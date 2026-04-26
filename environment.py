"""
FreshChain Post-Harvest Yield Loss Environment
server/environment.py

This is the BRAIN of the fake world.
It tracks all the batches, spoilage timers, trucks, and scores.
"""

import random
import signal
import functools
from uuid import uuid4
from typing import Optional
from models import (
    FreshChainObservation, FreshChainAction, FreshChainState,
    BatchInfo, TruckInfo
)
from whatsapp_alerts import WhatsAppAlertSystem, AlertType

alert_system = WhatsAppAlertSystem()

# ─────────────────────────────────────────────
# EXECUTION TIMEOUT GUARD  (hackathon guide §8)
# Prevents infinite loops inside step() from
# hanging the RL training loop.
# ─────────────────────────────────────────────

class StepTimeoutError(Exception):
    pass

def _step_timeout(seconds: int = 10):
    """Decorator: terminate step() if it runs longer than `seconds`."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self_obj, *args, **kwargs):
            def _handler(signum, frame):
                raise StepTimeoutError(
                    f"step() exceeded {seconds}s — possible infinite loop or exploit."
                )
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                return func(self_obj, *args, **kwargs)
            except StepTimeoutError:
                # Terminate episode cleanly instead of crashing the trainer
                self_obj._done = True
                return self_obj._build_observation(
                    "⚠️ Step timeout: episode terminated by safety guard.",
                    reward=-1.0,
                )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

# ─────────────────────────────────────────────
# ANTI-EXPLOIT ACTION VALIDATOR  (hackathon guide §8)
# Rejects actions that try to mutate protected
# state fields directly via the action payload.
# ─────────────────────────────────────────────

_FORBIDDEN_FIELD_WRITES = {
    "spoilage_risk", "step_count", "_done", "max_steps",
    "yield_saved", "yield_lost", "quality_score",
}

def _validate_action(action: "FreshChainAction") -> bool:
    """Return False if the action contains any protected field override attempt."""
    action_dict = action.__dict__ if hasattr(action, "__dict__") else {}
    for key in action_dict:
        if key in _FORBIDDEN_FIELD_WRITES:
            return False
    return True

# ─────────────────────────────────────────────
# TASK CONFIGURATIONS
# ─────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "description": "Single batch, one truck, straightforward dispatch",
        "num_batches": 1,
        "num_trucks": 1,
        "truck_failure": False,
        "price_volatility": False,
        "max_steps": 5,
    },
    "medium": {
        "description": "3 batches, one truck breaks down mid-episode, agent must prioritize",
        "num_batches": 3,
        "num_trucks": 2,
        "truck_failure": True,
        "price_volatility": False,
        "max_steps": 10,
    },
    "hard": {
        "description": "Full network with spoilage cascade, price volatility, limited trucks",
        "num_batches": 6,
        "num_trucks": 2,
        "truck_failure": True,
        "price_volatility": True,
        "max_steps": 15,
    },
}

CROPS = ["tomato", "potato", "onion", "banana", "mango", "cauliflower", "spinach", "carrot"]

# Real Indian APMC market price ranges (INR/kg) based on 2024-25 data
CROP_PRICE_RANGES = {
    "tomato":      (12, 45),
    "potato":      (8, 22),
    "onion":       (10, 35),
    "banana":      (15, 40),
    "mango":       (30, 120),
    "cauliflower": (8, 25),
    "spinach":     (10, 28),
    "carrot":      (12, 30),
}

# Spoilage speed by crop type (days before critical spoilage)
CROP_SPOILAGE_RATE = {
    "tomato":      0.18,  # Very perishable
    "spinach":     0.22,  # Most perishable
    "banana":      0.15,
    "mango":       0.14,
    "cauliflower": 0.12,
    "carrot":      0.08,  # More durable
    "potato":      0.06,  # Most durable
    "onion":       0.05,
}

MARKETS = [
    "Azadpur Mandi (Delhi)",
    "Vashi APMC (Mumbai)",
    "Koyambedu (Chennai)",
    "Gultekdi Market (Pune)",
    "Bowenpally (Hyderabad)",
    "Yeshwanthpur (Bangalore)",
]


class FreshChainEnvironment:
    """
    The Post-Harvest Yield Loss simulation environment.

    Rules:
    - Each step, spoilage_risk increases for all batches in storage
    - If spoilage_risk >= 1.0, the batch is lost (yield lost)
    - Agent can dispatch, store, reroute, or discard batches
    - Score = (yield_saved) / (yield_saved + yield_lost)
    """

    def __init__(self):
        self._state = FreshChainState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id="easy",
            max_steps=5
        )
        self._batches: dict[str, BatchInfo] = {}
        self._trucks: dict[str, TruckInfo] = {}
        self._total_saved = 0.0
        self._total_lost = 0.0
        self._initial_total_kg = 0.0
        self._config = TASK_CONFIGS["easy"]
        self._done = False

    # ─────────────────────────────────────────────
    # RESET — Start a fresh episode
    # ─────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> FreshChainObservation:
        """Initialize a new episode for the given task."""
        if task_id not in TASK_CONFIGS:
            task_id = "easy"

        self._config = TASK_CONFIGS[task_id]
        self._state = FreshChainState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=self._config["max_steps"]
        )
        self._batches = {}
        self._trucks = {}
        self._total_saved = 0.0
        self._total_lost = 0.0
        self._done = False
        self._truck_failure_step = random.randint(3, 7)

        # Generate batches with real Indian market data
        for i in range(self._config["num_batches"]):
            bid = f"B{i+1:03d}"
            qty = round(random.uniform(200, 800), 1)
            crop = random.choice(CROPS)
            price_min, price_max = CROP_PRICE_RANGES[crop]
            self._batches[bid] = BatchInfo(
                batch_id=bid,
                crop_type=crop,
                quantity_kg=qty,
                temperature_c=round(random.uniform(8, 22), 1),
                humidity_pct=round(random.uniform(55, 85), 1),
                spoilage_risk=round(random.uniform(0.1, 0.35), 2),
                days_in_storage=random.randint(1, 3),
                market_price_per_kg=round(random.uniform(price_min, price_max), 2),
            )

        self._initial_total_kg = sum(b.quantity_kg for b in self._batches.values())

        # Generate trucks
        for i in range(self._config["num_trucks"]):
            tid = f"T{i+1:02d}"
            self._trucks[tid] = TruckInfo(
                truck_id=tid,
                capacity_kg=round(random.uniform(500, 1000), 0),
                available=True,
                destination=random.choice(MARKETS)
            )

        # WhatsApp alert — episode start
        alert_system.reset()
        alert_system.send(
            AlertType.EPISODE_START,
            task_id=task_id.upper(),
            num_batches=len(self._batches),
            total_kg=round(self._initial_total_kg, 0),
        )

        return self._build_observation("Episode started. Assess batches and act.")

    # ─────────────────────────────────────────────
    # STEP — Agent takes an action
    # ─────────────────────────────────────────────

    @_step_timeout(seconds=10)
    def step(self, action: FreshChainAction) -> FreshChainObservation:
        """Execute one action and advance the simulation by one step."""
        if self._done:
            return self._build_observation("Episode already finished.")

        # Reject any action that tries to mutate protected state fields
        if not _validate_action(action):
            self._done = True
            return self._build_observation(
                "Invalid action: attempted protected field write. Episode terminated.",
                reward=-1.0,
            )

        self._state.step_count += 1
        reward = 0.0
        message = ""

        # ── Process action ──
        if action.action_type == "dispatch":
            reward, message = self._handle_dispatch(action)

        elif action.action_type == "store":
            reward = -0.05  # Small penalty for doing nothing
            message = f"Stored all batches for one more step. Spoilage risk increases."

        elif action.action_type == "reroute":
            reward, message = self._handle_reroute(action)

        elif action.action_type == "discard":
            reward, message = self._handle_discard(action)

        else:
            reward = -0.1
            message = f"Unknown action '{action.action_type}'. No effect."

        # ── Advance spoilage for remaining batches ──
        spoiled_this_step = self._advance_spoilage()
        if spoiled_this_step > 0:
            message += f" ⚠ {spoiled_this_step:.1f}kg spoiled this step."
            reward -= 0.2  # Penalize spoilage

        # ── Truck failure event (medium/hard tasks) ──
        if self._config["truck_failure"] and self._state.step_count == self._truck_failure_step:
            self._trigger_truck_failure()
            message += "  Truck T01 has broken down!"
            alert_system.send(
                AlertType.TRUCK_BREAKDOWN,
                truck_id="T01",
                batches_at_risk=len(self._batches),
            )

        # ── Critical batch alerts ──
        for batch in self._batches.values():
            if batch.spoilage_risk >= 0.85:
                alert_system.send(
                    AlertType.CRITICAL_BATCH,
                    batch_id=batch.batch_id,
                    crop_type=batch.crop_type,
                    quantity=round(batch.quantity_kg, 0),
                )

        # ── Price volatility (hard task) ──
        if self._config["price_volatility"]:
            self._fluctuate_prices()

        # ── Check episode end ──
        all_gone = len(self._batches) == 0
        max_steps_reached = self._state.step_count >= self._state.max_steps
        if all_gone or max_steps_reached:
            self._done = True
            message += " Episode complete."

        # Normalize reward to -1 to +1 range (FreshChain raw max ~2.5)
        normalized_reward = max(-1.0, min(1.0, reward / 2.5))
        return self._build_observation(message, normalized_reward)

    # ─────────────────────────────────────────────
    # STATE — Return episode metadata
    # ─────────────────────────────────────────────

    @property
    def state(self) -> FreshChainState:
        return self._state

    # ─────────────────────────────────────────────
    # ACTION HANDLERS
    # ─────────────────────────────────────────────

    def _handle_dispatch(self, action: FreshChainAction):
        batch = self._batches.get(action.batch_id)
        truck = self._trucks.get(action.truck_id)

        if not batch:
            return -0.15, f"Batch {action.batch_id} not found."
        if not truck:
            return -0.15, f"Truck {action.truck_id} not found."
        if not truck.available:
            return -0.15, f"Truck {action.truck_id} is not available."
        if batch.quantity_kg > truck.capacity_kg:
            # Partial dispatch
            dispatched = truck.capacity_kg
            batch.quantity_kg -= dispatched
            self._total_saved += dispatched
            reward = (dispatched / self._initial_total_kg) * 2.0
            # High-risk batch dispatched = bonus
            if batch.spoilage_risk > 0.6:
                reward += 0.2
            truck.available = False
            return round(reward, 3), (
                f"Partial dispatch: {dispatched:.0f}kg of {batch.batch_id} "
                f"sent to {truck.destination} via {truck.truck_id}."
            )
        else:
            dispatched = batch.quantity_kg
            self._total_saved += dispatched
            reward = (dispatched / self._initial_total_kg) * 2.0
            if batch.spoilage_risk > 0.6:
                reward += 0.3
            del self._batches[action.batch_id]
            truck.available = False
            alert_system.send(
                AlertType.DISPATCH_CONFIRMED,
                batch_id=action.batch_id,
                crop_type=batch.crop_type,
                quantity=round(dispatched, 0),
                destination=truck.destination,
                truck_id=truck.truck_id,
                value=round(dispatched * batch.market_price_per_kg, 0),
            )
            return round(reward, 3), (
                f"Dispatched {dispatched:.0f}kg of {batch.crop_type} "
                f"({batch.batch_id}) to {truck.destination}. "
                f"Market price: ₹{batch.market_price_per_kg}/kg."
            )

    def _handle_reroute(self, action: FreshChainAction):
        batch = self._batches.get(action.batch_id)
        if not batch:
            return -0.1, f"Batch {action.batch_id} not found for reroute."
        new_dest = action.destination or random.choice(MARKETS)

        # Real market price multipliers -- not a flat 5% bump
        DEST_MULTIPLIERS = {
            "Delhi APMC":     1.15,
            "Mumbai APMC":    1.20,
            "Chennai APMC":   1.10,
            "Kolkata APMC":   1.08,
            "Bangalore APMC": 1.12,
            "Local Mandi":    0.90,
        }
        multiplier = DEST_MULTIPLIERS.get(new_dest, 1.05)
        old_price  = batch.market_price_per_kg
        batch.market_price_per_kg = round(old_price * multiplier, 2)
        price_gain = batch.market_price_per_kg - old_price
        reward     = max(-0.2, min(0.5, price_gain / old_price))

        alert_system.send(
            AlertType.REROUTE,
            batch_id=action.batch_id,
            destination=new_dest,
            price=batch.market_price_per_kg,
        )
        return round(reward, 3), (
            f"Rerouted batch {action.batch_id} to {new_dest}. "
            f"Price: Rs{old_price} -> Rs{batch.market_price_per_kg}/kg ({multiplier:.2f}x)."
        )

    def _handle_discard(self, action: FreshChainAction):
        batch = self._batches.get(action.batch_id)
        if not batch:
            return -0.1, f"Batch {action.batch_id} not found for discard."
        # Discarding a high-risk batch prevents cascade spoilage — small reward
        if batch.spoilage_risk > 0.75:
            self._total_lost += batch.quantity_kg
            del self._batches[action.batch_id]
            return 0.05, (
                f"Discarded critical batch {action.batch_id} ({batch.quantity_kg:.0f}kg). "
                f"Prevented spoilage cascade."
            )
        else:
            # Discarding a healthy batch = bad decision
            self._total_lost += batch.quantity_kg
            del self._batches[action.batch_id]
            return -0.3, (
                f"Discarded batch {action.batch_id} unnecessarily. "
                f"{batch.quantity_kg:.0f}kg wasted."
            )

    # ─────────────────────────────────────────────
    # SIMULATION HELPERS
    # ─────────────────────────────────────────────

    def _advance_spoilage(self) -> float:
        """Increase spoilage risk for all stored batches. Returns kg lost this step."""
        lost_kg = 0.0
        to_remove = []
        for bid, batch in self._batches.items():
            # Crop-specific base spoilage rate
            base_increase = CROP_SPOILAGE_RATE.get(batch.crop_type, 0.12)
            # Temperature above 15°C accelerates spoilage
            if batch.temperature_c > 15:
                base_increase += (batch.temperature_c - 15) * 0.008
            # High humidity accelerates spoilage
            if batch.humidity_pct > 70:
                base_increase += (batch.humidity_pct - 70) * 0.003
            batch.spoilage_risk = round(min(1.0, batch.spoilage_risk + base_increase), 2)
            batch.days_in_storage += 1

            if batch.spoilage_risk >= 1.0:
                lost_kg += batch.quantity_kg
                self._total_lost += batch.quantity_kg
                to_remove.append(bid)

        for bid in to_remove:
            del self._batches[bid]

        return lost_kg

    def _trigger_truck_failure(self):
        """Make the first truck unavailable (breakdown event)."""
        if "T01" in self._trucks:
            self._trucks["T01"].available = False

    def _fluctuate_prices(self):
        """Randomly shift market prices to simulate price volatility."""
        for batch in self._batches.values():
            change = random.uniform(-0.08, 0.08)
            batch.market_price_per_kg = round(
                max(5.0, batch.market_price_per_kg * (1 + change)), 2
            )

    # ─────────────────────────────────────────────
    # GRADERS — Score the agent 0.0 to 1.0
    # ─────────────────────────────────────────────

    def grade(self) -> float:
        """
        Final task score from 0.0 to 1.0.
        Based on what fraction of total yield was saved vs lost.
        Partial credit for partial saves.
        """
        total = self._total_saved + self._total_lost
        if total == 0:
            return 0.0
        base_score = self._total_saved / total

        # Bonus: finishing before max steps = efficiency bonus
        steps_used = self._state.step_count
        max_steps = self._state.max_steps
        efficiency_bonus = max(0, (max_steps - steps_used) / max_steps) * 0.1

        return round(min(1.0, base_score + efficiency_bonus), 3)

    # ─────────────────────────────────────────────
    # BUILD OBSERVATION
    # ─────────────────────────────────────────────

    def _build_observation(self, message: str, reward: float = 0.0) -> FreshChainObservation:
        total = self._total_saved + self._total_lost
        task_score = self._total_saved / total if total > 0 else 0.0
        used_pct = (
            sum(b.quantity_kg for b in self._batches.values()) /
            max(1, self._initial_total_kg)
        ) * 100

        return FreshChainObservation(
            step=self._state.step_count,
            batches=list(self._batches.values()),
            trucks=list(self._trucks.values()),
            total_yield_saved_kg=round(self._total_saved, 1),
            total_yield_lost_kg=round(self._total_lost, 1),
            storage_capacity_used_pct=round(used_pct, 1),
            message=message,
            done=self._done,
            reward=reward,
            task_score=round(task_score, 3),
        )
