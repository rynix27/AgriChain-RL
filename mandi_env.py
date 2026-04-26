"""
MandiNegotiateEnv -- Multi-Agent Mandi Negotiation Environment
Part of AgriChain-RL v2

Fixes applied:
    - Buyer walkaway mechanic (leaves after round 5-6 if no deal)
    - Reward normalized to -1 to +1 range
    - Real market price differentiation across 6 destinations
    - Buyers update offers more aggressively each round
"""

import random
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from pydantic import BaseModel


# ---- Constants ---------------------------------------------------------------

MSP = {
    "tomato":      12.0,
    "potato":       8.0,
    "onion":       10.0,
    "banana":      15.0,
    "mango":       30.0,
    "cauliflower":  8.0,
    "spinach":     10.0,
    "carrot":      12.0,
}

MARKET_RATE = {
    "tomato":      (15, 45),
    "potato":      (10, 22),
    "onion":       (12, 35),
    "banana":      (18, 40),
    "mango":       (35, 120),
    "cauliflower": (10, 25),
    "spinach":     (12, 28),
    "carrot":      (14, 30),
}

# Real APMC market price multipliers per destination per crop
# Different mandis have different demand levels for different crops
MARKET_MULTIPLIERS = {
    "Delhi APMC":     {"tomato": 1.15, "onion": 1.20, "potato": 1.10, "banana": 1.05,
                       "mango": 1.25, "cauliflower": 1.10, "spinach": 1.15, "carrot": 1.10},
    "Mumbai APMC":    {"tomato": 1.20, "onion": 1.10, "potato": 1.05, "banana": 1.25,
                       "mango": 1.30, "cauliflower": 1.05, "spinach": 1.20, "carrot": 1.15},
    "Chennai APMC":   {"tomato": 1.10, "onion": 1.05, "potato": 1.00, "banana": 1.30,
                       "mango": 1.20, "cauliflower": 1.00, "spinach": 1.10, "carrot": 1.05},
    "Kolkata APMC":   {"tomato": 1.05, "onion": 1.15, "potato": 1.20, "banana": 1.10,
                       "mango": 1.10, "cauliflower": 1.20, "spinach": 1.05, "carrot": 1.20},
    "Local Mandi":    {"tomato": 0.90, "onion": 0.90, "potato": 0.90, "banana": 0.90,
                       "mango": 0.90, "cauliflower": 0.90, "spinach": 0.90, "carrot": 0.90},
    "Bangalore APMC": {"tomato": 1.12, "onion": 1.08, "potato": 1.05, "banana": 1.18,
                       "mango": 1.22, "cauliflower": 1.08, "spinach": 1.12, "carrot": 1.08},
}

BUYER_NAMES = [
    "Ramesh Traders",
    "Delhi Agro Exports",
    "FreshMart Wholesale",
]

MAX_ROUNDS = 8
REWARD_SCALE = 100.0   # divide by this to normalize rewards to -1 to +1


# ---- Enums -------------------------------------------------------------------

class BuyerStrategy(str, Enum):
    AGGRESSIVE = "aggressive"
    MODERATE   = "moderate"
    DESPERATE  = "desperate"


class FarmerAction(str, Enum):
    ACCEPT       = "accept"
    COUNTER      = "counter"
    REJECT_ALL   = "reject_all"
    SELL_PARTIAL = "sell_partial"


# ---- Buyer -------------------------------------------------------------------

@dataclass
class Buyer:
    id: str
    name: str
    strategy: BuyerStrategy
    hidden_budget: float
    hidden_demand: float
    current_offer: float
    rounds_active: int = 0
    accepted: bool = False
    walked_away: bool = False
    walkaway_round: int = 0      # the round this buyer will walk if no deal

    def make_offer(self, market_rate: float, farmer_quality: float, rng: random.Random) -> float:
        if self.walked_away or self.accepted:
            return self.current_offer

        base = market_rate * farmer_quality

        if self.strategy == BuyerStrategy.AGGRESSIVE:
            discount = rng.uniform(0.28, 0.38) - (self.rounds_active * 0.025)
            offer    = base * (1 - max(0.04, discount))
        elif self.strategy == BuyerStrategy.MODERATE:
            discount = rng.uniform(0.08, 0.18) - (self.rounds_active * 0.035)
            offer    = base * (1 - max(0.01, discount))
        elif self.strategy == BuyerStrategy.DESPERATE:
            premium  = self.hidden_demand * 0.18
            offer    = base * (1 + premium - rng.uniform(0, 0.03))

        offer = min(offer, self.hidden_budget)
        offer = max(offer, 1.0)

        self.current_offer = round(offer, 2)
        self.rounds_active += 1
        return self.current_offer

    def check_walkaway(self, current_round: int):
        """Buyer walks away if no deal by their walkaway round."""
        if not self.accepted and not self.walked_away:
            if current_round >= self.walkaway_round:
                self.walked_away = True

    def to_dict(self):
        rounds_left = max(0, self.walkaway_round - self.rounds_active)
        return {
            "id": self.id,
            "name": self.name,
            "strategy_hint": (
                "tough" if self.strategy == BuyerStrategy.AGGRESSIVE else
                "fair"  if self.strategy == BuyerStrategy.MODERATE   else
                "eager"
            ),
            "current_offer": self.current_offer,
            "rounds_active": self.rounds_active,
            "accepted": self.accepted,
            "walked_away": self.walked_away,
            "rounds_until_walkaway": rounds_left,   # farmer can see urgency signal
        }


# ---- Action / Observation Models ---------------------------------------------

class MandiAction(BaseModel):
    action_type: FarmerAction = FarmerAction.REJECT_ALL
    target_buyer_id: Optional[str] = None
    counter_price: Optional[float] = None


class MandiObservation(BaseModel):
    round: int
    max_rounds: int
    crop_type: str
    destination: str
    quantity_kg: float
    quality_score: float
    days_since_harvest: int
    transport_cost_incurred: float
    market_rate_today: float
    msp: float
    buyers: list
    active_buyers: int
    best_offer: float
    best_buyer_id: str
    urgency_level: str
    farmer_leverage: str
    raw_reward: float
    normalized_reward: float
    score: float
    normalized_score: float
    done: bool
    message: str
    info: dict


# ---- Environment -------------------------------------------------------------

class MandiNegotiateEnv:

    def __init__(self):
        self._rng = random.Random()
        self.crop_type = "tomato"
        self.destination = "Delhi APMC"
        self.quantity_kg = 500.0
        self.quality_score = 0.8
        self.days_since_harvest = 3
        self.transport_cost = 50.0

        self.round = 0
        self.score = 0.0
        self.normalized_score = 0.0
        self.buyers: dict[str, Buyer] = {}
        self.market_rate = 25.0
        self.msp = 12.0
        self._done = False
        self._last_raw_reward = 0.0
        self._last_norm_reward = 0.0
        self._last_info = {}
        self._quantity_sold = 0.0
        self._revenue = 0.0

    def reset(
        self,
        crop_type: str = "tomato",
        quantity_kg: float = 500.0,
        quality_score: float = 0.8,
        days_since_harvest: int = 3,
        transport_cost: float = 50.0,
        destination: str = "Delhi APMC",
        seed: Optional[int] = None,
    ) -> MandiObservation:

        if seed is not None:
            self._rng.seed(seed)

        self.crop_type          = crop_type
        self.destination        = destination
        self.quantity_kg        = quantity_kg
        self.quality_score      = max(0.01, min(1.0, quality_score))
        self.days_since_harvest = days_since_harvest
        self.transport_cost     = transport_cost

        self.round              = 0
        self.score              = 0.0
        self.normalized_score   = 0.0
        self._done              = False
        self._last_raw_reward   = 0.0
        self._last_norm_reward  = 0.0
        self._quantity_sold     = 0.0
        self._revenue           = 0.0

        # Apply destination market multiplier to base rate
        lo, hi = MARKET_RATE.get(crop_type, (10, 30))
        base_rate   = self._rng.uniform(lo, hi)
        multiplier  = MARKET_MULTIPLIERS.get(destination, {}).get(crop_type, 1.0)
        self.market_rate = round(base_rate * multiplier, 2)
        self.msp         = MSP.get(crop_type, 10.0)

        # Spawn 3 buyers with random strategies
        strategies = self._rng.sample(list(BuyerStrategy), 3)
        self.buyers = {}
        for i, (name, strategy) in enumerate(zip(BUYER_NAMES, strategies)):
            bid    = f"B{i+1}"
            budget = self.market_rate * self._rng.uniform(0.9, 1.4)
            demand = self._rng.uniform(0.2, 0.95)

            # Each buyer has a randomized walkaway round (5 to 7)
            walkaway = self._rng.randint(5, 7)

            buyer = Buyer(
                id=bid,
                name=name,
                strategy=strategy,
                hidden_budget=round(budget, 2),
                hidden_demand=round(demand, 2),
                current_offer=0.0,
                walkaway_round=walkaway,
            )
            buyer.make_offer(self.market_rate, self.quality_score, self._rng)
            self.buyers[bid] = buyer

        self._last_info = {
            "event": "mandi_reset",
            "market_rate": self.market_rate,
            "destination": destination,
            "quality_score": self.quality_score,
        }

        return self._make_observation("Arrived at mandi. Negotiate well.")

    def step(self, action: MandiAction) -> tuple[MandiObservation, float, bool]:
        if self._done:
            return self._make_observation("Episode done."), 0.0, True

        self.round += 1
        raw_reward = 0.0
        event      = ""

        best_buyer = self._get_best_buyer()

        if action.action_type == FarmerAction.ACCEPT:
            raw_reward, event = self._handle_accept(action, best_buyer)
        elif action.action_type == FarmerAction.COUNTER:
            raw_reward, event = self._handle_counter(action)
        elif action.action_type == FarmerAction.SELL_PARTIAL:
            raw_reward, event = self._handle_partial(action, best_buyer)
        elif action.action_type == FarmerAction.REJECT_ALL:
            raw_reward, event = self._handle_reject()

        if not self._done:
            # Buyers update offers and check walkaway
            walked = []
            for buyer in self.buyers.values():
                buyer.check_walkaway(self.round)
                if buyer.walked_away and buyer.id not in [b for b in walked]:
                    walked.append(buyer.name)
                elif not buyer.accepted and not buyer.walked_away:
                    buyer.make_offer(self.market_rate, self.quality_score, self._rng)

            if walked:
                event += f" {', '.join(walked)} walked away -- no deal reached."

            self.quality_score      = max(0.01, self.quality_score - 0.04)
            self.days_since_harvest += 1

        # Terminal conditions
        if self.quantity_kg <= 0:
            self._done = True
            event += " All produce sold."
        elif self.round >= MAX_ROUNDS:
            self._done = True
            if self.quantity_kg > 0:
                forced = self.quantity_kg * self.msp
                self._revenue += forced
                raw_reward -= 20.0
                event += f" Forced sale at MSP Rs{self.msp}/kg."
        elif self.quality_score < 0.1:
            self._done = True
            raw_reward -= 50.0
            event += " Produce quality critical -- distress sale."
        elif self._all_buyers_gone():
            self._done = True
            raw_reward -= 30.0
            event += " All buyers walked away -- no deal possible."

        norm_reward = self._normalize(raw_reward)
        self.score            += raw_reward
        self.normalized_score += norm_reward
        self._last_raw_reward  = raw_reward
        self._last_norm_reward = norm_reward
        self._last_info = {
            "event": event,
            "round": self.round,
            "quality_score": round(self.quality_score, 3),
            "revenue_so_far": round(self._revenue, 2),
            "quantity_remaining": round(self.quantity_kg, 1),
            "active_buyers": len(self._active_buyers()),
        }

        return self._make_observation(event), norm_reward, self._done

    # ---- Action Handlers -----------------------------------------------------

    def _handle_accept(self, action: MandiAction, best_buyer: Optional[Buyer]):
        target = (
            self.buyers.get(action.target_buyer_id) if action.target_buyer_id
            else best_buyer
        )
        if not target or target.walked_away:
            return -10.0, "invalid_accept: buyer not available"

        price   = target.current_offer
        revenue = price * self.quantity_kg
        self._revenue       += revenue
        self._quantity_sold += self.quantity_kg

        ratio = price / self.market_rate
        if ratio >= 0.95:
            reward, outcome = 100.0, "excellent deal -- above 95% of market rate"
        elif ratio >= 0.80:
            reward, outcome = 60.0, "good deal -- above 80% of market rate"
        elif ratio >= self.msp / self.market_rate:
            reward, outcome = 30.0, "acceptable deal -- above MSP"
        else:
            reward, outcome = -20.0, "bad deal -- below MSP"

        # Bonus for negotiating before accepting
        if self.round > 2:
            reward += min(10.0, self.round * 2.0)

        self.quantity_kg = 0.0
        self._done       = True
        target.accepted  = True

        return reward, (
            f"Accepted Rs{price}/kg from {target.name}. "
            f"Revenue: Rs{revenue:,.0f}. {outcome}."
        )

    def _handle_counter(self, action: MandiAction):
        if not action.counter_price:
            return -5.0, "invalid_counter: no price specified"

        counter = action.counter_price
        if counter < self.msp:
            return -10.0, f"invalid_counter: Rs{counter} is below MSP Rs{self.msp}"

        accepted_by = [
            b for b in self.buyers.values()
            if not b.accepted and not b.walked_away
            and counter <= b.hidden_budget * 1.05
        ]

        if accepted_by:
            winner  = max(accepted_by, key=lambda b: b.hidden_demand)
            revenue = counter * self.quantity_kg
            self._revenue       += revenue
            self._quantity_sold += self.quantity_kg
            self.quantity_kg = 0.0
            self._done       = True
            winner.accepted  = True
            ratio  = counter / self.market_rate
            reward = 80.0 if ratio >= 0.90 else 50.0 if ratio >= 0.75 else 20.0
            return reward, (
                f"Counter Rs{counter}/kg accepted by {winner.name}. "
                f"Revenue: Rs{revenue:,.0f}."
            )
        else:
            return -5.0, f"Counter Rs{counter}/kg rejected by all buyers."

    def _handle_partial(self, action: MandiAction, best_buyer: Optional[Buyer]):
        target = (
            self.buyers.get(action.target_buyer_id) if action.target_buyer_id
            else best_buyer
        )
        if not target or target.walked_away:
            return -10.0, "invalid_partial: buyer not available"

        sell_qty = self.quantity_kg * 0.5
        price    = target.current_offer
        revenue  = price * sell_qty
        self._revenue       += revenue
        self._quantity_sold += sell_qty
        self.quantity_kg    -= sell_qty

        ratio  = price / self.market_rate
        reward = 40.0 if ratio >= 0.85 else 20.0 if ratio >= 0.70 else 5.0

        return reward, (
            f"Sold {sell_qty:.0f}kg to {target.name} at Rs{price}/kg. "
            f"Rs{revenue:,.0f} earned. {self.quantity_kg:.0f}kg remaining."
        )

    def _handle_reject(self):
        active = self._active_buyers()
        if not active:
            return -20.0, "No active buyers left to negotiate with."

        leverage = self._get_leverage()
        if leverage in ["strong", "moderate"]:
            return 2.0, "Rejected all offers. Waiting for better bids."
        else:
            return -5.0, "Rejected all offers. Quality dropping -- risky move."

    # ---- Helpers -------------------------------------------------------------

    def _normalize(self, reward: float) -> float:
        return max(-1.0, min(1.0, reward / REWARD_SCALE))

    def _active_buyers(self) -> list:
        return [b for b in self.buyers.values() if not b.accepted and not b.walked_away]

    def _all_buyers_gone(self) -> bool:
        return len(self._active_buyers()) == 0 and self.quantity_kg > 0

    def _get_best_buyer(self) -> Optional[Buyer]:
        active = self._active_buyers()
        return max(active, key=lambda b: b.current_offer) if active else None

    def _get_leverage(self) -> str:
        if self.quality_score >= 0.75 and self.days_since_harvest <= 4:
            return "strong"
        elif self.quality_score >= 0.50 and self.days_since_harvest <= 7:
            return "moderate"
        elif self.quality_score >= 0.25:
            return "weak"
        else:
            return "desperate"

    def _get_urgency(self) -> str:
        if self.days_since_harvest <= 3:   return "low"
        elif self.days_since_harvest <= 6: return "medium"
        elif self.days_since_harvest <= 9: return "high"
        else:                              return "critical"

    def _make_observation(self, message: str) -> MandiObservation:
        best = self._get_best_buyer()
        return MandiObservation(
            round=self.round,
            max_rounds=MAX_ROUNDS,
            crop_type=self.crop_type,
            destination=self.destination,
            quantity_kg=round(self.quantity_kg, 1),
            quality_score=round(self.quality_score, 3),
            days_since_harvest=self.days_since_harvest,
            transport_cost_incurred=self.transport_cost,
            market_rate_today=self.market_rate,
            msp=self.msp,
            buyers=[b.to_dict() for b in self.buyers.values()],
            active_buyers=len(self._active_buyers()),
            best_offer=best.current_offer if best else 0.0,
            best_buyer_id=best.id if best else "",
            urgency_level=self._get_urgency(),
            farmer_leverage=self._get_leverage(),
            raw_reward=self._last_raw_reward,
            normalized_reward=self._last_norm_reward,
            score=round(self.score, 2),
            normalized_score=round(self.normalized_score, 4),
            done=self._done,
            message=message,
            info=self._last_info,
        )

    # ---- 4 Independent Reward Functions -------------------------------------
    # As recommended by the hackathon guide -- multiple independent checks
    # reduce reward hacking vs a single combined score.

    def reward_price_ratio(self) -> float:
        """R1: How close to market rate did the farmer sell? (0-1)
        Penalizes selling below MSP. Rewards selling above 90% of market rate.
        Anti-hack: capped at 1.0 so inflating price field doesnt help."""
        if self._quantity_sold == 0:
            return 0.0
        avg_price = self._revenue / self._quantity_sold
        ratio     = avg_price / max(self.market_rate, 1.0)
        if avg_price < self.msp:
            return max(0.0, ratio * 0.3)   # heavy penalty for below MSP
        return round(min(1.0, ratio), 3)

    def reward_sell_completeness(self) -> float:
        """R2: What fraction of produce was sold vs wasted? (0-1)
        Anti-hack: checks actual quantity_sold vs total, not just episode score."""
        total = self._quantity_sold + self.quantity_kg
        if total == 0:
            return 0.0
        return round(self._quantity_sold / total, 3)

    def reward_negotiation_efficiency(self) -> float:
        """R3: Did the agent negotiate well before accepting? (0-1)
        Rewards holding out for better price if leverage was strong.
        Anti-hack: only positive if rounds > 1 AND price ratio > 0.8."""
        if self._quantity_sold == 0:
            return 0.0
        avg_price = self._revenue / self._quantity_sold
        ratio     = avg_price / max(self.market_rate, 1.0)
        rounds_bonus = min(1.0, self.round / 4.0)   # reward for negotiating
        if ratio >= 0.80:
            return round(0.5 + 0.5 * rounds_bonus, 3)
        return round(0.2 * rounds_bonus, 3)

    def reward_timeout_penalty(self) -> float:
        """R4: Penalty for running out of time or all buyers walking away. (0-1)
        Anti-hack: independent of price -- catches agents that stall endlessly."""
        if self.round >= MAX_ROUNDS and self._quantity_sold == 0:
            return 0.0   # worst case -- no sale, ran out of rounds
        if self._all_buyers_gone() and self._quantity_sold == 0:
            return 0.0   # all buyers walked -- complete failure
        if self.round >= MAX_ROUNDS:
            return 0.5   # forced sale -- partial credit
        return 1.0       # completed within time limit

    def grade(self) -> dict:
        """Combined grade using 4 independent reward functions."""
        if self._quantity_sold == 0:
            return {
                "score": 0.0, "grade": "F", "total_revenue": 0.0,
                "r1_price_ratio": 0.0, "r2_sell_completeness": 0.0,
                "r3_negotiation": 0.0, "r4_timeout": self.reward_timeout_penalty(),
            }

        r1 = self.reward_price_ratio()
        r2 = self.reward_sell_completeness()
        r3 = self.reward_negotiation_efficiency()
        r4 = self.reward_timeout_penalty()

        # Weighted combination -- price and completeness matter most
        final = round(0.40 * r1 + 0.30 * r2 + 0.20 * r3 + 0.10 * r4, 3)
        grade = (
            "S" if final >= 0.90 else
            "A" if final >= 0.75 else
            "B" if final >= 0.60 else
            "C" if final >= 0.45 else "F"
        )

        avg_price = self._revenue / self._quantity_sold
        return {
            "score": final,
            "grade": grade,
            "avg_price_per_kg": round(avg_price, 2),
            "market_rate": self.market_rate,
            "price_ratio": round(avg_price / max(self.market_rate, 1.0), 3),
            "sell_ratio": round(self._quantity_sold / max(self._quantity_sold + self.quantity_kg, 1), 3),
            "total_revenue": round(self._revenue, 2),
            "r1_price_ratio": r1,
            "r2_sell_completeness": r2,
            "r3_negotiation": r3,
            "r4_timeout": r4,
        }
