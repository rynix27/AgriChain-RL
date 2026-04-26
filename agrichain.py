"""
AgriChain-RL -- OpenEnv-wrapped two-stage environment
agrichain.py v2

Fixes applied:
    - Proper OpenEnv BaseEnvironment / BaseAction / BaseObservation wrapper
    - Both stage rewards normalized to -1 to +1
    - Reroute uses real market price multipliers (6 destinations)
    - Truck failure timing randomized
    - Buyer walkaway mechanic in MandiNegotiateEnv
"""

import json
import random
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenEnv imports
try:
    from openenv import create_app, BaseAction, BaseObservation, BaseEnvironment
    OPENENV_AVAILABLE = True
except ImportError:
    # Fallback: define minimal base classes so server starts even if openenv-core
    # installs under a different module name
    from pydantic import BaseModel
    BaseAction = BaseModel
    BaseObservation = BaseModel
    class BaseEnvironment:
        pass
    def create_app(env_cls, action_cls, obs_cls, **kwargs):
        from fastapi import FastAPI
        return FastAPI()
    OPENENV_AVAILABLE = False

from mandi_env import (
    MandiNegotiateEnv, MandiAction, FarmerAction,
    MARKET_MULTIPLIERS,
)


# ---- Destinations (for reroute action) ----------------------------------------

DESTINATIONS = list(MARKET_MULTIPLIERS.keys())

FRESHCHAIN_REWARD_SCALE = 2.0   # max raw reward in FreshChain stage


# ---- Stage -------------------------------------------------------------------

class Stage:
    FRESHCHAIN = "freshchain"
    MANDI      = "mandi"
    COMPLETE   = "complete"


# ---- OpenEnv Action ----------------------------------------------------------

class AgriChainAction(BaseAction):
    stage: str = Stage.FRESHCHAIN

    # FreshChain fields
    freshchain_action_type: Optional[str] = None   # dispatch / store / reroute / discard
    batch_id: Optional[str] = None
    truck_id: Optional[str] = None
    destination: Optional[str] = None              # for reroute -- one of DESTINATIONS

    # Mandi fields
    mandi_action_type: Optional[str] = None
    target_buyer_id: Optional[str] = None
    counter_price: Optional[float] = None

    # Reset-time fields
    task_id: Optional[str] = "medium"
    seed: Optional[int] = None


# ---- OpenEnv Observation -----------------------------------------------------

class AgriChainObservation(BaseObservation):
    stage: str
    episode_raw_score: float
    episode_normalized_score: float
    done: bool
    message: str
    freshchain_obs: Optional[dict] = None
    freshchain_output: Optional[dict] = None
    mandi_obs: Optional[dict] = None
    final_summary: Optional[dict] = None


# ---- FreshChain Output (linking layer) ---------------------------------------

class FreshChainOutput:
    """
    Extracts Stage 1 results and maps them to Stage 2 inputs.

    Linking mechanism:
        quality_score       = spoilage-adjusted yield ratio (not just kg saved)
        quantity_kg         = actual kg saved
        days_since_harvest  = steps taken + base transit days (crop-dependent)
        transport_cost      = cost proxy from lost yield
        destination         = best market selected during reroute actions
    """

    TRANSIT_DAYS = {
        "tomato": 2, "potato": 3, "onion": 3,
        "banana": 2, "mango": 2, "cauliflower": 2,
        "spinach": 1, "carrot": 3,
    }

    def __init__(self, freshchain_state: dict, crop_type: str, chosen_destination: str):
        saved          = freshchain_state.get("total_yield_saved_kg", 200.0)
        lost           = freshchain_state.get("total_yield_lost_kg", 50.0)
        total          = saved + lost
        step_count     = freshchain_state.get("step_count", 5)

        # Quality score = weighted ratio: yield saved AND how fresh it is
        spoilage_ratio  = lost / max(total, 1.0)
        freshness       = max(0.0, 1.0 - spoilage_ratio)
        step_penalty    = min(0.3, step_count * 0.02)
        self.quality_score = round(max(0.05, freshness - step_penalty), 3)

        self.quantity_kg        = round(saved, 1)
        self.days_since_harvest = step_count + self.TRANSIT_DAYS.get(crop_type, 2)
        self.transport_cost     = round(lost * 1.5, 2)
        self.crop_type          = crop_type
        self.destination        = chosen_destination

        raw_fc_score            = saved / max(total, 1.0)
        self.freshchain_score   = round(raw_fc_score, 3)
        self.freshchain_grade   = self._to_grade(self.freshchain_score)

    def _to_grade(self, s: float) -> str:
        return "S" if s >= 0.90 else "A" if s >= 0.75 else "B" if s >= 0.60 else "C" if s >= 0.45 else "F"

    def to_dict(self):
        return {
            "quality_score":      self.quality_score,
            "quantity_kg":        self.quantity_kg,
            "days_since_harvest": self.days_since_harvest,
            "transport_cost":     self.transport_cost,
            "crop_type":          self.crop_type,
            "destination":        self.destination,
            "freshchain_score":   self.freshchain_score,
            "freshchain_grade":   self.freshchain_grade,
        }


# ---- Minimal FreshChain Sim (self-contained, no import dependency) -----------

CROPS = ["tomato", "potato", "onion", "banana", "mango", "cauliflower"]

SPOILAGE_RATE = {
    "easy":   0.03,
    "medium": 0.05,
    "hard":   0.08,
}

TRUCK_COUNT = {"easy": 4, "medium": 3, "hard": 2}
MAX_STEPS   = {"easy": 12, "medium": 10, "hard": 8}


class FreshChainSim:
    """
    Lightweight self-contained FreshChain simulation.
    Avoids environment.py import issues and gives cleaner state access.
    Fixes: randomized truck failure timing.
    """

    def __init__(self):
        self._rng = random.Random()
        self.crop_type = "tomato"
        self.task_id   = "medium"
        self.batches   = []
        self.trucks    = []
        self.step_count = 0
        self.max_steps  = 10
        self.done       = False
        self.total_saved = 0.0
        self.total_lost  = 0.0
        self.chosen_destination = "Delhi APMC"
        self._raw_score = 0.0

    def reset(self, crop_type: str, task_id: str = "medium", seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)

        self.crop_type  = crop_type
        self.task_id    = task_id
        self.step_count = 0
        self.max_steps  = MAX_STEPS.get(task_id, 10)
        self.done       = False
        self.total_saved = 0.0
        self.total_lost  = 0.0
        self._raw_score  = 0.0
        self.chosen_destination = self._rng.choice(DESTINATIONS)

        # Generate batches
        n_batches = {"easy": 4, "medium": 6, "hard": 8}.get(task_id, 6)
        self.batches = []
        for i in range(n_batches):
            qty = self._rng.uniform(80, 200)
            self.batches.append({
                "batch_id":     f"B{i+1:03d}",
                "quantity_kg":  round(qty, 1),
                "quality":      round(self._rng.uniform(0.7, 1.0), 2),
                "spoilage_risk": round(self._rng.uniform(0.1, 0.9), 2),
                "dispatched":   False,
                "discarded":    False,
            })

        # Generate trucks with randomized failure step
        n_trucks = TRUCK_COUNT.get(task_id, 3)
        self.trucks = []
        for i in range(n_trucks):
            # Randomized failure: anywhere from step 3 to step 7
            fail_step = self._rng.randint(3, 7)
            self.trucks.append({
                "truck_id":    f"T{i+1:02d}",
                "available":   True,
                "capacity_kg": self._rng.uniform(300, 600),
                "fail_step":   fail_step,   # truck breaks at this step
            })

        return self._obs()

    def step(self, action_type: str, batch_id: Optional[str] = None,
             truck_id: Optional[str] = None, destination: Optional[str] = None):

        self.step_count += 1
        reward = 0.0
        message = ""

        # Apply truck failures at randomized step
        for truck in self.trucks:
            if self.step_count == truck["fail_step"] and truck["available"]:
                truck["available"] = False
                message += f"Truck {truck['truck_id']} broke down. "

        # Apply spoilage to all undispatched batches
        rate = SPOILAGE_RATE.get(self.task_id, 0.05)
        for batch in self.batches:
            if not batch["dispatched"] and not batch["discarded"]:
                loss = batch["quantity_kg"] * rate * batch["spoilage_risk"]
                batch["quantity_kg"] = max(0.0, batch["quantity_kg"] - loss)
                batch["spoilage_risk"] = min(1.0, batch["spoilage_risk"] + 0.05)
                self.total_lost += loss
                reward -= loss * 0.01   # small per-step spoilage penalty

        if action_type == "dispatch" and batch_id and truck_id:
            reward += self._dispatch(batch_id, truck_id, destination)
            message += f"Dispatched {batch_id} via {truck_id}."

        elif action_type == "reroute" and destination:
            self.chosen_destination = destination
            # Real price multiplier benefit for chosen destination + crop
            mult    = MARKET_MULTIPLIERS.get(destination, {}).get(self.crop_type, 1.0)
            bonus   = (mult - 1.0) * 10.0   # real differentiation, not flat 5%
            reward += bonus
            message += f"Rerouted to {destination}. Market multiplier: {mult:.2f}x."

        elif action_type == "discard" and batch_id:
            reward += self._discard(batch_id)
            message += f"Discarded {batch_id}."

        else:
            message += "Stored produce this step."

        self._raw_score += reward

        # Episode ends when max steps reached or all batches handled
        all_handled = all(b["dispatched"] or b["discarded"] for b in self.batches)
        if self.step_count >= self.max_steps or all_handled:
            self.done = True
            # Count remaining undispatched as lost
            for b in self.batches:
                if not b["dispatched"] and not b["discarded"]:
                    self.total_lost += b["quantity_kg"]

        normalized_reward = max(-1.0, min(1.0, reward / FRESHCHAIN_REWARD_SCALE))
        return self._obs(), normalized_reward, self.done, message

    def _dispatch(self, batch_id: str, truck_id: str, destination: Optional[str]) -> float:
        batch = next((b for b in self.batches if b["batch_id"] == batch_id), None)
        truck = next((t for t in self.trucks if t["truck_id"] == truck_id), None)

        if not batch or not truck:
            return -0.5
        if batch["dispatched"] or batch["discarded"]:
            return -0.3
        if not truck["available"]:
            return -0.5

        batch["dispatched"] = True
        self.total_saved   += batch["quantity_kg"]

        if destination:
            self.chosen_destination = destination

        # Reward based on quality and spoilage risk prevented
        quality_bonus = batch["quality"] * 1.5
        urgency_bonus = batch["spoilage_risk"] * 1.0
        # FIX: truck stays available after dispatch; only fail_step disables trucks
        return round(quality_bonus + urgency_bonus, 3)

    def _discard(self, batch_id: str) -> float:
        batch = next((b for b in self.batches if b["batch_id"] == batch_id), None)
        if not batch or batch["dispatched"] or batch["discarded"]:
            return -0.3
        # Only reward discard if spoilage risk is very high (>0.8)
        if batch["spoilage_risk"] > 0.8:
            batch["discarded"] = True
            self.total_lost   += batch["quantity_kg"]
            return 0.2   # small reward for smart triage
        else:
            batch["discarded"] = True
            self.total_lost   += batch["quantity_kg"]
            return -0.5   # penalty for discarding good produce

    def _obs(self) -> dict:
        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "crop_type": self.crop_type,
            "task_id": self.task_id,
            "chosen_destination": self.chosen_destination,
            "batches": self.batches,
            "trucks": self.trucks,
            "total_yield_saved_kg": round(self.total_saved, 1),
            "total_yield_lost_kg":  round(self.total_lost, 1),
            "done": self.done,
        }

    def state_dict(self) -> dict:
        return {
            "total_yield_saved_kg": self.total_saved,
            "total_yield_lost_kg":  self.total_lost,
            "step_count":           self.step_count,
            "fail_step":            self._rng.randint(3, 7),   # expose for state reconstruction
        }


# ---- OpenEnv Environment -----------------------------------------------------

class AgriChainEnv(BaseEnvironment):
    """
    AgriChain-RL -- OpenEnv BaseEnvironment compliant two-stage environment.

    Implements: reset(), step(), state(), close()
    Rewards normalized to -1 to +1 throughout.
    """

    SPEC = {
        "name": "AgriChain-RL",
        "version": "2.0.0",
        "openenv_spec": "0.1",
        "tagline": (
            "AgriChain-RL is a two-stage AI environment where an agent first manages "
            "crop logistics to reduce spoilage and then negotiates prices in a "
            "multi-agent market, simulating the complete journey from farm to sale."
        ),
        "author": "Manasviii27",
        "themes_covered": [
            "Theme 1 -- Multi-Agent: 3 buyer agents vs farmer agent in Mandi stage",
            "Theme 2 -- Long-Horizon Planning: Stage 1 decisions affect Stage 2 outcomes",
            "Theme 3 -- World Modeling: full agri economic simulation with feedback",
        ],
        "reward_normalization": "All rewards normalized to [-1, +1]",
        "stages": {
            "freshchain": "Post-harvest logistics -- minimize spoilage, choose best market",
            "mandi": "Multi-agent price negotiation -- maximize sale price",
        },
        "linking_mechanism": {
            "quality_score": "Spoilage-adjusted yield ratio -> farmer leverage",
            "quantity_kg": "Saved produce -> negotiation volume",
            "days_since_harvest": "Steps used + transit days -> time pressure",
            "destination": "Reroute choices in Stage 1 -> market price multiplier in Stage 2",
        },
        "action_spaces": {
            "freshchain": ["dispatch", "store", "reroute", "discard"],
            "mandi": ["accept", "counter", "sell_partial", "reject_all"],
        },
        "normalized_reward_range": [-1.0, 1.0],
        "destinations": DESTINATIONS,
        "interfaces": ["HTTP (OpenEnv)", "WebSocket (/ws)"],
    }

    def __init__(self):
        self._rng        = random.Random()
        self._freshchain = FreshChainSim()
        self._mandi      = MandiNegotiateEnv()
        self._stage      = Stage.FRESHCHAIN
        self._ep_raw     = 0.0
        self._ep_norm    = 0.0
        self._done       = False
        self._crop_type  = "tomato"
        self._fc_output: Optional[FreshChainOutput] = None
        self._last_fc_obs: Optional[dict] = None

    async def reset(self, action: Optional[AgriChainAction] = None) -> AgriChainObservation:
        task_id = (action.task_id or "medium") if action else "medium"
        seed    = action.seed if action else None
        if seed is not None:
            self._rng.seed(seed)

        self._stage     = Stage.FRESHCHAIN
        self._ep_raw    = 0.0
        self._ep_norm   = 0.0
        self._done      = False
        self._fc_output = None

        self._crop_type = self._rng.choice(CROPS)
        fc_obs = self._freshchain.reset(self._crop_type, task_id, seed)
        self._last_fc_obs = fc_obs

        return AgriChainObservation(
            stage=Stage.FRESHCHAIN,
            episode_raw_score=0.0,
            episode_normalized_score=0.0,
            done=False,
            message=(
                f"AgriChain Episode Started. Crop: {self._crop_type.upper()} | Task: {task_id.upper()}\n"
                f"STAGE 1 -- FreshChain: Dispatch batches before they spoil.\n"
                f"Destinations available: {', '.join(DESTINATIONS)}\n"
                f"What you save here determines your mandi leverage."
            ),
            freshchain_obs=fc_obs,
        )

    async def step(self, action: AgriChainAction) -> tuple[AgriChainObservation, float, bool]:
        if self._done:
            return await self.state(), 0.0, True

        if self._stage == Stage.FRESHCHAIN:
            return await self._step_freshchain(action)
        elif self._stage == Stage.MANDI:
            return await self._step_mandi(action)

        return await self.state(), 0.0, True

    async def _step_freshchain(self, action: AgriChainAction):
        fc_obs, norm_reward, done, message = self._freshchain.step(
            action_type=action.freshchain_action_type or "store",
            batch_id=action.batch_id,
            truck_id=action.truck_id,
            destination=action.destination,
        )
        self._last_fc_obs = fc_obs
        self._ep_raw  += norm_reward * FRESHCHAIN_REWARD_SCALE
        self._ep_norm += norm_reward

        if done:
            # Build linking output
            self._fc_output = FreshChainOutput(
                self._freshchain.state_dict(),
                self._crop_type,
                self._freshchain.chosen_destination,
            )
            self._stage = Stage.MANDI

            mandi_obs = self._mandi.reset(
                crop_type=self._crop_type,
                quantity_kg=self._fc_output.quantity_kg,
                quality_score=self._fc_output.quality_score,
                days_since_harvest=self._fc_output.days_since_harvest,
                transport_cost=self._fc_output.transport_cost,
                destination=self._fc_output.destination,
            )

            leverage_msg = {
                "strong":   "Strong leverage. You saved most of your produce.",
                "moderate": "Moderate leverage. Negotiate carefully.",
                "weak":     "Weak leverage. Some spoilage hurt your position.",
                "desperate":"Desperate position. Accept reasonable offers quickly.",
            }.get(mandi_obs.farmer_leverage, "")

            return AgriChainObservation(
                stage=Stage.MANDI,
                episode_raw_score=round(self._ep_raw, 2),
                episode_normalized_score=round(self._ep_norm, 4),
                done=False,
                message=(
                    f"STAGE 1 COMPLETE -- Grade: {self._fc_output.freshchain_grade}\n"
                    f"Saved: {self._fc_output.quantity_kg}kg | "
                    f"Quality: {self._fc_output.quality_score} | "
                    f"Days: {self._fc_output.days_since_harvest} | "
                    f"Destination: {self._fc_output.destination}\n\n"
                    f"STAGE 2 -- Mandi Negotiation: {leverage_msg}\n"
                    f"Market rate: Rs{mandi_obs.market_rate_today}/kg | MSP: Rs{mandi_obs.msp}/kg"
                ),
                freshchain_output=self._fc_output.to_dict(),
                mandi_obs=mandi_obs.model_dump(),
            ), norm_reward, False

        return AgriChainObservation(
            stage=Stage.FRESHCHAIN,
            episode_raw_score=round(self._ep_raw, 2),
            episode_normalized_score=round(self._ep_norm, 4),
            done=False,
            message=message,
            freshchain_obs=fc_obs,
        ), norm_reward, False

    async def _step_mandi(self, action: AgriChainAction):
        mandi_action = MandiAction(
            action_type=FarmerAction(action.mandi_action_type or "reject_all"),
            target_buyer_id=action.target_buyer_id,
            counter_price=action.counter_price,
        )
        mandi_obs, norm_reward, done = self._mandi.step(mandi_action)
        self._ep_raw  += mandi_obs.raw_reward
        self._ep_norm += norm_reward

        if done:
            self._done  = True
            self._stage = Stage.COMPLETE
            summary     = self._build_summary()

            return AgriChainObservation(
                stage=Stage.COMPLETE,
                episode_raw_score=round(self._ep_raw, 2),
                episode_normalized_score=round(self._ep_norm, 4),
                done=True,
                message=self._build_final_message(summary),
                freshchain_output=self._fc_output.to_dict() if self._fc_output else None,
                mandi_obs=mandi_obs.model_dump(),
                final_summary=summary,
            ), norm_reward, True

        return AgriChainObservation(
            stage=Stage.MANDI,
            episode_raw_score=round(self._ep_raw, 2),
            episode_normalized_score=round(self._ep_norm, 4),
            done=False,
            message=mandi_obs.message,
            freshchain_output=self._fc_output.to_dict() if self._fc_output else None,
            mandi_obs=mandi_obs.model_dump(),
        ), norm_reward, False

    async def state(self) -> AgriChainObservation:
        return AgriChainObservation(
            stage=self._stage,
            episode_raw_score=round(self._ep_raw, 2),
            episode_normalized_score=round(self._ep_norm, 4),
            done=self._done,
            message="Current state.",
            freshchain_obs=self._last_fc_obs,
            freshchain_output=self._fc_output.to_dict() if self._fc_output else None,
        )

    async def close(self):
        self._done = True

    def _build_summary(self) -> dict:
        mg      = self._mandi.grade()
        fc_sc   = self._fc_output.freshchain_score if self._fc_output else 0.0
        m_sc    = mg["score"]
        combined = round(0.4 * fc_sc + 0.6 * m_sc, 3)
        grade    = "S" if combined >= 0.90 else "A" if combined >= 0.75 else \
                   "B" if combined >= 0.60 else "C" if combined >= 0.45 else "F"
        return {
            "combined_score":    combined,
            "combined_grade":    grade,
            "freshchain_score":  round(fc_sc, 3),
            "mandi_score":       round(m_sc, 3),
            "total_revenue_inr": mg.get("total_revenue", 0),
            "avg_price_per_kg":  mg.get("avg_price_per_kg", 0),
            "market_rate":       mg.get("market_rate", 0),
            "price_ratio":       mg.get("price_ratio", 0),
            "episode_normalized_score": round(self._ep_norm, 4),
            "crop_type":         self._crop_type,
            "destination":       self._fc_output.destination if self._fc_output else "",
            "r1_price_ratio":       mg.get("r1_price_ratio", 0.0),
            "r2_sell_completeness": mg.get("r2_sell_completeness", 0.0),
            "r3_negotiation":       mg.get("r3_negotiation", 0.0),
            "r4_timeout":           mg.get("r4_timeout", 1.0),
        }

    def _build_final_message(self, s: dict) -> str:
        return (
            f"AGRICHAIN EPISODE COMPLETE\n"
            f"Crop: {s['crop_type'].upper()} | Destination: {s['destination']}\n"
            f"Combined Grade: {s['combined_grade']} ({s['combined_score']})\n\n"
            f"Stage 1 (FreshChain): {s['freshchain_score']:.2f}/1.0\n"
            f"Stage 2 (Mandi):      {s['mandi_score']:.2f}/1.0\n\n"
            f"Revenue: Rs{s['total_revenue_inr']:,.0f} | "
            f"Avg Price: Rs{s['avg_price_per_kg']}/kg | "
            f"Market Rate: Rs{s['market_rate']}/kg | "
            f"Price Ratio: {s['price_ratio']:.1%}\n"
            f"Normalized Episode Score: {s['episode_normalized_score']}"
        )


# ---- OpenEnv App -------------------------------------------------------------

app = create_app(
    AgriChainEnv,
    AgriChainAction,
    AgriChainObservation,
    max_concurrent_envs=32,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"name": "AgriChain-RL", "status": "running", "version": "2.0.0"}


@app.get("/config")
async def config():
    return JSONResponse(content=AgriChainEnv.SPEC)


# ---- WebSocket ---------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket interface for AgriChain-RL.

    Protocol:
        {"type": "reset", "task_id": "medium", "seed": 42}
        {"type": "step", "stage": "freshchain", "freshchain_action_type": "dispatch",
         "batch_id": "B001", "truck_id": "T01", "destination": "Mumbai APMC"}
        {"type": "step", "stage": "mandi", "mandi_action_type": "counter",
         "counter_price": 28.5}
        {"type": "state"}
        {"type": "config"}
        {"type": "grade"}
        {"type": "close"}
    """
    await websocket.accept()
    env = AgriChainEnv()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "invalid JSON"})
                continue

            t = msg.get("type", "")

            if t == "reset":
                action = AgriChainAction(
                    task_id=msg.get("task_id", "medium"),
                    seed=msg.get("seed"),
                )
                obs = await env.reset(action)
                await websocket.send_json({"type": "reset", "observation": obs.model_dump()})

            elif t == "step":
                action = AgriChainAction(
                    stage=msg.get("stage", Stage.FRESHCHAIN),
                    freshchain_action_type=msg.get("freshchain_action_type"),
                    batch_id=msg.get("batch_id"),
                    truck_id=msg.get("truck_id"),
                    destination=msg.get("destination"),
                    mandi_action_type=msg.get("mandi_action_type"),
                    target_buyer_id=msg.get("target_buyer_id"),
                    counter_price=msg.get("counter_price"),
                )
                obs, reward, done = await env.step(action)
                await websocket.send_json({
                    "type": "step",
                    "observation": obs.model_dump(),
                    "reward": round(reward, 6),
                    "done": done,
                })

            elif t == "state":
                obs = await env.state()
                await websocket.send_json({"type": "state", "observation": obs.model_dump()})

            elif t == "config":
                await websocket.send_json({"type": "config", "config": AgriChainEnv.SPEC})

            elif t == "grade":
                if env._fc_output:
                    await websocket.send_json({"type": "grade", "grade": env._build_summary()})
                else:
                    await websocket.send_json({"type": "grade", "grade": {"combined_score": 0.0, "combined_grade": "F"}})

            elif t == "close":
                await env.close()
                await websocket.send_json({"type": "closed"})
                break

            else:
                await websocket.send_json({"type": "error", "message": f"unknown: {t}"})

    except WebSocketDisconnect:
        await env.close()
