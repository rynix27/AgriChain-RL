"""
FreshChain Post-Harvest Yield Loss Environment
Models: Action, Observation, State

These are the typed data structures that define what the AI agent
can SEE (Observation) and what it can DO (Action).
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# OBSERVATION — What the agent sees each step
# ─────────────────────────────────────────────

class BatchInfo(BaseModel):
    """Information about a single produce batch in the warehouse."""
    batch_id: str = Field(description="Unique identifier for this batch")
    crop_type: str = Field(description="Type of crop (e.g., tomato, potato, onion)")
    quantity_kg: float = Field(description="Remaining quantity in kilograms")
    temperature_c: float = Field(description="Current storage temperature in Celsius")
    humidity_pct: float = Field(description="Current humidity percentage")
    spoilage_risk: float = Field(description="Spoilage risk score 0.0 (safe) to 1.0 (critical)")
    days_in_storage: int = Field(description="Number of days this batch has been in storage")
    market_price_per_kg: float = Field(description="Current market price per kg in INR")


class TruckInfo(BaseModel):
    """Information about an available transport truck."""
    truck_id: str
    capacity_kg: float
    available: bool
    destination: str


class FreshChainObservation(BaseModel):
    """
    Full observation returned to the agent after each step.
    Think of this as everything the agent can 'see' about the warehouse.
    """
    step: int = Field(description="Current step number in this episode")
    batches: List[BatchInfo] = Field(description="List of all produce batches currently in storage")
    trucks: List[TruckInfo] = Field(description="List of available trucks for dispatch")
    total_yield_saved_kg: float = Field(description="Total kg successfully dispatched to market so far")
    total_yield_lost_kg: float = Field(description="Total kg lost to spoilage so far")
    storage_capacity_used_pct: float = Field(description="Percentage of warehouse storage currently used")
    message: str = Field(description="Human-readable description of what happened last step")
    done: bool = Field(description="True if the episode has ended")
    reward: float = Field(description="Reward received for the last action")
    task_score: float = Field(description="Current task completion score 0.0 to 1.0")


# ─────────────────────────────────────────────
# ACTION — What the agent can do each step
# ─────────────────────────────────────────────

class FreshChainAction(BaseModel):
    """
    Action the agent takes.
    The agent picks ONE action per step.

    action_type options:
      - "dispatch"  : Send a batch to market using a truck
      - "store"     : Keep a batch in storage (do nothing, wait)
      - "reroute"   : Change a batch's destination to a closer market
      - "discard"   : Write off a batch that is too far gone (prevents cascade spoilage)
    """
    action_type: str = Field(
        description="One of: dispatch, store, reroute, discard"
    )
    batch_id: Optional[str] = Field(
        default=None,
        description="ID of the batch to act on (required for dispatch/reroute/discard)"
    )
    truck_id: Optional[str] = Field(
        default=None,
        description="ID of the truck to use (required for dispatch)"
    )
    destination: Optional[str] = Field(
        default=None,
        description="New destination market (required for reroute)"
    )


# ─────────────────────────────────────────────
# STATE — Internal episode tracking
# ─────────────────────────────────────────────

class FreshChainState(BaseModel):
    """
    Internal state of the environment.
    Tracks episode metadata.
    """
    episode_id: str = Field(description="Unique ID for this episode")
    step_count: int = Field(default=0, description="Number of steps taken so far")
    task_id: str = Field(description="Which task is being run: easy, medium, or hard")
    max_steps: int = Field(description="Maximum steps allowed in this episode")
