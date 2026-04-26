"""
example_agent.py -- Greedy baseline agent for AgriChain-RL

Runs a full two-stage episode:
    Stage 1: Dispatch highest-risk batches first (FreshChain)
    Stage 2: Counter at 90% market rate, accept best offer (Mandi)

Usage:
    python example_agent.py
    python example_agent.py --task hard --seed 42
    python example_agent.py --url wss://Manasviii27-agrichain-rl.hf.space/ws
"""

import asyncio
import argparse
import json
import websockets


class GreedyAgriChainAgent:

    def choose_freshchain_action(self, obs: dict) -> dict:
        batches = obs.get("batches", [])
        trucks  = [t for t in obs.get("trucks", []) if t["available"]]

        if not batches or not trucks:
            return {
                "type": "step",
                "stage": "freshchain",
                "freshchain_action_type": "store",
            }

        batches_sorted = sorted(batches, key=lambda b: b["spoilage_risk"], reverse=True)
        target_batch   = batches_sorted[0]
        target_truck   = trucks[0]

        return {
            "type": "step",
            "stage": "freshchain",
            "freshchain_action_type": "dispatch",
            "batch_id": target_batch["batch_id"],
            "truck_id": target_truck["truck_id"],
        }

    def choose_mandi_action(self, obs: dict) -> dict:
        market_rate = obs.get("market_rate_today", 20.0)
        best_offer  = obs.get("best_offer", 0.0)
        best_buyer  = obs.get("best_buyer_id", "")
        leverage    = obs.get("farmer_leverage", "weak")
        round_num   = obs.get("round", 1)
        msp         = obs.get("msp", 10.0)

        if best_offer >= market_rate * 0.85:
            return {
                "type": "step",
                "stage": "mandi",
                "mandi_action_type": "accept",
                "target_buyer_id": best_buyer,
            }

        if leverage in ["strong", "moderate"] and round_num <= 4:
            counter = round(market_rate * 0.92, 2)
            return {
                "type": "step",
                "stage": "mandi",
                "mandi_action_type": "counter",
                "counter_price": counter,
            }

        if best_offer >= msp:
            return {
                "type": "step",
                "stage": "mandi",
                "mandi_action_type": "accept",
                "target_buyer_id": best_buyer,
            }

        return {
            "type": "step",
            "stage": "mandi",
            "mandi_action_type": "reject_all",
        }


async def run_episode(url: str, task: str, seed: int):
    agent = GreedyAgriChainAgent()

    async with websockets.connect(url) as ws:
        print(f"Connected | Task: {task} | Seed: {seed}")
        print("-" * 60)

        await ws.send(json.dumps({"type": "reset", "task_id": task, "seed": seed}))
        msg = json.loads(await ws.recv())
        obs = msg["observation"]
        print(obs["message"])
        print("-" * 60)

        done = False
        step = 0

        while not done:
            stage = obs["stage"]

            if stage == "freshchain":
                fc_obs = obs.get("freshchain_obs", {})
                action = agent.choose_freshchain_action(fc_obs)
            elif stage == "mandi":
                m_obs  = obs.get("mandi_obs", {})
                action = agent.choose_mandi_action(m_obs)
            else:
                break

            await ws.send(json.dumps(action))
            msg    = json.loads(await ws.recv())
            obs    = msg["observation"]
            reward = msg["reward"]
            done   = msg["done"]
            step  += 1

            if abs(reward) > 0.5 or obs["stage"] != stage:
                print(
                    f"Step {step:2d} | Stage: {obs['stage']:12s} | "
                    f"Reward: {reward:+7.3f} | EpScore: {obs.get('episode_normalized_score', obs.get('episode_raw_score', 0)):8.3f}"
                )
                if obs["stage"] != stage:
                    print(f"  {obs['message'][:120]}")

        print("-" * 60)
        print(obs["message"])

        await ws.send(json.dumps({"type": "grade"}))
        grade_msg = json.loads(await ws.recv())
        g = grade_msg["grade"]
        print(f"\nFinal Grade    : {g.get('combined_grade')} ({g.get('combined_score')})")
        print(f"FreshChain     : {g.get('freshchain_score')}")
        print(f"Mandi          : {g.get('mandi_score')}")
        print(f"Total Revenue  : Rs{g.get('total_revenue_inr', 0):,.0f}")

        await ws.send(json.dumps({"type": "close"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default="ws://localhost:7860/ws")
    parser.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    asyncio.run(run_episode(args.url, args.task, args.seed))
