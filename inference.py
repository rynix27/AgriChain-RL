"""
inference.py -- AgriChain-RL inference entry point

Loads the trained LoRA model (if available) and serves actions.
Falls back to the greedy policy if no model is found.

Usage:
    python inference.py                    # greedy fallback demo
    python inference.py --model <hf_id>   # load trained model from HF Hub
    python inference.py --model <hf_id> --demo  # run full episode
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- Model state (lazy-loaded) -----------------------------------------------

_model     = None
_tokenizer = None
_use_model = False

SYSTEM_PROMPT = (
    "You are an AI agent for Indian farmers. Minimize spoilage. Maximize sale price.\n\n"
    "FreshChain JSON only:\n"
    '{"action":"dispatch","batch_id":"B001","truck_id":"T01"}\n'
    'or {"action":"store"} or {"action":"reroute","dest":"Mumbai APMC"}\n\n'
    "Mandi JSON only:\n"
    '{"action":"accept","buyer_id":"B1"} or {"action":"counter","price":28.5} or {"action":"reject_all"}\n\n'
    "JSON only. No explanation."
)


def load_model(model_name_or_path="Manasviii27/agrichain-rl-grpo"):
    """Load the trained LoRA model via Unsloth. Call once at startup."""
    global _model, _tokenizer, _use_model
    try:
        from unsloth import FastLanguageModel
        print(f"[inference] Loading: {model_name_or_path}")
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=512,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)
        _use_model = True
        print("[inference] Model ready.")
    except Exception as e:
        print(f"[inference] Load failed ({e}) -- using greedy fallback.")
        _use_model = False


def _build_prompt(obs):
    stage = obs.get("stage", "freshchain")
    if stage == "freshchain":
        b = obs.get("batches", [])
        t = obs.get("trucks", [])
        p  = f"FreshChain step {obs.get('step',0)}/{obs.get('max_steps',10)}"
        p += f" | {obs.get('crop','?')} | dest:{obs.get('dest','?')}\n"
        p += f"Trucks:{[x['id'] for x in t]} Saved:{obs.get('saved',0)}kg Lost:{obs.get('lost',0)}kg\n"
        for x in sorted(b, key=lambda x: x.get("risk", x.get("spoilage_risk", 0)), reverse=True)[:3]:
            bid = x.get("id", x.get("batch_id", "?"))
            qty = x.get("qty", x.get("quantity_kg", 0))
            rsk = x.get("risk", x.get("spoilage_risk", 0))
            p  += f"  {bid}:{qty:.0f}kg risk={rsk:.2f}\n"
        return p + "Action:"
    else:
        p  = f"Mandi round {obs.get('m_round', obs.get('round',0))}/8"
        p += f" | qual:{obs.get('quality', obs.get('quality_score',0.8)):.2f}"
        p += f" lev:{obs.get('leverage', obs.get('farmer_leverage','?'))}\n"
        p += f"Market:Rs{obs.get('market', obs.get('market_rate_today',0))}/kg"
        p += f" MSP:Rs{obs.get('msp',0)}/kg active:{obs.get('active_buyers',0)}\n"
        for b in obs.get("buyers", []):
            if not b.get("gone", b.get("walked_away", False)):
                hint  = b.get("hint", b.get("strategy_hint", "?"))
                offer = b.get("offer", b.get("current_offer", 0))
                rl    = b.get("rl",   b.get("rounds_until_walkaway", "?"))
                p    += f"  {b['id']}({hint}):Rs{offer}/kg {rl}rds\n"
        p += f"Best:Rs{obs.get('best_offer',0)}/kg from {obs.get('best_id', obs.get('best_buyer_id',''))}\nAction:"
        return p


def _parse(resp, stage):
    try:
        s = resp.find("{"); e = resp.rfind("}") + 1
        if s < 0 or e <= 0: raise ValueError
        p = json.loads(resp[s:e])
        a = p.get("action", "store")
        if stage == "freshchain":
            return {"action_type": a, "batch_id": p.get("batch_id"),
                    "truck_id": p.get("truck_id"), "destination": p.get("dest")}
        return {"action_type": a, "target_buyer_id": p.get("buyer_id"),
                "counter_price": p.get("price")}
    except Exception:
        return {"action_type": "store" if stage == "freshchain" else "reject_all"}


def _model_inference(obs):
    import torch
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(obs)}]
    text   = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=48, do_sample=True,
                              temperature=0.7, pad_token_id=_tokenizer.eos_token_id)
    resp = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse(resp, obs.get("stage", "freshchain"))


def _greedy_inference(obs):
    stage = obs.get("stage", "freshchain")
    if stage == "freshchain":
        batches = obs.get("batches", [])
        trucks  = [t for t in obs.get("trucks", [])
                   if t.get("avail", t.get("available", False))]
        if not batches or not trucks:
            return {"action_type": "store"}
        target = sorted(batches,
                        key=lambda b: b.get("risk", b.get("spoilage_risk", 0)),
                        reverse=True)[0]
        return {"action_type": "dispatch",
                "batch_id":    target.get("id", target.get("batch_id", "")),
                "truck_id":    trucks[0].get("id", trucks[0].get("truck_id", ""))}
    # Mandi
    market_rate = obs.get("market", obs.get("market_rate_today", 20.0))
    best_offer  = obs.get("best_offer", 0.0)
    best_buyer  = obs.get("best_id",    obs.get("best_buyer_id", ""))
    leverage    = obs.get("leverage",   obs.get("farmer_leverage", "weak"))
    round_num   = obs.get("m_round",    obs.get("round", 1))
    msp         = obs.get("msp", 10.0)
    active      = obs.get("active_buyers", 0)
    if active == 0:                                        return {"action_type": "reject_all"}
    if best_offer >= market_rate * 0.85:                  return {"action_type": "accept", "target_buyer_id": best_buyer}
    if leverage in ["strong", "moderate"] and round_num <= 4:
        return {"action_type": "counter", "counter_price": round(market_rate * 0.92, 2)}
    if best_offer >= msp:                                  return {"action_type": "accept", "target_buyer_id": best_buyer}
    return {"action_type": "reject_all"}


# ---- Public API --------------------------------------------------------------

def inference(observation: dict) -> dict:
    """
    Run one inference step on AgriChain-RL.

    Uses the trained LLM if load_model() was called, otherwise greedy fallback.

    Args:
        observation: raw env obs dict (from WebSocket or env._obs())
    Returns:
        action dict with action_type and optional fields
    """
    if _use_model and _model is not None:
        return _model_inference(observation)
    return _greedy_inference(observation)


def demo_episode():
    """Run a quick self-test with greedy policy."""
    test_cases = [
        {"stage": "freshchain",
         "batches": [{"id": "B001", "qty": 150.0, "qual": 0.9, "risk": 0.75, "disp": False, "disc": False},
                     {"id": "B002", "qty": 90.0,  "qual": 0.7, "risk": 0.30, "disp": False, "disc": False}],
         "trucks": [{"id": "T01", "avail": True}, {"id": "T02", "avail": True}],
         "saved": 0.0, "lost": 5.0, "step": 1, "max_steps": 10, "crop": "tomato", "dest": "Mumbai APMC"},
        {"stage": "mandi", "market": 28.0, "msp": 12.0, "best_offer": 22.0, "best_id": "B1",
         "leverage": "strong", "m_round": 2, "active_buyers": 2,
         "buyers": [{"id": "B1", "hint": "moderate", "offer": 22.0, "gone": False, "rl": 4},
                    {"id": "B2", "hint": "aggressive","offer": 18.0, "gone": False, "rl": 3}]},
        {"stage": "mandi", "market": 28.0, "msp": 12.0, "best_offer": 26.5, "best_id": "B1",
         "leverage": "weak", "m_round": 6, "active_buyers": 1,
         "buyers": [{"id": "B1", "hint": "moderate", "offer": 26.5, "gone": False, "rl": 1}]},
    ]
    expected = ["dispatch", "counter", "accept"]
    all_pass  = True
    for i, (obs, exp) in enumerate(zip(test_cases, expected)):
        result = inference(obs)
        ok     = result["action_type"] == exp
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Test {i+1}: stage={obs['stage']} -> {result['action_type']} (expected {exp})")
        if not ok: all_pass = False
    print(f"\nAll tests {'passed' if all_pass else 'FAILED'}.")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="AgriChain-RL inference")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model id or local path (e.g. Manasviii27/agrichain-rl-grpo)")
    parser.add_argument("--demo", action="store_true", help="Run self-test")
    args = parser.parse_args()
    if args.model:
        load_model(args.model)
    else:
        print("[inference] No --model given -- greedy policy active.")
    if args.demo:
        demo_episode()
    else:
        print("AgriChain-RL inference ready.")
        print("  from inference import inference, load_model")
        print("  load_model('Manasviii27/agrichain-rl-grpo')  # load trained model")
        print("  action = inference(obs_dict)")


if __name__ == "__main__":
    main()
