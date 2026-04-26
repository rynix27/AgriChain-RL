# AgriChain-RL: We Trained an AI to Think Like an Indian Farmer

## The Problem Nobody Talks About

Every day, millions of Indian farmers wake up before dawn, load their produce onto a truck, and drive to the nearest mandi. They arrive with no price data, no negotiation experience, and produce that has been degrading since it left the field.

On the other side of the table sits a professional buyer. He negotiates crop prices 300 days a year. He knows the exact market rate. He knows how much supply arrived today. He knows which farmers are desperate and which ones can wait.

That information gap costs Indian farmers 30% of their income to spoilage and another 20% to unfair negotiation. Rs 1.5 lakh crore lost every year. 600 million people affected.

We built AgriChain-RL to close that gap.

---

## What We Built

AgriChain-RL is a two-stage reinforcement learning environment built on the OpenEnv framework. It simulates the complete journey of produce from farm to final sale price — and trains an AI agent to optimize every decision along the way.

**Stage 1: FreshChainEnv**

The agent manages produce batches across multiple steps. It decides when to dispatch, when to store, when to reroute to a better market, and when to discard critically spoiled produce. Trucks break down at random. Every step, unsaved produce degrades. The agent has to think ahead.

**The Linking Layer**

This is what makes AgriChain novel.

Every decision in Stage 1 has a direct causal effect on Stage 2. The quality score, quantity saved, days elapsed, and market destination all flow from Stage 1 into Stage 2 as farmer leverage inputs. An agent that manages logistics well arrives at the mandi with fresh produce and negotiating power. An agent that lets produce spoil arrives desperate.

**Stage 2: MandiNegotiateEnv**

The agent negotiates against 3 buyer agents simultaneously. Each buyer has a hidden budget, hidden demand level, and a walkaway round after which they leave if no deal is reached. The agent can accept, counter-offer, sell partial quantity, or reject and wait.

The quality score from Stage 1 directly sets the negotiating leverage. High quality = strong leverage = better prices. It is the same dynamic that plays out at every mandi in India, every day.

---

## Training

We trained a Qwen2.5-7B model using GRPO with Unsloth on a free T4 GPU.

**Results after 80 training steps (20-episode benchmark, medium difficulty):**

| Agent | Mean Score (0–1) | Grade distribution |
|---|---|---|
| Greedy baseline | 0.41 | 60% C, 30% F, 10% B |
| GRPO-trained LLM | 0.68 | 50% B, 35% A, 15% C |

The combined score improved from **0.41 → 0.68**, a **+66% gain** over the greedy baseline. The reward signal was near-zero for the first 10 steps (as expected on a fresh model), then climbed steadily from step 15 onward as the curriculum shifted from easy to medium tasks and the model began producing valid JSON actions consistently.

The trained agent learned three concrete behaviours the greedy policy cannot replicate: dispatching the highest-spoilage-risk batch first rather than the largest batch, holding firm on counter-price when produce quality is above 0.75, and recognising the aggressive buyer strategy from offer patterns and using `reject_all` to force a second round bid.

**What made it work:**

Five independent reward functions (R1–R5) prevented reward hacking — gaming any one check does not satisfy the others. Curriculum training from easy to hard gave the model non-zero reward signal from the very first steps, avoiding the stalled-learning failure mode. Both FreshChain and Mandi stages were included in every training prompt so the model learned the full two-stage episode rather than optimising each stage in isolation.

---

## Why This Matters

A trained AgriChain agent is not just a research demo. It is the foundation of:

- A WhatsApp bot that tells a farmer "your mango quality is 0.82, hold firm, counter at Rs 45/kg" during a live negotiation
- A mobile app that shows the optimal dispatch timeline for each crop in each region
- A government policy tool that identifies where information asymmetry costs farmers the most

The environment is fully open source. The trained model is on HuggingFace. Anyone can build on it.

---

## Links

- Live demo: https://huggingface.co/spaces/Manasviii27/agrichain-rl
- Trained model: https://huggingface.co/Manasviii27/agrichain-rl-grpo
- GitHub: https://github.com/Manasviii27/agrichain-rl

Built for the Meta PyTorch OpenEnv Hackathon x SST India AI Hackathon 2026.
