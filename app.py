"""
app.py -- AgriChain-RL Interactive Gradio Demo
Judges can play a full two-stage episode live in the browser.
No WebSocket needed -- runs the sync env directly.
"""

import gradio as gr
import random, json, textwrap

# ── Self-contained environment (same as training notebook) ────────────────────

CROPS       = ["tomato","potato","onion","banana","mango","cauliflower"]
MARKET_RATE = {"tomato":(15,45),"potato":(10,22),"onion":(12,35),
               "banana":(18,40),"mango":(35,120),"cauliflower":(10,25)}
MSP         = {"tomato":12.0,"potato":8.0,"onion":10.0,
               "banana":15.0,"mango":30.0,"cauliflower":8.0}
DEST_MULT   = {"Delhi APMC":1.15,"Mumbai APMC":1.20,"Chennai APMC":1.10,
               "Kolkata APMC":1.08,"Bangalore APMC":1.12,"Local Mandi":0.90}
DESTINATIONS = list(DEST_MULT.keys())

class AgriChainEnv:
    def __init__(self): self._rng = random.Random()

    def reset(self, task="medium", seed=None):
        if seed is not None: self._rng.seed(seed)
        self.task=task; self.crop=self._rng.choice(CROPS)
        self.max_steps={"easy":12,"medium":10,"hard":8}[task]
        self.spoilage={"easy":0.03,"medium":0.05,"hard":0.08}[task]
        self.step_count=0; self.saved=0.0; self.lost=0.0
        self.dest=self._rng.choice(DESTINATIONS); self.fail_step=self._rng.randint(3,7)
        self.stage="freshchain"; self.task=task
        nb={"easy":4,"medium":6,"hard":8}[task]
        self.batches=[{"id":f"B{i+1:03d}","qty":round(self._rng.uniform(80,200),1),
                       "qual":round(self._rng.uniform(0.7,1.0),2),
                       "risk":round(self._rng.uniform(0.1,0.9),2),
                       "disp":False,"disc":False} for i in range(nb)]
        nt={"easy":4,"medium":3,"hard":2}[task]
        self.trucks=[{"id":f"T{i+1:02d}","avail":True} for i in range(nt)]
        lo,hi=MARKET_RATE.get(self.crop,(10,30))
        self.market=round(self._rng.uniform(lo,hi)*DEST_MULT.get(self.dest,1.0),2)
        self.msp=MSP.get(self.crop,10.0)
        self.m_round=0; self.quality=0.8; self.days=3
        self.revenue=0.0; self.qty_sold=0.0
        strats=["aggressive","moderate","desperate"]; self._rng.shuffle(strats)
        self.buyers=[{"id":f"B{i+1}","strat":strats[i],
                      "budget":round(self.market*self._rng.uniform(0.9,1.4),2),
                      "demand":round(self._rng.uniform(0.2,0.95),2),
                      "offer":0.0,"gone":False,"walkaway":self._rng.randint(5,7)}
                     for i in range(3)]
        self._update_offers(); return self._obs()

    def step_fc(self, action, batch_id=None, truck_id=None, dest=None):
        self.step_count+=1; r=0.0
        if self.step_count==self.fail_step:
            avail=[t for t in self.trucks if t["avail"]]
            if avail: avail[0]["avail"]=False
        for b in self.batches:
            if not b["disp"] and not b["disc"]:
                loss=b["qty"]*self.spoilage*b["risk"]
                b["qty"]=max(0,b["qty"]-loss); b["risk"]=min(1.0,b["risk"]+0.05)
                self.lost+=loss; r-=loss*0.01
        if action=="dispatch" and batch_id and truck_id:
            b=next((x for x in self.batches if x["id"]==batch_id),None)
            t=next((x for x in self.trucks  if x["id"]==truck_id), None)
            if b and t and t["avail"] and not b["disp"] and not b["disc"]:
                b["disp"]=True; self.saved+=b["qty"]
                r+=b["qual"]*1.5+b["risk"]*1.0
            else: r-=0.5
        elif action=="reroute" and dest:
            self.dest=dest; r+=(DEST_MULT.get(dest,1.0)-1.0)*10.0
        elif action=="discard" and batch_id:
            b=next((x for x in self.batches if x["id"]==batch_id),None)
            if b and not b["disp"]:
                b["disc"]=True; self.lost+=b["qty"]
                r+=0.2 if b["risk"]>0.8 else -0.5
        nr=max(-1.0,min(1.0,r/2.0))
        done=(self.step_count>=self.max_steps or
              all(b["disp"] or b["disc"] for b in self.batches))
        if done:
            tot=self.saved+self.lost; sr=self.lost/max(tot,1)
            self.quality=max(0.05,(1-sr)-min(0.3,self.step_count*0.02))
            self.days=self.step_count+2; self.stage="mandi"; self._update_offers()
        return self._obs(),nr,done

    def step_mandi(self, action, buyer_id=None, price=None):
        self.m_round+=1
        active=[b for b in self.buyers if not b["gone"]]
        best=max(active,key=lambda b:b["offer"]) if active else None
        if not active: return self._obs(),-1.0,True
        raw=0.0
        if action=="accept":
            tgt=next((b for b in active if b["id"]==buyer_id),best)
            if tgt:
                p=tgt["offer"]; ratio=p/max(self.market,1)
                self.revenue+=p*max(self.saved,1); self.qty_sold+=max(self.saved,1)
                raw=(100.0 if ratio>=0.95 else 60.0 if ratio>=0.80
                     else 30.0 if ratio>=(self.msp/self.market) else -20.0)
                return self._obs(),max(-1.0,min(1.0,raw/100.0)),True
        elif action=="counter" and price is not None:
            acc=[b for b in active if price<=b["budget"]*1.05]
            if acc:
                self.revenue+=price*max(self.saved,1); self.qty_sold+=max(self.saved,1)
                ratio=price/max(self.market,1)
                raw=80.0 if ratio>=0.90 else 50.0 if ratio>=0.75 else 20.0
                return self._obs(),max(-1.0,min(1.0,raw/100.0)),True
            else: raw=-3.0
        elif action=="sell_partial" and best:
            qty=max(self.saved,1)*0.5; p=best["offer"]
            self.revenue+=p*qty; self.qty_sold+=qty; self.saved-=qty
            raw=40.0 if p/max(self.market,1)>=0.85 else 20.0
        else:
            lev=self._lev(); raw=2.0 if lev in["strong","moderate"] else -5.0
        self.quality=max(0.01,self.quality-0.04); self.days+=1
        for b in self.buyers:
            if not b["gone"] and self.m_round>=b["walkaway"]: b["gone"]=True
        self._update_offers()
        done=self.m_round>=8
        if done and self.qty_sold==0:
            self.revenue+=max(self.saved,1)*self.msp; raw-=20.0
        return self._obs(),max(-1.0,min(1.0,raw/100.0)),done

    def _update_offers(self):
        base=self.market*self.quality
        for b in self.buyers:
            if b["gone"]: continue
            r=self.m_round
            if   b["strat"]=="aggressive": o=base*(1-max(0.04,0.35-r*0.025))
            elif b["strat"]=="moderate":   o=base*(1-max(0.01,0.15-r*0.035))
            else:                          o=base*(1+b["demand"]*0.18)
            b["offer"]=round(min(o,b["budget"]),2)

    def _lev(self):
        if   self.quality>=0.75 and self.days<=4: return "strong"
        elif self.quality>=0.50 and self.days<=7: return "moderate"
        elif self.quality>=0.25:                  return "weak"
        else:                                     return "desperate"

    def grade(self):
        if self.qty_sold==0: return {"score":0.0,"grade":"F","r1":0,"r2":0,"r3":0,"r4":0,"revenue":0}
        avg=self.revenue/self.qty_sold; ratio=avg/max(self.market,1)
        r1=round(min(1.0,ratio*0.3 if avg<self.msp else ratio),3)
        tot=self.qty_sold+max(self.saved,0)
        r2=round(self.qty_sold/max(tot,1),3)
        rb=min(1.0,self.m_round/4.0)
        r3=round((0.5+0.5*rb) if ratio>=0.80 else 0.2*rb,3)
        r4=0.0 if self.m_round>=8 and self.qty_sold==0 else (0.5 if self.m_round>=8 else 1.0)
        combined=round(0.40*r1+0.30*r2+0.20*r3+0.10*r4,3)
        g=("S" if combined>=0.90 else "A" if combined>=0.75 else
           "B" if combined>=0.60 else "C" if combined>=0.45 else "F")
        fc=self.saved/max(self.saved+self.lost,1)
        return {"score":round(0.4*fc+0.6*combined,3),"grade":g,
                "r1":r1,"r2":r2,"r3":r3,"r4":r4,"revenue":round(self.revenue,0)}

    def _obs(self):
        active=[b for b in self.buyers if not b["gone"]]
        best=max(active,key=lambda b:b["offer"]) if active else None
        undisp=[b for b in self.batches if not b["disp"] and not b["disc"]]
        avt=[t for t in self.trucks if t["avail"]]
        return {"stage":self.stage,"crop":self.crop,"step":self.step_count,
                "max_steps":self.max_steps,"task":self.task,
                "saved":round(self.saved,1),"lost":round(self.lost,1),
                "batches":undisp,"trucks":avt,"dest":self.dest,
                "market":self.market,"msp":self.msp,
                "m_round":self.m_round,"quality":round(self.quality,3),"days":self.days,
                "buyers":[{"id":b["id"],"hint":b["strat"],"offer":b["offer"],
                           "gone":b["gone"],"rl":max(0,b["walkaway"]-self.m_round)}
                          for b in self.buyers],
                "active_buyers":len(active),
                "best_offer":best["offer"] if best else 0.0,
                "best_id":best["id"] if best else "",
                "leverage":self._lev()}

# ── Greedy agent ──────────────────────────────────────────────────────────────

class GreedyAgent:
    def fc_action(self, obs):
        b=obs["batches"]; t=obs["trucks"]
        if not b or not t: return "store",{}
        best=sorted(b,key=lambda x:x["risk"],reverse=True)[0]
        return "dispatch",{"batch_id":best["id"],"truck_id":t[0]["id"]}
    def mandi_action(self, obs):
        mkt=obs["market"]; best=obs["best_offer"]; bid=obs["best_id"]
        lev=obs["leverage"]; rnd=obs["m_round"]; msp=obs["msp"]; act=obs["active_buyers"]
        if act==0: return "reject_all",{}
        if best>=mkt*0.85: return "accept",{"buyer_id":bid}
        if lev in["strong","moderate"] and rnd<=4: return "counter",{"price":round(mkt*0.92,2)}
        if best>=msp: return "accept",{"buyer_id":bid}
        return "reject_all",{}

# ── Global state ──────────────────────────────────────────────────────────────
_env   = AgriChainEnv()
_obs   = None
_log   = []
_done  = False
_agent = GreedyAgent()

def _fmt_obs(obs):
    """Format observation into readable status string."""
    if obs["stage"] == "freshchain":
        lines = [
            f"STAGE 1 -- FRESHCHAIN  |  Step {obs['step']}/{obs['max_steps']}",
            f"Crop: {obs['crop'].upper()}  |  Destination: {obs['dest']}",
            f"Market rate: Rs {obs['market']}/kg  |  MSP: Rs {obs['msp']}/kg",
            f"Saved: {obs['saved']} kg  |  Lost to spoilage: {obs['lost']} kg",
            "",
            "Available trucks: " + ", ".join(t["id"] for t in obs["trucks"]) or "NONE",
            "",
            "Batches remaining:",
        ]
        for b in obs["batches"]:
            bar = "#" * int(b["risk"]*10) + "." * (10-int(b["risk"]*10))
            lines.append(f"  {b['id']}  {b['qty']:.0f}kg  qual={b['qual']:.2f}  risk [{bar}] {b['risk']:.2f}")
    else:
        lines = [
            f"STAGE 2 -- MANDI NEGOTIATION  |  Round {obs['m_round']}/8",
            f"Crop: {obs['crop'].upper()}  |  Quality: {obs['quality']:.2f}  |  Leverage: {obs['leverage'].upper()}",
            f"Market rate: Rs {obs['market']}/kg  |  MSP: Rs {obs['msp']}/kg",
            f"Produce available: {obs['saved']:.0f} kg",
            "",
            "Buyer offers:",
        ]
        for b in obs["buyers"]:
            if b["gone"]:
                lines.append(f"  {b['id']} ({b['hint']:10s})  -- WALKED AWAY --")
            else:
                lines.append(f"  {b['id']} ({b['hint']:10s})  Rs {b['offer']:.2f}/kg  ({b['rl']} rounds left)")
        lines.append(f"\nBest offer: Rs {obs['best_offer']:.2f}/kg from {obs['best_id']}")
    return "\n".join(lines)

def _fmt_log():
    return "\n".join(_log[-30:]) if _log else "(no actions yet)"

# ── Gradio handlers ───────────────────────────────────────────────────────────

def reset_episode(task, seed_str):
    global _obs, _log, _done
    try:    seed = int(seed_str)
    except: seed = random.randint(0, 9999)
    _obs  = _env.reset(task=task, seed=seed)
    _log  = [f"Episode started. Task: {task} | Seed: {seed}",
             f"Crop: {_obs['crop']} | Market: Rs {_obs['market']}/kg | Dest: {_obs['dest']}",
             "-"*50]
    _done = False
    return _fmt_obs(_obs), _fmt_log(), "", gr.update(interactive=True)

def step_dispatch(batch_id, truck_id):
    global _obs, _log, _done
    if _done or _obs is None or _obs["stage"] != "freshchain":
        return _fmt_obs(_obs) if _obs else "", _fmt_log(), "Not in FreshChain stage."
    if not batch_id or not truck_id:
        return _fmt_obs(_obs), _fmt_log(), "Select a batch and truck first."
    _obs, r, _done = _env.step_fc("dispatch", batch_id=batch_id, truck_id=truck_id)
    _log.append(f"DISPATCH {batch_id} via {truck_id}  ->  reward {r:+.3f}")
    if _done and _obs["stage"]=="mandi":
        _log.append(f"Stage 1 complete. Quality: {_obs['quality']:.2f} | Leverage: {_obs['leverage']}")
        _log.append("-"*50)
    msg = f"Dispatched {batch_id} via {truck_id}. Reward: {r:+.3f}"
    if _done and _obs["stage"]!="mandi": msg += " | Episode ended."
    return _fmt_obs(_obs), _fmt_log(), msg

def step_store():
    global _obs, _log, _done
    if _done or _obs is None or _obs["stage"] != "freshchain":
        return _fmt_obs(_obs) if _obs else "", _fmt_log(), "Not in FreshChain stage."
    _obs, r, _done = _env.step_fc("store")
    _log.append(f"STORE (wait)  ->  reward {r:+.3f}")
    return _fmt_obs(_obs), _fmt_log(), f"Stored all batches. Reward: {r:+.3f}"

def step_reroute(dest):
    global _obs, _log, _done
    if _done or _obs is None or _obs["stage"] != "freshchain":
        return _fmt_obs(_obs) if _obs else "", _fmt_log(), "Not in FreshChain stage."
    _obs, r, _done = _env.step_fc("reroute", dest=dest)
    _log.append(f"REROUTE to {dest}  ->  reward {r:+.3f}")
    return _fmt_obs(_obs), _fmt_log(), f"Rerouted to {dest}. Reward: {r:+.3f}"

def step_accept(buyer_id):
    global _obs, _log, _done
    if _done or _obs is None or _obs["stage"] != "mandi":
        return _fmt_obs(_obs) if _obs else "", _fmt_log(), "Not in Mandi stage."
    _obs, r, _done = _env.step_mandi("accept", buyer_id=buyer_id)
    _log.append(f"ACCEPT {buyer_id}  ->  reward {r:+.3f}")
    if _done: _log.append("Deal closed!"); _log.append(_grade_summary())
    return _fmt_obs(_obs), _fmt_log(), f"Accepted offer from {buyer_id}. Reward: {r:+.3f}"

def step_counter(price_str):
    global _obs, _log, _done
    if _done or _obs is None or _obs["stage"] != "mandi":
        return _fmt_obs(_obs) if _obs else "", _fmt_log(), "Not in Mandi stage."
    try:    price = float(price_str)
    except: return _fmt_obs(_obs), _fmt_log(), "Enter a valid price number."
    _obs, r, _done = _env.step_mandi("counter", price=price)
    _log.append(f"COUNTER Rs {price:.2f}/kg  ->  reward {r:+.3f}")
    if _done: _log.append(_grade_summary())
    return _fmt_obs(_obs), _fmt_log(), f"Counter offer at Rs {price:.2f}. Reward: {r:+.3f}"

def step_reject():
    global _obs, _log, _done
    if _done or _obs is None or _obs["stage"] != "mandi":
        return _fmt_obs(_obs) if _obs else "", _fmt_log(), "Not in Mandi stage."
    _obs, r, _done = _env.step_mandi("reject_all")
    _log.append(f"REJECT ALL  ->  reward {r:+.3f}")
    if _done: _log.append(_grade_summary())
    return _fmt_obs(_obs), _fmt_log(), f"Rejected all offers. Reward: {r:+.3f}"

def run_greedy():
    global _obs, _log, _done
    if _obs is None: return "", _fmt_log(), "Reset first."
    steps = 0
    while not _done:
        if _obs["stage"] == "freshchain":
            a,kw = _agent.fc_action(_obs); _obs,r,_done = _env.step_fc(a,**kw)
            _log.append(f"GREEDY {a.upper()} {kw}  ->  r={r:+.3f}")
        else:
            a,kw = _agent.mandi_action(_obs); _obs,r,_done = _env.step_mandi(a,**kw)
            _log.append(f"GREEDY {a.upper()} {kw}  ->  r={r:+.3f}")
        steps += 1
        if steps > 30: break
    _log.append(_grade_summary())
    return _fmt_obs(_obs), _fmt_log(), f"Greedy completed in {steps} steps."

def _grade_summary():
    g = _env.grade()
    return (f"\n{'='*40}\n"
            f"FINAL GRADE: {g['grade']}  |  Score: {g['score']:.3f}\n"
            f"Revenue: Rs {g['revenue']:,.0f}\n"
            f"R1 Price ratio: {g['r1']:.3f}  R2 Sell: {g['r2']:.3f}\n"
            f"R3 Negotiation: {g['r3']:.3f}  R4 Timeout: {g['r4']:.1f}\n"
            f"{'='*40}")

# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
.gr-button { font-family: monospace !important; }
.status-box { font-family: monospace !important; font-size: 13px !important; }
.log-box    { font-family: monospace !important; font-size: 12px !important; }
#header { text-align: center; padding: 24px 0 8px; }
#header h1 { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
#header p  { color: #888; font-size: 14px; }
"""

with gr.Blocks(css=CSS, title="AgriChain-RL") as demo:

    gr.HTML("""
    <div id="header">
      <h1>AgriChain-RL</h1>
      <p>Two-stage RL environment &mdash; Post-harvest logistics + Mandi price negotiation</p>
      <p style="font-size:12px;color:#aaa;">Meta PyTorch OpenEnv Hackathon 2026 &nbsp;|&nbsp; Built by Manasviii27</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            status = gr.Textbox(label="Environment State", lines=18,
                                elem_classes=["status-box"], interactive=False,
                                value="Press Reset to start a new episode.")
            result_msg = gr.Textbox(label="Last Action Result", lines=1, interactive=False)

        with gr.Column(scale=1):
            action_log = gr.Textbox(label="Action Log", lines=18,
                                    elem_classes=["log-box"], interactive=False)

    gr.HTML("<hr style='border-color:#333;margin:8px 0'>")

    with gr.Row():
        task_dd   = gr.Dropdown(["easy","medium","hard"], value="medium", label="Difficulty")
        seed_box  = gr.Textbox(value="42", label="Seed (integer)")
        reset_btn = gr.Button("RESET EPISODE", variant="primary")

    gr.HTML("<p style='font-size:12px;color:#888;margin:4px 0'>Stage 1 -- FreshChain Actions</p>")
    with gr.Row():
        batch_dd = gr.Dropdown(
            [f"B{i+1:03d}" for i in range(8)], value="B001", label="Batch ID")
        truck_dd = gr.Dropdown(
            ["T01","T02","T03","T04"], value="T01", label="Truck ID")
        dest_dd  = gr.Dropdown(DESTINATIONS, value="Mumbai APMC", label="Reroute destination")
    with gr.Row():
        dispatch_btn = gr.Button("DISPATCH")
        store_btn    = gr.Button("STORE (wait)")
        reroute_btn  = gr.Button("REROUTE")

    gr.HTML("<p style='font-size:12px;color:#888;margin:4px 0'>Stage 2 -- Mandi Actions</p>")
    with gr.Row():
        buyer_dd   = gr.Dropdown(["B1","B2","B3"], value="B1", label="Buyer ID")
        price_box  = gr.Textbox(value="28.5", label="Counter price (Rs/kg)")
    with gr.Row():
        accept_btn  = gr.Button("ACCEPT")
        counter_btn = gr.Button("COUNTER OFFER")
        reject_btn  = gr.Button("REJECT ALL")

    gr.HTML("<hr style='border-color:#333;margin:8px 0'>")
    greedy_btn = gr.Button("RUN GREEDY AGENT (auto-complete episode)", variant="secondary")

    # ── Wire up ───────────────────────────────────────────────────────────────
    outs = [status, action_log, result_msg]

    reset_btn.click(reset_episode, [task_dd, seed_box],
                    [status, action_log, result_msg, dispatch_btn])

    dispatch_btn.click(step_dispatch, [batch_dd, truck_dd], outs)
    store_btn.click(step_store,   [], outs)
    reroute_btn.click(step_reroute, [dest_dd], outs)
    accept_btn.click(step_accept,  [buyer_dd],   outs)
    counter_btn.click(step_counter, [price_box], outs)
    reject_btn.click(step_reject,  [], outs)
    greedy_btn.click(run_greedy,   [], outs)

    gr.HTML("""
    <div style="font-size:11px;color:#666;text-align:center;padding:16px 0 8px">
      Training: Qwen2.5-7B + GRPO + Unsloth &nbsp;|&nbsp;
      Results: updated after onsite training — see README
      Improvement: +2.067
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
