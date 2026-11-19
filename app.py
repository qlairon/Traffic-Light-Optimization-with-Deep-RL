"""
Streamlit UI — Traffic Light Optimization with DQN
====================================================
Interactive dashboard that trains the agent from the traffic_dqn package live.

    pip install -r requirements.txt
    streamlit run app.py
"""

import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from traffic_dqn import TrafficIntersection, DQNAgent

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Light DQN",
    page_icon="🚦",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  Custom CSS — clean dark-ish card style
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"]   { font-size: 1.6rem; }
[data-testid="stMetricLabel"]   { font-size: 0.75rem; color: #888; }
.phase-badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:0.8rem; font-weight:600; margin-left:8px;
}
.ns-green  { background:#d4edda; color:#1a6630; }
.ns-yellow { background:#fff3cd; color:#856404; }
.ew-green  { background:#cce5ff; color:#004085; }
.ew-yellow { background:#fff3cd; color:#856404; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Session-state initialisation
# ─────────────────────────────────────────────────────────────
def _init():
    if "agent" not in st.session_state:
        st.session_state.agent        = DQNAgent()
        st.session_state.env          = TrafficIntersection("uniform")
        st.session_state.state        = st.session_state.env.reset()
        st.session_state.episode      = 0
        st.session_state.ep_step      = 0
        st.session_state.ep_reward    = 0.0
        st.session_state.ep_wait      = 0.0
        st.session_state.ep_thru      = 0
        st.session_state.history      = {"rewards": [], "avg_wait": [], "losses": [], "throughput": []}
        st.session_state.running      = False
        st.session_state.trained      = False

_init()

STEPS_PER_EP = 200
PHASE_NAMES  = {0: "NS 🟢 Green", 1: "NS 🟡 Yellow", 2: "EW 🟢 Green", 3: "EW 🟡 Yellow"}
PHASE_CSS    = {0: "ns-green",    1: "ns-yellow",    2: "ew-green",    3: "ew-yellow"}

# ─────────────────────────────────────────────────────────────
#  Sidebar — hyperparameters & controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚦 DQN Traffic Light")
    st.caption("Single intersection · 4 lanes · ε-greedy DQN")
    st.divider()

    st.subheader("Environment")
    traffic_mode = st.selectbox(
        "Traffic mode",
        ["uniform", "rush_hour", "heavy", "imbalanced"],
        format_func=lambda x: {
            "uniform":    "Uniform (balanced)",
            "rush_hour":  "Rush Hour (N↔S heavy)",
            "heavy":      "Heavy (all directions)",
            "imbalanced": "Imbalanced (E heavy)",
        }[x],
    )

    st.subheader("Hyperparameters")
    lr            = st.select_slider("Learning rate (α)",    [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    gamma         = st.slider("Discount factor (γ)",  0.80, 0.999, 0.95, 0.005)
    eps_decay     = st.slider("ε decay rate",         0.990, 0.999, 0.995, 0.001)
    batch_size    = st.select_slider("Batch size",    [16, 32, 64, 128], value=64)
    target_update = st.select_slider("Target update (steps)", [50, 100, 200, 500], value=100)

    st.subheader("Training")
    n_episodes   = st.slider("Episodes",      50, 1000, 300, 50)
    steps_ep_ui  = st.slider("Steps / episode", 50, 500, 200, 50)
    speed        = st.slider("Simulation speed  (ep/render)", 1, 20, 5)

    st.divider()
    col_a, col_b = st.columns(2)

    start_btn = col_a.button(
        "▶ Start" if not st.session_state.running else "⏸ Pause",
        width="stretch", type="primary",
    )
    reset_btn = col_b.button("↺ Reset", width="stretch")

    if start_btn:
        if not st.session_state.trained or st.session_state.episode == 0:
            # Fresh run — rebuild agent & env with chosen params
            st.session_state.agent = DQNAgent(
                lr=lr, gamma=gamma,
                epsilon_decay=eps_decay,
                batch_size=batch_size,
                target_update=target_update,
            )
            st.session_state.env   = TrafficIntersection(traffic_mode)
            st.session_state.state = st.session_state.env.reset()
            st.session_state.episode   = 0
            st.session_state.ep_step   = 0
            st.session_state.ep_reward = 0.0
            st.session_state.ep_wait   = 0.0
            st.session_state.ep_thru   = 0
            st.session_state.history   = {"rewards": [], "avg_wait": [], "losses": [], "throughput": []}
        st.session_state.running = not st.session_state.running

    if reset_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        _init()
        st.rerun()

# ─────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────
phase     = st.session_state.env.phase
phase_css = PHASE_CSS[phase]
phase_lbl = PHASE_NAMES[phase]

st.markdown(
    f"## Traffic Light Optimization — DQN &nbsp;"
    f'<span class="phase-badge {phase_css}">{phase_lbl}</span>',
    unsafe_allow_html=True,
)
st.caption(f"Episode **{st.session_state.episode}** / {n_episodes} &nbsp;·&nbsp; "
           f"ε = **{st.session_state.agent.epsilon:.3f}** &nbsp;·&nbsp; "
           f"Replay buffer: **{len(st.session_state.agent.buffer)}** transitions")

# ─────────────────────────────────────────────────────────────
#  Top metrics row
# ─────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
q = st.session_state.env.queues
total_q = sum(q.values())
avg_wait = (st.session_state.ep_wait / max(st.session_state.ep_step, 1))
ep_thru  = st.session_state.ep_thru

hist = st.session_state.history
prev_reward = hist["rewards"][-2] if len(hist["rewards"]) >= 2 else None
cur_reward  = hist["rewards"][-1] if hist["rewards"] else 0

m1.metric("Episode reward",    f"{st.session_state.ep_reward:.0f}",
          delta=f"{cur_reward - prev_reward:.0f}" if prev_reward is not None else None)
m2.metric("Avg wait (queue)",  f"{avg_wait:.2f} veh")
m3.metric("Throughput",        f"{ep_thru} veh")
m4.metric("Total queue now",   f"{total_q} veh")
m5.metric("Episodes done",     st.session_state.episode)

st.divider()

# ─────────────────────────────────────────────────────────────
#  Main layout — intersection | charts
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

# ── Left: Intersection diagram ──────────────────────────────
with left:
    st.subheader("Intersection view")

    phase = st.session_state.env.phase
    ns_color = "#2ecc71" if phase == 0 else "#f1c40f" if phase == 1 else "#e74c3c"
    ew_color = "#2ecc71" if phase == 2 else "#f1c40f" if phase == 3 else "#e74c3c"
    qN, qS, qE, qW = q["N"], q["S"], q["E"], q["W"]

    # SVG intersection
    def car_row(count, x, y, dx, dy, color="#4a9fd4"):
        svgs = []
        for i in range(min(count, 5)):
            cx, cy = x + i*dx, y + i*dy
            svgs.append(
                f'<rect x="{cx-7}" y="{cy-4}" width="14" height="9" rx="2" '
                f'fill="{color}" stroke="#fff" stroke-width="0.5"/>'
            )
        if count > 5:
            svgs.append(f'<text x="{x+5*dx}" y="{y+dy//2+4}" font-size="9" fill="#aaa">+{count-5}</text>')
        return "".join(svgs)

    svg = f"""
    <svg viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:300px;display:block;margin:auto;">
      <!-- Background -->
      <rect width="260" height="260" fill="#2a2a28"/>
      <!-- Grass/pavement blocks -->
      <rect x="0" y="0" width="108" height="108" fill="#1e2d1e"/>
      <rect x="152" y="0" width="108" height="108" fill="#1e2d1e"/>
      <rect x="0" y="152" width="108" height="108" fill="#1e2d1e"/>
      <rect x="152" y="152" width="108" height="108" fill="#1e2d1e"/>
      <!-- Roads -->
      <rect x="108" y="0" width="44" height="260" fill="#3a3a38"/>
      <rect x="0" y="108" width="260" height="44" fill="#3a3a38"/>
      <!-- Center box -->
      <rect x="108" y="108" width="44" height="44" fill="#444"/>
      <!-- Lane markings -->
      <line x1="130" y1="0" x2="130" y2="108" stroke="#f0e070" stroke-width="1" stroke-dasharray="8,8"/>
      <line x1="130" y1="152" x2="130" y2="260" stroke="#f0e070" stroke-width="1" stroke-dasharray="8,8"/>
      <line x1="0" y1="130" x2="108" y2="130" stroke="#f0e070" stroke-width="1" stroke-dasharray="8,8"/>
      <line x1="152" y1="130" x2="260" y2="130" stroke="#f0e070" stroke-width="1" stroke-dasharray="8,8"/>
      <!-- Zebra crossings -->
      {''.join(f'<rect x="112" y="{96+i*4}" width="36" height="2" fill="#eee" opacity="0.4"/>' for i in range(4))}
      {''.join(f'<rect x="112" y="{156+i*4}" width="36" height="2" fill="#eee" opacity="0.4"/>' for i in range(4))}
      {''.join(f'<rect x="{96+i*4}" y="112" width="2" height="36" fill="#eee" opacity="0.4"/>' for i in range(4))}
      {''.join(f'<rect x="{156+i*4}" y="112" width="2" height="36" fill="#eee" opacity="0.4"/>' for i in range(4))}
      <!-- Traffic lights — NS -->
      <rect x="155" y="80" width="10" height="26" rx="3" fill="#333"/>
      <circle cx="160" cy="86" r="4" fill="#e74c3c"/>
      <circle cx="160" cy="93" r="4" fill="#f1c40f"/>
      <circle cx="160" cy="100" r="4" fill="#2ecc71"/>
      <circle cx="160" cy="{'86' if ns_color=='#e74c3c' else '93' if ns_color=='#f1c40f' else '100'}" r="4" fill="{ns_color}" filter="url(#glow)"/>
      <rect x="95" y="154" width="10" height="26" rx="3" fill="#333"/>
      <circle cx="100" cy="160" r="4" fill="#e74c3c"/>
      <circle cx="100" cy="167" r="4" fill="#f1c40f"/>
      <circle cx="100" cy="174" r="4" fill="#2ecc71"/>
      <circle cx="100" cy="{'160' if ns_color=='#e74c3c' else '167' if ns_color=='#f1c40f' else '174'}" r="4" fill="{ns_color}" filter="url(#glow)"/>
      <!-- Traffic lights — EW -->
      <rect x="80" y="80" width="10" height="26" rx="3" fill="#333"/>
      <circle cx="85" cy="86" r="4" fill="#e74c3c"/>
      <circle cx="85" cy="93" r="4" fill="#f1c40f"/>
      <circle cx="85" cy="100" r="4" fill="#2ecc71"/>
      <circle cx="85" cy="{'86' if ew_color=='#e74c3c' else '93' if ew_color=='#f1c40f' else '100'}" r="4" fill="{ew_color}" filter="url(#glow)"/>
      <rect x="170" y="154" width="10" height="26" rx="3" fill="#333"/>
      <circle cx="175" cy="160" r="4" fill="#e74c3c"/>
      <circle cx="175" cy="167" r="4" fill="#f1c40f"/>
      <circle cx="175" cy="174" r="4" fill="#2ecc71"/>
      <circle cx="175" cy="{'160' if ew_color=='#e74c3c' else '167' if ew_color=='#f1c40f' else '174'}" r="4" fill="{ew_color}" filter="url(#glow)"/>
      <!-- Cars -->
      {car_row(qN, 121, 90, 0, -16, "#4a9fd4")}
      {car_row(qS, 139, 170, 0,  16, "#4a9fd4")}
      {car_row(qE, 170, 121, 16, 0,  "#e8a135")}
      {car_row(qW,  90, 139, -16, 0, "#e8a135")}
      <!-- Direction labels -->
      <text x="130" y="14" text-anchor="middle" font-size="11" fill="#888" font-family="monospace">N</text>
      <text x="130" y="254" text-anchor="middle" font-size="11" fill="#888" font-family="monospace">S</text>
      <text x="8" y="134" font-size="11" fill="#888" font-family="monospace">W</text>
      <text x="248" y="134" font-size="11" fill="#888" font-family="monospace">E</text>
      <!-- Glow filter -->
      <defs>
        <filter id="glow"><feGaussianBlur stdDeviation="2.5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)

    # Queue bars
    st.subheader("Lane queues")
    bar_fig = go.Figure()
    dirs   = ["N", "S", "E", "W"]
    vals   = [q[d] for d in dirs]
    colors = ["#378ADD" if v <= 5 else "#EF9F27" if v <= 8 else "#D85A30" for v in vals]
    bar_fig.add_trace(go.Bar(
        x=dirs, y=vals, marker_color=colors,
        text=vals, textposition="outside",
    ))
    bar_fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 13], showgrid=False),
        xaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#aaa",
        showlegend=False,
    )
    st.plotly_chart(bar_fig, width="stretch", config={"displayModeBar": False})

# ── Right: Charts ────────────────────────────────────────────
with right:
    hist = st.session_state.history
    eps_list = list(range(1, len(hist["rewards"]) + 1))

    tab1, tab2, tab3 = st.tabs(["📈 Reward", "⏱ Wait time & Throughput", "📉 Loss"])

    with tab1:
        fig = go.Figure()
        if hist["rewards"]:
            window = 20
            avg_r = [np.mean(hist["rewards"][max(0,i-window):i+1]) for i in range(len(hist["rewards"]))]
            fig.add_trace(go.Scatter(
                x=eps_list, y=hist["rewards"],
                name="Episode reward", line=dict(color="#378ADD", width=1), opacity=0.5,
            ))
            fig.add_trace(go.Scatter(
                x=eps_list, y=avg_r,
                name=f"{window}-ep avg", line=dict(color="#1D9E75", width=2.5),
            ))
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Episode", yaxis_title="Reward",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#aaa", legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with tab2:
        fig2 = go.Figure()
        if hist["avg_wait"]:
            fig2.add_trace(go.Scatter(
                x=eps_list, y=hist["avg_wait"],
                name="Avg queue (wait proxy)", line=dict(color="#D85A30", width=2),
            ))
        if hist["throughput"]:
            fig2.add_trace(go.Scatter(
                x=eps_list, y=hist["throughput"],
                name="Throughput (veh/step)", line=dict(color="#1D9E75", width=2),
                yaxis="y2",
            ))
        fig2.update_layout(
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Episode",
            yaxis=dict(title="Avg queue length", color="#D85A30"),
            yaxis2=dict(title="Throughput", overlaying="y", side="right", color="#1D9E75"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#aaa", legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    with tab3:
        fig3 = go.Figure()
        if hist["losses"]:
            fig3.add_trace(go.Scatter(
                x=eps_list, y=hist["losses"],
                name="Huber loss", line=dict(color="#BA7517", width=1.5), opacity=0.8,
            ))
        fig3.update_layout(
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Episode", yaxis_title="Loss",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#aaa",
        )
        st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

    # Q-values heatmap (current state)
    st.subheader("Q-values for current state")
    agent = st.session_state.agent
    state_t = torch.FloatTensor(st.session_state.state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        qvals = agent.q_net(state_t).cpu().numpy()[0].tolist()

    action_names = ["NS 🟢 Green", "NS 🟡 Yellow", "EW 🟢 Green", "EW 🟡 Yellow"]
    best_action  = int(np.argmax(qvals))

    qv_fig = go.Figure(go.Bar(
        x=action_names, y=qvals,
        marker_color=["#1D9E75" if i == best_action else "#378ADD" for i in range(4)],
        text=[f"{v:.3f}" for v in qvals], textposition="outside",
    ))
    qv_fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Q-value",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#aaa", showlegend=False,
    )
    st.plotly_chart(qv_fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"🤖 Agent prefers: **{action_names[best_action]}**  (highest Q-value highlighted in green)")

# ─────────────────────────────────────────────────────────────
#  Training loop — runs `speed` episodes per rerun
# ─────────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.episode < n_episodes:
    agent = st.session_state.agent
    env   = st.session_state.env

    for _ in range(speed):
        if st.session_state.episode >= n_episodes:
            break

        # Run one full episode
        state      = env.reset()
        ep_reward  = 0.0
        ep_wait    = 0.0
        ep_loss    = []
        ep_thru    = 0

        for _ in range(steps_ep_ui):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            loss = agent.learn()
            if loss is not None:
                ep_loss.append(loss)
            ep_reward += reward
            ep_wait   += info["total_queue"] / 4.0
            ep_thru   += info["departed"]
            state      = next_state

        avg_wait = ep_wait / steps_ep_ui
        avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0

        h = st.session_state.history
        h["rewards"].append(ep_reward)
        h["avg_wait"].append(avg_wait)
        h["losses"].append(avg_loss)
        h["throughput"].append(ep_thru / steps_ep_ui)

        st.session_state.episode   += 1
        st.session_state.ep_reward  = ep_reward
        st.session_state.ep_wait    = ep_wait
        st.session_state.ep_thru    = ep_thru
        st.session_state.ep_step    = steps_ep_ui
        st.session_state.state      = state
        st.session_state.trained    = True

    if st.session_state.episode >= n_episodes:
        st.session_state.running = False
        st.success(f"✅ Training complete — {n_episodes} episodes finished!")
    else:
        time.sleep(0.02)
        st.rerun()

elif st.session_state.episode >= n_episodes and st.session_state.trained:
    st.success(f"✅ Training complete — {n_episodes} episodes finished!")

    col1, col2 = st.columns(2)
    with col1:
        h = st.session_state.history
        last20 = h["rewards"][-20:]
        st.metric("Best episode reward",  f"{max(h['rewards']):.0f}")
    with col2:
        st.metric("Final 20-ep avg reward", f"{np.mean(last20):.1f}")
