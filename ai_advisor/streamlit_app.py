"""
AI Financial Advisor — Conversational Streamlit App
Run:  streamlit run streamlit_app.py
"""

from dotenv import load_dotenv
load_dotenv()

import base64
import time
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from ai_advisor.run_advisor import run_initial_pipeline, run_followup_reoptimize, AdvisorResult

# ── Page Config ─────────────────────────────────────────────────────────────

try:
    from PIL import Image as _PIL_Image
    _fav = Path(__file__).parent / "favicon.ico"
    _page_icon = _PIL_Image.open(_fav) if _fav.exists() else "◆"
except Exception:
    _page_icon = "◆"

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Theme-aware variables ── */
    :root {
        --accent-gold: #c9a24e;
        --accent-gold-dim: rgba(201,162,78,0.12);
        --text-muted: #6b7280;
        --border-color: rgba(128, 128, 128, 0.18);
        --card-bg: rgba(128, 128, 128, 0.05);
    }
    @media (prefers-color-scheme: dark) {
        :root { --text-secondary: #94a3b8; }
    }
    @media (prefers-color-scheme: light) {
        :root { --text-secondary: #475569; }
    }

    .stApp { font-family: 'DM Sans', sans-serif; }
    #MainMenu, header, footer, .stDeployButton { display: none !important; }
    .block-container { padding-top: 1rem !important; }
    /* Prevent animated scroll so rerun scroll-jumps are instant, not a sweep */
    section[data-testid="stMain"] { scroll-behavior: auto !important; }
    /* During options questions the chat-input bottom bar is hidden but kept in
       the DOM so its height never changes — this prevents the layout-shift that
       causes Streamlit to auto-scroll when the bar reappears for text questions. */
    body:has(#options-q-active) [data-testid="stBottomBlockContainer"] {
        visibility: hidden !important;
    }

    /* Constrain chat/survey/processing to readable width;
       the sentinel #chat-mode div is injected for all non-welcome phases. */
    body:has(#chat-mode) .block-container {
        max-width: 760px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    /* Constrain the sticky chat input bar (Streamlit 1.40+ uses stBottomBlockContainer) */
    body:has(#chat-mode) [data-testid="stBottomBlockContainer"] {
        max-width: 760px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    /* Fallback for other Streamlit versions */
    body:has(#chat-mode) [data-testid="stBottom"] {
        max-width: 760px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Landing page content: cap width on very wide monitors */
    .landing-content { max-width: 1200px; margin: 0 auto; }

    /* User avatar */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #22c55e !important;
        color: #fff !important;
    }

    /* Brand bar */
    .brand-bar {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; font-weight: 500; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--text-muted);
        padding: 1rem 0 0.5rem 0; border-bottom: 1px solid var(--border-color);
        margin-bottom: 1rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .brand-name { color: var(--accent-gold); }
    .brand-logo { height: 28px; width: auto; display: block; margin-bottom: 3px; }
    .brand-logo-dark { display: none; }
    @media (prefers-color-scheme: dark) {
        .brand-logo-light { display: none; }
        .brand-logo-dark { display: inline-block; }
    }

    /* ── User messages: avatar on right ── */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        text-align: right !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        display: inline-flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        max-width: 80% !important;
        width: auto !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] > div {
        background: rgba(128,128,128,0.08) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.85rem !important;
        width: fit-content !important;
        max-width: 100% !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: transparent !important;
    }

    /* Survey option buttons */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.4rem 1rem !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
        white-space: normal !important;
        line-height: 1.3 !important;
        min-height: 0 !important;
    }
    .stButton > button:hover {
        border-color: var(--accent-gold) !important;
        background: var(--accent-gold-dim) !important;
    }
    .stButton > button[kind="primary"] {
        text-align: center !important;
        padding: 0.6rem 2rem !important;
        font-size: 0.9rem !important;
    }

    /* Gold chat input */
    [data-testid="stChatInput"] > div {
        border-color: var(--accent-gold) !important;
        box-shadow: 0 0 0 1px var(--accent-gold) !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent-gold) !important;
        box-shadow: 0 0 0 2px rgba(201,162,78,0.45) !important;
    }
    [data-testid="stChatInputTextArea"] { caret-color: var(--accent-gold) !important; }
    [data-testid="stChatInputSubmitButton"] button { color: var(--accent-gold) !important; }

    /* ── Portfolio card ── */
    .portfolio-card {
        border: 1px solid var(--border-color);
        border-radius: 10px; padding: 1.5rem; margin: 0.75rem 0;
        background: var(--card-bg);
    }
    .portfolio-card h3 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--accent-gold); margin: 0 0 1rem 0;
    }
    .portfolio-metrics { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .pm-item {
        flex: 1; min-width: 100px; padding: 0.6rem 0.8rem;
        border: 1px solid var(--border-color); border-radius: 6px; background: var(--card-bg);
    }
    .pm-label {
        font-size: 0.65rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.15rem;
    }
    .pm-value {
        font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600;
        /* No explicit color — inherits from Streamlit theme (works in light + dark) */
    }
    .pm-value.gold { color: var(--accent-gold); }

    .alloc-table { width: 100%; margin-top: 0.5rem; }
    .alloc-table-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);
    }
    .alloc-table-row:last-child { border-bottom: none; }
    .at-ticker {
        font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.85rem;
        /* Inherits theme color */
    }
    .at-bar-container {
        flex: 1; margin: 0 1rem; height: 4px;
        background: var(--border-color); border-radius: 2px; overflow: hidden;
    }
    .at-bar { height: 100%; background: var(--accent-gold); border-radius: 2px; }
    .at-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem; color: var(--accent-gold); min-width: 45px; text-align: right;
    }

    /* Typing indicator */
    .typing-dot {
        display: inline-block; width: 6px; height: 6px;
        border-radius: 50%; background: var(--text-muted);
        margin-right: 4px; animation: blink 1.4s infinite both;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
        0%, 80%, 100% { opacity: 0.3; }
        40% { opacity: 1; }
    }

    .input-hint {
        font-size: 0.78rem; color: var(--text-muted);
        margin-top: -0.5rem; margin-bottom: 0.5rem;
    }

    /* ── Pipeline status ── */
    .pipeline-stage {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        color: var(--text-muted); padding: 0.15rem 0;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .pipeline-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
    .pipeline-dot.done { background: var(--accent-gold); }
    .pipeline-dot.active {
        background: var(--text-muted);
        animation: pulse-stage 1s ease-in-out infinite;
    }
    @keyframes pulse-stage {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.3); }
    }
    .pipeline-block { padding: 0.4rem 0; }

    /* ── Landing page ── */
    .landing-hero { padding: 0; }
    .landing-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--accent-gold); border: 1px solid rgba(201,162,78,0.45);
        border-radius: 20px; padding: 0.3rem 1rem; margin-bottom: 0.7rem;
    }
    .landing-title {
        font-size: 2.75rem; font-weight: 700; line-height: 1.1; margin: 0 0 0.6rem 0;
    }
    .landing-sub {
        font-size: 1rem; line-height: 1.6; color: var(--text-muted); display: block;
    }
    /* Stats bar — shown below hero */
    .stats-bar {
        display: flex; align-items: center; justify-content: center;
        gap: 0; margin: 2rem 0 0 0;
    }
    .stat-item { text-align: center; padding: 0 2rem; }
    .stat-item + .stat-item { border-left: 1px solid var(--border-color); }
    .stat-value {
        font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600;
        color: var(--accent-gold);
    }
    .stat-label { font-size: 0.82rem; color: var(--text-muted); margin-top: 0.2rem; }
    /* Divider */
    .landing-divider { height: 1px; background: var(--border-color); margin: 2.5rem 0; }
    .landing-section-label {
        font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
        font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--accent-gold); margin-bottom: 1.5rem;
    }
    /* How it works — 3 horizontal cards */
    .steps-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.25rem; }
    .step-card {
        padding: 1.6rem 1.5rem; border: 1px solid var(--border-color);
        border-radius: 8px; background: var(--card-bg);
    }
    .step-num {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600;
        color: var(--accent-gold); opacity: 0.8; margin-bottom: 0.75rem;
    }
    .step-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.45rem; }
    .step-desc { font-size: 0.88rem; color: var(--text-muted); line-height: 1.55; }
    /* Strategies — 4-column grid */
    .strategies-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; }
    .strategy-card {
        padding: 1.1rem 1.2rem; border: 1px solid var(--border-color);
        border-radius: 8px; background: var(--card-bg);
    }
    .strategy-name { font-size: 0.92rem; font-weight: 600; margin-bottom: 0.35rem; }
    .strategy-desc { font-size: 0.8rem; color: var(--text-muted); line-height: 1.45; margin-bottom: 0.65rem; }
    .strategy-risk {
        display: inline-block; font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.18rem 0.55rem; border-radius: 3px;
    }
    .risk-conservative { background: rgba(16,185,129,0.12); color: #10b981; }
    .risk-moderate     { background: rgba(59,130,246,0.12); color: #3b82f6; }
    .risk-balanced     { background: rgba(139,92,246,0.12); color: #8b5cf6; }
    .risk-growth       { background: rgba(245,158,11,0.12); color: #f59e0b; }
    .risk-aggressive   { background: rgba(239,68,68,0.12);  color: #ef4444; }
    /* Disclaimer */
    .landing-disclaimer {
        font-size: 0.78rem; color: var(--text-muted); text-align: center;
        padding-top: 1.5rem; border-top: 1px solid var(--border-color);
        line-height: 1.6; margin-bottom: 2.5rem;
    }

    /* ── Hero section: self-contained split background ── */
    .hero-section {
        position: relative;
        padding: 0;
        min-height: 400px;
        display: flex;
        align-items: flex-start;
    }
    .hero-section::before {
        content: '';
        position: absolute; top: 0; bottom: 0;
        left: -200vw; right: 40%;
        background: #0d1b2e;
        z-index: 0;
    }
    .hero-section::after {
        content: '';
        position: absolute; top: 0; bottom: 0;
        left: 60%; right: -200vw;
        background: #091320;
        z-index: 0;
    }
    .hero-inner {
        position: relative; z-index: 1;
        display: flex; gap: 0; align-items: center;
        width: 100%;
    }
    .hero-left { flex: 0 0 60%; padding-right: 5rem; padding-top: 9vh; }
    .hero-right {
        flex: 0 0 40%; position: relative;
        overflow: hidden; min-height: 280px;
        display: flex; align-items: flex-start; justify-content: center;
        padding-top: 9vh;
    }
    .hero-section .landing-title { color: #ffffff; }
    .hero-section .landing-sub {
        color: rgba(255,255,255,0.7);
    }
    .hero-cta-btn, .hero-cta-btn:link, .hero-cta-btn:visited {
        display: inline-block;
        background: var(--accent-gold); color: #ffffff !important;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700; font-size: 0.95rem;
        padding: 0.75rem 2.2rem; border-radius: 6px;
        text-decoration: none !important; margin-top: 1rem;
        cursor: pointer; transition: background 0.15s ease;
        letter-spacing: 0.01em;
    }
    .hero-cta-btn:hover { background: #d4b06a; color: #ffffff !important; text-decoration: none !important; }

    /* ── Hero right: chart decoration ── */
    .hero-svg {
        width: min(420px, 92%);
        height: auto;
        flex-shrink: 0;
    }

    /* ── Stats bar: full-width black section ── */
    .landing-stats-dark {
        position: relative;
        padding: 2.75rem 0;
    }
    .landing-stats-dark::before {
        content: '';
        position: absolute; top: 0; bottom: 0;
        left: -200vw; right: -200vw;
        background: #080808;
        z-index: 0;
    }
    .landing-stats-dark .stats-bar { position: relative; z-index: 1; margin: 0; }
    .landing-stats-dark .stat-label { color: #9ca3af !important; }
    .landing-stats-dark .stat-value { font-size: 1.8rem; }

</style>
""", unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = -1          # -1 = welcome, 0..N-1 = survey, N = done
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "phase" not in st.session_state:
    st.session_state.phase = "welcome"  # welcome → survey → processing → chat
if "advisor_result" not in st.session_state:
    st.session_state.advisor_result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


# ── Questions ───────────────────────────────────────────────────────────────

QUESTIONS = [
    {
        "key": "age",
        "message": "Let's start with the basics. How old are you?",
        "type": "freetext",
        "hint": "Enter your age",
    },
    {
        "key": "employment_status",
        "message": "What's your current employment situation?",
        "type": "options",
        "options": [
            "Full-time employed",
            "Part-time employed",
            "Self-employed / Freelance",
            "Retired",
            "Student",
            "Unemployed",
        ],
    },
    {
        "key": "dependents",
        "message": "How many people depend on you financially? (Children, elderly parents, etc.)",
        "type": "freetext",
        "hint": "Enter a number",
    },
    {
        "key": "annual_income",
        "message": "What's your approximate annual income before tax?",
        "type": "freetext",
        "hint": "e.g. $80,000",
    },
    {
        "key": "monthly_expenses",
        "message": "And roughly how much do you spend each month in total?",
        "type": "freetext",
        "hint": "e.g. $3,500",
    },
    {
        "key": "monthly_savings",
        "message": "How much are you able to save or invest each month?",
        "type": "freetext",
        "hint": "e.g. $1,000",
    },
    {
        "key": "total_savings",
        "message": "What's the total in your savings and checking accounts right now?",
        "type": "freetext",
        "hint": "e.g. $25,000",
    },
    {
        "key": "existing_investments",
        "message": "Do you have any existing investments? If so, what's their approximate total value? (401k, IRA, brokerage, etc.)",
        "type": "freetext",
        "hint": "e.g. $10,000 or none",
    },
    {
        "key": "total_debt",
        "message": "What about debt — what's your total outstanding balance across all debts?",
        "type": "freetext",
        "hint": "e.g. $15,000 or none",
    },
    {
        "key": "debt_details",
        "message": "Could you briefly describe what those debts are?",
        "type": "freetext",
        "hint": "e.g. $20k student loan at 5%, $3k credit card — or type 'none'",
    },
    {
        "key": "experience_level",
        "message": "How would you describe your investing experience?",
        "type": "options",
        "options": [
            "None — I've never invested",
            "Beginner — I've done a little investing",
            "Intermediate — I invest regularly",
            "Advanced — I actively manage a diversified portfolio",
        ],
    },
    {
        "key": "current_holdings",
        "message": "What types of investments do you currently hold, if any?",
        "type": "freetext",
        "hint": "e.g. index funds, stocks, crypto, real estate — or type 'none'",
    },
    {
        "key": "investment_horizon",
        "message": "How long do you plan to keep this money invested?",
        "type": "options",
        "options": [
            "Less than 1 year",
            "1–3 years",
            "3–5 years",
            "5–10 years",
            "10–20 years",
            "20+ years",
        ],
    },
    {
        "key": "investment_amount",
        "message": "How much are you looking to invest initially? We will construct your portfolio based on this",
        "type": "freetext",
        "hint": "e.g. $10,000",
    },
    {
        "key": "short_term_goals",
        "message": "What are your financial goals for the next 1–3 years?",
        "type": "freetext",
        "hint": "e.g. build emergency fund, save for a vacation, down payment",
    },
    {
        "key": "long_term_goals",
        "message": "And what are your longer-term financial goals?",
        "type": "freetext",
        "hint": "e.g. retire by 55, children's education, financial independence",
    },
    {
        "key": "risk_comfort",
        "message": "Now let's talk about risk. Imagine your portfolio dropped 20% in a single month. What would you do?",
        "type": "options",
        "options": [
            "Sell everything immediately",
            "Very uncomfortable — would consider selling some",
            "Concerned but would hold and wait for recovery",
            "Stay the course — market dips are normal",
            "See it as a buying opportunity and invest more",
        ],
    },
    {
        "key": "return_vs_safety",
        "message": "Which of these best describes your investment priority?",
        "type": "options",
        "options": [
            "Protect my capital — I'd rather earn less than risk losing",
            "Mostly safe with a little growth potential",
            "A balance between growth and safety",
            "Mostly growth — I can handle short-term losses",
            "Maximum growth — comfortable with high volatility",
        ],
    },
    {
        "key": "loss_tolerance",
        "message": "What's the maximum loss you could stomach in a single year before you'd want to change strategy?",
        "type": "options",
        "options": [
            "5% or less",
            "10%",
            "20%",
            "30%",
            "40% or more",
        ],
    },
    {
        "key": "investment_style",
        "message": "Do you prefer passive investing (tracking the market) or active investing (trying to beat it)?",
        "type": "options",
        "options": [
            "Passive — I just want to match the market (e.g. index funds)",
            "Slightly passive — prefer index funds but open to some active",
            "No preference — either is fine",
            "Slightly active — prefer trying to beat the market",
            "Active — I want to maximize returns and beat the market",
        ],
    },
    {
        "key": "leverage_comfort",
        "message": "Are you comfortable with leverage — using borrowed money to amplify returns (which also amplifies losses)?",
        "type": "options",
        "options": [
            "No — I only want to invest what I have",
            "Maybe — open to a small amount if strongly recommended",
            "Yes — I'm comfortable with leverage",
        ],
    },
    {
        "key": "num_stocks",
        "message": "How many holdings would you like in your portfolio?",
        "type": "options",
        "options": [
            "5 or fewer — concentrated, high-conviction",
            "6–10 — moderately diversified",
            "11–15 — well diversified",
            "16–20 — broadly diversified",
            "20+ — maximum diversification",
        ],
    },
    {
        "key": "sector_diversification",
        "message": "Would you like your portfolio spread across different industries and sectors?",
        "type": "options",
        "options": [
            "Yes — spread across as many sectors as possible",
            "Somewhat — some sector diversity, but don't force it",
            "No preference — let the optimizer decide",
            "No — concentrate in a few sectors I believe in",
        ],
    },
    {
        "key": "special_considerations",
        "message": "Last question — anything else I should know? Any stock that you like or dislike? ESG preferences, upcoming big expenses, tax situations, constraints?",
        "type": "freetext",
        "hint": "Type anything relevant, or 'none'",
    },
]

TOTAL = len(QUESTIONS)

SURVEY_LABELS = {
    "age": "Age", "employment_status": "Employment Status",
    "dependents": "Number of Dependents",
    "annual_income": "Annual Income (pre-tax)",
    "monthly_expenses": "Monthly Expenses",
    "monthly_savings": "Monthly Savings/Investment",
    "total_savings": "Total Liquid Savings",
    "existing_investments": "Existing Investments Value",
    "total_debt": "Total Outstanding Debt",
    "debt_details": "Debt Details",
    "experience_level": "Investment Experience",
    "current_holdings": "Current Holdings",
    "investment_horizon": "Investment Horizon",
    "investment_amount": "Initial Investment Amount",
    "short_term_goals": "Short-term Goals (1-3 years)",
    "long_term_goals": "Long-term Goals",
    "risk_comfort": "Reaction to 20% Portfolio Drop",
    "return_vs_safety": "Growth vs Safety Priority",
    "loss_tolerance": "Maximum Tolerable Annual Loss",
    "investment_style": "Investment Style",
    "leverage_comfort": "Leverage Comfort",
    "num_stocks": "Preferred Number of Holdings",
    "sector_diversification": "Sector Diversification Preference",
    "special_considerations": "Special Considerations",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def brand():
    _here = Path(__file__).parent
    _logo_light = _here / "logo_light.png"
    _logo_dark = _here / "logo_dark.png"
    if _logo_light.exists() and _logo_dark.exists():
        def _b64(p):
            return base64.b64encode(p.read_bytes()).decode()
        light_src = f"data:image/png;base64,{_b64(_logo_light)}"
        dark_src = f"data:image/png;base64,{_b64(_logo_dark)}"
        brand_html = (
            f'<img class="brand-logo brand-logo-light" src="{light_src}" alt="Your Robo-Advisor">'
            f'<img class="brand-logo brand-logo-dark" src="{dark_src}" alt="Your Robo-Advisor">'
        )
    else:
        brand_html = '<span class="brand-name">Your Robo-Advisor</span>'
    st.markdown(
        '<div class="brand-bar">'
        f'<span style="display:inline-flex;align-items:flex-end;gap:0.5em;">'
        f'{brand_html}'
        f'<span> | Intelligent Portfolio Management</span>'
        f'</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    components.html("""<script>
(function() {
    var doc = window.parent.document;
    function fitHero() {
        var bar = doc.querySelector('.brand-bar');
        var hero = doc.querySelector('.hero-section');
        if (bar && hero) {
            hero.style.minHeight = (window.parent.innerHeight - bar.offsetHeight - 38) + 'px';
        }
    }
    fitHero();
    window.parent.addEventListener('resize', fitHero);
})();
</script>""", height=0)


def autofocus_input():
    """Inject JS to focus the chat input textarea after it renders."""
    components.html("""
        <script>
            (function() {
                var doc = window.parent.document;
                // Scroll to bottom immediately — mirrors button-question behaviour,
                // which pre-empts Streamlit's post-render animated sweep.
                var mainEl = doc.querySelector('section[data-testid="stMain"]')
                           || doc.querySelector('section.main')
                           || doc.querySelector('.main');
                if (mainEl) mainEl.scrollTop = mainEl.scrollHeight;
                function focus() {
                    var el = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                    if (el) { el.focus({ preventScroll: true }); return true; }
                    return false;
                }
                if (!focus()) {
                    var t = setInterval(function() { if (focus()) clearInterval(t); }, 50);
                    setTimeout(function() { clearInterval(t); }, 1500);
                }
            })();
        </script>
    """, height=0)


def typing_generator(text: str):
    """
    Yield text in small chunks to simulate a typing effect.
    Speed is adaptive: total animation time is capped at ~1.5 s so short
    questions feel snappy and long responses don't drag on forever.
    """
    chunk_size = 3
    chunks = max(len(text) // chunk_size, 1)
    delay = min(0.025, max(0.008, 1.5 / chunks))

    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        time.sleep(delay)


def add_assistant(text: str, animated: bool = True):
    """Append an assistant message. `animated=True` means it will be
    streamed character-by-character the first time it is rendered."""
    st.session_state.messages.append({
        "role": "assistant",
        "content": text,
        "animated": animated,
    })


def add_user(text):
    st.session_state.messages.append({"role": "user", "content": text})


def format_survey_for_crew():
    lines = []
    for k, v in st.session_state.answers.items():
        lines.append(f"{SURVEY_LABELS.get(k, k)}: {v}")
    return "\n".join(lines)


_HORIZON_YEARS = {
    "<1yr": 1, "1-3yr": 2, "3-5yr": 4,
    "5-10yr": 7, "10-20yr": 15, "20+yr": 25,
}


def _make_pie_chart(allocations: list[dict]) -> go.Figure:
    labels = [a["ticker"] for a in allocations]
    values = [a["weight"] for a in allocations]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4,
        textinfo="label+percent",
        textfont_size=13,
        marker=dict(colors=[
            "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981",
            "#3b82f6", "#ef4444", "#14b8a6", "#f97316", "#a855f7",
            "#06b6d4", "#84cc16", "#e11d48", "#0ea5e9", "#d97706",
        ]),
    ))
    fig.update_layout(
        title=dict(text="Portfolio Allocation", font=dict(size=16)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _make_monte_carlo(
    expected_return: float,
    expected_volatility: float,
    investment_amount: float,
    investment_horizon: str,
    n_sims: int = 500,
) -> go.Figure:
    years = _HORIZON_YEARS.get(investment_horizon, 5)
    n_steps = years * 12
    dt = 1 / 12
    mu, sigma = expected_return, expected_volatility

    rng = np.random.default_rng(42)
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = investment_amount
    for t in range(1, n_steps + 1):
        Z = rng.standard_normal(n_sims)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )

    times = np.linspace(0, years, n_steps + 1)
    pct = {p: np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)}

    accent = "99, 102, 241"
    fig = go.Figure()
    # Outer band (5–95th)
    fig.add_trace(go.Scatter(
        x=np.concatenate([times, times[::-1]]),
        y=np.concatenate([pct[95], pct[5][::-1]]),
        fill="toself", fillcolor=f"rgba({accent},0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95th %ile", hoverinfo="skip",
    ))
    # Inner band (25–75th)
    fig.add_trace(go.Scatter(
        x=np.concatenate([times, times[::-1]]),
        y=np.concatenate([pct[75], pct[25][::-1]]),
        fill="toself", fillcolor=f"rgba({accent},0.30)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th %ile", hoverinfo="skip",
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=times, y=pct[50],
        line=dict(color=f"rgb({accent})", width=2.5), name="Median",
    ))
    # Starting value dotted line
    fig.add_hline(
        y=investment_amount, line_dash="dot",
        line_color="gray", opacity=0.5,
        annotation_text=f"  Initial ${investment_amount:,.0f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(
            text=f"Monte Carlo Simulation — {n_sims} paths over {years} yr",
            font=dict(size=16),
        ),
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(200,200,200,0.2)"),
        yaxis=dict(gridcolor="rgba(200,200,200,0.2)"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=60),
    )
    return fig


def _render_charts(params: dict):
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_make_pie_chart(params["allocations"]), width='stretch')
    with col2:
        st.plotly_chart(
            _make_monte_carlo(
                params["expected_return"],
                params["expected_volatility"],
                params["investment_amount"],
                params["investment_horizon"],
            ),
            width='stretch',
        )


def build_portfolio_card(result: AdvisorResult) -> str:
    """Build HTML card for portfolio results displayed in chat."""
    opt = result.optimization_result

    metrics = f"""
    <div class="portfolio-card">
        <h3>Portfolio Overview</h3>
        <div class="portfolio-metrics">
            <div class="pm-item">
                <div class="pm-label">Risk Category</div>
                <div class="pm-value gold">{result.risk_category}</div>
            </div>
            <div class="pm-item">
                <div class="pm-label">Strategy</div>
                <div class="pm-value" style="font-size:0.85rem;">{opt.strategy_display_name}</div>
            </div>
            <div class="pm-item">
                <div class="pm-label">Exp. Return</div>
                <div class="pm-value">{opt.expected_return:.1%}</div>
            </div>
            <div class="pm-item">
                <div class="pm-label">Volatility</div>
                <div class="pm-value">{opt.expected_volatility:.1%}</div>
            </div>
            <div class="pm-item">
                <div class="pm-label">Sharpe</div>
                <div class="pm-value">{opt.sharpe_ratio:.2f}</div>
            </div>
        </div>
    """

    max_w = max((a["weight"] for a in result.allocations), default=1)
    rows = ""
    for a in result.allocations:
        pct = a["weight"] * 100
        bar_w = (a["weight"] / max_w) * 100
        rows += f"""
        <div class="alloc-table-row">
            <span class="at-ticker">{a["ticker"]}</span>
            <div class="at-bar-container">
                <div class="at-bar" style="width:{bar_w}%"></div>
            </div>
            <span class="at-pct">{pct:.1f}%</span>
        </div>"""

    metrics += f'<div class="alloc-table">{rows}</div></div>'
    return metrics


def build_stages_html(completed: list[str], active: str = None) -> str:
    """Build the HTML for pipeline status stages."""
    html = '<div class="pipeline-block">'
    for stage in completed:
        html += (
            f'<div class="pipeline-stage">'
            f'<span class="pipeline-dot done"></span>'
            f'{stage}'
            f'</div>'
        )
    if active:
        html += (
            f'<div class="pipeline-stage">'
            f'<span class="pipeline-dot active"></span>'
            f'{active}'
            f'</div>'
        )
    html += '</div>'
    return html


def advance_to_next_question():
    """Add the next survey question as an assistant message."""
    step = st.session_state.step
    if step < TOTAL:
        q = QUESTIONS[step]
        add_assistant(q["message"])


def _no_debt(value: str) -> bool:
    """Return True if the answer indicates zero debt."""
    s = value.strip().lower().replace("$", "").replace(",", "").replace(" ", "")
    return s in {"none", "no", "n/a", "na", "0", "0.0", "nil", "nothing", "zero", "nope", "not any", "0debt"}


def record_answer(value: str):
    """Record answer, advance step, add next question or trigger processing."""
    step = st.session_state.step
    q = QUESTIONS[step]

    add_user(value)
    st.session_state.answers[q["key"]] = value
    st.session_state.step += 1

    # Skip debt_details when user has no debt
    if q["key"] == "total_debt" and _no_debt(value):
        st.session_state.answers["debt_details"] = "none"
        st.session_state.step += 1

    if st.session_state.step >= TOTAL:
        st.session_state.phase = "processing"
    else:
        advance_to_next_question()


def reset():
    for k in ["messages", "step", "answers", "phase", "advisor_result", "error", "pending_input"]:
        if k in st.session_state:
            del st.session_state[k]


def build_followup_context(current: str) -> str:
    """Pack prior follow-up conversation for agent memory."""
    msgs = st.session_state.messages
    start = 0
    for i, m in enumerate(msgs):
        if m.get("is_result"):
            start = i + 1
            break

    prior = msgs[start:]
    if not prior:
        return current

    lines = ["PRIOR CONVERSATION:"]
    for m in prior:
        speaker = "User" if m["role"] == "user" else "Advisor"
        lines.append(f"{speaker}: {m.get('content', '')[:800]}")
    lines.append(f"\nCURRENT QUESTION: {current}")
    return "\n".join(lines)


# ── Render ──────────────────────────────────────────────────────────────────

def main():
    # Inject sentinel for non-welcome phases — triggers CSS to constrain block-container width
    if st.session_state.phase != "welcome":
        st.markdown('<div id="chat-mode"></div>', unsafe_allow_html=True)
        # Scroll to bottom before messages render so the typing animation is
        # visible from the start (rerun resets scroll to top by default).
        # Multiple retries handle the async gap between Python execution and
        # the browser actually having painted the content.
        components.html("""
            <script>
            (function() {
                var doc = window.parent.document;
                function scrollBottom() {
                    var el = doc.querySelector('section[data-testid="stMain"]')
                           || doc.querySelector('.main');
                    if (el) el.scrollTop = el.scrollHeight;
                }
                scrollBottom();
                setTimeout(scrollBottom, 100);
                setTimeout(scrollBottom, 300);
                setTimeout(scrollBottom, 600);
            })();
            </script>
        """, height=0)

    brand()

    # ── Welcome state ──
    if st.session_state.phase == "welcome":
        # Detect "Get started" click — set by the hero HTML button via query param
        if st.query_params.get("started"):
            st.query_params.clear()
            st.session_state.phase = "survey"
            st.session_state.step = 0
            add_assistant("Great, let's build your portfolio. I'll walk you through a few questions — it should take about 3–5 minutes.")
            advance_to_next_question()
            st.rerun()
            return

        welcome = st.empty()
        with welcome.container():
            # ── Hero: self-contained 65/35 split, backgrounds via CSS pseudo-elements ──
            st.markdown("""
<div class="hero-section">
    <div class="hero-inner">
        <div class="hero-left">
            <div class="landing-badge">Intelligent &middot; Quantitative &middot; Dynamic</div>
            <h1 class="landing-title">Your Portfolio,<br>Optimized.</h1>
            <p class="landing-sub">
                Our systematic assessment aligns your goals with a bespoke,<br>
                quantitatively optimized portfolio driven by rigorous real-time market data.
            </p>
            <a href="?started=1" target="_self" onclick="window.location.href='?started=1'; return false;" class="hero-cta-btn">Get started →</a>
        </div>
        <div class="hero-right">
            <svg class="hero-svg" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="af" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="rgba(201,162,78,0.20)"/>
                        <stop offset="100%" stop-color="rgba(201,162,78,0)"/>
                    </linearGradient>
                </defs>
                <line x1="35" y1="82" x2="35" y2="332" stroke="rgba(201,162,78,0.22)" stroke-width="1"/>
                <line x1="35" y1="332" x2="378" y2="332" stroke="rgba(201,162,78,0.22)" stroke-width="1"/>
                <line x1="29" y1="122" x2="35" y2="122" stroke="rgba(201,162,78,0.38)" stroke-width="1"/>
                <line x1="29" y1="172" x2="35" y2="172" stroke="rgba(201,162,78,0.28)" stroke-width="1"/>
                <line x1="29" y1="222" x2="35" y2="222" stroke="rgba(201,162,78,0.28)" stroke-width="1"/>
                <line x1="29" y1="272" x2="35" y2="272" stroke="rgba(201,162,78,0.28)" stroke-width="1"/>
                <line x1="115" y1="332" x2="115" y2="338" stroke="rgba(201,162,78,0.32)" stroke-width="1"/>
                <line x1="200" y1="332" x2="200" y2="338" stroke="rgba(201,162,78,0.32)" stroke-width="1"/>
                <line x1="285" y1="332" x2="285" y2="338" stroke="rgba(201,162,78,0.32)" stroke-width="1"/>
                <line x1="368" y1="332" x2="368" y2="338" stroke="rgba(201,162,78,0.32)" stroke-width="1"/>
                <line x1="35" y1="344" x2="378" y2="344" stroke="rgba(201,162,78,0.10)" stroke-width="1"/>
                <rect x="44"  y="367" width="7" height="13" fill="rgba(201,162,78,0.18)"/>
                <rect x="82"  y="361" width="7" height="19" fill="rgba(201,162,78,0.18)"/>
                <rect x="119" y="356" width="7" height="24" fill="rgba(201,162,78,0.22)"/>
                <rect x="156" y="363" width="7" height="17" fill="rgba(201,162,78,0.18)"/>
                <rect x="193" y="351" width="7" height="29" fill="rgba(201,162,78,0.26)"/>
                <rect x="231" y="358" width="7" height="22" fill="rgba(201,162,78,0.20)"/>
                <rect x="268" y="354" width="7" height="26" fill="rgba(201,162,78,0.22)"/>
                <rect x="306" y="346" width="7" height="34" fill="rgba(201,162,78,0.30)"/>
                <rect x="348" y="348" width="7" height="32" fill="rgba(201,162,78,0.28)"/>
                <path d="M 35,308 C 75,290 95,262 115,270 S 150,285 165,278 S 205,230 225,222 S 268,196 285,188 S 318,148 338,140 S 358,102 368,90 L 368,332 L 35,332 Z" fill="url(#af)"/>
                <path d="M 35,318 C 85,306 130,285 165,278 C 200,270 220,248 248,234 C 275,220 305,198 335,174 S 362,150 368,134" stroke="rgba(201,162,78,0.30)" stroke-width="1.5" fill="none" stroke-linecap="round"/>
                <path d="M 35,308 C 75,290 95,262 115,270 S 150,285 165,278 S 205,230 225,222 S 268,196 285,188 S 318,148 338,140 S 358,102 368,90" stroke="rgba(201,162,78,0.90)" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="115" cy="270" r="2.5" fill="rgba(201,162,78,0.55)"/>
                <circle cx="225" cy="222" r="2.5" fill="rgba(201,162,78,0.68)"/>
                <circle cx="338" cy="140" r="2.5" fill="rgba(201,162,78,0.78)"/>
                <line x1="35" y1="90" x2="356" y2="90" stroke="rgba(201,162,78,0.12)" stroke-width="1" stroke-dasharray="4,4"/>
                <circle cx="368" cy="90" r="5" fill="rgba(201,162,78,0.95)"/>
                <circle cx="368" cy="90" r="10" stroke="rgba(201,162,78,0.28)" stroke-width="1" fill="none"/>
            </svg>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            # ── Stats bar: full-width black background ──
            st.markdown("""
<div class="landing-stats-dark">
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">22</div>
        <div class="stat-label">questions</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">8</div>
        <div class="stat-label">strategies</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">50+</div>
        <div class="stat-label">securities</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">~5</div>
        <div class="stat-label">minutes</div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

            # ── How it works + Strategies + Disclaimer ──
            st.markdown("""
<div class="landing-content">
<div class="landing-divider"></div>

<div class="landing-section-label">How it works</div>
<div class="steps-grid">
    <div class="step-card">
        <div class="step-num">01</div>
        <div class="step-title">Complete your profile</div>
        <div class="step-desc">22 questions covering income, goals, risk comfort, and investment preferences.</div>
    </div>
    <div class="step-card">
        <div class="step-num">02</div>
        <div class="step-title">AI analysis &amp; optimization</div>
        <div class="step-desc">We assess your risk tier, fetch live market data, select securities, and run portfolio optimization.</div>
    </div>
    <div class="step-card">
        <div class="step-num">03</div>
        <div class="step-title">Receive your portfolio</div>
        <div class="step-desc">Optimized allocations, growth projections, a risk profile summary, and an AI advisor for follow-up questions.</div>
    </div>
</div>

<div class="landing-divider"></div>

<div class="landing-section-label">Optimization strategies</div>
<div class="strategies-grid">
    <div class="strategy-card">
        <div class="strategy-name">Minimum Variance</div>
        <div class="strategy-desc">Minimize portfolio volatility. Prioritizes capital preservation.</div>
        <span class="strategy-risk risk-conservative">Conservative</span>
    </div>
    <div class="strategy-card">
        <div class="strategy-name">Equal Risk Contribution</div>
        <div class="strategy-desc">Each asset contributes equally to total portfolio risk.</div>
        <span class="strategy-risk risk-moderate">Moderate</span>
    </div>
    <div class="strategy-card">
        <div class="strategy-name">Robust Mean-Variance</div>
        <div class="strategy-desc">Minimize variance while maintaining a minimum return floor.</div>
        <span class="strategy-risk risk-balanced">Balanced</span>
    </div>
    <div class="strategy-card">
        <div class="strategy-name">Market Tracking</div>
        <div class="strategy-desc">Tracks VOO. Passive broad-market exposure with growth potential.</div>
        <span class="strategy-risk risk-growth">Growth</span>
    </div>
    <div class="strategy-card">
        <div class="strategy-name">Equally Weighted</div>
        <div class="strategy-desc">Each asset receives an equal allocation. Simple and diversified.</div>
        <span class="strategy-risk risk-growth">Growth</span>
    </div>
    <div class="strategy-card">
        <div class="strategy-name">Max Sharpe Ratio</div>
        <div class="strategy-desc">Maximize risk-adjusted return. Best return per unit of risk taken.</div>
        <span class="strategy-risk risk-aggressive">Aggressive</span>
    </div>
    <div class="strategy-card">
        <div class="strategy-name">Max Expected Return</div>
        <div class="strategy-desc">Maximize expected annual return. Higher risk tolerance required.</div>
        <span class="strategy-risk risk-aggressive">Aggressive</span>
    </div>
    <div class="strategy-card">
        <div class="strategy-name">Leveraged Max Sharpe</div>
        <div class="strategy-desc">Max Sharpe portfolio with 1.5× leverage. Maximum growth potential.</div>
        <span class="strategy-risk risk-aggressive">Aggressive</span>
    </div>
</div>

<div class="landing-divider"></div>

<div class="landing-disclaimer">
    AI-generated guidance for informational purposes only &mdash; not professional financial advice.
    Consult a licensed financial advisor before making investment decisions.
</div>
</div>
""", unsafe_allow_html=True)

        # Navigation is handled by query param detection above (href="?started=1")
        return

    # ── Render all chat messages ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_charts"):
                params = msg.get("chart_params") or st.session_state.get("chart_params")
                if params:
                    _render_charts(params)
            elif msg.get("is_html"):
                st.markdown(msg["content"], unsafe_allow_html=True)
            elif msg["role"] == "assistant" and msg.get("animated", False):
                st.write_stream(typing_generator(msg["content"]))
                msg["animated"] = False
            else:
                st.markdown(msg["content"])

    # ── Processing phase with live status ──
    if st.session_state.phase == "processing":
        status_container = st.empty()
        completed_stages = []
        current_stage = [None]

        def on_progress(stage_label: str):
            if current_stage[0]:
                completed_stages.append(current_stage[0])
            current_stage[0] = stage_label
            status_container.markdown(
                build_stages_html(completed_stages, current_stage[0]),
                unsafe_allow_html=True,
            )

        try:
            survey_text = format_survey_for_crew()
            result = run_initial_pipeline(survey_text, on_progress=on_progress)
            st.session_state.advisor_result = result
            st.session_state.error = None

            # Mark final stage done
            if current_stage[0]:
                completed_stages.append(current_stage[0])
                current_stage[0] = None

        except Exception as e:
            st.session_state.error = str(e)
            result = None
            if current_stage[0]:
                completed_stages.append(current_stage[0])

        # Replace live container with nothing (we'll persist stages as a message)
        status_container.empty()

        # Save completed stages as a permanent chat message
        if completed_stages:
            st.session_state.messages.append({
                "role": "assistant",
                "content": build_stages_html(completed_stages),
                "is_html": True,
                "animated": False,
            })

        if st.session_state.error:
            add_assistant(f"I ran into an issue while processing: {st.session_state.error}\n\nPlease try again.")
            st.session_state.phase = "chat"
            st.rerun()
            return

        # Portfolio card
        st.session_state.messages.append({
            "role": "assistant",
            "content": build_portfolio_card(result),
            "is_html": True,
            "is_result": True,
            "animated": False,
        })

        # Store chart parameters and add charts sentinel message
        opt = result.optimization_result
        _chart_params = {
            "allocations": result.allocations,
            "expected_return": opt.expected_return,
            "expected_volatility": opt.expected_volatility,
            "investment_amount": opt.metadata.get("investment_amount", 10000.0),
            "investment_horizon": opt.metadata.get("investment_horizon", "5-10yr"),
        }
        st.session_state.chart_params = _chart_params
        st.session_state.messages.append({
            "role": "assistant",
            "is_charts": True,
            "chart_params": _chart_params,
        })

        # Text recommendation
        add_assistant(result.portfolio_recommendation)

        # Transition message
        add_assistant("That's your optimized portfolio. Feel free to ask me anything — why I chose a particular stock, how a different strategy would look, what happens if your situation changes, or anything else.")

        st.session_state.phase = "chat"
        st.rerun()
        return

    # ── Survey phase ──
    if st.session_state.phase == "survey" and st.session_state.step < TOTAL:
        q = QUESTIONS[st.session_state.step]

        if q["type"] == "options":
            # Sentinel div: CSS uses :has(#options-q-active) to hide the bottom
            # bar without removing it, so the bar's height is stable across the
            # button→text transition and Streamlit never fires a layout-shift scroll.
            st.markdown('<div id="options-q-active"></div>', unsafe_allow_html=True)
            clicked_option = None
            bottom_ctx = st.bottom() if hasattr(st, "bottom") else st.container()
            with bottom_ctx:
                for i, option in enumerate(q["options"]):
                    if st.button(option, key=f"opt_{st.session_state.step}_{i}", width='stretch'):
                        clicked_option = option
            # Dummy chat_input keeps stBottomBlockContainer in the DOM (hidden via
            # CSS above) so its presence/absence never changes between question types.
            st.chat_input("", key=f"_opts_{st.session_state.step}")
            if clicked_option:
                record_answer(clicked_option)
                st.rerun()
            else:
                components.html("""
                    <script>
                        (function() {
                            var doc = window.parent.document;
                            var el = doc.querySelector('section[data-testid="stMain"]')
                                    || doc.querySelector('section.main')
                                    || doc.querySelector('.main');
                            if (el) el.scrollTop = el.scrollHeight;
                        })();
                    </script>
                """, height=0)

        else:
            hint = q.get("hint", "Type your answer...")
            if user_input := st.chat_input(hint):
                record_answer(user_input)
                st.rerun()
            else:
                autofocus_input()

        return

    # ── Chat phase ──
    if st.session_state.phase == "chat":
        if prompt := st.chat_input("Ask about your portfolio..."):
            add_user(prompt)

            with st.chat_message("user"):
                st.markdown(prompt)

            # Live pipeline-stage display during any re-optimization
            progress_placeholder = st.empty()
            completed_stages = []
            current_stage = [None]

            def _on_followup_progress(stage_label: str):
                if current_stage[0]:
                    completed_stages.append(current_stage[0])
                current_stage[0] = stage_label
                progress_placeholder.markdown(
                    build_stages_html(completed_stages, current_stage[0]),
                    unsafe_allow_html=True,
                )

            try:
                context = build_followup_context(prompt)
                answer, new_result = run_followup_reoptimize(
                    prompt, context, st.session_state.advisor_result,
                    on_progress=_on_followup_progress,
                )
            except Exception as e:
                answer = f"I ran into an error: {e}"
                new_result = None

            # Finalise stage list and clear the live placeholder
            if current_stage[0]:
                completed_stages.append(current_stage[0])
                current_stage[0] = None
            progress_placeholder.empty()

            if new_result is not None:
                # ── Re-optimization occurred: persist stages, new card, new charts ──
                st.session_state.advisor_result = new_result

                if completed_stages:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": build_stages_html(completed_stages),
                        "is_html": True,
                        "animated": False,
                    })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": build_portfolio_card(new_result),
                    "is_html": True,
                    "is_result": True,
                    "animated": False,
                })

                opt = new_result.optimization_result
                _new_chart_params = {
                    "allocations": new_result.allocations,
                    "expected_return": opt.expected_return,
                    "expected_volatility": opt.expected_volatility,
                    "investment_amount": opt.metadata.get("investment_amount", 10000.0),
                    "investment_horizon": st.session_state.get("chart_params", {}).get(
                        "investment_horizon", "5-10yr"
                    ),
                }
                st.session_state.chart_params = _new_chart_params
                st.session_state.messages.append({
                    "role": "assistant",
                    "is_charts": True,
                    "chart_params": _new_chart_params,
                    "animated": False,
                })

                # Queue answer as animated; rerun renders everything in order
                add_assistant(answer)
                st.rerun()
            else:
                # ── Simple follow-up: stream answer directly ──
                with st.chat_message("assistant"):
                    st.write_stream(typing_generator(answer))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "animated": False,
                })
        else:
            autofocus_input()


if __name__ == "__main__":
    main()