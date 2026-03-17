"""
AI Financial Advisor — Conversational Streamlit App
Run:  streamlit run streamlit_app.py
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components
from ai_advisor.run_advisor import run_initial_pipeline, run_followup, AdvisorResult

# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --accent-gold: #c9a24e;
        --accent-gold-dim: rgba(201,162,78,0.12);
        --text-muted: #64748b;
        --text-secondary: #94a3b8;
        --text-primary: #f1f5f9;
        --border-subtle: #1e293b;
    }

    .stApp { font-family: 'DM Sans', sans-serif; }
    #MainMenu, header, footer, .stDeployButton { display: none !important; }

    /* Brand */
    .brand-bar {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; font-weight: 500; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--text-muted);
        padding: 1rem 0 0.5rem 0; border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1rem;
    }
    .brand-bar span { color: var(--accent-gold); }

    /* ── User messages: avatar on the right ── */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        text-align: right !important;
    }
    /* Shrink user bubble to fit text, max 80% width */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        display: inline-flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        max-width: 80% !important;
        width: auto !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.85rem !important;
        width: fit-content !important;
        max-width: 100% !important;
    }

    /* Compact stacked buttons (survey options) */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.4rem 1rem !important;
        border: 1px solid var(--border-subtle) !important;
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

    /* Portfolio card inside chat */
    .portfolio-card {
        border: 1px solid var(--border-subtle);
        border-radius: 10px; padding: 1.5rem;
        margin: 0.75rem 0;
    }
    .portfolio-card h3 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--accent-gold);
        margin: 0 0 1rem 0;
    }
    .portfolio-metrics {
        display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;
    }
    .pm-item {
        flex: 1; min-width: 100px; padding: 0.6rem 0.8rem;
        border: 1px solid var(--border-subtle); border-radius: 6px;
    }
    .pm-label {
        font-size: 0.65rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.15rem;
    }
    .pm-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem; font-weight: 600; color: var(--text-primary);
    }
    .pm-value.gold { color: var(--accent-gold); }

    .alloc-table { width: 100%; margin-top: 0.5rem; }
    .alloc-table-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem 0; border-bottom: 1px solid var(--border-subtle);
    }
    .alloc-table-row:last-child { border-bottom: none; }
    .at-ticker {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600; font-size: 0.85rem; color: var(--text-primary);
    }
    .at-bar-container {
        flex: 1; margin: 0 1rem; height: 4px;
        background: var(--border-subtle); border-radius: 2px; overflow: hidden;
    }
    .at-bar {
        height: 100%; background: var(--accent-gold); border-radius: 2px;
    }
    .at-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem; color: var(--accent-gold); min-width: 45px;
        text-align: right;
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

    /* Subtle helper text */
    .input-hint {
        font-size: 0.78rem; color: var(--text-muted);
        margin-top: -0.5rem; margin-bottom: 0.5rem;
    }

    /* Start button */
    .start-section { text-align: center; padding: 3rem 0 1rem 0; }
    .start-section h1 {
        font-size: 1.8rem; font-weight: 700;
        color: var(--text-primary); margin-bottom: 0.5rem;
    }
    .start-section p {
        font-size: 1rem; color: var(--text-secondary);
        line-height: 1.6; max-width: 460px; margin: 0 auto 2rem auto;
    }
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
    "special_considerations": "Special Considerations",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def brand():
    st.markdown(
        '<div class="brand-bar">◆ <span>Your Robo-Advisor</span> · An AI-powered tailored portfolio built just for you.</div>',
        unsafe_allow_html=True,
    )


def add_assistant(text):
    st.session_state.messages.append({"role": "assistant", "content": text})


def add_user(text):
    st.session_state.messages.append({"role": "user", "content": text})


def format_survey_for_crew():
    lines = []
    for k, v in st.session_state.answers.items():
        lines.append(f"{SURVEY_LABELS.get(k, k)}: {v}")
    return "\n".join(lines)


def build_portfolio_card(result: AdvisorResult) -> str:
    """Build HTML card for portfolio results displayed in chat."""
    opt = result.optimization_result

    # Metrics row
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

    # Allocation rows with bars
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


def advance_to_next_question():
    """Add the next survey question as an assistant message."""
    step = st.session_state.step
    if step < TOTAL:
        q = QUESTIONS[step]
        add_assistant(q["message"])


def record_answer(value: str):
    """Record answer, advance step, add next question or trigger processing."""
    step = st.session_state.step
    q = QUESTIONS[step]

    add_user(value)
    st.session_state.answers[q["key"]] = value
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
    # Find messages after the portfolio recommendation
    msgs = st.session_state.messages
    # Find the index of the portfolio card message
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
        lines.append(f"{speaker}: {m['content'][:800]}")
    lines.append(f"\nCURRENT QUESTION: {current}")
    return "\n".join(lines)


# ── Render ──────────────────────────────────────────────────────────────────

def main():
    brand()

    # ── Welcome state: show greeting ──
    if st.session_state.phase == "welcome":
        st.markdown("")
        st.markdown(
            '<div class="start-section">'
            "<h1>Your Portfolio, Optimized.</h1>"
            "<p>I'll ask you a few questions about your financial situation, "
            "goals, and risk tolerance, then build a personalized, "
            "quantitatively optimized portfolio for you.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        _, col_c, _ = st.columns([1, 1, 1])
        with col_c:
            if st.button("Let's get started →", use_container_width=True, type="primary"):
                st.session_state.phase = "survey"
                st.session_state.step = 0
                add_assistant("Great, let's build your portfolio. I'll walk you through a few questions — it should take about 3–5 minutes.")
                advance_to_next_question()
                st.rerun()
        return

    # ── Render all chat messages ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Check if this is a result card (HTML)
            if msg.get("is_html"):
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    # ── Processing phase ──
    if st.session_state.phase == "processing":
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your profile and running portfolio optimization — this will take a moment..."):
                try:
                    survey_text = format_survey_for_crew()
                    result = run_initial_pipeline(survey_text)
                    st.session_state.advisor_result = result
                    st.session_state.error = None
                except Exception as e:
                    st.session_state.error = str(e)
                    result = None

        if st.session_state.error:
            add_assistant(f"I ran into an issue while processing: {st.session_state.error}\n\nPlease try again.")
            st.session_state.phase = "chat"
            st.rerun()
            return

        # Add the portfolio card as a message
        card_html = build_portfolio_card(result)
        st.session_state.messages.append({
            "role": "assistant",
            "content": card_html,
            "is_html": True,
            "is_result": True,
        })

        # Add the text recommendation
        add_assistant(result.portfolio_recommendation)

        # Transition message
        add_assistant("That's your optimized portfolio. Feel free to ask me anything — why I chose a particular stock, how a different strategy would look, what happens if your situation changes, or anything else.")

        st.session_state.phase = "chat"
        st.rerun()
        return

    # ── Survey phase: show input for current question ──
    if st.session_state.phase == "survey" and st.session_state.step < TOTAL:
        q = QUESTIONS[st.session_state.step]

        if q["type"] == "options":
            with st.chat_message("assistant"):
                for i, option in enumerate(q["options"]):
                    if st.button(option, key=f"opt_{st.session_state.step}_{i}", use_container_width=True):
                        record_answer(option)
                        st.rerun()
            components.html("""
                <script>
                    function scrollToBottom() {
                        var doc = window.parent.document;
                        var el = doc.querySelector('section[data-testid="stMain"]')
                                || doc.querySelector('section.main')
                                || doc.querySelector('.main');
                        if (el) el.scrollTop = el.scrollHeight;
                    }
                    scrollToBottom();
                    setTimeout(scrollToBottom, 100);
                    setTimeout(scrollToBottom, 300);
                    setTimeout(scrollToBottom, 600);
                </script>
            """, height=1)

        else:
            # Free text input via chat input
            hint = q.get("hint", "Type your answer...")
            if user_input := st.chat_input(hint):
                record_answer(user_input)
                st.rerun()

        return

    # ── Chat phase: follow-up conversation ──
    if st.session_state.phase == "chat":
        if prompt := st.chat_input("Ask about your portfolio..."):
            add_user(prompt)

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(""):
                    try:
                        context = build_followup_context(prompt)
                        answer = run_followup(context, st.session_state.advisor_result)
                    except Exception as e:
                        answer = f"I ran into an error: {e}"
                st.markdown(answer)

            add_assistant(answer)


if __name__ == "__main__":
    main()
