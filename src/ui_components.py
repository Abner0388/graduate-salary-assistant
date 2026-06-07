"""
Reusable Streamlit UI components for the salary analysis app.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from .config import (
    FEATURE_DISPLAY_NAMES, BRANCHES, BINARY_COLS,
    NUMERICAL_COLS, ORDINAL_COLS, FEATURE_COLS,
)
from .ui_theme import setup_matplotlib_chinese, COLORS

# Chinese font auto-configured at import time
setup_matplotlib_chinese()


# Cache the feature name mapping
@st.cache_data
def get_display_name(feature_key: str) -> str:
    """Look up the Chinese display name for a feature key."""
    return FEATURE_DISPLAY_NAMES.get(feature_key, feature_key)


def render_student_input_form(
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """
    Render a student profile input form.

    Args:
        prefix: Unique prefix for widget keys (for multiple forms on one page).
        defaults: Optional dict of default values.

    Returns:
        dict with all FEATURE_COLS keys and their values.
    """
    if defaults is None:
        defaults = {}

    student = {}

    with st.container(border=True):
        st.markdown(f"**📋 学生信息**")

        col1, col2, col3 = st.columns(3)

        with col1:
            student["cgpa"] = st.slider(
                "CGPA",
                min_value=5.0, max_value=10.0,
                value=float(defaults.get("cgpa", 7.5)),
                step=0.1,
                key=f"{prefix}cgpa",
            )
            student["branch"] = st.selectbox(
                "专业 (Branch)",
                options=BRANCHES,
                index=BRANCHES.index(defaults.get("branch", "CSE")) if defaults.get("branch", "CSE") in BRANCHES else 0,
                key=f"{prefix}branch",
            )
            student["college_tier"] = st.selectbox(
                "大学等级 (College Tier)",
                options=[1, 2, 3],
                index=[1, 2, 3].index(defaults.get("college_tier", 2)),
                key=f"{prefix}tier",
            )

        with col2:
            st.markdown("**技能**")
            student["python_skill"] = 1 if st.checkbox(
                "Python", value=bool(defaults.get("python_skill", 1)),
                key=f"{prefix}py"
            ) else 0
            student["dsa_skill"] = 1 if st.checkbox(
                "数据结构与算法 (DSA)", value=bool(defaults.get("dsa_skill", 1)),
                key=f"{prefix}dsa"
            ) else 0
            student["ml_skill"] = 1 if st.checkbox(
                "机器学习 (ML)", value=bool(defaults.get("ml_skill", 0)),
                key=f"{prefix}ml"
            ) else 0
            student["web_dev_skill"] = 1 if st.checkbox(
                "Web 开发", value=bool(defaults.get("web_dev_skill", 0)),
                key=f"{prefix}web"
            ) else 0

        with col3:
            student["internships"] = st.number_input(
                "实习次数",
                min_value=0, max_value=10,
                value=int(defaults.get("internships", 1)),
                key=f"{prefix}intern",
            )
            student["projects"] = st.number_input(
                "项目数量",
                min_value=0, max_value=15,
                value=int(defaults.get("projects", 3)),
                key=f"{prefix}proj",
            )
            student["backlogs"] = st.number_input(
                "挂科数量",
                min_value=0, max_value=10,
                value=int(defaults.get("backlogs", 0)),
                key=f"{prefix}back",
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            student["coding_score"] = st.slider(
                "编程能力评分",
                min_value=0.0, max_value=100.0,
                value=float(defaults.get("coding_score", 50.0)),
                step=1.0,
                key=f"{prefix}cod",
            )
        with col5:
            student["communication_score"] = st.slider(
                "沟通能力评分",
                min_value=0.0, max_value=10.0,
                value=float(defaults.get("communication_score", 7.0)),
                step=0.1,
                key=f"{prefix}comm",
            )
        with col6:
            student["aptitude_score"] = st.slider(
                "能力倾向评分",
                min_value=0.0, max_value=100.0,
                value=float(defaults.get("aptitude_score", 70.0)),
                step=1.0,
                key=f"{prefix}apt",
            )

        col7, col8 = st.columns(2)
        with col7:
            student["resume_score"] = st.slider(
                "简历评分",
                min_value=0.0, max_value=100.0,
                value=float(defaults.get("resume_score", 65.0)),
                step=1.0,
                key=f"{prefix}res",
            )
        with col8:
            student["skill_score"] = st.slider(
                "综合技能评分",
                min_value=0, max_value=4,
                value=int(defaults.get("skill_score", 2)),
                key=f"{prefix}skill",
            )

    return student


def render_prediction_result(prediction: dict):
    """Display the predicted salary result prominently."""
    salary = prediction["predicted_salary_lpa"]

    # Determine salary tier for color
    if salary >= 80:
        color = "#4CAF50"  # green - high
        tier = "高薪区间"
    elif salary >= 50:
        color = "#2196F3"  # blue - mid
        tier = "中等偏上"
    elif salary >= 30:
        color = "#FF9800"  # orange - low-mid
        tier = "中等区间"
    else:
        color = "#F44336"  # red - low
        tier = "起步区间"

    st.markdown(f"""
    <div style="text-align:center; padding:20px; border-radius:10px; background-color:{color}15; border:2px solid {color};">
        <p style="font-size:14px; color:gray; margin:0;">预测年薪</p>
        <h1 style="font-size:48px; color:{color}; margin:10px 0;">₹ {salary:.2f} LPA</h1>
        <p style="font-size:14px; color:gray; margin:0;">
            约合 ₹ {salary * 100000:,.0f} / 年 ｜ {tier}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show top factors
    if prediction.get("top_factors"):
        st.markdown("**🎯 主要影响因素：**")
        for i, f in enumerate(prediction["top_factors"]):
            feature_cn = get_display_name(f["feature"])
            st.markdown(f"{i+1}. **{feature_cn}** — 贡献度: {f['score']:.4f}")


def render_factor_bar_chart(importance_data: list[dict], top_n: int = 15):
    """Plot a horizontal bar chart of feature importance."""
    if not importance_data:
        st.warning("暂无特征重要性数据。请先训练模型。")
        return

    data = importance_data[:top_n]
    features = [get_display_name(d["feature"]) for d in data]
    scores = [d["importance"] for d in data]

    # Reverse for horizontal bar (top at top)
    features = features[::-1]
    scores = scores[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis([i / len(scores) for i in range(len(scores))])[::-1]

    bars = ax.barh(range(len(features)), scores, color=colors, height=0.6)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importance Ranking", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + max(scores) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    st.pyplot(fig)


def render_comparison_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Render a comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(data[x_col], data[y_col], color="#2196F3", width=0.5)
    ax.set_xlabel(get_display_name(x_col), fontsize=11)
    ax.set_ylabel("Average Salary (LPA)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)


def render_comparison_table(data: pd.DataFrame, title: str = ""):
    """Render a styled comparison table."""
    if title:
        st.markdown(f"**{title}**")
    st.dataframe(
        data.style.format(precision=2).highlight_max(subset=data.select_dtypes("number").columns, color="#e8f5e9"),
        use_container_width=True,
    )


def show_llm_section(label: str, response: str):
    """Display an LLM response in a styled box."""
    if response.startswith("[AI"):
        st.info(response)
    else:
        with st.chat_message("assistant"):
            st.markdown(response)


def show_dataset_overview(stats: dict):
    """Show a high-level dataset overview in metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("学生总数", f"{stats['total_students']:,}")
    with col2:
        st.metric("已就业", f"{stats['placed_students']:,}")
    with col3:
        st.metric("就业率", f"{stats['placement_rate']}%")
    with col4:
        st.metric("平均年薪", f"₹{stats['avg_salary']:.2f} LPA")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("中位数年薪", f"₹{stats['median_salary']:.2f} LPA")
    with col6:
        st.metric("最低年薪", f"₹{stats['min_salary']:.2f} LPA")
    with col7:
        st.metric("最高年薪", f"₹{stats['max_salary']:.2f} LPA")
    with col8:
        st.metric("标准差", f"₹{stats['std_salary']:.2f} LPA")


# ═══════════════════════════════════════════════════════════
# Phase 5 New Components
# ═══════════════════════════════════════════════════════════


def render_card_form(
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """
    Card-based student input form with visual grouping.
    Groups fields into logical cards instead of a flat 8-column grid.

    Args:
        prefix: Unique prefix for widget keys.
        defaults: Optional dict of default values.

    Returns:
        dict with keys matching FEATURE_COLS.
    """
    if defaults is None:
        defaults = {}

    student = {}
    p = prefix

    # ── Card 1: Basic Info ─────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">👤 基本信息</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        student["cgpa"] = st.slider(
            "CGPA",
            min_value=5.0, max_value=10.0,
            value=float(defaults.get("cgpa", 7.5)),
            step=0.1, key=f"{p}cgpa",
        )
    with c2:
        branch_idx = BRANCHES.index(defaults.get("branch", "CSE")) if defaults.get("branch", "CSE") in BRANCHES else 0
        student["branch"] = st.selectbox(
            "专业 (Branch)",
            options=BRANCHES, index=branch_idx,
            key=f"{p}branch",
        )
    with c3:
        student["college_tier"] = st.selectbox(
            "大学等级",
            options=[1, 2, 3],
            index=int(defaults.get("college_tier", 2)) - 1,
            format_func=lambda t: {1: "Tier 1 — 顶尖院校", 2: "Tier 2 — 中等院校", 3: "Tier 3 — 一般院校"}[t],
            key=f"{p}tier",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 2: Technical Skills ──────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">💻 技术技能</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        student["python_skill"] = 1 if st.checkbox(
            "🐍 Python",
            value=bool(defaults.get("python_skill", 0)),
            key=f"{p}python",
        ) else 0
    with s2:
        student["dsa_skill"] = 1 if st.checkbox(
            "📊 数据结构与算法",
            value=bool(defaults.get("dsa_skill", 0)),
            key=f"{p}dsa",
        ) else 0
    with s3:
        student["ml_skill"] = 1 if st.checkbox(
            "🤖 机器学习",
            value=bool(defaults.get("ml_skill", 0)),
            key=f"{p}ml",
        ) else 0
    with s4:
        student["web_dev_skill"] = 1 if st.checkbox(
            "🌐 Web 开发",
            value=bool(defaults.get("web_dev_skill", 0)),
            key=f"{p}web",
        ) else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 3: Scores & Experience ───────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">📈 成绩与经历</div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        student["coding_score"] = st.slider(
            "编程能力评分",
            min_value=0.0, max_value=100.0,
            value=float(defaults.get("coding_score", 50.0)),
            step=1.0, key=f"{p}coding",
        )
        student["communication_score"] = st.slider(
            "沟通能力评分 (1-10)",
            min_value=0.0, max_value=10.0,
            value=float(defaults.get("communication_score", 7.0)),
            step=0.1, key=f"{p}comm",
        )
    with r2:
        student["aptitude_score"] = st.slider(
            "能力倾向评分",
            min_value=0.0, max_value=100.0,
            value=float(defaults.get("aptitude_score", 70.0)),
            step=1.0, key=f"{p}apt",
        )
        student["internships"] = st.number_input(
            "实习次数",
            min_value=0, max_value=10,
            value=int(defaults.get("internships", 1)),
            step=1, key=f"{p}intern",
        )
    with r3:
        student["projects"] = st.number_input(
            "项目数量",
            min_value=0, max_value=15,
            value=int(defaults.get("projects", 3)),
            step=1, key=f"{p}proj",
        )
        student["backlogs"] = st.number_input(
            "挂科数",
            min_value=0, max_value=10,
            value=int(defaults.get("backlogs", 0)),
            step=1, key=f"{p}back",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 4: Overall Assessment ────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">📋 综合评价</div>',
        unsafe_allow_html=True,
    )

    o1, o2 = st.columns(2)
    with o1:
        student["resume_score"] = st.slider(
            "简历评分",
            min_value=0.0, max_value=100.0,
            value=float(defaults.get("resume_score", 65.0)),
            step=1.0, key=f"{p}resume",
        )
    with o2:
        student["skill_score"] = st.slider(
            "综合技能评分",
            min_value=0, max_value=4,
            value=int(defaults.get("skill_score", 2)),
            step=1, key=f"{p}skill",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    return student


def render_radar_chart(skills: dict, title: str = "技能雷达图"):
    """
    Render a Plotly radar chart showing skill scores.

    Args:
        skills: Dict mapping skill names to values (0-1 or 0-100).
        title: Chart title.
    """
    import plotly.graph_objects as go

    categories = list(skills.keys())
    values = list(skills.values())

    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(26, 115, 232, 0.2)",
        line=dict(color="#1a73e8", width=2),
        name="技能水平",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(values) * 1.2, 1)],
                showticklabels=False,
            ),
        ),
        title=dict(text=title, x=0.5, font=dict(size=14)),
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=20),
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"radar_{title}")


def render_salary_comparison_gauge(
    salary: float,
    benchmark: float,
    title: str = "",
):
    """
    Render a gauge-like comparison showing personal salary vs. benchmark.
    Uses Plotly indicator.

    Args:
        salary: The predicted/actual salary in LPA.
        benchmark: The benchmark value (e.g., median salary for the branch).
        title: Optional title.
    """
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=salary,
        delta={"reference": benchmark, "increasing": {"color": "#34a853"}},
        title={"text": title or "薪资 vs 基准"},
        gauge={
            "axis": {"range": [0, max(salary, benchmark) * 1.3]},
            "bar": {"color": "#1a73e8"},
            "steps": [
                {"range": [0, benchmark * 0.5], "color": "#fce8e6"},
                {"range": [benchmark * 0.5, benchmark], "color": "#e8f0fe"},
                {"range": [benchmark, benchmark * 1.2], "color": "#e6f4ea"},
            ],
            "threshold": {
                "line": {"color": "#ea4335", "width": 2},
                "thickness": 0.75,
                "value": benchmark,
            },
        },
        number={"suffix": " LPA", "font": {"size": 22}},
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{title}")


def render_context_indicator(used: int, max_tokens: int):
    """
    Render a small context window usage indicator.

    Args:
        used: Number of tokens currently used.
        max_tokens: Maximum token budget.
    """
    ratio = min(used / max_tokens, 1.0) if max_tokens > 0 else 0

    if ratio < 0.6:
        color = "#34a853"
    elif ratio < 0.85:
        color = "#f9ab00"
    else:
        color = "#ea4335"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
            <span style="font-size:12px;color:#5f6368;">上下文</span>
            <div class="context-bar-container" style="flex:1;">
                <div class="context-bar-fill" style="width:{ratio*100}%;background:{color};"></div>
            </div>
            <span style="font-size:11px;color:#5f6368;">{used}/{max_tokens} tokens</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_llm_section_streaming(label: str, stream_generator):
    """
    Render an LLM response section with word-by-word streaming.
    Uses st.write_stream for incremental display.

    Args:
        label: Section label for ARIA.
        stream_generator: Yields text chunks.
    """
    st.write_stream(stream_generator)
