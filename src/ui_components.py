"""
Reusable Streamlit UI components for the salary analysis app.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

from .config import (
    FEATURE_DISPLAY_NAMES, BRANCHES, BINARY_COLS,
    NUMERICAL_COLS, ORDINAL_COLS, FEATURE_COLS,
)


# Try to set a Chinese-compatible font for matplotlib
def _setup_chinese_font():
    """Configure matplotlib to support Chinese characters."""
    # Try common Chinese fonts on Windows
    chinese_fonts = [
        "Microsoft YaHei", "SimHei", "KaiTi", "FangSong",
        "SimSun", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    ]
    available = set(f.name for f in fm.fontManager.ttflist)
    for font in chinese_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    # Fallback: use sans-serif
    plt.rcParams["font.family"] = "sans-serif"


_setup_chinese_font()
# Suppress matplotlib font warnings
plt.rcParams["axes.unicode_minus"] = False


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
