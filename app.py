"""
Graduate Salary Analysis & Prediction Assistant
Streamlit web application with 5 tabs:
  1. Salary Prediction
  2. Factor Importance
  3. Comparison Analysis
  4. Improvement Recommendations
  5. AI Chat
"""
import sys
import os

# Ensure the src package is importable
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

# Page config — must be the first Streamlit call
st.set_page_config(
    page_title="毕业生薪资分析与预测助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apply custom theme (Phase 1) ─────────────────────
from src.ui_theme import apply_custom_theme
apply_custom_theme()

# ── Load secrets before importing config ──────────────
# Read API key from Streamlit Secrets (for Streamlit Cloud) or local secrets.toml
try:
    if "DEEPSEEK_API_KEY" in st.secrets:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass  # secrets not available, fall back to env var or config default

# Local imports (after page config)
from src.config import (
    FEATURE_DISPLAY_NAMES, BRANCHES, BINARY_COLS,
    NUMERICAL_COLS, FEATURE_COLS, MODEL_PATH, DATA_PATH,
    DEEPSEEK_MODEL,
)
from src.preprocessing import load_data, get_dataset_stats
from src.inference import predict_salary, load_model, get_feature_importance, get_model_metrics
from src.factor_analysis import (
    counterfactual_gains, compare_branches, compare_skills,
    compare_tiers, get_group_stats,
)
from src.llm_client import ask_deepseek, ask_deepseek_stream, ChatLLMClient
from src.memory import ChatMemory
from src.context_manager import ContextManager
from src.llm_prompts import (
    explain_prediction, explain_factor_importance,
    generate_advice, chat_system_prompt,
)
from src.rag import (
    load_corpus as load_rag_corpus,
    retrieve_similar_profiles,
    format_retrieved_context,
)
from src.memory import SessionMemory
from src.ui_components import (
    render_student_input_form, render_prediction_result,
    render_factor_bar_chart, render_comparison_chart,
    render_comparison_table, show_llm_section, show_dataset_overview,
    render_card_form, render_radar_chart, render_salary_comparison_gauge,
    render_context_indicator,
)


# ── Session State Initialization ──────────────────────
def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "chat_history": [],
        "model_loaded": False,
        "model": None,
        "scaler": None,
        "preprocessor": None,
        "metadata": None,
        "data_loaded": False,
        "df": None,
        "stats": None,
        "student_a": {},
        "student_b": {},
        # Session memory keys (Phase 2)
        "sm_profile": None,
        "sm_prediction": None,
        # RAG corpus (Phase 3)
        "rag_corpus": None,
        "rag_ready": False,
        # Chat memory & context (Phase 4)
        "chat_memory": None,
        "context_manager": None,
        "chat_client": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Model & Data Loading (cached) ─────────────────────
@st.cache_resource
def load_ml_model():
    """Load the trained ML model. Returns None if not trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    return load_model()


@st.cache_data
def load_dataset():
    """Load and cache the dataset plus aggregate stats."""
    df = load_data(DATA_PATH)
    stats = get_dataset_stats(df)
    return df, stats


@st.cache_resource
def load_rag_corpus_cached():
    """Load and cache the RAG corpus (TF-IDF index + aggregates)."""
    return load_rag_corpus()


@st.cache_data
def get_cached_comparison_data():
    """Cache comparison data."""
    return {
        "branches": compare_branches(),
        "skills": compare_skills(),
        "tiers": compare_tiers(),
    }


# Load on first access
if not st.session_state.model_loaded:
    model, scaler, preprocessor, metadata = load_ml_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.preprocessor = preprocessor
        st.session_state.metadata = metadata
        st.session_state.model_loaded = True

if not st.session_state.data_loaded:
    df, stats = load_dataset()
    st.session_state.df = df
    st.session_state.stats = stats
    st.session_state.data_loaded = True

# Initialize RAG corpus (Phase 3) — runs once, cached
if not st.session_state.rag_ready and st.session_state.data_loaded:
    try:
        st.session_state.rag_corpus = load_rag_corpus_cached()
        st.session_state.rag_ready = True
    except Exception:
        st.session_state.rag_corpus = None
        st.session_state.rag_ready = True  # Don't retry

# Initialize Chat Memory + Context Manager (Phase 4)
if st.session_state.chat_memory is None:
    st.session_state.chat_memory = ChatMemory()
    st.session_state.context_manager = ContextManager(st.session_state.chat_memory)
    st.session_state.chat_client = ChatLLMClient(
        st.session_state.chat_memory, st.session_state.context_manager
    )


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-size:60px;'>🎓</h1>", unsafe_allow_html=True)
    st.markdown("### 薪资分析助手")
    st.caption("基于 9000 名工程毕业生的就业数据")

    st.divider()

    # Model status
    if st.session_state.model_loaded:
        metrics = get_model_metrics()
        st.success(f"✅ 模型已加载 | R² = {metrics.get('r2', 'N/A')}")
    else:
        st.warning("⚠️ 模型尚未训练，请先运行 train.py")

    st.divider()

    # Dataset overview
    if st.session_state.stats:
        stats = st.session_state.stats
        st.markdown("**📊 数据概览**")
        st.caption(f"总学生: {stats['total_students']:,}")
        st.caption(f"就业率: {stats['placement_rate']}%")
        st.caption(f"平均薪资: ₹{stats['avg_salary']:.2f} LPA")

    st.divider()

    # LLM info
    st.markdown("**🤖 LLM 配置**")
    st.caption(f"模型: {DEEPSEEK_MODEL}")
    st.caption("API: DeepSeek")

    st.divider()
    st.caption("© 2026 Graduate Salary Analyzer")


# ── Main Content ──────────────────────────────────────
st.title("🎓 毕业生薪资分析与预测助手")
st.caption("基于机器学习 + AI 大模型的智能职业规划工具")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 薪资预测",
    "📊 因素重要性分析",
    "📈 专业/技能对比",
    "💡 个性化建议",
    "💬 AI 对话",
])


# ═══════════════════════════════════════════════════════
# TAB 1: Salary Prediction
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("### 💰 毕业生薪资预测")
    st.markdown("填写以下学生信息，模型将预测其毕业年薪，并由 AI 给出详细解读。")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        # Use card-based form (Phase 5)
        student = render_card_form(prefix="tab1_")

        predict_btn = st.button("🔮 预测薪资", type="primary", use_container_width=True,
                                disabled=not st.session_state.model_loaded)

    with col_result:
        if predict_btn and st.session_state.model_loaded:
            with st.spinner("ML 模型预测中..."):
                prediction = predict_salary(
                    student,
                    st.session_state.model,
                    st.session_state.scaler,
                    st.session_state.preprocessor,
                    st.session_state.metadata,
                )
                st.session_state.last_prediction = prediction
                st.session_state.last_student = student
                # Persist profile across tabs (Phase 2)
                SessionMemory.save_profile(student)
                SessionMemory.save_prediction(prediction)

            render_prediction_result(prediction)

            # Radar chart + Salary gauge (Phase 5)
            radar_skills = {
                "Python": student.get("python_skill", 0),
                "DSA": student.get("dsa_skill", 0),
                "ML": student.get("ml_skill", 0),
                "WebDev": student.get("web_dev_skill", 0),
                "实习/项目": min(
                    (student.get("internships", 0) + student.get("projects", 0)) / 10, 1.0
                ),
            }
            render_radar_chart(radar_skills, "技能画像")

            # Benchmark: median salary from stats
            benchmark = st.session_state.stats.get("median_salary", 50) if st.session_state.stats else 50
            render_salary_comparison_gauge(
                prediction["predicted_salary_lpa"],
                benchmark,
                "预测薪资 vs 数据集中位数",
            )

            # LLM explanation with RAG context (Phase 3)
            with st.spinner("AI 正在解读预测结果..."):
                # Retrieve similar profiles from corpus
                rag_context = ""
                if st.session_state.rag_corpus:
                    try:
                        similar = retrieve_similar_profiles(
                            student, st.session_state.rag_corpus
                        )
                        rag_context = format_retrieved_context(
                            similar, st.session_state.rag_corpus
                        )
                    except Exception:
                        rag_context = ""
                sys_prompt, user_msg = explain_prediction(
                    student, prediction, rag_context=rag_context
                )
                llm_response = ask_deepseek(sys_prompt, user_msg)

            st.divider()
            st.markdown("**🤖 AI 解读：**")
            show_llm_section("预测解读", llm_response)

        elif predict_btn and not st.session_state.model_loaded:
            st.error("❌ 模型未加载。请先运行 `python src/train.py` 训练模型。")

        else:
            # Placeholder when no prediction yet
            st.info("👈 请在左侧填写学生信息，然后点击「预测薪资」按钮。")
            st.markdown("""
            **使用说明：**
            1. 调整各项参数以匹配一名学生的真实情况
            2. 点击「预测薪资」按钮
            3. ML 模型将给出薪资预测
            4. AI 将为预测结果提供详细解读和职业建议
            """)


# ═══════════════════════════════════════════════════════
# TAB 2: Factor Importance Analysis
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 薪资影响因素分析")
    st.markdown("了解哪些因素对毕业生薪资影响最大，为职业规划提供数据支撑。")

    if not st.session_state.model_loaded:
        st.warning("⚠️ 模型尚未训练，请先运行 `python src/train.py`")
    else:
        col_chart, col_insight = st.columns([3, 2], gap="large")

        with col_chart:
            importance_data = get_feature_importance()
            st.markdown("**🏆 特征重要性排名**")
            render_factor_bar_chart(importance_data, top_n=15)

        with col_insight:
            st.markdown("**🤖 AI 分析洞察**")

            if st.button("🔄 生成 AI 分析", key="tab2_llm", type="primary"):
                with st.spinner("AI 分析中..."):
                    # RAG: aggregate stats (Phase 3)
                    rag_context = ""
                    if st.session_state.rag_corpus:
                        try:
                            from src.rag import retrieve_aggregate_stats
                            agg = retrieve_aggregate_stats(
                                "all", st.session_state.rag_corpus
                            )
                            # Format as compact summary
                            lines = ["[数据集统计]\n"]
                            for b, s in agg.get("by_branch", {}).items():
                                lines.append(f"{b}专业 均薪{s['mean']}LPA")
                            rag_context = "; ".join(lines[:8])
                        except Exception:
                            rag_context = ""
                    sys_prompt, user_msg = explain_factor_importance(
                        importance_data, rag_context=rag_context
                    )
                    llm_response = ask_deepseek(sys_prompt, user_msg)
                    st.session_state.factor_llm_response = llm_response

            if "factor_llm_response" in st.session_state:
                show_llm_section("因素分析", st.session_state.factor_llm_response)
            else:
                st.info("点击上方按钮，让 AI 为你解读这些特征重要性的含义。")

        # Feature importance table
        st.divider()
        st.markdown("**📋 详细特征重要性表**")
        imp_df = pd.DataFrame(importance_data)
        imp_df["feature_cn"] = imp_df["feature"].apply(
            lambda f: FEATURE_DISPLAY_NAMES.get(f, f)
        )
        imp_df = imp_df[["feature_cn", "feature", "importance"]]
        imp_df.columns = ["特征名称", "特征键", "重要性分数"]
        st.dataframe(imp_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# TAB 3: Comparison Analysis
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 专业与技能对比分析")
    st.markdown("对比不同专业、技能组合的薪资差异，以及两名学生之间的预测对比。")

    # Sub-tabs for different comparison types
    comp_tab1, comp_tab2, comp_tab3 = st.tabs([
        "👥 双学生对比",
        "🏫 专业/等级对比",
        "🛠️ 技能溢价分析",
    ])

    # -- Sub-tab: Two-Student Comparison --
    with comp_tab1:
        st.markdown("**对比两名学生的预测薪资**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 学生 A")
            student_a = render_student_input_form(prefix="cmp_a_",
                                                   defaults={"branch": "CSE", "cgpa": 8.0})
        with col_b:
            st.markdown("##### 学生 B")
            student_b = render_student_input_form(prefix="cmp_b_",
                                                   defaults={"branch": "Mechanical", "cgpa": 6.5})

        if st.button("🔍 对比分析", key="compare_btn", type="primary",
                     disabled=not st.session_state.model_loaded):
            with st.spinner("分析中..."):
                pred_a = predict_salary(student_a, st.session_state.model,
                                        st.session_state.scaler, st.session_state.preprocessor, st.session_state.metadata)
                pred_b = predict_salary(student_b, st.session_state.model,
                                        st.session_state.scaler, st.session_state.preprocessor, st.session_state.metadata)

            col_res_a, col_res_b = st.columns(2)
            with col_res_a:
                st.markdown("##### 学生 A 预测结果")
                render_prediction_result(pred_a)
            with col_res_b:
                st.markdown("##### 学生 B 预测结果")
                render_prediction_result(pred_b)

            # Difference
            diff = pred_a["predicted_salary_lpa"] - pred_b["predicted_salary_lpa"]
            st.info(f"💰 薪资差距: **{abs(diff):.2f} LPA** "
                    f"({'学生 A 更高' if diff > 0 else '学生 B 更高' if diff < 0 else '持平'})")

            # LLM comparison
            with st.spinner("AI 正在分析差异..."):
                sys_prompt = "你是一个数据分析专家，擅长对比分析。用中文回复，不超过 200 字。"
                user_msg = (
                    f"学生 A（{student_a['branch']}专业，CGPA {student_a['cgpa']}）"
                    f"预测薪资 {pred_a['predicted_salary_lpa']} LPA。\n"
                    f"学生 B（{student_b['branch']}专业，CGPA {student_b['cgpa']}）"
                    f"预测薪资 {pred_b['predicted_salary_lpa']} LPA。\n"
                    f"请简要分析造成薪资差异的关键因素。"
                )
                llm_compare = ask_deepseek(sys_prompt, user_msg)
            show_llm_section("差异分析", llm_compare)

    # -- Sub-tab: Branch/Tier Comparison --
    with comp_tab2:
        comp_data = get_cached_comparison_data()

        st.markdown("**各专业薪资对比**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            render_comparison_chart(
                comp_data["branches"], "branch", "avg_salary",
                "Average Salary by Branch"
            )
        with col_c2:
            render_comparison_table(comp_data["branches"], "专业详细对比")

        st.divider()
        st.markdown("**大学等级薪资对比**")
        render_comparison_table(comp_data["tiers"], "各等级大学平均薪资")

    # -- Sub-tab: Skill Premium --
    with comp_tab3:
        comp_data = get_cached_comparison_data()

        st.markdown("**各项技能的薪资溢价**")
        render_comparison_table(comp_data["skills"], "技能薪资溢价分析")

        st.markdown("""
        **说明：**
        - **premium** = 拥有该技能的平均薪资 - 没有该技能的平均薪资
        - 正值表示该技能能带来薪资提升
        - 数值越大，该技能在就业市场中的溢价越明显
        """)


# ═══════════════════════════════════════════════════════
# TAB 4: Improvement Recommendations
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("### 💡 个性化提升建议")
    st.markdown("输入学生当前条件，系统将分析各因素改善后的预期薪资提升，AI 会给出优先级建议。")

    col_form, col_advice = st.columns([1, 1], gap="large")

    with col_form:
        # Pre-fill from session memory if available (Phase 2)
        saved_profile = SessionMemory.get_profile()
        imp_defaults = saved_profile if saved_profile else None
        student_imp = render_card_form(prefix="imp_", defaults=imp_defaults)

        analyze_btn = st.button("💡 分析提升空间", type="primary", use_container_width=True,
                                disabled=not st.session_state.model_loaded)

    with col_advice:
        if analyze_btn and st.session_state.model_loaded:
            with st.spinner("计算各因素的边际增益..."):
                base_pred = predict_salary(
                    student_imp, st.session_state.model,
                    st.session_state.scaler, st.session_state.preprocessor, st.session_state.metadata,
                )
                gains = counterfactual_gains(student_imp)

            st.markdown(f"**当前预测薪资: ₹{base_pred['predicted_salary_lpa']:.2f} LPA**")

            if gains:
                st.markdown("**📈 各因素提升空间：**")
                gains_df = pd.DataFrame(gains)
                gains_df["feature_cn"] = gains_df["feature"].apply(
                    lambda f: FEATURE_DISPLAY_NAMES.get(f, f)
                )
                display_df = gains_df[[
                    "feature_cn", "current_value", "improved_value", "salary_gain_lpa"
                ]]
                display_df.columns = ["因素", "当前值", "改善后", "预期薪资提升(LPA)"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # LLM advice with RAG context (Phase 3)
            with st.spinner("AI 正在生成个性化建议..."):
                rag_context = ""
                if st.session_state.rag_corpus:
                    try:
                        similar = retrieve_similar_profiles(
                            student_imp, st.session_state.rag_corpus
                        )
                        rag_context = format_retrieved_context(
                            similar, st.session_state.rag_corpus
                        )
                    except Exception:
                        rag_context = ""
                sys_prompt, user_msg = generate_advice(
                    student_imp, base_pred["predicted_salary_lpa"], gains,
                    rag_context=rag_context,
                )
                llm_advice = ask_deepseek(sys_prompt, user_msg)

            st.divider()
            st.markdown("**🤖 AI 个性化建议：**")
            show_llm_section("职业建议", llm_advice)

        elif analyze_btn and not st.session_state.model_loaded:
            st.error("❌ 模型未加载。请先运行 `python src/train.py` 训练模型。")
        else:
            st.info("👈 请在左侧填写学生信息，点击「分析提升空间」获取个性化建议。")


# ═══════════════════════════════════════════════════════
# TAB 5: AI Chat
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("### 💬 AI 职业顾问对话")
    st.markdown("与 AI 助手自由对话，询问关于薪资、就业、技能提升等各方面的问题。")

    # Display chat history from ChatMemory (Phase 4)
    for msg in st.session_state.chat_memory.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    # Chat input
    if user_input := st.chat_input("输入你的问题，例如：Python 技能对薪资有多大影响？"):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate streaming AI response (Phase 4)
        with st.chat_message("assistant"):
            # Build context
            stats = st.session_state.stats
            stats_context = ""
            if stats:
                stats_context = (
                    f"当前数据集包含 {stats['total_students']} 名学生，"
                    f"就业率 {stats['placement_rate']}%，平均薪资 {stats['avg_salary']:.2f} LPA。"
                )

            # Profile context from session memory
            profile_ctx = SessionMemory.profile_to_context(
                SessionMemory.get_profile()
            )

            # Auto-summarize if chat history is too long
            st.session_state.context_manager.maybe_summarize(
                lambda text: ask_deepseek(
                    "请用简洁中文总结以下对话要点（不超过150字）",
                    text,
                )
            )

            # Build RAG context if corpus available
            rag_ctx = ""
            if st.session_state.rag_corpus and SessionMemory.has_profile():
                try:
                    similar = retrieve_similar_profiles(
                        SessionMemory.get_profile(),
                        st.session_state.rag_corpus,
                    )
                    rag_ctx = format_retrieved_context(
                        similar, st.session_state.rag_corpus
                    )
                except Exception:
                    rag_ctx = ""

            # Use ChatLLMClient with streaming
            sys_prompt = chat_system_prompt()
            full_user_msg = f"{user_input}\n\n[数据上下文] {stats_context}"

            response_placeholder = st.empty()
            streamed_text = ""
            try:
                for chunk in st.session_state.chat_client.chat_stream(
                    user_message=full_user_msg,
                    system_prompt=sys_prompt,
                    rag_context=rag_ctx,
                    profile_context=profile_ctx,
                ):
                    streamed_text += chunk
                    response_placeholder.markdown(streamed_text + "▌")
                response_placeholder.markdown(streamed_text)
            except Exception:
                if not streamed_text:
                    streamed_text = ask_deepseek(sys_prompt, full_user_msg)
                    st.markdown(streamed_text)

        # Context usage indicator (Phase 5)
        tokens_used = st.session_state.context_manager.count_tokens(streamed_text) if streamed_text else 0
        render_context_indicator(tokens_used, 3500)

    # Clear chat button
    if st.session_state.chat_memory.size() > 0:
        if st.button("🗑️ 清空对话", key="clear_chat"):
            st.session_state.chat_memory.clear()
            st.rerun()


# ── Footer ────────────────────────────────────────────
st.divider()
st.caption(
    "💡 提示：薪资预测基于 XGBoost 机器学习模型 + DeepSeek AI 解读。"
    "预测结果仅供参考，实际薪资受市场环境、个人表现等多种因素影响。"
)
