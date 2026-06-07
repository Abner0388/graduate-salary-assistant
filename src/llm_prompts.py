"""
Prompt templates for the LLM-powered features.
Each function returns a (system_prompt, user_message) tuple.
"""
import json
from .config import FEATURE_DISPLAY_NAMES, BRANCHES


def explain_prediction(
    student: dict, prediction: dict, rag_context: str = ""
) -> tuple[str, str]:
    """Generate a prompt to explain an individual salary prediction."""
    # Inject RAG context into system prompt if provided
    rag_block = f"\n\n[参考数据]\n{rag_context}" if rag_context else ""

    # Map feature values to Chinese display names
    profile_lines = []
    name_map = FEATURE_DISPLAY_NAMES
    for key, val in student.items():
        display = name_map.get(key, key)
        if key == "branch":
            profile_lines.append(f"- {display}: {val}")
        elif isinstance(val, float):
            profile_lines.append(f"- {display}: {val:.2f}")
        else:
            profile_lines.append(f"- {display}: {val}")

    top_factors_str = "\n".join(
        f"  {i+1}. {f['feature']} (feature value: {f['value']:.3f}, contribution score: {f['score']:.4f})"
        for i, f in enumerate(prediction.get("top_factors", []))
    )

    system = (
        "你是一名资深职业顾问兼数据科学专家，专精于分析毕业生薪资预测模型的结果。"
        "你需要用中文回复，语言亲切、专业、有鼓励性。回复不超过 200 字。"
        f"{rag_block}"
    )

    user = f"""一位学生的个人资料如下：

{chr(10).join(profile_lines)}

机器学习模型预测该学生的年薪为 **{prediction['predicted_salary_lpa']} LPA**（约合 {prediction['predicted_salary_lpa']:.0f} 万印度卢比/年）。

对该预测影响最大的前三个因素：
{top_factors_str if top_factors_str else '无数据'}

请用自然语言解释这个薪资预测的含义，包括：
1. 这个薪资在当前就业市场中处于什么水平
2. 哪些因素对预测结果帮助最大，哪些拖了后腿
3. 给出一条具体、可操作的提升建议

请用中文回答，鼓励性的语气。"""

    return system, user


def explain_factor_importance(
    importance_data: list[dict], rag_context: str = ""
) -> tuple[str, str]:
    """Generate a prompt to narrate feature importance findings."""
    rag_block = f"\n\n[数据集统计]\n{rag_context}" if rag_context else ""
    top_features = importance_data[:10]
    features_str = "\n".join(
        f"{i+1}. {f['feature']}: {f['importance']:.6f}"
        for i, f in enumerate(top_features)
    )

    system = (
        "你是一名数据科学教育者，擅长用通俗易懂的语言解释机器学习模型的结果。"
        "请用中文回复，控制在 250 字以内。"
        f"{rag_block}"
    )

    user = f"""我们的薪资预测模型（基于 9000 名工程毕业生的数据训练）显示出以下前 10 个最重要的影响因素及其重要性分数：

{features_str}

请解释这些发现对在校学生意味着什么。请将相关因素分组（如技术能力、学习成绩、项目经验），并给出实用的建议。
用中文回答，对话式语气。"""

    return system, user


def generate_advice(
    student: dict,
    base_salary: float,
    gains: list[dict],
    rag_context: str = "",
) -> tuple[str, str]:
    """Generate a prompt for personalized career improvement advice."""
    rag_block = f"\n\n[参考数据]\n{rag_context}" if rag_context else ""
    profile_lines = []
    name_map = FEATURE_DISPLAY_NAMES
    for key, val in student.items():
        display = name_map.get(key, key)
        profile_lines.append(f"- {display}: {val}")

    gains_lines = "\n".join(
        f"- 改善 **{name_map.get(g['feature'], g['feature'])}**（{g['current_value']} → {g['improved_value']}）：预计薪资提升 **{g['salary_gain_lpa']} LPA**"
        for g in gains[:8]
    )

    system = (
        "你是一名富有同理心的工程类毕业生职业顾问。你的建议应该具体、可操作、有优先级。"
        "用中文回复，控制在 300 字以内，鼓励性语气。"
        f"{rag_block}"
    )

    user = f"""一位 {student.get('branch', '未知专业')} 专业的学生，当前预测薪资为 **{base_salary} LPA**。

当前条件：
{chr(10).join(profile_lines)}

模型分析显示，改善以下各方面可以带来的薪资增长：
{gains_lines}

请给出 3-4 条具体、有优先级的改善建议。需要考虑现实约束（学生无法立刻改变 CGPA 或大学等级）。重点放在技能提升、项目经验和实习上。用中文回答。"""

    return system, user


def chat_system_prompt() -> str:
    """Return the system prompt for the free-form chat tab."""
    return (
        "你是一个毕业生薪资分析与预测助手，拥有对 9000 名工程毕业生的数据分析能力。"
        "数据涵盖 CSE、IT、ECE、Mechanical、Civil、EEE 六个专业，3 个大学等级。"
        "就业公司包括 MNC（跨国公司）、Mid-size（中型企业）和 Startup（初创公司）。"
        "年薪范围大约在 15 到 100+ LPA（1 LPA = 10 万印度卢比/年）。"
        "你可以回答关于薪资趋势、技能价值、就业市场分析等各方面的问题。"
        "请用中文回复，提供有数据支撑的见解。如果没有具体数据支持，请诚实说明。"
        "保持回答简洁、有帮助性。"
    )
