# 🎓 Graduate Salary Analysis & Prediction Assistant

> 基于机器学习 + 大语言模型的毕业生薪资智能分析与预测系统

[![Streamlit](https://img.shields.io/badge/Streamlit-Online-red?logo=streamlit)](https://你的应用名.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green)](https://xgboost.readthedocs.io/)
[![LLM](https://img.shields.io/badge/LLM-DeepSeek_V4_Pro-purple)](https://deepseek.com)

## 📌 项目概述

针对 9,000 名工程类毕业生的就业数据，构建了一个集**薪资预测**、**因素分析**和**个性化建议**于一体的智能分析平台。系统采用 **XGBoost 回归模型**实现精确薪资预测（MAPE 12%），结合 **DeepSeek V4 Pro 大语言模型**提供自然语言解读与职业指导，通过 **Streamlit** 构建交互式 Web 界面并部署上线。

## ✨ 核心功能

| 功能模块 | 说明 |
|---------|------|
| 💰 **薪资预测** | 输入学生专业、技能、成绩等指标，ML 模型实时预测毕业年薪 + AI 解读 |
| 📊 **因素重要性分析** | 量化 18 个特征对薪资的影响权重，揭示技能溢价与市场规律 |
| 📈 **专业/技能对比** | 多维度横向对比：不同专业薪资差异、技能组合溢价分析 |
| 💡 **个性化建议** | 反事实分析 + LLM 生成优先级提升方案，告诉学生"从哪里入手最划算" |
| 💬 **AI 职业顾问** | 基于数据集上下文的自由对话，回答就业市场相关问题 |

## 🛠 技术架构

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ 9000 条学生  │────▶│ XGBoost Regressor │────▶│ 薪资预测 (LPA)   │
│ 就业数据集   │     │ R²=0.63 MAE=7.9   │     └─────────────────┘
└─────────────┘     └──────────────────┘              │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ 用户输入     │────▶│ DeepSeek V4 Pro  │────▶│ 自然语言解读      │
│ (Streamlit) │     │ (Prompt Engineering)   │ 职业建议          │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

- **ML 层**: XGBoost 回归 + StandardScaler + OneHotEncoder，log1p 目标变换，GridSearchCV 调参
- **LLM 层**: OpenAI-compatible API 调用 DeepSeek V4 Pro，4 套 Prompt 模板（预测解读、因素分析、职业建议、通用问答）
- **前端层**: Streamlit 5 Tab 交互式界面，matplotlib 可视化，会话状态管理
- **部署**: Streamlit Community Cloud，Secrets 管理 API 密钥

## 📊 模型性能

| 指标 | 数值 |
|------|------|
| R² | 0.63 |
| MAE | 7.90 LPA |
| RMSE | 10.14 LPA |
| MAPE | 12.04% |
| 训练样本 | 6,161（已就业学生） |

## 🔑 关键发现

- **复合技能评分** 是薪资的首要驱动因素（特征重要性 87.6%）
- **Python 技能溢价**: +₹12.09 LPA（拥有者 vs 无此技能者）
- **DSA 技能溢价**: +₹11.60 LPA
- **大学等级**影响较小：Tier 1/2/3 薪资差异 < ₹1.3 LPA
- **六类专业**（CSE/IT/ECE/Mechanical/Civil/EEE）薪资差异在 3% 以内

## 🚀 本地运行

```bash
# 克隆项目
git clone https://github.com/你的用户名/graduate-salary-assistant.git
cd graduate-salary-assistant

# 创建虚拟环境并安装依赖
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 配置 API 密钥（创建 .streamlit/secrets.toml）
echo DEEPSEEK_API_KEY = \"你的密钥\" > .streamlit/secrets.toml

# 训练模型（生成 models/*.pkl）
python -m src.train

# 启动应用
streamlit run app.py
```

## 📁 项目结构

```
├── app.py              # Streamlit 主应用（5 Tab）
├── src/
│   ├── config.py       # 全局配置与常量
│   ├── preprocessing.py # 数据加载与特征工程（18维 → 19维编码后）
│   ├── train.py        # XGBoost 训练管线 + GridSearchCV
│   ├── inference.py    # 推理模块（单条/批量预测）
│   ├── factor_analysis.py # 特征重要性、反事实分析、分组统计
│   ├── llm_client.py   # DeepSeek API 封装（重试、超时、降级）
│   ├── llm_prompts.py  # 4 套 Prompt 模板工程
│   └── ui_components.py # 可复用 Streamlit UI 组件
├── models/             # 训练好的模型文件（pkl + json）
├── data/               # 原始数据集
└── requirements.txt
```

## 🎯 技术亮点

- **混合架构**: ML 模型提供精确数值预测，LLM 负责语义理解和自然语言生成，各取所长
- **优雅降级**: LLM API 不可用时自动回退，仅展示 ML 预测结果，不影响核心功能
- **特征工程**: 筛选 14 个就业前可观测特征，排除 target leakage（公司类型、职位等事后变量）
- **Prompt Engineering**: 针对 4 种场景设计结构化 Prompt，控制输出长度和质量
- **反事实分析**: 逐特征计算边际增益，量化"提升某项技能能涨多少薪资"

---

*Built with Python, XGBoost, DeepSeek V4 Pro, and Streamlit*
