"""
Centralized configuration for the Graduate Salary Analysis & Prediction Assistant.
"""
import os

# ── Paths ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "student_placement_salary_elite_v2.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "salary_predictor.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ── DeepSeek LLM ───────────────────────────────────────
# ⚠️ Never hardcode API keys. Use environment variable or Streamlit Secrets.
# Streamlit Cloud: set in .streamlit/secrets.toml or via app dashboard
# Local dev: create .streamlit/secrets.toml with DEEPSEEK_API_KEY = "sk-..."
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    ""  # Must be set via env var or st.secrets
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # Change to "deepseek-reasoner" for reasoning tasks
LLM_TIMEOUT = 30  # seconds
LLM_MAX_RETRIES = 3
LLM_TEMPERATURE = 0.7

# ── Dataset ────────────────────────────────────────────
TARGET_COL = "salary_lpa"
ID_COL = "student_id"
PLACED_COL = "placed"

# Features available BEFORE placement (used for prediction)
NUMERICAL_COLS = [
    "cgpa", "coding_score", "communication_score", "aptitude_score",
    "internships", "projects", "backlogs", "resume_score", "skill_score",
]
CATEGORICAL_COLS = ["branch"]
ORDINAL_COLS = ["college_tier"]
BINARY_COLS = ["python_skill", "dsa_skill", "ml_skill", "web_dev_skill"]

# Post-placement columns (excluded from prediction features)
POST_PLACEMENT_COLS = ["company_type", "job_role"]

# All feature columns used for training
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS + ORDINAL_COLS + BINARY_COLS

# ── Branch values ──────────────────────────────────────
BRANCHES = ["CSE", "IT", "ECE", "Mechanical", "Civil", "EEE"]

# ── ML Training ────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# XGBoost hyperparameter search space
XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.9],
    "colsample_bytree": [1.0],
    "reg_alpha": [1],
    "reg_lambda": [1.5],
}

# Display name mapping for UI
FEATURE_DISPLAY_NAMES = {
    "cgpa": "CGPA",
    "branch": "专业 (Branch)",
    "college_tier": "大学等级 (College Tier)",
    "python_skill": "Python 技能",
    "dsa_skill": "数据结构与算法 (DSA)",
    "ml_skill": "机器学习 (ML)",
    "web_dev_skill": "Web 开发",
    "coding_score": "编程能力评分",
    "communication_score": "沟通能力评分",
    "aptitude_score": "能力倾向评分",
    "internships": "实习次数",
    "projects": "项目数量",
    "backlogs": "挂科数量",
    "resume_score": "简历评分",
    "skill_score": "综合技能评分",
}
