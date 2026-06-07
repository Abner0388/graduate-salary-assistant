"""
TF-IDF corpus builder for the RAG module.
Converts structured student records into searchable text documents,
pre-computes aggregate statistics for fast retrieval.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..preprocessing import load_data
from ..config import (
    DATA_PATH, FEATURE_COLS, BRANCHES,
    BINARY_COLS, NUMERICAL_COLS, ORDINAL_COLS,
)


def student_to_text(row: pd.Series) -> str:
    """
    Convert a student row into a Chinese-readable text document.
    Used as the unit of TF-IDF indexing.
    """
    parts = []

    # Branch
    branch = row.get("branch", "")
    parts.append(f"专业:{branch}")

    # College tier (lower = better)
    tier = row.get("college_tier", "")
    parts.append(f"大学等级:{tier}级")

    # CGPA
    cgpa = row.get("cgpa", "")
    parts.append(f"CGPA:{cgpa:.1f}")

    # Skills (binary: 有/无)
    for col, label in [
        ("python_skill", "Python"),
        ("dsa_skill", "DSA"),
        ("ml_skill", "机器学习"),
        ("web_dev_skill", "Web开发"),
    ]:
        val = int(row.get(col, 0))
        parts.append(f"{label}:{'是' if val else '否'}")

    # Scores
    coding = row.get("coding_score", "")
    parts.append(f"编程评分:{coding:.0f}")
    communication = row.get("communication_score", "")
    parts.append(f"沟通评分:{communication:.1f}")
    aptitude = row.get("aptitude_score", "")
    parts.append(f"能力倾向评分:{aptitude:.0f}")

    # Experience
    internships = int(row.get("internships", 0))
    parts.append(f"实习次数:{internships}")
    projects = int(row.get("projects", 0))
    parts.append(f"项目数量:{projects}")
    backlogs = int(row.get("backlogs", 0))
    parts.append(f"挂科数:{backlogs}")

    # Scores
    resume = row.get("resume_score", "")
    parts.append(f"简历评分:{resume:.0f}")
    skill_score = int(row.get("skill_score", 0))
    parts.append(f"综合技能评分:{skill_score}")

    # Salary (for similarity — we want similar students to have similar outcomes)
    salary = row.get("salary_lpa", "")
    if salary:
        parts.append(f"薪资:{salary:.0f}LPA")

    return ", ".join(parts)


def compute_aggregates(df: pd.DataFrame) -> dict:
    """
    Pre-compute aggregate statistics for fast retrieval:
    - Salary distribution by branch, tier, skill buckets
    - Salary percentiles overall
    - Average profiles for high/medium/low salary groups
    """
    placed = df[df["placed"] == 1].copy()

    aggregates = {}

    # By branch
    branch_stats = placed.groupby("branch")["salary_lpa"].agg([
        "mean", "median", "std", "count"
    ]).round(2)
    aggregates["by_branch"] = {
        b: {
            "mean": row["mean"], "median": row["median"],
            "std": row["std"], "count": int(row["count"])
        }
        for b, row in branch_stats.iterrows()
    }

    # By college tier
    tier_stats = placed.groupby("college_tier")["salary_lpa"].agg([
        "mean", "median", "std", "count"
    ]).round(2)
    aggregates["by_tier"] = {
        int(t): {
            "mean": row["mean"], "median": row["median"],
            "std": row["std"], "count": int(row["count"])
        }
        for t, row in tier_stats.iterrows()
    }

    # By skill (binary)
    for col in BINARY_COLS:
        has_skill = placed[placed[col] == 1]["salary_lpa"]
        no_skill = placed[placed[col] == 0]["salary_lpa"]
        aggregates[f"skill_{col}"] = {
            "has_skill_mean": round(float(has_skill.mean()), 2),
            "no_skill_mean": round(float(no_skill.mean()), 2),
            "premium": round(float(has_skill.mean() - no_skill.mean()), 2),
            "has_skill_count": len(has_skill),
            "no_skill_count": len(no_skill),
        }

    # Salary percentiles overall
    for pct in [10, 25, 50, 75, 90, 95]:
        aggregates[f"salary_p{pct}"] = round(
            float(np.percentile(placed["salary_lpa"], pct)), 2
        )

    # CGPA buckets
    placed["cgpa_bucket"] = pd.cut(
        placed["cgpa"], bins=[0, 6.5, 7.5, 8.5, 10.0],
        labels=["低(≤6.5)", "中(6.5-7.5)", "良(7.5-8.5)", "优(>8.5)"]
    )
    cgpa_stats = placed.groupby("cgpa_bucket", observed=False)["salary_lpa"].agg([
        "mean", "count"
    ]).round(2)
    aggregates["by_cgpa_bucket"] = {
        str(b): {"mean": row["mean"], "count": int(row["count"])}
        for b, row in cgpa_stats.iterrows()
    }

    return aggregates


def load_corpus(df: pd.DataFrame | None = None):
    """
    Build the TF-IDF corpus from the CSV dataset.

    Args:
        df: Optional pre-loaded DataFrame. If None, loads from DATA_PATH.

    Returns:
        dict with keys: df, vectorizer, tfidf_matrix, texts, feature_cols, aggregates
    """
    if df is None:
        df = load_data(DATA_PATH)

    # Only index placed students (they have salary data)
    placed = df[df["placed"] == 1].copy()
    # Reset index so we can map back
    placed = placed.reset_index(drop=True)

    # Convert each row to text
    texts = placed.apply(student_to_text, axis=1).tolist()

    # Build TF-IDF vectorizer with Chinese-friendly settings
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),       # unigrams + bigrams
        max_features=1000,         # reasonable vocabulary size
        sublinear_tf=True,         # 1 + log(tf)
        max_df=0.95,              # ignore very frequent terms
        min_df=2,                  # ignore rare terms
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Pre-compute aggregates
    aggregates = compute_aggregates(df)

    return {
        "df": placed,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "texts": texts,
        "aggregates": aggregates,
    }


def query_to_text(student: dict) -> str:
    """
    Convert a student profile dict to text for similarity query.
    Mirrors student_to_text format but excludes salary.
    """
    parts = []
    branch = student.get("branch", "")
    parts.append(f"专业:{branch}")
    tier = student.get("college_tier", "")
    parts.append(f"大学等级:{tier}级")
    cgpa = student.get("cgpa", 0)
    parts.append(f"CGPA:{cgpa:.1f}")

    for key, label in [
        ("python_skill", "Python"),
        ("dsa_skill", "DSA"),
        ("ml_skill", "机器学习"),
        ("web_dev_skill", "Web开发"),
    ]:
        val = int(student.get(key, 0))
        parts.append(f"{label}:{'是' if val else '否'}")

    coding = student.get("coding_score", 0)
    parts.append(f"编程评分:{coding:.0f}")
    communication = student.get("communication_score", 0)
    parts.append(f"沟通评分:{communication:.1f}")
    aptitude = student.get("aptitude_score", 0)
    parts.append(f"能力倾向评分:{aptitude:.0f}")

    internships = int(student.get("internships", 0))
    parts.append(f"实习次数:{internships}")
    projects = int(student.get("projects", 0))
    parts.append(f"项目数量:{projects}")
    backlogs = int(student.get("backlogs", 0))
    parts.append(f"挂科数:{backlogs}")

    resume = student.get("resume_score", 0)
    parts.append(f"简历评分:{resume:.0f}")
    skill_score = int(student.get("skill_score", 0))
    parts.append(f"综合技能评分:{skill_score}")

    return ", ".join(parts)
