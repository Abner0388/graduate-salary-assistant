"""
Retrieval functions for the RAG module.
Given a student profile or natural-language query, retrieves:
- Top-k similar student profiles from the corpus
- Relevant aggregate statistics
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..config import RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, RAG_CONTEXT_MAX_CHARS
from .corpus import query_to_text


def retrieve_similar_profiles(
    student: dict,
    corpus: dict,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """
    Find the top-k most similar student profiles in the corpus.

    Args:
        student: Student profile dict (same keys as FEATURE_COLS).
        corpus: The corpus dict from load_corpus().
        top_k: Number of similar profiles to return.

    Returns:
        List of dicts with keys: profile (student dict), salary, similarity, text
    """
    vectorizer = corpus["vectorizer"]
    tfidf_matrix = corpus["tfidf_matrix"]
    df = corpus["df"]

    # Convert query to text and vectorize
    query_text = query_to_text(student)
    query_vec = vectorizer.transform([query_text])

    # Cosine similarity against all documents
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim < RAG_SIMILARITY_THRESHOLD:
            continue

        row = df.iloc[idx]
        results.append({
            "profile": {col: row[col] for col in df.columns if col in (
                "branch", "cgpa", "college_tier", "python_skill", "dsa_skill",
                "ml_skill", "web_dev_skill", "coding_score", "communication_score",
                "aptitude_score", "internships", "projects", "backlogs",
                "resume_score", "skill_score", "salary_lpa",
            )},
            "salary": float(row["salary_lpa"]),
            "similarity": round(sim, 4),
            "text": corpus["texts"][idx],
        })

    return results


def retrieve_aggregate_stats(
    query_type: str,
    corpus: dict,
    params: dict | None = None,
) -> dict | str:
    """
    Retrieve pre-computed aggregate statistics matching the query.

    Args:
        query_type: One of "by_branch", "by_tier", "by_skill", "by_cgpa",
                    "salary_distribution", "all"
        corpus: The corpus dict from load_corpus().
        params: Optional filter params, e.g. {"branch": "CSE"} or {"skill": "python_skill"}

    Returns:
        Aggregated data dict or formatted string.
    """
    aggregates = corpus["aggregates"]

    if query_type == "by_branch":
        data = aggregates.get("by_branch", {})
        if params and "branch" in params:
            return data.get(params["branch"], {})
        return data

    elif query_type == "by_tier":
        return aggregates.get("by_tier", {})

    elif query_type == "by_skill":
        data = aggregates.get("skill", {})
        if params and "skill" in params:
            return data.get(params["skill"], {})
        return aggregates

    elif query_type == "salary_distribution":
        return {
            f"p{p}": aggregates.get(f"salary_p{p}", 0)
            for p in [10, 25, 50, 75, 90, 95]
        }

    elif query_type == "by_cgpa":
        return aggregates.get("by_cgpa_bucket", {})

    elif query_type == "all":
        return aggregates

    return {}


def format_retrieved_context(
    similar_profiles: list[dict],
    corpus: dict,
    max_chars: int = RAG_CONTEXT_MAX_CHARS,
) -> str:
    """
    Format retrieved profiles and aggregate stats into a compact
    Chinese text block for LLM prompt injection.

    Args:
        similar_profiles: Results from retrieve_similar_profiles().
        corpus: The corpus dict from load_corpus().
        max_chars: Maximum character length for the returned context.

    Returns:
        Formatted Chinese text string, or empty string if no data.
    """
    if not similar_profiles:
        return ""

    lines = ["[参考数据 — 数据集中的相似学生]\n"]

    char_count = len(lines[0])

    for i, sp in enumerate(similar_profiles[:5]):
        p = sp["profile"]
        skill_str = "、".join([
            sk for sk, key in [
                ("Python", "python_skill"),
                ("DSA", "dsa_skill"),
                ("ML", "ml_skill"),
                ("WebDev", "web_dev_skill"),
            ]
            if p.get(key, 0) == 1
        ]) or "无特殊技能"

        entry = (
            f"- 学生{i+1}: {p['branch']}专业, CGPA {p['cgpa']:.1f}, "
            f"实习{p.get('internships',0):.0f}次, 项目{p.get('projects',0):.0f}个, "
            f"技能:[{skill_str}], "
            f"实际薪资: {sp['salary']:.1f} LPA "
            f"(相似度{sp['similarity']:.0%})\n"
        )
        if char_count + len(entry) > max_chars:
            break
        lines.append(entry)
        char_count += len(entry)

    # Add aggregate stats if they fit
    agg = corpus["aggregates"]
    if agg:
        salary_p50 = agg.get("salary_p50", 0)
        salary_p75 = agg.get("salary_p75", 0)
        salary_p90 = agg.get("salary_p90", 0)
        stat_line = (
            f"\n[数据集基准] 薪资中位数: {salary_p50} LPA, "
            f"75分位: {salary_p75} LPA, "
            f"90分位: {salary_p90} LPA\n"
        )
        if char_count + len(stat_line) <= max_chars:
            lines.append(stat_line)

    return "".join(lines)
