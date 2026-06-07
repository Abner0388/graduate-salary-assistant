"""
Cross-tab session memory for user profile persistence.
Typed wrapper around st.session_state to avoid scattered key access.
"""
import streamlit as st

# Namespaced session-state key names
_KEY_PROFILE = "sm_profile"
_KEY_PREDICTION = "sm_prediction"
_KEY_COMPARISON_A = "sm_cmp_a"
_KEY_COMPARISON_B = "sm_cmp_b"


class SessionMemory:
    """
    Typed session-state wrapper that persists a user's student profile
    and prediction result across tabs within the same browser session.

    Usage:
        # After prediction (Tab 1):
        SessionMemory.save_profile(student)

        # In another tab (Tab 4, Tab 5):
        profile = SessionMemory.get_profile()
        if profile:
            # pre-fill form or inject into chat context
    """

    # ── Profile ─────────────────────────────────────

    @staticmethod
    def save_profile(student: dict) -> None:
        """Save a student profile dict to session state."""
        st.session_state[_KEY_PROFILE] = dict(student)

    @staticmethod
    def get_profile() -> dict | None:
        """Retrieve the saved student profile, or None."""
        return st.session_state.get(_KEY_PROFILE)

    @staticmethod
    def has_profile() -> bool:
        """Check whether a profile has been saved."""
        return _KEY_PROFILE in st.session_state and st.session_state[_KEY_PROFILE] is not None

    @staticmethod
    def clear_profile() -> None:
        """Remove the saved profile."""
        st.session_state.pop(_KEY_PROFILE, None)

    # ── Prediction ──────────────────────────────────

    @staticmethod
    def save_prediction(prediction: dict) -> None:
        """Save the most recent prediction result."""
        st.session_state[_KEY_PREDICTION] = prediction

    @staticmethod
    def get_prediction() -> dict | None:
        """Retrieve the saved prediction, or None."""
        return st.session_state.get(_KEY_PREDICTION)

    # ── Comparison pair ─────────────────────────────

    @staticmethod
    def save_comparison(student_a: dict, student_b: dict) -> None:
        """Save the two students being compared (Tab 3)."""
        st.session_state[_KEY_COMPARISON_A] = dict(student_a)
        st.session_state[_KEY_COMPARISON_B] = dict(student_b)

    @staticmethod
    def get_comparison() -> tuple[dict | None, dict | None]:
        """Retrieve the saved comparison pair."""
        return (
            st.session_state.get(_KEY_COMPARISON_A),
            st.session_state.get(_KEY_COMPARISON_B),
        )

    # ── Profile-as-text for LLM injection ───────────

    @staticmethod
    def profile_to_context(student: dict | None) -> str:
        """
        Format a student profile as a compact context string for LLM prompts.
        Returns empty string if profile is None.
        """
        if not student:
            return ""
        lines = []
        # Display-friendly field ordering
        order = [
            ("branch", "专业"),
            ("cgpa", "CGPA"),
            ("college_tier", "大学等级"),
            ("python_skill", "Python"),
            ("dsa_skill", "DSA"),
            ("ml_skill", "ML"),
            ("web_dev_skill", "WebDev"),
            ("coding_score", "编程评分"),
            ("communication_score", "沟通评分"),
            ("aptitude_score", "能力倾向"),
            ("internships", "实习次数"),
            ("projects", "项目数量"),
            ("backlogs", "挂科数"),
            ("resume_score", "简历评分"),
            ("skill_score", "综合技能"),
        ]
        for key, label in order:
            if key in student:
                val = student[key]
                if key in ("python_skill", "dsa_skill", "ml_skill", "web_dev_skill"):
                    val = "是" if val else "否"
                lines.append(f"{label}: {val}")
        return "当前学生信息:\n" + "\n".join(lines)
