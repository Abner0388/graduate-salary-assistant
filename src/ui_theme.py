"""
CSS theme, color palette, and Chinese font configuration for the app.
"""
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


# ── Color Palette ──────────────────────────────────────
COLORS = {
    "primary": "#1a73e8",
    "primary_light": "#e8f0fe",
    "secondary": "#34a853",
    "secondary_light": "#e6f4ea",
    "accent": "#ea4335",
    "accent_light": "#fce8e6",
    "warning": "#f9ab00",
    "bg": "#f8f9fa",
    "card": "#ffffff",
    "text": "#202124",
    "text_secondary": "#5f6368",
    "border": "#dadce0",
    "salary_high": "#34a853",
    "salary_mid": "#1a73e8",
    "salary_low_mid": "#f9ab00",
    "salary_low": "#ea4335",
}


def apply_custom_theme():
    """Inject custom CSS styles into the Streamlit app."""
    css = f"""
    <style>
    /* ── Global ──────────────────────────── */
    .stApp {{
        background-color: {COLORS['bg']};
    }}

    /* ── Cards ───────────────────────────── */
    .card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .card-title {{
        font-size: 16px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .card-subtitle {{
        font-size: 13px;
        color: {COLORS['text_secondary']};
        margin-bottom: 4px;
    }}

    /* ── Metric Cards ────────────────────── */
    .metric-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    .metric-label {{
        font-size: 12px;
        color: {COLORS['text_secondary']};
        margin-top: 4px;
    }}

    /* ── Chat Bubbles ────────────────────── */
    .chat-bubble {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
    }}

    /* ── Sidebar ─────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['card']};
    }}

    /* ── Buttons ─────────────────────────── */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
    }}
    div[data-testid="stButton"] button[kind="primary"] {{
        background-color: {COLORS['primary']};
    }}

    /* ── Input labels ────────────────────── */
    .stSlider label, .stSelectbox label, .stNumberInput label {{
        font-size: 13px;
        color: {COLORS['text_secondary']};
    }}

    /* ── Tabs ────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }}

    /* ── Context indicator bar ───────────── */
    .context-bar-container {{
        width: 100%;
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
        margin: 4px 0;
    }}
    .context-bar-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Chinese Font Setup ────────────────────────────────

_FONT_SETUP_DONE = False


def get_chinese_font_family() -> str:
    """
    Return the best available Chinese font for the current platform.
    Checks Windows, macOS, and Linux fonts in priority order.
    """
    # Try common Chinese fonts across platforms
    chinese_fonts = [
        "Microsoft YaHei",       # Windows
        "SimHei",                # Windows
        "PingFang SC",           # macOS
        "Heiti SC",              # macOS
        "Noto Sans CJK SC",      # Linux / general
        "WenQuanYi Micro Hei",   # Linux
        "WenQuanYi Zen Hei",     # Linux
        "KaiTi",                 # Windows
        "FangSong",              # Windows
        "SimSun",                # Windows
    ]
    available = set(f.name for f in fm.fontManager.ttflist)
    for font in chinese_fonts:
        if font in available:
            return font
    return "sans-serif"


def setup_matplotlib_chinese():
    """
    Configure matplotlib to support Chinese characters.
    Called once at module import time.
    """
    global _FONT_SETUP_DONE
    if _FONT_SETUP_DONE:
        return
    font_family = get_chinese_font_family()
    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_SETUP_DONE = True


# Auto-setup on import
setup_matplotlib_chinese()
