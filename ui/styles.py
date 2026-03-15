"""Centralized Streamlit styling for the workbench UI."""

from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    """Apply scoped styling without changing application behavior."""
    st.markdown(
        """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

div[data-testid="stMultiSelect"] {
    max-width: 680px;
}

div[data-testid="stSelectbox"] {
    max-width: 330px;
}

.guide-card {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 10px;
    padding: 1rem;
    min-height: 144px;
}

.author-card {
    border-left: 4px solid #888;
    padding: 0.65rem 1rem;
    background: rgba(128, 128, 128, 0.07);
    border-radius: 4px;
}

.small-muted {
    opacity: 0.75;
    font-size: 0.9rem;
}

/* Make the main benchmark action more prominent without affecting downloads. */
div[data-testid="stButton"] > button[kind="primary"] {
    min-height: 3.45rem;
    padding: 0.75rem 1.5rem;
    font-size: 1.12rem;
    font-weight: 700;
    border-radius: 0.65rem;
}

/* Compact result cards replace oversized native metric values. */
.summary-card {
    border: 1px solid rgba(128, 128, 128, 0.24);
    border-radius: 0.8rem;
    padding: 0.95rem 1rem;
    min-height: 9.2rem;
    background: rgba(128, 128, 128, 0.045);
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 0.3rem;
}

.summary-card-label {
    font-size: 0.82rem;
    line-height: 1.2;
    font-weight: 650;
    opacity: 0.78;
}

.summary-card-primary {
    margin-top: 0.15rem;
    font-size: clamp(1.35rem, 1.75vw, 1.62rem);
    line-height: 1.16;
    font-weight: 680;
    overflow-wrap: anywhere;
}

.summary-card-secondary {
    min-height: 1.1rem;
    font-size: 0.88rem;
    line-height: 1.25;
    font-weight: 500;
    opacity: 0.72;
}

.summary-card-badge {
    align-self: flex-start;
    margin-top: auto;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.82rem;
    line-height: 1.25;
    font-weight: 700;
    background: rgba(128, 128, 128, 0.14);
}

.summary-card-badge.positive {
    background: rgba(34, 197, 94, 0.14);
}

/* Larger, bolder result tabs with full labels retained. */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.4rem;
    overflow-x: auto;
    scrollbar-width: thin;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    min-height: 3.75rem;
    padding: 1rem 1.3rem;
    white-space: nowrap;
}

div[data-testid="stTabs"] button[data-baseweb="tab"],
div[data-testid="stTabs"] button[data-baseweb="tab"] p,
div[data-testid="stTabs"] button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] {
    font-size: 1.2rem !important;
    line-height: 1.2 !important;
    font-weight: 800 !important;
    letter-spacing: 0.005em;
}

div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {
    font-weight: 900 !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    height: 5px;
    border-radius: 5px 5px 0 0;
}

@media (max-width: 900px) {
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        padding: 0.95rem 1.05rem;
    }

    .summary-card {
        min-height: 8.4rem;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )
