# apmt_dashboard/pages/goat_offtake.py
import streamlit as st
from pages.sheep_offtake import render_offtake_analysis


def render_goat_offtake(processor):
    """
    Render the Goat Offtake dashboard page.

    This is a thin wrapper around the generic `render_offtake_analysis`
    so that goats get their own menu entry while reusing the same logic
    as sheep_of_ftake.
    """
    return render_offtake_analysis(processor, 'goats')
