# apmt_dashboard/config.py
from pathlib import Path
import os

# Data path
DATA_PATH = str((Path(__file__).resolve().parent / "APMT_Longitudinal_Survey.csv"))

# App configuration
APP_TITLE = "APMT Panel Data Dashboard"
APP_ICON = "🐑"

# Chart colors
COLORS = {
    'kpmd': '#1f77b4',
    'non_kpmd': '#ff6b6b',
    'treatment': '#2ca02c',
    'control': '#d62728'
}

# Panel data configuration
PANEL_CONFIG = {
    'hhid_columns': ['Household ID', 'household_id', 'HHID', '_id', '_uuid', 'respondent_id'],
    'date_columns': ['int_date', '_submission_time', 'start', 'end'],
    'wave_columns': ['wave', 'round', 'survey_round', 'visit', 'time_period']
}