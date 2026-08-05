"""
Configuration for the LoL Worlds Predictor
"""

import os
from datetime import datetime

# ==========================================
# FILE PATHS
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATHS = {
    'raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'results': os.path.join(PROJECT_ROOT, 'results'),
}

# ==========================================
# DATA SOURCE CONFIG
# ==========================================
# The raw data file should be in data/raw/
RAW_FILE_PATTERN = '2026_LoL_esports_match_data_from_OraclesElixir.csv'

# ==========================================
# MAJOR LEAGUES (Worlds 2026)
# ==========================================
MAJOR_LEAGUES = ['LPL', 'LCK', 'LEC', 'LCS', 'CBLOL', 'LCP']

# ==========================================
# INTERNATIONAL EVENTS
# ==========================================
INTERNATIONAL_LEAGUES = ['FST', 'EWC', 'MSI', 'Worlds']
# ==========================================
# ELO CONFIG
# ==========================================
ELO_CONFIG = {
    'base_rating': 1200,
    'k_factor_regional': 32,
    'k_factor_international': 48,
    'k_factor_worlds': 64,  # Higher stakes for Worlds
}

# ==========================================
# FEATURES
# ==========================================
PRE_TOURNAMENT_FEATURES = [
    'team_elo',            # From ELO system
    'team_form_5',         # Win % in last 5 games
    'team_form_10',        # Win % in last 10 games
    'team_h2h',            # Head-to-head win rate
    'league_strength',     # Region strength factor
]
# ==========================================
# LOGGING
# ==========================================
def get_update_log():
    """Get the path to the update log file"""
    return os.path.join(DATA_PATHS['processed'], 'update_log.txt')

def log_update(status, details=""):
    """Log when data was last updated"""
    log_path = get_update_log()
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {status} | {details}\n")
