"""
Phase 1: Data Preparation (2026)
- Load raw Oracle's Elixir data for 2026 season
- Tag international events (FST, EWC, MSI) on ALL data first
- Filter to major leagues + keep international events
- Aggregate from player-level to team-level
- Clean missing values
- Sort chronologically
- Save prepared data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATHS = {
    'raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'results': os.path.join(PROJECT_ROOT, 'results'),
}

MAJOR_LEAGUES = ['LPL', 'LCK', 'LEC', 'LCS', 'CBLOL', 'LCP']
INTERNATIONAL_LEAGUES = ['FST', 'EWC', 'MSI']

def find_raw_file():
    """Find the raw data file in data/raw/"""
    raw_dir = DATA_PATHS['raw']
    if not os.path.exists(raw_dir):
        return None
    files = os.listdir(raw_dir)
    for file in files:
        if 'LoL_esports_match_data' in file and file.endswith('.csv'):
            return os.path.join(raw_dir, file)
    return None

def main():
    print("=" * 70)
    print("PHASE 1: DATA PREPARATION (2026)")
    print("=" * 70)

    # ==========================================
    # STEP 1: FIND AND LOAD RAW DATA
    # ==========================================
    print("\n📂 STEP 1: Finding raw data...")
    raw_path = find_raw_file()
    if raw_path is None:
        print("   ❌ No Oracle's Elixir data found in data/raw/")
        print("   Please place the CSV file in data/raw/")
        return

    print(f"   ✅ Found: {os.path.basename(raw_path)}")

    df = pd.read_csv(raw_path, low_memory=False)
    print(f"   ✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # ==========================================
    # STEP 2: TAG INTERNATIONAL EVENTS (ON ALL DATA)
    # ==========================================
    print("\n🌍 STEP 2: Tagging international events...")

    def tag_match_type(league):
        return 'international' if league in INTERNATIONAL_LEAGUES else 'regional'
    df['match_type'] = df['league'].apply(tag_match_type)

    intl_count = df[df['match_type'] == 'international'].shape[0]
    reg_count = df[df['match_type'] == 'regional'].shape[0]

    print(f"   Total Regional: {reg_count:,}")
    print(f"   Total International: {intl_count:,}")

    if intl_count > 0:
        intl_leagues = df[df['match_type'] == 'international']['league'].unique()
        print(f"   International leagues found: {intl_leagues}")

    # ==========================================
    # STEP 3: FILTER TO MAJOR LEAGUES + KEEP INTERNATIONAL
    # ==========================================
    print("\n🎯 STEP 3: Filtering to major leagues + keeping international events...")

    # Keep rows that are either in major leagues OR are international events
    df = df[df['league'].isin(MAJOR_LEAGUES) | df['match_type'].isin(['international'])]
    print(f"   ✅ Kept {df.shape[0]:,} rows")

    # Show league distribution
    print("\n   League distribution:")
    for league, count in df['league'].value_counts().items():
        match_type = df[df['league'] == league]['match_type'].iloc[0]
        print(f"      {league}: {count:,} rows ({match_type})")

    # ==========================================
    # STEP 4: AGGREGATE TO TEAM LEVEL
    # ==========================================
    print("\n🏗️ STEP 4: Aggregating to team level...")

    agg_dict = {
        'side': 'first',
        'result': 'first',
        'league': 'first',
        'match_type': 'first',
        'firstdragon': 'first',
        'firstbaron': 'first',
        'firsttower': 'first',
        'gamelength': 'first',
        'date': 'first',
        'kills': 'sum',
        'deaths': 'sum',
        'assists': 'sum',
        'teamkills': 'first',
        'teamdeaths': 'first',
        'totalgold': 'sum',
        'earnedgold': 'sum',
        'goldspent': 'sum',
        'damagetochampions': 'sum',
        'wardsplaced': 'sum',
        'wardskilled': 'sum',
        'controlwardsbought': 'sum',
        'visionscore': 'sum',
        'dragons': 'first',
        'barons': 'first',
        'towers': 'first',
        'inhibitors': 'first',
        'golddiffat10': 'first',
        'xpdiffat10': 'first',
        'csdiffat10': 'first',
        'golddiffat15': 'first',
        'xpdiffat15': 'first',
        'csdiffat15': 'first',
    }

    team_df = df.groupby(['gameid', 'teamname']).agg(agg_dict).reset_index()
    print(f"   ✅ Aggregated to {team_df.shape[0]:,} rows (team level)")

    # ==========================================
    # STEP 5: HANDLE MISSING VALUES
    # ==========================================
    print("\n🧹 STEP 5: Handling missing values...")

    # Fill missing objectives with 0 (meaning "not taken")
    for col in ['firstdragon', 'firstbaron', 'firsttower']:
        if col in team_df.columns and team_df[col].isnull().sum() > 0:
            before = team_df[col].isnull().sum()
            team_df[col] = team_df[col].fillna(0)
            print(f"   ✅ {col}: filled {before:,} missing values with 0")

    # ==========================================
    # STEP 6: PROCESS DATES
    # ==========================================
    print("\n📅 STEP 6: Processing date column...")

    if 'date' in team_df.columns:
        team_df['date_parsed'] = pd.to_datetime(team_df['date'], errors='coerce')
        valid_dates = team_df['date_parsed'].notna().sum()
        print(f"   ✅ {valid_dates:,} rows have valid dates")

        if valid_dates < len(team_df):
            print(f"   ⚠️ {len(team_df) - valid_dates:,} rows missing dates")
            print("   Using gameid as chronological proxy for missing dates")
    else:
        print("   ⚠️ No date column found, using gameid as proxy")

    # ==========================================
    # STEP 7: SORT CHRONOLOGICALLY
    # ==========================================
    print("\n🔄 STEP 7: Sorting chronologically...")

    if 'date_parsed' in team_df.columns and team_df['date_parsed'].notna().any():
        team_df = team_df.sort_values('date_parsed').reset_index(drop=True)
        print("   ✅ Sorted by date")
    else:
        team_df = team_df.sort_values('gameid').reset_index(drop=True)
        print("   ✅ Sorted by gameid (fallback)")

    # ==========================================
    # STEP 8: DROP HIGH-MISSING COLUMNS
    # ==========================================
    print("\n🗑️ STEP 8: Dropping columns with >80% missing values...")

    missing_pct = team_df.isnull().sum() / len(team_df) * 100
    high_missing_cols = missing_pct[missing_pct > 80].index.tolist()

    if high_missing_cols:
        print(f"   Dropping {len(high_missing_cols)} columns")
        team_df = team_df.drop(columns=high_missing_cols)
        print(f"   ✅ Remaining columns: {team_df.shape[1]}")
    else:
        print("   ✅ No columns with >80% missing values")

    # ==========================================
    # STEP 9: SAVE
    # ==========================================
    print("\n💾 STEP 9: Saving prepared data...")

    os.makedirs(DATA_PATHS['processed'], exist_ok=True)

    output_path = os.path.join(DATA_PATHS['processed'], '2026_data_prepared.csv')
    team_df.to_csv(output_path, index=False)
    print(f"   ✅ Saved to: {output_path}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("✅ PHASE 1 COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final dataset:")
    print(f"   Rows: {team_df.shape[0]:,}")
    print(f"   Columns: {team_df.shape[1]}")
    print(f"   Teams: {team_df['teamname'].nunique()}")
    print(f"   Leagues: {team_df['league'].nunique()}")

    if 'date_parsed' in team_df.columns:
        min_date = team_df['date_parsed'].min()
        max_date = team_df['date_parsed'].max()
        print(f"   Date range: {min_date} to {max_date}")

    # Show match type breakdown
    match_type_counts = team_df['match_type'].value_counts()
    print(f"\n   Match type breakdown:")
    for match_type, count in match_type_counts.items():
        print(f"      {match_type}: {count:,} rows")

if __name__ == "__main__":
    main()
