"""
Phase 2: ELO System with Scaling
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

BASE_ELO = 1200
K_BASE = 32

# League strength based on international performance
LEAGUE_STRENGTH = {
    'LCK': 1.00,
    'LPL': 0.92,
    'LEC': 0.85,
    'LCS': 0.69,
    'CBLOL': 0.54,
    'LCP': 0.46,
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_expected(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k_factor):
    return rating + k_factor * (actual - expected)

def main():
    print("=" * 70)
    print("PHASE 2: ELO SYSTEM (FINAL)")
    print("=" * 70)

    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    print("\n📂 STEP 1: Loading prepared data...")
    data_path = os.path.join(PROCESSED_DIR, '2026_data_prepared.csv')
    if not os.path.exists(data_path):
        print(f"   ❌ File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"   ✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # ==========================================
    # STEP 2: SORT CHRONOLOGICALLY
    # ==========================================
    print("\n🔄 STEP 2: Sorting chronologically...")
    df_sorted = df.sort_values('date_parsed').reset_index(drop=True)
    print(f"   ✅ Sorted {len(df_sorted):,} matches")

    # ==========================================
    # STEP 3: INITIALIZE ELO
    # ==========================================
    print("\n🏁 STEP 3: Initializing ELO ratings...")
    teams = df['teamname'].unique()
    elo = {team: BASE_ELO for team in teams}
    print(f"   ✅ Initialized {len(teams)} teams with {BASE_ELO} ELO")

    # ==========================================
    # STEP 4: PROCESS MATCHES
    # ==========================================
    print("\n🔄 STEP 4: Processing matches chronologically...")
    total_matches = len(df_sorted)
    international_matches = df_sorted[df_sorted['match_type'] == 'international'].shape[0]

    print(f"   Total matches: {total_matches:,}")
    print(f"   International matches: {international_matches:,}")
    print()

    for idx, match in df_sorted.iterrows():
        team = match['teamname']
        opponent_row = df_sorted[(df_sorted['gameid'] == match['gameid']) & (df_sorted['teamname'] != team)].iloc[0]
        opponent = opponent_row['teamname']

        team_elo = elo[team]
        opp_elo = elo[opponent]

        expected = calculate_expected(team_elo, opp_elo)
        actual = match['result']

        # Apply league strength to domestic matches
        if match['match_type'] == 'international':
            k = K_BASE * 1.5
        else:
            team_strength = LEAGUE_STRENGTH.get(match['league'], 0.80)
            opp_strength = LEAGUE_STRENGTH.get(opponent_row['league'], 0.80)
            avg_strength = (team_strength + opp_strength) / 2
            k = K_BASE * avg_strength

        new_elo = update_elo(team_elo, expected, actual, k)
        elo[team] = new_elo

        opp_new_elo = update_elo(opp_elo, expected, 1 - actual, k)
        elo[opponent] = opp_new_elo

    print(f"\n   ✅ Processed {total_matches:,} matches")

    # ==========================================
    # STEP 5: CREATE RATINGS
    # ==========================================
    print("\n📊 STEP 5: Creating final ratings...")
    final_ratings = []
    for team in teams:
        final_ratings.append({
            'teamname': team,
            'league': df[df['teamname'] == team]['league'].iloc[0],
            'elo': elo[team],
        })

    ratings_df = pd.DataFrame(final_ratings)

    # ==========================================
    # STEP 6: SCALE TO OFFICIAL RANGE
    # ==========================================
    print("\n📊 STEP 6: Scaling to official range (1000-1600)...")

    min_elo = ratings_df['elo'].min()
    max_elo = ratings_df['elo'].max()

    # Scale to 1000-1600 range
    ratings_df['scaled_elo'] = 1000 + (ratings_df['elo'] - min_elo) / (max_elo - min_elo) * 600

    # Sort by scaled ELO
    ratings_df = ratings_df.sort_values('scaled_elo', ascending=False).reset_index(drop=True)

    print(f"\n   🏆 Top 20 Teams by Scaled ELO:")
    print("   " + "-" * 55)
    for i, row in ratings_df.head(20).iterrows():
        print(f"   {i+1:2d}. {row['teamname']:30s} ({row['league']:4s}) | {row['scaled_elo']:.0f}")

    # ==========================================
    # STEP 7: SAVE
    # ==========================================
    print("\n💾 STEP 7: Saving results...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, '2026_elo_ratings.csv')
    ratings_df.to_csv(output_path, index=False)
    print(f"   ✅ Saved: {output_path}")

    print("\n" + "=" * 70)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
