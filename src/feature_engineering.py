"""
Phase 3: Feature Engineering
- Build pre-game features for the ML model
- Calculate form, streaks, head-to-head, and more
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def load_data():
    """Load prepared data and ELO ratings."""
    # Load team-level match data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, '2026_data_prepared.csv'))

    # Load ELO ratings
    elo_df = pd.read_csv(os.path.join(PROCESSED_DIR, '2026_elo_ratings.csv'))

    # Create ELO lookup dictionary
    elo_dict = dict(zip(elo_df['teamname'], elo_df['scaled_elo']))

    return df, elo_dict


def calculate_form(team_df, n_games=5):
    """Calculate win rate in last n games."""
    if len(team_df) == 0:
        return 0.5

    last_n = team_df.tail(n_games)
    if len(last_n) == 0:
        return 0.5

    return last_n['result'].mean()


def calculate_streak(team_df):
    """Calculate current win/loss streak."""
    if len(team_df) == 0:
        return 0

    results = team_df['result'].values

    # Get the last result
    if len(results) == 0:
        return 0

    last_result = results[-1]
    streak = 0

    # Count consecutive same results
    for result in reversed(results):
        if result == last_result:
            streak += 1
        else:
            break

    # Make negative for losses
    if last_result == 0:
        streak = -streak

    return streak


def calculate_h2h(df, team1, team2):
    """Calculate head-to-head win rate for team1 vs team2."""
    # Find all matches between these teams
    matches = df[
        ((df['teamname'] == team1) & (df['opponent'] == team2)) |
        ((df['teamname'] == team2) & (df['opponent'] == team1))
        ]

    if len(matches) == 0:
        return 0.5  # No history, assume 50%

    # Get wins for team1
    team1_matches = matches[matches['teamname'] == team1]
    if len(team1_matches) == 0:
        return 0.5

    return team1_matches['result'].mean()


def build_features(df, elo_dict):
    """
    Build all features for each match.
    """
    features = []

    # Get all unique game IDs
    game_ids = df['gameid'].unique()

    print("Building features for", len(game_ids), "games...")

    for game_id in game_ids:
        # Get both teams for this game
        game_df = df[df['gameid'] == game_id]

        if len(game_df) != 2:
            continue

        team1_row = game_df.iloc[0]
        team2_row = game_df.iloc[1]

        team1 = team1_row['teamname']
        team2 = team2_row['teamname']

        # Get all previous matches for each team (before this game)
        prev_matches_team1 = df[
            (df['teamname'] == team1) &
            (df['date_parsed'] < team1_row['date_parsed'])
            ]
        prev_matches_team2 = df[
            (df['teamname'] == team2) &
            (df['date_parsed'] < team2_row['date_parsed'])
            ]

        # Calculate features
        team1_elo = elo_dict.get(team1, 1200)
        team2_elo = elo_dict.get(team2, 1200)
        elo_diff = team1_elo - team2_elo

        team1_form_5 = calculate_form(prev_matches_team1, 5)
        team2_form_5 = calculate_form(prev_matches_team2, 5)
        team1_form_10 = calculate_form(prev_matches_team1, 10)
        team2_form_10 = calculate_form(prev_matches_team2, 10)

        team1_streak = calculate_streak(prev_matches_team1)
        team2_streak = calculate_streak(prev_matches_team2)

        h2h = calculate_h2h(df, team1, team2)

        # League strength (from ELO data)
        team1_league = team1_row['league']
        team2_league = team2_row['league']

        # Build feature row
        features.append({
            'gameid': game_id,
            'team1': team1,
            'team2': team2,
            'result': team1_row['result'],  # 1 if team1 wins
            'team1_elo': team1_elo,
            'team2_elo': team2_elo,
            'elo_diff': elo_diff,
            'team1_form_5': team1_form_5,
            'team2_form_5': team2_form_5,
            'team1_form_10': team1_form_10,
            'team2_form_10': team2_form_10,
            'team1_streak': team1_streak,
            'team2_streak': team2_streak,
            'h2h_win_rate': h2h,
            'team1_league': team1_league,
            'team2_league': team2_league,
            'date': team1_row['date_parsed'],
        })

    return pd.DataFrame(features)


def main():
    print("=" * 70)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 70)

    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    print("\n📂 STEP 1: Loading data...")
    df, elo_dict = load_data()
    print(f"   ✅ Loaded {len(df):,} matches, {len(elo_dict)} teams")

    # ==========================================
    # STEP 2: ADD OPPONENT COLUMN
    # ==========================================
    print("\n🔄 STEP 2: Adding opponent column...")

    # For each match, find the opponent
    opponents = []
    for idx, row in df.iterrows():
        game_id = row['gameid']
        team = row['teamname']

        # Find opponent in the same game
        opponent_row = df[(df['gameid'] == game_id) & (df['teamname'] != team)]
        if len(opponent_row) > 0:
            opponents.append(opponent_row.iloc[0]['teamname'])
        else:
            opponents.append(None)

    df['opponent'] = opponents
    print(f"   ✅ Added opponent column")

    # ==========================================
    # STEP 3: BUILD FEATURES
    # ==========================================
    print("\n🔧 STEP 3: Building features...")
    features_df = build_features(df, elo_dict)
    print(f"   ✅ Built {len(features_df):,} feature rows")

    # ==========================================
    # STEP 4: SAVE
    # ==========================================
    print("\n💾 STEP 4: Saving features...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, '2026_features.csv')
    features_df.to_csv(output_path, index=False)
    print(f"   ✅ Saved to: {output_path}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("✅ PHASE 3 COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Feature Summary:")
    print(f"   Total features: {features_df.shape[1]}")
    print(f"   Total rows: {features_df.shape[0]:,}")
    print(f"\n📋 Feature columns:")
    for col in features_df.columns:
        print(f"   - {col}")

    # Show sample
    print(f"\n📊 Sample data:")
    print(features_df.head(10).to_string())


if __name__ == "__main__":
    main()
