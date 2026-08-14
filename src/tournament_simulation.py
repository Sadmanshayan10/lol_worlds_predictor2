"""
Phase 5: Tournament Simulation (Full Bracket)
- Swiss Stage: 20 teams → 8 teams
- Knockout Stage: Quarterfinals → Semifinals → Finals
- Simulates 10,000 tournaments
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# WORLDS 2026 TEAMS (20 Teams)
# ==========================================

WORLDS_TEAMS = {
    'LCK': ['T1', 'Gen.G', 'Hanwha Life Esports', 'Dplus Kia'],
    'LPL': ['Bilibili Gaming', 'Top Esports', "Anyone's Legend", 'JD Gaming'],
    'LEC': ['G2 Esports', 'Karmine Corp', 'Team Vitality'],
    'LCS': ['LYON', 'Team Liquid', 'Cloud9'],
    'LCP': ['Team Secret Whales', 'GAM Esports'],
    'CBLOL': ['LØS'],
    'Play-in': ['FURIA', 'RED Canids', 'Sentinels'],
}

def load_model_and_data():
    """Load the trained model and feature data."""
    df = pd.read_csv('../data/processed/2026_features.csv')
    raw_df = pd.read_csv('../data/processed/2026_data_prepared.csv')
    elo_df = pd.read_csv('../data/processed/2026_elo_ratings.csv')
    elo_dict = dict(zip(elo_df['teamname'], elo_df['scaled_elo']))

    # ==========================================
    # ADD OPPONENT COLUMN TO RAW_DF
    # ==========================================
    opponents = []
    for idx, row in raw_df.iterrows():
        game_id = row['gameid']
        team = row['teamname']
        opponent_row = raw_df[(raw_df['gameid'] == game_id) & (raw_df['teamname'] != team)]
        if len(opponent_row) > 0:
            opponents.append(opponent_row.iloc[0]['teamname'])
        else:
            opponents.append(None)
    raw_df['opponent'] = opponents

    # Train the model on all data
    features = ['elo_diff', 'h2h_win_rate', 'team1_form_5', 'team2_form_5',
                'team1_form_10', 'team2_form_10', 'team1_streak', 'team2_streak']
    X = df[features].fillna(0)
    y = df['result']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, df, raw_df, elo_dict, features

def predict_match(model, team1, team2, df, raw_df, elo_dict, features):
    """Predict the probability of team1 beating team2."""

    # Head-to-head
    h2h = raw_df[((raw_df['teamname'] == team1) & (raw_df['opponent'] == team2)) |
                 ((raw_df['teamname'] == team2) & (raw_df['opponent'] == team1))]

    if len(h2h) == 0:
        h2h_win_rate = 0.5
    else:
        team1_h2h = h2h[h2h['teamname'] == team1]
        h2h_win_rate = team1_h2h['result'].mean() if len(team1_h2h) > 0 else 0.5

    # Recent form
    team1_matches = raw_df[raw_df['teamname'] == team1].tail(10)
    team2_matches = raw_df[raw_df['teamname'] == team2].tail(10)

    team1_form_5 = team1_matches.tail(5)['result'].mean() if len(team1_matches) >= 5 else 0.5
    team2_form_5 = team2_matches.tail(5)['result'].mean() if len(team2_matches) >= 5 else 0.5
    team1_form_10 = team1_matches['result'].mean() if len(team1_matches) >= 10 else 0.5
    team2_form_10 = team2_matches['result'].mean() if len(team2_matches) >= 10 else 0.5

    # Streaks
    def get_streak(team_df):
        if len(team_df) == 0:
            return 0
        results = team_df['result'].values
        if len(results) == 0:
            return 0
        last_result = results[-1]
        streak = 0
        for r in reversed(results):
            if r == last_result:
                streak += 1
            else:
                break
        return streak if last_result == 1 else -streak

    team1_streak = get_streak(team1_matches)
    team2_streak = get_streak(team2_matches)

    # ELO
    team1_elo = elo_dict.get(team1, 1200)
    team2_elo = elo_dict.get(team2, 1200)
    elo_diff = team1_elo - team2_elo

    # Feature vector
    features_values = [
        elo_diff,
        h2h_win_rate,
        team1_form_5,
        team2_form_5,
        team1_form_10,
        team2_form_10,
        team1_streak,
        team2_streak
    ]

    X = np.array([features_values]).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]

    return prob

def simulate_match(model, team1, team2, df, raw_df, elo_dict, features):
    """Simulate a single match and return the winner."""
    prob = predict_match(model, team1, team2, df, raw_df, elo_dict, features)
    winner = team1 if np.random.random() < prob else team2
    return winner, prob

def simulate_swiss_stage(model, teams, df, raw_df, elo_dict, features):
    """Simulate Swiss stage (20 teams → 8 teams)."""
    # Use ELO to seed and advance top 8
    elo_dict_local = elo_dict.copy()
    team_elos = [(team, elo_dict_local.get(team, 1200)) for team in teams]
    team_elos.sort(key=lambda x: x[1], reverse=True)
    return [team for team, _ in team_elos[:8]]

def simulate_knockout_stage(model, teams, df, raw_df, elo_dict, features):
    """Simulate quarterfinals → semifinals → finals."""
    # Seed: 1st vs 8th, 2nd vs 7th, 3rd vs 6th, 4th vs 5th
    quarterfinals = [
        (teams[0], teams[7]),
        (teams[1], teams[6]),
        (teams[2], teams[5]),
        (teams[3], teams[4]),
    ]

    semifinalists = []
    for team1, team2 in quarterfinals:
        winner, _ = simulate_match(model, team1, team2, df, raw_df, elo_dict, features)
        semifinalists.append(winner)

    # Semifinals
    winner1, _ = simulate_match(model, semifinalists[0], semifinalists[3], df, raw_df, elo_dict, features)
    winner2, _ = simulate_match(model, semifinalists[1], semifinalists[2], df, raw_df, elo_dict, features)

    # Finals
    champion, _ = simulate_match(model, winner1, winner2, df, raw_df, elo_dict, features)

    return champion

def main():
    print("=" * 70)
    print("PHASE 5: TOURNAMENT SIMULATION (FULL BRACKET)")
    print("=" * 70)

    # ==========================================
    # STEP 1: LOAD MODEL AND DATA
    # ==========================================
    print("\n📂 STEP 1: Loading model and data...")
    model, df, raw_df, elo_dict, features = load_model_and_data()
    print(f"   ✅ Model loaded, {len(df):,} data points")
    print(f"   ✅ Raw data loaded, {len(raw_df):,} rows")

    # ==========================================
    # STEP 2: GET ALL WORLDS TEAMS
    # ==========================================
    print("\n🏆 STEP 2: Worlds 2026 Qualifying Teams")
    print("   " + "-" * 50)

    all_teams = []
    for region, teams in WORLDS_TEAMS.items():
        all_teams.extend(teams)
        print(f"   {region}: {', '.join(teams)}")

    print(f"\n   Total teams: {len(all_teams)}")

    # ==========================================
    # STEP 3: SIMULATE TOURNAMENT
    # ==========================================
    print("\n🎲 STEP 3: Simulating tournament (10,000 simulations)...")

    champion_counts = {}

    for sim in range(10000):
        if sim % 1000 == 0:
            print(f"   Simulation {sim}/10000...")

        # Swiss stage: 20 teams → 8 teams
        swiss_teams = simulate_swiss_stage(model, all_teams, df, raw_df, elo_dict, features)

        # Knockout stage: 8 teams → champion
        champion = simulate_knockout_stage(model, swiss_teams, df, raw_df, elo_dict, features)

        champion_counts[champion] = champion_counts.get(champion, 0) + 1

    # ==========================================
    # STEP 4: RESULTS
    # ==========================================
    print("\n📊 STEP 4: Results")
    print("   " + "-" * 50)

    total = sum(champion_counts.values())
    sorted_results = sorted(champion_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n🏆 Champion Probabilities (based on 10,000 simulations):")
    for team, wins in sorted_results[:15]:
        pct = wins / total * 100
        print(f"   {team}: {pct:.2f}%")

    # ==========================================
    # STEP 5: SAVE
    # ==========================================
    print("\n💾 STEP 5: Saving results...")
    os.makedirs('../results', exist_ok=True)

    results_df = pd.DataFrame([
        {'Team': team, 'Wins': wins, 'Win_Percentage': wins / total * 100}
        for team, wins in sorted_results
    ])
    results_df.to_csv('../results/worlds_prediction_full.csv', index=False)
    print("   ✅ Saved: ../results/worlds_prediction_full.csv")

    print("\n" + "=" * 70)
    print("✅ PHASE 5 COMPLETE!")
    print("=" * 70)

    print(f"\n🏆 Predicted World Champion:")
    print(f"   {sorted_results[0][0]} with {sorted_results[0][1] / total * 100:.2f}% chance")

if __name__ == "__main__":
    main()
