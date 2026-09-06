"""
Walk-forward backtest of the Worlds prediction pipeline (2023, 2024, 2025).

For each year:
- Retrain the model using only matches that finished before that year's
  Worlds started (no look-ahead).
- Simulate the tournament the same way as tournament_simulation.py (top 8
  by ELO advance, then a seeded single-elimination bracket) and read off
  championship probabilities.
- Compare the predicted champion to the team that actually won, and compare
  per-match accuracy on the games played at that Worlds to a baseline of
  picking the higher-ELO team every time.

Needs the Oracle's Elixir yearly CSVs in data/raw/ (any file whose name
contains "LoL_esports_match_data"). Get the 2023-2025 files from
https://oracleselixir.com/tools/downloads; adding 2022 gives the 2023
training slice more history.

The Worlds start date, participant list and actual champion are all read
from the data, so nothing about the outcomes is hard-coded. Oracle's Elixir
tags the regional Worlds-qualifier series with the same 'WLDs' league weeks
before the tournament; date-gap clustering separates those games out and
keeps them as training data.

Output: results/backtest_summary.csv plus a printed summary.

    python src/backtest.py         # 10,000 simulations per year
    python src/backtest.py 2000    # override the simulation count
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

BACKTEST_YEARS = [2023, 2024, 2025]
N_SIMS = 10_000
N_KNOCKOUT_SEEDS = 8
SEED = 42

# Mirrors elo_system.py
BASE_ELO = 1200
K_BASE = 32
LEAGUE_STRENGTH = {
    'LCK': 1.00, 'LPL': 0.92, 'LEC': 0.85, 'LCS': 0.69, 'CBLOL': 0.54, 'LCP': 0.46,
}

# Mirrors model_training.py / tournament_simulation.py
FEATURE_COLS = [
    'elo_diff', 'h2h_win_rate',
    'team1_form_5', 'team2_form_5',
    'team1_form_10', 'team2_form_10',
    'team1_streak', 'team2_streak',
]

WORLDS_RE = re.compile(r'world|wlds', re.I)
INTERNATIONAL_RE = re.compile(r'world|wlds|msi|ewc|iem|rift|fst', re.I)


# ==========================================
# SMALL HELPERS (mirror elo_system / feature_engineering)
# ==========================================
def calculate_expected(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating, expected, actual, k_factor):
    return rating + k_factor * (actual - expected)


def form(results, n):
    """Win rate over the last n results; 0.5 if fewer than n are available."""
    if len(results) < n:
        return 0.5
    return float(np.mean(results[-n:]))


def signed_streak(results):
    """Current streak from a chronological 0/1 list: + for wins, - for losses."""
    if len(results) == 0:
        return 0
    last = results[-1]
    s = 0
    for r in reversed(results):
        if r == last:
            s += 1
        else:
            break
    return s if last == 1 else -s


# ==========================================
# DATA LOADING
# ==========================================
def load_raw_team_rows():
    """Load and concatenate every Oracle's Elixir CSV in data/raw/ (team rows only)."""
    files = sorted(set(glob.glob(os.path.join(RAW_DIR, '*LoL_esports_match_data*.csv'))))
    if not files:
        return None, []

    wanted = {'gameid', 'league', 'year', 'date', 'teamname', 'result', 'position'}
    frames = []
    file_info = []
    for path in files:
        d = pd.read_csv(path, low_memory=False, usecols=lambda c: c in wanted)
        d = d[d['position'] == 'team'].copy()
        d['date_parsed'] = pd.to_datetime(d['date'], errors='coerce')
        yrs = d['date_parsed'].dt.year.dropna()
        file_info.append((os.path.basename(path),
                          int(yrs.min()) if len(yrs) else None,
                          int(yrs.max()) if len(yrs) else None,
                          len(d)))
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['gameid', 'teamname', 'result', 'date_parsed'])
    df['result'] = df['result'].astype(int)
    df = df.drop_duplicates(subset=['gameid', 'teamname'])

    # opponent lookup (each game has exactly two team rows)
    teams_by_game = df.groupby('gameid')['teamname'].agg(list).to_dict()
    opponents = []
    for gid, tn in zip(df['gameid'].values, df['teamname'].values):
        others = [t for t in teams_by_game.get(gid, []) if t != tn]
        opponents.append(others[0] if len(others) == 1 else None)
    df['opponent'] = opponents
    df = df.dropna(subset=['opponent'])

    df['match_type'] = np.where(
        df['league'].map(lambda l: bool(INTERNATIONAL_RE.search(str(l)))),
        'international', 'regional',
    )
    df = df.sort_values('date_parsed', kind='stable').reset_index(drop=True)
    return df, file_info


# ==========================================
# PIPELINE STEPS (parameterised by a training slice)
# ==========================================
def compute_scaled_elo(train_df):
    """ELO over the training slice, scaled to 1000-1600 (mirrors elo_system.py)."""
    teams = pd.unique(pd.concat([train_df['teamname'], train_df['opponent']], ignore_index=True))
    elo = {t: BASE_ELO for t in teams}

    for row in train_df.itertuples(index=False):
        team, opp, res, mtype, league = (
            row.teamname, row.opponent, row.result, row.match_type, row.league,
        )
        ra, rb = elo[team], elo[opp]
        expected = calculate_expected(ra, rb)
        if mtype == 'international':
            k = K_BASE * 1.5
        else:
            # both team rows of a regional game share a league, so the
            # average of team/opp league strength collapses to one lookup
            k = K_BASE * LEAGUE_STRENGTH.get(league, 0.80)
        elo[team] = update_elo(ra, expected, res, k)
        elo[opp] = update_elo(rb, expected, 1 - res, k)

    vals = pd.Series(elo)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        scaled = {t: 1300.0 for t in elo}
    else:
        scaled = (1000 + (vals - lo) / (hi - lo) * 600).to_dict()
    return scaled


def build_training_matrix(train_df, scaled_elo):
    """One pre-game feature row per game (mirrors feature_engineering.py)."""
    mean_elo = float(np.mean(list(scaled_elo.values()))) if scaled_elo else 1300.0
    hist = {}          # team -> chronological list of results
    h2h = {}           # (team, opp) -> chronological list of team's results
    seen_games = set()
    rows = []

    for row in train_df.itertuples(index=False):
        gid, team, opp, res = row.gameid, row.teamname, row.opponent, row.result

        if gid not in seen_games:
            e1 = scaled_elo.get(team, mean_elo)
            e2 = scaled_elo.get(opp, mean_elo)
            t_hist, o_hist = hist.get(team, []), hist.get(opp, [])
            pair = h2h.get((team, opp), [])
            rows.append({
                'result': res,
                'elo_diff': e1 - e2,
                'h2h_win_rate': float(np.mean(pair)) if pair else 0.5,
                'team1_form_5': form(t_hist, 5),
                'team2_form_5': form(o_hist, 5),
                'team1_form_10': form(t_hist, 10),
                'team2_form_10': form(o_hist, 10),
                'team1_streak': signed_streak(t_hist[-10:]),
                'team2_streak': signed_streak(o_hist[-10:]),
            })
            seen_games.add(gid)

        hist.setdefault(team, []).append(res)
        h2h.setdefault((team, opp), []).append(res)

    return pd.DataFrame(rows)


def train_model(matrix):
    """StandardScaler + LogisticRegression (mirrors model_training.py)."""
    X = matrix[FEATURE_COLS].fillna(0).values
    y = matrix['result'].values
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(scaler.transform(X), y)
    return model, scaler


def team_snapshots(train_df, scaled_elo):
    """Full result history / h2h as of the training cutoff (the pre-Worlds state)."""
    hist, h2h = {}, {}
    for row in train_df.itertuples(index=False):
        hist.setdefault(row.teamname, []).append(row.result)
        h2h.setdefault((row.teamname, row.opponent), []).append(row.result)
    mean_elo = float(np.mean(list(scaled_elo.values()))) if scaled_elo else 1300.0
    return hist, h2h, mean_elo


def matchup_features(t1, t2, hist, h2h, scaled_elo, mean_elo):
    e1 = scaled_elo.get(t1, mean_elo)
    e2 = scaled_elo.get(t2, mean_elo)
    pair = h2h.get((t1, t2), [])
    h1, h2 = hist.get(t1, []), hist.get(t2, [])
    return [
        e1 - e2,
        float(np.mean(pair)) if pair else 0.5,
        form(h1, 5), form(h2, 5),
        form(h1, 10), form(h2, 10),
        signed_streak(h1[-10:]), signed_streak(h2[-10:]),
    ]


def win_prob(model, scaler, feats):
    x = scaler.transform(np.array([feats], dtype=float))
    return float(model.predict_proba(x)[0][1])


# ==========================================
# EVALUATION
# ==========================================
def evaluate_worlds_matches(worlds_df, model, scaler, hist, h2h, scaled_elo, mean_elo):
    """Per-match accuracy on the games actually played at this Worlds."""
    model_correct = base_correct = n = 0
    for gid, grp in worlds_df.groupby('gameid'):
        if len(grp) != 2:
            continue
        r1, r2 = grp.iloc[0], grp.iloc[1]
        t1, t2 = r1['teamname'], r2['teamname']
        actual_t1_win = int(r1['result']) == 1

        p = win_prob(model, scaler,
                     matchup_features(t1, t2, hist, h2h, scaled_elo, mean_elo))
        model_correct += int((p > 0.5) == actual_t1_win)

        e1 = scaled_elo.get(t1, mean_elo)
        e2 = scaled_elo.get(t2, mean_elo)
        base_correct += int((e1 > e2) == actual_t1_win)
        n += 1
    return model_correct, base_correct, n


def simulate_tournament(participants, model, scaler, hist, h2h, scaled_elo, mean_elo, n_sims):
    """Top-8 by ELO -> seeded single-elim bracket, Monte Carlo (mirrors tournament_simulation.py)."""
    seeds = sorted(participants, key=lambda t: scaled_elo.get(t, mean_elo), reverse=True)
    bracket = seeds[:N_KNOCKOUT_SEEDS]
    if len(bracket) < N_KNOCKOUT_SEEDS:
        return None

    prob = {}
    for a in bracket:
        for b in bracket:
            if a != b:
                prob[(a, b)] = win_prob(
                    model, scaler,
                    matchup_features(a, b, hist, h2h, scaled_elo, mean_elo),
                )

    quarters = [
        (bracket[0], bracket[7]), (bracket[1], bracket[6]),
        (bracket[2], bracket[5]), (bracket[3], bracket[4]),
    ]

    rng = np.random.default_rng(SEED)
    draws = rng.random((n_sims, 7))
    wins = {t: 0 for t in bracket}
    for s in range(n_sims):
        d = draws[s]
        sf = [a if d[i] < prob[(a, b)] else b for i, (a, b) in enumerate(quarters)]
        f1 = sf[0] if d[4] < prob[(sf[0], sf[3])] else sf[3]
        f2 = sf[1] if d[5] < prob[(sf[1], sf[2])] else sf[2]
        champ = f1 if d[6] < prob[(f1, f2)] else f2
        wins[champ] += 1

    probs_in_bracket = {t: wins[t] / n_sims for t in bracket}
    full_probs = {t: probs_in_bracket.get(t, 0.0) for t in participants}
    return full_probs, bracket


def isolate_main_event(worlds_df, max_gap_days=9):
    """Keep only the main Worlds tournament.

    Oracle's Elixir tags regional Worlds-qualifier series with the same
    'WLDs' league as the main event, weeks earlier. Split the WLDs games
    into date clusters (a gap of more than `max_gap_days` starts a new one)
    and keep the cluster with the most games, which is always the main event.
    """
    days = sorted(pd.to_datetime(worlds_df['date_parsed'].dt.normalize().unique()))
    clusters, current = [], [days[0]]
    for d in days[1:]:
        if (d - current[-1]).days > max_gap_days:
            clusters.append(current)
            current = [d]
        else:
            current.append(d)
    clusters.append(current)

    def n_games(cluster):
        lo, hi = cluster[0], cluster[-1]
        block = worlds_df[(worlds_df['date_parsed'] >= lo)
                          & (worlds_df['date_parsed'] < hi + pd.Timedelta(days=1))]
        return block['gameid'].nunique()

    main = max(clusters, key=n_games)
    lo, hi = main[0], main[-1] + pd.Timedelta(days=1)
    return worlds_df[(worlds_df['date_parsed'] >= lo) & (worlds_df['date_parsed'] < hi)]


def champion_report(full_probs, actual_champion):
    ranking = sorted(full_probs.items(), key=lambda kv: kv[1], reverse=True)
    top_pick, top_prob = ranking[0]
    actual_prob = full_probs.get(actual_champion, 0.0)
    rank = 1 + sum(1 for _, p in ranking if p > actual_prob)
    brier = sum((p - (1.0 if t == actual_champion else 0.0)) ** 2
                for t, p in full_probs.items())
    return top_pick, top_prob, actual_prob, rank, brier, ranking


# ==========================================
# MAIN
# ==========================================
def main():
    n_sims = N_SIMS
    if len(sys.argv) > 1:
        try:
            n_sims = int(sys.argv[1])
        except ValueError:
            print(f"⚠️ Ignoring non-integer sim count '{sys.argv[1]}'")

    print("=" * 70)
    print("WALK-FORWARD BACKTEST: Worlds 2023 / 2024 / 2025")
    print("=" * 70)

    raw, file_info = load_raw_team_rows()
    if raw is None or raw.empty:
        print("\n❌ No Oracle's Elixir data found in data/raw/")
        print("   Download the 2022-2025 yearly CSVs from")
        print("   https://oracleselixir.com/tools/downloads and place them in data/raw/")
        return

    print("\n📂 Raw files detected:")
    for name, y0, y1, nrows in file_info:
        span = f"{y0}-{y1}" if y0 else "no dated rows"
        print(f"   - {name:60s} {span:12s} {nrows:>8,} team rows")

    print(f"\n🎲 Simulations per year: {n_sims:,}")

    summary_rows = []
    for year in BACKTEST_YEARS:
        is_worlds = raw['league'].map(lambda l: bool(WORLDS_RE.search(str(l))))
        wdf = raw[is_worlds & (raw['date_parsed'].dt.year == year)]

        print("\n" + "-" * 70)
        print(f"WORLDS {year}")
        print("-" * 70)

        if wdf.empty:
            print(f"   ⚠️ No Worlds {year} matches found in data/raw/. Skipping.")
            print(f"      (need a file containing {year} data with a 'Worlds'/'WLDs' league)")
            continue

        # strip the regional Worlds-qualifier games that share the 'WLDs' tag
        wdf = isolate_main_event(wdf)
        start = wdf['date_parsed'].min()
        train_df = raw[raw['date_parsed'] < start]
        if len(train_df) < 200:
            print(f"   ⚠️ Only {len(train_df):,} training matches before {start.date()}. "
                  f"Skipping (add the previous year's CSV).")
            continue

        last_gid = wdf.sort_values('date_parsed', kind='stable')['gameid'].iloc[-1]
        last_game = wdf[wdf['gameid'] == last_gid]
        actual_champion = last_game.loc[last_game['result'] == 1, 'teamname'].iloc[0]
        participants = sorted(wdf['teamname'].unique())
        n_train_games = train_df['gameid'].nunique()

        print(f"   Worlds start (from data): {start.date()}")
        print(f"   Training slice:           {n_train_games:,} games, "
              f"{train_df['date_parsed'].min().date()} to {train_df['date_parsed'].max().date()}")
        print(f"   Participants:             {len(participants)}")
        print(f"   Actual champion:          {actual_champion}")

        scaled_elo = compute_scaled_elo(train_df)
        matrix = build_training_matrix(train_df, scaled_elo)
        if matrix['result'].nunique() < 2:
            print("   ⚠️ Training slice has only one outcome class. Skipping.")
            continue
        model, scaler = train_model(matrix)
        hist, h2h, mean_elo = team_snapshots(train_df, scaled_elo)

        model_correct, base_correct, n_matches = evaluate_worlds_matches(
            wdf, model, scaler, hist, h2h, scaled_elo, mean_elo,
        )
        model_acc = model_correct / n_matches if n_matches else float('nan')
        base_acc = base_correct / n_matches if n_matches else float('nan')

        sim = simulate_tournament(
            participants, model, scaler, hist, h2h, scaled_elo, mean_elo, n_sims,
        )
        if sim is None:
            print("   ⚠️ Fewer than 8 participants matched, cannot simulate the bracket.")
            continue
        full_probs, bracket = sim
        (top_pick, top_prob, actual_prob, actual_rank,
         brier, ranking) = champion_report(full_probs, actual_champion)

        elo_baseline_champion = max(participants, key=lambda t: scaled_elo.get(t, mean_elo))
        top_pick_correct = top_pick == actual_champion

        print(f"\n   Predicted title odds (top 5):")
        for t, p in ranking[:5]:
            mark = "  <-- actual champion" if t == actual_champion else ""
            print(f"      {t:28s} {p * 100:5.1f}%{mark}")
        if actual_prob == 0.0:
            print(f"   Actual champion ({actual_champion}) was outside the ELO top-8 seed, "
                  f"so 0% in this simulation format.")
        else:
            print(f"   Actual champion odds:     {actual_prob * 100:.1f}%  "
                  f"(rank {actual_rank} of {len(participants)})")
        print(f"   Model top pick:           {top_pick} ({top_prob * 100:.1f}%)  "
              f"{'✅ correct' if top_pick_correct else '❌ wrong'}")
        print(f"   Higher-ELO baseline pick: {elo_baseline_champion} "
              f"{'✅ correct' if elo_baseline_champion == actual_champion else '❌ wrong'}")
        print(f"\n   Per-match accuracy on the {n_matches} games played at Worlds {year}:")
        print(f"      Model (logistic reg.):  {model_acc * 100:5.1f}%  "
              f"({model_correct}/{n_matches})")
        print(f"      Baseline (higher ELO):  {base_acc * 100:5.1f}%  "
              f"({base_correct}/{n_matches})")
        print(f"      Model edge over baseline: {(model_acc - base_acc) * 100:+.1f} pp")

        summary_rows.append({
            'year': year,
            'worlds_start_date': start.date().isoformat(),
            'n_train_games': n_train_games,
            'n_participants': len(participants),
            'n_worlds_games': n_matches,
            'actual_champion': actual_champion,
            'model_top_pick': top_pick,
            'model_top_pick_prob': round(top_prob, 4),
            'top_pick_correct': top_pick_correct,
            'actual_champion_prob': round(actual_prob, 4),
            'actual_champion_rank': actual_rank,
            'champion_brier': round(brier, 4),
            'elo_baseline_champion': elo_baseline_champion,
            'elo_baseline_champion_correct': elo_baseline_champion == actual_champion,
            'model_match_accuracy': round(model_acc, 4),
            'baseline_match_accuracy': round(base_acc, 4),
            'model_correct_matches': model_correct,
            'baseline_correct_matches': base_correct,
        })

    # ==========================================
    # AGGREGATE + SAVE
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not summary_rows:
        print("\n   No years could be backtested. Add the 2022-2025 Oracle's Elixir")
        print("   yearly CSVs to data/raw/ and re-run.")
        return

    summary_df = pd.DataFrame(summary_rows)
    n_years = len(summary_df)
    champ_hits = int(summary_df['top_pick_correct'].sum())
    base_champ_hits = int(summary_df['elo_baseline_champion_correct'].sum())
    mean_model_acc = summary_df['model_match_accuracy'].mean()
    mean_base_acc = summary_df['baseline_match_accuracy'].mean()
    mean_actual_prob = summary_df['actual_champion_prob'].mean()
    mean_actual_rank = summary_df['actual_champion_rank'].mean()

    print(f"\n   Years backtested: {n_years}  ({', '.join(map(str, summary_df['year']))})")
    print(f"\n   Champion called correctly (model top pick):    {champ_hits}/{n_years}")
    print(f"   Champion called correctly (higher-ELO team):   {base_champ_hits}/{n_years}")
    print(f"\n   Mean per-match accuracy:")
    print(f"      Model:    {mean_model_acc * 100:5.1f}%")
    print(f"      Baseline: {mean_base_acc * 100:5.1f}%")
    print(f"      Edge:     {(mean_model_acc - mean_base_acc) * 100:+.1f} pp")
    print(f"\n   Mean probability the model gave the eventual champion: {mean_actual_prob * 100:.1f}%")
    print(f"   Mean rank of the eventual champion in the model's odds: {mean_actual_rank:.1f}")

    verdict = "beats" if mean_model_acc > mean_base_acc else (
        "matches" if abs(mean_model_acc - mean_base_acc) < 1e-9 else "trails")
    print(f"\n   The model {verdict} the higher-ELO baseline on per-match accuracy,")
    print(f"   and its top pick won {champ_hits} of {n_years} backtested Worlds.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    agg_row = {
        'year': 'ALL', 'worlds_start_date': '', 'n_train_games': '',
        'n_participants': '', 'n_worlds_games': int(summary_df['n_worlds_games'].sum()),
        'actual_champion': '', 'model_top_pick': '',
        'model_top_pick_prob': '', 'top_pick_correct': f"{champ_hits}/{n_years}",
        'actual_champion_prob': round(mean_actual_prob, 4),
        'actual_champion_rank': round(mean_actual_rank, 2),
        'champion_brier': round(summary_df['champion_brier'].mean(), 4),
        'elo_baseline_champion': '',
        'elo_baseline_champion_correct': f"{base_champ_hits}/{n_years}",
        'model_match_accuracy': round(mean_model_acc, 4),
        'baseline_match_accuracy': round(mean_base_acc, 4),
        'model_correct_matches': int(summary_df['model_correct_matches'].sum()),
        'baseline_correct_matches': int(summary_df['baseline_correct_matches'].sum()),
    }
    out_df = pd.concat([summary_df, pd.DataFrame([agg_row])], ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, 'backtest_summary.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\n💾 Saved: {out_path}")

    print("\n" + "=" * 70)
    print("✅ BACKTEST COMPLETE!")
    print("=" * 70)
    print("\nNote: the simulation reuses tournament_simulation.py's simplified format")
    print("(deterministic top-8-by-ELO group stage, then a fixed 1v8..4v5 bracket),")
    print("so a real champion seeded outside our ELO top 8 shows 0% title odds.")


if __name__ == "__main__":
    main()
