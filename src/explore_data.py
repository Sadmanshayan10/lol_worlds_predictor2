"""
Data Exploration Script
- Check what leagues are available
- Check what columns exist
- Check date range
- Check missing values
- Understand the data before processing
"""

import pandas as pd
import os
from datetime import datetime


def main():
    print("=" * 70)
    print("DATA EXPLORATION - 2026 SEASON")
    print("=" * 70)

    # ==========================================
    # LOAD RAW DATA
    # ==========================================
    print("\n📂 Loading raw data...")
    raw_path = '../data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv'

    if not os.path.exists(raw_path):
        print(f"   ❌ File not found: {raw_path}")
        print("   Please make sure the file is in data/raw/")
        return

    df = pd.read_csv(raw_path, low_memory=False)
    print(f"   ✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # ==========================================
    # 1. CHECK ALL LEAGUES
    # ==========================================
    print("\n" + "=" * 70)
    print("1. ALL LEAGUES IN DATA")
    print("=" * 70)

    leagues = df['league'].value_counts()
    print(f"\nTotal leagues: {len(leagues)}")
    print("\nTop 20 leagues by matches:")
    for i, (league, count) in enumerate(leagues.head(20).items(), 1):
        pct = count / len(df) * 100
        print(f"   {i:2d}. {league:10s}: {count:6,} rows ({pct:.1f}%)")

    # ==========================================
    # 2. CHECK MAJOR REGIONS
    # ==========================================
    print("\n" + "=" * 70)
    print("2. MAJOR REGIONS (Worlds 2026)")
    print("=" * 70)

    major_leagues = ['LPL', 'LCK', 'LEC', 'LCS', 'CBLOL', 'LCP']

    print("\nChecking for each major region:")
    for league in major_leagues:
        if league in df['league'].unique():
            count = df[df['league'] == league].shape[0]
            print(f"   ✅ {league:6s}: {count:6,} rows found")
        else:
            print(f"   ❌ {league:6s}: NOT FOUND in data")

    # Count total major region rows
    major_df = df[df['league'].isin(major_leagues)]
    print(f"\n   Total major region rows: {major_df.shape[0]:,} ({major_df.shape[0] / len(df) * 100:.1f}% of data)")

    # ==========================================
    # 3. CHECK INTERNATIONAL EVENTS
    # ==========================================
    print("\n" + "=" * 70)
    print("3. INTERNATIONAL EVENTS")
    print("=" * 70)

    international_keywords = ['MSI', 'Worlds', 'EWC', 'Rift', 'IEM']

    intl_leagues = []
    for league in df['league'].unique():
        for keyword in international_keywords:
            if keyword.lower() in league.lower():
                intl_leagues.append(league)
                break

    if intl_leagues:
        print("\n   International leagues found:")
        for league in intl_leagues:
            count = df[df['league'] == league].shape[0]
            print(f"      {league}: {count:,} rows")
    else:
        print("   ⚠️ No international events found in data")

    # ==========================================
    # 4. CHECK DATE RANGE
    # ==========================================
    print("\n" + "=" * 70)
    print("4. DATE RANGE")
    print("=" * 70)

    if 'date' in df.columns:
        # Convert to datetime
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        valid_dates = df['date_parsed'].notna()

        if valid_dates.any():
            min_date = df[valid_dates]['date_parsed'].min()
            max_date = df[valid_dates]['date_parsed'].max()
            print(f"\n   Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            print(f"   Valid dates: {valid_dates.sum():,} rows ({valid_dates.sum() / len(df) * 100:.1f}%)")
        else:
            print("\n   ⚠️ No valid dates found")
    else:
        print("\n   ⚠️ No 'date' column found in data")

    # ==========================================
    # 5. CHECK KEY COLUMNS
    # ==========================================
    print("\n" + "=" * 70)
    print("5. KEY COLUMNS AVAILABLE")
    print("=" * 70)

    key_columns = [
        'gameid', 'league', 'teamname', 'side', 'result',
        'firstdragon', 'firstbaron', 'firsttower',
        'kills', 'deaths', 'assists',
        'gamelength', 'date',
        'golddiffat10', 'xpdiffat10', 'csdiffat10',
        'golddiffat15', 'xpdiffat15', 'csdiffat15'
    ]

    print("\n   Checking for key columns:")
    for col in key_columns:
        if col in df.columns:
            # Check if column has data
            non_null = df[col].notna().sum()
            print(f"   ✅ {col:15s}: present ({non_null:,} non-null)")
        else:
            print(f"   ❌ {col:15s}: NOT FOUND")

    # ==========================================
    # 6. CHECK MISSING VALUES
    # ==========================================
    print("\n" + "=" * 70)
    print("6. MISSING VALUES (top 10 columns)")
    print("=" * 70)

    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(1)

    print("\n   Columns with most missing values:")
    for i, (col, count) in enumerate(missing.head(10).items(), 1):
        pct = missing_pct[col]
        print(f"   {i:2d}. {col:20s}: {count:6,} ({pct:.1f}%)")

    # ==========================================
    # 7. CHECK TEAMS PER LEAGUE
    # ==========================================
    print("\n" + "=" * 70)
    print("7. TEAMS PER MAJOR LEAGUE")
    print("=" * 70)

    for league in major_leagues:
        if league in df['league'].unique():
            league_df = df[df['league'] == league]
            teams = league_df['teamname'].unique()
            print(f"\n   {league}:")
            print(f"      Total teams: {len(teams)}")
            print(f"      Teams: {', '.join(teams[:5])}" + ("..." if len(teams) > 5 else ""))

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("EXPLORATION SUMMARY")
    print("=" * 70)

    print(f"\n📊 Dataset Size: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"🏆 Total Leagues: {len(leagues)}")
    print(f"🏆 Major Leagues Found: {len([l for l in major_leagues if l in df['league'].unique()])}/6")
    print(
        f"📅 Date Range: {min_date.strftime('%Y-%m-%d') if valid_dates.any() else 'N/A'} to {max_date.strftime('%Y-%m-%d') if valid_dates.any() else 'N/A'}")
    print(f"🧹 Missing Values: {df.isnull().sum().sum():,} total missing cells")

    print("\n" + "=" * 70)
    print("✅ EXPLORATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
