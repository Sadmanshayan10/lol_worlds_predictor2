"""
Phase 4: Model Training
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Compare performance
- Select the best model
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def load_data():
    """Load the feature dataset."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, '2026_features.csv'))
    return df


def prepare_features(df):
    """Prepare features and target for training."""
    # Features to use
    feature_cols = [
        'elo_diff',
        'h2h_win_rate',
        'team1_form_5',
        'team2_form_5',
        'team1_form_10',
        'team2_form_10',
        'team1_streak',
        'team2_streak',
    ]

    X = df[feature_cols]
    y = df['result']

    # Handle any missing values
    X = X.fillna(0)

    return X, y, feature_cols


def report_probability_metrics(y_test, y_proba):
    """Print probability-quality metrics for a model's test-set predictions.

    These matter because tournament_simulation.py feeds these probabilities
    into a Monte Carlo simulation, so miscalibration compounds across the
    10,000 simulated matches.
    """
    brier = brier_score_loss(y_test, y_proba)
    ll = log_loss(y_test, y_proba)
    print("\n📊 Probability Quality (lower is better):")
    print(f"   Brier score: {brier:.4f}")
    print(f"   Log loss:    {ll:.4f}")
    return brier, ll


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    # Standardize features so coefficients are on a comparable scale.
    # Fit the scaler on the training set only, then transform both sets.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    report_probability_metrics(y_test, y_proba)

    # Feature importance (standardized coefficients, comparable across features)
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]
    })
    importances['Abs'] = importances['Coefficient'].abs()
    importances = importances.sort_values('Abs', ascending=False)
    print("\n📊 Feature Importance (Standardized Coefficients):")
    for i, row in importances.iterrows():
        print(f"   {row['Feature']}: {row['Coefficient']:.4f}")

    return model, accuracy, y_proba


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    report_probability_metrics(y_test, y_proba)

    # Feature importance
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
    importances = importances.sort_values('Importance', ascending=False)
    print("\n📊 Feature Importance:")
    for i, row in importances.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

    return model, accuracy, y_proba


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost."""
    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)

    if not XGB_AVAILABLE:
        print("❌ XGBoost not available. Skipping.")
        return None, 0, None

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    report_probability_metrics(y_test, y_proba)

    # Feature importance
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
    importances = importances.sort_values('Importance', ascending=False)
    print("\n📊 Feature Importance:")
    for i, row in importances.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

    return model, accuracy, y_proba


def plot_calibration_curve(best_model_name, y_test, y_proba, out_path):
    """Save a reliability curve for the best model's test-set probabilities."""
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.plot(mean_pred, frac_pos, 'o-', label=best_model_name)
    ax.set_xlabel('Mean predicted probability (team1 win)')
    ax.set_ylabel('Observed fraction of team1 wins')
    ax.set_title(f'Calibration Curve ({best_model_name})')
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 4: MODEL TRAINING")
    print("=" * 70)

    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    print("\n📂 STEP 1: Loading features...")
    df = load_data()
    print(f"   ✅ Loaded {len(df):,} rows, {df.shape[1]} columns")

    # ==========================================
    # STEP 2: PREPARE FEATURES
    # ==========================================
    print("\n🔧 STEP 2: Preparing features...")
    # Sort chronologically so the split below is temporal, not random.
    df = df.sort_values('date').reset_index(drop=True)
    X, y, feature_cols = prepare_features(df)
    print(f"   ✅ Features: {feature_cols}")
    print(f"   ✅ Target: result (1 = team1 wins)")

    # ==========================================
    # STEP 3: TEMPORAL TRAIN/TEST SPLIT
    # ==========================================
    print("\n🔀 STEP 3: Temporal split (earliest 80% train, most recent 20% test)...")
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"   ✅ Train: {len(X_train):,} rows (through {df['date'].iloc[split_idx - 1][:10]})")
    print(f"   ✅ Test:  {len(X_test):,} rows (from {df['date'].iloc[split_idx][:10]})")

    # ==========================================
    # STEP 4: TRAIN MODELS
    # ==========================================
    print("\n🤖 STEP 4: Training models...")

    results = {}
    probabilities = {}

    # Logistic Regression
    lr_model, lr_acc, lr_proba = train_logistic_regression(X_train, y_train, X_test, y_test)
    results['Logistic Regression'] = lr_acc
    probabilities['Logistic Regression'] = lr_proba

    # Random Forest
    rf_model, rf_acc, rf_proba = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = rf_acc
    probabilities['Random Forest'] = rf_proba

    # XGBoost
    xgb_model, xgb_acc, xgb_proba = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_model is not None:
        results['XGBoost'] = xgb_acc
        probabilities['XGBoost'] = xgb_proba

    # ==========================================
    # STEP 5: COMPARE RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print("\n📊 Accuracy Comparison:")
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model_name}: {acc:.4f} ({acc * 100:.2f}%)")

    print("\n📊 Probability Quality Comparison (lower is better):")
    print(f"   {'Model':22s} {'Brier':>8s} {'LogLoss':>9s}")
    for model_name, proba in probabilities.items():
        brier = brier_score_loss(y_test, proba)
        ll = log_loss(y_test, proba)
        print(f"   {model_name:22s} {brier:8.4f} {ll:9.4f}")

    best_model = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model} ({results[best_model]:.4f} accuracy)")

    # ==========================================
    # STEP 6: SAVE RESULTS
    # ==========================================
    print("\n💾 STEP 6: Saving results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save model comparison
    comparison_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': acc,
            'Brier': brier_score_loss(y_test, probabilities[name]),
            'LogLoss': log_loss(y_test, probabilities[name]),
        }
        for name, acc in results.items()
    ])
    output_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"   ✅ Saved: {output_path}")

    # Calibration (reliability) curve for the best model
    plot_path = os.path.join(RESULTS_DIR, 'calibration_plot.png')
    plot_calibration_curve(best_model, y_test, probabilities[best_model], plot_path)
    print(f"   ✅ Saved: {plot_path}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 70)

    print(f"\n📊 Summary:")
    print(f"   Features used: {len(feature_cols)}")
    print(f"   Training rows: {len(X_train):,}")
    print(f"   Testing rows: {len(X_test):,}")
    print(f"   Best model: {best_model} ({results[best_model]:.4f})")


if __name__ == "__main__":
    main()
