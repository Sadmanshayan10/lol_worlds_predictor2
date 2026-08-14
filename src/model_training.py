"""
Phase 4: Model Training
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Compare performance
- Select the best model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")


def load_data():
    """Load the feature dataset."""
    df = pd.read_csv('../data/processed/2026_features.csv')
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


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]
    })
    importances['Abs'] = importances['Coefficient'].abs()
    importances = importances.sort_values('Abs', ascending=False)
    print("\n📊 Feature Importance (Coefficients):")
    for i, row in importances.iterrows():
        print(f"   {row['Feature']}: {row['Coefficient']:.4f}")

    return model, accuracy


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
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
    importances = importances.sort_values('Importance', ascending=False)
    print("\n📊 Feature Importance:")
    for i, row in importances.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

    return model, accuracy


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost."""
    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)

    if not XGB_AVAILABLE:
        print("❌ XGBoost not available. Skipping.")
        return None, 0

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
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    })
    importances = importances.sort_values('Importance', ascending=False)
    print("\n📊 Feature Importance:")
    for i, row in importances.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")

    return model, accuracy


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
    X, y, feature_cols = prepare_features(df)
    print(f"   ✅ Features: {feature_cols}")
    print(f"   ✅ Target: result (1 = team1 wins)")

    # ==========================================
    # STEP 3: TRAIN/TEST SPLIT
    # ==========================================
    print("\n🔀 STEP 3: Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   ✅ Train: {len(X_train):,} rows")
    print(f"   ✅ Test: {len(X_test):,} rows")

    # ==========================================
    # STEP 4: TRAIN MODELS
    # ==========================================
    print("\n🤖 STEP 4: Training models...")

    results = {}

    # Logistic Regression
    lr_model, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test)
    results['Logistic Regression'] = lr_acc

    # Random Forest
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = rf_acc

    # XGBoost
    xgb_model, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_model is not None:
        results['XGBoost'] = xgb_acc

    # ==========================================
    # STEP 5: COMPARE RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print("\n📊 Accuracy Comparison:")
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model_name}: {acc:.4f} ({acc * 100:.2f}%)")

    best_model = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model} ({results[best_model]:.4f} accuracy)")

    # ==========================================
    # STEP 6: SAVE RESULTS
    # ==========================================
    print("\n💾 STEP 6: Saving results...")
    import os
    os.makedirs('../results', exist_ok=True)

    # Save model comparison
    comparison_df = pd.DataFrame([
        {'Model': name, 'Accuracy': acc} for name, acc in results.items()
    ])
    comparison_df.to_csv('../results/model_comparison.csv', index=False)
    print("   ✅ Saved: ../results/model_comparison.csv")

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
