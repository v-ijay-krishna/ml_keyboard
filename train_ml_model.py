import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CSV_FILE = 'midi/ML_SCORES.csv'
MODEL_FILE = 'midi/human_score_model.pkl'
PLOT_DIR = 'midi/plots'
OUTPUT_PREDICTION_CSV = 'midi/ML_PREDICTIONS.csv'


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_level(score):
    if score >= 8:
        return "Good"
    elif score >= 5:
        return "Average"
    else:
        return "Poor"


def natural_key(name):
    try:
        return int(str(name).replace('.mid', ''))
    except:
        return str(name)


# ─────────────────────────────────────────────
# SIMPLE GRAPH FUNCTIONS
# ─────────────────────────────────────────────

def create_layman_graphs(result_df, importance_df, mae):
    os.makedirs(PLOT_DIR, exist_ok=True)

    result_df = result_df.copy()
    result_df = result_df.sort_values(
        by='Pattern',
        key=lambda col: col.map(natural_key)
    )

    x = range(len(result_df))

    # ─────────────────────────────────────────
    # GRAPH 1: Human Score vs ML Predicted Score
    # ─────────────────────────────────────────

    plt.figure(figsize=(14, 6))

    plt.plot(
        x,
        result_df['human_score_10'],
        marker='o',
        linewidth=3,
        label='Human Score'
    )

    plt.plot(
        x,
        result_df['predicted_score_10'],
        marker='o',
        linewidth=3,
        label='ML Predicted Score'
    )

    plt.xlabel("MIDI File")
    plt.ylabel("Score out of 10")
    plt.title("Human Score vs ML Predicted Score")
    plt.xticks(x, result_df['Pattern'], rotation=45)
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "01_human_vs_ml_prediction.png"), dpi=300)
    plt.close()

    # ─────────────────────────────────────────
    # GRAPH 2: Prediction Error per File
    # ─────────────────────────────────────────

    plt.figure(figsize=(14, 6))

    plt.bar(result_df['Pattern'], result_df['error'])
    plt.axhline(0, linewidth=2)

    plt.xlabel("MIDI File")
    plt.ylabel("Error")
    plt.title(f"Prediction Error per MIDI File | MAE = {mae:.2f} marks")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "02_prediction_error_per_file.png"), dpi=300)
    plt.close()

    # ─────────────────────────────────────────
    # GRAPH 3: Feature Importance
    # ─────────────────────────────────────────

    readable_names = {
        'pitch_score': 'Pitch / Note Correctness',
        'velocity_score': 'Velocity / Dynamics',
        'rhythm_score': 'Rhythm Pattern',
        'timing_score': 'Timing Accuracy'
    }

    importance_df = importance_df.copy()
    importance_df['Readable Feature'] = importance_df['Feature'].map(readable_names)

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df['Readable Feature'],
        importance_df['Importance']
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_random_forest_feature_importance.png"), dpi=300) 
    plt.close()

    # ─────────────────────────────────────────
    # GRAPH 4: Human Level vs ML Predicted Level
    # ─────────────────────────────────────────

    level_df = pd.DataFrame({
        'Human Level': result_df['human_level'].value_counts(),
        'ML Predicted Level': result_df['predicted_level'].value_counts()
    }).fillna(0)

    level_df = level_df.reindex(['Poor', 'Average', 'Good']).fillna(0)

    plt.figure(figsize=(8, 5))

    level_df.plot(kind='bar')

    plt.xlabel("Performance Level")
    plt.ylabel("Number of MIDI Files")
    plt.title("Human Level vs ML Predicted Level Count")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "04_level_count_comparison.png"), dpi=300)
    plt.close()

    # ─────────────────────────────────────────
    # GRAPH 5: Top 5 Performances
    # ─────────────────────────────────────────

    top_df = result_df.sort_values(
        by='human_score_10',
        ascending=False
    ).head(5)

    plt.figure(figsize=(10, 5))

    plt.bar(top_df['Pattern'], top_df['human_score_10'])

    plt.xlabel("MIDI File")
    plt.ylabel("Human Score out of 10")
    plt.title("Top 5 Best Performances Based on Human Score")
    plt.ylim(0, 10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR, "05_top_5_performances.png"), dpi=300)
    plt.close()

    print(f"\nSimple graphs saved inside: {PLOT_DIR}")


# ─────────────────────────────────────────────
# TRAIN RANDOM FOREST MODEL
# ─────────────────────────────────────────────

def train_model():
    print("\n── Random Forest Human Score Prediction ──")

    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] CSV file not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()

    print("\nCSV loaded successfully.")
    print(f"CSV path: {os.path.abspath(CSV_FILE)}")
    print(f"Total rows found: {len(df)}")

    # In case Excel shortened column names
    rename_map = {
        'pitch_scor': 'pitch_score',
        'velocity_sc': 'velocity_score',
        'rhythm_scr': 'rhythm_score',
        'timing_sco': 'timing_score',
        'human_score': 'human_score_10'
    }

    df = df.rename(columns=rename_map)

    required_columns = [
        'Pattern',
        'pitch_score',
        'velocity_score',
        'rhythm_score',
        'timing_score',
        'human_score_10'
    ]

    for col in required_columns:
        if col not in df.columns:
            print(f"\n[ERROR] Missing column: {col}")
            print("Columns found:")
            print(list(df.columns))
            return

    numeric_columns = [
        'pitch_score',
        'velocity_score',
        'rhythm_score',
        'timing_score',
        'human_score_10'
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove empty rows
    df = df.dropna(subset=numeric_columns)

    print(f"Valid rows after cleaning: {len(df)}")

    if len(df) < 5:
        print("\n[ERROR] You need at least 5 valid filled human scores.")
        return

    features = [
        'pitch_score',
        'velocity_score',
        'rhythm_score',
        'timing_score'
    ]

    X = df[features]
    y = df['human_score_10']

    test_size = 0.25 if len(df) >= 8 else 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Test prediction
    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions) if len(y_test) > 1 else None

    # Predict all rows for final CSV output
    all_predictions = model.predict(X)

    result_df = df.copy()

    result_df['predicted_score_10'] = all_predictions.round(1)
    result_df['error'] = (
        result_df['human_score_10'] - result_df['predicted_score_10']
    ).round(1)
    result_df['absolute_error'] = result_df['error'].abs().round(1)

    result_df['human_level'] = result_df['human_score_10'].apply(get_level)
    result_df['predicted_level'] = result_df['predicted_score_10'].apply(get_level)

    result_df['result_explanation'] = result_df.apply(
        lambda row: f"Human gave {row['human_score_10']}/10, ML predicted {row['predicted_score_10']}/10",
        axis=1
    )

    output_columns = [
        'Pattern',
        'pitch_score',
        'velocity_score',
        'rhythm_score',
        'timing_score',
        'human_score_10',
        'predicted_score_10',
        'error',
        'absolute_error',
        'human_level',
        'predicted_level',
        'result_explanation'
    ]

    result_df[output_columns].to_csv(OUTPUT_PREDICTION_CSV, index=False)

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("\nTraining complete.")
    print(f"Total samples used: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"MAE: {mae:.2f} marks out of 10")

    if r2 is not None:
        print(f"R2 Score: {r2:.2f}")

    print(f"\nPrediction CSV saved to: {OUTPUT_PREDICTION_CSV}")
    print(f"Model saved to: {MODEL_FILE}")

    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    print("\nSample Output:")
    print(result_df[[
        'Pattern',
        'human_score_10',
        'predicted_score_10',
        'error',
        'human_level',
        'predicted_level'
    ]].to_string(index=False))

    create_layman_graphs(result_df, importance_df, mae)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_model()