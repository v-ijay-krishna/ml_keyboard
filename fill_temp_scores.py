import pandas as pd

CSV_FILE = 'midi/ML_SCORES.csv'

df = pd.read_csv(CSV_FILE)

df.columns = df.columns.str.strip()

df['human_score_10'] = (
    0.35 * df['pitch_score'] +
    0.20 * df['velocity_score'] +
    0.20 * df['rhythm_score'] +
    0.25 * df['timing_score']
) / 10

df['human_score_10'] = df['human_score_10'].round(1)

df.to_csv(CSV_FILE, index=False)

print("Temporary human_score_10 added successfully.")
print(df[['Pattern', 'human_score_10']])