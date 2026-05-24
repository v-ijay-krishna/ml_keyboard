import pandas as pd

CSV_FILE = 'midi/ML_SCORES.csv'

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

if 'human_score_10' not in df.columns:
    df['human_score_10'] = ''
else:
    df['human_score_10'] = ''

df.to_csv(CSV_FILE, index=False)

print("human_score_10 column cleared successfully.")
print("Now fill real human scores manually after listening.")