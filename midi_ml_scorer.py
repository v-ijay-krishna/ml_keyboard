import pretty_midi
import os
import glob
import csv
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MASTER_FILE = 'midi/master/master.mid'
DATA_DIR    = 'midi/suriya'
OUTPUT_FILE = 'midi/ML_SCORES.csv'

TEMPO_BPM = 80
BEAT_SEC  = 60.0 / TEMPO_BPM

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────

def extract_features(notes):
    pitches = np.array([n.pitch for n in notes], dtype=float)
    velocities = np.array([n.velocity for n in notes], dtype=float)
    starts_b = np.array([n.start / BEAT_SEC for n in notes], dtype=float)
    ioi_b = np.diff(starts_b) if len(starts_b) > 1 else np.array([0.0])

    grid = 0.25
    snap_err = np.abs(starts_b % grid - grid / 2)
    tightness = float(np.clip(1.0 - np.mean(snap_err) / (grid / 2), 0, 1))

    return {
        'pitches': pitches,
        'velocities': velocities,
        'starts_b': starts_b,
        'ioi_b': ioi_b,
        'timing_tightness': tightness,
        'note_count': len(notes),
        'pitch_mean': float(np.mean(pitches)),
        'velocity_mean': float(np.mean(velocities)),
    }

# ─────────────────────────────────────────────
# SCORERS
# ─────────────────────────────────────────────

def _pad(a, b):
    n = max(len(a), len(b))
    return np.pad(a, (0, n - len(a))), np.pad(b, (0, n - len(b)))

def _cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

def score_pitch(f, r):
    p, rp = _pad(
        np.diff(f['pitches']) if len(f['pitches']) > 1 else np.array([0.0]),
        np.diff(r['pitches']) if len(r['pitches']) > 1 else np.array([0.0])
    )

    cos = (_cosine(p, rp) + 1) / 2
    close = max(0.0, 1.0 - abs(f['pitch_mean'] - r['pitch_mean']) / 24.0)

    return 0.6 * cos + 0.4 * close

def score_velocity(f, r):
    v, rv = _pad(f['velocities'], r['velocities'])

    cos = (_cosine(v, rv) + 1) / 2
    close = max(0.0, 1.0 - abs(f['velocity_mean'] - r['velocity_mean']) / 64.0)

    return 0.5 * cos + 0.5 * close

def score_rhythm(f, r):
    bins = [0, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0]

    h1, _ = np.histogram(f['ioi_b'], bins=bins)
    h2, _ = np.histogram(r['ioi_b'], bins=bins)

    h1 = h1.astype(float)
    h2 = h2.astype(float)

    if h1.sum() > 0:
        h1 /= h1.sum()

    if h2.sum() > 0:
        h2 /= h2.sum()

    return float(np.sum(np.sqrt(h1 * h2)))

def score_timing(f, r):
    phase_diff = abs(f['starts_b'][0] % 1.0 - r['starts_b'][0] % 1.0)
    phase_diff = min(phase_diff, 1.0 - phase_diff)

    return 0.6 * f['timing_tightness'] + 0.4 * (1.0 - phase_diff)

def compute_score(f, r):
    sp = score_pitch(f, r)
    sv = score_velocity(f, r)
    sr = score_rhythm(f, r)
    st = score_timing(f, r)

    return {
        'pitch_score': round(sp * 100, 1),
        'velocity_score': round(sv * 100, 1),
        'rhythm_score': round(sr * 100, 1),
        'timing_score': round(st * 100, 1),
    }

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_all_notes(midi_file_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_file_path)

        notes = []

        for inst in midi.instruments:
            notes.extend(inst.notes)

        return sorted(notes, key=lambda n: n.start)

    except Exception as e:
        print(f"[ERROR] Could not load {midi_file_path}: {e}")
        return []

def save_csv(path, label, notes):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)

        w.writerow([
            'Label',
            'Pitch',
            'Note_Name',
            'Start_s',
            'Start_beat',
            'Duration_s',
            'Duration_beat',
            'Velocity'
        ])

        for n in notes:
            w.writerow([
                label,
                n.pitch,
                pretty_midi.note_number_to_name(n.pitch),
                round(n.start, 4),
                round(n.start / BEAT_SEC, 4),
                round(n.end - n.start, 4),
                round((n.end - n.start) / BEAT_SEC, 4),
                n.velocity
            ])

# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run():
    print("\n── Initializing MIDI Feature Scorer ──")

    # 1. Load master reference
    if not os.path.exists(MASTER_FILE):
        print(f"[FATAL] Master file not found at: {MASTER_FILE}")
        return

    ref_notes = get_all_notes(MASTER_FILE)

    if not ref_notes:
        print("[FATAL] Master file has 0 notes.")
        return

    ref_feat = extract_features(ref_notes)

    print(f"[REFERENCE] Loaded master.mid ({len(ref_notes)} notes)\n")

    # Save master note-level CSV
    save_csv(MASTER_FILE.replace('.mid', '.csv'), 'master_reference', ref_notes)

    # 2. Locate student MIDI files
    if not os.path.exists(DATA_DIR):
        print(f"[FATAL] Data directory not found at: {DATA_DIR}")
        return

    midi_files = glob.glob(os.path.join(DATA_DIR, '*.mid'))

    if not midi_files:
        print(f"[WARN] No MIDI files found in {DATA_DIR}")
        return

    results = []

    print(f"  {'Data File':<18} {'Notes':>5}  {'Pitch':>6}  {'Vel':>6}  {'Rhythm':>7}  {'Timing':>7}")
    print("  " + "─" * 56)

    # 3. Score every MIDI file
    for filepath in midi_files:
        filename = os.path.basename(filepath)
        label = filename.replace('.mid', '')

        notes = get_all_notes(filepath)

        if not notes:
            continue

        # Save note-level CSV for this file
        csv_path = os.path.join(DATA_DIR, f"{label}.csv")
        save_csv(csv_path, f"surya_{label}", notes)

        # Extract features and calculate parameter scores
        feat = extract_features(notes)
        sc = compute_score(feat, ref_feat)

        results.append({
            'Pattern': filename,
            'Notes': feat['note_count'],
            'pitch_score': sc['pitch_score'],
            'velocity_score': sc['velocity_score'],
            'rhythm_score': sc['rhythm_score'],
            'timing_score': sc['timing_score'],
            'human_score_10': ''
        })

        print(f"  {filename:<18} {feat['note_count']:>5}  "
              f"{sc['pitch_score']:>6.1f}  {sc['velocity_score']:>6.1f}  "
              f"{sc['rhythm_score']:>7.1f}  {sc['timing_score']:>7.1f}")

    if not results:
        print("[WARN] No valid notes found in data tracks.")
        return

    # 4. Sort and save final CSV
    def natural_sort_key(x):
        name = x['Pattern'].replace('.mid', '')
        return int(name) if name.isdigit() else name

    results.sort(key=natural_sort_key)

    fields = [
        'Pattern',
        'Notes',
        'pitch_score',
        'velocity_score',
        'rhythm_score',
        'timing_score',
        'human_score_10'
    ]

    with open(OUTPUT_FILE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print("\n" + "─" * 60)
    print(f"  Scored {len(results)} files successfully.")
    print(f"  Final CSV saved to: {OUTPUT_FILE}")
    print("  Fill human_score_10 manually after listening.")
    print("─" * 60 + "\n")

if __name__ == "__main__":
    run()