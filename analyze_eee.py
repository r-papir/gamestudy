import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents Tcl crash on macOS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path.home() / "Downloads"
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'Explore':   "#10A99A",
    'Establish': "#C75170",
    'Exploit':   "#EFDF51",
}

MANUAL_LABEL_MAP = {
    'explore':   'Explore',   'Explore':   'Explore',
    'establish': 'Establish', 'Establish': 'Establish',
    'exploit':   'Exploit',   'Exploit':   'Exploit',
}

NLP_LABEL_MAP = {
    'exploratory': 'Explore',
    'confirmatory': 'Establish',
    'exploitative': 'Exploit',
}

TRAJECTORY_MAP = {
    'mechanic':   'Mechanic',   'Mechanic':   'Mechanic',
    'objective':  'Objective',  'Objective':  'Objective',
    'unclear':    None,         'Unclear':    None,
    '--':         None,
}

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path if path else None

def normalize_manual(label):
    if pd.isna(label):
        return None
    return MANUAL_LABEL_MAP.get(str(label).strip(), None)

def normalize_nlp(label):
    if pd.isna(label):
        return None
    s = str(label).strip().lower()
    for k, v in NLP_LABEL_MAP.items():
        if k in s:
            return v
    return None

def normalize_trajectory(label):
    if pd.isna(label):
        return None
    return TRAJECTORY_MAP.get(str(label).strip(), None)

def load_data():
    print("Select your COMPLETED CODING SPREADSHEET (EEE_coding_sheet.xlsx)...")
    manual_path = select_file(
        "Select completed coding spreadsheet",
        [("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if not manual_path:
        raise SystemExit("No file selected.")

    print("Select your NLP OUTPUT FILE (classified_speech_segments.xlsx)...")
    nlp_path = select_file(
        "Select NLP output file",
        [("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if not nlp_path:
        raise SystemExit("No file selected.")

    manual_df = pd.read_excel(manual_path, sheet_name='Coding Sheet')
    nlp_df    = pd.read_excel(nlp_path, sheet_name='Confident Classifications')

    return manual_df, nlp_df

def prepare_manual(manual_df):
    df = manual_df.copy()
    df['manual_label'] = df['Model Category'].apply(normalize_manual)

    # Handle Learning Trajectory column — may be named differently
    traj_col = None
    for col in ['Learning Trajectory', 'Trajectory', 'trajectory', 'learning_trajectory']:
        if col in df.columns:
            traj_col = col
            break
    if traj_col:
        df['trajectory'] = df[traj_col].apply(normalize_trajectory)
    else:
        df['trajectory'] = None
        print("Warning: No 'Learning Trajectory' column found. Figure 4 will be skipped.")

    df = df[df['manual_label'].isin(['Explore', 'Establish', 'Exploit'])].copy()
    df['Participant ID'] = df['Participant ID'].astype(str).str.strip()
    df = df[~df['Participant ID'].isin(['P029', 'P031'])].copy()
    df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
    return df

def prepare_nlp(nlp_df):
    df = nlp_df.copy()
    cat_col = None
    for col in ['context_adjusted_category', 'auto_category', 'category']:
        if col in df.columns:
            cat_col = col
            break
    if cat_col is None:
        raise ValueError(f"Could not find category column. Available: {list(df.columns)}")

    df['nlp_label'] = df[cat_col].apply(normalize_nlp)
    df = df[df['nlp_label'].isin(['Explore', 'Establish', 'Exploit'])].copy()

    pid_col = None
    for col in ['participant_id', 'Participant ID', 'participant']:
        if col in df.columns:
            pid_col = col
            break
    if pid_col:
        df['Participant ID'] = df[pid_col].astype(str).str.strip()

    ts_col = None
    for col in ['start_time', 'Timestamp', 'timestamp']:
        if col in df.columns:
            ts_col = col
            break
    if ts_col:
        df['Timestamp'] = df[ts_col].astype(str).str.strip()

    return df

def _ts_to_seconds(ts):
    """Convert MM:SS or HH:MM:SS timestamp string to total seconds. Returns None on failure."""
    try:
        parts = str(ts).strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return None

def compute_agreement(manual_df, nlp_df, max_gap_seconds=5):
    """Match manual and NLP rows by Participant ID + game and fuzzy timestamp (≤max_gap_seconds).
    For each manual row, finds the closest NLP row within the gap window."""

    manual = manual_df[['Participant ID', 'Timestamp', 'manual_label']].copy()
    nlp    = nlp_df[['Participant ID', 'Timestamp', 'nlp_label']].copy()

    # Add game column to NLP if available (used to restrict matching within same game)
    if 'game' in nlp_df.columns:
        nlp['game'] = nlp_df['game'].astype(str).str.strip()
    else:
        nlp['game'] = ''

    manual['_ts_sec'] = manual['Timestamp'].apply(_ts_to_seconds)
    nlp['_ts_sec']    = nlp['Timestamp'].apply(_ts_to_seconds)

    rows = []
    for _, m_row in manual.iterrows():
        pid  = m_row['Participant ID']
        m_ts = m_row['_ts_sec']
        if m_ts is None:
            continue

        # Candidates: same participant, valid timestamp
        cands = nlp[(nlp['Participant ID'] == pid) & (nlp['_ts_sec'].notna())].copy()
        if cands.empty:
            continue

        cands['_diff'] = (cands['_ts_sec'] - m_ts).abs()
        best = cands.loc[cands['_diff'].idxmin()]

        if best['_diff'] <= max_gap_seconds:
            rows.append({
                'Participant ID': pid,
                'manual_ts':      m_row['Timestamp'],
                'nlp_ts':         best['Timestamp'],
                'ts_diff_sec':    best['_diff'],
                'manual_label':   m_row['manual_label'],
                'nlp_label':      best['nlp_label'],
            })

    merged = pd.DataFrame(rows)

    print(f"\nMatched {len(merged)} utterances for agreement analysis "
          f"(fuzzy ≤{max_gap_seconds}s, {len(manual)} manual rows, {len(nlp)} NLP rows)")

    if len(merged) < 10:
        print("Warning: fewer than 10 matched utterances — agreement stats may be unreliable.")

    if len(merged) == 0:
        print("No matched utterances found. Check Participant ID formats and timestamp ranges.")
        return None

    y_manual = merged['manual_label'].tolist()
    y_nlp    = merged['nlp_label'].tolist()

    kappa     = cohen_kappa_score(y_manual, y_nlp)
    pct_agree = np.mean([m == n for m, n in zip(y_manual, y_nlp)])

    print(f"\nOverall Cohen's Kappa: {kappa:.3f}")
    print(f"Overall Percent Agreement: {pct_agree:.1%}")

    cats = ['Explore', 'Establish', 'Exploit']
    print("\nPer-category agreement:")
    cat_stats = {}
    for cat in cats:
        indices = [i for i, m in enumerate(y_manual) if m == cat]
        if not indices:
            continue
        cat_manual = [y_manual[i] for i in indices]
        cat_nlp    = [y_nlp[i] for i in indices]
        cat_agree  = np.mean([m == n for m, n in zip(cat_manual, cat_nlp)])
        cat_stats[cat] = cat_agree
        print(f"  {cat}: {cat_agree:.1%} agreement (n={len(indices)})")

    cm = confusion_matrix(y_manual, y_nlp, labels=cats)
    print("\nConfusion matrix (rows=manual, cols=NLP):")
    print(pd.DataFrame(cm, index=cats, columns=cats))

    return merged, kappa, pct_agree, cat_stats

def plot_eee_distribution(manual_df):
    counts = manual_df['manual_label'].value_counts()
    cats   = ['Explore', 'Establish', 'Exploit']
    values = [counts.get(c, 0) for c in cats]
    total  = sum(values)
    pcts   = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(cats, pcts, color=[COLORS[c] for c in cats],
                  width=0.5, edgecolor='white', linewidth=1.5)

    for bar, pct, val in zip(bars, pcts, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%\n(n={val})', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_ylabel('Proportion of Utterances (%)', fontsize=12)
    ax.set_title('Distribution of EEE Reasoning States\nAcross Participants (Games A & B)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / "figure1_eee_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved Figure 1: {out}")
    plt.close()

def plot_sequential_pattern(manual_df):
    df = manual_df.copy()

    def ts_to_seconds(ts):
        try:
            parts = str(ts).strip().split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except:
            return None
        return None

    df['seconds'] = df['Timestamp'].apply(ts_to_seconds)
    df = df.dropna(subset=['seconds'])

    def assign_third(group):
        mn, mx = group['seconds'].min(), group['seconds'].max()
        if mx == mn:
            group['third'] = 'Middle'
            return group
        rng = mx - mn
        group['third'] = group['seconds'].apply(
            lambda s: 'Early' if s <= mn + rng/3
                      else ('Late' if s >= mn + 2*rng/3 else 'Middle')
        )
        return group

    df = df.groupby('Participant ID', group_keys=False).apply(assign_third)

    thirds = ['Early', 'Middle', 'Late']
    cats   = ['Explore', 'Establish', 'Exploit']

    props = {}
    for third in thirds:
        sub    = df[df['third'] == third]
        counts = sub['manual_label'].value_counts()
        total  = len(sub)
        props[third] = {c: counts.get(c, 0) / total * 100 if total > 0 else 0 for c in cats}

    fig, ax = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(thirds))
    width  = 0.5
    bottom = np.zeros(len(thirds))

    for cat in cats:
        values = [props[t][cat] for t in thirds]
        bars = ax.bar(x, values, width, bottom=bottom,
                      label=cat, color=COLORS[cat], edgecolor='white', linewidth=1)
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bottom[i] + val/2,
                        f'{val:.0f}%', ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold')
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(['Early\n(first third)', 'Middle\n(second third)', 'Late\n(final third)'],
                       fontsize=11)
    ax.set_ylabel('Proportion of Utterances (%)', fontsize=12)
    ax.set_title('EEE Reasoning States Across Transcript Thirds\n(Sequential Pattern)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "figure2_sequential_pattern.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2: {out}")
    plt.close()

def plot_agreement_by_category(cat_stats):
    cats   = list(cat_stats.keys())
    values = [cat_stats[c] * 100 for c in cats]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(cats, values, color=[COLORS[c] for c in cats],
                  width=0.5, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylabel('Agreement with Manual Coding (%)', fontsize=12)
    ax.set_title('NLP Classifier Agreement by EEE Category\n(vs. Manual Ground Truth)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / "figure3_nlp_agreement.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 3: {out}")
    plt.close()

def plot_trajectory_breakdown(manual_df):
    """Figure 4: EEE distribution broken down by learning trajectory."""
    if 'trajectory' not in manual_df.columns or manual_df['trajectory'].isna().all():
        print("Skipping Figure 4: Learning Trajectory column is blank.")
        return

    df = manual_df[manual_df['trajectory'].isin(['Mechanic', 'Objective'])].copy()

    if len(df) == 0:
        print("Skipping Figure 4: no trajectory-coded utterances found.")
        return

    trajectories = ['Mechanic', 'Objective']
    cats         = ['Explore', 'Establish', 'Exploit']

    props = {}
    ns    = {}
    for traj in trajectories:
        sub    = df[df['trajectory'] == traj]
        counts = sub['manual_label'].value_counts()
        total  = len(sub)
        ns[traj]    = total
        props[traj] = {c: counts.get(c, 0) / total * 100 if total > 0 else 0 for c in cats}

    fig, ax = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(trajectories))
    width  = 0.5
    bottom = np.zeros(len(trajectories))

    for cat in cats:
        values = [props[t][cat] for t in trajectories]
        bars = ax.bar(x, values, width, bottom=bottom,
                      label=cat, color=COLORS[cat], edgecolor='white', linewidth=1)
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bottom[i] + val/2,
                        f'{val:.0f}%', ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold')
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'Mechanic-Based\n(n={ns["Mechanic"]})', f'Objective-Based\n(n={ns["Objective"]})'],
        fontsize=11
    )
    ax.set_ylabel('Proportion of Utterances (%)', fontsize=12)
    ax.set_title('EEE Reasoning States by Learning Trajectory\n(Mechanic-Based vs. Objective-Based)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "figure4_trajectory_breakdown.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 4: {out}")
    plt.close()

def main():
    print("=" * 60)
    print("EEE ANALYSIS PIPELINE")
    print("=" * 60)

    manual_df, nlp_df = load_data()

    manual_clean = prepare_manual(manual_df)
    nlp_clean    = prepare_nlp(nlp_df)

    print(f"\nManual coded utterances (EEE only): {len(manual_clean)}")
    print(f"NLP classified utterances (EEE only): {len(nlp_clean)}")
    print(f"\nManual category breakdown:")
    print(manual_clean['manual_label'].value_counts())

    if 'trajectory' in manual_clean.columns and manual_clean['trajectory'].notna().any():
        print(f"\nLearning trajectory breakdown:")
        print(manual_clean['trajectory'].value_counts())

    # Figure 1: Overall EEE distribution
    plot_eee_distribution(manual_clean)

    # Figure 2: Sequential pattern across transcript thirds
    plot_sequential_pattern(manual_clean)

    # Figure 3: NLP agreement by category
    result = compute_agreement(manual_clean, nlp_clean)
    if result is not None:
        merged, kappa, pct_agree, cat_stats = result
        plot_agreement_by_category(cat_stats)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()