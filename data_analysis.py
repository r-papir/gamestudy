#!/usr/bin/env python3
"""
ARC Puzzle Data Analysis Program
=================================

Comprehensive statistical analysis and visualization for ARC puzzle behavioral data.

Includes:
- Completion time extraction from JSON game data
- Descriptive statistics (mean, median, SD, outliers)
- Statistical tests (Mann-Whitney U, Chi-squared, Spearman, Linear Regression)
- Knowledge-search behavior analysis (exploratory/confirmatory/exploitative)
- Age correlation analysis
- Data visualizations (histograms, boxplots, violin plots, heatmaps, scatter plots)

All outputs saved to ~/Downloads

Author: Rachel (Thesis Research)
Date: January 2026
"""

import json
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import (
    mannwhitneyu, chi2_contingency, spearmanr, pearsonr,
    shapiro, f_oneway, kruskal, wilcoxon
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# ── PLOT COLORS ──────────────────────────────────────────────────────────────
PLOT_COLORS = {
    'primary':    'cornflowerblue',   # scatter plots, paired lines, boxplots
    'histogram':  'skyblue',     # histogram bars
    'game_a':     "#BB2E4F",     # Puzzle A line/markers
    'game_b':     "#4F0E99",     # Puzzle B line/markers
    'mean_line':  'darkred',         # mean reference lines
    'median_line': 'orange',      # median reference lines
    'categorical': ['#ff9999', "#667dff", "#92d238", "#edde7d"],  # stacked bar chart
}

def set_plot_colors(primary=None, histogram=None, game_a=None, game_b=None,
                    mean_line=None, median_line=None, categorical=None, palette=None):
    """Change figure colors for all visualizations. Call before running analyses.

    Args:
        primary:     Color for scatter plots, paired lines, and boxplots (default: 'steelblue')
        histogram:   Color for histogram bars (default: 'skyblue')
        game_a:      Color for Puzzle A data (default: '#2E86AB')
        game_b:      Color for Puzzle B data (default: '#A23B72')
        mean_line:   Color for mean reference lines (default: 'red')
        median_line: Color for median reference lines (default: 'green')
        categorical: List of colors for the stacked category bar chart
        palette:     Seaborn palette name for multi-color plots (default: 'husl')

    Example:
        set_plot_colors(primary='purple', game_a='orange', game_b='teal', palette='Set2')
    """
    if primary is not None:     PLOT_COLORS['primary'] = primary
    if histogram is not None:   PLOT_COLORS['histogram'] = histogram
    if game_a is not None:      PLOT_COLORS['game_a'] = game_a
    if game_b is not None:      PLOT_COLORS['game_b'] = game_b
    if mean_line is not None:   PLOT_COLORS['mean_line'] = mean_line
    if median_line is not None: PLOT_COLORS['median_line'] = median_line
    if categorical is not None: PLOT_COLORS['categorical'] = categorical
    if palette is not None:     sns.set_palette(palette)
# ─────────────────────────────────────────────────────────────────────────────

# Output directory - always Downloads
OUTPUT_DIR = Path.home() / "Downloads"


class FileSelector:
    """Handles file/folder selection dialogs with a single Tk instance"""

    def __init__(self):
        self._root = None

    def _get_root(self):
        if self._root is None:
            self._root = tk.Tk()
            self._root.withdraw()
        return self._root

    def select_folder(self, title="Select a folder"):
        root = self._get_root()
        root.lift()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title=title, initialdir=str(OUTPUT_DIR))
        root.attributes('-topmost', False)
        return folder if folder else None

    def select_file(self, title="Select a file", filetypes=None):
        if filetypes is None:
            filetypes = [("All files", "*.*")]
        root = self._get_root()
        root.lift()
        root.attributes('-topmost', True)
        file = filedialog.askopenfilename(title=title, initialdir=str(OUTPUT_DIR), filetypes=filetypes)
        root.attributes('-topmost', False)
        return file if file else None

    def cleanup(self):
        if self._root is not None:
            self._root.destroy()
            self._root = None


# Global file selector
_file_selector = FileSelector()


def select_folder(title="Select a folder"):
    return _file_selector.select_folder(title)


def select_file(title="Select a file", filetypes=None):
    return _file_selector.select_file(title, filetypes)


def read_spreadsheet(path, sheet_name=0):
    """Read a CSV or Excel file into a DataFrame"""
    if str(path).endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


class ParticipantTracker:
    """Loads participant tracking data for mapping files to participant IDs"""

    def __init__(self, tracker_path):
        self.df = read_spreadsheet(tracker_path, sheet_name=0)
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        self.valid_pids = set()
        self.participant_info = {}
        self.quit_games = {}  # pid -> set of game letters quit, e.g. {'C'} or {'B', 'C'}

        notes_col = next((c for c in self.df.columns if str(c).startswith('Notes')), None)

        for _, row in self.df.iterrows():
            pid = re.sub(r'[^A-Za-z0-9]+$', '', str(row.get('PID:', '')).strip())
            if not pid or pid.lower() == 'nan':
                continue

            self.valid_pids.add(pid)
            quit_val = row.get('Puz. Quit:', '')
            self.participant_info[pid] = {
                'puzzle_order': row.get('Puz. Order:', ''),
                'puzzles_quit': quit_val,
                'notes': row.get(notes_col, '') if notes_col else ''
            }
            self.quit_games[pid] = self._parse_quit_games(str(quit_val))

        # P036 and P045 have gamestate files but are absent from the tracker.
        # Register them as valid participants with no quit games recorded.
        for pid in ('P036', 'P045'):
            if pid not in self.valid_pids:
                self.valid_pids.add(pid)
                self.participant_info[pid] = {'puzzle_order': '', 'puzzles_quit': '', 'notes': ''}
                self.quit_games[pid] = set()
                print(f"  Note: {pid} absent from tracker — added as valid participant (known omission)")

        print(f"  Loaded {len(self.valid_pids)} participants from tracker")

    def _parse_quit_games(self, quit_val):
        """Parse 'Puz. Quit:' value into a set of game letters, e.g. {'C'} or {'B', 'C'}"""
        if not quit_val or quit_val.lower() in ('nan', 'none', '--', ''):
            return set()
        return set(re.findall(r'\b([ABC])\b', quit_val))

    def get_quit_games(self, pid):
        """Return set of game letters (A/B/C) that this participant quit"""
        return self.quit_games.get(pid, set())

    def _extract_pid(self, filename):
        match = re.search(r'(P\d{3})', filename)
        return match.group(1) if match else None

    def get_participant_id_from_json(self, json_filename, game_type):
        pid = self._extract_pid(json_filename)
        if not pid:
            return None
        return pid if pid in self.valid_pids else None

    def get_valid_participants(self):
        return sorted(self.valid_pids)


class ARCDataAnalyzer:
    """Main analysis class for ARC puzzle behavioral data"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.participant_tracker = None
        self.demographic_df = None
        self.nlp_df = None
        self.completion_times = {'Game A': {}, 'Game B': {}}
        self.quit_levels = {'Game A': {}, 'Game B': {}}        # pid -> last level reached
        self.completion_status = {'Game A': {}, 'Game B': {}}  # pid -> 'completed' or 'withdrawn'
        self.results = {}

        print("=" * 70)
        print("ARC PUZZLE DATA ANALYSIS PROGRAM")
        print("=" * 70)
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Timestamp: {self.timestamp}")

    def load_participant_tracker(self, tracker_path=None):
        """Load participant tracker CSV"""
        if tracker_path is None:
            print("\n" + "=" * 60)
            print("FILE 1 of 3: PARTICIPANT TRACKER")
            print("=" * 60)
            print("Select the CSV file that maps Session IDs to game files.")
            print("File name example: 'Participant Tracker.xlsx'")
            print("Contains columns: Session ID, Game A Data, Game B Data, etc.")
            print("=" * 60)
            input(">>> Press ENTER to open file picker...")
            tracker_path = select_file(
                "FILE 1: Select Participant Tracker",
                filetypes=[("Excel/CSV files", "*.xlsx *.xls *.csv"), ("All files", "*.*")]
            )
        if not tracker_path:
            print("No file selected.")
            return False

        print(f"\nLoading participant tracker: {tracker_path}")
        self.participant_tracker = ParticipantTracker(tracker_path)
        return True

    def load_demographic_data(self, demographic_path=None):
        """Load demographic/consent form data"""
        if demographic_path is None:
            print("\n" + "=" * 60)
            print("FILE 2 of 3: SURVEY DATA")
            print("=" * 60)
            print("Select the Survey Data Excel file (separate from Participant Tracker).")
            print("File name example: 'Survey Data.xlsx'")
            print("Contains: Age, Gender, Video Game Enjoyment, Puzzle Enjoyment, etc.")
            print("=" * 60)
            input(">>> Press ENTER to open file picker...")
            demographic_path = select_file(
                "FILE 2: Select Survey Data",
                filetypes=[("Excel/CSV files", "*.xlsx *.xls *.csv"), ("All files", "*.*")]
            )
        if not demographic_path:
            print("No file selected.")
            return False

        print(f"\nLoading demographic data: {demographic_path}")
        self.demographic_df = read_spreadsheet(demographic_path)
        print(f"  Loaded {len(self.demographic_df)} demographic records")
        return True

    def load_nlp_classifications(self, nlp_path=None):
        """Load NLP classification results"""
        if nlp_path is None:
            print("\n" + "=" * 60)
            print("FILE 3 of 3: NLP CLASSIFICATIONS (OPTIONAL)")
            print("=" * 60)
            print("Select the NLP classification output from NLP_program.py")
            print("File name example: 'classified_speech_segments.xlsx' or")
            print("                   'final_classified_segments.xlsx'")
            print("If you don't have this yet, press Cancel to skip.")
            print("=" * 60)
            input(">>> Press ENTER to open file picker (or Cancel to skip)...")
            nlp_path = select_file(
                "FILE 3: Select NLP Classifications (Cancel to skip)",
                filetypes=[("Excel/CSV files", "*.xlsx *.xls *.csv"), ("All files", "*.*")]
            )
        if not nlp_path:
            print("No file selected - skipping NLP analysis.")
            return False

        print(f"\nLoading NLP classifications: {nlp_path}")
        self.nlp_df = read_spreadsheet(nlp_path)
        print(f"  Loaded {len(self.nlp_df)} classified segments")

        # Determine category column
        if 'final_category' in self.nlp_df.columns:
            self.category_col = 'final_category'
        elif 'speech_category' in self.nlp_df.columns:
            self.category_col = 'speech_category'
        else:
            self.category_col = 'auto_category'

        print(f"  Using category column: {self.category_col}")
        print(f"  Category breakdown:")
        print(self.nlp_df[self.category_col].value_counts())
        return True

    def extract_completion_times(self, data_dir=None):
        """Extract completion times from gamestate JSON files in a single data folder"""
        print("\n" + "=" * 50)
        print("EXTRACTING COMPLETION TIMES FROM JSON FILES")
        print("=" * 50)

        if data_dir is None:
            print("\n" + "=" * 60)
            print("DATA FOLDER")
            print("=" * 60)
            print("Select the 'Data' folder containing all gamestate JSON files.")
            print("Eyetracking and audio files will be ignored automatically.")
            print("File names look like: 'P001_gA_gamestate_03272026.json'")
            print("=" * 60)
            input(">>> Press ENTER to open folder picker...")
            data_dir = select_folder("Select Data folder")
        if not data_dir:
            print("No folder selected.")
            return False

        data_path = Path(data_dir)
        gamestate_files = [f for f in data_path.glob("*.json") if 'gamestate' in f.name.lower()]
        print(f"\nFound {len(gamestate_files)} gamestate files...")

        for json_file in gamestate_files:
            self._process_json_file(json_file)

        print(f"\n  Extracted {len(self.completion_times['Game A'])} Game A completion times")
        print(f"  Extracted {len(self.completion_times['Game B'])} Game B completion times")

        # Known data collection gap: P008's Game A session was not recorded.
        # Their Game B data is retained. This is not a code issue.
        if 'P008' not in self.completion_times['Game A']:
            print("  Note: P008 has no Game A gamestate file (session not recorded) — excluded from Game A only")

        self._determine_completion_status()
        return True

    def _determine_completion_status(self):
        """Cross-reference gamestate max levels with tracker quit data to determine completion status"""
        print("\n" + "=" * 50)
        print("COMPLETION STATUS")
        print("=" * 50)

        game_letter_map = {'Game A': 'A', 'Game B': 'B'}

        for game in ['Game A', 'Game B']:
            game_letter = game_letter_map[game]
            for pid in self.quit_levels[game]:
                quit_games = set()
                if self.participant_tracker:
                    # Also try stripping trailing asterisk (e.g. P009*)
                    quit_games = (self.participant_tracker.get_quit_games(pid) |
                                  self.participant_tracker.get_quit_games(pid.rstrip('*')))
                if game_letter in quit_games:
                    self.completion_status[game][pid] = 'withdrawn'
                else:
                    self.completion_status[game][pid] = 'completed'

            completed = [p for p, s in self.completion_status[game].items() if s == 'completed']
            withdrawn = [p for p, s in self.completion_status[game].items() if s == 'withdrawn']
            print(f"\n  {game}: {len(completed)} completed, {len(withdrawn)} withdrawn")
            for pid in withdrawn:
                level = self.quit_levels[game].get(pid)
                level_str = f" (quit at level {level})" if level is not None else ""
                print(f"    - {pid} withdrew{level_str}")

    def _process_json_file(self, json_path):
        """Process a single gamestate JSON file to extract completion time"""
        try:
            # Determine game type from filename
            name_lower = json_path.name.lower()
            if '_ga_' in name_lower:
                game = 'Game A'
            elif '_gb_' in name_lower:
                game = 'Game B'
            else:
                return

            with open(json_path, 'r') as f:
                data = json.load(f)

            session_start = data.get('sessionStart')
            movements = data.get('movements', [])

            if not session_start or not movements:
                return

            last_movement = movements[-1].get('timestamp')
            if not last_movement:
                return

            # Parse timestamps
            start_dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(last_movement.replace('Z', '+00:00'))
            completion_time = end_dt - start_dt

            # Get participant ID
            if self.participant_tracker:
                participant_id = self.participant_tracker.get_participant_id_from_json(json_path.name, game)
            else:
                match = re.search(r'(P\d{3})', json_path.name)
                participant_id = match.group(1) if match else None

            if participant_id:
                self.completion_times[game][participant_id] = completion_time

                # Extract the highest level reached from movement data
                max_level = None
                for movement in movements:
                    lvl = movement.get('level')
                    if lvl is not None:
                        if max_level is None or lvl > max_level:
                            max_level = lvl
                if max_level is not None:
                    self.quit_levels[game][participant_id] = max_level

        except Exception as e:
            print(f"  Warning: Could not process {json_path.name}: {e}")

    def compute_descriptive_statistics(self):
        """Compute descriptive statistics for completion times"""
        print("\n" + "=" * 50)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 50)

        self.results['descriptive_stats'] = {}

        for game in ['Game A', 'Game B']:
            times = [t for pid, t in self.completion_times[game].items()
                     if self.completion_status[game].get(pid) != 'withdrawn']
            if not times:
                print(f"\n  Warning: No completion times for {game}")
                continue

            times_seconds = [t.total_seconds() for t in times]

            stats_dict = {
                'count': len(times_seconds),
                'mean': np.mean(times_seconds),
                'median': np.median(times_seconds),
                'std': np.std(times_seconds),
                'min': np.min(times_seconds),
                'max': np.max(times_seconds),
                'q1': np.percentile(times_seconds, 25),
                'q3': np.percentile(times_seconds, 75)
            }

            # Outlier detection using IQR
            iqr = stats_dict['q3'] - stats_dict['q1']
            lower_bound = stats_dict['q1'] - 1.5 * iqr
            upper_bound = stats_dict['q3'] + 1.5 * iqr
            outliers = [t for t in times_seconds if t < lower_bound or t > upper_bound]
            stats_dict['outliers'] = outliers
            stats_dict['n_outliers'] = len(outliers)

            self.results['descriptive_stats'][game] = stats_dict

            print(f"\n  {game} Statistics (N={stats_dict['count']}):")
            print(f"    Mean:   {timedelta(seconds=stats_dict['mean'])}")
            print(f"    Median: {timedelta(seconds=stats_dict['median'])}")
            print(f"    SD:     {timedelta(seconds=stats_dict['std'])}")
            print(f"    Range:  {timedelta(seconds=stats_dict['min'])} - {timedelta(seconds=stats_dict['max'])}")
            print(f"    Outliers detected: {stats_dict['n_outliers']}")

    def mann_whitney_u_test(self, group1_ids, group2_ids, game='Game A',
                           group1_name="Group 1", group2_name="Group 2"):
        """Perform Mann-Whitney U test comparing two groups on completion time"""
        print(f"\n  Mann-Whitney U: {group1_name} vs {group2_name} ({game})")

        group1_times = [self.completion_times[game][pid].total_seconds()
                       for pid in group1_ids if pid in self.completion_times[game]]
        group2_times = [self.completion_times[game][pid].total_seconds()
                       for pid in group2_ids if pid in self.completion_times[game]]

        if len(group1_times) < 2 or len(group2_times) < 2:
            print(f"    Insufficient data (n1={len(group1_times)}, n2={len(group2_times)})")
            return None

        statistic, p_value = mannwhitneyu(group1_times, group2_times, alternative='two-sided')

        # Effect size: rank-biserial correlation
        n1, n2 = len(group1_times), len(group2_times)
        rank_biserial = 1 - (2 * statistic) / (n1 * n2)

        results = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_n': n1,
            'group2_n': n2,
            'group1_median': np.median(group1_times),
            'group2_median': np.median(group2_times),
            'U_statistic': statistic,
            'p_value': p_value,
            'rank_biserial': rank_biserial,
            'significant': p_value < 0.05
        }

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"    U={statistic:.2f}, p={p_value:.4f} {sig}, r={rank_biserial:.3f}")

        return results

    def chi_squared_test(self):
        """Chi-squared test: Knowledge-search categories vs Performance groups"""
        if self.nlp_df is None:
            print("  Error: NLP data not loaded")
            return None

        print("\n  Chi-squared Test: Categories vs Performance")

        # Create performance groups based on completion time quartiles
        all_times = {}
        for game in ['Game A', 'Game B']:
            for pid, time in self.completion_times[game].items():
                if pid not in all_times:
                    all_times[pid] = []
                all_times[pid].append(time.total_seconds())

        # Calculate mean completion time per participant
        mean_times = {pid: np.mean(times) for pid, times in all_times.items()}

        if len(mean_times) < 4:
            print("    Insufficient data for quartile analysis")
            return None

        # Create quartile groups
        times_series = pd.Series(mean_times)
        quartiles = pd.qcut(times_series, q=4, labels=['Fast', 'Med-Fast', 'Med-Slow', 'Slow'])
        performance_groups = quartiles.to_dict()

        # Get dominant category per participant
        participant_categories = self.nlp_df.groupby('participant_id')[self.category_col].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        )

        # Build contingency table
        contingency_data = []
        for pid in set(performance_groups.keys()) & set(participant_categories.index):
            cat = participant_categories[pid]
            perf = performance_groups[pid]
            if cat and perf and cat != 'NEEDS_MANUAL_REVIEW':
                contingency_data.append({'category': cat, 'performance': perf})

        if len(contingency_data) < 10:
            print("    Insufficient data for chi-squared test")
            return None

        contingency_df = pd.DataFrame(contingency_data)
        contingency_table = pd.crosstab(contingency_df['category'], contingency_df['performance'])

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Cramer's V
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        results = {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'cramers_v': cramers_v,
            'contingency_table': contingency_table,
            'significant': p_value < 0.05
        }

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"    X2({dof})={chi2:.3f}, p={p_value:.4f} {sig}, Cramer's V={cramers_v:.3f}")

        self.results['chi_squared'] = results
        return results

    def spearman_correlation(self):
        """Spearman correlation: Category proportions vs Efficiency rank"""
        if self.nlp_df is None:
            print("  Error: NLP data not loaded")
            return None

        print("\n  Spearman Correlation: Category Proportions vs Efficiency Rank")

        # Calculate efficiency rank (faster = rank 1)
        all_times = {}
        for game in ['Game A', 'Game B']:
            for pid, time in self.completion_times[game].items():
                if pid not in all_times:
                    all_times[pid] = []
                all_times[pid].append(time.total_seconds())

        mean_times = {pid: np.mean(times) for pid, times in all_times.items()}
        efficiency_rank = pd.Series(mean_times).rank()

        # Calculate category proportions per participant
        category_props = self.nlp_df.groupby('participant_id')[self.category_col].value_counts(normalize=True).unstack(fill_value=0)

        results = {}
        for category in ['exploratory', 'confirmatory', 'exploitative']:
            if category not in category_props.columns:
                continue

            # Get common participants
            common_pids = set(efficiency_rank.index) & set(category_props.index)
            if len(common_pids) < 5:
                continue

            ranks = [efficiency_rank[pid] for pid in common_pids]
            props = [category_props.loc[pid, category] for pid in common_pids]

            rho, p_value = spearmanr(ranks, props)

            results[category] = {
                'rho': rho,
                'p_value': p_value,
                'n': len(common_pids),
                'significant': p_value < 0.05
            }

            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"    {category}: rho={rho:.3f}, p={p_value:.4f} {sig}, n={len(common_pids)}")

        self.results['spearman'] = results
        return results

    def linear_regression_analysis(self):
        """Linear regression: Efficiency rank predicted by 'confirmatory' proportion"""
        if self.nlp_df is None:
            print("  Error: NLP data not loaded")
            return None

        print("\n  Linear Regression: Efficiency Rank ~ Confirmatory Proportion")

        # Calculate efficiency rank
        all_times = {}
        for game in ['Game A', 'Game B']:
            for pid, time in self.completion_times[game].items():
                if pid not in all_times:
                    all_times[pid] = []
                all_times[pid].append(time.total_seconds())

        mean_times = {pid: np.mean(times) for pid, times in all_times.items()}
        efficiency_rank = pd.Series(mean_times).rank()

        # Calculate confirmatory proportion
        category_props = self.nlp_df.groupby('participant_id')[self.category_col].value_counts(normalize=True).unstack(fill_value=0)

        if 'confirmatory' not in category_props.columns:
            print("    No 'confirmatory' category found")
            return None

        # Get common participants
        common_pids = list(set(efficiency_rank.index) & set(category_props.index))
        if len(common_pids) < 5:
            print("    Insufficient data")
            return None

        X = category_props.loc[common_pids, 'confirmatory'].values.reshape(-1, 1)
        y = efficiency_rank[common_pids].values

        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        coef = model.coef_[0]
        intercept = model.intercept_

        # Calculate p-value for coefficient
        n = len(y)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se = np.sqrt(mse / np.sum((X.flatten() - X.mean())**2))
        t_stat = coef / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        results = {
            'r2': r2,
            'coefficient': coef,
            'intercept': intercept,
            'p_value': p_value,
            'n': n,
            'X': X.flatten(),
            'y': y,
            'y_pred': y_pred
        }

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"    R2={r2:.3f}, coef={coef:.3f}, p={p_value:.4f} {sig}, n={n}")

        self.results['linear_regression'] = results
        return results

    def age_correlation_analysis(self):
        """
        Analyze relationship between age and completion time.

        Per game:
          - Pearson or Spearman correlation (chosen by Shapiro-Wilk normality test)
          - Linear regression: completion time ~ age (R², coefficient, p-value)
          - Descriptive statistics broken down by age group (<25, 25-34, 35-44, 45+)
        """
        if self.demographic_df is None:
            print("  Demographic data not loaded - skipping age correlation")
            return None

        print("\n  Age Correlation Analysis")

        # Find age column
        age_cols = [c for c in self.demographic_df.columns if 'age' in c.lower()]
        if not age_cols:
            print("    No age column found in demographic data")
            return None

        age_col = age_cols[0]

        # Try to find session ID column
        id_cols = [c for c in self.demographic_df.columns if 'session' in c.lower() or 'id' in c.lower()]
        id_col = id_cols[0] if id_cols else self.demographic_df.columns[0]

        age_bins = [0, 25, 35, 45, 200]
        age_labels = ['<25', '25-34', '35-44', '45+']

        results = {}
        for game in ['Game A', 'Game B']:
            age_time_data = []
            for pid, completion_time in self.completion_times[game].items():
                # Find matching demographic record
                mask = self.demographic_df[id_col].astype(str).str.contains(pid, na=False)
                if mask.any():
                    age = self.demographic_df.loc[mask, age_col].iloc[0]
                    if pd.notna(age):
                        try:
                            age_time_data.append({
                                'participant_id': pid,
                                'age': float(age),
                                'completion_time': completion_time.total_seconds()
                            })
                        except:
                            pass

            if len(age_time_data) < 5:
                print(f"    {game}: Insufficient data (n={len(age_time_data)})")
                continue

            df_age = pd.DataFrame(age_time_data)

            # --- Correlation (Pearson or Spearman) ---
            _, normality_p = shapiro(df_age['completion_time'])
            if normality_p > 0.05:
                corr, p_value = pearsonr(df_age['age'], df_age['completion_time'])
                method = "Pearson"
            else:
                corr, p_value = spearmanr(df_age['age'], df_age['completion_time'])
                method = "Spearman"

            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"    {game}: {method} r={corr:.3f}, p={p_value:.4f} {sig}, n={len(df_age)}")

            # --- Linear regression: age → completion time ---
            X_reg = df_age['age'].values.reshape(-1, 1)
            y_reg = df_age['completion_time'].values
            reg_model = LinearRegression()
            reg_model.fit(X_reg, y_reg)
            y_reg_pred = reg_model.predict(X_reg)
            r2 = r2_score(y_reg, y_reg_pred)
            coef = reg_model.coef_[0]
            intercept = reg_model.intercept_
            n_reg = len(y_reg)
            residuals = y_reg - y_reg_pred
            mse = np.sum(residuals ** 2) / (n_reg - 2) if n_reg > 2 else 0
            x_var = np.sum((X_reg.flatten() - X_reg.mean()) ** 2)
            se = np.sqrt(mse / x_var) if x_var > 0 and mse > 0 else 0
            t_stat = coef / se if se > 0 else 0
            p_reg = 2 * (1 - stats.t.cdf(abs(t_stat), n_reg - 2)) if n_reg > 2 else 1

            sig_reg = '***' if p_reg < 0.001 else '**' if p_reg < 0.01 else '*' if p_reg < 0.05 else 'ns'
            print(f"      Linear regression: R²={r2:.3f}, coef={coef:.1f} sec/year, "
                  f"p={p_reg:.4f} {sig_reg}")
            direction = 'longer' if coef > 0 else 'shorter'
            print(f"      Interpretation: Each additional year of age associated with "
                  f"{abs(coef):.1f} sec {direction} completion time")

            regression = {
                'r2': r2,
                'coefficient': coef,
                'intercept': float(intercept),
                'p_value': p_reg,
                'n': n_reg,
                'y_pred': y_reg_pred,
            }

            # --- Age group descriptive statistics ---
            df_age['age_group'] = pd.cut(
                df_age['age'], bins=age_bins, labels=age_labels, right=False
            )
            age_group_stats = {}
            print(f"      Completion time by age group ({game}):")
            for grp_label in age_labels:
                grp_df = df_age[df_age['age_group'] == grp_label]
                if len(grp_df) == 0:
                    continue
                times = grp_df['completion_time'].values
                age_group_stats[grp_label] = {
                    'n': len(times),
                    'mean_sec': float(np.mean(times)),
                    'median_sec': float(np.median(times)),
                    'std_sec': float(np.std(times)),
                }
                print(f"        {grp_label}: n={len(times)}, "
                      f"mean={timedelta(seconds=int(np.mean(times)))}, "
                      f"median={timedelta(seconds=int(np.median(times)))}")

            results[game] = {
                'method': method,
                'correlation': corr,
                'p_value': p_value,
                'n': len(df_age),
                'data': df_age,
                'regression': regression,
                'age_group_stats': age_group_stats,
            }

        self.results['age_correlation'] = results
        return results

    def enjoyment_correlation_analysis(self):
        """
        Analyze correlation between game enjoyment (Likert scales) and completion time.

        Tests:
        - Video game enjoyment vs completion time (Spearman + Linear Regression)
        - Puzzle enjoyment vs completion time (Spearman + Linear Regression)
        """
        if self.demographic_df is None:
            print("  Demographic data not loaded - skipping enjoyment correlation")
            return None

        print("\n  Enjoyment vs Completion Time Correlation Analysis")

        # Find enjoyment columns
        # Video game enjoyment: look for 'video game' in column name
        videogame_cols = [c for c in self.demographic_df.columns if 'video game' in c.lower()]

        # Puzzle enjoyment: look for the specific survey question about puzzle games
        # The column is: "How much do you enjoy puzzle games, such as crossword puzzles, sudoku, or Tetris?"
        puzzle_enjoy_cols = [c for c in self.demographic_df.columns
                           if ('puzzle game' in c.lower() or 'sudoku' in c.lower() or
                               'tetris' in c.lower() or 'crossword' in c.lower())]

        # Fallback: look for columns with 'enjoy' AND 'puzzle'
        if not puzzle_enjoy_cols:
            puzzle_enjoy_cols = [c for c in self.demographic_df.columns
                               if 'puzzle' in c.lower() and 'enjoy' in c.lower()]

        if not videogame_cols and not puzzle_enjoy_cols:
            print("    No enjoyment columns found in demographic data")
            return None

        videogame_col = videogame_cols[0] if videogame_cols else None
        puzzle_col = puzzle_enjoy_cols[0] if puzzle_enjoy_cols else None

        print(f"    Video game enjoyment column: {videogame_col}")
        print(f"    Puzzle enjoyment column: {puzzle_col}")

        # Find session ID column
        id_cols = [c for c in self.demographic_df.columns if 'session' in c.lower() or 'id' in c.lower()]
        id_col = id_cols[0] if id_cols else self.demographic_df.columns[0]

        # Calculate mean completion time for each participant
        all_times = {}
        for game in ['Game A', 'Game B']:
            for pid, time in self.completion_times[game].items():
                if pid not in all_times:
                    all_times[pid] = []
                all_times[pid].append(time.total_seconds())

        if not all_times:
            print("    No completion time data available")
            return None

        mean_times = {pid: np.mean(times) for pid, times in all_times.items()}

        results = {}

        for enjoyment_type, col in [('video_game', videogame_col), ('puzzle', puzzle_col)]:
            if col is None:
                continue

            print(f"\n    {enjoyment_type.replace('_', ' ').title()} Enjoyment:")

            # Collect data
            enjoyment_data = []
            for pid in mean_times.keys():
                mask = self.demographic_df[id_col].astype(str).str.contains(str(pid), na=False)
                if mask.any():
                    enjoyment_score = self.demographic_df.loc[mask, col].iloc[0]
                    if pd.notna(enjoyment_score):
                        try:
                            # Convert Likert score to numeric
                            score = float(enjoyment_score)
                            enjoyment_data.append({
                                'participant_id': pid,
                                'enjoyment': score,
                                'completion_time': mean_times[pid],
                                'completion_time_min': mean_times[pid] / 60  # Also store in minutes
                            })
                        except (ValueError, TypeError):
                            pass

            if len(enjoyment_data) < 5:
                print(f"      Insufficient data (n={len(enjoyment_data)})")
                continue

            df_enjoy = pd.DataFrame(enjoyment_data)

            # Spearman correlation (rank-based, appropriate for Likert scales)
            rho, p_spearman = spearmanr(df_enjoy['enjoyment'], df_enjoy['completion_time'])

            # Linear regression (enjoyment predicting completion time)
            X = df_enjoy['enjoyment'].values.reshape(-1, 1)
            y = df_enjoy['completion_time'].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            coef = model.coef_[0]
            intercept = model.intercept_

            # Calculate p-value for regression coefficient
            n = len(y)
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (n - 2) if n > 2 else 0
            x_var = np.sum((X.flatten() - X.mean())**2)
            se = np.sqrt(mse / x_var) if x_var > 0 and mse > 0 else 0
            t_stat = coef / se if se > 0 else 0
            p_regression = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if n > 2 else 1

            results[enjoyment_type] = {
                'spearman_rho': rho,
                'spearman_p': p_spearman,
                'r2': r2,
                'coefficient': coef,
                'intercept': intercept,
                'regression_p': p_regression,
                'n': n,
                'data': df_enjoy,
                'y_pred': y_pred
            }

            sig_spearman = '***' if p_spearman < 0.001 else '**' if p_spearman < 0.01 else '*' if p_spearman < 0.05 else 'ns'
            sig_reg = '***' if p_regression < 0.001 else '**' if p_regression < 0.01 else '*' if p_regression < 0.05 else 'ns'

            print(f"      Spearman: rho={rho:.3f}, p={p_spearman:.4f} {sig_spearman}")
            print(f"      Linear Regression: R2={r2:.3f}, coef={coef:.3f} sec/point, p={p_regression:.4f} {sig_reg}")
            print(f"      N={n}")
            print(f"      Interpretation: {'Higher' if coef > 0 else 'Lower'} enjoyment associated with {'longer' if coef > 0 else 'shorter'} completion times")

        self.results['enjoyment_correlation'] = results
        return results

    def analyze_nlp_by_category(self):
        """Analyze movement features by speech category (ANOVA/Kruskal-Wallis)"""
        if self.nlp_df is None:
            print("  NLP data not loaded")
            return None

        print("\n  Movement Features by Speech Category")

        features = ['movement_entropy', 'direction_changes', 'repeated_sequences',
                   'unique_positions', 'num_moves', 'num_revisits']

        # Filter to available features
        features = [f for f in features if f in self.nlp_df.columns]

        if not features:
            print("    No movement features found in NLP data")
            return None

        results = {}
        for feature in features:
            groups = {}
            for cat in ['exploratory', 'confirmatory', 'exploitative']:
                data = self.nlp_df[self.nlp_df[self.category_col] == cat][feature].dropna()
                if len(data) > 0:
                    groups[cat] = data.values

            if len(groups) < 2:
                continue

            # Kruskal-Wallis test (non-parametric ANOVA)
            group_data = list(groups.values())
            h_stat, p_value = kruskal(*group_data)

            results[feature] = {
                'H_statistic': h_stat,
                'p_value': p_value,
                'groups': {k: {'mean': v.mean(), 'std': v.std(), 'n': len(v)}
                          for k, v in groups.items()}
            }

            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"    {feature}: H={h_stat:.3f}, p={p_value:.4f} {sig}")

        self.results['nlp_anova'] = results
        return results

    def game_a_vs_game_b_comparison(self):
        """Wilcoxon signed-rank test comparing Game A and Game B completion times (paired)"""
        print("\n  Game A vs Game B Completion Time Comparison (Wilcoxon signed-rank)")

        both_pids = sorted(set(self.completion_times['Game A'].keys()) & set(self.completion_times['Game B'].keys()))

        if len(both_pids) < 5:
            print(f"    Insufficient paired data (n={len(both_pids)})")
            return None

        times_a = [self.completion_times['Game A'][pid].total_seconds() for pid in both_pids]
        times_b = [self.completion_times['Game B'][pid].total_seconds() for pid in both_pids]

        statistic, p_value = wilcoxon(times_a, times_b)

        results = {
            'n_paired': len(both_pids),
            'median_a': np.median(times_a),
            'median_b': np.median(times_b),
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"    n={len(both_pids)}, W={statistic:.2f}, p={p_value:.4f} {sig}")
        print(f"    Median Game A: {timedelta(seconds=results['median_a'])}")
        print(f"    Median Game B: {timedelta(seconds=results['median_b'])}")

        self.results['game_comparison'] = results
        return results

    def order_effects_analysis(self):
        """Mann-Whitney U test: did puzzle order affect completion time?"""
        if self.participant_tracker is None:
            print("  Participant tracker not loaded - skipping order effects analysis")
            return None

        print("\n  Order Effects Analysis (Mann-Whitney U)")

        # Split participants by whether they did Game A before Game B
        a_before_b = []
        b_before_a = []

        for pid, info in self.participant_tracker.participant_info.items():
            order = str(info.get('puzzle_order', '')).strip()
            if not order or order == 'nan':
                continue
            puzzles = [p.strip().upper() for p in order.split(',')]
            if 'A' in puzzles and 'B' in puzzles:
                if puzzles.index('A') < puzzles.index('B'):
                    a_before_b.append(pid)
                else:
                    b_before_a.append(pid)

        print(f"    A before B: n={len(a_before_b)}, B before A: n={len(b_before_a)}")

        results = {}
        for game in ['Game A', 'Game B']:
            group1 = [self.completion_times[game][pid].total_seconds()
                      for pid in a_before_b if pid in self.completion_times[game]]
            group2 = [self.completion_times[game][pid].total_seconds()
                      for pid in b_before_a if pid in self.completion_times[game]]

            if len(group1) < 3 or len(group2) < 3:
                print(f"    {game}: Insufficient data")
                continue

            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

            results[game] = {
                'n_a_before_b': len(group1),
                'n_b_before_a': len(group2),
                'median_a_before_b': np.median(group1),
                'median_b_before_a': np.median(group2),
                'U_statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"    {game}: U={statistic:.2f}, p={p_value:.4f} {sig}")

        self.results['order_effects'] = results
        return results

    def completion_time_by_speech_category(self):
        """Kruskal-Wallis test: mean completion time by dominant speech category"""
        if self.nlp_df is None:
            print("  NLP data not loaded - skipping completion time by speech category")
            return None

        print("\n  Completion Time by Dominant Speech Category (Kruskal-Wallis)")

        # Get dominant category per participant
        dominant_category = self.nlp_df.groupby('participant_id')[self.category_col].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        )

        # Get mean completion time per participant across both games
        all_times = {}
        for game in ['Game A', 'Game B']:
            for pid, time in self.completion_times[game].items():
                if pid not in all_times:
                    all_times[pid] = []
                all_times[pid].append(time.total_seconds())
        mean_times = {pid: np.mean(times) for pid, times in all_times.items()}

        # Group by dominant category
        groups = {}
        for pid in set(mean_times.keys()) & set(dominant_category.index):
            cat = dominant_category[pid]
            if cat and cat != 'NEEDS_MANUAL_REVIEW':
                groups.setdefault(cat, []).append(mean_times[pid])

        if len(groups) < 2:
            print("    Insufficient category data")
            return None

        for cat, times in sorted(groups.items()):
            print(f"    {cat}: n={len(times)}, median={timedelta(seconds=int(np.median(times)))}")

        h_stat, p_value = kruskal(*groups.values())

        results = {
            'groups': {k: {'n': len(v), 'median': np.median(v), 'mean': np.mean(v), 'data': v}
                       for k, v in groups.items()},
            'H_statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"    Kruskal-Wallis: H={h_stat:.3f}, p={p_value:.4f} {sig}")

        self.results['speech_category_completion'] = results
        return results

    def analyze_proportion_into_level(self):
        """Kruskal-Wallis test: does proportion_into_level differ across speech categories?"""
        if self.nlp_df is None:
            print("  NLP data not loaded - skipping proportion into level analysis")
            return None

        if 'proportion_into_level' not in self.nlp_df.columns:
            print("  'proportion_into_level' column not found in NLP data - skipping")
            return None

        print("\n  Proportion Into Level by Speech Category (Kruskal-Wallis)")

        df = self.nlp_df.dropna(subset=['proportion_into_level', self.category_col])
        df = df[~df[self.category_col].isin(['NEEDS_MANUAL_REVIEW', 'RA Speech', 'Unrelated'])]

        groups = {}
        for cat, grp in df.groupby(self.category_col):
            vals = grp['proportion_into_level'].tolist()
            if len(vals) >= 2:
                groups[cat] = vals

        if len(groups) < 2:
            print("    Insufficient data across categories")
            return None

        for cat, vals in sorted(groups.items()):
            print(f"    {cat}: n={len(vals)}, median={np.median(vals):.3f}, mean={np.mean(vals):.3f}")

        h_stat, p_value = kruskal(*groups.values())
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"    Kruskal-Wallis: H={h_stat:.3f}, p={p_value:.4f} {sig}")

        results = {
            'groups': {k: {'n': len(v), 'median': np.median(v), 'mean': np.mean(v), 'data': v}
                       for k, v in groups.items()},
            'H_statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        self.results['proportion_into_level'] = results
        return results

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def create_completion_time_histograms(self):
        """Create histogram visualizations for completion times"""
        print("\n  Creating completion time histograms...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, game in enumerate(['Game A', 'Game B']):
            times = [t.total_seconds() / 60 for t in self.completion_times[game].values()]
            if not times:
                continue

            ax = axes[idx]
            ax.hist(times, bins=15, color=PLOT_COLORS['histogram'], edgecolor='black', alpha=0.7)
            ax.set_xlabel('Completion Time (minutes)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{game} Completion Time Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            mean_time = np.mean(times)
            median_time = np.median(times)
            ax.axvline(mean_time, color=PLOT_COLORS['mean_line'], linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f} min')
            ax.axvline(median_time, color=PLOT_COLORS['median_line'], linestyle='--', linewidth=2, label=f'Median: {median_time:.1f} min')
            ax.legend()

        plt.tight_layout()
        output_path = OUTPUT_DIR / f'completion_time_histograms_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_game_comparison_plot(self):
        """Paired plot and boxplot comparing Game A vs Game B completion times"""
        both_pids = sorted(set(self.completion_times['Game A'].keys()) & set(self.completion_times['Game B'].keys()))
        if len(both_pids) < 2:
            return

        print("  Creating Game A vs Game B comparison plot...")

        times_a = [self.completion_times['Game A'][pid].total_seconds() / 60 for pid in both_pids]
        times_b = [self.completion_times['Game B'][pid].total_seconds() / 60 for pid in both_pids]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Paired line plot
        ax = axes[0]
        for i in range(len(both_pids)):
            ax.plot([0, 1], [times_a[i], times_b[i]], 'o-', color=PLOT_COLORS['primary'], alpha=0.4, linewidth=1)
        ax.plot([0, 1], [np.median(times_a), np.median(times_b)], 'o-', color='red',
                linewidth=3, label='Median', zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Game A', 'Game B'])
        ax.set_ylabel('Completion Time (minutes)')
        ax.set_title('Paired Completion Times', fontsize=13, fontweight='bold')
        ax.legend()

        # Boxplot
        axes[1].boxplot([times_a, times_b], labels=['Game A', 'Game B'])
        axes[1].set_ylabel('Completion Time (minutes)')
        axes[1].set_title('Completion Time Distribution by Game', fontsize=13, fontweight='bold')

        plt.suptitle('Game A vs Game B Completion Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = OUTPUT_DIR / f'game_comparison_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_order_effects_plot(self):
        """Boxplot comparing completion times by puzzle order group"""
        if not self.results.get('order_effects'):
            return

        print("  Creating order effects plot...")

        a_before_b = []
        b_before_a = []
        for pid, info in self.participant_tracker.participant_info.items():
            order = str(info.get('puzzle_order', '')).strip()
            if not order or order == 'nan':
                continue
            puzzles = [p.strip().upper() for p in order.split(',')]
            if 'A' in puzzles and 'B' in puzzles:
                if puzzles.index('A') < puzzles.index('B'):
                    a_before_b.append(pid)
                else:
                    b_before_a.append(pid)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, game in enumerate(['Game A', 'Game B']):
            group1 = [self.completion_times[game][pid].total_seconds() / 60
                      for pid in a_before_b if pid in self.completion_times[game]]
            group2 = [self.completion_times[game][pid].total_seconds() / 60
                      for pid in b_before_a if pid in self.completion_times[game]]
            if group1 and group2:
                axes[idx].boxplot([group1, group2], labels=['A before B', 'B before A'])
            axes[idx].set_ylabel('Completion Time (minutes)')
            axes[idx].set_title(f'{game}: Completion Time by Puzzle Order', fontsize=13, fontweight='bold')

        plt.suptitle('Order Effects on Completion Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = OUTPUT_DIR / f'order_effects_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_speech_category_completion_plot(self):
        """Boxplot of mean completion time by dominant speech category"""
        if self.nlp_df is None or not self.results.get('speech_category_completion'):
            return

        print("  Creating speech category completion time plot...")

        groups = self.results['speech_category_completion']['groups']
        categories = sorted(groups.keys())
        data = [np.array(groups[cat]['data']) / 60 for cat in categories]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=categories)
        plt.ylabel('Mean Completion Time (minutes)')
        plt.xlabel('Dominant Speech Category')
        plt.title('Completion Time by Dominant Speech Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = OUTPUT_DIR / f'speech_category_completion_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_nlp_boxplots(self):
        """Create boxplots of movement features by speech category"""
        if self.nlp_df is None:
            return

        print("  Creating NLP category boxplots...")

        features = ['movement_entropy', 'direction_changes', 'repeated_sequences',
                   'unique_positions', 'num_moves', 'num_revisits']
        features = [f for f in features if f in self.nlp_df.columns]

        if not features:
            return

        if len(features) < 6:
            fig, axes = plt.subplots(1, len(features), figsize=(5*len(features), 5))
            if len(features) == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

        for idx, feature in enumerate(features):
            sns.boxplot(data=self.nlp_df, x=self.category_col, y=feature, ax=axes[idx])
            axes[idx].set_title(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].tick_params(axis='x', rotation=30)

        plt.suptitle('Movement Features by Speech Category', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        output_path = OUTPUT_DIR / f'nlp_boxplots_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_feature_heatmap(self):
        """Create heatmap of mean features by category"""
        if self.nlp_df is None:
            return

        print("  Creating feature profiles heatmap...")

        features = ['movement_entropy', 'direction_changes', 'repeated_sequences',
                   'unique_positions', 'num_revisits', 'num_moves']
        features = [f for f in features if f in self.nlp_df.columns]

        if not features:
            return

        category_means = self.nlp_df.groupby(self.category_col)[features].mean()

        # Normalize
        scaler = MinMaxScaler()
        category_means_scaled = pd.DataFrame(
            scaler.fit_transform(category_means.T).T,
            index=category_means.index,
            columns=category_means.columns
        )

        # Reorder
        order = ['exploratory', 'confirmatory', 'exploitative']
        category_means_scaled = category_means_scaled.reindex([c for c in order if c in category_means_scaled.index])

        plt.figure(figsize=(10, 4))
        sns.heatmap(category_means_scaled, annot=True, fmt='.2f', cmap='RdYlGn',
                   cbar_kws={'label': 'Normalized Mean Value'},
                   linewidths=1, linecolor='white')
        plt.title('Movement Feature Profiles by Speech Category', fontsize=14, fontweight='bold')
        plt.ylabel('Speech Category', fontsize=12)
        plt.xlabel('Movement Features', fontsize=12)
        plt.tight_layout()
        output_path = OUTPUT_DIR / f'feature_heatmap_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_age_scatter_plots(self):
        """Create scatter plots for age vs completion time"""
        if 'age_correlation' not in self.results or not self.results['age_correlation']:
            return

        print("  Creating age correlation scatter plots...")

        for game, data in self.results['age_correlation'].items():
            df_age = data['data']

            plt.figure(figsize=(8, 6))
            plt.scatter(df_age['age'], df_age['completion_time'] / 60, alpha=0.6, s=100, color=PLOT_COLORS['primary'])

            # Regression line
            z = np.polyfit(df_age['age'], df_age['completion_time'] / 60, 1)
            p = np.poly1d(z)
            plt.plot(df_age['age'], p(df_age['age']), "r--", alpha=0.8, linewidth=2)

            plt.xlabel('Age (years)', fontsize=12)
            plt.ylabel('Completion Time (minutes)', fontsize=12)
            reg = data.get('regression', {})
            reg_str = (f"  |  Linear regression R²={reg['r2']:.3f}, p={reg['p_value']:.4f}"
                       if reg else "")
            plt.title(
                f'{game}: Age vs Completion Time\n'
                f'{data["method"]} r={data["correlation"]:.3f}, p={data["p_value"]:.4f}'
                f'{reg_str}',
                fontsize=13, fontweight='bold'
            )
            plt.grid(alpha=0.3)

            output_path = OUTPUT_DIR / f'age_correlation_{game.replace(" ", "_")}_{self.timestamp}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {output_path.name}")
            plt.close()

    def create_age_group_boxplot(self):
        """
        Boxplot of completion time by age group (<25, 25-34, 35-44, 45+) for each game.

        Shows median, IQR, and individual data points (strip plot overlay) so that
        small-N groups remain interpretable. Saved to Downloads.
        """
        if 'age_correlation' not in self.results or not self.results['age_correlation']:
            return

        print("  Creating age group completion time boxplot...")

        age_labels = ['<25', '25-34', '35-44', '45+']
        games_with_data = [g for g, d in self.results['age_correlation'].items()
                           if d.get('age_group_stats')]

        if not games_with_data:
            return

        fig, axes = plt.subplots(1, len(games_with_data), figsize=(7 * len(games_with_data), 6))
        if len(games_with_data) == 1:
            axes = [axes]

        for ax, game in zip(axes, games_with_data):
            data = self.results['age_correlation'][game]
            df_age = data['data'].copy()

            # Keep only age groups that have at least one participant
            present_groups = [g for g in age_labels if g in data['age_group_stats']]
            df_plot = df_age[df_age['age_group'].isin(present_groups)].copy()
            df_plot['completion_time_min'] = df_plot['completion_time'] / 60
            df_plot['age_group'] = pd.Categorical(df_plot['age_group'],
                                                   categories=present_groups, ordered=True)

            sns.boxplot(
                data=df_plot, x='age_group', y='completion_time_min',
                order=present_groups, ax=ax, color=PLOT_COLORS['primary'], width=0.5,
                flierprops=dict(marker='', linestyle='none')
            )
            sns.stripplot(
                data=df_plot, x='age_group', y='completion_time_min',
                order=present_groups, ax=ax, color='black', alpha=0.5,
                size=6, jitter=True
            )

            # Annotate n per group
            for i, grp in enumerate(present_groups):
                n = data['age_group_stats'][grp]['n']
                ax.text(i, ax.get_ylim()[0] - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f'n={n}', ha='center', va='top', fontsize=9, color='dimgray')

            reg = data.get('regression', {})
            subtitle = (f"Linear regression: R²={reg['r2']:.3f}, "
                        f"coef={reg['coefficient']:.1f} sec/yr, p={reg['p_value']:.4f}"
                        if reg else "")
            ax.set_title(f'{game}: Completion Time by Age Group\n{subtitle}',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Age Group', fontsize=11)
            ax.set_ylabel('Completion Time (minutes)', fontsize=11)

        plt.tight_layout()
        output_path = OUTPUT_DIR / f'age_group_completion_time_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_participant_distribution_chart(self):
        """Create stacked bar chart of category distribution by participant"""
        if self.nlp_df is None:
            return

        print("  Creating participant category distribution chart...")

        participant_categories = self.nlp_df.groupby(['participant_id', self.category_col]).size().unstack(fill_value=0)
        participant_categories = participant_categories.div(participant_categories.sum(axis=1), axis=0)

        # Sort by dominant category
        participant_categories = participant_categories.sort_values(
            by=participant_categories.columns.tolist(),
            ascending=False
        )

        fig, ax = plt.subplots(figsize=(14, 6))
        participant_categories.plot(kind='bar', stacked=True, ax=ax,
                                   color=PLOT_COLORS['categorical'])
        ax.set_title('Proportion of Speech Categories by Participant', fontsize=14, fontweight='bold')
        ax.set_xlabel('Participant', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = OUTPUT_DIR / f'participant_distribution_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_proportion_into_level_plot(self):
        """Violin + strip plot of proportion_into_level by speech category"""
        if self.nlp_df is None or 'proportion_into_level' not in self.nlp_df.columns:
            return
        if 'proportion_into_level' not in self.results.get('proportion_into_level', {}).get('groups', {}):
            if not self.results.get('proportion_into_level'):
                return

        print("  Creating proportion-into-level plot...")

        df = self.nlp_df.dropna(subset=['proportion_into_level', self.category_col])
        df = df[~df[self.category_col].isin(['NEEDS_MANUAL_REVIEW', 'RA Speech', 'Unrelated'])]

        if df.empty:
            return

        category_order = [c for c in ['exploratory', 'confirmatory', 'exploitative'] if c in df[self.category_col].values]

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.violinplot(data=df, x=self.category_col, y='proportion_into_level',
                       order=category_order, ax=ax, inner=None, alpha=0.4,
                       palette=sns.color_palette('husl', len(category_order)))
        sns.stripplot(data=df, x=self.category_col, y='proportion_into_level',
                      order=category_order, ax=ax, size=4, alpha=0.6, jitter=True,
                      palette=sns.color_palette('husl', len(category_order)))

        # Median markers
        groups = self.results['proportion_into_level']['groups']
        for i, cat in enumerate(category_order):
            if cat in groups:
                ax.hlines(groups[cat]['median'], i - 0.2, i + 0.2,
                          colors='black', linewidths=2.5, zorder=5)

        ax.set_xlabel('Speech Category', fontsize=12)
        ax.set_ylabel('Proportion Into Level (0 = start, 1 = end)', fontsize=12)

        res = self.results['proportion_into_level']
        sig = '***' if res['p_value'] < 0.001 else '**' if res['p_value'] < 0.01 else '*' if res['p_value'] < 0.05 else 'ns'
        ax.set_title(
            f'When During a Level Does Each Speech Category Occur?\n'
            f'Kruskal-Wallis H={res["H_statistic"]:.2f}, p={res["p_value"]:.4f} {sig}',
            fontsize=13, fontweight='bold'
        )
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(len(category_order) - 0.5, 0.51, 'midpoint', fontsize=8, color='gray', va='bottom')

        plt.tight_layout()
        output_path = OUTPUT_DIR / f'proportion_into_level_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_regression_plot(self):
        """Create scatter plot with regression line"""
        if 'linear_regression' not in self.results or not self.results['linear_regression']:
            return

        print("  Creating regression plot...")

        data = self.results['linear_regression']

        plt.figure(figsize=(8, 6))
        plt.scatter(data['X'], data['y'], alpha=0.6, s=100, color=PLOT_COLORS['primary'], label='Participants')
        plt.plot(data['X'], data['y_pred'], 'r-', linewidth=2,
                label=f'Regression line (R2={data["r2"]:.3f})')

        plt.xlabel('Proportion of Confirmatory Speech', fontsize=12)
        plt.ylabel('Efficiency Rank (1 = fastest)', fontsize=12)
        plt.title('Efficiency Rank vs Confirmatory Speech Proportion', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)

        output_path = OUTPUT_DIR / f'regression_plot_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_enjoyment_scatter_plots(self):
        """Create scatter plots for enjoyment vs completion time"""
        if 'enjoyment_correlation' not in self.results or not self.results['enjoyment_correlation']:
            return

        print("  Creating enjoyment vs completion time scatter plots...")

        enjoyment_types = list(self.results['enjoyment_correlation'].keys())

        if len(enjoyment_types) == 0:
            return

        fig, axes = plt.subplots(1, len(enjoyment_types), figsize=(7*len(enjoyment_types), 6))
        if len(enjoyment_types) == 1:
            axes = [axes]

        for idx, enjoyment_type in enumerate(enjoyment_types):
            data = self.results['enjoyment_correlation'][enjoyment_type]
            df = data['data']
            ax = axes[idx]

            # Convert completion time to minutes for readability
            completion_time_min = df['completion_time'] / 60

            # Scatter plot
            ax.scatter(df['enjoyment'], completion_time_min, alpha=0.6, s=100, color=PLOT_COLORS['primary'])

            # Regression line (convert y_pred to minutes too)
            X = df['enjoyment'].values
            y_pred_min = data['y_pred'] / 60
            sort_idx = np.argsort(X)
            ax.plot(X[sort_idx], y_pred_min[sort_idx], 'r-', linewidth=2,
                   label=f'Regression (R²={data["r2"]:.3f})')

            title_name = enjoyment_type.replace('_', ' ').title()
            ax.set_xlabel(f'{title_name} Enjoyment (Likert Scale)', fontsize=12)
            ax.set_ylabel('Completion Time (minutes)', fontsize=12)
            ax.set_title(f'{title_name} Enjoyment vs Completion Time\n'
                        f'Spearman rho={data["spearman_rho"]:.3f}, p={data["spearman_p"]:.4f}',
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / f'enjoyment_vs_completion_time_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    def create_completion_time_line_graph(self):
        """Create line graph showing completion times for each participant (Game A and B)"""
        if not self.completion_times['Game A'] and not self.completion_times['Game B']:
            print("  No completion time data for line graph")
            return

        print("  Creating completion time line graph...")

        # Get all participants who have data for either game
        all_participants = sorted(set(self.completion_times['Game A'].keys()) |
                                 set(self.completion_times['Game B'].keys()))

        if not all_participants:
            print("    No participants with completion times")
            return

        # Prepare data
        game_a_times = []
        game_b_times = []

        for pid in all_participants:
            # Game A time (in minutes)
            if pid in self.completion_times['Game A']:
                game_a_times.append(self.completion_times['Game A'][pid].total_seconds() / 60)
            else:
                game_a_times.append(np.nan)

            # Game B time (in minutes)
            if pid in self.completion_times['Game B']:
                game_b_times.append(self.completion_times['Game B'][pid].total_seconds() / 60)
            else:
                game_b_times.append(np.nan)

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(all_participants))

        # Plot both lines
        ax.plot(x, game_a_times, 'o', color=PLOT_COLORS['game_a'], markersize=8,
               label='Puzzle A', alpha=0.8)
        ax.plot(x, game_b_times, 's', color=PLOT_COLORS['game_b'], markersize=8,
               label='Puzzle B', alpha=0.8)

        # Customize the plot
        ax.set_xlabel('Participant', fontsize=12)
        ax.set_ylabel('Completion Time (minutes)', fontsize=12)
        ax.set_title('Puzzle Completion Times by Participant', fontsize=14, fontweight='bold')

        # Set x-axis labels to participant IDs
        ax.set_xticks(x)
        ax.set_xticklabels(all_participants, rotation=45, ha='right')

        ax.legend(loc='upper right', fontsize=11)
        ax.grid(alpha=0.3, axis='y')

        # Add mean lines
        mean_a = np.nanmean(game_a_times)
        mean_b = np.nanmean(game_b_times)
        ax.axhline(y=mean_a, color=PLOT_COLORS['game_a'], linestyle='--', alpha=0.5,
                  label=f'Puzzle A Mean: {mean_a:.1f} min')
        ax.axhline(y=mean_b, color=PLOT_COLORS['game_b'], linestyle='--', alpha=0.5,
                  label=f'Puzzle B Mean: {mean_b:.1f} min')

        # Update legend to include means
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        output_path = OUTPUT_DIR / f'completion_time_by_participant_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path.name}")
        plt.close()

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 50)
        print("GENERATING SUMMARY REPORT")
        print("=" * 50)

        report_path = OUTPUT_DIR / f'analysis_summary_report_{self.timestamp}.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ARC PUZZLE BEHAVIORAL ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Participant info
            if self.participant_tracker:
                valid = self.participant_tracker.get_valid_participants()
                f.write(f"Valid Participants: N = {len(valid)}\n")

            # Descriptive Statistics
            f.write("\n" + "=" * 80 + "\n")
            f.write("DESCRIPTIVE STATISTICS - COMPLETION TIMES\n")
            f.write("=" * 80 + "\n")

            if 'descriptive_stats' in self.results:
                for game, stats in self.results['descriptive_stats'].items():
                    f.write(f"\n{game}:\n")
                    f.write(f"  N = {stats['count']}\n")
                    f.write(f"  Mean: {timedelta(seconds=stats['mean'])}\n")
                    f.write(f"  Median: {timedelta(seconds=stats['median'])}\n")
                    f.write(f"  SD: {timedelta(seconds=stats['std'])}\n")
                    f.write(f"  Range: {timedelta(seconds=stats['min'])} - {timedelta(seconds=stats['max'])}\n")
                    f.write(f"  IQR: Q1={timedelta(seconds=stats['q1'])}, Q3={timedelta(seconds=stats['q3'])}\n")
                    f.write(f"  Outliers: {stats['n_outliers']} detected\n")

            # NLP Classification Summary
            if self.nlp_df is not None:
                f.write("\n" + "=" * 80 + "\n")
                f.write("NLP SPEECH CLASSIFICATION SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"\nTotal segments: {len(self.nlp_df)}\n")
                f.write(f"Participants: {self.nlp_df['participant_id'].nunique()}\n")
                f.write(f"\nCategory breakdown:\n{self.nlp_df[self.category_col].value_counts()}\n")

            # Statistical Tests
            f.write("\n" + "=" * 80 + "\n")
            f.write("STATISTICAL TESTS\n")
            f.write("=" * 80 + "\n")

            if 'chi_squared' in self.results and self.results['chi_squared']:
                res = self.results['chi_squared']
                f.write(f"\nChi-squared Test (Categories vs Performance):\n")
                f.write(f"  X2({res['dof']}) = {res['chi2']:.3f}\n")
                f.write(f"  p-value = {res['p_value']:.4f}\n")
                f.write(f"  Cramer's V = {res['cramers_v']:.3f}\n")
                f.write(f"  Significant: {'Yes' if res['significant'] else 'No'}\n")

            if 'spearman' in self.results and self.results['spearman']:
                f.write(f"\nSpearman Correlations (Category Proportions vs Efficiency):\n")
                for cat, res in self.results['spearman'].items():
                    f.write(f"  {cat}: rho={res['rho']:.3f}, p={res['p_value']:.4f}, n={res['n']}\n")

            if 'linear_regression' in self.results and self.results['linear_regression']:
                res = self.results['linear_regression']
                f.write(f"\nLinear Regression (Efficiency ~ Confirmatory Proportion):\n")
                f.write(f"  R2 = {res['r2']:.3f}\n")
                f.write(f"  Coefficient = {res['coefficient']:.3f}\n")
                f.write(f"  p-value = {res['p_value']:.4f}\n")
                f.write(f"  N = {res['n']}\n")

            if 'age_correlation' in self.results and self.results['age_correlation']:
                f.write(f"\nAge vs Completion Time:\n")
                for game, res in self.results['age_correlation'].items():
                    sig = ('***' if res['p_value'] < 0.001 else '**' if res['p_value'] < 0.01
                           else '*' if res['p_value'] < 0.05 else 'ns')
                    f.write(f"\n  {game} (N={res['n']}):\n")
                    f.write(f"    Correlation: {res['method']} r={res['correlation']:.3f}, "
                            f"p={res['p_value']:.4f} {sig}\n")
                    reg = res.get('regression', {})
                    if reg:
                        sig_reg = ('***' if reg['p_value'] < 0.001 else '**' if reg['p_value'] < 0.01
                                   else '*' if reg['p_value'] < 0.05 else 'ns')
                        direction = 'longer' if reg['coefficient'] > 0 else 'shorter'
                        f.write(f"    Linear Regression: R²={reg['r2']:.3f}, "
                                f"coef={reg['coefficient']:.1f} sec/year, "
                                f"p={reg['p_value']:.4f} {sig_reg}\n")
                        f.write(f"    Interpretation: Each year of age associated with "
                                f"{abs(reg['coefficient']):.1f} sec {direction} completion time\n")
                    age_groups = res.get('age_group_stats', {})
                    if age_groups:
                        f.write(f"    Completion time by age group:\n")
                        for grp, gstats in age_groups.items():
                            f.write(f"      {grp}: n={gstats['n']}, "
                                    f"mean={timedelta(seconds=int(gstats['mean_sec']))}, "
                                    f"median={timedelta(seconds=int(gstats['median_sec']))}, "
                                    f"SD={timedelta(seconds=int(gstats['std_sec']))}\n")

            if 'enjoyment_correlation' in self.results and self.results['enjoyment_correlation']:
                f.write(f"\nEnjoyment vs Completion Time Correlations:\n")
                for enjoyment_type, res in self.results['enjoyment_correlation'].items():
                    title = enjoyment_type.replace('_', ' ').title()
                    f.write(f"\n  {title} Enjoyment:\n")
                    f.write(f"    Spearman: rho={res['spearman_rho']:.3f}, p={res['spearman_p']:.4f}\n")
                    f.write(f"    Linear Regression: R2={res['r2']:.3f}, coef={res['coefficient']:.3f} sec/point, p={res['regression_p']:.4f}\n")
                    f.write(f"    N={res['n']}\n")
                    sig = '*' if res['spearman_p'] < 0.05 else ''
                    direction = 'longer' if res['coefficient'] > 0 else 'shorter'
                    f.write(f"    Interpretation: {'Significant' if sig else 'Not significant'} relationship - higher {title.lower()} enjoyment associated with {direction} completion times\n")

            if 'nlp_anova' in self.results and self.results['nlp_anova']:
                f.write(f"\nKruskal-Wallis Tests (Movement Features by Category):\n")
                for feature, res in self.results['nlp_anova'].items():
                    sig = '*' if res['p_value'] < 0.05 else ''
                    f.write(f"  {feature}: H={res['H_statistic']:.3f}, p={res['p_value']:.4f}{sig}\n")
                    for cat, stats in res['groups'].items():
                        f.write(f"    {cat}: M={stats['mean']:.3f}, SD={stats['std']:.3f}, n={stats['n']}\n")

            if 'proportion_into_level' in self.results and self.results['proportion_into_level']:
                res = self.results['proportion_into_level']
                f.write(f"\nProportion Into Level by Speech Category:\n")
                f.write(f"  Kruskal-Wallis: H={res['H_statistic']:.3f}, p={res['p_value']:.4f}\n")
                f.write(f"  Significant: {'Yes' if res['significant'] else 'No'}\n")
                for cat, stats in sorted(res['groups'].items()):
                    f.write(f"  {cat}: median={stats['median']:.3f}, mean={stats['mean']:.3f}, n={stats['n']}\n")
                if res['significant']:
                    # Find earliest and latest category by median
                    earliest = min(res['groups'], key=lambda c: res['groups'][c]['median'])
                    latest   = max(res['groups'], key=lambda c: res['groups'][c]['median'])
                    f.write(f"  Interpretation: Categories differ significantly in when they occur "
                            f"within a level (p={res['p_value']:.4f}). '{earliest}' utterances tend "
                            f"to occur earliest (median={res['groups'][earliest]['median']:.2f}) and "
                            f"'{latest}' utterances latest (median={res['groups'][latest]['median']:.2f}), "
                            f"where 0.0 = start of level and 1.0 = end of level.\n")
                else:
                    f.write(f"  Interpretation: No significant difference in when categories occur "
                            f"within a level (p={res['p_value']:.4f}). Speech category timing is "
                            f"roughly evenly distributed across levels.\n")

            # =================================================================
            # INTERPRETATION GUIDE
            # =================================================================
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("How to read every statistic and plot in this analysis\n")
            f.write("=" * 80 + "\n")

            f.write("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESCRIPTIVE STATISTICS (completion_time_histograms, completion_time_by_participant)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Mean      — The average completion time across all participants. Sensitive to
              outliers (one very slow participant pulls it up).

  Median    — The middle value when times are sorted. More robust than the mean
              when there are outliers. If the median is much lower than the mean,
              a few slow participants are skewing the average.

  SD        — Standard deviation. Tells you how spread out the times are. A large
              SD means participants varied a lot; a small SD means they were
              similar.

  IQR (Q1–Q3) — The range covering the middle 50% of participants. Q1 is the
              25th percentile, Q3 is the 75th. A wide IQR means high variability
              even among typical participants.

  Outliers  — Detected using the 1.5×IQR rule (values below Q1−1.5×IQR or above
              Q3+1.5×IQR). Not necessarily errors — just unusually fast or slow
              participants worth noting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GAME A vs GAME B COMPARISON (game_comparison plot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Mann-Whitney U Test — A non-parametric test comparing whether one game tends
              to produce faster or slower times than the other, without assuming
              a normal distribution. Use this instead of a t-test when data are
              skewed or the sample is small.

  U statistic — The raw test statistic. On its own it's hard to interpret;
              focus on the p-value and effect size instead.

  p-value   — The probability of seeing a difference this large by chance if
              the two games were actually equivalent. p < 0.05 is the conventional
              threshold for "statistically significant."

  Effect size (r) — How large the difference is in practical terms, independent
              of sample size. r ≈ 0.1 = small, r ≈ 0.3 = medium, r ≈ 0.5 = large.
              A result can be statistically significant but have a tiny effect size
              (especially with large N), so always check both.

  Paired line plot — Each line connects one participant's Game A time to their
              Game B time. Lines going up = slower on B; lines going down = faster
              on B. The thick red line shows the median trajectory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORDER EFFECTS (order_effects plot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  This tests whether the order participants played the games (A first vs B first)
  affected their completion times — i.e., whether practice or fatigue carried
  over between puzzles.

  Mann-Whitney U — Same interpretation as above, but here the groups are "played
              Game X first" vs "played Game X second."

  A significant result means order mattered — participants who played a game
  second performed differently (likely faster due to learning transfer, or slower
  due to fatigue).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHI-SQUARED TEST (speech_category_completion plot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Tests whether the distribution of speech categories (exploratory / confirmatory
  / exploitative) differs between fast and slow performers. In other words: do
  fast solvers talk differently than slow solvers?

  X² statistic — Larger values indicate more deviation from what you'd expect if
              category use and performance were unrelated.

  p-value   — As above, p < 0.05 = the association is unlikely due to chance.

  Cramer's V — Effect size for chi-squared, ranging 0–1. V < 0.1 = negligible,
              V ≈ 0.3 = moderate, V > 0.5 = strong.

  Degrees of freedom (dof) — Related to the number of categories/groups. Reported
              for completeness; focus on p and Cramer's V.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPEARMAN CORRELATION (category proportions vs efficiency)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Measures whether participants who use more of a given speech category tend to
  be faster or slower solvers, without assuming a linear relationship.

  rho (ρ)   — Ranges from −1 to +1. Positive = more of this category → faster;
              negative = more of this category → slower. Values near 0 = no
              relationship.

  p-value   — Whether the correlation is statistically significant.

  Use Spearman (not Pearson) when your data may not be normally distributed or
  when you care about rank order rather than exact values.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LINEAR REGRESSION (regression plot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Models the relationship between confirmatory speech proportion (predictor) and
  efficiency rank (outcome) as a straight line.

  R²        — "R-squared." The proportion of variance in efficiency explained by
              the predictor. R²=0.25 means the predictor accounts for 25% of the
              differences in efficiency across participants. Higher is better, but
              even modest R² can be meaningful in behavioral research.

  Coefficient — The slope of the line. E.g., coefficient=−0.4 means each 10%
              increase in confirmatory speech proportion is associated with a 0.04
              decrease in efficiency rank (where lower rank = faster).

  p-value   — Whether the slope is significantly different from zero.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGE CORRELATIONS (age_correlation scatter plots, age_group_completion_time plot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Tests whether older participants took longer (or shorter) to complete puzzles.

  Spearman r — Same as above. A positive value means older → slower; negative
              means older → faster.

  Linear regression coefficient — E.g., "12.3 sec/year" means each additional
              year of age is associated with ~12 more seconds on that puzzle, on
              average.

  Age group boxplot — Splits participants into age brackets and shows the
              distribution of completion times per bracket. The horizontal line
              inside each box is the median; the box spans Q1–Q3; dots are
              individual participants.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENJOYMENT CORRELATIONS (enjoyment_vs_completion_time plot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Tests whether how much participants enjoyed the puzzle predicted their
  completion time.

  Spearman rho — As above. A positive value means higher enjoyment → longer
              time (perhaps engaged/absorbed); negative means higher enjoyment
              → faster time.

  The scatter plots show each participant as a dot, with a regression line
  overlaid. The R² in the legend tells you how well enjoyment predicts time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NLP SPEECH CATEGORIES (participant_distribution chart, nlp_boxplots)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Speech is classified into three EEE categories:

  Exploratory   — Orienting, uncertain, gathering information without a clear
                  hypothesis. ("what does this do?", "I'm not sure...")

  Confirmatory  — Testing a specific hypothesis. ("if I do X, then Y should
                  happen...", "let me check if...")

  Exploitative  — Applying a confirmed rule strategically toward the goal.
                  ("I know I need to...", "just have to...")

  Participant distribution chart — Stacked bar showing each participant's
              proportion of speech in each category. Useful for seeing individual
              differences at a glance.

  NLP boxplots — Show how movement features (e.g., entropy, direction changes)
              differ across speech categories. These link verbal behavior to
              physical puzzle-solving behavior.

  Kruskal-Wallis (movement features by category) — Tests whether movement
              features differ significantly across speech categories. H is the
              test statistic; p < 0.05 means at least one category differs.
              Individual feature rows tell you which features drive the difference.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROPORTION INTO LEVEL (proportion_into_level plot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  For each utterance, this value captures how far through the current level it
  occurred: 0.0 = very beginning, 1.0 = right at the end.

  Violin plot — The width of the shape at any height shows how many utterances
              occurred at that proportion. A wide base means lots of speech early
              in the level; a wide top means lots of speech late.

  Strip plot (dots) — Each dot is one utterance. Overlaid on the violin to show
              the actual data distribution and sample size.

  Black horizontal bar — The median for that category. The most important
              summary value to compare across categories.

  Dashed midpoint line — Marks the 0.5 midpoint (halfway through the level) for
              visual reference.

  Kruskal-Wallis — Tests whether the three categories differ significantly in
              when they occur. A significant result (p < 0.05) supports the
              hypothesis that explore speech clusters early, establish speech in
              the middle, and exploit speech late in a level — consistent with
              the EEE model of knowledge-search behavior.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNIFICANCE NOTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ***  p < 0.001  (very strong evidence against the null hypothesis)
  **   p < 0.01   (strong evidence)
  *    p < 0.05   (conventional significance threshold)
  ns   p ≥ 0.05   (not statistically significant — could be due to chance)

  Important: "not significant" does not mean "no effect." With small samples,
  real effects can fail to reach significance. Always report effect sizes
  alongside p-values.

""")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"  Saved: {report_path.name}")
        return report_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_analysis():
    """Run the complete data analysis pipeline with file picker dialogs"""

    analyzer = ARCDataAnalyzer()

    # Load data
    print("\n" + "=" * 50)
    print("STEP 1: LOADING DATA FILES")
    print("=" * 50)

    if not analyzer.load_participant_tracker():
        return None

    if not analyzer.load_demographic_data():
        print("Continuing without demographic data...")

    if not analyzer.load_nlp_classifications():
        print("Continuing without NLP data...")

    if not analyzer.extract_completion_times():
        print("Error: Could not extract completion times")
        return None

    # Compute statistics
    print("\n" + "=" * 50)
    print("STEP 2: STATISTICAL ANALYSIS")
    print("=" * 50)

    analyzer.compute_descriptive_statistics()
    analyzer.game_a_vs_game_b_comparison()
    analyzer.order_effects_analysis()
    analyzer.chi_squared_test()
    analyzer.spearman_correlation()
    analyzer.linear_regression_analysis()
    analyzer.age_correlation_analysis()
    analyzer.enjoyment_correlation_analysis()
    analyzer.analyze_nlp_by_category()
    analyzer.completion_time_by_speech_category()
    analyzer.analyze_proportion_into_level()

    # Create visualizations
    print("\n" + "=" * 50)
    print("STEP 3: CREATING VISUALIZATIONS")
    print("=" * 50)

    analyzer.create_completion_time_histograms()
    analyzer.create_game_comparison_plot()
    analyzer.create_order_effects_plot()
    analyzer.create_completion_time_line_graph()
    analyzer.create_nlp_boxplots()
    analyzer.create_speech_category_completion_plot()
    analyzer.create_feature_heatmap()
    analyzer.create_age_scatter_plots()
    analyzer.create_age_group_boxplot()
    analyzer.create_enjoyment_scatter_plots()
    analyzer.create_participant_distribution_chart()
    analyzer.create_proportion_into_level_plot()
    analyzer.create_regression_plot()

    # Generate report
    analyzer.generate_summary_report()

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Timestamp: {analyzer.timestamp}")

    # Cleanup
    _file_selector.cleanup()

    return analyzer


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Automatically run the full analysis when script is executed
    run_full_analysis()
    print("  analyzer.compute_descriptive_statistics()")
    print("  # ... etc")
