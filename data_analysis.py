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
    shapiro, f_oneway, kruskal
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


class ParticipantTracker:
    """Loads participant tracking data for mapping files to participant IDs"""

    def __init__(self, tracker_csv_path):
        self.df = pd.read_csv(tracker_csv_path)
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        self.game_a_lookup = {}
        self.game_b_lookup = {}
        self.excluded_participants = set()
        self.participant_info = {}

        for _, row in self.df.iterrows():
            session_id = str(row.get('Session ID:', '')).strip()
            if not session_id or session_id == 'nan':
                continue

            if 'exclude' in session_id.lower():
                match = re.search(r'(P\d+)', session_id)
                if match:
                    self.excluded_participants.add(match.group(1))
                continue

            participant_id = session_id.split()[0] if ' ' in session_id else session_id

            # Store participant info
            self.participant_info[participant_id] = {
                'game_order': row.get('Game Order:', ''),
                'games_quit': row.get('Games Quit:', ''),
                'notes': row.get('Notes:', '')
            }

            # Game A data file
            game_a_file = str(row.get('Game A Data (file name):', '')).strip()
            if game_a_file and game_a_file != 'nan' and game_a_file != '--' and not game_a_file.startswith('GameState'):
                timestamp = self._extract_timestamp(game_a_file)
                if timestamp:
                    self.game_a_lookup[timestamp] = participant_id

            # Game B data file
            game_b_file = str(row.get('Game B Data (file name):', '')).strip()
            if game_b_file and game_b_file != 'nan' and game_b_file != '--' and not game_b_file.startswith('GameState'):
                timestamp = self._extract_timestamp(game_b_file)
                if timestamp:
                    self.game_b_lookup[timestamp] = participant_id

        print(f"  Loaded {len(self.game_a_lookup)} Game A mappings")
        print(f"  Loaded {len(self.game_b_lookup)} Game B mappings")
        print(f"  Excluded participants: {sorted(self.excluded_participants)}")

    def _extract_timestamp(self, filename):
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    def get_participant_id_from_json(self, json_filename, game_type):
        timestamp = self._extract_timestamp(json_filename)
        if not timestamp:
            return None
        if game_type == 'Game A':
            return self.game_a_lookup.get(timestamp)
        elif game_type == 'Game B':
            return self.game_b_lookup.get(timestamp)
        return None

    def is_excluded(self, participant_id):
        return participant_id in self.excluded_participants

    def get_valid_participants(self):
        all_participants = set(self.game_a_lookup.values()) | set(self.game_b_lookup.values())
        return sorted(all_participants - self.excluded_participants)


class ARCDataAnalyzer:
    """Main analysis class for ARC puzzle behavioral data"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.participant_tracker = None
        self.demographic_df = None
        self.nlp_df = None
        self.completion_times = {'Game A': {}, 'Game B': {}}
        self.results = {}

        print("=" * 70)
        print("ARC PUZZLE DATA ANALYSIS PROGRAM")
        print("=" * 70)
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Timestamp: {self.timestamp}")

    def load_participant_tracker(self, tracker_path=None):
        """Load participant tracker CSV"""
        if tracker_path is None:
            print("\nPlease select the PARTICIPANT TRACKER CSV file...")
            tracker_path = select_file(
                "Select Participant Tracker CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
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
            print("\nPlease select the PARTICIPANT FORM DATA (demographics) CSV file...")
            demographic_path = select_file(
                "Select Participant Form Data CSV (demographics)",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        if not demographic_path:
            print("No file selected.")
            return False

        print(f"\nLoading demographic data: {demographic_path}")
        self.demographic_df = pd.read_csv(demographic_path)
        print(f"  Loaded {len(self.demographic_df)} demographic records")
        return True

    def load_nlp_classifications(self, nlp_path=None):
        """Load NLP classification results"""
        if nlp_path is None:
            print("\nPlease select the NLP CLASSIFICATIONS CSV file...")
            print("(This is the output from NLP_program.py)")
            nlp_path = select_file(
                "Select NLP Classifications CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        if not nlp_path:
            print("No file selected.")
            return False

        print(f"\nLoading NLP classifications: {nlp_path}")
        self.nlp_df = pd.read_csv(nlp_path)
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

    def extract_completion_times(self, game_a_dir=None, game_b_dir=None):
        """Extract completion times from JSON game data files"""
        print("\n" + "=" * 50)
        print("EXTRACTING COMPLETION TIMES FROM JSON FILES")
        print("=" * 50)

        # Select Game A folder
        if game_a_dir is None:
            print("\nPlease select the PUZZLE A GAME DATA folder (JSON files)...")
            game_a_dir = select_folder("Select Puzzle A Game Data folder")
        if not game_a_dir:
            print("No folder selected for Game A.")
            return False

        # Select Game B folder
        if game_b_dir is None:
            print("\nPlease select the PUZZLE B GAME DATA folder (JSON files)...")
            game_b_dir = select_folder("Select Puzzle B Game Data folder")
        if not game_b_dir:
            print("No folder selected for Game B.")
            return False

        # Process Game A
        game_a_path = Path(game_a_dir)
        game_a_files = list(game_a_path.glob("*.json"))
        print(f"\nProcessing {len(game_a_files)} Game A files...")
        for json_file in game_a_files:
            self._process_json_file(json_file, 'Game A')

        # Process Game B
        game_b_path = Path(game_b_dir)
        game_b_files = list(game_b_path.glob("*.json"))
        print(f"Processing {len(game_b_files)} Game B files...")
        for json_file in game_b_files:
            self._process_json_file(json_file, 'Game B')

        print(f"\n  Extracted {len(self.completion_times['Game A'])} Game A completion times")
        print(f"  Extracted {len(self.completion_times['Game B'])} Game B completion times")
        return True

    def _process_json_file(self, json_path, game):
        """Process a single JSON file to extract completion time"""
        try:
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
                participant_id = json_path.stem

            if participant_id:
                # Skip excluded participants
                if self.participant_tracker and self.participant_tracker.is_excluded(participant_id):
                    return
                self.completion_times[game][participant_id] = completion_time

        except Exception as e:
            print(f"  Warning: Could not process {json_path.name}: {e}")

    def compute_descriptive_statistics(self):
        """Compute descriptive statistics for completion times"""
        print("\n" + "=" * 50)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 50)

        self.results['descriptive_stats'] = {}

        for game in ['Game A', 'Game B']:
            times = list(self.completion_times[game].values())
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
        """Analyze correlation between age and completion time"""
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

            # Test normality
            _, normality_p = shapiro(df_age['completion_time'])

            if normality_p > 0.05:
                corr, p_value = pearsonr(df_age['age'], df_age['completion_time'])
                method = "Pearson"
            else:
                corr, p_value = spearmanr(df_age['age'], df_age['completion_time'])
                method = "Spearman"

            results[game] = {
                'method': method,
                'correlation': corr,
                'p_value': p_value,
                'n': len(df_age),
                'data': df_age
            }

            sig = '*' if p_value < 0.05 else 'ns'
            print(f"    {game}: {method} r={corr:.3f}, p={p_value:.4f} {sig}, n={len(df_age)}")

        self.results['age_correlation'] = results
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
            ax.hist(times, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Completion Time (minutes)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{game} Completion Time Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            mean_time = np.mean(times)
            median_time = np.median(times)
            ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f} min')
            ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.1f} min')
            ax.legend()

        plt.tight_layout()
        output_path = OUTPUT_DIR / f'completion_time_histograms_{self.timestamp}.png'
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
            plt.scatter(df_age['age'], df_age['completion_time'] / 60, alpha=0.6, s=100, color='steelblue')

            # Regression line
            z = np.polyfit(df_age['age'], df_age['completion_time'] / 60, 1)
            p = np.poly1d(z)
            plt.plot(df_age['age'], p(df_age['age']), "r--", alpha=0.8, linewidth=2)

            plt.xlabel('Age (years)', fontsize=12)
            plt.ylabel('Completion Time (minutes)', fontsize=12)
            plt.title(f'{game}: Age vs Completion Time\n{data["method"]} r = {data["correlation"]:.3f}, p = {data["p_value"]:.4f}',
                     fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)

            output_path = OUTPUT_DIR / f'age_correlation_{game.replace(" ", "_")}_{self.timestamp}.png'
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
                                   color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
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

    def create_regression_plot(self):
        """Create scatter plot with regression line"""
        if 'linear_regression' not in self.results or not self.results['linear_regression']:
            return

        print("  Creating regression plot...")

        data = self.results['linear_regression']

        plt.figure(figsize=(8, 6))
        plt.scatter(data['X'], data['y'], alpha=0.6, s=100, color='steelblue', label='Participants')
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
                f.write(f"Excluded: {sorted(self.participant_tracker.excluded_participants)}\n")

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
                f.write(f"\nAge Correlations:\n")
                for game, res in self.results['age_correlation'].items():
                    f.write(f"  {game}: {res['method']} r={res['correlation']:.3f}, p={res['p_value']:.4f}, n={res['n']}\n")

            if 'nlp_anova' in self.results and self.results['nlp_anova']:
                f.write(f"\nKruskal-Wallis Tests (Movement Features by Category):\n")
                for feature, res in self.results['nlp_anova'].items():
                    sig = '*' if res['p_value'] < 0.05 else ''
                    f.write(f"  {feature}: H={res['H_statistic']:.3f}, p={res['p_value']:.4f}{sig}\n")
                    for cat, stats in res['groups'].items():
                        f.write(f"    {cat}: M={stats['mean']:.3f}, SD={stats['std']:.3f}, n={stats['n']}\n")

            f.write("\n" + "=" * 80 + "\n")
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
    analyzer.chi_squared_test()
    analyzer.spearman_correlation()
    analyzer.linear_regression_analysis()
    analyzer.age_correlation_analysis()
    analyzer.analyze_nlp_by_category()

    # Create visualizations
    print("\n" + "=" * 50)
    print("STEP 3: CREATING VISUALIZATIONS")
    print("=" * 50)

    analyzer.create_completion_time_histograms()
    analyzer.create_nlp_boxplots()
    analyzer.create_feature_heatmap()
    analyzer.create_age_scatter_plots()
    analyzer.create_participant_distribution_chart()
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
    print("=" * 60)
    print("ARC Puzzle Data Analysis Program")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nUsage:")
    print("  Run: run_full_analysis()")
    print("\nOr use the ARCDataAnalyzer class directly:")
    print("  analyzer = ARCDataAnalyzer()")
    print("  analyzer.load_participant_tracker('path/to/tracker.csv')")
    print("  analyzer.load_nlp_classifications('path/to/nlp.csv')")
    print("  analyzer.extract_completion_times('game_a_dir', 'game_b_dir')")
    print("  analyzer.compute_descriptive_statistics()")
    print("  # ... etc")
