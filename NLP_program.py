#!/usr/bin/env python3
"""
NLP Speech Classification Program
==================================

Classifies participant speech segments from ARC puzzle experiments into
knowledge-search behavior categories:
- Exploratory (explore): Uncertainty, questions, random exploration
- Confirmatory (establish): Hypothesis testing, predictions, conditionals
- Exploitative (exploit): Certainty, execution, completion

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

# Output directory - always Downloads
OUTPUT_DIR = Path.home() / "Downloads"


class FileSelector:
    """Handles file/folder selection dialogs with a single Tk instance"""

    def __init__(self):
        self._root = None

    def _get_root(self):
        """Get or create the Tk root window"""
        if self._root is None:
            self._root = tk.Tk()
            self._root.withdraw()
        return self._root

    def select_folder(self, title="Select a folder"):
        """Open a folder selection dialog"""
        root = self._get_root()
        root.lift()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title=title, initialdir=str(OUTPUT_DIR))
        root.attributes('-topmost', False)
        return folder if folder else None

    def select_file(self, title="Select a file", filetypes=None):
        """Open a file selection dialog"""
        if filetypes is None:
            filetypes = [("All files", "*.*")]
        root = self._get_root()
        root.lift()
        root.attributes('-topmost', True)
        file = filedialog.askopenfilename(title=title, initialdir=str(OUTPUT_DIR), filetypes=filetypes)
        root.attributes('-topmost', False)
        return file if file else None

    def cleanup(self):
        """Destroy the Tk root window"""
        if self._root is not None:
            self._root.destroy()
            self._root = None


# Global file selector instance
_file_selector = FileSelector()


def select_folder(title="Select a folder"):
    """Open a folder selection dialog"""
    return _file_selector.select_folder(title)


def select_file(title="Select a file", filetypes=None):
    """Open a file selection dialog"""
    return _file_selector.select_file(title, filetypes)


class ParticipantTracker:
    """
    Loads and manages participant tracking data for mapping files to participant IDs.
    """

    def __init__(self, tracker_csv_path):
        """Load the participant tracker CSV"""
        self.df = pd.read_csv(tracker_csv_path)
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build lookup tables for fast matching"""
        self.game_a_lookup = {}  # timestamp -> participant_id
        self.game_b_lookup = {}  # timestamp -> participant_id
        self.audio_a_lookup = {}  # audio timestamp -> participant_id
        self.audio_b_lookup = {}  # audio timestamp -> participant_id

        # Hardcoded exclusion list - these participants are always excluded
        self.excluded_participants = {
            'P001', 'P002', 'P007', 'P013', 'P014', 'P015', 'P016', 'P020', 'P024'
        }

        for _, row in self.df.iterrows():
            session_id = str(row.get('Session ID:', '')).strip()
            if not session_id or session_id == 'nan':
                continue

            # Check if excluded
            if 'exclude' in session_id.lower():
                # Extract just the P### part
                match = re.search(r'(P\d+)', session_id)
                if match:
                    self.excluded_participants.add(match.group(1))
                continue

            # Extract participant ID (P001, P002, etc.)
            participant_id = session_id.split()[0] if ' ' in session_id else session_id

            # Game A data file
            game_a_file = str(row.get('Game A Data (file name):', '')).strip()
            if game_a_file and game_a_file != 'nan' and game_a_file != '--':
                timestamp = self._extract_timestamp(game_a_file)
                if timestamp:
                    self.game_a_lookup[timestamp] = participant_id
                # Also extract audio timestamp
                audio_ts = self._extract_audio_timestamp_from_game(game_a_file)
                if audio_ts:
                    self.audio_a_lookup[audio_ts] = participant_id

            # Game B data file
            game_b_file = str(row.get('Game B Data (file name):', '')).strip()
            if game_b_file and game_b_file != 'nan' and game_b_file != '--':
                timestamp = self._extract_timestamp(game_b_file)
                if timestamp:
                    self.game_b_lookup[timestamp] = participant_id
                # Also extract audio timestamp
                audio_ts = self._extract_audio_timestamp_from_game(game_b_file)
                if audio_ts:
                    self.audio_b_lookup[audio_ts] = participant_id

        print(f"Loaded {len(self.game_a_lookup)} Game A mappings")
        print(f"Loaded {len(self.game_b_lookup)} Game B mappings")
        print(f"Excluded participants: {sorted(self.excluded_participants)}")

    def _extract_timestamp(self, filename):
        """
        Extract timestamp from JSON filename.

        Patterns:
        - game-session-2025-12-30T16-51-25-570Z.json (mehdi)
        - puzzle-game1-state-2026-01-05T17-15-03-300Z.json
        """
        # Match ISO-like timestamp: YYYY-MM-DDTHH-MM-SS
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', filename)
        if match:
            return match.group(1)
        return None

    def _extract_audio_timestamp_from_game(self, game_filename):
        """
        Extract the timestamp that would match audio files.
        Audio files use format: puzzle-game1-audio-2025-12-30T16-51-25.webm
        Game files use: game-session-2025-12-30T16-51-25-570Z.json

        They share the same base timestamp (without milliseconds).
        """
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', game_filename)
        if match:
            return match.group(1)
        return None

    def get_participant_id_from_audio(self, audio_filename, game_type):
        """
        Get participant ID from audio/transcript filename.

        Args:
            audio_filename: e.g., 'puzzle-game1-audio-2025-12-30T16-51-25_transcription.txt'
            game_type: 'Game A' or 'Game B'
        """
        timestamp = self._extract_timestamp(audio_filename)
        if not timestamp:
            return None

        if game_type == 'Game A':
            return self.audio_a_lookup.get(timestamp)
        elif game_type == 'Game B':
            return self.audio_b_lookup.get(timestamp)
        return None

    def get_participant_id_from_json(self, json_filename, game_type):
        """
        Get participant ID from JSON game data filename.

        Args:
            json_filename: e.g., 'game-session-2025-12-30T16-51-25-570Z.json (mehdi)'
            game_type: 'Game A' or 'Game B'
        """
        timestamp = self._extract_timestamp(json_filename)
        if not timestamp:
            return None

        if game_type == 'Game A':
            return self.game_a_lookup.get(timestamp)
        elif game_type == 'Game B':
            return self.game_b_lookup.get(timestamp)
        return None

    def is_excluded(self, participant_id):
        """Check if a participant is excluded"""
        return participant_id in self.excluded_participants

    def get_valid_participants(self):
        """Get list of valid (non-excluded) participant IDs"""
        all_participants = set(self.game_a_lookup.values()) | set(self.game_b_lookup.values())
        return sorted(all_participants - self.excluded_participants)


class GameStateAnalyzer:
    """
    Analyzes game-state JSON files to extract level timing information.
    Used to correlate speech segments with game levels.
    """

    def __init__(self, gamestate_json_path):
        """Load and parse a game-state JSON file."""
        self.filepath = Path(gamestate_json_path)
        self.data = None
        self.session_start = None
        self.level_timestamps = {}  # level_num -> {'start': datetime, 'end': datetime}

        self._load_and_parse()

    def _load_and_parse(self):
        """Load JSON and extract level timing information."""
        with open(self.filepath, 'r') as f:
            self.data = json.load(f)

        # Parse session start time
        session_start_str = self.data.get('sessionStart', '')
        if session_start_str:
            self.session_start = self._parse_iso_timestamp(session_start_str)

        # Extract level timestamps from movements
        movements = self.data.get('movements', [])
        for movement in movements:
            level = movement.get('level')
            timestamp_str = movement.get('timestamp', '')
            if level is None or not timestamp_str:
                continue

            timestamp = self._parse_iso_timestamp(timestamp_str)
            if timestamp is None:
                continue

            if level not in self.level_timestamps:
                self.level_timestamps[level] = {'start': timestamp, 'end': timestamp}
            else:
                # Update end time to the latest timestamp for this level
                if timestamp > self.level_timestamps[level]['end']:
                    self.level_timestamps[level]['end'] = timestamp
                if timestamp < self.level_timestamps[level]['start']:
                    self.level_timestamps[level]['start'] = timestamp

    def _parse_iso_timestamp(self, iso_string):
        """Parse ISO 8601 timestamp string to datetime object."""
        try:
            # Handle format: 2026-01-11T20:22:08.894Z
            iso_string = iso_string.replace('Z', '+00:00')
            return datetime.fromisoformat(iso_string.replace('Z', ''))
        except (ValueError, AttributeError):
            try:
                # Try without timezone
                return datetime.strptime(iso_string[:19], '%Y-%m-%dT%H:%M:%S')
            except:
                return None

    def get_level_for_timestamp(self, relative_seconds):
        """
        Determine which level a speech segment occurred in.

        Args:
            relative_seconds: Seconds from start of session/recording

        Returns:
            Level number (int) or None if can't be determined
        """
        if self.session_start is None:
            return None

        # Convert relative time to absolute time
        absolute_time = self.session_start + timedelta(seconds=relative_seconds)

        # Find which level this falls into
        for level, times in self.level_timestamps.items():
            # Add a small buffer (5 seconds) before level start for pre-level speech
            level_start = times['start'] - timedelta(seconds=5)
            level_end = times['end'] + timedelta(seconds=5)

            if level_start <= absolute_time <= level_end:
                return level

        # If between levels, return the next level (anticipatory speech)
        sorted_levels = sorted(self.level_timestamps.keys())
        for i, level in enumerate(sorted_levels):
            if absolute_time < self.level_timestamps[level]['start']:
                return level  # Speech before this level starts

        # If after all levels, return the last level
        if sorted_levels:
            return sorted_levels[-1]

        return None

    def get_level_info(self):
        """Return summary of level timing information."""
        info = {
            'session_start': self.session_start,
            'num_levels': len(self.level_timestamps),
            'levels': {}
        }
        for level, times in sorted(self.level_timestamps.items()):
            duration = (times['end'] - times['start']).total_seconds()
            info['levels'][level] = {
                'start': times['start'],
                'end': times['end'],
                'duration_seconds': duration
            }
        return info

    def is_near_level_completion(self, relative_seconds, window_seconds=10):
        """
        Check if a timestamp is near the end of a level (approaching solution).

        Args:
            relative_seconds: Seconds from start of session/recording
            window_seconds: How many seconds before level end to consider "near completion"

        Returns:
            dict with:
                - 'near_completion': bool
                - 'level': level number if near completion
                - 'seconds_to_completion': seconds until level ends
                - 'is_final_levels': True if this is one of the last 2 levels
        """
        if self.session_start is None:
            return {'near_completion': False, 'level': None, 'seconds_to_completion': None, 'is_final_levels': False}

        # Convert relative time to absolute time
        absolute_time = self.session_start + timedelta(seconds=relative_seconds)

        sorted_levels = sorted(self.level_timestamps.keys())
        num_levels = len(sorted_levels)

        for level in sorted_levels:
            times = self.level_timestamps[level]
            level_end = times['end']

            # Check if we're within the window before level completion
            seconds_to_end = (level_end - absolute_time).total_seconds()

            if 0 <= seconds_to_end <= window_seconds:
                # We're within the window before level completion
                is_final = level >= num_levels - 1  # Last 2 levels (0-indexed: levels 4 and 5 if 6 total)
                return {
                    'near_completion': True,
                    'level': level,
                    'seconds_to_completion': seconds_to_end,
                    'is_final_levels': is_final
                }

        return {'near_completion': False, 'level': None, 'seconds_to_completion': None, 'is_final_levels': False}

    def get_max_level(self):
        """Return the highest level number in this session."""
        if not self.level_timestamps:
            return None
        return max(self.level_timestamps.keys())


def parse_timestamp_to_seconds(timestamp_str):
    """
    Convert timestamp string (MM:SS or HH:MM:SS) to total seconds.

    Args:
        timestamp_str: e.g., "01:23" or "01:23:45"

    Returns:
        Total seconds as float, or None if parsing fails
    """
    if not timestamp_str:
        return None

    try:
        parts = timestamp_str.strip().split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return None


class SpeechCategoryClassifier:
    """Classify speech segments based on linguistic markers"""

    def __init__(self):
        # Linguistic markers for each category
        # Exploratory = explore, Confirmatory = establish, Exploitative = exploit
        self.exploratory_markers = {
            'questions': ['what', 'why', 'how', 'where', 'which', 'when'],
            'uncertainty': ['not sure', "don't know", 'trying to figure', 'confused'],
            'exploration_verbs': ['exploring', 'checking', 'seeing',
                                  'trying different', 'experimenting', 'random', 'randomly'],
            'observations': ['I see', "there's", 'there are', 'there is',
                            'this is', 'interesting', "doesn't make sense"]
        }

        self.confirmatory_markers = {
            'hypothesis_statements': ['I think', 'my hypothesis', 'if...then',
                                     'probably', 'should be', 'seems like', 'bet', 'wonder if',
                                     'I think maybe', 'kind of', 'sort of',
                                     'maybe', 'might', 'could', 'perhaps', 'wonder'],
            'testing_language': ['let me test', 'testing', 'checking if', 'see if',
                                'trying to confirm', 'verify', 'test whether',
                                'gonna try', 'let me see if', 'let me check',
                                "I'm going to test", 'confirm'],
            'conditional': ['if I', 'when I', 'if this', 'assuming', 'suppose'],
            'prediction': ['will', 'should', 'expect', 'predict', 'would']
        }

        self.exploitative_markers = {
            'certainty': ['I know', 'definitely', 'obviously', 'clearly', 'for sure',
                         'certain', 'figured it out', 'got it', 'understand', 'I get it',
                         'makes sense', 'that makes sense', 'ah okay', 'oh okay'],
            'execution': ["now I'll just", 'just need to', 'all I have to do',
                         'just going to', 'now I can', 'okay now', "let me just",
                         'alright now', 'easy', 'match', 'so I need to',
                         "I'm gonna", "I'll go", 'move to', 'go to', 'align'],
            'completion': ['almost done', 'finish', 'complete', 'final step',
                          'last step', 'last thing', 'done', 'solved', 'there we go',
                          "that's it", 'yes', 'nice', 'perfect', 'boom', 'there it is',
                          'got it', 'yep', 'okay good']
        }

    def score_category(self, text, markers_dict):
        """Count markers from a category in text"""
        text_lower = text.lower()
        score = 0
        matched_markers = []

        for category, markers in markers_dict.items():
            for marker in markers:
                if marker in text_lower:
                    score += 1
                    matched_markers.append(marker)

        return score, matched_markers

    def classify(self, text):
        """Classify a speech segment"""
        exp_score, exp_markers = self.score_category(text, self.exploratory_markers)
        conf_score, conf_markers = self.score_category(text, self.confirmatory_markers)
        expl_score, expl_markers = self.score_category(text, self.exploitative_markers)

        scores = {
            'exploratory': exp_score,
            'confirmatory': conf_score,
            'exploitative': expl_score
        }

        markers_matched = {
            'exploratory': exp_markers,
            'confirmatory': conf_markers,
            'exploitative': expl_markers
        }

        max_score = max(scores.values())
        total_score = sum(scores.values())

        # Flag for manual review if:
        # 1. No markers matched
        # 2. Tie between categories
        # 3. Only 1 marker matched (too ambiguous)
        if max_score == 0:
            return 'NEEDS_MANUAL_REVIEW', 'low', scores, markers_matched

        max_categories = [cat for cat, score in scores.items() if score == max_score]
        if len(max_categories) > 1:
            return 'NEEDS_MANUAL_REVIEW', 'low', scores, markers_matched

        # Single marker match is ambiguous - flag for review
        if total_score == 1:
            category = max(scores, key=scores.get)
            return 'NEEDS_MANUAL_REVIEW', 'low', scores, markers_matched

        category = max(scores, key=scores.get)

        # Calculate confidence level
        # High: max_score >= 3 AND dominates other categories
        # Medium: max_score >= 2 OR clear winner
        # Low: max_score == 1 or close competition
        if max_score >= 3 and max_score >= total_score * 0.6:
            confidence = 'high'
        elif max_score >= 2 and max_score > total_score * 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
            # Low confidence should go to manual review
            return 'NEEDS_MANUAL_REVIEW', confidence, scores, markers_matched

        return category, confidence, scores, markers_matched


def apply_sequential_context(results_df, time_window_seconds=3):
    """
    Apply sequential context analysis to ALL segments.

    For each segment, look at surrounding segments within the time window.
    If the segment's classification conflicts with the dominant pattern of
    nearby segments, flag it for review.

    Args:
        results_df: DataFrame with classification results
        time_window_seconds: Max time gap to consider segments as contextually related

    Returns:
        Updated DataFrame with 'context_adjusted_category' and 'context_flag' columns
    """
    df = results_df.copy()

    # Add columns for context analysis
    df['context_adjusted_category'] = df['auto_category']
    df['context_source'] = ''  # Track where the context came from
    df['context_flag'] = ''  # Flag potential misclassifications

    # Convert timestamps to seconds for comparison
    df['start_seconds'] = df['start_time'].apply(parse_timestamp_to_seconds)

    # Process ALL segments (not just NEEDS_MANUAL_REVIEW)
    for idx in df.index:
        current_time = df.loc[idx, 'start_seconds']
        if current_time is None:
            continue

        current_category = df.loc[idx, 'auto_category']

        # Get the same participant and game only
        participant = df.loc[idx, 'participant_id']
        game = df.loc[idx, 'game']

        # Find surrounding segments within time window (excluding current segment)
        mask = (
            (df['participant_id'] == participant) &
            (df['game'] == game) &
            (df['start_seconds'].notna()) &
            (df.index != idx)
        )

        nearby = df[mask].copy()
        if len(nearby) == 0:
            continue

        # Calculate time differences
        nearby['time_diff'] = abs(nearby['start_seconds'] - current_time)

        # Get segments within the time window
        within_window = nearby[nearby['time_diff'] <= time_window_seconds]

        if len(within_window) == 0:
            continue

        # Count categories of nearby segments (excluding NEEDS_MANUAL_REVIEW for counting)
        classified_nearby = within_window[within_window['auto_category'] != 'NEEDS_MANUAL_REVIEW']
        if len(classified_nearby) == 0:
            continue

        category_counts = classified_nearby['auto_category'].value_counts()
        closest_time_diff = within_window['time_diff'].min()

        if len(category_counts) == 0:
            continue

        dominant_category = category_counts.index[0]
        dominant_count = category_counts.iloc[0]
        total_nearby = len(classified_nearby)

        # For NEEDS_MANUAL_REVIEW segments: inherit from context if clear pattern
        if current_category == 'NEEDS_MANUAL_REVIEW':
            if dominant_count >= 2 or (dominant_count >= 1 and closest_time_diff <= 1.5):
                df.loc[idx, 'context_adjusted_category'] = dominant_category
                df.loc[idx, 'context_source'] = f'inherited_{closest_time_diff:.1f}s'

        # For already-classified segments: check if context conflicts
        else:
            # If dominant nearby category differs AND is strong pattern, flag for review
            if dominant_category != current_category:
                dominance_ratio = dominant_count / total_nearby if total_nearby > 0 else 0

                # Flag if context strongly suggests different category
                if dominance_ratio >= 0.7 and dominant_count >= 2:
                    df.loc[idx, 'context_flag'] = f'context_suggests_{dominant_category}'
                    df.loc[idx, 'needs_review'] = True  # Flag for manual review

    # Clean up temporary column
    df = df.drop(columns=['start_seconds'], errors='ignore')

    # TRANSITION RULE: No explore → exploit transitions allowed
    # Must go through establish (confirmatory) first
    # Sort by participant, game, and time to check sequential transitions
    df['_start_seconds'] = df['start_time'].apply(parse_timestamp_to_seconds)
    df = df.sort_values(['participant_id', 'game', '_start_seconds']).reset_index(drop=True)

    transition_fixes = 0
    for idx in range(1, len(df)):
        # Check if same participant and game
        if (df.loc[idx, 'participant_id'] != df.loc[idx-1, 'participant_id'] or
            df.loc[idx, 'game'] != df.loc[idx-1, 'game']):
            continue

        prev_category = df.loc[idx-1, 'context_adjusted_category']
        curr_category = df.loc[idx, 'context_adjusted_category']

        # Invalid transition: explore → exploit
        if prev_category == 'exploratory' and curr_category == 'exploitative':
            # Reclassify as confirmatory (establish)
            df.loc[idx, 'context_adjusted_category'] = 'confirmatory'
            df.loc[idx, 'context_flag'] = 'transition_rule_explore_to_exploit'
            transition_fixes += 1

    if transition_fixes > 0:
        print(f"  -> {transition_fixes} explore→exploit transitions fixed (changed to confirmatory)")

    df = df.drop(columns=['_start_seconds'], errors='ignore')

    return df


def parse_whisper_transcript(transcript_file):
    """
    Parse a Whisper transcript .txt file into segments

    Supports multiple formats:

    Format 1 (timestamp on own line):
    GAME A

    00:00
    Text here

    00:03
    More text here

    Format 2 (bracketed timestamps):
    [00:00:00 - 00:00:05] Hello, this is the beginning.

    Returns:
        dict with 'puzzle_label' and 'segments' list
    """
    segments = []
    puzzle_label = None

    with open(transcript_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    # Extract puzzle/game label if present
    puzzle_match = re.search(r'=== (Puzzle [A-C]) ===', content)
    if puzzle_match:
        puzzle_label = puzzle_match.group(1)

    # Check for "GAME A" or "GAME B" at the start
    game_match = re.search(r'^(GAME [A-C])', content, re.MULTILINE)
    if game_match:
        puzzle_label = game_match.group(1).replace('GAME', 'Puzzle')

    # Try Format 2 first: [00:00:00 - 00:00:05] text
    pattern_bracketed = r'\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\] (.+)'
    matches = re.findall(pattern_bracketed, content)

    if matches:
        for i, (start_time, end_time, text) in enumerate(matches):
            segments.append({
                'segment_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'text': text.strip()
            })
    else:
        # Format 1: timestamp on its own line, text on following lines
        # Pattern: MM:SS or HH:MM:SS on its own line
        current_timestamp = None
        current_text_lines = []
        segment_id = 0

        for line in lines:
            line = line.strip()

            # Check if this line is a timestamp (MM:SS or HH:MM:SS)
            timestamp_match = re.match(r'^(\d{1,2}:\d{2}(?::\d{2})?)$', line)

            if timestamp_match:
                # Save previous segment if we have one
                if current_timestamp is not None and current_text_lines:
                    text = ' '.join(current_text_lines).strip()
                    if text:
                        segments.append({
                            'segment_id': segment_id,
                            'start_time': current_timestamp,
                            'end_time': '',  # Will calculate from next timestamp
                            'text': text
                        })
                        segment_id += 1

                # Start new segment
                current_timestamp = timestamp_match.group(1)
                # Normalize to HH:MM:SS format
                if current_timestamp.count(':') == 1:
                    current_timestamp = '00:' + current_timestamp
                current_text_lines = []

            elif line and current_timestamp is not None:
                # Skip header lines like "GAME A"
                if not re.match(r'^GAME [A-C]$', line):
                    current_text_lines.append(line)

        # Don't forget the last segment
        if current_timestamp is not None and current_text_lines:
            text = ' '.join(current_text_lines).strip()
            if text:
                segments.append({
                    'segment_id': segment_id,
                    'start_time': current_timestamp,
                    'end_time': '',
                    'text': text
                })

        # Calculate end times from next segment's start time
        for i in range(len(segments) - 1):
            segments[i]['end_time'] = segments[i + 1]['start_time']
        if segments:
            segments[-1]['end_time'] = segments[-1]['start_time']  # Last segment

    return {
        'puzzle_label': puzzle_label,
        'segments': segments
    }


def parse_game_from_filename(filename):
    """
    Extract game type from filename

    Expected format: puzzle-game2-audio-2026-01-14T04-28-46_transcription.txt
    Returns: 'Game A', 'Game B', or 'Game C'
    """
    filename_lower = filename.lower()

    if 'game1' in filename_lower or 'puzzle-a' in filename_lower:
        return 'Game A'
    elif 'game2' in filename_lower or 'puzzle-b' in filename_lower:
        return 'Game B'
    elif 'game3' in filename_lower or 'puzzle-c' in filename_lower:
        return 'Game C'
    else:
        return 'Unknown'


def process_transcript_file(transcript_file, participant_tracker=None, participant_id=None,
                           game_name=None, gamestate_analyzer=None):
    """
    Process one participant's transcript file

    Args:
        transcript_file: Path to transcript file (.txt from Whisper or .json)
        participant_tracker: ParticipantTracker instance for ID lookup
        participant_id: Participant ID (optional, will try to extract)
        game_name: 'Game A', 'Game B', or 'Game C' (optional, will try to extract)
        gamestate_analyzer: GameStateAnalyzer instance for level detection (optional)

    Returns:
        DataFrame with classified speech segments
    """
    transcript_path = Path(transcript_file)
    filename = transcript_path.name

    # Determine file format and parse accordingly
    if transcript_path.suffix == '.txt':
        # Parse Whisper .txt format
        transcript_data = parse_whisper_transcript(transcript_file)
        segments = transcript_data['segments']

        # Use puzzle label from file if available
        if transcript_data['puzzle_label'] and not game_name:
            # Convert "Puzzle A" to "Game A"
            puzzle_label = transcript_data['puzzle_label']
            game_name = puzzle_label.replace('Puzzle', 'Game')
    else:
        # Parse JSON format
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        segments = transcript_data.get('segments', [])

    # Extract game type from filename if not provided
    if not game_name:
        game_name = parse_game_from_filename(filename)

    # Extract participant ID using tracker if not provided
    if not participant_id and participant_tracker:
        participant_id = participant_tracker.get_participant_id_from_audio(filename, game_name)

    if not participant_id:
        # Fallback: try to extract from filename or use filename as ID
        match = re.search(r'(P\d+)', filename, re.IGNORECASE)
        if match:
            participant_id = match.group(1)
        else:
            # Use timestamp from filename as identifier
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', filename)
            participant_id = ts_match.group(1) if ts_match else filename.split('_')[0]

    # Check if participant is excluded
    if participant_tracker and participant_tracker.is_excluded(participant_id):
        print(f"  -> Skipping excluded participant: {participant_id}")
        return pd.DataFrame()

    classifier = SpeechCategoryClassifier()
    results = []

    for segment in segments:
        text = segment.get('text', '')
        start_time = segment.get('start_time', '')
        end_time = segment.get('end_time', '')
        segment_id = segment.get('segment_id', 0)

        # Skip empty segments
        if not text.strip():
            continue

        # Classify
        category, confidence, scores, markers = classifier.classify(text)

        # Determine level from game-state data if available
        level = None
        near_completion_info = {'near_completion': False, 'level': None, 'seconds_to_completion': None, 'is_final_levels': False}
        if gamestate_analyzer:
            start_seconds = parse_timestamp_to_seconds(start_time)
            if start_seconds is not None:
                level = gamestate_analyzer.get_level_for_timestamp(start_seconds)
                # Check if near level completion (within 10 seconds of solving)
                near_completion_info = gamestate_analyzer.is_near_level_completion(start_seconds, window_seconds=10)

        # GAME B LEVELS 1-2: Strong exploration bias
        # In Game B, levels 1-2 teach different rules, so exploration is expected
        is_game_b_early_levels = (game_name == 'Game B' and level is not None and level <= 2)

        if is_game_b_early_levels:
            # In Game B levels 1-2, classify ambiguous segments as exploration
            if category == 'NEEDS_MANUAL_REVIEW':
                category = 'exploratory'
                confidence = 'medium'
                markers['exploratory'].append('game_b_early_level')
            elif category == 'exploitative':
                # Can't exploit in early learning levels - reclassify
                if scores['confirmatory'] > 0:
                    category = 'confirmatory'
                    confidence = 'medium'
                else:
                    category = 'exploratory'
                    confidence = 'medium'
                markers['exploratory'].append('game_b_early_level_override')

        # Apply level completion boost for exploit classification
        # If near level completion and not already classified as exploit, boost exploit
        near_completion = near_completion_info['near_completion']
        is_final_levels = near_completion_info['is_final_levels']

        # Only apply exploit boost for levels 3+ in Game B (or any level in Game A)
        if near_completion and not is_game_b_early_levels:
            if category != 'exploitative':
                # Near level completion strongly suggests exploit behavior
                # For final levels: override to exploit if not clearly another category
                # For other levels: boost exploit score and flag for review
                if is_final_levels and category == 'NEEDS_MANUAL_REVIEW':
                    # Final levels + ambiguous = likely exploit
                    category = 'exploitative'
                    confidence = 'medium'
                    markers['exploitative'].append('near_level_completion_final')
                elif is_final_levels and confidence == 'low':
                    # Final levels + low confidence = override to exploit
                    category = 'exploitative'
                    confidence = 'medium'
                    markers['exploitative'].append('near_level_completion_final')
                elif near_completion and category == 'NEEDS_MANUAL_REVIEW':
                    # Near any level completion + ambiguous = likely exploit
                    category = 'exploitative'
                    confidence = 'low'
                    markers['exploitative'].append('near_level_completion')

        results.append({
            'participant_id': participant_id,
            'game': game_name,
            'level': level,
            'segment_id': segment_id,
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'audio_filename': filename,
            'transcript_file_path': str(transcript_path),  # Full path for linking
            'auto_category': category,
            'confidence': confidence,
            'exploratory_score': scores['exploratory'],
            'confirmatory_score': scores['confirmatory'],
            'exploitative_score': scores['exploitative'],
            'exploratory_markers': ', '.join(markers['exploratory']),
            'confirmatory_markers': ', '.join(markers['confirmatory']),
            'exploitative_markers': ', '.join(markers['exploitative']),
            'near_level_completion': near_completion,
            'is_final_levels': is_final_levels,
            'needs_review': category == 'NEEDS_MANUAL_REVIEW'
        })

    return pd.DataFrame(results)


def process_all_transcripts(transcript_dir, participant_tracker=None, output_file=None,
                           gamestate_dir=None, apply_context=True):
    """
    Process all transcript files in a directory

    Args:
        transcript_dir: Directory containing transcript files (.txt or .json)
        participant_tracker: ParticipantTracker instance for ID lookup
        output_file: Where to save the classified segments
        gamestate_dir: Directory containing game-state JSON files (optional)
        apply_context: Whether to apply sequential context analysis (default True)
    """
    if output_file is None:
        output_file = OUTPUT_DIR / "classified_speech_segments.xlsx"

    all_results = []

    # Look for both .txt (Whisper) and .json files
    transcript_path = Path(transcript_dir)
    transcript_files = list(transcript_path.glob('*.txt')) + list(transcript_path.glob('*.json'))

    print(f"Found {len(transcript_files)} transcript files in {transcript_dir}")

    if len(transcript_files) == 0:
        print(f"Warning: No transcript files found in {transcript_dir}")
        print("Make sure your transcriptions are in the correct folder.")
        return pd.DataFrame()

    # Load game-state files if directory provided
    gamestate_lookup = {}
    if gamestate_dir:
        gamestate_path = Path(gamestate_dir)
        gamestate_files = list(gamestate_path.glob('*.json'))
        print(f"Found {len(gamestate_files)} game-state files in {gamestate_dir}")

        for gs_file in gamestate_files:
            # Extract timestamp from filename to match with transcripts
            match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', gs_file.name)
            if match:
                timestamp = match.group(1)
                try:
                    gamestate_lookup[timestamp] = GameStateAnalyzer(gs_file)
                except Exception as e:
                    print(f"  Warning: Could not load game-state {gs_file.name}: {e}")

    for transcript_file in transcript_files:
        print(f"Processing {transcript_file.name}...")

        # Find matching game-state analyzer
        gamestate_analyzer = None
        if gamestate_lookup:
            match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', transcript_file.name)
            if match:
                timestamp = match.group(1)
                gamestate_analyzer = gamestate_lookup.get(timestamp)
                if gamestate_analyzer:
                    print(f"  -> Matched game-state file for level detection")

        try:
            df = process_transcript_file(
                transcript_file,
                participant_tracker=participant_tracker,
                gamestate_analyzer=gamestate_analyzer
            )
            if len(df) > 0:
                all_results.append(df)
                level_info = ""
                if 'level' in df.columns and df['level'].notna().any():
                    levels = df['level'].dropna().unique()
                    level_info = f", levels: {sorted([int(l) for l in levels])}"
                print(f"  -> {len(df)} segments, participant: {df['participant_id'].iloc[0]}{level_info}")
        except Exception as e:
            print(f"  Error processing {transcript_file.name}: {e}")
            continue

    if not all_results:
        print("No transcripts were successfully processed.")
        return pd.DataFrame()

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Apply sequential context analysis
    if apply_context:
        print("\nApplying sequential context analysis...")
        final_df = apply_sequential_context(final_df, time_window_seconds=3)
        context_adjusted = (final_df['context_adjusted_category'] != final_df['auto_category']).sum()
        print(f"  -> {context_adjusted} segments adjusted based on context")

    # Calculate summary statistics
    total_segments = len(final_df)
    confidence_counts = final_df['confidence'].value_counts()
    high_count = confidence_counts.get('high', 0)
    medium_count = confidence_counts.get('medium', 0)
    low_count = confidence_counts.get('low', 0)

    high_pct = (high_count / total_segments * 100) if total_segments > 0 else 0
    medium_pct = (medium_count / total_segments * 100) if total_segments > 0 else 0
    low_pct = (low_count / total_segments * 100) if total_segments > 0 else 0

    # Create summary dataframe
    summary_data = {
        'Metric': [
            'Total Segments',
            '',
            'High Confidence',
            'Medium Confidence',
            'Low Confidence',
            '',
            'Needs Manual Review',
            '',
            'Category: exploratory',
            'Category: confirmatory',
            'Category: exploitative',
            'Category: NEEDS_MANUAL_REVIEW'
        ],
        'Count': [
            total_segments,
            '',
            high_count,
            medium_count,
            low_count,
            '',
            final_df['needs_review'].sum(),
            '',
            (final_df['auto_category'] == 'exploratory').sum(),
            (final_df['auto_category'] == 'confirmatory').sum(),
            (final_df['auto_category'] == 'exploitative').sum(),
            (final_df['auto_category'] == 'NEEDS_MANUAL_REVIEW').sum()
        ],
        'Percentage': [
            '100%',
            '',
            f'{high_pct:.1f}%',
            f'{medium_pct:.1f}%',
            f'{low_pct:.1f}%',
            '',
            f'{(final_df["needs_review"].sum() / total_segments * 100):.1f}%' if total_segments > 0 else '0%',
            '',
            f'{((final_df["auto_category"] == "exploratory").sum() / total_segments * 100):.1f}%' if total_segments > 0 else '0%',
            f'{((final_df["auto_category"] == "confirmatory").sum() / total_segments * 100):.1f}%' if total_segments > 0 else '0%',
            f'{((final_df["auto_category"] == "exploitative").sum() / total_segments * 100):.1f}%' if total_segments > 0 else '0%',
            f'{((final_df["auto_category"] == "NEEDS_MANUAL_REVIEW").sum() / total_segments * 100):.1f}%' if total_segments > 0 else '0%'
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # Save to Excel with summary sheet first
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        final_df.to_excel(writer, sheet_name='Classified Segments', index=False)

    print(f"\nSaved classified segments to {output_file}")
    print(f"\n{'='*40}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*40}")
    print(f"Total segments: {total_segments}")
    print(f"\nConfidence breakdown:")
    print(f"  High:   {high_count:>5} ({high_pct:.1f}%)")
    print(f"  Medium: {medium_count:>5} ({medium_pct:.1f}%)")
    print(f"  Low:    {low_count:>5} ({low_pct:.1f}%)")
    print(f"\nNeeds manual review: {final_df['needs_review'].sum()}")
    print(f"\nCategory breakdown (auto):")
    print(final_df['auto_category'].value_counts())
    if 'context_adjusted_category' in final_df.columns:
        print(f"\nCategory breakdown (after context adjustment):")
        print(final_df['context_adjusted_category'].value_counts())

    return final_df


def create_manual_review_file(classified_df, output_file=None):
    """
    Create an Excel file with segments needing manual review.
    Sheet 1: Review Data
    Sheet 2: Category Definitions
    """
    if output_file is None:
        output_file = OUTPUT_DIR / "manual_review_needed.xlsx"

    review_df = classified_df[classified_df['needs_review'] == True].copy()

    # Add empty column for manual coding
    review_df['manual_category'] = ''
    review_df['reviewer_notes'] = ''

    # Reorder columns for easier review
    columns_order = [
        'participant_id', 'game', 'level',  # Level from game-state data
        'segment_id', 'start_time', 'end_time',  # TIMESTAMPS for finding in audio
        'text', 'audio_filename', 'transcript_file_path',  # Full path to find the file
        'auto_category', 'context_adjusted_category', 'context_source', 'context_flag',  # Context analysis
        'exploratory_score', 'confirmatory_score', 'exploitative_score',
        'exploratory_markers', 'confirmatory_markers', 'exploitative_markers',
        'confidence',
        'manual_category',  # YOU FILL THIS IN
        'reviewer_notes'    # YOU FILL THIS IN
    ]

    # Only include columns that exist
    columns_order = [c for c in columns_order if c in review_df.columns]
    review_df = review_df[columns_order]

    # Create definitions dataframe for second sheet
    definitions_df = pd.DataFrame({
        'Category': ['EXPLORATORY (explore)', 'CONFIRMATORY (establish)', 'EXPLOITATIVE (exploit)'],
        'Markers': [
            "what, why, how, where, which, when, maybe, might, could, perhaps, wonder, not sure, don't know, trying to figure, confused, hmm, uh, exploring, looking, checking, seeing, trying different, experimenting, random, randomly, I think maybe, kind of, sort of, seems like",
            "I think, my hypothesis, if...then, probably, should be, seems like, bet, wonder if, let me test, testing, checking if, see if, trying to confirm, verify, test whether, gonna try, let me see if, let me check, I'm going to test, confirm, if I, when I, if this, assuming, suppose, will, should, expect, predict, would",
            "I know, definitely, obviously, clearly, for sure, certain, figured it out, got it, understand, now I'll just, just need to, all I have to do, simply, just going to, now I can, okay now, alright now, just, easy, almost done, finish, complete, final step, last thing, done, solved"
        ]
    })

    # Write Excel file with two sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        review_df.to_excel(writer, sheet_name='Review Data', index=False)
        definitions_df.to_excel(writer, sheet_name='Category Definitions', index=False)

    print(f"\nCreated manual review file: {output_file}")
    print(f"  Sheet 'Review Data': {len(review_df)} segments need review")
    print(f"  Sheet 'Category Definitions': marker definitions")
    print("\nInstructions:")
    print("1. Open the Excel file")
    print("2. Jump to 'start_time' in the audio file to hear each segment")
    print("3. Fill in 'manual_category' column with: exploratory, confirmatory, or exploitative")
    print("4. Save the file")

    return review_df


def merge_manual_reviews(auto_classified_df, manual_review_file=None):
    """
    Merge manual reviews back into the main dataframe
    """
    if manual_review_file is None:
        manual_review_file = OUTPUT_DIR / "manual_review_needed.xlsx"

    # Load manual reviews from Excel (Sheet 1: Review Data)
    manual_df = pd.read_excel(manual_review_file, sheet_name='Review Data')

    # Create final category column
    auto_classified_df = auto_classified_df.copy()
    auto_classified_df['final_category'] = auto_classified_df['auto_category']
    auto_classified_df['reviewer_notes'] = ''

    # Update with manual reviews
    for idx, row in manual_df.iterrows():
        if pd.notna(row.get('manual_category')) and str(row['manual_category']).strip():
            # Find matching segment
            mask = (
                (auto_classified_df['participant_id'] == row['participant_id']) &
                (auto_classified_df['segment_id'] == row['segment_id']) &
                (auto_classified_df['game'] == row['game'])
            )
            auto_classified_df.loc[mask, 'final_category'] = str(row['manual_category']).strip().lower()
            if 'reviewer_notes' in row and pd.notna(row.get('reviewer_notes')):
                auto_classified_df.loc[mask, 'reviewer_notes'] = row['reviewer_notes']

    print("Merged manual reviews")
    print(f"\nFinal category breakdown:")
    print(auto_classified_df['final_category'].value_counts())

    return auto_classified_df


def time_str_to_seconds(time_str):
    """Convert HH:MM:SS to seconds"""
    parts = str(time_str).split(':')
    try:
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return int(h * 3600 + m * 60 + s)
        elif len(parts) == 2:
            m, s = map(float, parts)
            return int(m * 60 + s)
    except ValueError:
        pass
    return 0


def link_speech_to_movements(speech_segment, game_data, window_seconds=10):
    """
    Find movements that occurred within X seconds after speech segment

    Args:
        speech_segment: Row from classified speech DataFrame
        game_data: Loaded JSON game data
        window_seconds: How many seconds after speech to look for movements

    Returns:
        List of movements
    """
    # Parse speech end time (format: HH:MM:SS or ISO)
    end_time_str = speech_segment.get('end_time', '')
    if not end_time_str:
        end_time_str = speech_segment.get('start_time', '')

    # Handle different time formats
    if 'T' in str(end_time_str) or 'Z' in str(end_time_str):
        # ISO format
        try:
            speech_end = datetime.fromisoformat(str(end_time_str).replace('Z', '+00:00'))
            window_end = speech_end + timedelta(seconds=window_seconds)
            use_datetime = True
        except:
            return []
    else:
        # HH:MM:SS format
        speech_end_seconds = time_str_to_seconds(str(end_time_str))
        window_end_seconds = speech_end_seconds + window_seconds
        use_datetime = False

    # Get movements in window
    relevant_movements = []

    if 'movements' not in game_data:
        return relevant_movements

    for move in game_data['movements']:
        if 'timestamp' not in move:
            continue

        move_timestamp = move['timestamp']

        if use_datetime:
            try:
                move_time = datetime.fromisoformat(str(move_timestamp).replace('Z', '+00:00'))
                if speech_end <= move_time <= window_end:
                    relevant_movements.append(move)
            except:
                continue
        else:
            # Try to extract seconds from movement timestamp
            try:
                if 'T' in str(move_timestamp):
                    # ISO format - extract time part
                    time_part = str(move_timestamp).split('T')[1].split('.')[0].split('Z')[0]
                    move_seconds = time_str_to_seconds(time_part)
                else:
                    move_seconds = time_str_to_seconds(str(move_timestamp))

                if speech_end_seconds <= move_seconds <= window_end_seconds:
                    relevant_movements.append(move)
            except:
                continue

    return relevant_movements


def extract_movement_features(movements):
    """Extract features from movement sequence"""
    if len(movements) == 0:
        return {
            'num_moves': 0,
            'movement_entropy': 0,
            'direction_changes': 0,
            'prop_direction_changes': 0,
            'repeated_sequences': 0,
            'unique_directions': 0,
            'unique_positions': 0,
            'num_revisits': 0,
            'color_changes': 0
        }

    directions = [m.get('direction', '') for m in movements]
    positions = [(m.get('positionBefore', {}).get('x', 0),
                  m.get('positionBefore', {}).get('y', 0)) for m in movements]
    colors = [m.get('colorBefore', '') for m in movements]

    # Filter out empty values
    directions = [d for d in directions if d]

    if len(directions) == 0:
        return {
            'num_moves': len(movements),
            'movement_entropy': 0,
            'direction_changes': 0,
            'prop_direction_changes': 0,
            'repeated_sequences': 0,
            'unique_directions': 0,
            'unique_positions': len(set(positions)),
            'num_revisits': len(positions) - len(set(positions)),
            'color_changes': 0
        }

    # Calculate entropy
    counts = Counter(directions)
    probs = [c/len(directions) for c in counts.values()]
    entropy_val = -sum(p * np.log2(p) for p in probs if p > 0)

    # Direction changes
    direction_changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])

    # Repeated sequences (2-move patterns)
    if len(directions) > 1:
        sequences = [tuple(directions[i:i+2]) for i in range(len(directions)-1)]
        repeated_sequences = len(sequences) - len(set(sequences))
    else:
        repeated_sequences = 0

    # Unique positions visited
    unique_positions = len(set(positions))

    # Backtracking (revisiting positions)
    num_revisits = len(positions) - unique_positions

    # Color changes (indicates learning)
    colors = [c for c in colors if c]
    color_changes = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i-1]) if len(colors) > 1 else 0

    return {
        'num_moves': len(movements),
        'movement_entropy': entropy_val,
        'direction_changes': direction_changes,
        'prop_direction_changes': direction_changes / max(1, len(directions) - 1),
        'repeated_sequences': repeated_sequences,
        'unique_directions': len(set(directions)),
        'unique_positions': unique_positions,
        'num_revisits': num_revisits,
        'color_changes': color_changes
    }


def create_final_dataset(classified_speech_df, game_data_dir, participant_tracker=None, output_file=None):
    """
    Create final dataset linking speech categories to movement features

    Args:
        classified_speech_df: DataFrame with classified speech segments
        game_data_dir: Directory containing game JSON files
        participant_tracker: ParticipantTracker instance
        output_file: Where to save final CSV
    """
    if output_file is None:
        output_file = OUTPUT_DIR / "NLP_features_for_analysis.xlsx"

    results = []
    game_data_path = Path(game_data_dir)

    # Determine which category column to use
    category_col = 'final_category' if 'final_category' in classified_speech_df.columns else 'auto_category'

    # Build a lookup from participant_id + game -> JSON file
    json_files = list(game_data_path.glob('*.json'))

    for idx, speech_row in classified_speech_df.iterrows():
        participant_id = speech_row['participant_id']
        game = speech_row['game']

        # Find matching game data file
        game_file = None
        for jf in json_files:
            if participant_tracker:
                jf_participant = participant_tracker.get_participant_id_from_json(jf.name, game)
                if jf_participant == participant_id:
                    game_file = jf
                    break

        if game_file is None:
            continue

        try:
            with open(game_file) as f:
                game_data = json.load(f)
        except Exception as e:
            print(f"Error loading {game_file}: {e}")
            continue

        # Get movements linked to this speech
        movements = link_speech_to_movements(speech_row, game_data, window_seconds=10)

        # Skip if too few movements
        if len(movements) < 3:
            continue

        # Extract movement features
        features = extract_movement_features(movements)

        # Combine speech and movement data
        row_data = {
            'participant_id': participant_id,
            'game': game,
            'segment_id': speech_row['segment_id'],
            'speech_text': speech_row['text'],
            'speech_category': speech_row[category_col],
            'confidence': speech_row['confidence'],
            **features  # Movement features
        }

        results.append(row_data)

    final_df = pd.DataFrame(results)

    if len(final_df) > 0:
        final_df.to_excel(output_file, index=False)
        print(f"\nCreated final dataset: {output_file}")
        print(f"{len(final_df)} speech-movement pairs")
    else:
        print("\nWarning: No speech-movement pairs were created.")
        print("Check that your game data files exist and match the expected format.")

    return final_df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline():
    """Run the complete NLP classification pipeline with file picker dialogs"""

    print("=" * 60)
    print("NLP SPEECH CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"\nAll outputs will be saved to: {OUTPUT_DIR}")

    # Step 1: Select Participant Tracker CSV
    print("\n" + "=" * 60)
    print("FILE 1 of 2: PARTICIPANT TRACKER")
    print("=" * 60)
    print("Select the CSV file that maps Session IDs to game files.")
    print("File name example: 'Participant Tracker.csv'")
    print("Contains columns: Session ID, Game A Data, Game B Data, etc.")
    print("=" * 60)
    input(">>> Press ENTER to open file picker...")
    tracker_file = select_file(
        "FILE 1: Select Participant Tracker CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not tracker_file:
        print("No file selected. Exiting.")
        return None
    print(f"  Selected: {tracker_file}")

    # Load participant tracker
    print("\nLoading participant tracker...")
    participant_tracker = ParticipantTracker(tracker_file)

    # Step 2: Select Puzzle A transcript folder
    print("\n" + "=" * 60)
    print("FOLDER 1 of 4: PUZZLE A (GAME 1) TRANSCRIPTS")
    print("=" * 60)
    print("Select the FOLDER containing Puzzle A / Game 1 transcript files (.txt)")
    print("These are the Whisper transcription output files.")
    print("File names look like: 'puzzle-game1-audio-2026-01-05T17-15-03_transcription.txt'")
    print("=" * 60)
    input(">>> Press ENTER to open folder picker...")
    transcript_dir_a = select_folder("FOLDER 1: Select Puzzle A Transcripts folder")
    if not transcript_dir_a:
        print("No folder selected. Exiting.")
        return None
    print(f"  Selected: {transcript_dir_a}")

    # Step 3: Select Puzzle A game-state folder
    print("\n" + "=" * 60)
    print("FOLDER 2 of 4: PUZZLE A (GAME 1) GAME-STATE DATA")
    print("=" * 60)
    print("Select the FOLDER containing Puzzle A / Game 1 game-state JSON files.")
    print("These contain movement data and level timing information.")
    print("File names look like: 'puzzle-game1-state-2026-01-05T17-15-03-300Z.json'")
    print("=" * 60)
    input(">>> Press ENTER to open folder picker...")
    gamestate_dir_a = select_folder("FOLDER 2: Select Puzzle A Game-State folder")
    if not gamestate_dir_a:
        print("No folder selected - will proceed without level detection for Puzzle A.")
        gamestate_dir_a = None
    else:
        print(f"  Selected: {gamestate_dir_a}")

    # Step 4: Select Puzzle B transcript folder
    print("\n" + "=" * 60)
    print("FOLDER 3 of 4: PUZZLE B (GAME 2) TRANSCRIPTS")
    print("=" * 60)
    print("Select the FOLDER containing Puzzle B / Game 2 transcript files (.txt)")
    print("These are the Whisper transcription output files.")
    print("File names look like: 'puzzle-game2-audio-2026-01-05T17-20-15_transcription.txt'")
    print("=" * 60)
    input(">>> Press ENTER to open folder picker...")
    transcript_dir_b = select_folder("FOLDER 3: Select Puzzle B Transcripts folder")
    if not transcript_dir_b:
        print("No folder selected. Exiting.")
        return None
    print(f"  Selected: {transcript_dir_b}")

    # Step 5: Select Puzzle B game-state folder
    print("\n" + "=" * 60)
    print("FOLDER 4 of 4: PUZZLE B (GAME 2) GAME-STATE DATA")
    print("=" * 60)
    print("Select the FOLDER containing Puzzle B / Game 2 game-state JSON files.")
    print("These contain movement data and level timing information.")
    print("File names look like: 'puzzle-game2-state-2026-01-05T17-20-15-300Z.json'")
    print("=" * 60)
    input(">>> Press ENTER to open folder picker...")
    gamestate_dir_b = select_folder("FOLDER 4: Select Puzzle B Game-State folder")
    if not gamestate_dir_b:
        print("No folder selected - will proceed without level detection for Puzzle B.")
        gamestate_dir_b = None
    else:
        print(f"  Selected: {gamestate_dir_b}")

    # Step 4: Process transcripts with NLP
    print("\n" + "=" * 50)
    print("STEP 1: Classifying speech segments with NLP")
    print("=" * 50)

    # Process both folders
    print("\nProcessing Puzzle A transcripts...")
    classified_df_a = process_all_transcripts(
        transcript_dir_a,
        participant_tracker=participant_tracker,
        gamestate_dir=gamestate_dir_a,
        apply_context=True
    )

    print("\nProcessing Puzzle B transcripts...")
    classified_df_b = process_all_transcripts(
        transcript_dir_b,
        participant_tracker=participant_tracker,
        gamestate_dir=gamestate_dir_b,
        apply_context=True
    )

    # Combine results
    if len(classified_df_a) > 0 and len(classified_df_b) > 0:
        classified_df = pd.concat([classified_df_a, classified_df_b], ignore_index=True)
    elif len(classified_df_a) > 0:
        classified_df = classified_df_a
    elif len(classified_df_b) > 0:
        classified_df = classified_df_b
    else:
        classified_df = pd.DataFrame()

    # Save combined results with summary statistics
    if len(classified_df) > 0:
        output_file = OUTPUT_DIR / "classified_speech_segments.xlsx"

        # Calculate summary statistics
        total_segments = len(classified_df)
        confidence_counts = classified_df['confidence'].value_counts()
        high_count = confidence_counts.get('high', 0)
        medium_count = confidence_counts.get('medium', 0)
        low_count = confidence_counts.get('low', 0)

        high_pct = (high_count / total_segments * 100) if total_segments > 0 else 0
        medium_pct = (medium_count / total_segments * 100) if total_segments > 0 else 0
        low_pct = (low_count / total_segments * 100) if total_segments > 0 else 0

        needs_review_count = classified_df['needs_review'].sum()
        needs_review_pct = (needs_review_count / total_segments * 100) if total_segments > 0 else 0

        # Create summary dataframe
        summary_data = {
            'Metric': [
                'Total Segments',
                '',
                'High Confidence',
                'Medium Confidence',
                'Low Confidence',
                '',
                'Needs Manual Review'
            ],
            'Count': [
                total_segments,
                '',
                high_count,
                medium_count,
                low_count,
                '',
                needs_review_count
            ],
            'Percentage': [
                '100%',
                '',
                f'{high_pct:.1f}%',
                f'{medium_pct:.1f}%',
                f'{low_pct:.1f}%',
                '',
                f'{needs_review_pct:.1f}%'
            ]
        }
        summary_df = pd.DataFrame(summary_data)

        # Save with Summary sheet first
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            classified_df.to_excel(writer, sheet_name='Classified Segments', index=False)

        print(f"\nSaved combined classified segments to {output_file}")
        print(f"\n--- SUMMARY STATISTICS ---")
        print(f"Total segments: {total_segments}")
        print(f"  High confidence:   {high_count} ({high_pct:.1f}%)")
        print(f"  Medium confidence: {medium_count} ({medium_pct:.1f}%)")
        print(f"  Low confidence:    {low_count} ({low_pct:.1f}%)")
        print(f"  Needs manual review: {needs_review_count} ({needs_review_pct:.1f}%)")
        print(f"\nCategory breakdown:")
        print(classified_df['auto_category'].value_counts())

    if len(classified_df) == 0:
        print("\nPipeline stopped: No transcripts to process.")
        return None

    # Step 4: Create manual review file
    print("\n" + "=" * 50)
    print("STEP 2: Creating manual review file")
    print("=" * 50)
    create_manual_review_file(classified_df)

    print("\n" + "=" * 50)
    print(">>> PAUSE HERE <<<")
    print(f">>> Manually review 'manual_review_needed.xlsx' in {OUTPUT_DIR} <<<")
    print(">>> Then run: continue_pipeline_after_review() <<<")
    print("=" * 50)

    # Cleanup file selector
    _file_selector.cleanup()

    return classified_df


def continue_pipeline_after_review():
    """Continue pipeline after manual review is complete"""

    print("\n" + "=" * 60)
    print("CONTINUING NLP PIPELINE AFTER MANUAL REVIEW")
    print("=" * 60)

    # Select Participant Tracker CSV again
    print("\nPlease select the PARTICIPANT TRACKER CSV file...")
    tracker_file = select_file(
        "Select Participant Tracker CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not tracker_file:
        print("No file selected. Exiting.")
        return None

    participant_tracker = ParticipantTracker(tracker_file)

    # Prompt for game data folder
    print("\nPlease select your GAME DATA folder (JSON files)...")
    game_data_dir = select_folder("Select folder containing game data files (.json)")
    if not game_data_dir:
        print("No folder selected. Exiting.")
        return None
    print(f"  Game data: {game_data_dir}")

    # Load the classified data
    classified_file = OUTPUT_DIR / "classified_speech_segments.xlsx"
    if not classified_file.exists():
        print(f"Error: Could not find {classified_file}")
        print("Please run run_full_pipeline() first.")
        return None

    classified_df = pd.read_csv(classified_file)

    # Merge manual reviews
    print("\n" + "=" * 50)
    print("STEP 3: Merging manual reviews")
    print("=" * 50)
    final_classified_df = merge_manual_reviews(classified_df)
    final_classified_df.to_excel(OUTPUT_DIR / "final_classified_segments.xlsx", index=False)

    # Link to movements and create final dataset
    print("\n" + "=" * 50)
    print("STEP 4: Linking speech to movements")
    print("=" * 50)
    final_dataset = create_final_dataset(
        final_classified_df,
        game_data_dir,
        participant_tracker=participant_tracker
    )

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - classified_speech_segments.xlsx")
    print("  - manual_review_needed.xlsx")
    print("  - final_classified_segments.xlsx")
    print("  - NLP_features_for_analysis.xlsx")
    print("\nReady for data_analysis.py!")

    # Cleanup file selector
    _file_selector.cleanup()

    return final_dataset


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Automatically run the full pipeline when script is executed
    run_full_pipeline()
