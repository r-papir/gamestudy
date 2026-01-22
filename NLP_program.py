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
        self.excluded_participants = set()

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


class SpeechCategoryClassifier:
    """Classify speech segments based on linguistic markers"""

    def __init__(self):
        # Linguistic markers for each category
        # Exploratory = explore, Confirmatory = establish, Exploitative = exploit
        self.exploratory_markers = {
            'questions': ['what', 'why', 'how', 'where', 'which', 'when'],
            'uncertainty': ['maybe', 'might', 'could', 'perhaps', 'wonder', 'not sure',
                           "don't know", 'trying to figure', 'confused', 'hmm', 'uh'],
            'exploration_verbs': ['exploring', 'looking', 'checking', 'seeing',
                                  'trying different', 'experimenting', 'random', 'randomly'],
            'hedges': ['I think maybe', 'kind of', 'sort of', 'seems like']
        }

        self.confirmatory_markers = {
            'hypothesis_statements': ['I think', 'my hypothesis', 'if...then',
                                     'probably', 'should be', 'seems like', 'bet'],
            'testing_language': ['let me test', 'testing', 'checking if', 'see if',
                                'trying to confirm', 'verify', 'test whether',
                                'gonna try', 'let me see if', 'let me check',
                                "I'm going to test", 'confirm'],
            'conditional': ['if I', 'when I', 'if this', 'assuming', 'suppose'],
            'prediction': ['will', 'should', 'expect', 'predict', 'would']
        }

        self.exploitative_markers = {
            'certainty': ['I know', 'definitely', 'obviously', 'clearly', 'for sure',
                         'certain', 'figured it out', 'got it', 'understand'],
            'execution': ["now I'll just", 'just need to', 'all I have to do',
                         'simply', 'just going to', 'now I can', 'okay now',
                         'alright now', 'just', 'easy'],
            'completion': ['almost done', 'finish', 'complete', 'final step',
                          'last thing', 'done', 'solved']
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

        # Flag for manual review if:
        # 1. No markers matched
        # 2. Tie between categories
        # 3. Low confidence
        if max_score == 0:
            return 'NEEDS_MANUAL_REVIEW', 0.0, scores, markers_matched

        max_categories = [cat for cat, score in scores.items() if score == max_score]
        if len(max_categories) > 1:
            return 'NEEDS_MANUAL_REVIEW', 0.0, scores, markers_matched

        category = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0

        if confidence < 0.6:
            return 'NEEDS_MANUAL_REVIEW', confidence, scores, markers_matched

        return category, confidence, scores, markers_matched


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


def process_transcript_file(transcript_file, participant_tracker=None, participant_id=None, game_name=None):
    """
    Process one participant's transcript file

    Args:
        transcript_file: Path to transcript file (.txt from Whisper or .json)
        participant_tracker: ParticipantTracker instance for ID lookup
        participant_id: Participant ID (optional, will try to extract)
        game_name: 'Game A', 'Game B', or 'Game C' (optional, will try to extract)

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

        results.append({
            'participant_id': participant_id,
            'game': game_name,
            'segment_id': segment_id,
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'audio_filename': filename,
            'auto_category': category,
            'confidence': confidence,
            'exploratory_score': scores['exploratory'],
            'confirmatory_score': scores['confirmatory'],
            'exploitative_score': scores['exploitative'],
            'exploratory_markers': ', '.join(markers['exploratory']),
            'confirmatory_markers': ', '.join(markers['confirmatory']),
            'exploitative_markers': ', '.join(markers['exploitative']),
            'needs_review': category == 'NEEDS_MANUAL_REVIEW'
        })

    return pd.DataFrame(results)


def process_all_transcripts(transcript_dir, participant_tracker=None, output_file=None):
    """
    Process all transcript files in a directory

    Args:
        transcript_dir: Directory containing transcript files (.txt or .json)
        participant_tracker: ParticipantTracker instance for ID lookup
        output_file: Where to save the classified segments CSV
    """
    if output_file is None:
        output_file = OUTPUT_DIR / "classified_speech_segments.csv"

    all_results = []

    # Look for both .txt (Whisper) and .json files
    transcript_path = Path(transcript_dir)
    transcript_files = list(transcript_path.glob('*.txt')) + list(transcript_path.glob('*.json'))

    print(f"Found {len(transcript_files)} transcript files in {transcript_dir}")

    if len(transcript_files) == 0:
        print(f"Warning: No transcript files found in {transcript_dir}")
        print("Make sure your transcriptions are in the correct folder.")
        return pd.DataFrame()

    for transcript_file in transcript_files:
        print(f"Processing {transcript_file.name}...")

        try:
            df = process_transcript_file(transcript_file, participant_tracker=participant_tracker)
            if len(df) > 0:
                all_results.append(df)
                print(f"  -> {len(df)} segments, participant: {df['participant_id'].iloc[0]}")
        except Exception as e:
            print(f"  Error processing {transcript_file.name}: {e}")
            continue

    if not all_results:
        print("No transcripts were successfully processed.")
        return pd.DataFrame()

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save to CSV
    final_df.to_csv(output_file, index=False)

    print(f"\nSaved classified segments to {output_file}")
    print(f"Total segments: {len(final_df)}")
    print(f"Needs manual review: {final_df['needs_review'].sum()}")
    print(f"\nCategory breakdown:")
    print(final_df['auto_category'].value_counts())

    return final_df


def create_manual_review_file(classified_df, output_file=None):
    """
    Create a CSV file with only segments needing manual review
    """
    if output_file is None:
        output_file = OUTPUT_DIR / "manual_review_needed.csv"

    review_df = classified_df[classified_df['needs_review'] == True].copy()

    # Add empty column for manual coding
    review_df['manual_category'] = ''
    review_df['reviewer_notes'] = ''

    # Reorder columns for easier review
    columns_order = [
        'participant_id', 'game', 'segment_id',
        'start_time', 'end_time',  # TIMESTAMPS for finding in audio
        'text', 'audio_filename',
        'exploratory_score', 'confirmatory_score', 'exploitative_score',
        'exploratory_markers', 'confirmatory_markers', 'exploitative_markers',
        'confidence',
        'manual_category',  # YOU FILL THIS IN
        'reviewer_notes'    # YOU FILL THIS IN
    ]

    # Only include columns that exist
    columns_order = [c for c in columns_order if c in review_df.columns]
    review_df = review_df[columns_order]

    # Create key/instructions header
    key_text = """=== MANUAL REVIEW INSTRUCTIONS ===

HOW TO USE THIS FILE:
1. Find the audio file in 'audio_filename' column
2. Jump to 'start_time' in the audio to hear the segment
3. Read the 'text' and listen to context
4. Enter your classification in 'manual_category': exploratory OR confirmatory OR exploitative
5. (Optional) Add notes in 'reviewer_notes'

=== SCORE DEFINITIONS ===

EXPLORATORY SCORE (explore): Counts markers of uncertainty and exploration
  Markers: what, why, how, where, which, when, maybe, might, could, perhaps, wonder, not sure, don't know, trying to figure, confused, hmm, uh, exploring, looking, checking, seeing, trying different, experimenting, random, randomly, I think maybe, kind of, sort of, seems like

CONFIRMATORY SCORE (establish): Counts markers of hypothesis testing and prediction
  Markers: I think, my hypothesis, if...then, probably, should be, seems like, bet, let me test, testing, checking if, see if, trying to confirm, verify, test whether, gonna try, let me see if, let me check, I'm going to test, confirm, if I, when I, if this, assuming, suppose, will, should, expect, predict, would

EXPLOITATIVE SCORE (exploit): Counts markers of certainty and execution
  Markers: I know, definitely, obviously, clearly, for sure, certain, figured it out, got it, understand, now I'll just, just need to, all I have to do, simply, just going to, now I can, okay now, alright now, just, easy, almost done, finish, complete, final step, last thing, done, solved

=== WHY SEGMENTS NEED REVIEW ===
- All scores are 0: No markers detected - use context to judge
- Scores are tied: Unclear winner - use your judgment
- Low confidence: Winner didn't have a strong lead

=== DATA STARTS BELOW ==="""

    # Write file with key at top
    with open(output_file, 'w') as f:
        for line in key_text.strip().split('\n'):
            f.write(f"# {line}\n")
        f.write("#\n")

    # Append the data
    review_df.to_csv(output_file, mode='a', index=False)

    print(f"\nCreated manual review file: {output_file}")
    print(f"{len(review_df)} segments need review")
    print("\nInstructions:")
    print("1. Open the CSV file in Excel or Google Sheets")
    print("2. Jump to 'start_time' in the audio file to hear each segment")
    print("3. Fill in 'manual_category' column with: exploratory, confirmatory, or exploitative")
    print("4. Add any notes in 'reviewer_notes' column")
    print("5. Save the file")

    return review_df


def merge_manual_reviews(auto_classified_df, manual_review_file=None):
    """
    Merge manual reviews back into the main dataframe
    """
    if manual_review_file is None:
        manual_review_file = OUTPUT_DIR / "manual_review_needed.csv"

    # Load manual reviews (skip comment lines starting with #)
    manual_df = pd.read_csv(manual_review_file, comment='#')

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
        output_file = OUTPUT_DIR / "NLP_features_for_analysis.csv"

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
        final_df.to_csv(output_file, index=False)
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
    print("FOLDER 1 of 2: PUZZLE A (GAME 1) TRANSCRIPTS")
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

    # Step 3: Select Puzzle B transcript folder
    print("\n" + "=" * 60)
    print("FOLDER 2 of 2: PUZZLE B (GAME 2) TRANSCRIPTS")
    print("=" * 60)
    print("Select the FOLDER containing Puzzle B / Game 2 transcript files (.txt)")
    print("These are the Whisper transcription output files.")
    print("File names look like: 'puzzle-game2-audio-2026-01-05T17-20-15_transcription.txt'")
    print("=" * 60)
    input(">>> Press ENTER to open folder picker...")
    transcript_dir_b = select_folder("FOLDER 2: Select Puzzle B Transcripts folder")
    if not transcript_dir_b:
        print("No folder selected. Exiting.")
        return None
    print(f"  Selected: {transcript_dir_b}")

    # Step 4: Process transcripts with NLP
    print("\n" + "=" * 50)
    print("STEP 1: Classifying speech segments with NLP")
    print("=" * 50)

    # Process both folders
    print("\nProcessing Puzzle A transcripts...")
    classified_df_a = process_all_transcripts(transcript_dir_a, participant_tracker=participant_tracker)

    print("\nProcessing Puzzle B transcripts...")
    classified_df_b = process_all_transcripts(transcript_dir_b, participant_tracker=participant_tracker)

    # Combine results
    if len(classified_df_a) > 0 and len(classified_df_b) > 0:
        classified_df = pd.concat([classified_df_a, classified_df_b], ignore_index=True)
    elif len(classified_df_a) > 0:
        classified_df = classified_df_a
    elif len(classified_df_b) > 0:
        classified_df = classified_df_b
    else:
        classified_df = pd.DataFrame()

    # Save combined results
    if len(classified_df) > 0:
        output_file = OUTPUT_DIR / "classified_speech_segments.csv"
        classified_df.to_csv(output_file, index=False)
        print(f"\nSaved combined classified segments to {output_file}")
        print(f"Total segments: {len(classified_df)}")
        print(f"Needs manual review: {classified_df['needs_review'].sum()}")
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
    print(f">>> Manually review 'manual_review_needed.csv' in {OUTPUT_DIR} <<<")
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
    classified_file = OUTPUT_DIR / "classified_speech_segments.csv"
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
    final_classified_df.to_csv(OUTPUT_DIR / "final_classified_segments.csv", index=False)

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
    print("  - classified_speech_segments.csv")
    print("  - manual_review_needed.csv")
    print("  - final_classified_segments.csv")
    print("  - NLP_features_for_analysis.csv")
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
