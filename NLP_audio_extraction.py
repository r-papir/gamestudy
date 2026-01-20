import json
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
from tkinter import Tk, filedialog

# All data files read from and saved to Downloads (not in git repo)
DATA_DIR = "/Users/rachelpapirmeister/Downloads"
TRANSCRIPT_DIR = None  # Will be set by user via file picker
GAME_DATA_DIR = None   # Will be set by user via file picker
OUTPUT_DIR = DATA_DIR


def select_folder(title="Select a folder"):
    """Open a folder selection dialog"""
    Tk().withdraw()
    folder = filedialog.askdirectory(title=title, initialdir=DATA_DIR)
    return folder if folder else None


def select_file(title="Select a file", filetypes=None):
    """Open a file selection dialog"""
    if filetypes is None:
        filetypes = [("All files", "*.*")]
    Tk().withdraw()
    file = filedialog.askopenfilename(title=title, initialdir=DATA_DIR, filetypes=filetypes)
    return file if file else None


class SpeechCategoryClassifier:
    """Classify speech segments based on linguistic markers"""

    def __init__(self):
        # Linguistic markers for each category
        self.exploratory_markers = {
            'questions': ['what', 'why', 'how', 'where', 'which', 'when'],
            'uncertainty': ['maybe', 'might', 'could', 'perhaps', 'wonder', 'not sure',
                           'don\'t know', 'trying to figure', 'confused', 'hmm', 'uh'],
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
                                'I\'m going to test', 'confirm'],
            'conditional': ['if I', 'when I', 'if this', 'assuming', 'suppose'],
            'prediction': ['will', 'should', 'expect', 'predict', 'would']
        }

        self.exploitative_markers = {
            'certainty': ['I know', 'definitely', 'obviously', 'clearly', 'for sure',
                         'certain', 'figured it out', 'got it', 'understand'],
            'execution': ['now I\'ll just', 'just need to', 'all I have to do',
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

    Expected format from transcribe.py:
    === Puzzle B ===

    [00:00:00 - 00:00:05] Hello, this is the beginning.
    [00:00:05 - 00:00:12] Now I'm doing something else.

    Returns:
        dict with 'puzzle_label' and 'segments' list
    """
    segments = []
    puzzle_label = None

    with open(transcript_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract puzzle label if present
    puzzle_match = re.search(r'=== (Puzzle [A-C]) ===', content)
    if puzzle_match:
        puzzle_label = puzzle_match.group(1)

    # Parse timestamp lines: [00:00:00 - 00:00:05] text
    pattern = r'\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\] (.+)'
    matches = re.findall(pattern, content)

    for i, (start_time, end_time, text) in enumerate(matches):
        segments.append({
            'segment_id': i,
            'start_time': start_time,
            'end_time': end_time,
            'text': text.strip()
        })

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

    if 'game1' in filename_lower:
        return 'Game A'
    elif 'game2' in filename_lower:
        return 'Game B'
    elif 'game3' in filename_lower:
        return 'Game C'
    else:
        return 'Unknown'


def parse_participant_from_filename(filename):
    """
    Extract participant ID from filename

    Adjust this function to match your actual naming convention.
    """
    # Try to extract participant ID - adjust pattern as needed
    # Example: if files are named P001_game1_transcription.txt
    match = re.search(r'(P\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)

    # If no participant ID found, use part of filename
    return filename.split('_')[0]


def process_transcript_file(transcript_file, participant_id=None, game_name=None):
    """
    Process one participant's transcript file

    Args:
        transcript_file: Path to transcript file (.txt from Whisper or .json)
        participant_id: Participant ID (optional, will try to extract from filename)
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
            game_name = transcript_data['puzzle_label']
    else:
        # Parse JSON format
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        segments = transcript_data.get('segments', [])

    # Extract participant ID and game from filename if not provided
    if not participant_id:
        participant_id = parse_participant_from_filename(filename)
    if not game_name:
        game_name = parse_game_from_filename(filename)

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


def process_all_transcripts(transcript_dir=TRANSCRIPT_DIR, output_file=None):
    """
    Process all transcript files in a directory

    Args:
        transcript_dir: Directory containing transcript files (.txt or .json)
        output_file: Where to save the classified segments CSV
    """
    if output_file is None:
        output_file = f"{OUTPUT_DIR}/classified_speech_segments.csv"

    all_results = []

    # Look for both .txt (Whisper) and .json files
    transcript_files = list(Path(transcript_dir).glob('*.txt')) + list(Path(transcript_dir).glob('*.json'))

    print(f"Found {len(transcript_files)} transcript files in {transcript_dir}")

    if len(transcript_files) == 0:
        print(f"Warning: No transcript files found in {transcript_dir}")
        print("Make sure your transcriptions are in the correct folder.")
        return pd.DataFrame()

    for transcript_file in transcript_files:
        print(f"Processing {transcript_file.name}...")

        try:
            df = process_transcript_file(transcript_file)
            all_results.append(df)
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
        output_file = f"{OUTPUT_DIR}/manual_review_needed.csv"

    review_df = classified_df[classified_df['needs_review'] == True].copy()

    # Add empty column for manual coding
    review_df['manual_category'] = ''
    review_df['reviewer_notes'] = ''

    # Reorder columns for easier review
    columns_order = [
        'participant_id', 'game', 'segment_id',
        'text',
        'exploratory_score', 'confirmatory_score', 'exploitative_score',
        'exploratory_markers', 'confirmatory_markers', 'exploitative_markers',
        'manual_category',  # YOU FILL THIS IN
        'reviewer_notes'    # YOU FILL THIS IN
    ]

    review_df = review_df[columns_order]
    review_df.to_csv(output_file, index=False)

    print(f"Created manual review file: {output_file}")
    print(f"{len(review_df)} segments need review")
    print("\nInstructions:")
    print("1. Open the CSV file")
    print("2. Fill in 'manual_category' column with: exploratory, confirmatory, or exploitative")
    print("3. Add any notes in 'reviewer_notes' column")
    print("4. Save the file")

    return review_df


def merge_manual_reviews(auto_classified_df, manual_review_file=None):
    """
    Merge manual reviews back into the main dataframe
    """
    if manual_review_file is None:
        manual_review_file = f"{OUTPUT_DIR}/manual_review_needed.csv"

    # Load manual reviews
    manual_df = pd.read_csv(manual_review_file)

    # Create final category column
    auto_classified_df = auto_classified_df.copy()
    auto_classified_df['final_category'] = auto_classified_df['auto_category']
    auto_classified_df['reviewer_notes'] = ''

    # Update with manual reviews
    for idx, row in manual_df.iterrows():
        if pd.notna(row['manual_category']) and str(row['manual_category']).strip():
            # Find matching segment
            mask = (
                (auto_classified_df['participant_id'] == row['participant_id']) &
                (auto_classified_df['segment_id'] == row['segment_id']) &
                (auto_classified_df['game'] == row['game'])
            )
            auto_classified_df.loc[mask, 'final_category'] = str(row['manual_category']).strip().lower()
            if 'reviewer_notes' in row and pd.notna(row['reviewer_notes']):
                auto_classified_df.loc[mask, 'reviewer_notes'] = row['reviewer_notes']

    print("Merged manual reviews")
    print(f"\nFinal category breakdown:")
    print(auto_classified_df['final_category'].value_counts())

    return auto_classified_df


def time_str_to_seconds(time_str):
    """Convert HH:MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
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
    end_time_str = speech_segment['end_time']
    if not end_time_str:
        end_time_str = speech_segment['start_time']

    # Handle different time formats
    if 'T' in str(end_time_str) or 'Z' in str(end_time_str):
        # ISO format
        speech_end_seconds = None  # Will use datetime comparison
        speech_end = datetime.fromisoformat(str(end_time_str).replace('Z', '+00:00'))
        window_end = speech_end + timedelta(seconds=window_seconds)
        use_datetime = True
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


def create_final_dataset(classified_speech_df, game_data_dir=GAME_DATA_DIR, output_file=None):
    """
    Create final dataset linking speech categories to movement features

    Args:
        classified_speech_df: DataFrame with classified speech segments (must have 'final_category')
        game_data_dir: Directory containing game JSON files
        output_file: Where to save final CSV
    """
    if output_file is None:
        output_file = f"{OUTPUT_DIR}/NLP_features_for_LTA.csv"

    results = []

    # Determine which category column to use
    category_col = 'final_category' if 'final_category' in classified_speech_df.columns else 'auto_category'

    for idx, speech_row in classified_speech_df.iterrows():
        participant_id = speech_row['participant_id']
        game = speech_row['game']

        # Try different filename patterns to find game data
        possible_filenames = [
            f"{participant_id}_{game}.json",
            f"{participant_id}_{game.replace(' ', '')}.json",
            f"{participant_id}_game{game[-1]}.json" if game.startswith('Game') else None,
        ]

        game_file = None
        for filename in possible_filenames:
            if filename:
                test_path = Path(game_data_dir) / filename
                if test_path.exists():
                    game_file = test_path
                    break

        if game_file is None:
            # Try to find any matching file
            pattern = f"{participant_id}*{game.replace(' ', '')}*.json"
            matches = list(Path(game_data_dir).glob(pattern))
            if matches:
                game_file = matches[0]

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
        print(f"Created final dataset: {output_file}")
        print(f"{len(final_df)} speech-movement pairs")
    else:
        print("Warning: No speech-movement pairs were created.")
        print("Check that your game data files exist and match the expected format.")

    return final_df


# FULL PIPELINE SCRIPT
def run_full_pipeline():
    """Run the complete analysis pipeline"""
    global TRANSCRIPT_DIR, GAME_DATA_DIR

    print("=" * 60)
    print("NLP SPEECH CLASSIFICATION PIPELINE")
    print("=" * 60)

    # Prompt user to select folders
    print("\nPlease select your TRANSCRIPTS folder...")
    TRANSCRIPT_DIR = select_folder("Select folder containing transcript files (.txt)")
    if not TRANSCRIPT_DIR:
        print("No folder selected. Exiting.")
        return None
    print(f"  Transcripts: {TRANSCRIPT_DIR}")

    print("\nPlease select your GAME DATA folder...")
    GAME_DATA_DIR = select_folder("Select folder containing game data files (.json)")
    if not GAME_DATA_DIR:
        print("No folder selected. Exiting.")
        return None
    print(f"  Game data: {GAME_DATA_DIR}")

    print(f"  Output: {OUTPUT_DIR}")

    # 1. Process transcripts with NLP
    print("\n" + "=" * 50)
    print("STEP 1: Classifying speech segments with NLP")
    print("=" * 50)
    classified_df = process_all_transcripts(transcript_dir=TRANSCRIPT_DIR)

    if len(classified_df) == 0:
        print("\nPipeline stopped: No transcripts to process.")
        return None

    # 2. Create manual review file
    print("\n" + "=" * 50)
    print("STEP 2: Creating manual review file")
    print("=" * 50)
    create_manual_review_file(classified_df)

    print("\n" + "=" * 50)
    print(">>> PAUSE HERE <<<")
    print(">>> Manually review 'manual_review_needed.csv' in Downloads <<<")
    print(">>> Then run: continue_pipeline_after_review() <<<")
    print("=" * 50)

    return classified_df


def continue_pipeline_after_review():
    """Continue pipeline after manual review is complete"""
    global GAME_DATA_DIR

    # Prompt for game data folder
    print("\nPlease select your GAME DATA folder...")
    GAME_DATA_DIR = select_folder("Select folder containing game data files (.json)")
    if not GAME_DATA_DIR:
        print("No folder selected. Exiting.")
        return None
    print(f"  Game data: {GAME_DATA_DIR}")

    # Load the classified data
    classified_file = f"{OUTPUT_DIR}/classified_speech_segments.csv"
    classified_df = pd.read_csv(classified_file)

    # 3. Merge manual reviews
    print("\n" + "=" * 50)
    print("STEP 3: Merging manual reviews")
    print("=" * 50)
    final_classified_df = merge_manual_reviews(classified_df)
    final_classified_df.to_csv(f"{OUTPUT_DIR}/final_classified_segments.csv", index=False)

    # 4. Link to movements and create final dataset
    print("\n" + "=" * 50)
    print("STEP 4: Linking speech to movements")
    print("=" * 50)
    final_dataset = create_final_dataset(final_classified_df, game_data_dir=GAME_DATA_DIR)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"Final dataset saved to: {OUTPUT_DIR}/NLP_features_for_LTA.csv")
    print("Ready for data_viz.py analysis!")

    return final_dataset


# Run if executed directly
if __name__ == "__main__":
    print("NLP Audio Extraction Tool")
    print("-" * 40)
    print("\nUsage:")
    print("  1. Run run_full_pipeline() to start")
    print("  2. Manually review the CSV file")
    print("  3. Run continue_pipeline_after_review() to finish")
    print("\nOr run steps individually:")
    print("  classified_df = process_all_transcripts()")
    print("  create_manual_review_file(classified_df)")
    print("  final_df = merge_manual_reviews(classified_df)")
    print("  final_dataset = create_final_dataset(final_df)")
