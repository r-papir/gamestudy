import whisper
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import re

def transcription():
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_puzzle_label(filename):
        match = re.search(r'_g([ABC])_', filename, re.IGNORECASE)
        if match:
            return f"Puzzle {match.group(1).upper()}"
        return None

    Tk().withdraw()

    output_dir = "/Users/rachelpapirmeister/Documents/TAP_transcriptions"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Whisper model...")
    model = whisper.load_model("large-v3")

    while True:
        print("\nSelect an audio file to transcribe (cancel to stop)...")
        audio_path = askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.flac *.webm"), ("All files", "*.*")]
        )

        if not audio_path:
            print("No file selected. Exiting.")
            break

        print(f"Processing: {os.path.basename(audio_path)}")

        try:
            result = model.transcribe(audio_path, language="en")

            filename = os.path.basename(audio_path).rsplit('.', 1)[0]
            output_file = os.path.join(output_dir, filename + '_transcription.txt')

            with open(output_file, 'w') as f:
                puzzle_label = get_puzzle_label(filename)
                if puzzle_label:
                    f.write(f"=== {puzzle_label} ===\n\n")
                for segment in result["segments"]:
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"[{start} - {end}] {text}\n")

            print(f"✓ Transcription saved to: {output_file}")
            preview = result["text"][:100] if result["text"] else ""
            print(f"  Preview: {preview}...\n")

        except Exception as e:
            print(f"✗ Error processing {os.path.basename(audio_path)}: {str(e)}\n")

    print("Done!")

transcription()