import whisper
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import os
import re

def transcription():
    def fmt_mmss(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    Tk().withdraw()

    output_dir = "/Users/rachelpapirmeister/Documents/TAP_transcriptions"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Whisper model...")
    model = whisper.load_model("large-v3")

    print("\nSelect audio files to transcribe...")
    audio_paths = askopenfilenames(
        title="Select audio files",
        filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.flac *.webm"), ("All files", "*.*")]
    )

    if not audio_paths:
        print("No files selected. Exiting.")
        return

    print(f"{len(audio_paths)} file(s) selected.\n")

    for audio_path in audio_paths:
        print(f"Processing: {os.path.basename(audio_path)}")

        try:
            result = model.transcribe(audio_path, language="en", word_timestamps=True)

            base = os.path.basename(audio_path).rsplit('.', 1)[0]
            m = re.match(r'^(P\d+_g[ABC])_audio_(\d{8})$', base, re.IGNORECASE)
            if m:
                prefix, date = m.group(1), m.group(2)
                txt_file  = os.path.join(output_dir, f"{prefix}_audio_{date}.txt")
                json_file = os.path.join(output_dir, f"{prefix}_transcript_{date}.json")
            else:
                txt_file  = os.path.join(output_dir, base + "_transcription.txt")
                json_file = os.path.join(output_dir, base + "_transcript.json")

            # Human-readable .txt in MM:SS format (compatible with replay.html)
            with open(txt_file, 'w') as f:
                for seg in result["segments"]:
                    f.write(fmt_mmss(seg["start"]) + "\n")
                    f.write(seg["text"].strip() + "\n\n")

            # Word-timestamp .json for word-by-word highlighting in replay.html
            words = []
            for seg in result["segments"]:
                for w in seg.get("words", []):
                    words.append({
                        "word":  w["word"].strip(),
                        "start": round(w["start"], 3),
                        "end":   round(w["end"],   3),
                    })
            with open(json_file, 'w') as f:
                json.dump({"words": words}, f)

            print(f"✓ Text saved to:  {txt_file}")
            print(f"✓ Words saved to: {json_file}")
            preview = result["text"][:100] if result["text"] else ""
            print(f"  Preview: {preview}...\n")

        except Exception as e:
            print(f"✗ Error processing {os.path.basename(audio_path)}: {str(e)}\n")

    print("All done!")

transcription()
