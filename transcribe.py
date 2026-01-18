import speech_recognition as sr
from pydub import AudioSegment
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import os

Tk().withdraw()

output_dir = "/Users/rachel/Documents/TAP_transcriptions"
os.makedirs(output_dir, exist_ok=True)

print("Select audio files to transcribe...")
audio_files = askopenfilenames(
    title="Select audio files",
    filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.flac"), ("All files", "*.*")]
)

if not audio_files:
    print("No files selected. Exiting.")
    exit()

print(f"\nSelected {len(audio_files)} file(s)\n")

r = sr.Recognizer()

for audio_path in audio_files:
    print(f"Processing: {os.path.basename(audio_path)}")
    
    try:
        if not audio_path.lower().endswith('.wav'):
            sound = AudioSegment.from_file(audio_path)
            wav_path = audio_path.rsplit('.', 1)[0] + '_temp.wav'
            sound.export(wav_path, format="wav")
        else:
            wav_path = audio_path
        
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        
        transcription = r.recognize_google(audio)
        
        filename = os.path.basename(audio_path).rsplit('.', 1)[0]
        output_file = os.path.join(output_dir, filename + '_transcription.txt')
        with open(output_file, 'w') as f:
            f.write(transcription)
        
        print(f"✓ Transcription saved to: {output_file}")
        print(f"  Text: {transcription[:100]}...\n")
        
        if wav_path != audio_path:
            os.remove(wav_path)
            
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(audio_path)}: {str(e)}\n")

print("All files processed!")