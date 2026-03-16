"""Quick test: file path vs numpy array input to Whisper on the 30-min clip."""
import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

AUDIO_FILE = "c:/_/Whisper/test_new/1_16k_mono_nr090.wav"
CLIP_FILE = "c:/_/Whisper/test_new/clip_30m_nr090.wav"

# Create a 30-minute clip file on disk
print("Creating 30-minute clip file...")
audio, sr = sf.read(AUDIO_FILE, dtype='float32')
clip = audio[:30*60*sr]
sf.write(CLIP_FILE, clip, sr, subtype='PCM_16')
print(f"  Saved {CLIP_FILE} ({len(clip)/sr/60:.1f} min)")

print("\nLoading model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

PROMPT = "以下是一节关于空气动力学的大学课程录音。"

# Test A: file path
print("\n=== Test A: FILE PATH ===")
t0 = time.time()
segs_a, _ = model.transcribe(CLIP_FILE, language="zh", beam_size=5,
    vad_filter=True, condition_on_previous_text=True,
    repetition_penalty=1.2, no_repeat_ngram_size=3, initial_prompt=PROMPT)
count_a = 0
for s in segs_a:
    count_a += 1
print(f"  Segments: {count_a} in {time.time()-t0:.1f}s")

# Test B: numpy array (float32)
print("\n=== Test B: NUMPY FLOAT32 ===")
t0 = time.time()
segs_b, _ = model.transcribe(clip, language="zh", beam_size=5,
    vad_filter=True, condition_on_previous_text=True,
    repetition_penalty=1.2, no_repeat_ngram_size=3, initial_prompt=PROMPT)
count_b = 0
for s in segs_b:
    count_b += 1
print(f"  Segments: {count_b} in {time.time()-t0:.1f}s")

print(f"\nResult: filepath={count_a} segs, numpy={count_b} segs")
