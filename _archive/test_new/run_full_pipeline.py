import os
import time
import numpy as np
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel

INPUT_FILE = "c:/_/Whisper/test_new/1_16k_mono.wav"
DENOISED_FILE = "c:/_/Whisper/test_new/1_16k_mono_nr090.wav"
OUTPUT_TXT = "c:/_/Whisper/test_new/1_16k_mono_transcript.txt"

PROMPT = (
    "以下是一节关于空气动力学的大学课程录音。包含专业术语如："
    "法向、切向、流线、迹线、脉线、马赫数、雷诺数、层流、湍流、"
    "边界层、压强梯度、伯努利方程、欧拉方程、连续性方程、速度势、"
    "势流、涡量、环量、升力、阻力、攻角、翼型。"
)

# ========== Step 1: Denoise ==========
print("Step 1: Loading audio...")
audio, sr = sf.read(INPUT_FILE, dtype='float64')
print(f"  Loaded: {len(audio)/sr/60:.1f} min, sr={sr}")

print("Step 2: Denoising with noisereduce (prop_decrease=0.90)...")
t0 = time.time()
audio_denoised = nr.reduce_noise(
    y=audio, sr=sr,
    stationary=True,
    prop_decrease=0.90,
    n_fft=2048,
    freq_mask_smooth_hz=500
)
print(f"  Denoised in {time.time()-t0:.1f}s")

# ========== Step 2: Peak Normalize ==========
print("Step 3: Peak normalization to 0.95...")
peak = np.max(np.abs(audio_denoised))
audio_norm = audio_denoised * (0.95 / peak)
print(f"  Peak: {peak:.4f} -> 0.9500")

# Save denoised file
sf.write(DENOISED_FILE, audio_norm, sr, subtype='PCM_16')
print(f"  Saved: {DENOISED_FILE}")

# ========== Step 3: Transcribe ==========
print("Step 4: Loading Whisper model (large-v3, fp16)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("  Model loaded.")

print("Step 5: Transcribing with W08 optimal parameters...")
t0 = time.time()
segments, info = model.transcribe(
    audio_norm.astype(np.float32),
    language="zh",
    beam_size=5,
    temperature=0,
    vad_filter=True,
    condition_on_previous_text=True,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    hallucination_silence_threshold=2.0,
    initial_prompt=PROMPT,
)

# Write output
seg_count = 0
char_count = 0
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for s in segments:
        seg_count += 1
        char_count += len(s.text)
        f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
        if seg_count % 100 == 0:
            print(f"  ... {seg_count} segments processed")

elapsed = time.time() - t0
print(f"\nDONE! Transcription complete.")
print(f"  Segments: {seg_count}")
print(f"  Characters: {char_count}")
print(f"  Time: {elapsed:.1f}s")
print(f"  Output: {OUTPUT_TXT}")
