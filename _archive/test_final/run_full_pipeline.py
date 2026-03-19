"""
端到端验证脚本：从 0_original.aac 到最终文本
完整执行 SOP 三步流程，并与 test_new 的 V3 输出对比验证稳定性。
"""
import os, time, re, subprocess
import numpy as np
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel

# ======== 路径 ========
SRC_AAC     = "c:/_/Whisper/test_new/0_original.aac"
WORK_DIR    = "c:/_/Whisper/test_final"
WAV_16K     = os.path.join(WORK_DIR, "1_16k_mono.wav")
DENOISED    = os.path.join(WORK_DIR, "1_16k_mono_nr090.wav")
OUTPUT_TXT  = os.path.join(WORK_DIR, "transcript_v3_final.txt")
V3_BASELINE = "c:/_/Whisper/test_new/1_16k_mono_transcript_v3_novad.txt"

os.makedirs(WORK_DIR, exist_ok=True)

PROMPT = (
    "以下是一节关于空气动力学的大学课程录音。包含专业术语如："
    "法向、切向、流线、迹线、脉线、马赫数、雷诺数、层流、湍流、"
    "边界层、压强梯度、伯努利方程、欧拉方程、连续性方程、速度势、"
    "势流、涡量、环量、升力、阻力、攻角、翼型、散度、旋度。"
)

# ================================================================
# Step 0: AAC → 16kHz mono WAV (ffmpeg)
# ================================================================
print("=" * 60)
print("STEP 0: AAC → 16kHz mono WAV")
print("=" * 60)
t0 = time.time()
cmd = [
    r"C:\DownKyi-1.6.1\ffmpeg.exe", "-y", "-i", SRC_AAC,
    "-ar", "16000", "-ac", "1",
    "-c:a", "pcm_s16le", WAV_16K
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ ffmpeg failed:\n{result.stderr}")
    exit(1)

info = sf.info(WAV_16K)
print(f"  Output: {WAV_16K}")
print(f"  Duration: {info.duration:.1f}s ({info.duration/60:.1f} min)")
print(f"  Samplerate: {info.samplerate}, Channels: {info.channels}")
print(f"  Done in {time.time()-t0:.1f}s")

# Validation
assert info.samplerate == 16000, "Samplerate must be 16kHz"
assert info.channels == 1, "Must be mono"
assert info.duration > 6000, "Audio too short"
print("✅ STEP 0 PASSED\n")

# ================================================================
# Step 1: Denoise + Normalize
# ================================================================
print("=" * 60)
print("STEP 1: Denoise (noisereduce) + Peak Normalize")
print("=" * 60)
t0 = time.time()
audio, sr = sf.read(WAV_16K, dtype='float64')
print(f"  Loaded: {len(audio)} samples, {len(audio)/sr:.1f}s")

audio_denoised = nr.reduce_noise(
    y=audio, sr=sr,
    stationary=True,
    prop_decrease=0.90,
    n_fft=2048,
    freq_mask_smooth_hz=500
)

peak = np.max(np.abs(audio_denoised))
audio_norm = audio_denoised * (0.95 / peak)
sf.write(DENOISED, audio_norm, sr, subtype='PCM_16')

dn_info = sf.info(DENOISED)
print(f"  Output: {DENOISED}")
print(f"  Duration: {dn_info.duration:.1f}s")
print(f"  Peak before norm: {peak:.4f} → 0.9500")
print(f"  Done in {time.time()-t0:.1f}s")

# Validation
assert abs(dn_info.duration - info.duration) < 1.0, "Duration mismatch after denoise"
print("✅ STEP 1 PASSED\n")

# ================================================================
# Step 2: Whisper V3 Transcription
# ================================================================
print("=" * 60)
print("STEP 2: Whisper Transcription (V3 optimal params)")
print("=" * 60)
print("  Loading model (large-v3, CUDA, float16)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("  Model loaded.\n")

print("  Transcribing with V3 params:")
print("    language=zh, beam_size=5")
print("    vad_filter=False")
print("    condition_on_previous_text=False")
print("    initial_prompt=[术语列表]")
print("    temperature=default fallback")

t0 = time.time()
segments, _ = model.transcribe(
    DENOISED,
    language="zh",
    beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
    initial_prompt=PROMPT,
)

seg_count = 0
char_count = 0
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for s in segments:
        seg_count += 1
        char_count += len(s.text)
        f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
        if seg_count % 500 == 0:
            print(f"    ... {seg_count} segs, {char_count} chars")

elapsed = time.time() - t0
print(f"\n  Result: {seg_count} segs, {char_count} chars, {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  Output: {OUTPUT_TXT}")

# Validation
file_size = os.path.getsize(OUTPUT_TXT)
print(f"  File size: {file_size} bytes")
assert seg_count > 1000, f"Too few segments ({seg_count})"
assert char_count > 15000, f"Too few chars ({char_count})"
print("✅ STEP 2 PASSED\n")

# ================================================================
# Step 3: Compare with baseline V3
# ================================================================
print("=" * 60)
print("STEP 3: Compare with baseline V3 from test_new")
print("=" * 60)

ts_pattern = re.compile(r'\[[\d.]+s\s*->\s*[\d.]+s\]\s*(.*)')

def extract_text(filepath):
    text = ""
    lines = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = ts_pattern.match(line.strip())
            if m:
                text += m.group(1)
                lines += 1
    return text, lines

new_text, new_lines = extract_text(OUTPUT_TXT)
old_text, old_lines = extract_text(V3_BASELINE)

print(f"\n  Baseline (test_new V3): {old_lines} segs, {len(old_text)} chars")
print(f"  New      (test_final): {new_lines} segs, {len(new_text)} chars")

seg_diff = abs(new_lines - old_lines)
char_diff = abs(len(new_text) - len(old_text))
seg_pct = seg_diff / old_lines * 100
char_pct = char_diff / len(old_text) * 100

print(f"\n  Segment diff: {seg_diff} ({seg_pct:.1f}%)")
print(f"  Char diff:    {char_diff} ({char_pct:.1f}%)")

# Check first 200 chars overlap
overlap_len = min(200, len(new_text), len(old_text))
match_chars = sum(1 for a, b in zip(new_text[:overlap_len], old_text[:overlap_len]) if a == b)
match_pct = match_chars / overlap_len * 100
print(f"  First {overlap_len} chars match: {match_pct:.1f}%")

# Final verdict
if seg_pct < 10 and char_pct < 15:
    print("\n✅ PIPELINE STABILITY CONFIRMED: Output is consistent with baseline.")
    print("   The SOP is reproducible and stable.")
else:
    print(f"\n⚠️ SIGNIFICANT DEVIATION DETECTED (seg {seg_pct:.1f}%, char {char_pct:.1f}%)")
    print("   This may be due to non-deterministic decoding (temperature fallback).")
    print("   Check if content quality is still acceptable.")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE!")
print("=" * 60)
