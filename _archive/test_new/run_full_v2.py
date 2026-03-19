import os
import time
import json
from faster_whisper import WhisperModel

AUDIO_FILE = "c:/_/Whisper/test_new/1_16k_mono_nr090.wav"
OUTPUT_TXT = "c:/_/Whisper/test_new/1_16k_mono_transcript_v2.txt"

PROMPT = (
    "以下是一节关于空气动力学的大学课程录音。包含专业术语如："
    "法向、切向、流线、迹线、脉线、马赫数、雷诺数、层流、湍流、"
    "边界层、压强梯度、伯努利方程、欧拉方程、连续性方程、速度势、"
    "势流、涡量、环量、升力、阻力、攻角、翼型。"
)

print("Loading Whisper model (large-v3, fp16)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Model loaded.\n")

print("Transcribing 114-minute audio with corrected parameters...")
t0 = time.time()
segments, info = model.transcribe(
    AUDIO_FILE,
    language="zh",
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False,
    initial_prompt=PROMPT,
    # temperature: default fallback [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # NO repetition_penalty
    # NO no_repeat_ngram_size
    # NO hallucination_silence_threshold
)

seg_count = 0
char_count = 0
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for s in segments:
        seg_count += 1
        char_count += len(s.text)
        f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
        if seg_count % 200 == 0:
            print(f"  ... {seg_count} segments, {char_count} chars")

elapsed = time.time() - t0
print(f"\nDONE!")
print(f"  Segments: {seg_count}")
print(f"  Characters: {char_count}")
print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  Output: {OUTPUT_TXT}")
