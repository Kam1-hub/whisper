import os
import time
import json
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

AUDIO_FILE = "c:/_/Whisper/test_new/1_16k_mono_nr090.wav"
OUT_DIR = "c:/_/Whisper/test_new/fulltest_isolate"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = (
    "以下是一节关于空气动力学的大学课程录音。包含专业术语如："
    "法向、切向、流线、迹线、脉线、马赫数、雷诺数、层流、湍流、"
    "边界层、压强梯度、伯努利方程、欧拉方程、连续性方程、速度势、"
    "势流、涡量、环量、升力、阻力、攻角、翼型。"
)

# Load and clip to first 30 minutes
print("Loading denoised audio...")
audio, sr = sf.read(AUDIO_FILE, dtype='float32')
clip_samples = 30 * 60 * sr  # 30 minutes
audio_30m = audio[:clip_samples]
print(f"  Clipped to {len(audio_30m)/sr/60:.1f} min")

# 2x2 factorial: temperature × hallucination_silence_threshold
TESTS = [
    ("F1_temp_default_thresh_none", [0, 0.2, 0.4, 0.6, 0.8, 1.0], None),
    ("F2_temp_zero_thresh_none",    0,                              None),
    ("F3_temp_default_thresh_2",    [0, 0.2, 0.4, 0.6, 0.8, 1.0], 2.0),
    ("F4_temp_zero_thresh_2",       0,                              2.0),
]

print("Loading Whisper model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Model loaded.\n")

results = []
for name, temp, thresh in TESTS:
    print(f"=== {name} ===")
    print(f"  temp={temp}, threshold={thresh}")
    
    t0 = time.time()
    segments, info = model.transcribe(
        audio_30m,
        language="zh",
        beam_size=5,
        temperature=temp,
        vad_filter=True,
        condition_on_previous_text=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        hallucination_silence_threshold=thresh,
        initial_prompt=PROMPT,
    )
    
    txt_path = os.path.join(OUT_DIR, f"{name}.txt")
    seg_count = 0
    char_count = 0
    max_duration = 0
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in segments:
            seg_count += 1
            char_count += len(s.text)
            dur = s.end - s.start
            if dur > max_duration:
                max_duration = dur
            f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
    
    elapsed = time.time() - t0
    res = {
        "name": name,
        "time_s": round(elapsed, 1),
        "segments": seg_count,
        "chars": char_count,
        "max_seg_duration_s": round(max_duration, 1),
    }
    results.append(res)
    print(f"  Done in {elapsed:.1f}s | segs={seg_count} | chars={char_count} | max_dur={max_duration:.0f}s\n")

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("ALL TESTS COMPLETE!")
