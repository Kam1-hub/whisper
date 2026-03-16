"""
用 clip.wav (5分钟短片段) 快速对比 condition_on_previous_text True vs False
"""
import os, time, json
from faster_whisper import WhisperModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP = os.path.join(SCRIPT_DIR, "clip.wav")

print("Loading model...")
t0 = time.time()
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"Model loaded in {time.time()-t0:.1f}s\n")

common_params = dict(
    language="zh",
    beam_size=5,
    word_timestamps=True,
    vad_filter=False,
    temperature=0,
    hallucination_silence_threshold=1.0,
)

results = {}

for cond in [False, True]:
    label = "cond_ON" if cond else "cond_OFF"
    print(f"=== condition_on_previous_text={cond} ===")

    t0 = time.time()
    segments, info = model.transcribe(CLIP, condition_on_previous_text=cond, **common_params)

    lines = []
    for seg in segments:
        lines.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")

    elapsed = time.time() - t0

    # Save to file
    txt_path = os.path.join(SCRIPT_DIR, f"clip_{label}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    total_chars = sum(len(l.split("] ", 1)[1]) if "] " in l else 0 for l in lines)

    results[label] = {
        "segments": len(lines),
        "chars": total_chars,
        "time_s": round(elapsed, 1),
        "file": txt_path,
    }

    print(f"  Segments: {len(lines)}, Chars: {total_chars}, Time: {elapsed:.1f}s")
    print(f"  Saved: {txt_path}\n")

# Save comparison
with open(os.path.join(SCRIPT_DIR, "clip_comparison.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Done!")
