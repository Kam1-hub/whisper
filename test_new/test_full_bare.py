"""Test bare Whisper on FULL 114-min denoised file vs the ORIGINAL non-denoised file."""
import time
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

PROMPT = "以下是一节关于空气动力学的大学课程录音。"

tests = [
    ("FULL_denoised_bare", "c:/_/Whisper/test_new/1_16k_mono_nr090.wav"),
    ("FULL_original_bare", "c:/_/Whisper/test_new/1_16k_mono.wav"),
]

for name, path in tests:
    print(f"\n=== {name} ===")
    t0 = time.time()
    segs, _ = model.transcribe(path, language="zh", vad_filter=True,
        condition_on_previous_text=False, initial_prompt=PROMPT)
    count = 0
    chars = 0
    for s in segs:
        count += 1
        chars += len(s.text)
        if count <= 3:
            print(f"  [{s.start:.1f}s -> {s.end:.1f}s] {s.text[:60]}")
        if count % 500 == 0:
            print(f"  ... {count} segments so far")
    elapsed = time.time() - t0
    print(f"  TOTAL: {count} segs, {chars} chars, {elapsed:.1f}s")

print("\nDONE!")
