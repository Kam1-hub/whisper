"""Bare minimum test: NO protection at all, just plain Whisper on 30-min denoised audio."""
import time
import soundfile as sf
from faster_whisper import WhisperModel

CLIP = "c:/_/Whisper/test_new/clip_30m_nr090.wav"

print("Loading model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Test: absolutely ZERO extras, just language + VAD
print("\n=== BARE MINIMUM: VAD + language only ===")
t0 = time.time()
segs, _ = model.transcribe(CLIP, language="zh", vad_filter=True)
count = 0
chars = 0
for s in segs:
    count += 1
    chars += len(s.text)
    if count <= 5:
        print(f"  [{s.start:.1f}s -> {s.end:.1f}s] {s.text[:80]}")
print(f"  Total: {count} segs, {chars} chars, {time.time()-t0:.1f}s")

# Test: no VAD at all
print("\n=== NO VAD, language only ===")
t0 = time.time()
segs2, _ = model.transcribe(CLIP, language="zh", vad_filter=False)
count2 = 0
chars2 = 0
for s in segs2:
    count2 += 1
    chars2 += len(s.text)
    if count2 <= 5:
        print(f"  [{s.start:.1f}s -> {s.end:.1f}s] {s.text[:80]}")
print(f"  Total: {count2} segs, {chars2} chars, {time.time()-t0:.1f}s")

print("\nDONE!")
