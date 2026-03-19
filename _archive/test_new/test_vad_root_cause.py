"""Test VAD vs no-VAD, and also test on original (non-denoised) audio."""
import time
import soundfile as sf
from faster_whisper import WhisperModel

DENOISED = "c:/_/Whisper/test_new/clip_30m_nr090.wav"
ORIGINAL = "c:/_/Whisper/test_new/1_16k_mono.wav"

print("Loading model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

PROMPT = "以下是一节关于空气动力学的大学课程录音。"

tests = [
    ("T1_denoised_vad_on",  DENOISED, True,  True),
    ("T2_denoised_vad_off", DENOISED, False, True),
    ("T3_denoised_vad_on_nocond", DENOISED, True, False),
    ("T4_original_vad_on",  ORIGINAL, True,  True),
]

for name, path, vad, cond in tests:
    print(f"\n=== {name} ===")
    # For original, only read first 30 min
    if "original" in name:
        import numpy as np
        audio, sr = sf.read(path, dtype='float32')
        audio = audio[:30*60*sr]
        src = audio
    else:
        src = path
    
    t0 = time.time()
    segs, _ = model.transcribe(src, language="zh", beam_size=5,
        vad_filter=vad, condition_on_previous_text=cond,
        repetition_penalty=1.2, no_repeat_ngram_size=3, initial_prompt=PROMPT)
    count = 0
    chars = 0
    for s in segs:
        count += 1
        chars += len(s.text)
        if count <= 3:
            print(f"  [{s.start:.1f}s -> {s.end:.1f}s] {s.text[:60]}...")
    print(f"  Total: {count} segs, {chars} chars, {time.time()-t0:.1f}s")

print("\nDONE!")
