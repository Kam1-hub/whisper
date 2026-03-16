"""Final diagnostic: test full 114-min denoised file WITHOUT VAD."""
import time
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
PROMPT = "以下是一节关于空气动力学的大学课程录音。"
AUDIO = "c:/_/Whisper/test_new/1_16k_mono_nr090.wav"
OUT = "c:/_/Whisper/test_new/1_16k_mono_transcript_v3_novad.txt"

print("Transcribing full 114-min WITHOUT VAD...")
t0 = time.time()
segs, _ = model.transcribe(AUDIO, language="zh", beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
    initial_prompt=PROMPT)

count = 0
chars = 0
with open(OUT, "w", encoding="utf-8") as f:
    for s in segs:
        count += 1
        chars += len(s.text)
        f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
        if count % 500 == 0:
            print(f"  ... {count} segs, {chars} chars")

elapsed = time.time() - t0
print(f"\nDONE! {count} segs, {chars} chars, {elapsed:.1f}s ({elapsed/60:.1f} min)")
