"""
严格控制变量测试：使用与 test_new/test_full_novad.py 完全一致的参数
（含完全相同的短 initial_prompt）在 test_final 的降噪音频上重跑转录。
"""
import time, os, re
from faster_whisper import WhisperModel

AUDIO    = "c:/_/Whisper/test_final/1_16k_mono_nr090.wav"
OUT_TXT  = "c:/_/Whisper/test_final/transcript_v3_strict.txt"
BASELINE = "c:/_/Whisper/test_new/1_16k_mono_transcript_v3_novad.txt"

# 与 test_full_novad.py 完全一致的短 prompt
PROMPT = "以下是一节关于空气动力学的大学课程录音。"

print("Loading model (large-v3, cuda, float16)...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

print("Transcribing with IDENTICAL params to baseline V3...")
print(f"  PROMPT = \"{PROMPT}\"")
print(f"  language=zh, beam_size=5, vad_filter=False, condition=False")

t0 = time.time()
segs, _ = model.transcribe(
    AUDIO,
    language="zh",
    beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
    initial_prompt=PROMPT,
)

count = 0
chars = 0
with open(OUT_TXT, "w", encoding="utf-8") as f:
    for s in segs:
        count += 1
        chars += len(s.text)
        f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
        if count % 500 == 0:
            print(f"  ... {count} segs, {chars} chars")

elapsed = time.time() - t0
print(f"\nDONE! {count} segs, {chars} chars, {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"File: {OUT_TXT}, Size: {os.path.getsize(OUT_TXT)} bytes")

# === Compare with baseline ===
print("\n" + "=" * 60)
print("COMPARISON: strict rerun vs baseline V3")
print("=" * 60)

ts_pat = re.compile(r'\[[\d.]+s\s*->\s*[\d.]+s\]\s*(.*)')

def stats(path):
    text = ""
    lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = ts_pat.match(line.strip())
            if m:
                text += m.group(1)
                lines += 1
    return lines, len(text), os.path.getsize(path)

bl_segs, bl_chars, bl_size = stats(BASELINE)
nw_segs, nw_chars, nw_size = stats(OUT_TXT)

print(f"  Baseline:     {bl_segs} segs, {bl_chars} chars, {bl_size} bytes")
print(f"  Strict rerun: {nw_segs} segs, {nw_chars} chars, {nw_size} bytes")

seg_d = abs(nw_segs - bl_segs) / bl_segs * 100
chr_d = abs(nw_chars - bl_chars) / bl_chars * 100
siz_d = abs(nw_size - bl_size) / bl_size * 100

print(f"\n  Segment diff: {abs(nw_segs-bl_segs)} ({seg_d:.2f}%)")
print(f"  Char diff:    {abs(nw_chars-bl_chars)} ({chr_d:.2f}%)")
print(f"  Size diff:    {abs(nw_size-bl_size)} ({siz_d:.2f}%)")

# First 500 chars text overlap
overlap = min(500, len(nw_chars.__class__.__name__) or 500)
# Actually let's re-read texts
with open(BASELINE, "r", encoding="utf-8") as f:
    bl_lines = f.readlines()
with open(OUT_TXT, "r", encoding="utf-8") as f:
    nw_lines = f.readlines()

# Compare first 20 lines
print(f"\n  First 20 lines side-by-side check:")
match = 0
for i in range(min(20, len(bl_lines), len(nw_lines))):
    identical = bl_lines[i].strip() == nw_lines[i].strip()
    if identical:
        match += 1
    marker = "==" if identical else "!="
    print(f"    L{i+1}: {marker}")
print(f"  Match rate (first 20 lines): {match}/20 = {match/20*100:.0f}%")

if seg_d < 5 and chr_d < 5:
    print("\n✅ STRICT CONTROL TEST PASSED: Pipeline is stable and reproducible.")
else:
    print(f"\n⚠️ Deviation detected: seg {seg_d:.1f}%, char {chr_d:.1f}%")
