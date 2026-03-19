"""
方案 B 执行脚本：Step 2 → Step 3 → Step 4
  Step 2: 逐段转录 + 逐段验证
  Step 3: 时间轴偏移合并 + 完整性验证
  Step 4: 与 V3 基线量化比对
"""
import os, re, time, json
from faster_whisper import WhisperModel

CHUNK_DIR   = "c:/_/Whisper/test_new/chunks"
CHUNK_TXT   = "c:/_/Whisper/test_new/chunks_v4b_txt"
MERGED_FILE = "c:/_/Whisper/test_new/1_16k_mono_transcript_v4b_chunked.txt"
V3_FILE     = "c:/_/Whisper/test_new/1_16k_mono_transcript_v3_novad.txt"
REPORT_FILE = "c:/_/Whisper/test_new/v4b_vs_v3_report.json"
CHUNK_SEC   = 600  # each chunk = 10 minutes = 600 seconds

PROMPT = (
    "以下是一节关于空气动力学的大学课程录音。包含专业术语如："
    "法向、切向、流线、迹线、脉线、马赫数、雷诺数、层流、湍流、"
    "边界层、压强梯度、伯努利方程、欧拉方程、连续性方程、速度势、"
    "势流、涡量、环量、升力、阻力、攻角、翼型。"
)

TERMS = ["法向", "切向", "流线", "迹线", "马赫数", "雷诺数", "层流", "湍流",
         "边界层", "伯努利", "欧拉", "连续方程", "连续性方程", "速度势",
         "涡量", "环量", "升力", "阻力", "攻角", "翼型", "散度", "旋度"]

os.makedirs(CHUNK_TXT, exist_ok=True)

# ======================================================================
# Step 2: Chunk-Level Transcription
# ======================================================================
print("=" * 60)
print("STEP 2: Transcribing chunks with safe parameters")
print("=" * 60)

print("Loading model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".wav")])
chunk_stats = []
all_ok = True
t_total = time.time()

for i, filename in enumerate(chunks):
    filepath = os.path.join(CHUNK_DIR, filename)
    outpath  = os.path.join(CHUNK_TXT, filename.replace(".wav", ".txt"))

    print(f"\n[{i+1}/{len(chunks)}] {filename}")
    t0 = time.time()

    segments, _ = model.transcribe(
        filepath,
        language="zh",
        beam_size=5,
        vad_filter=False,                    # VAD proven broken on this audio
        condition_on_previous_text=True,     # KEY BENEFIT of chunking
        initial_prompt=PROMPT,
        # NO repetition_penalty
        # NO no_repeat_ngram_size
        # NO hallucination_silence_threshold
        # temperature: default fallback
    )

    seg_count = 0
    char_count = 0
    max_dur = 0
    lines = []
    for s in segments:
        seg_count += 1
        char_count += len(s.text)
        dur = s.end - s.start
        if dur > max_dur:
            max_dur = dur
        lines.append(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")

    with open(outpath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    elapsed = time.time() - t0
    stat = {"chunk": filename, "segs": seg_count, "chars": char_count,
            "max_dur": round(max_dur, 1), "time": round(elapsed, 1)}
    chunk_stats.append(stat)

    # --- Per-chunk validation ---
    warnings = []
    if seg_count < 5 or char_count < 100:
        warnings.append("COLLAPSE: too few segments/chars")
        all_ok = False
    if max_dur > 60:
        warnings.append(f"LONG_SEG: max segment duration {max_dur:.0f}s > 60s")

    status = "✅" if not warnings else "⚠️ " + "; ".join(warnings)
    print(f"  {seg_count} segs | {char_count} chars | max_dur={max_dur:.0f}s | {elapsed:.1f}s | {status}")

total_segs  = sum(s["segs"]  for s in chunk_stats)
total_chars = sum(s["chars"] for s in chunk_stats)
print(f"\n--- Step 2 Summary ---")
print(f"Total: {total_segs} segs, {total_chars} chars, {(time.time()-t_total)/60:.1f} min")

if total_segs < 500 or total_chars < 10000:
    print("❌ STEP 2 VALIDATION FAILED: total output too thin.")
    all_ok = False
else:
    print("✅ STEP 2 VALIDATION PASSED")

# ======================================================================
# Step 3: Time-Offset Merge
# ======================================================================
print("\n" + "=" * 60)
print("STEP 3: Merging with time offsets")
print("=" * 60)

merged_lines = []
ts_pattern = re.compile(r'\[(\d+\.\d+)s\s*->\s*(\d+\.\d+)s\](.*)') 

for i, filename in enumerate(sorted(os.listdir(CHUNK_TXT))):
    if not filename.endswith(".txt"):
        continue
    offset = i * CHUNK_SEC
    filepath = os.path.join(CHUNK_TXT, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = ts_pattern.match(line.strip())
            if m:
                start = float(m.group(1)) + offset
                end   = float(m.group(2)) + offset
                text  = m.group(3)
                merged_lines.append((start, end, text))

with open(MERGED_FILE, "w", encoding="utf-8") as f:
    for start, end, text in merged_lines:
        f.write(f"[{start:.2f}s -> {end:.2f}s]{text}\n")

# --- Merge validation ---
merge_ok = True

# Check monotonic timestamps
for j in range(1, len(merged_lines)):
    if merged_lines[j][0] < merged_lines[j-1][0]:
        print(f"❌ Timestamp order violation at line {j+1}")
        merge_ok = False
        break

# Check final timestamp
if merged_lines:
    final_ts = merged_lines[-1][1]
    if abs(final_ts - 6840) > 60:
        print(f"⚠️ Final timestamp {final_ts:.1f}s deviates from expected ~6840s")
    else:
        print(f"✅ Final timestamp {final_ts:.1f}s is within expected range")

merged_segs  = len(merged_lines)
merged_chars = sum(len(t) for _, _, t in merged_lines)
print(f"Merged: {merged_segs} segs, {merged_chars} chars")

if merged_segs == total_segs and merge_ok:
    print("✅ STEP 3 VALIDATION PASSED: merge is lossless and ordered")
else:
    print("❌ STEP 3 VALIDATION FAILED")

# ======================================================================
# Step 4: Compare with V3 baseline
# ======================================================================
print("\n" + "=" * 60)
print("STEP 4: Quality comparison V4 (chunked) vs V3 (no-VAD)")
print("=" * 60)

# Load V3
with open(V3_FILE, "r", encoding="utf-8") as f:
    v3_lines = f.readlines()
v3_text = ""
for line in v3_lines:
    m = ts_pattern.match(line.strip())
    if m:
        v3_text += m.group(3)

# Load V4
with open(MERGED_FILE, "r", encoding="utf-8") as f:
    v4_lines = f.readlines()
v4_text = ""
for line in v4_lines:
    m = ts_pattern.match(line.strip())
    if m:
        v4_text += m.group(3)

# 4a: Pure text char count
print(f"\n[4a] Character Count")
print(f"  V3: {len(v3_text)} chars")
print(f"  V4: {len(v4_text)} chars")

# 4b: Segment count
print(f"\n[4b] Segment Count")
print(f"  V3: {len(v3_lines)} segs")
print(f"  V4: {len(v4_lines)} segs")

# 4c: Term hit count
print(f"\n[4c] Terminology Hit Count")
v3_hits = {}
v4_hits = {}
for term in TERMS:
    v3_hits[term] = v3_text.count(term)
    v4_hits[term] = v4_text.count(term)

v3_total_hits = sum(v3_hits.values())
v4_total_hits = sum(v4_hits.values())
print(f"  V3 total term hits: {v3_total_hits}")
print(f"  V4 total term hits: {v4_total_hits}")
print(f"  Top differences:")
for term in TERMS:
    diff = v4_hits[term] - v3_hits[term]
    if diff != 0:
        marker = "↑" if diff > 0 else "↓"
        print(f"    {term}: V3={v3_hits[term]} V4={v4_hits[term]} ({marker}{abs(diff)})")

# 4d: Tail hallucination count
print(f"\n[4d] Tail Hallucination Detection")
v3_hao = sum(1 for l in v3_lines if l.strip().endswith("好"))
v4_hao = sum(1 for l in v4_lines if l.strip().endswith("好"))
v3_sub = sum(1 for l in v3_lines if "订阅" in l or "点赞" in l or "转发" in l)
v4_sub = sum(1 for l in v4_lines if "订阅" in l or "点赞" in l or "转发" in l)
print(f"  Lines ending with just '好': V3={v3_hao}, V4={v4_hao}")
print(f"  Lines with YouTube hallucination: V3={v3_sub}, V4={v4_sub}")

# Save report
report = {
    "v3": {"segs": len(v3_lines), "chars": len(v3_text), "term_hits": v3_total_hits,
           "hao_lines": v3_hao, "youtube_lines": v3_sub},
    "v4": {"segs": len(v4_lines), "chars": len(v4_text), "term_hits": v4_total_hits,
           "hao_lines": v4_hao, "youtube_lines": v4_sub},
    "chunk_stats": chunk_stats,
}
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n--- Final Verdict ---")
improvements = []
if v4_total_hits > v3_total_hits:
    improvements.append(f"术语命中 +{v4_total_hits - v3_total_hits}")
if v4_hao < v3_hao:
    improvements.append(f"'好'杂音行 -{v3_hao - v4_hao}")
if v4_sub < v3_sub:
    improvements.append(f"YouTube幻觉 -{v3_sub - v4_sub}")

if improvements:
    print("✅ V4 IMPROVEMENTS: " + ", ".join(improvements))
else:
    print("⚠️ V4 did not show clear improvements over V3")

print(f"\nReport saved to {REPORT_FILE}")
print("ALL STEPS COMPLETE!")
