"""
对比三次转录结果的关键指标
"""
import json

files = {
    "transcript.txt": "1_Raw_NoVAD",
    "transcript_enhanced.txt": "2_Enhanced_VAD",
    "transcript_enhanced_novad.txt": "3_Enhanced_NoVAD",
}

results = []
for fname, label in files.items():
    path = f"c:\\_\\Whisper\\test\\{fname}"
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    total_chars = sum(len(l.split("] ", 1)[1]) if "] " in l else 0 for l in lines)

    # Parse timestamps to get coverage
    import re
    timestamps = []
    for l in lines:
        m = re.match(r'\[(\d+\.\d+)s -> (\d+\.\d+)s\]', l)
        if m:
            timestamps.append((float(m.group(1)), float(m.group(2))))

    if timestamps:
        first_ts = timestamps[0][0]
        last_ts = timestamps[-1][1]
        total_speech_duration = sum(e - s for s, e in timestamps)
    else:
        first_ts = last_ts = total_speech_duration = 0

    results.append({
        "label": label,
        "segments": len(lines),
        "total_chars": total_chars,
        "first_ts": round(first_ts, 1),
        "last_ts": round(last_ts, 1),
        "speech_duration_s": round(total_speech_duration, 1),
    })

with open("c:\\_\\Whisper\\test\\comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("OK")
