"""
音频快速分段工具
用法: python clip_audio.py [开始秒数] [时长秒数]
例: python clip_audio.py 600 300  → 从10分钟处截取5分钟
默认: 从 800s (13min) 截取 300s (5min) — 课程正文最密集的区域
"""
import os, sys, time
import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(SCRIPT_DIR, "step3_normalized.wav")

start_sec = float(sys.argv[1]) if len(sys.argv) > 1 else 800
duration_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300

print(f"Reading: {INPUT}")
audio, sr = sf.read(INPUT)

start_sample = int(start_sec * sr)
end_sample = int((start_sec + duration_sec) * sr)
end_sample = min(end_sample, len(audio))

clip = audio[start_sample:end_sample]
out_path = os.path.join(SCRIPT_DIR, "clip.wav")
sf.write(out_path, clip, sr)

print(f"Clip: {start_sec:.0f}s ~ {start_sec+duration_sec:.0f}s ({duration_sec:.0f}s = {duration_sec/60:.1f}min)")
print(f"Output: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")
