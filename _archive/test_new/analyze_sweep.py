import numpy as np
import soundfile as sf
import os
import re

DIR = "c:/_/Whisper/test_new"
log_path = os.path.join(DIR, "log.md")

files = [
    ("clip_10m_20m.wav", "原始基准", "无"),
    ("clip_nr_0.85.wav", "noisereduce", "prop_decrease=0.85"),
    ("clip_nr_0.90.wav", "noisereduce", "prop_decrease=0.90"),
    ("clip_nr_0.95.wav", "noisereduce", "prop_decrease=0.95"),
    ("clip_nr_1.00.wav", "noisereduce", "prop_decrease=1.00"),
    ("clip_dfn_6.wav", "DeepFilterNet", "--atten-lim-db 6"),
    ("clip_dfn_9.wav", "DeepFilterNet", "--atten-lim-db 9"),
    ("clip_dfn_12.wav", "DeepFilterNet", "--atten-lim-db 12"),
]

print("Analyzing metrics...")
new_rows = []
for fname, method, param in files:
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        print(f"Skipping {fname} (not found)")
        continue
        
    a, sr = sf.read(path)
    
    # RMS
    rms = float(20 * np.log10(np.sqrt(np.mean(a**2)) + 1e-10))
    
    # Noise Floor (bottom 10% of 0.5s chunks)
    chunk_size = 8000
    n_chunks = len(a) // chunk_size
    chunks = np.array_split(a[:n_chunks*chunk_size], n_chunks)
    chunk_rms = np.array([np.sqrt(np.mean(c**2)) for c in chunks])
    nf = float(20 * np.log10(np.mean(np.sort(chunk_rms)[:max(1, n_chunks//10)]) + 1e-10))
    
    row = f"| `{fname}` | {method} | `{param}` | {nf:.2f} dB | {rms:.2f} dB | 待评 |"
    new_rows.append(row)

# Read log.md and append the new tests
with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

out_lines = []
in_table = False
for line in lines:
    out_lines.append(line)
    if line.startswith("| `2_denoised_deepfilternet.wav`"):
        # Append our new rows right after the old tracking section
        out_lines.append("### 10分钟切片参数测试 (10m~20m)\n\n")
        out_lines.append("| 文件名 | 降噪方法 | 所用参数 | 底噪(NoiseFloor) | 音量(RMS) | 听感评价 / 备注 |\n")
        out_lines.append("|---|---|---|---|---|---|\n")
        for r in new_rows:
            out_lines.append(r + "\n")

with open(log_path, "w", encoding="utf-8") as f:
    f.writelines(out_lines)

print("log.md updated!")
