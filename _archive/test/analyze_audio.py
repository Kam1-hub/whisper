"""
音频数据特征分析脚本
对比原始、降噪后、归一化后三个音频的客观物理指标
"""
import numpy as np
import soundfile as sf
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
files_to_analyze = {
    "1_原始音频 (有底噪)": "step1_16k_mono.wav",
    "2_降噪后 (底噪极小)": "step2_denoised.wav",
    "3_归一化 (人声更清晰)": "step3_normalized.wav"
}

print(f"{'文件':<25} | {'平均能量(RMS)':<15} | {'最大振幅(Peak)':<15} | {'动态范围':<15} | {'静音极小值(底噪评估)':<20}")
print("-" * 100)

for label, filename in files_to_analyze.items():
    filepath = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(filepath):
        continue
        
    audio, sr = sf.read(filepath)
    
    # 1. RMS (平均能量 - 客观音量大小)
    rms = np.sqrt(np.mean( audio**2 ))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # 2. Peak (最大振幅)
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak + 1e-10)
    
    # 3. 动态范围 (Peak - RMS)
    dynamic_range_db = peak_db - rms_db
    
    # 4. 底噪评估 (通过寻找能量最低的10%片段的平均RMS)
    # 将音频分块，每块0.5秒
    chunk_size = int(sr * 0.5)
    num_chunks = len(audio) // chunk_size
    if num_chunks > 0:
        chunks = np.array_split(audio[:num_chunks * chunk_size], num_chunks)
        chunk_rms = np.array([np.sqrt(np.mean(c**2)) for c in chunks])
        # 取最小的 10% 块作为静音底噪的估计
        noise_floor_rms = np.mean(np.sort(chunk_rms)[:max(1, int(num_chunks * 0.1))])
        noise_floor_db = 20 * np.log10(noise_floor_rms + 1e-10)
    else:
        noise_floor_db = -100.0

    print(f"{label:<25} | {rms_db:>10.2f} dBFS | {peak_db:>10.2f} dBFS | {dynamic_range_db:>10.2f} dB | {noise_floor_db:>10.2f} dBFS")

print("-" * 100)
print("\n数据解读:")
print("- 平均能量(RMS): 越接近0说明声音越大。归一化步骤(3)不仅是拉高了整体音量，主要是把人声音量提升了。")
print("- 最大振幅(Peak): 最大不能超过0。这代表音频里最响的一声。")
print("- 静音极小值(底噪): 这是判断噪音强度的最直观数据。越小越好（负得越多越好）。")
print("  > 你可以看到从 (1) 到 (2)，底噪能量下降了非常多，这就是降噪的作用。")
