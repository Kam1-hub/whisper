"""
音频预处理流水线 — 分步处理，每步输出独立文件供试听
输入: test_audio.aac
输出:
  step1_16k_mono.wav   — 解码为 16kHz 单声道
  step2_denoised.wav   — AI 降噪后
  step3_normalized.wav — 音量归一化后（最终版本）
"""
import os
import sys
import time
import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "test_audio.aac")

# ============================================================
# 步骤 1: 解码为 16kHz 单声道 WAV
# ============================================================
print("=" * 50)
print("步骤 1: 解码为 16kHz 单声道 WAV")
print("=" * 50)

step1_output = os.path.join(SCRIPT_DIR, "step1_16k_mono.wav")
t0 = time.time()

# 使用 PyAV (faster-whisper 自带) 解码音频
import av

container = av.open(INPUT_FILE)
audio_stream = container.streams.audio[0]

resampler = av.AudioResampler(
    format="s16",
    layout="mono",
    rate=16000,
)

samples = []
for frame in container.decode(audio_stream):
    resampled = resampler.resample(frame)
    for r in resampled:
        arr = r.to_ndarray().flatten()
        samples.append(arr)

container.close()

audio_data = np.concatenate(samples).astype(np.float32) / 32768.0  # int16 -> float32

sf.write(step1_output, audio_data, 16000)
duration = len(audio_data) / 16000

print(f"  原始文件: {os.path.basename(INPUT_FILE)} ({os.path.getsize(INPUT_FILE)/1024/1024:.1f} MB)")
print(f"  输出文件: {os.path.basename(step1_output)} ({os.path.getsize(step1_output)/1024/1024:.1f} MB)")
print(f"  音频时长: {duration:.1f}s ({duration/60:.1f}min)")
print(f"  采样率: 16000 Hz, 单声道")
print(f"  耗时: {time.time()-t0:.1f}s")

# ============================================================
# 步骤 2: 降噪 (noisereduce)
# ============================================================
print("\n" + "=" * 50)
print("步骤 2: AI 降噪 (noisereduce)")
print("=" * 50)

step2_output = os.path.join(SCRIPT_DIR, "step2_denoised.wav")
t0 = time.time()

import noisereduce as nr

# 使用 stationary 模式降噪 (对空调/风扇等稳态噪音效果最好)
# prop_decrease: 降噪强度 0~1, 0.8 是适中值
print("  正在降噪，请稍候（可能需要 1-3 分钟）...")
audio_denoised = nr.reduce_noise(
    y=audio_data,
    sr=16000,
    stationary=True,
    prop_decrease=1.0,  # 提高到最大降噪强度
    n_fft=2048,
    freq_mask_smooth_hz=500,
)

sf.write(step2_output, audio_denoised, 16000)

print(f"  输出文件: {os.path.basename(step2_output)} ({os.path.getsize(step2_output)/1024/1024:.1f} MB)")
print(f"  降噪模式: stationary (稳态噪音)")
print(f"  降噪强度: 1.0 (最大强度)")
print(f"  耗时: {time.time()-t0:.1f}s")

# ============================================================
# 步骤 3: 音量归一化 (LUFS)
# ============================================================
print("\n" + "=" * 50)
print("步骤 3: 音量归一化")
print("=" * 50)

step3_output = os.path.join(SCRIPT_DIR, "step3_normalized.wav")
t0 = time.time()

# 计算当前 RMS 并归一化到目标 RMS
# 目标: -16 dBFS (比较洪亮的语音标准)
target_dbfs = -16.0

rms = np.sqrt(np.mean(audio_denoised ** 2))
current_dbfs = 20 * np.log10(rms + 1e-10)
gain_db = target_dbfs - current_dbfs
gain = 10 ** (gain_db / 20)

audio_normalized = audio_denoised * gain

# 限幅防爆
audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

sf.write(step3_output, audio_normalized, 16000)

print(f"  输出文件: {os.path.basename(step3_output)} ({os.path.getsize(step3_output)/1024/1024:.1f} MB)")
print(f"  原始响度: {current_dbfs:.1f} dBFS")
print(f"  目标响度: {target_dbfs:.1f} dBFS")
print(f"  增益: {gain_db:+.1f} dB")
print(f"  耗时: {time.time()-t0:.1f}s")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 50)
print("全部完成! 请试听以下文件对比效果:")
print("=" * 50)
print(f"  原始 (AAC):    {INPUT_FILE}")
print(f"  步骤1 (16kHz): {step1_output}")
print(f"  步骤2 (降噪):  {step2_output}")
print(f"  步骤3 (归一化): {step3_output}")
