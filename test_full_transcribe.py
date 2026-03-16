"""
faster-whisper 最终验证脚本（无手动 PATH 设置，依赖系统环境变量）
"""
import sys, time, math, struct, wave
from faster_whisper import WhisperModel
import ctranslate2

print("=== 环境信息 ===")
print(f"Python: {sys.version.split()[0]}")
print(f"ctranslate2: {ctranslate2.__version__}")
print(f"CUDA 设备: {ctranslate2.get_cuda_device_count()}")

# 生成测试音频
with wave.open("test_audio.wav", 'w') as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
    for i in range(32000):
        wf.writeframesraw(struct.pack('<h', int(32767 * math.sin(440 * 2 * math.pi * i / 16000))))
print("\n=== 测试音频已生成 ===")

# GPU 推理测试
print("\n=== 加载模型 (GPU fp16) ===")
t = time.time()
model = WhisperModel("tiny", device="cuda", compute_type="float16")
print(f"加载耗时: {time.time()-t:.2f}s")

print("\n=== 转录测试 ===")
t = time.time()
segments, info = model.transcribe("test_audio.wav", beam_size=5)
print(f"语言: {info.language} (概率: {info.language_probability:.2f})")
for seg in segments:
    print(f"  [{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
print(f"转录耗时: {time.time()-t:.2f}s")

print("\n=== ALL TESTS PASSED ===")
