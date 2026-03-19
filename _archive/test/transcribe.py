"""
faster-whisper 中文课程录音转录脚本 (v3 - 标准模式)
使用 WhisperModel.transcribe 以获得准确的细粒度分段
输出: test/transcript.txt + test/transcript.srt
"""
import os
import sys
import time
from faster_whisper import WhisperModel

# ===== 配置 =====
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "test_audio.aac")
OUTPUT_DIR = os.path.dirname(__file__)
MODEL_SIZE = "large-v3"

# ===== 加载模型 =====
print(f"正在加载模型 {MODEL_SIZE} 到 GPU (fp16)...")
t0 = time.time()
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"模型加载完成，耗时 {time.time()-t0:.1f}s\n")

# ===== 开始转录 =====
print(f"开始转录: {AUDIO_PATH}")
print(f"文件大小: {os.path.getsize(AUDIO_PATH) / 1024 / 1024:.1f} MB\n")

t0 = time.time()
segments, info = model.transcribe(
    AUDIO_PATH,
    language="zh",
    beam_size=5,
    word_timestamps=True,
    vad_filter=False,
    condition_on_previous_text=True,
)

print(f"语言: {info.language} (概率: {info.language_probability:.2f})")
print(f"音频时长: {info.duration:.1f}s ({info.duration/60:.1f}min)")
print(f"VAD 后有效时长: {info.duration_after_vad:.1f}s ({info.duration_after_vad/60:.1f}min)\n")

# ===== 收集结果并写入文件 =====
txt_path = os.path.join(OUTPUT_DIR, "transcript.txt")
srt_path = os.path.join(OUTPUT_DIR, "transcript.srt")

def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

seg_count = 0
with open(txt_path, "w", encoding="utf-8") as txt_f, \
     open(srt_path, "w", encoding="utf-8") as srt_f:

    for segment in segments:
        seg_count += 1

        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        txt_f.write(line + "\n")

        if seg_count % 100 == 0 or seg_count <= 5:
            print(f"  #{seg_count} [{segment.start:.1f}s->{segment.end:.1f}s] {segment.text[:40]}")

        srt_f.write(f"{seg_count}\n")
        srt_f.write(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n")
        srt_f.write(f"{segment.text.strip()}\n\n")

elapsed = time.time() - t0
print(f"\n===== 转录完成 =====")
print(f"共 {seg_count} 个片段")
print(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"速度倍率: {info.duration / elapsed:.1f}x 实时")
print(f"TXT 输出: {txt_path}")
print(f"SRT 输出: {srt_path}")
