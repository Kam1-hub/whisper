"""
faster-whisper 降噪+归一化音频转录脚本
输入: step3_normalized.wav (预处理后)
输出: transcript_enhanced.txt + transcript_enhanced.srt
"""
import os
import sys
import time
from faster_whisper import WhisperModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(SCRIPT_DIR, "step3_normalized.wav")
MODEL_SIZE = "large-v3"

print(f"Loading model {MODEL_SIZE} (GPU fp16)...")
t0 = time.time()
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"Model loaded in {time.time()-t0:.1f}s\n")

print(f"Input: {AUDIO_PATH}")
print(f"Size: {os.path.getsize(AUDIO_PATH)/1024/1024:.1f} MB\n")

t0 = time.time()
segments, info = model.transcribe(
    AUDIO_PATH,
    language="zh",
    beam_size=5,
    word_timestamps=True,
    vad_filter=True,
    condition_on_previous_text=False,
    temperature=0,
    hallucination_silence_threshold=1.0,
)

print(f"Language: {info.language} (prob: {info.language_probability:.2f})")
print(f"Duration: {info.duration:.1f}s ({info.duration/60:.1f}min)")
print(f"After VAD: {info.duration_after_vad:.1f}s ({info.duration_after_vad/60:.1f}min)\n")

def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

txt_path = os.path.join(SCRIPT_DIR, "transcript_enhanced.txt")
srt_path = os.path.join(SCRIPT_DIR, "transcript_enhanced.srt")

seg_count = 0
with open(txt_path, "w", encoding="utf-8") as txt_f, \
     open(srt_path, "w", encoding="utf-8") as srt_f:
    for segment in segments:
        seg_count += 1
        txt_f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
        srt_f.write(f"{seg_count}\n{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n{segment.text.strip()}\n\n")
        if seg_count % 100 == 0 or seg_count <= 3:
            print(f"  #{seg_count} [{segment.start:.1f}s] {segment.text[:40]}")

elapsed = time.time() - t0
print(f"\nDone: {seg_count} segments in {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Speed: {info.duration/elapsed:.1f}x realtime")
print(f"TXT: {txt_path}")
print(f"SRT: {srt_path}")
