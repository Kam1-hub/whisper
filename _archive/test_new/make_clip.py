import os, sys
import soundfile as sf

start_min = float(sys.argv[1]) if len(sys.argv) > 1 else 10
end_min = float(sys.argv[2]) if len(sys.argv) > 2 else 20

start_sec = start_min * 60
end_sec = end_min * 60

INPUT = "c:/_/Whisper/test_new/1_16k_mono.wav"
print(f"Reading: {INPUT}")
audio, sr = sf.read(INPUT)

start_sample = int(start_sec * sr)
end_sample = int(end_sec * sr)
end_sample = min(end_sample, len(audio))

clip = audio[start_sample:end_sample]
out_path = "c:/_/Whisper/test_new/clip_10m_20m.wav"
sf.write(out_path, clip, sr)

print(f"Clip: {start_min:.1f}m ~ {end_min:.1f}m ({end_sec-start_sec:.1f}s)")
print(f"Output: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")
