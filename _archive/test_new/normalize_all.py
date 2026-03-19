import numpy as np
import soundfile as sf
import os

DIR = "c:/_/Whisper/test_new"
TARGET_PEAK = 0.95  # -0.45 dBFS headroom

files = [
    "clip_10m_20m.wav",
    "clip_nr_0.85.wav",
    "clip_nr_0.90.wav",
    "clip_nr_0.95.wav",
    "clip_nr_1.00.wav",
    "clip_dfn_6.wav",
    "clip_dfn_9.wav",
    "clip_dfn_12.wav",
]

for fname in files:
    path = os.path.join(DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP: {fname} not found")
        continue

    audio, sr = sf.read(path)
    
    # Current stats
    current_peak = np.max(np.abs(audio))
    current_rms = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
    
    # Peak normalize
    gain = TARGET_PEAK / (current_peak + 1e-10)
    audio_norm = audio * gain
    
    new_rms = 20 * np.log10(np.sqrt(np.mean(audio_norm**2)) + 1e-10)
    new_peak = np.max(np.abs(audio_norm))
    
    # Save with _norm suffix (overwrite previous failed normalization)
    base, ext = os.path.splitext(fname)
    out_path = os.path.join(DIR, f"{base}_norm{ext}")
    sf.write(out_path, audio_norm, sr)
    
    print(f"{fname}: RMS {current_rms:.2f} -> {new_rms:.2f} dB | Peak {current_peak:.4f} -> {new_peak:.4f}")

print("\nAll peak normalization done!")
