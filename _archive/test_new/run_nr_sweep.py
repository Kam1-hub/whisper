import soundfile as sf
import noisereduce as nr
import time
import os

input_file = "c:/_/Whisper/test_new/clip_10m_20m.wav"
print(f"Reading {input_file}...")
audio, sr = sf.read(input_file)

props = [0.85, 0.90, 0.95, 1.0]

for p in props:
    print(f"\nRunning noisereduce with prop_decrease={p} ...")
    t0 = time.time()
    audio_denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=p,
        n_fft=2048,
        freq_mask_smooth_hz=500,
    )
    elapsed = time.time() - t0
    
    out_path = f"c:/_/Whisper/test_new/clip_nr_{p:.2f}.wav"
    print(f"Saving to {out_path} (took {elapsed:.1f}s)...")
    sf.write(out_path, audio_denoised, sr)

print("\nAll NR sweeps done!")
