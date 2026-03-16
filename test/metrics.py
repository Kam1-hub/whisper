import numpy as np
import soundfile as sf
import json

files = [
    ("c:\\_\\Whisper\\test\\step1_16k_mono.wav", "Step1_Raw"),
    ("c:\\_\\Whisper\\test\\step2_denoised.wav", "Step2_Denoise"),
    ("c:\\_\\Whisper\\test\\step3_normalized.wav", "Step3_Norm"),
]

results = []
for path, label in files:
    a, sr = sf.read(path)
    rms = float(20 * np.log10(np.sqrt(np.mean(a**2)) + 1e-10))
    peak = float(20 * np.log10(np.max(np.abs(a)) + 1e-10))
    chunks = np.array_split(a[:len(a) // 8000 * 8000], len(a) // 8000)
    chunk_rms = np.array([np.sqrt(np.mean(c**2)) for c in chunks])
    nf = float(20 * np.log10(np.mean(np.sort(chunk_rms)[:max(1, len(chunk_rms) // 10)]) + 1e-10))
    results.append({"label": label, "RMS_dB": round(rms, 2), "Peak_dB": round(peak, 2), "NoiseFloor_dB": round(nf, 2)})

with open("c:\\_\\Whisper\\test\\metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("OK")
