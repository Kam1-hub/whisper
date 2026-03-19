import os
import subprocess

lims = [6, 9, 12]
input_file = "c:/_/Whisper/test_new/clip_10m_20m.wav"

for l in lims:
    print(f"\nRunning DeepFilterNet with --atten-lim-db {l} ...")
    cmd = f'C:\\_\\Apps\\deep-filter.exe --atten-lim-db {l} -o "c:\\_\\Whisper\\test_new" "{input_file}"'
    subprocess.run(cmd, shell=True)
    
    # Rename output
    old_name = "c:/_/Whisper/test_new/clip_10m_20m_DeepFilterNet3.wav"
    new_name = f"c:/_/Whisper/test_new/clip_dfn_{l}.wav"
    if os.path.exists(old_name):
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)

print("\nAll DFN sweeps done!")
