import os
import subprocess
import shutil

lims = [6, 9, 12]
src_file = "c:/_/Whisper/test_new/clip_10m_20m.wav"

for l in lims:
    print(f"\n========================================")
    print(f"Running DeepFilterNet with --atten-lim-db {l}")
    
    # DFN overwrites the file if output dir is the same as input dir
    # So we copy the source file to a temporary name first
    temp_target = f"c:/_/Whisper/test_new/temp_dfn_input_{l}.wav"
    out_target = f"c:/_/Whisper/test_new/clip_dfn_{l}.wav"
    
    shutil.copy2(src_file, temp_target)
    
    cmd = f'C:\\_\\Apps\\deep-filter.exe --atten-lim-db {l} -o "c:\\_\\Whisper\\test_new" "{temp_target}"'
    subprocess.run(cmd, shell=True)
    
    # Now temp_target has been overwritten by DFN. Rename it to final target.
    if os.path.exists(out_target):
        os.remove(out_target)
    
    if os.path.exists(temp_target):
        os.rename(temp_target, out_target)
        print(f"Successfully generated {out_target}")
    else:
        print(f"ERROR: {temp_target} not found after DFN run!")

print("\nAll DFN sweeps done!")
