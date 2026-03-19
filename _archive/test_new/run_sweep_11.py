import os
import time
import json
import re
from faster_whisper import WhisperModel

PROMPT = "以下是一节关于空气动力学的大学课程录音。包含专业术语如：法向、切向、流线、迹线、脉线、马赫数、雷诺数、层流、湍流、边界层、压强梯度、伯努利方程、欧拉方程、连续性方程、速度势、势流、涡量、环量、升力、阻力、攻角、翼型。"

# Format: (Name, VAD, Condition, RepPenalty, NGram, UsePrompt, BeamSize)
TESTS = [
    ("W01", False, False, 1.0, 0, False, 5),
    ("W02", True,  False, 1.0, 0, False, 5),
    ("W03", True,  True,  1.0, 0, False, 5),
    ("W04", True,  True,  1.2, 0, False, 5),
    ("W05", True,  True,  1.0, 3, False, 5),
    ("W06", True,  True,  1.0, 0, True,  5),
    ("W07", True,  True,  1.2, 3, False, 5),
    ("W08", True,  True,  1.2, 3, True,  5),
    ("W09", True,  False, 1.0, 0, True,  5),
    ("W10", True,  True,  1.5, 3, True,  5),
    ("W11", True,  True,  1.2, 3, True,  1)
]

AUDIO_FILES = {
    "nr_0.90": "c:/_/Whisper/test_new/clip_nr_0.90_norm.wav",
    "dfn_9": "c:/_/Whisper/test_new/clip_dfn_9_norm.wav"
}

print("Loading model large-v3...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Model loaded.\n")

for label, audio_path in AUDIO_FILES.items():
    if not os.path.exists(audio_path):
        print(f"Skipping {audio_path}, file not found")
        continue

    out_dir = f"c:/_/Whisper/test_new/sweep_results_{label}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"========== Processing {label} ==========")
    
    results = []
    for (name, vad, cond, rep, ngram, use_prompt, beam) in TESTS:
        print(f"  -> Running {name}: VAD={vad}, Cond={cond}, Rep={rep}, Ngram={ngram}, Prp={use_prompt}, Beam={beam}")
        
        prompt_str = PROMPT if use_prompt else None
        
        t0 = time.time()
        segments, info = model.transcribe(
            audio_path,
            language="zh",
            beam_size=beam,
            temperature=0,
            vad_filter=vad,
            condition_on_previous_text=cond,
            repetition_penalty=rep,
            no_repeat_ngram_size=ngram,
            initial_prompt=prompt_str,
            hallucination_silence_threshold=2.0
        )
        
        txt_path = os.path.join(out_dir, f"{name}.txt")
        full_text = ""
        seg_count = 0
        
        with open(txt_path, "w", encoding="utf-8") as f:
            for s in segments:
                seg_count += 1
                full_text += s.text
                f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
                
        elapsed = time.time() - t0
        
        faxiang = len(re.findall(r'法向', full_text))
        faxian = len(re.findall(r'发现(?!出|有|了)', full_text))
        
        res = {
            "name": name,
            "time_s": round(elapsed, 1),
            "segments": seg_count,
            "chars": len(full_text),
            "faxiang": faxiang,
            "faxian": faxian
        }
        results.append(res)
        print(f"     Done in {elapsed:.1f}s | segs: {seg_count} | 法向: {faxiang} | 发现: {faxian}")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print("\nALL OPERATIONS COMPLETE!")
