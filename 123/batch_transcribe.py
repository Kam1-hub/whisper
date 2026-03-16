"""
批量转录脚本：11 个课程录音 → 11 个文本
===================================================
防中断设计：
  1. 断点续跑：已完成的文件自动跳过（检查输出 txt 是否已存在且非空）
  2. 单文件隔离：每个文件独立 try/except，一个失败不影响其余
  3. 内存回收：每个文件处理完后显式释放大数组，防止累积 OOM
  4. 中间文件清理：WAV 和降噪 WAV 转录完毕后自动删除，节省磁盘
  5. 进度日志：实时写入 progress.json，随时可查看当前进度
===================================================
"""
import os, sys, time, json, gc, subprocess, traceback
import numpy as np
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel

# ====================== 配置 ======================
INPUT_DIR   = "c:/_/Whisper/123"
OUTPUT_DIR  = "c:/_/Whisper/123/1234"
FFMPEG_PATH = r"C:\DownKyi-1.6.1\ffmpeg.exe"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== Prompt 定义 ======================
PROMPTS = {
    "空气动力学基础": (
        "以下是一节关于空气动力学基础的大学课程录音。包含专业术语如："
        "连续介质假设、克努森数、控制体、流体微团、完全气体、压缩性、黏性、"
        "理想流体、物质导数、速度散度、升力、阻力、力矩、压力中心、压心、"
        "气动力系数、动压、压强系数、升阻系数、连续方程、质量守恒、动量方程、"
        "能量方程、N-S方程、黏性流动、欧拉方程、无黏流动、流线、迹线、"
        "流函数、速度位、角变形率、有旋流动、无旋流动、涡线、涡管、"
        "速度环量、毕奥-萨伐尔定律、诱导速度。"
    ),
    "材料力学": (
        "以下是一节关于材料力学的大学课程录音。包含专业术语如："
        "强度、刚度、稳定性、均质连续性假设、各向同性假设、微小变形假设、"
        "弹性变形、塑性变形、轴向拉压、剪切、扭转、弯曲、内力、截面法、"
        "轴力、正应力、切应力、圣维南原理、线应变、纵向应变、横向应变、泊松比、"
        "切应力互等定理、胡克定律、弹性模量、杨氏模量、抗拉刚度、切变模量、"
        "应力-应变曲线、比例极限、弹性极限、屈服点、抗拉强度、颈缩、伸长率、"
        "断面收缩率、冷作硬化、塑性材料、脆性材料、蠕变、应力松弛、冲击韧度、"
        "疲劳破坏、交变应力、疲劳极限、许用应力、安全因数、超静定结构、"
        "挤压、挤压应力。"
    ),
    "机械原理": (
        "以下是一节关于机械原理的大学课程录音。包含专业术语如："
        "构件、运动副、低副、高副、转动副、移动副、自由度、约束、运动链、"
        "闭链、开链、机构、机架、原动件、主动件、从动件、机构运动简图、"
        "机构示意图、平面机构、复合铰链、局部自由度、虚约束、高副低代、"
        "平面机构组成原理、杆组、机构的级别、Ⅱ级组、Ⅲ级组、结构分析。"
    ),
    "c_cpp语言程序设计": (
        "以下是一节关于C/C++语言程序设计的大学课程录音。包含专业术语如："
        "变量、常量、数据类型、int、float、double、char、指针、数组、"
        "结构体、函数、递归、循环、条件语句、switch、for、while、"
        "头文件、宏定义、预处理、编译、链接、内存分配、malloc、free、"
        "栈、堆、作用域、生命周期、文件操作、标准输入输出、printf、scanf、"
        "字符串、链表、二叉树、排序算法、冒泡排序、选择排序、插入排序。"
    ),
}

def get_prompt(filename):
    """根据文件名前缀匹配课程 prompt"""
    for course, prompt in PROMPTS.items():
        if filename.startswith(course):
            return course, prompt
    # 通用 fallback：仍然告诉 Whisper 这是中文课程录音
    return "未知课程", "以下是一节大学课程录音。"

def load_progress():
    """加载进度文件"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "results": {}}

def save_progress(progress):
    """保存进度文件"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def process_one_file(aac_path, model):
    """处理单个 AAC 文件的完整流水线，返回统计信息"""
    basename = os.path.splitext(os.path.basename(aac_path))[0]
    course, prompt = get_prompt(basename)

    # 临时文件（用完即删）
    wav_tmp = os.path.join(OUTPUT_DIR, f"{basename}_16k_mono.wav")
    nr_tmp  = os.path.join(OUTPUT_DIR, f"{basename}_nr090.wav")
    # 最终输出（与音频同名，扩展名 .txt）
    out_txt = os.path.join(OUTPUT_DIR, f"{basename}.txt")

    stats = {"file": basename, "course": course}

    # ---- Step A: ffmpeg AAC → 16kHz mono WAV ----
    print(f"    [A] ffmpeg 转码...", end=" ", flush=True)
    cmd = [FFMPEG_PATH, "-y", "-i", aac_path,
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="ignore", timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-200:]}")
    info = sf.info(wav_tmp)
    stats["duration_s"] = round(info.duration, 1)
    print(f"{info.duration:.0f}s ✅")

    # ---- Step B: noisereduce 降噪 + 归一化 ----
    print(f"    [B] 降噪 + 归一化...", end=" ", flush=True)
    audio, sr = sf.read(wav_tmp, dtype='float64')
    audio_dn = nr.reduce_noise(
        y=audio, sr=sr,
        stationary=True, prop_decrease=0.90,
        n_fft=2048, freq_mask_smooth_hz=500
    )
    peak = np.max(np.abs(audio_dn))
    if peak > 0:
        audio_dn = audio_dn * (0.95 / peak)
    sf.write(nr_tmp, audio_dn, sr, subtype='PCM_16')
    del audio        # 释放原始音频内存
    del audio_dn     # 释放降噪音频内存
    gc.collect()
    print("✅")

    # 删除原始 WAV（降噪版已保存）
    try:
        os.remove(wav_tmp)
    except:
        pass

    # ---- Step C: Whisper V3 转录 ----
    print(f"    [C] Whisper 转录 (prompt={course})...", flush=True)
    t0 = time.time()
    segments, _ = model.transcribe(
        nr_tmp,
        language="zh",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=prompt,
    )

    seg_count = 0
    char_count = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for s in segments:
            seg_count += 1
            char_count += len(s.text)
            f.write(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}\n")
            if seg_count % 500 == 0:
                print(f"        ... {seg_count} segs, {char_count} chars")

    elapsed = time.time() - t0
    stats["segs"] = seg_count
    stats["chars"] = char_count
    stats["time_s"] = round(elapsed, 1)
    stats["file_size"] = os.path.getsize(out_txt)
    print(f"        → {seg_count} segs, {char_count} chars, {elapsed:.0f}s ✅")

    # 删除降噪 WAV（文本已保存）
    try:
        os.remove(nr_tmp)
    except:
        pass

    gc.collect()
    return stats


# ====================== 主流程 ======================
if __name__ == "__main__":
    print("=" * 60)
    print("批量转录 - 自动检测新文件")
    print("=" * 60)

    # 扫描输入
    aac_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".aac")])
    print(f"发现 {len(aac_files)} 个 AAC 文件\n")

    # 加载进度
    progress = load_progress()

    # 检查哪些已经完成（输出 txt 存在且 > 1KB）
    for f in aac_files:
        basename = os.path.splitext(f)[0]
        out_txt = os.path.join(OUTPUT_DIR, f"{basename}.txt")
        if os.path.exists(out_txt) and os.path.getsize(out_txt) > 1024:
            if basename not in progress["completed"]:
                progress["completed"].append(basename)

    todo = [f for f in aac_files
            if os.path.splitext(f)[0] not in progress["completed"]]
    print(f"已完成: {len(progress['completed'])} 个")
    print(f"待处理: {len(todo)} 个\n")

    if not todo:
        print("全部已完成，无需重跑！")
    else:
        # 加载模型（只加载一次）
        print("加载 Whisper 模型 (large-v3, CUDA, float16)...")
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print("模型加载完毕\n")

        for i, filename in enumerate(todo):
            basename = os.path.splitext(filename)[0]
            course, _ = get_prompt(basename)
            aac_path = os.path.join(INPUT_DIR, filename)

            print(f"\n{'─' * 60}")
            print(f"[{len(progress['completed'])+1}/{len(aac_files)}] {filename} ({course})")
            print(f"{'─' * 60}")

            try:
                stats = process_one_file(aac_path, model)
                progress["completed"].append(basename)
                progress["results"][basename] = stats
                save_progress(progress)

                # 验证
                if stats.get("segs", 0) < 250 or stats.get("chars", 0) < 2500:
                    print(f"    ⚠️ 输出偏薄 (segs={stats.get('segs', 0)}, chars={stats.get('chars', 0)})")
                else:
                    print(f"    ✅ 验证通过")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"    ❌ 失败: {error_msg}")
                traceback.print_exc()
                progress["failed"].append({"file": basename, "error": error_msg})
                save_progress(progress)
                # 清理可能残留的临时文件
                for tmp in [os.path.join(OUTPUT_DIR, f"{basename}_16k_mono.wav"),
                            os.path.join(OUTPUT_DIR, f"{basename}_nr090.wav")]:
                    try: os.remove(tmp)
                    except: pass
                gc.collect()
                continue  # 继续处理下一个文件

    # ====================== 最终汇总 ======================
    print("\n" + "=" * 60)
    print("最终汇总")
    print("=" * 60)

    total_segs = 0
    total_chars = 0
    total_time = 0

    print(f"\n{'文件':<35} {'课程':<12} {'段数':>6} {'字数':>7} {'耗时':>6}")
    print("─" * 70)
    for basename, stats in sorted(progress.get("results", {}).items()):
        segs = stats.get("segs", 0)
        chars = stats.get("chars", 0)
        t = stats.get("time_s", 0)
        total_segs += segs
        total_chars += chars
        total_time += t
        print(f"{basename:<35} {stats.get('course',''):<12} {segs:>6} {chars:>7} {t:>5.0f}s")

    print("─" * 70)
    print(f"{'总计':<35} {'':<12} {total_segs:>6} {total_chars:>7} {total_time:>5.0f}s")

    if progress.get("failed"):
        print(f"\n❌ 失败文件 ({len(progress['failed'])}):")
        for f in progress["failed"]:
            print(f"  - {f['file']}: {f['error']}")
    else:
        print(f"\n✅ 全部 {len(progress['completed'])} 个文件处理完毕，无失败！")

    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"进度文件: {PROGRESS_FILE}")
