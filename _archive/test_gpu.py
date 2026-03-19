"""
faster-whisper GPU 环境验证脚本
验证 ctranslate2 能否检测到 GPU 并正常加载模型
"""
import sys
print(f"Python 版本: {sys.version}")

# 检查 ctranslate2 CUDA 支持
import ctranslate2
print(f"ctranslate2 版本: {ctranslate2.__version__}")

cuda_device_count = ctranslate2.get_cuda_device_count()
print(f"检测到 CUDA 设备数量: {cuda_device_count}")

if cuda_device_count > 0:
    print("✅ GPU 可用！faster-whisper 可以使用 GPU 推理。")
else:
    print("❌ 未检测到 CUDA 设备，将只能使用 CPU 推理。")

# 尝试加载一个小模型测试
print("\n正在下载并加载 tiny 模型进行测试...")
from faster_whisper import WhisperModel

try:
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    print("✅ 模型加载成功！GPU 推理一切正常。")
    del model
except Exception as e:
    print(f"⚠️ GPU 加载失败: {e}")
    print("尝试 CPU 模式...")
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ CPU 模式加载成功。")
        del model
    except Exception as e2:
        print(f"❌ CPU 模式也失败了: {e2}")

print("\n验证完成！")
