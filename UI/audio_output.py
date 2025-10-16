import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt


class AudioOutput:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate  # 标准音频采样率

    def play_waveform(self, signal, volume=0.5):
        """
        播放数字信号转换的音频
        :param signal: 输入的离散信号值（数组）
        :param volume: 音量调节系数 (0.0-1.0)
        """
        # 确保信号在合法范围内 (-1.0 到 1.0)
        signal = np.clip(signal * volume, -0.99, 0.99).astype(np.float32)

        print(f"正在播放 {len(signal) / self.sample_rate:.2f} 秒音频...")
        sd.play(signal, self.sample_rate)
        sd.wait()  # 等待播放完成

    def visualize_waveform(self, signal, title="音频波形"):
        """绘制信号波形图"""
        plt.figure(figsize=(10, 4))
        plt.plot(signal[:1000])  # 只显示前1000个采样点
        plt.title(title)
        plt.xlabel("采样点")
        plt.ylabel("振幅")
        plt.show()


# ---------------------------------------------------
# 示例使用：将Turbo解码后的数据转为音频
# ---------------------------------------------------
if __name__ == "__main__":
    # 1. 模拟DA转换输出（示例：生成1kHz正弦波）
    duration = 2  # 秒
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880Hz音调

    # 2. 创建音频输出实例
    audio = AudioOutput(sample_rate)

    # 3. 可视化波形
    audio.visualize_waveform(signal)

    # 4. 播放音频
    audio.play_waveform(signal, volume=0.8)