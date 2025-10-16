import sys
import os
from audio_output import AudioOutput
from beamforming import SatelliteBeamController
from decoder import PhysicalLayerDecoder, extract_original_data
import logging
from change import ConstellationRouting
from protocol import  PhysicalLayerEncoder
from signalcreat import create_voice_function

resource_dir = os.path.join(os.path.dirname(__file__), "images")  # 替换为实际路径
sys.path.append(resource_dir)  # 临时添加路径
import code#引入资源
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox, QTextBrowser, QSpinBox, \
    QLineEdit, QComboBox
from PyQt5 import uic
import numpy as np
from ADDA import PCM_ADC
from encoder import preprocess_data, TurboEncoder
from DA import PCM_DAC
# 获取 code.py 的绝对路径
resource_dir = os.path.join(os.path.dirname(__file__), "images")  # 替换为实际路径
sys.path.append(resource_dir)  # 临时添加路径
import code#引入资源
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SatModem')
class Stats():
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("stats.ui")
        self.button1 = self.ui.findChild(QPushButton, "pushButton_9")  #AD
        self.button2 = self.ui.findChild(QPushButton, "pushButton_8")#turbo码
        self.button3 = self.ui.findChild(QPushButton, "pushButton_10")  #调制
        self.button4 = self.ui.findChild(QPushButton, "pushButton_7")#协议
        self.button5 = self.ui.findChild(QPushButton, "pushButton_2")#DA获取信号
        self.button6 = self.ui.findChild(QPushButton, "pushButton_4")#输入
        self.button7 = self.ui.findChild(QPushButton, "pushButton")#解调
        self.button8 = self.ui.findChild(QPushButton, "pushButton_6")#获取信号
        self.button9 = self.ui.findChild(QPushButton, "pushButton_5")  #星地交换
        self.button10 = self.ui.findChild(QPushButton, "pushButton_3")  #波束控制

        # 3. 绑定按钮点击事件
        if self.button1:
            self.button1.clicked.connect(self.handleCalc1)
        if self.button2:
            self.button2.clicked.connect(self.handleCalc2)
        if self.button3:
            self.button3.clicked.connect(self.handleCalc3)
        if self.button4:
            self.button4.clicked.connect(self.handleCalc4)
        if self.button5:
            self.button5.clicked.connect(self.handleCalc5)
        if self.button6:
            self.button6.clicked.connect(self.handleCalc6)
        if self.button7:
            self.button7.clicked.connect(self.handleCalc7)
        if self.button8:
            self.button8.clicked.connect(self.handleCalc8)
        if self.button9:
            self.button9.clicked.connect(self.handleCalc9)
        if self.button10:
            self.button10.clicked.connect(self.handleCalc10)
        else:
            print("错误：未找到按钮对象！")
        self.text_browser = self.ui.findChild(QTextBrowser, "textBrowser")
        self.text_browser1 = self.ui.findChild(QTextBrowser, "textBrowser_2")
        if not self.text_browser:
            print("错误：未找到文本框对象！")
        self.box = self.ui.findChild(QSpinBox, "spinBox")#采样率
        self.box.setValue(8000)
        self.box1 = self.ui.findChild(QSpinBox, "spinBox_2")#采样时间
        self.box1.setValue(2)
        self.edit= self.ui.findChild(QLineEdit, "lineEdit")#文件选择
        self.cbox1= self.ui.findChild(QComboBox, "comboBox")#信号选择
        self.cbox2= self.ui.findChild(QComboBox, "comboBox_2")#输入端
        self.cbox3= self.ui.findChild(QComboBox, "comboBox_3")#输出端


    def handleCalc1(self):
        adc = PCM_ADC(fs=self.box.value(), bits=8)
        text = self.edit.text()
        voice_signal = create_voice_function(text)
        # 模拟语音信号 (300-3400Hz)
        """def voice_signal(t):
            # 基频 + 谐波成分模拟语音信号
            return (0.5 * np.sin(2 * np.pi * 800 * t) +
                    0.3 * np.sin(2 * np.pi * 1500 * t) +
                    0.2 * np.sin(2 * np.pi * 2500 * t))"""

        # 处理信号并获取PCM编码
        pcm_codes = adc.process(voice_signal, duration=self.box1.value(),plot=True)
        print(pcm_codes)
        self.pcm_codes=pcm_codes



    def handleCalc2(self): # 1. 生成测试数据
        self.text_browser.append("编码模块:")
        # 1. AD模块输出
        ad_output = self.pcm_codes

        # 2. 数据预处理
        input_bits = preprocess_data(ad_output, interleaver_size=128)
        self.input_bits=input_bits
        self.text_browser.append(f"预处理后比特数: {len(input_bits)} (原始{len(self.pcm_codes)*8} + 填充{len(input_bits) - len(self.pcm_codes)*8})")

        # 3. Turbo编码
        turbo_encoder = TurboEncoder()
        encoded_bits = turbo_encoder.encode(input_bits)
        self.encoded_bit=encoded_bits
        self.text_browser.append(f"Turbo编码后长度: {len(encoded_bits)} (码率≈1/3)")

        # 4. 验证编码结构（系统位保留原始信息）
        assert all(encoded_bits[i * 3] == input_bits[i] for i in range(len(input_bits))), "系统位校验失败"
        self.text_browser.append("验证通过：系统位正确保留原始信息")
        print(encoded_bits)

    def handleCalc3(self):
        self.text_browser.append("调制模块:")
        input_bits = self.encoded_bit
        encoder = PhysicalLayerEncoder()
        output_signal = encoder.process_binary_data(input_bits)
        self.modomed=output_signal

        len_input = len(input_bits)
        len_output = len(output_signal)

    # 正确写法：将所有内容转为单个字符串
        self.text_browser.append(f"输入比特数: {len_input}")
        self.text_browser.append(f"输出符号数: {len_output} (QPSK I/Q交替)")
        self.text_browser.append(f"前20个输出样本: {str(output_signal[:20])}")  # 关键修正：使用f-string

    def handleCalc4(self):
        self.text_browser.append("物理层-调制方式：QPSK（适用于卫星业务）- 同步机制：16-bit Barker码（0xF0F0）-信道编码：Turbo码")
        self.text_browser.append("数据链路层- 成帧：固定帧长256字节- ARQ：选择性重传- 多址接入：TDMA预分配时隙（10ms/时隙）")
        self.text_browser.append("网络层- 路由策略：静态路由表（预设Transmitter1-7和end1-5拓扑）,使用Disk算法求出最短路径-封装格式：简化AOS帧实现IP封装")


    def handleCalc5(self):
        # (来自ADC模块的输出)
        test_codes = self.recovered_data

        # 创建DAC实例 (参数需与ADC匹配)
        dac = PCM_DAC(fs=8000, bits=8)

        # 处理PCM编码并重建模拟信号
        time_points, analog_signal = dac.process(test_codes, target_power=-20)

        # 打印重建信号的峰值
        print(f"重建信号峰值: {np.max(np.abs(analog_signal)):.4f}")
        # 1. DA转换输出
        duration = self.box1.value()  # 秒
        sample_rate = self.box.value()*10
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = analog_signal

        # 2. 创建音频输出实例
        audio = AudioOutput(sample_rate)

        # 3. 可视化波形
        audio.visualize_waveform(signal)

        # 4. 播放音频
        audio.play_waveform(signal, volume=0.8)

    def handleCalc6(self):
        adc = PCM_ADC(fs=self.box.value(), bits=8)
        text = self.edit.text()
        voice_signal = create_voice_function(text)
        pcm_codes = adc.process(voice_signal, duration=self.box1.value(),plot=False)
        self.pcm_codes = pcm_codes
        ad_output = self.pcm_codes
        # 2. 数据预处理
        input_bits = preprocess_data(ad_output, interleaver_size=128)
        self.input_bits=input_bits
        # 3. Turbo编码
        turbo_encoder = TurboEncoder()
        encoded_bits = turbo_encoder.encode(input_bits)
        self.encoded_bit=encoded_bits
        encoder = PhysicalLayerEncoder()
        output_signal = encoder.process_binary_data(self.encoded_bit)
        self.modomed=output_signal
        decoder = PhysicalLayerDecoder()
        output_bits = decoder.demodulate_signal(self.modomed)
        self.recovered_data = extract_original_data(output_bits,original_length=len(self.pcm_codes)*8)
        self.text_browser.append("信号已输入")

    def handleCalc7(self):
        self.text_browser.append("解调")
        input_bits = self.encoded_bit
        modulated = self.modomed
    # 解码过程
        decoder = PhysicalLayerDecoder()
        output_bits = decoder.demodulate_signal(modulated)

    # 验证结果
        self.text_browser.append("==== 解码验证结果 ====")
        self.text_browser.append(f"原始数据长度: {len(input_bits)} bits")
        self.text_browser.append(f"解码数据长度: {len(output_bits)} bits")
        self.recovered_data = extract_original_data(output_bits,original_length=len(self.pcm_codes)*8)
        print(self.recovered_data)
        if type(self.pcm_codes) == type(self.recovered_data):
            self.text_browser.append("数据类型相同")
        else:
            self.text_browser.append(f"数据类型不同: pcm_codes是{type(self.pcm_codes)}, recovered_data是{type(self.recovered_data)}")

    def handleCalc8(self):
        router = ConstellationRouting()
        self.text_browser1.append("信号传输路径:")
        source = self.cbox2.currentText()
        target = self.cbox3.currentText()
        # 执行模拟
        data_size=1
        result = router.simulate_transmission(source, target, data_size)

        # 处理结果
        if "error" in result:
            self.text_browser1.append(f"传输错误: {result['error']}")
            return

        # 构建显示内容
        path_str = " → ".join(result["path"])
        latency_str = f"{result['total_latency']:.2f}ms"

        # 显示结果
        self.text_browser1.append("=== 信号传输模拟结果 ===")
        self.text_browser1.append(f"路径: {path_str}")
        self.text_browser1.append(f"总时延: {latency_str}")
        self.text_browser1.append("\n逐跳时延详情:")

        for hop in result["per_hop_latency"]:
            self.text_browser1.append(
                f"  {hop[0]} → {hop[1]}: {hop[2]:.2f}ms"
            )
        test_codes = self.pcm_codes

        # 创建DAC实例 (参数需与ADC匹配)
        dac = PCM_DAC(fs=8000, bits=8)

        # 处理PCM编码并重建模拟信号
        time_points, analog_signal = dac.process(test_codes, target_power=-20)

        # 打印重建信号的峰值
        print(f"重建信号峰值: {np.max(np.abs(analog_signal)):.4f}")
        # 1. DA转换输出
        duration = self.box1.value()  # 秒
        sample_rate = self.box.value()*10
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = analog_signal

        # 2. 创建音频输出实例
        audio = AudioOutput(sample_rate)

        # 3. 可视化波形
        audio.visualize_waveform(signal)

        # 4. 播放音频
        audio.play_waveform(signal, volume=0.8)

    def handleCalc9(self):
        self.text_browser.append(
            "{\n"
            "  \"end1\": [\"Transmitter1\"],\n"
            "  \"end2\": [\"Transmitter2\"],\n"
            "  \"end3\": [\"Transmitter7\"],\n"
            "  \"end4\": [\"Transmitter4\"],\n"
            "  \"end5\": [\"Transmitter6\"]\n"
            "}\n\n"  # ← 注意这里多了一个\n
            "Transmitter1: ['end1', 'Transmitter2', 'Transmitter7'],\n"
            "Transmitter2: ['end2', 'Transmitter1', 'Transmitter3'],\n"
            "Transmitter3: ['Transmitter2', 'Transmitter4'],\n"
            "Transmitter4: ['end4', 'Transmitter3', 'Transmitter5'],\n"
            "Transmitter5: ['Transmitter4', 'Transmitter6'],\n"
            "Transmitter6: ['end5', 'Transmitter5', 'Transmitter7'],\n"
            "Transmitter7: ['end3', 'Transmitter6', 'Transmitter1']"
        )

    def handleCalc10(self):
            # 初始化控制器（匹配图片中的节点）
            controller = SatelliteBeamController(
                transmitters=[f"Transmitter{i}" for i in range(1, 8)],
                ends=[f"end{i}" for i in range(1, 6)]
            )

            # 模拟动态波束控制
            for t in np.arange(0, 1, 0.1):  # 模拟1秒内10个时隙
                beams = controller.dynamic_beamforming(t)
                print(f"Time {t:.1f}s:")
                for tx, params in beams.items():
                    print(f"  {tx} -> {params['target']} "
                          f"| Freq: {params['freq'] / 1e9:.2f}GHz "
                          f"| Power: {params['power']:.1f}W "
                          f"| Interf: {controller.interference.get(tx, 0):.2f}")
                # 可视化拓扑（模拟原图风格）
            controller.visualize_topology()
app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec()