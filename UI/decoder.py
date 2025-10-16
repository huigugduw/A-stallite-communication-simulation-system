import numpy as np
from typing import List
class PhysicalLayerDecoder:
    def __init__(self):
        # 必须与编码器参数一致（根据图片中的节点配置）
        self.SYNC_HEADER = [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]  # 13位Barker码
        self.SYNC_LEN = len(self.SYNC_HEADER)
        self.FRAME_SIZE = 256  # 字节单位，需与编码器完全一致
        self.FRAME_BITS = self.FRAME_SIZE * 8

    def demodulate_signal(self, modulated_signal: List[float]) -> List[int]:
        """完整解调流程（输入QPSK信号，输出原始二进制）"""
        # 1. QPSK解调
        bits = self._qpsk_demodulate(modulated_signal)

        # 2. 帧同步与数据提取
        frames = self._extract_frames(bits)

        # 3. 合并数据并去除填充位
        return self._remove_padding(np.concatenate(frames))

    def _qpsk_demodulate(self, signal: List[float]) -> List[int]:
        """QPSK解调（匹配编码器的星座图）"""
        # 将I/Q交替信号分离
        i_signal = signal[::2]
        q_signal = signal[1::2]

        # 解调为比特流
        bits = []
        for i, q in zip(i_signal, q_signal):
            if i >= 0 and q >= 0:
                bits.extend([0, 0])
            elif i < 0 and q >= 0:
                bits.extend([0, 1])
            elif i >= 0 and q < 0:
                bits.extend([1, 0])
            else:
                bits.extend([1, 1])
        return bits

    def _extract_frames(self, bits: List[int]) -> List[List[int]]:
        """帧同步与数据提取（处理图片中的多帧情况）"""
        frames = []
        idx = 0

        while idx < len(bits):
            # 查找同步头（适应图片中的节点同步需求）
            sync_pos = self._find_sync(bits[idx:idx + self.FRAME_BITS + self.SYNC_LEN])
            if sync_pos == -1:
                break

            # 提取有效数据（跳过同步头）
            frame_start = idx + sync_pos + self.SYNC_LEN
            frame_end = frame_start + self.FRAME_BITS
            frames.append(bits[frame_start:frame_end])

            idx = frame_end

        return frames

    def _find_sync(self, search_window: List[int]) -> int:
        """在数据流中定位同步头（匹配图片中的Barker码）"""
        for i in range(len(search_window) - self.SYNC_LEN):
            if np.array_equal(search_window[i:i + self.SYNC_LEN], self.SYNC_HEADER):
                return i
        return -1

    def _remove_padding(self, data: List[int]) -> List[int]:
        """去除编码时的填充零（恢复原始数据长度）"""
        # 查找最后一个非零位（根据图片中的帧结构特点）
        last_nonzero = len(data)
        while last_nonzero > 0 and data[last_nonzero - 1] == 0:
            last_nonzero -= 1
        return data[:last_nonzero]
def extract_original_data(encoded_bits, original_length=640):
    """
    直接从Turbo编码数据中提取系统位还原原始数据
    :param encoded_bits: Turbo编码后的比特流（系统位+校验位1+校验位2交替排列）
    :param original_length: 原始数据长度（80 * 8=640）
    :return: 80个8位二进制字符串列表
    """
    # 1. 提取系统位（每3个比特中的第1位）
    sys_bits = encoded_bits[::3]  # 步长3取系统位

    # 2. 去除填充位
    sys_bits = sys_bits[:original_length]

    # 3. 转换为8位二进制字符串
    binary_codes = []
    for i in range(0, original_length, 8):
        byte = sys_bits[i:i + 8]
        binary_str = ''.join(str(bit) for bit in byte)
        binary_codes.append(binary_str)

    return binary_codes

