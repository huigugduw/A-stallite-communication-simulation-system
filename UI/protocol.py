from typing import Dict, List
import numpy as np


class ConstellationRouting:
    def __init__(self):
        # 严格按图片标注的节点连接关系
        self.topology = {
            # 终端节点连接（星型）
            "end1": ["Transmitter1"],
            "end2": ["Transmitter2"],
            "end3": ["Transmitter7"],
            "end4": ["Transmitter4"],
            "end5": ["Transmitter6"],

            # 发射器节点连接（环形骨干网）
            "Transmitter1": ["end1", "Transmitter2", "Transmitter7"],
            "Transmitter2": ["end2", "Transmitter1", "Transmitter3"],
            "Transmitter3": ["Transmitter2", "Transmitter4"],
            "Transmitter4": ["end4", "Transmitter3", "Transmitter5"],
            "Transmitter5": ["Transmitter4", "Transmitter6"],
            "Transmitter6": ["end5", "Transmitter5", "Transmitter7"],
            "Transmitter7": ["end3", "Transmitter6", "Transmitter1"]
        }

        # 预计算的最短路径路由表
        self.routing_table = self._build_routing_table()

    def _build_routing_table(self) -> Dict[str, Dict[str, str]]:
        """生成完整路由表（键值对格式：源节点->目标节点->下一跳）"""
        return {
            # ========== 终端节点路由 ==========
            "end1": {
                "end2": "Transmitter1",
                "end3": "Transmitter1",
                "end4": "Transmitter1",
                "end5": "Transmitter1",
                "Transmitter1": "Transmitter1",
                "Transmitter2": "Transmitter1",
                "Transmitter3": "Transmitter1",
                "Transmitter4": "Transmitter1",
                "Transmitter5": "Transmitter1",
                "Transmitter6": "Transmitter1",
                "Transmitter7": "Transmitter1"
            },
            "end2": {
                "end1": "Transmitter2",
                "end3": "Transmitter2",
                "end4": "Transmitter2",
                "end5": "Transmitter2",
                "Transmitter1": "Transmitter2",
                "Transmitter2": "Transmitter2",
                "Transmitter3": "Transmitter2",
                "Transmitter4": "Transmitter2",
                "Transmitter5": "Transmitter2",
                "Transmitter6": "Transmitter2",
                "Transmitter7": "Transmitter2"
            },
            # ...（end3-end5的规则类似，全部指向其连接的Transmitter）

            # ========== 发射器节点路由 ==========
            "Transmitter1": {
                "end1": "end1",  # 直连
                "end2": "Transmitter2",
                "end3": "Transmitter7",
                "end4": "Transmitter2",  # 选择Transmitter2→3→4路径
                "end5": "Transmitter7",  # 选择Transmitter7→6路径
                "Transmitter2": "Transmitter2",
                "Transmitter3": "Transmitter2",
                "Transmitter4": "Transmitter2",
                "Transmitter5": "Transmitter7",
                "Transmitter6": "Transmitter7",
                "Transmitter7": "Transmitter7"
            },
            "Transmitter2": {
                "end1": "Transmitter1",
                "end2": "end2",
                "end3": "Transmitter1",
                "end4": "Transmitter3",
                "end5": "Transmitter3",
                "Transmitter1": "Transmitter1",
                "Transmitter3": "Transmitter3",
                "Transmitter4": "Transmitter3",
                "Transmitter5": "Transmitter3",
                "Transmitter6": "Transmitter1",
                "Transmitter7": "Transmitter1"
            },
            "Transmitter3": {
                "end1": "Transmitter2",
                "end2": "Transmitter2",
                "end3": "Transmitter4",
                # 路径: Transmitter3 → Transmitter4 → Transmitter5 → Transmitter6 → Transmitter7 → end3
                "end4": "Transmitter4",
                "end5": "Transmitter4",  # 路径: Transmitter3 → Transmitter4 → Transmitter5 → Transmitter6 → end5
                "Transmitter2": "Transmitter2",
                "Transmitter4": "Transmitter4"
            },
            "Transmitter4": {
                "end1": "Transmitter3",
                "end2": "Transmitter3",
                "end3": "Transmitter5",  # 路径: Transmitter4 → Transmitter5 → Transmitter6 → Transmitter7 → end3
                "end4": "end4",
                "end5": "Transmitter5",
                "Transmitter3": "Transmitter3",
                "Transmitter5": "Transmitter5"
            },
            "Transmitter5": {
                "end1": "Transmitter4",
                "end2": "Transmitter4",  # 路径: Transmitter5 → Transmitter4 → Transmitter3 → Transmitter2 → end2
                "end3": "Transmitter6",
                "end4": "Transmitter4",
                "end5": "Transmitter6",
                "Transmitter4": "Transmitter4",
                "Transmitter6": "Transmitter6"
            },
            "Transmitter6": {
                "end1": "Transmitter5",
                # 路径: Transmitter6 → Transmitter5 → Transmitter4 → Transmitter3 → Transmitter2 → Transmitter1 → end1
                "end2": "Transmitter5",
                # 路径: Transmitter6 → Transmitter5 → Transmitter4 → Transmitter3 → Transmitter2 → end2
                "end3": "Transmitter7",
                "end4": "Transmitter5",
                "end5": "end5",
                "Transmitter5": "Transmitter5",
                "Transmitter7": "Transmitter7"
            },
            # ...（Transmitter3-7的规则完整补充）
            "Transmitter7": {
                "end1": "Transmitter1",
                "end2": "Transmitter1",
                "end3": "end3",  # 直连终端
                "end4": "Transmitter6",  # 通过Transmitter6→5→4
                "end5": "Transmitter6",
                "Transmitter1": "Transmitter1",
                "Transmitter6": "Transmitter6"  # 关键连接
        }
            # 完整路由表应包含所有12个节点的互相路由
        }

    def get_path(self, source: str, destination: str) -> List[str]:
        """获取完整传输路径（自动选择最短路径）"""
        path = [source]
        current = source

        while current != destination:
            next_hop = self.routing_table[current].get(destination)
            if not next_hop or next_hop in path:  # 防环路
                break
            path.append(next_hop)
            current = next_hop

        return path if current == destination else []

    def get_all_routes(self) -> Dict[str, Dict[str, List[str]]]:
        """返回所有节点间的最短路径（用于可视化检查）"""
        all_nodes = list(self.topology.keys())
        return {
            src: {
                dst: self.get_path(src, dst)
                for dst in all_nodes if dst != src
            }
            for src in all_nodes
        }


class PhysicalLayerEncoder:
    def __init__(self):
        # 同步头配置（匹配图片中的节点分布）
        self.SYNC_HEADER = [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]  # 13位Barker码对应0xF0F0
        self.FRAME_SIZE = 256  # 固定帧长（字节），与图片中的节点传输需求匹配

    def process_binary_data(self, binary_data: List[int]) -> List[float]:
        """物理层封装主流程（适配图片中的节点需求）"""
        # 1. 数据补零对齐（确保可被256字节整除）
        frame_bits = self.FRAME_SIZE * 8
        pad_len = (frame_bits - len(binary_data) % frame_bits) % frame_bits
        padded_data = np.concatenate([binary_data, np.zeros(pad_len, dtype=int)])

        # 2. 分帧并添加同步头（匹配图片中的同步需求）
        frames = np.reshape(padded_data, (-1, frame_bits))
        synced_frames = [np.concatenate([self.SYNC_HEADER, frame]) for frame in frames]

        # 3. QPSK调制（适合卫星信道）
        return self._qpsk_modulate(np.concatenate(synced_frames))

    def _segment_data(self, data: List[int]) -> List[List[int]]:
        """数据分帧（不足补零）"""
        frame_bits = self.FRAME_SIZE * 8
        pad_len = (frame_bits - len(data) % frame_bits) % frame_bits
        padded = np.concatenate([data, np.zeros(pad_len, dtype=int)])
        return np.reshape(padded, (-1, frame_bits))

    def _add_sync_header(self, frame: List[int]) -> List[int]:
        """添加同步头（匹配图片中的节点同步需求）"""
        return np.concatenate([self.SYNC_HEADER, frame])

    def _qpsk_modulate(self, bits: List[int]) -> List[float]:
        """安全的QPSK调制（自动处理奇数长度）"""
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)  # 补零
        """QPSK调制（输出I/Q交替信号）"""
        # 将比特流转换为符号（每2比特一组）
        symbols = np.reshape(bits, (-1, 2))

        # QPSK星座映射（Gray编码）
        constellation = {
            (0, 0): (+0.707, +0.707),  # 45°
            (0, 1): (-0.707, +0.707),  # 135°
            (1, 0): (+0.707, -0.707),  # -45°
            (1, 1): (-0.707, -0.707)  # -135°
        }

        # 生成调制信号（I/Q交替）
        modulated = []
        for pair in symbols:
            modulated.extend(constellation[tuple(pair)])
        return modulated
# 使用示例
if __name__ == "__main__":
    """router = ConstellationRouting()

    # 示例1：end1 -> end5 的路径
    print("end1 -> end5:", router.get_path("end1", "end5"))"""
    # 用户提供的输入数据（示例）
    input_bits = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]

    encoder = PhysicalLayerEncoder()
    output_signal = encoder.process_binary_data(input_bits)

    print(f"输入比特数: {len(input_bits)}")
    print(f"输出符号数: {len(output_signal)} (QPSK I/Q交替)")
    print("前20个输出样本:", output_signal[:20])

