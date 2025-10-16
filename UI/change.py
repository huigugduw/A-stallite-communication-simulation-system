import math
from typing import Dict, List, Tuple
import random


class ConstellationRouting:
    def __init__(self):
        # 节点坐标（模拟位置，用于计算时延）
        self.node_positions = {
            "end1": (0.2, 0.8),
            "end2": (0.4, 0.6),
            "end3": (0.8, 0.2),
            "end4": (0.6, 0.4),
            "end5": (0.8, 0.8),
            "Transmitter1": (0.1, 0.9),
            "Transmitter2": (0.3, 0.7),
            "Transmitter3": (0.5, 0.5),
            "Transmitter4": (0.7, 0.3),
            "Transmitter5": (0.7, 0.7),
            "Transmitter6": (0.9, 0.9),
            "Transmitter7": (0.9, 0.1)
        }

        # 拓扑结构（包含时延权重）
        self.topology = self._build_topology_with_latency()

        # 预计算路由表（包含最短路径和时延）
        self.routing_table = self._build_routing_table()

    def _calculate_latency(self, node1: str, node2: str) -> float:
        """计算两节点之间的模拟时延（基于距离+随机扰动）"""
        pos1 = self.node_positions[node1]
        pos2 = self.node_positions[node2]

        # 欧氏距离（模拟物理距离）
        distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

        # 基础时延（ms）：距离*100 + 5~15ms随机处理时延
        return distance * 100 + random.uniform(5, 15)

    def _build_topology_with_latency(self) -> Dict[str, Dict[str, float]]:
        """构建完整的带时延权重的拓扑结构"""
        return {
            # 终端节点连接
            "end1": {"Transmitter1": self._calculate_latency("end1", "Transmitter1")},
            "end2": {"Transmitter2": self._calculate_latency("end2", "Transmitter2")},
            "end3": {"Transmitter7": self._calculate_latency("end3", "Transmitter7")},
            "end4": {"Transmitter4": self._calculate_latency("end4", "Transmitter4")},
            "end5": {"Transmitter6": self._calculate_latency("end5", "Transmitter6")},

            # 发射器节点连接（环形骨干网）
            "Transmitter1": {
                "end1": self._calculate_latency("Transmitter1", "end1"),
                "Transmitter2": self._calculate_latency("Transmitter1", "Transmitter2"),
                "Transmitter7": self._calculate_latency("Transmitter1", "Transmitter7")
            },
            "Transmitter2": {
                "end2": self._calculate_latency("Transmitter2", "end2"),
                "Transmitter1": self._calculate_latency("Transmitter2", "Transmitter1"),
                "Transmitter3": self._calculate_latency("Transmitter2", "Transmitter3")
            },
            "Transmitter3": {
                "Transmitter2": self._calculate_latency("Transmitter3", "Transmitter2"),
                "Transmitter4": self._calculate_latency("Transmitter3", "Transmitter4")
            },
            "Transmitter4": {
                "end4": self._calculate_latency("Transmitter4", "end4"),
                "Transmitter3": self._calculate_latency("Transmitter4", "Transmitter3"),
                "Transmitter5": self._calculate_latency("Transmitter4", "Transmitter5")
            },
            "Transmitter5": {
                "Transmitter4": self._calculate_latency("Transmitter5", "Transmitter4"),
                "Transmitter6": self._calculate_latency("Transmitter5", "Transmitter6")
            },
            "Transmitter6": {
                "end5": self._calculate_latency("Transmitter6", "end5"),
                "Transmitter5": self._calculate_latency("Transmitter6", "Transmitter5"),
                "Transmitter7": self._calculate_latency("Transmitter6", "Transmitter7")
            },
            "Transmitter7": {
                "end3": self._calculate_latency("Transmitter7", "end3"),
                "Transmitter6": self._calculate_latency("Transmitter7", "Transmitter6"),
                "Transmitter1": self._calculate_latency("Transmitter7", "Transmitter1")
            }
        }

    def _build_routing_table(self) -> Dict[str, Dict[str, Tuple[str, float]]]:
        """使用Dijkstra算法计算最短时延路径（完整实现）"""
        # 初始化路由表
        routing_table = {}
        all_nodes = list(self.topology.keys())

        for src in all_nodes:
            routing_table[src] = {}
            for dst in all_nodes:
                if src == dst:
                    continue

                # 获取最短路径
                path = self._dijkstra_shortest_path(src, dst)
                if path:
                    next_hop = path[1] if len(path) > 1 else None
                    total_latency = self._get_path_latency(path)
                    routing_table[src][dst] = (next_hop, total_latency)

        return routing_table

    def _dijkstra_shortest_path(self, src: str, dst: str) -> List[str]:
        """Dijkstra算法实现最短路径查找"""
        distances = {node: float('inf') for node in self.topology}
        previous_nodes = {node: None for node in self.topology}
        distances[src] = 0
        unvisited = set(self.topology.keys())

        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            unvisited.remove(current)

            if current == dst:
                break

            for neighbor, latency in self.topology[current].items():
                new_distance = distances[current] + latency
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current

        # 回溯构建路径
        path = []
        current = dst
        while previous_nodes[current] is not None:
            path.insert(0, current)
            current = previous_nodes[current]

        if path:
            path.insert(0, src)
            return path
        return []

    def _get_path_latency(self, path: List[str]) -> float:
        """计算路径总时延"""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.topology[path[i]][path[i + 1]]
        return total

    def get_path(self, source: str, destination: str) -> List[str]:
        """获取完整传输路径（自动选择最短路径）"""
        path = [source]
        current = source

        while current != destination:
            next_hop = self.routing_table[current].get(destination, (None, None))[0]
            if not next_hop or next_hop in path:  # 防环路
                break
            path.append(next_hop)
            current = next_hop

        return path if current == destination else []

    def get_path_with_latency(self, source: str, destination: str) -> Tuple[List[str], float]:
        """获取路径和总时延"""
        path = self.get_path(source, destination)
        latency = self._get_path_latency(path) if path else float('inf')
        return path, latency

    def simulate_transmission(self, source: str, destination: str, data_size: int) -> Dict:
        """
        模拟完整传输过程
        返回: {
            "path": ["end1", "Transmitter1", ...],
            "total_latency": 125.3,  # ms
            "hop_count": 3,
            "per_hop_latency": [("end1", "Transmitter1", 32.1), ...],
            "data_size": 1  # MB
        }
        """
        path = self.get_path(source, destination)
        if not path:
            return {"error": "No valid path"}

        result = {
            "path": path,
            "total_latency": 0.0,
            "hop_count": len(path) - 1,
            "per_hop_latency": [],
            "data_size": data_size
        }

        # 计算每跳时延
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            latency = self.topology[from_node][to_node]
            result["per_hop_latency"].append((from_node, to_node, latency))
            result["total_latency"] += latency

        # 添加数据量影响（每MB增加0.1ms）
        result["total_latency"] += data_size * 0.1

        return result

    def visualize_topology(self):
        """可视化网络拓扑（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle

            plt.figure(figsize=(10, 8))

            # 绘制连接线
            for src, connections in self.topology.items():
                for dst, _ in connections.items():
                    x1, y1 = self.node_positions[src]
                    x2, y2 = self.node_positions[dst]
                    plt.plot([x1, x2], [y1, y2], 'b-', alpha=0.3)

            # 绘制节点
            for node, (x, y) in self.node_positions.items():
                color = 'red' if node.startswith('end') else 'green'
                plt.plot(x, y, 'o', markersize=15, markerfacecolor=color, markeredgecolor='black')
                plt.text(x, y + 0.02, node, ha='center', fontsize=9)

            plt.title("Satellite Network Topology with Latency")
            plt.grid(True)
            plt.show()

        except ImportError:
            print("Visualization requires matplotlib. Please install with: pip install matplotlib")

