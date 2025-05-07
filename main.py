import re
import sys
import random
import heapq
import matplotlib.pyplot as plt  # 新增导入matplotlib
import networkx as nx  # 新增导入networkx
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Optional


class TextGraph:
    def __init__(self):
        self.graph = defaultdict(dict)  # 邻接表表示的有向图
        self.words = set()  # 所有单词集合
        self.in_degree = defaultdict(int)  # 入度统计
        self.out_degree = defaultdict(int)  # 出度统计

    def build_graph(self, text: str):
        """从文本构建有向图"""
        # 预处理文本：替换非字母字符为空格，转换为小写
        processed = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        words = processed.split()  # 分割单词

        if len(words) < 2:
            return

        self.words = set(words)

        # 构建邻接表
        for i in range(len(words) - 1):
            from_word = words[i]
            to_word = words[i + 1]

            if to_word in self.graph[from_word]:
                self.graph[from_word][to_word] += 1
            else:
                self.graph[from_word][to_word] = 1

        # 计算入度和出度
        for from_word in self.graph:
            for to_word in self.graph[from_word]:
                self.out_degree[from_word] += 1
                self.in_degree[to_word] += 1

    def save_graph_to_file(self, filename: str = "text_graph.png"):
        """
        使用networkx和matplotlib绘制有向图并保存为图片
        :param filename: 输出文件名（如 "text_graph.png"）
        """
        # 创建有向图
        G = nx.DiGraph()

        # 添加所有节点和边
        for from_word in self.graph:
            for to_word, weight in self.graph[from_word].items():
                G.add_edge(from_word, to_word, weight=weight)

        # 设置图形布局
        pos = nx.spring_layout(G, k=0.5, iterations=50)  # 调整k和iterations使布局更清晰

        # 绘制节点和边
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue", alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray", arrows=True, arrowsize=20)

        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

        # 添加边权重标签
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

        # 保存图形
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"有向图已保存为: {filename}")

    def display_graph(self):
        """生成并保存有向图图形文件"""
        print("\n正在生成有向图图形文件...")
        self.save_graph_to_file()
    def find_bridge_words(self, word1: str, word2: str) -> List[str]:
        """查找桥接词"""
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.words or word2 not in self.words:
            return None

        bridge_words = []
        # 查找word1的直接后继
        successors = set(self.graph[word1].keys())
        # 查找word2的直接前驱
        predecessors = set()
        for word in self.graph:
            if word2 in self.graph[word]:
                predecessors.add(word)

        # 桥接词是同时是word1的后继和word2的前驱
        bridge_words = list(successors & predecessors)

        return bridge_words

    def generate_new_text(self, text: str) -> str:
        """根据桥接词生成新文本"""
        processed = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        words = processed.split()

        if len(words) < 2:
            return text

        new_text = [words[0]]

        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            if word1 in self.words and word2 in self.words:
                bridge_words = self.find_bridge_words(word1, word2)
                if bridge_words:
                    chosen = random.choice(bridge_words)
                    new_text.append(chosen)

            new_text.append(word2)

        return ' '.join(new_text)

    def shortest_path(self, word1: str, word2: str) -> Tuple[List[str], int]:
        """计算两个单词之间的最短路径（Dijkstra算法）"""
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.words or word2 not in self.words:
            return None, -1

        if word1 == word2:
            return [word1], 0

        # 优先队列: (累计距离, 当前节点, 路径)
        heap = []
        heapq.heappush(heap, (0, word1, [word1]))

        visited = set()

        while heap:
            dist, current, path = heapq.heappop(heap)

            if current == word2:
                return path, dist

            if current in visited:
                continue

            visited.add(current)

            for neighbor, weight in self.graph[current].items():
                if neighbor not in visited:
                    new_dist = dist + weight
                    new_path = path + [neighbor]
                    heapq.heappush(heap, (new_dist, neighbor, new_path))

        return None, -1  # 不可达

    def pagerank(self, damping_factor=0.85, max_iter=100, tol=1e-6) -> Dict[str, float]:
        """计算PageRank值"""
        num_nodes = len(self.words)
        if num_nodes == 0:
            return {}

        # 初始化PR值
        pr = {word: 1.0 / num_nodes for word in self.words}

        for _ in range(max_iter):
            new_pr = {}
            leak = 0.0

            # 计算漏出的PR值（无出边的节点）
            for word in self.words:
                if self.out_degree[word] == 0:
                    leak += damping_factor * pr[word] / num_nodes

            # 计算每个节点的新PR值
            for word in self.words:
                # 来自其他节点的贡献
                incoming = 0.0
                for from_word in self.graph:
                    if word in self.graph[from_word]:
                        incoming += pr[from_word] / self.out_degree[from_word]

                # 随机跳转 + 来自其他节点的贡献
                new_pr[word] = (1 - damping_factor) / num_nodes + damping_factor * incoming + leak

            # 检查收敛
            diff = sum(abs(new_pr[word] - pr[word]) for word in self.words)
            pr = new_pr

            if diff < tol:
                break

        # 归一化PR值
        total = sum(pr.values())
        if total > 0:
            pr = {word: value / total for word, value in pr.items()}

        return pr

    def random_walk(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """随机游走，直到遇到重复边或没有出边"""
        if not self.words:
            return [], []

        current = random.choice(list(self.words))
        path = [current]
        edges_visited = set()
        edges_traversed = []

        while True:
            if current not in self.graph or not self.graph[current]:
                break  # 没有出边

            neighbors = list(self.graph[current].keys())
            weights = list(self.graph[current].values())
            # 按权重随机选择
            next_word = random.choices(neighbors, weights=weights, k=1)[0]

            edge = (current, next_word)
            if edge in edges_visited:
                edges_traversed.append(edge)
                break

            edges_visited.add(edge)
            edges_traversed.append(edge)
            path.append(next_word)
            current = next_word

        return path, edges_traversed


def read_file(filename: str) -> str:
    """读取文本文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def main():
    print("文本图处理程序")
    print("=" * 40)

    # 获取文件路径
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("请输入文本文件路径: ")

    # 读取文件
    text = read_file(filename)
    if text is None:
        return

    # 构建图
    graph = TextGraph()
    graph.build_graph(text)

    while True:
        print("\n菜单:")
        print("1. 生成并保存有向图图形文件")
        print("2. 查询桥接词")
        print("3. 根据桥接词生成新文本")
        print("4. 计算最短路径")
        print("5. 计算PageRank")
        print("6. 随机游走")
        print("0. 退出")

        choice = input("请选择功能 (0-6): ")

        if choice == "1":
            graph.display_graph()

        elif choice == "2":
            word1 = input("输入第一个单词: ")
            word2 = input("输入第二个单词: ")
            bridges = graph.find_bridge_words(word1, word2)

            if bridges is None:
                print(f"No {word1} or {word2} in the graph!")
            elif not bridges:
                print(f"No bridge words from {word1} to {word2}!")
            else:
                if len(bridges) == 1:
                    print(f"The bridge word from {word1} to {word2} is: {bridges[0]}")
                else:
                    print(f"The bridge words from {word1} to {word2} are: ", end="")
                    print(", ".join(bridges[:-1]), end="")
                    print(f" and {bridges[-1]}.")

        elif choice == "3":
            new_text = input("输入新文本: ")
            generated = graph.generate_new_text(new_text)
            print("\n生成的新文本:")
            print(generated)

        elif choice == "4":
            word1 = input("输入起始单词: ")
            word2 = input("输入目标单词: ")
            path, length = graph.shortest_path(word1, word2)

            if path is None:
                print(f"无法从 {word1} 到达 {word2}!")
            else:
                print(f"最短路径 (长度: {length}):")
                print(" -> ".join(path))

        elif choice == "5":
            pr = graph.pagerank()
            print("\nPageRank值 (前20个):")
            print("=" * 40)
            sorted_pr = sorted(pr.items(), key=lambda x: -x[1])
            for word, score in sorted_pr[:20]:
                print(f"{word}: {score:.6f}")
            print("=" * 40)

        elif choice == "6":
            print("开始随机游走... (按Enter键停止)")
            path, edges = graph.random_walk()

            print("\n游走路径:")
            print(" -> ".join(path))

            # 保存到文件
            filename = f"random_walk_{len(path)}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(" ".join(path))
            print(f"路径已保存到 {filename}")

        elif choice == "0":
            print("退出程序")
            break

        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()