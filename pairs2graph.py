import os
import networkx as nx


def read_pairs(file_path):
    """从给定的文件中读取图像匹配对"""
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            # 分隔每一行的两个图像名称
            img1, img2 = line.strip().split()
            pairs.append((img1, img2))
    return pairs


def build_graph(pairs):
    """使用匹配对构建无向图"""
    G = nx.Graph()
    G.add_edges_from(pairs)
    return G


def calculate_weights(G):
    """计算每个节点的权重"""
    total_edges = G.number_of_edges()
    weights = {}
    for node in G.nodes():
        degree = G.degree(node)
        # 计算权重
        weight = degree / total_edges if total_edges > 0 else 0
        weights[node] = weight
    return weights


def write_weights_to_file(weights, output_file):
    """将权重写入到文件中"""
    with open(output_file, 'w') as f:
        for img, weight in weights.items():
            img_name = os.path.splitext(img)[0]  # 去掉文件后缀
            f.write(f"{img_name} {weight}\n")


def main(input_file):
    # 读取图像匹配对
    pairs = read_pairs(input_file)

    # 构建图
    G = build_graph(pairs)

    # 计算节点权重
    weights = calculate_weights(G)

    # 生成输出文件名
    output_file = os.path.splitext(input_file)[0] + "_graph_weight.txt"

    # 输出权重到文件
    write_weights_to_file(weights, output_file)
    print(f"Weights written to: {output_file}")


def generate_weights(dir_path):
    pair_file_name = dir_path.split('/')[-1]
    pair_file_path = os.path.join(dir_path, f'pairs_{pair_file_name}.txt')
    main(pair_file_path)


if __name__ == "__main__":
    dir_path = "datasets/waymo_data/individual_files_training_segment-10061305430875486848_1080_000_1100_000_with_camera_labels/Q_concentric_5_20_1_False"  # 请在此替换匹配dir
    generate_weights(dir_path)
