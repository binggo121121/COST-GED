"""
匈牙利算法近似计算图编辑距离 (GED)

给定两个图 G1 和 G2，先用匈牙利算法求节点匹配，再根据该匹配统计编辑代价。
不依赖任何项目内模块，仅需 numpy + scipy。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_ged(adj1, labels1, adj2, labels2,
                  node_sub_cost=None, node_del_cost=1.0, node_ins_cost=1.0,
                  edge_sub_cost=None, edge_del_cost=1.0, edge_ins_cost=1.0):
    """
    用匈牙利算法近似计算 G1 到 G2 的图编辑距离。

    Parameters
    ----------
    adj1, adj2 : np.ndarray (n1, n1) / (n2, n2)
        邻接矩阵 (无向图假设，使用上三角)。值为 0/1 或边标签整数。
    labels1, labels2 : np.ndarray (n1,) / (n2,)
        节点标签 (整数)。
    node_sub_cost : function(label, label) -> float, optional
        节点替换代价，默认: 标签相同=0，不同=1。
    node_del_cost, node_ins_cost : float
        节点删除/插入代价，默认 1.0。
    edge_sub_cost : function(label, label) -> float, optional
        边替换代价，默认: 标签相同=0，不同=1。
    edge_del_cost, edge_ins_cost : float
        边删除/插入代价，默认 1.0。

    Returns
    -------
    total_cost : float
        近似 GED。
    matching : np.ndarray (n1,), dtype=int
        源节点匹配到的目标节点索引，-1 表示删除。
    details : dict
        {node_sub, node_del, node_ins, edge_sub, edge_del, edge_ins}
    """
    n1, n2 = len(labels1), len(labels2)

    if node_sub_cost is None:
        def node_sub_cost(a, b): return 0.0 if a == b else 1.0
    if edge_sub_cost is None:
        def edge_sub_cost(a, b): return 0.0 if a == b else 1.0

    # ——— 1. 匈牙利匹配 (基于节点标签) ———
    N = max(n1, n2)
    cost_matrix = np.full((N, N), node_del_cost + node_ins_cost)  # 默认: 删除+插入
    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = node_sub_cost(labels1[i], labels2[j])
    # 虚节点 ↔ 虚节点: 代价为 0
    for i in range(n1, N):
        for j in range(n2, N):
            cost_matrix[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matching = np.full(n1, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            matching[r] = c

    # ——— 2. 统计节点编辑代价 ———
    matched_mask = matching >= 0
    node_sub = 0
    for i in np.where(matched_mask)[0]:
        node_sub += node_sub_cost(labels1[i], labels2[matching[i]])
    node_del = np.sum(~matched_mask) * node_del_cost
    node_ins = (n2 - np.sum(matched_mask)) * node_ins_cost

    # ——— 3. 统计边编辑代价 ———
    # 取出上三角 (无向图)
    tri1 = np.triu_indices(n1, k=1)
    tri2 = np.triu_indices(n2, k=1)

    # 构建 g2 边索引字典，方便查找
    edge2_dict = {}
    for idx in range(len(tri2[0])):
        u, v = tri2[0][idx], tri2[1][idx]
        if adj2[u, v] != 0:  # 有边
            edge2_dict[(u, v)] = adj2[u, v]

    edge_sub, edge_del, edge_ins = 0, 0, 0
    matched_edges = set()

    for idx in range(len(tri1[0])):
        u, v = tri1[0][idx], tri1[1][idx]
        if adj1[u, v] == 0:
            continue  # g1 中没有边
        label1 = adj1[u, v]

        # 映射到 g2
        mu, mv = matching[u], matching[v]
        if mu >= 0 and mv >= 0:
            key = (min(mu, mv), max(mu, mv))
            if key in edge2_dict:
                # 边替换
                label2 = edge2_dict[key]
                edge_sub += edge_sub_cost(label1, label2)
                matched_edges.add(key)
            else:
                # g1 有边，g2 对应位置无边 → 边删除
                edge_del += edge_del_cost
        else:
            # 源节点未匹配 → 边删除
            edge_del += edge_del_cost

    # g2 中未被匹配到的边 → 边插入
    for key, label2 in edge2_dict.items():
        if key not in matched_edges:
            edge_ins += edge_ins_cost

    # ——— 4. 汇总 ———
    total = node_sub + node_del + node_ins + edge_sub + edge_del + edge_ins
    details = {
        'node_sub': node_sub, 'node_del': node_del, 'node_ins': node_ins,
        'edge_sub': edge_sub, 'edge_del': edge_del, 'edge_ins': edge_ins,
    }
    return total, matching, details


# ============================================================
# 示例 & OGB-molhiv 加载器
# ============================================================
if __name__ == "__main__":
    import gzip, csv, os, pickle, random

    # ——— 简单的示例 ———
    print("=" * 50)
    print("示例: 两个小图")
    adj1 = np.array([[0, 1, 1],
                     [1, 0, 0],
                     [1, 0, 0]])
    labels1 = np.array([0, 1, 2])
    adj2 = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]])
    labels2 = np.array([0, 1, 1])

    cost, matching, details = hungarian_ged(adj1, labels1, adj2, labels2)
    print(f"  匹配: {matching}")
    print(f"  总代价: {cost}")
    print(f"  明细: {details}")

    # ——— OGB-molhiv 大图示例 ———
    print("\n" + "=" * 50)
    print("OGB molhiv 30-50 数据集")

    CACHE = 'data/ogb/cache/molhiv_30_50.pkl'
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            graphs = pickle.load(f)

        random.shuffle(graphs)
        split = int(len(graphs) * 0.8)
        test_pool = graphs[split:]

        # 构造 500 对测试
        pairs = []
        while len(pairs) < 500:
            i = random.randint(0, len(test_pool) - 1)
            j = random.randint(0, len(test_pool) - 1)
            if i != j:
                g1, g2 = test_pool[i], test_pool[j]
                if g1['num_nodes'] > g2['num_nodes']:
                    g1, g2 = g2, g1
                pairs.append((g1, g2))

        def graph_to_adj_labels(g):
            """OGB 图 dict → 邻接矩阵 + 标签数组"""
            n = g['num_nodes']
            labels = g['node_labels'].numpy()
            ei = g['edge_index'].numpy()
            ea = g['edge_attr'].numpy().flatten() if hasattr(g['edge_attr'], 'numpy') else np.array(g['edge_attr']).flatten()

            adj = np.zeros((n, n), dtype=int)
            for k in range(ei.shape[1]):
                u, v = ei[0, k], ei[1, k]
                lbl = ea[k] if k < len(ea) else 1
                if adj[u, v] == 0:
                    adj[u, v] = lbl + 1  # +1 避免 0 表示无边
                    adj[v, u] = lbl + 1
            return adj, labels

        costs = []
        from tqdm import tqdm
        for g1, g2 in tqdm(pairs, desc="  计算中"):
            a1, l1 = graph_to_adj_labels(g1)
            a2, l2 = graph_to_adj_labels(g2)
            c, _, _ = hungarian_ged(a1, l1, a2, l2)
            costs.append(c)

        print(f"  {len(pairs)} 对图, 平均代价: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
    else:
        print(f"  OGB cache 不存在: {CACHE}")
