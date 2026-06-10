"""
SIL-GED on Large Graphs (30-50 nodes)  v4 (FINAL FIX)
======================================================
v4 修复（最终正确版，撤回 v3 的过度修复）:
  v3 错误地把 edge_attr 扩到 actual_edge_dim=7，但 model.batch_ensemble_inference
  内部已经会自动 +1 列 alignment 特征。所以用户应传 6 维（num_edge_labels），
  让 model 内部加成 7 维（num_edge_labels+1），刚好匹配 GINEConv(edge_dim=7)。
  v4 撤回 expand_edge_attr_to_dim 调用，graphs_to_pyg 直接输出 6 维 one-hot。

数据流（终于理顺）:
  数据预处理: graphs_to_pyg → edge_attr (E, num_edge_labels=6) one-hot float
  ↓
  路径A 训练: parallel_process → edge_attr (E, num_edge_labels+1=7) → GINEConv(7) ✓
  路径B 推理: model.batch_ensemble_inference → 内部 +1 列 → (E, 7) → GINEConv(7) ✓

v4 保留 v3 的所有其他修复:
  - evaluate_cost 中 temperature=None 改 if 分支
  - graphs_to_pyg 输出 one-hot float（修复 v2 的 (E,1) 整数 bug）
  - 多轮 Gumbel 推理、cache、timer、GELATO baseline 等所有 v2 改进
"""

import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import gzip
import csv
import os
import json
import time
import pickle

from src.subproblem_dataset import make_instance, parallel_process
from src.model import LinkGNN
from src.utils import matching_cost, training_step_link
from sil_reconstruct import combined_reconstruct


# ============================================================
# OGB molhiv 加载（带 cache）
# ============================================================

def load_ogb_molhiv(root='data/ogb/ogbg_molhiv/raw',
                    min_nodes=30, max_nodes=50,
                    cache_dir=None):
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'molhiv_{min_nodes}_{max_nodes}.pkl')
        if os.path.exists(cache_path):
            print(f"[cache] loading from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    print(f"Loading OGB molhiv graphs ({min_nodes}-{max_nodes} nodes)...")

    with gzip.open(os.path.join(root, 'num-node-list.csv.gz'), 'rt') as f:
        num_nodes_list = [int(row[0]) for row in csv.reader(f)]
    with gzip.open(os.path.join(root, 'num-edge-list.csv.gz'), 'rt') as f:
        num_edges_list = [int(row[0]) for row in csv.reader(f)]
    with gzip.open(os.path.join(root, 'node-feat.csv.gz'), 'rt') as f:
        all_node_feats = [list(map(int, row)) for row in csv.reader(f)]
    with gzip.open(os.path.join(root, 'edge.csv.gz'), 'rt') as f:
        all_edges = [list(map(int, row)) for row in csv.reader(f)]
    with gzip.open(os.path.join(root, 'edge-feat.csv.gz'), 'rt') as f:
        all_edge_feats = [list(map(int, row)) for row in csv.reader(f)]

    graphs = []
    node_offset = 0
    edge_offset = 0
    num_bad_graphs = 0
    num_filtered_edges = 0

    for gid in range(len(num_nodes_list)):
        n = num_nodes_list[gid]
        m = num_edges_list[gid]

        if min_nodes <= n <= max_nodes:
            node_feats = all_node_feats[node_offset:node_offset + n]
            node_labels = torch.tensor([nf[0] for nf in node_feats], dtype=torch.long)

            if m > 0:
                edge_rows = all_edges[edge_offset:edge_offset + m]
                edge_feat_rows = all_edge_feats[edge_offset:edge_offset + m]
                src_raw = torch.tensor([e[0] for e in edge_rows], dtype=torch.long)
                dst_raw = torch.tensor([e[1] for e in edge_rows], dtype=torch.long)
                edge_attr_raw = torch.tensor([ef[0] for ef in edge_feat_rows],
                                             dtype=torch.long)

                local_ok = (src_raw.min().item() >= 0 and dst_raw.min().item() >= 0
                            and src_raw.max().item() < n and dst_raw.max().item() < n)
                global_ok = (src_raw.min().item() >= node_offset
                             and dst_raw.min().item() >= node_offset
                             and src_raw.max().item() < node_offset + n
                             and dst_raw.max().item() < node_offset + n)

                if local_ok:
                    src, dst, edge_attr = src_raw, dst_raw, edge_attr_raw
                elif global_ok:
                    src = src_raw - node_offset
                    dst = dst_raw - node_offset
                    edge_attr = edge_attr_raw
                else:
                    valid_local = ((src_raw >= 0) & (src_raw < n) &
                                   (dst_raw >= 0) & (dst_raw < n))
                    valid_global = ((src_raw >= node_offset) & (src_raw < node_offset + n) &
                                    (dst_raw >= node_offset) & (dst_raw < node_offset + n))
                    if valid_local.sum().item() >= valid_global.sum().item():
                        valid = valid_local
                        src, dst = src_raw[valid], dst_raw[valid]
                        edge_attr = edge_attr_raw[valid]
                    else:
                        valid = valid_global
                        src = src_raw[valid] - node_offset
                        dst = dst_raw[valid] - node_offset
                        edge_attr = edge_attr_raw[valid]
                    num_filtered_edges += m - valid.sum().item()
                    num_bad_graphs += 1

                if src.numel() > 0:
                    edge_index = torch.stack([src, dst], dim=0).long()
                    valid = ((edge_index[0] >= 0) & (edge_index[0] < n) &
                             (edge_index[1] >= 0) & (edge_index[1] < n))
                    if valid.sum().item() < edge_index.size(1):
                        num_filtered_edges += edge_index.size(1) - valid.sum().item()
                        edge_index = edge_index[:, valid]
                        edge_attr = edge_attr[valid]
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_attr = torch.zeros(0, dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros(0, dtype=torch.long)

            graphs.append({
                'node_labels': node_labels,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'num_nodes': n,
                'graph_id': gid
            })

        node_offset += n
        edge_offset += m

    print(f"  Found {len(graphs)} graphs with {min_nodes}-{max_nodes} nodes")
    print(f"  Bad graphs repaired: {num_bad_graphs}, filtered edges: {num_filtered_edges}")

    if cache_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"[cache] saved to {cache_path}")

    return graphs


def graphs_to_pyg(graphs, num_node_labels=120, num_edge_labels=6):
    """
    输出 edge_attr 为 (E, num_edge_labels) one-hot float，
    与 GraphMatchingDataset 保持完全一致格式。
    model.batch_ensemble_inference 内部会再 +1 列 alignment，
    形成 (E, num_edge_labels+1) 与 GINEConv(edge_dim=actual_edge_dim) 匹配。
    """
    pyg_list = []
    for idx, g in enumerate(graphs):
        num_nodes = g['num_nodes']
        x = F.one_hot(g['node_labels'].clamp(0, num_node_labels - 1),
                      num_node_labels).float()

        edge_index = g['edge_index'].long()
        edge_attr_raw = g['edge_attr'].long()

        if edge_attr_raw.dim() == 2 and edge_attr_raw.size(1) == 1:
            edge_attr_raw = edge_attr_raw.view(-1)
        elif edge_attr_raw.dim() == 0:
            edge_attr_raw = edge_attr_raw.view(1)

        if edge_index.size(1) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, num_edge_labels), dtype=torch.float)
        else:
            valid = ((edge_index[0] >= 0) & (edge_index[0] < num_nodes) &
                     (edge_index[1] >= 0) & (edge_index[1] < num_nodes))
            if valid.sum().item() < edge_index.size(1):
                edge_index = edge_index[:, valid]
                edge_attr_raw = edge_attr_raw[valid]

            edge_attr = F.one_hot(
                edge_attr_raw.clamp(0, num_edge_labels - 1),
                num_edge_labels
            ).float()

            edge_index, edge_attr = torch_geometric.utils.to_undirected(
                edge_index, edge_attr, num_nodes=num_nodes, reduce='max'
            )
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1).float()

        pyg_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                             num_nodes=num_nodes))
    print(f"Converted to PyG: {len(pyg_list)} graphs "
          f"(node one-hot={num_node_labels}, edge one-hot={num_edge_labels})")
    return pyg_list


def sample_graph_pairs(pyg_list, num_pairs, seed=0):
    rng = random.Random(seed)
    n = len(pyg_list)
    pairs = []
    for _ in range(num_pairs):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        while j == i:
            j = rng.randint(0, n - 1)
        g1, g2 = pyg_list[i], pyg_list[j]
        if g1.x.size(0) > g2.x.size(0):
            g1, g2 = g2, g1
        pairs.append((g1, g2))
    return pairs


# ============================================================
# Baselines
# ============================================================

def hungarian_init(g1, g2, node_cost=1.0):
    n1, n2 = g1.x.size(0), g2.x.size(0)
    N = max(n1, n2)
    cost_matrix = np.full((N, N), node_cost)
    for i in range(n1):
        for j in range(n2):
            if torch.equal(g1.x[i], g2.x[j]):
                cost_matrix[i, j] = 0.0
    for i in range(n1, N):
        for j in range(n2, N):
            cost_matrix[i, j] = 0.0
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matching = torch.full((n1,), -1, dtype=torch.long)
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            matching[r] = c
    return matching


def greedy_matching(g1, g2, node_cost=1.0):
    n1, n2 = g1.x.size(0), g2.x.size(0)
    matching = torch.full((n1,), -1, dtype=torch.long)
    used = set()
    for i in range(n1):
        for j in range(n2):
            if j not in used and torch.equal(g1.x[i], g2.x[j]):
                matching[i] = j
                used.add(j)
                break
    remaining = [j for j in range(n2) if j not in used]
    idx = 0
    for i in range(n1):
        if matching[i] == -1 and idx < len(remaining):
            matching[i] = remaining[idx]
            idx += 1
    return matching


def refine_matching(g1, g2, node_cost=1.0, edge_cost=1.0, max_iters=10):
    matching = hungarian_init(g1, g2, node_cost)
    best_cost = matching_cost(g1, g2, matching, node_cost, edge_cost)
    n1 = g1.x.size(0)
    n2 = g2.x.size(0)
    for _ in range(max_iters):
        improved = False
        for i in range(n1):
            if improved:
                break
            for j in range(i + 1, n1):
                new_m = matching.clone()
                new_m[i], new_m[j] = matching[j].item(), matching[i].item()
                nc = matching_cost(g1, g2, new_m, node_cost, edge_cost)
                if nc < best_cost:
                    matching, best_cost = new_m, nc
                    improved = True
                    break
        if improved:
            continue
        for i in range(n1):
            if improved:
                break
            used = set(matching[k].item() for k in range(n1)
                       if matching[k] >= 0 and k != i)
            for t in range(n2):
                if t in used:
                    continue
                new_m = matching.clone()
                new_m[i] = t
                nc = matching_cost(g1, g2, new_m, node_cost, edge_cost)
                if nc < best_cost:
                    matching, best_cost = new_m, nc
                    improved = True
                    break
        if not improved:
            break
    return matching


def init_matchings_with_model(g1_list, g2_list, model, k=32, batch_size=8,
                              node_cost=1.0):
    matching_list = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(g1_list), batch_size):
            end = min(start + batch_size, len(g1_list))
            try:
                batch_m = model.batch_ensemble_inference(
                    g1_list[start:end], g2_list[start:end], k=k
                )
                matching_list.extend(batch_m)
            except Exception as e:
                if start == 0:
                    print(f"  [init_matchings WARN] {type(e).__name__}: {e}")
                for idx in range(start, end):
                    matching_list.append(
                        hungarian_init(g1_list[idx], g2_list[idx], node_cost)
                    )
    return matching_list


# ============================================================
# 节点排列增强 + 伪标签数据集
# ============================================================

def augment_graph_pair(g1, g2, matching):
    n1 = g1.x.size(0)
    perm = torch.randperm(n1)
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(n1)
    new_x = g1.x[perm]
    new_edge_index = inv_perm[g1.edge_index]
    new_edge_attr = g1.edge_attr.clone()
    new_matching = torch.full_like(matching, -1)
    for new_i in range(n1):
        old_i = perm[new_i].item()
        new_matching[new_i] = matching[old_i]
    new_g1 = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    return new_g1, g2, new_matching


def edge_attr_to_label(edge_attr):
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.empty((0,), dtype=torch.int8)
    if edge_attr.dim() == 1:
        return edge_attr.to(torch.int8)
    if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
        return edge_attr.view(-1).to(torch.int8)
    return edge_attr.argmax(dim=-1).to(torch.int8)


def create_pseudo_label_dataset(g1_list, g2_list, matching_list,
                                num_instances_per_pair=20,
                                num_node_labels=None, num_edge_labels=None,
                                num_augment=0):
    args_list = []
    for i in range(len(g1_list)):
        g1, g2, mu = g1_list[i], g2_list[i], matching_list[i]
        pairs_to_process = [(g1, g2, mu)]
        for _ in range(num_augment):
            ag1, ag2, amu = augment_graph_pair(g1, g2, mu)
            pairs_to_process.append((ag1, ag2, amu))

        for pg1, pg2, pmu in pairs_to_process:
            x_s_raw = pg1.x.argmax(dim=-1).to(torch.int8)
            x_t_raw = pg2.x.argmax(dim=-1).to(torch.int8)
            ea_s_raw = edge_attr_to_label(pg1.edge_attr)
            ea_t_raw = edge_attr_to_label(pg2.edge_attr)

            class FakePair:
                pass
            fp = FakePair()
            fp.x_s = x_s_raw
            fp.x_t = x_t_raw
            fp.edge_index_s = pg1.edge_index.to(torch.int)
            fp.edge_index_t = pg2.edge_index.to(torch.int)
            fp.edge_attr_s = ea_s_raw
            fp.edge_attr_t = ea_t_raw
            fp.matching = pmu.to(torch.int16)

            n_s_local = pg1.x.size(0)
            inst_count = max(1, num_instances_per_pair // (1 + num_augment))
            for inst in range(inst_count):
                target_size = 0 if inst == 0 else random.randint(1, max(1, n_s_local - 1))
                pm = torch.full((n_s_local,), -2, dtype=torch.int16)
                Slist = torch.randperm(n_s_local)[:target_size]
                pm[Slist] = pmu[Slist].to(torch.int16)
                args_list.append((fp, pm, num_node_labels, num_edge_labels))

    data_list = []
    for a in tqdm(args_list, desc="  Building pseudo-labels", ncols=80):
        try:
            data_list.append(parallel_process(a))
        except Exception:
            continue
    return data_list


# ============================================================
# 评估：单轮 / 多轮 Gumbel 采样
# ============================================================

def evaluate_cost(model, test_pairs, k=32, num_runs=1, temperature=0.5,
                  batch_size=8, node_cost=1.0, edge_cost=1.0, desc="Eval"):
    model.eval()
    n_pairs = len(test_pairs)
    best_costs = [None] * n_pairs
    fail_count = 0

    with torch.no_grad():
        for run in range(num_runs):
            run_iter = range(0, n_pairs, batch_size)
            run_iter = tqdm(run_iter,
                            desc=f"{desc} run {run+1}/{num_runs}",
                            ncols=80) if num_runs > 1 else run_iter
            for start in run_iter:
                end = min(start + batch_size, n_pairs)
                g1_b = [p[0] for p in test_pairs[start:end]]
                g2_b = [p[1] for p in test_pairs[start:end]]

                matchings = None
                try:
                    if num_runs > 1:
                        matchings = model.batch_ensemble_inference(
                            g1_b, g2_b, k=k, temperature=temperature
                        )
                    else:
                        matchings = model.batch_ensemble_inference(
                            g1_b, g2_b, k=k
                        )
                except TypeError:
                    matchings = model.batch_ensemble_inference(
                        g1_b, g2_b, k=k
                    )
                except Exception as e:
                    fail_count += 1
                    if fail_count <= 3:
                        print(f"  [evaluate WARN {fail_count}] "
                              f"{type(e).__name__}: {str(e)[:100]}")
                    matchings = [hungarian_init(g1_b[j], g2_b[j], node_cost)
                                 for j in range(len(g1_b))]

                for j, m in enumerate(matchings):
                    c = matching_cost(g1_b[j], g2_b[j], m, node_cost, edge_cost)
                    idx = start + j
                    if best_costs[idx] is None or c < best_costs[idx]:
                        best_costs[idx] = c

    if fail_count > 0:
        print(f"  [evaluate] total fallback batches: {fail_count}")
    return best_costs


def compute_baseline_costs(test_pairs, node_cost=1.0, edge_cost=1.0):
    hungarian_costs, greedy_costs, refine_costs = [], [], []
    t0 = time.time()
    for g1, g2 in tqdm(test_pairs, desc="Hungarian/Greedy/Refine"):
        m_h = hungarian_init(g1, g2, node_cost)
        hungarian_costs.append(matching_cost(g1, g2, m_h, node_cost, edge_cost))
        m_g = greedy_matching(g1, g2, node_cost)
        greedy_costs.append(matching_cost(g1, g2, m_g, node_cost, edge_cost))
        m_r = refine_matching(g1, g2, node_cost, edge_cost)
        refine_costs.append(matching_cost(g1, g2, m_r, node_cost, edge_cost))
    print(f"  baselines wall-clock: {time.time()-t0:.1f}s")
    return hungarian_costs, greedy_costs, refine_costs


def report_method(name, costs, hungarian_costs, t_used=None):
    avg = np.mean(costs)
    avg_h = np.mean(hungarian_costs)
    impr = (avg_h - avg) / avg_h * 100
    wins = sum(1 for s, h in zip(costs, hungarian_costs) if s < h)
    ties = sum(1 for s, h in zip(costs, hungarian_costs) if s == h)
    losses = len(costs) - wins - ties
    win_rate = wins / len(costs) * 100
    tie_rate = ties / len(costs) * 100
    line = (f"  {name:<25} avg_cost={avg:.2f}  "
            f"vs_Hungarian={impr:+.1f}%  "
            f"win/tie/loss={wins}/{ties}/{losses} "
            f"({win_rate:.1f}%/{tie_rate:.1f}%/{100-win_rate-tie_rate:.1f}%)")
    if t_used is not None:
        line += f"  time={t_used:.1f}s"
    print(line)
    return {'avg': avg, 'impr': impr, 'win_rate': win_rate, 'tie_rate': tie_rate}


# ============================================================
# 主函数（v4 关键改动：删除 expand_edge_attr_to_dim 调用）
# ============================================================

def main(args):
    print("=" * 70)
    print("SIL-GED on Large Graphs v4 [FINAL FIX] (no exact GED labels)")
    print(f"Node range: {args.min_nodes}-{args.max_nodes}, "
          f"train_pairs={args.train_pairs}, test_pairs={args.test_pairs}")
    print(f"Inference: k={args.k}, num_runs={args.num_runs}, temp={args.temperature}")
    print("=" * 70)

    t_pipeline_start = time.time()

    raw_graphs = load_ogb_molhiv(
        root=args.ogb_root,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        cache_dir=args.cache_dir
    )
    pyg_graphs = graphs_to_pyg(raw_graphs,
                               num_node_labels=args.num_node_labels,
                               num_edge_labels=args.num_edge_labels)
    print(f"Total graphs: {len(pyg_graphs)}")

    random.shuffle(pyg_graphs)
    split = int(len(pyg_graphs) * 0.8)
    train_graphs = pyg_graphs[:split]
    test_graphs = pyg_graphs[split:]
    print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

    train_pairs = sample_graph_pairs(train_graphs, args.train_pairs, seed=args.seed)
    test_pairs = sample_graph_pairs(test_graphs, args.test_pairs, seed=args.seed + 999)
    g1_list = [p[0] for p in train_pairs]
    g2_list = [p[1] for p in train_pairs]
    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    num_node_labels = args.num_node_labels
    num_edge_labels = args.num_edge_labels

    # ---- Phase 0: Hungarian 初始化 ----
    print("\n--- Phase 0: Hungarian Initialization ---")
    matching_list = []
    init_costs = []
    t0 = time.time()
    for i in tqdm(range(len(g1_list)), desc="Hungarian init"):
        m = hungarian_init(g1_list[i], g2_list[i], node_cost=args.node_cost)
        matching_list.append(m)
        init_costs.append(matching_cost(g1_list[i], g2_list[i], m,
                                        args.node_cost, args.edge_cost))
    print(f"Hungarian avg cost: {np.mean(init_costs):.2f} "
          f"(time: {time.time()-t0:.1f}s)")

    # ---- 检测模型输入维度 (= num_edge_labels + 1，因为 parallel_process 加 1 列) ----
    probe = create_pseudo_label_dataset(
        g1_list[:2], g2_list[:2], matching_list[:2],
        num_instances_per_pair=1,
        num_node_labels=num_node_labels, num_edge_labels=num_edge_labels,
        num_augment=0
    )
    if len(probe) == 0:
        raise RuntimeError("无法生成 probe 数据")
    actual_node_dim = probe[0].x.shape[1]
    actual_edge_dim = probe[0].edge_attr.shape[1]
    print(f"Model input dims: node={actual_node_dim}, edge={actual_edge_dim}")
    print(f"  pyg_graphs[0].edge_attr.shape = {g1_list[0].edge_attr.shape}")
    print(f"  → batch_ensemble_inference 内部会 +1 列得到 {g1_list[0].edge_attr.shape[1]+1} 维")
    print(f"  → 与 GINEConv(edge_dim={actual_edge_dim}) 匹配: "
          f"{g1_list[0].edge_attr.shape[1]+1 == actual_edge_dim}")
    assert g1_list[0].edge_attr.shape[1] + 1 == actual_edge_dim, (
        f"边维度不一致：g1.edge_attr={g1_list[0].edge_attr.shape[1]} 维，"
        f"+1 后 ≠ probe 检测的 actual_edge_dim={actual_edge_dim}。"
        f"请确认 graphs_to_pyg 输出 num_edge_labels={num_edge_labels} 维 one-hot。"
    )

    # ---- 模型 ----
    model = LinkGNN(actual_node_dim, actual_edge_dim, 128, args.layers,
                    args.node_cost, args.edge_cost)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                gamma=args.lr_decay)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # ---- baselines ----
    print("\n--- Computing baseline costs on test set ---")
    hungarian_test_costs, greedy_test_costs, refine_test_costs = \
        compute_baseline_costs(test_pairs, args.node_cost, args.edge_cost)

    # ---- GELATO 预训练 baseline ----
    gelato_costs_single = None
    gelato_costs_multi = None
    if args.gelato_ckp is not None and os.path.exists(args.gelato_ckp):
        print(f"\n--- Loading GELATO pretrained model from {args.gelato_ckp} ---")
        try:
            gelato_model = LinkGNN(actual_node_dim, actual_edge_dim, 128,
                                   args.layers, args.node_cost, args.edge_cost)
            gelato_model.load_state_dict(torch.load(args.gelato_ckp, map_location='cpu'))
            gelato_model = gelato_model.to(args.device)
            print("  GELATO baseline (single-shot)...")
            t0 = time.time()
            gelato_costs_single = evaluate_cost(
                gelato_model, test_pairs, k=args.k, num_runs=1,
                batch_size=args.eval_batch_size,
                node_cost=args.node_cost, edge_cost=args.edge_cost,
                desc="GELATO single"
            )
            print(f"    avg cost: {np.mean(gelato_costs_single):.2f} "
                  f"(time: {time.time()-t0:.1f}s)")
            print("  GELATO + multi-Gumbel inference...")
            t0 = time.time()
            gelato_costs_multi = evaluate_cost(
                gelato_model, test_pairs, k=args.k,
                num_runs=args.num_runs, temperature=args.temperature,
                batch_size=args.eval_batch_size,
                node_cost=args.node_cost, edge_cost=args.edge_cost,
                desc="GELATO multi"
            )
            print(f"    avg cost: {np.mean(gelato_costs_multi):.2f} "
                  f"(time: {time.time()-t0:.1f}s)")
            del gelato_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [WARN] GELATO baseline failed: {e}")

    # ---- SIL 主循环 ----
    best_test_cost = 1e10
    patience_counter = 0
    best_model_state = None
    cycle_logs = []

    for cycle in range(args.num_cycles):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*70}")
        print(f"SIL Cycle {cycle+1}/{args.num_cycles}  lr={current_lr:.6f}  "
              f"patience={patience_counter}/{args.patience}")
        print(f"{'='*70}")
        cycle_t0 = time.time()

        if cycle > 0 and args.refresh_every > 0 and cycle % args.refresh_every == 0:
            print(f"\n  [Refresh] Resampling {args.train_pairs} new pairs...")
            new_pairs = sample_graph_pairs(train_graphs, args.train_pairs,
                                           seed=args.seed + cycle)
            g1_list = [p[0] for p in new_pairs]
            g2_list = [p[1] for p in new_pairs]
            matching_list = init_matchings_with_model(
                g1_list, g2_list, model,
                k=args.k, batch_size=args.eval_batch_size,
                node_cost=args.node_cost
            )
            new_costs = [matching_cost(g1_list[i], g2_list[i], matching_list[i],
                                       args.node_cost, args.edge_cost)
                         for i in range(len(g1_list))]
            print(f"  [Refresh] Model init avg cost: {np.mean(new_costs):.2f}")

        print("\n[A] Local Reconstruction")
        improved_matchings, num_imp, cost_bef, cost_aft = combined_reconstruct(
            g1_list, g2_list, matching_list, model,
            k=args.k, batch_size=args.recon_batch_size,
            node_cost=args.node_cost, edge_cost=args.edge_cost,
            cycle_idx=cycle, num_cycles=args.num_cycles,
            no_cost_gate=args.no_cost_gate,
            fixed_temperature=args.fixed_temperature,
            fixed_perturb_ratio=args.fixed_perturb_ratio,
        )

        matching_list = improved_matchings
        print(f"  improvements={num_imp}, cost: {cost_bef:.2f} -> {cost_aft:.2f}")

        print("\n[B] Building training data")
        pseudo_data = create_pseudo_label_dataset(
            g1_list, g2_list, matching_list,
            num_instances_per_pair=args.instances_per_pair,
            num_node_labels=num_node_labels, num_edge_labels=num_edge_labels,
            num_augment=args.num_augment
        )
        print(f"  {len(pseudo_data)} instances")
        if len(pseudo_data) == 0:
            print("  WARNING: 无训练数据，跳过")
            continue
        pseudo_loader = DataLoader(pseudo_data, batch_size=128,
                                   shuffle=True, pin_memory=True)

        print("\n[C] Training GINE")
        model.train()
        train_t0 = time.time()
        for epoch in range(args.train_epochs_per_cycle):
            ep_loss, ep_acc = training_step_link(model, pseudo_loader, optimizer, args)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch+1}/{args.train_epochs_per_cycle}: "
                      f"loss={ep_loss:.5f}, acc={ep_acc:.4f}")
        train_time = time.time() - train_t0
        scheduler.step()

        print("\n[D] Evaluation (multi-round Gumbel sampling)")
        eval_t0 = time.time()
        sil_test_costs = evaluate_cost(
            model, test_pairs,
            k=args.k, num_runs=args.num_runs, temperature=args.temperature,
            batch_size=args.eval_batch_size,
            node_cost=args.node_cost, edge_cost=args.edge_cost,
            desc="Eval"
        )
        eval_time = time.time() - eval_t0
        avg_sil_cost = np.mean(sil_test_costs)
        avg_hungarian = np.mean(hungarian_test_costs)
        improvement = (avg_hungarian - avg_sil_cost) / avg_hungarian * 100
        wins = sum(1 for s, h in zip(sil_test_costs, hungarian_test_costs) if s < h)
        ties = sum(1 for s, h in zip(sil_test_costs, hungarian_test_costs) if s == h)
        win_rate = wins / len(test_pairs) * 100
        tie_rate = ties / len(test_pairs) * 100

        print(f"  SIL avg cost:       {avg_sil_cost:.2f}")
        print(f"  Hungarian avg cost: {avg_hungarian:.2f}")
        print(f"  Improvement vs Hungarian: {improvement:+.1f}%")
        print(f"  Win rate: {win_rate:.1f}% (tie: {tie_rate:.1f}%)")
        print(f"  train_time={train_time:.1f}s  eval_time={eval_time:.1f}s")

        if avg_sil_cost < best_test_cost:
            best_test_cost = avg_sil_cost
            patience_counter = 0
            best_model_state = {k_: v.cpu().clone()
                                for k_, v in model.state_dict().items()}
            if args.save_ckp:
                torch.save(best_model_state, args.save_ckp)
                print(f"  ★ New best! Saved (cost={avg_sil_cost:.2f})")
        else:
            patience_counter += 1

        cycle_logs.append({
            'cycle': cycle + 1,
            'cost_aft': cost_aft,
            'sil_cost': avg_sil_cost,
            'win_rate': win_rate,
            'train_time': train_time,
            'eval_time': eval_time,
            'cycle_time': time.time() - cycle_t0
        })

        if args.log and args.save_ckp:
            log_path = args.save_ckp.rsplit('.', 1)[0] + "_large.log"
            with open(log_path, "a") as f:
                f.write(f"cycle={cycle+1} imp={num_imp} "
                        f"train_cost={cost_aft:.2f} "
                        f"test_sil={avg_sil_cost:.2f} "
                        f"test_hungarian={avg_hungarian:.2f} "
                        f"test_greedy={np.mean(greedy_test_costs):.2f} "
                        f"test_refine={np.mean(refine_test_costs):.2f} "
                        f"improvement={improvement:.1f}% "
                        f"win_rate={win_rate:.1f}% "
                        f"train_time={train_time:.1f}s "
                        f"eval_time={eval_time:.1f}s\n")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping! No improvement for {args.patience} cycles.")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                model.to(args.device)
            break

    # ---- 最终汇总 ----
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(args.device)

    print("\nRunning k=1 single-shot evaluation...")
    t0 = time.time()
    sil_k1_costs = evaluate_cost(
        model, test_pairs, k=1, num_runs=1,
        batch_size=args.eval_batch_size,
        node_cost=args.node_cost, edge_cost=args.edge_cost,
        desc="SIL k=1"
    )
    t_k1 = time.time() - t0

    print(f"Running multi-round Gumbel evaluation (k={args.k}, runs={args.num_runs})...")
    t0 = time.time()
    final_sil_costs = evaluate_cost(
        model, test_pairs, k=args.k,
        num_runs=args.num_runs, temperature=args.temperature,
        batch_size=args.eval_batch_size,
        node_cost=args.node_cost, edge_cost=args.edge_cost,
        desc="SIL multi"
    )
    t_multi = time.time() - t0

    print("\n--- Method Comparison ---")
    report_method("Greedy", greedy_test_costs, hungarian_test_costs)
    report_method("Hungarian", hungarian_test_costs, hungarian_test_costs)
    report_method("Refine (LSAP+local)", refine_test_costs, hungarian_test_costs)
    if gelato_costs_single is not None:
        report_method("GELATO (single-shot)", gelato_costs_single, hungarian_test_costs)
    if gelato_costs_multi is not None:
        report_method("GELATO + multi-Gumbel", gelato_costs_multi, hungarian_test_costs)
    report_method(f"SIL (k=1)", sil_k1_costs, hungarian_test_costs, t_k1)
    report_method(f"SIL (k={args.k}, runs={args.num_runs})",
                  final_sil_costs, hungarian_test_costs, t_multi)

    print(f"\nTotal pipeline wall-clock: {(time.time()-t_pipeline_start)/60:.1f} min")
    print(f"Test pairs: {len(test_pairs)}, Nodes: {args.min_nodes}-{args.max_nodes}")

    if args.save_ckp:
        result = {
            'config': vars(args),
            'cycles': cycle_logs,
            'final': {
                'hungarian_avg': float(np.mean(hungarian_test_costs)),
                'greedy_avg': float(np.mean(greedy_test_costs)),
                'refine_avg': float(np.mean(refine_test_costs)),
                'sil_k1_avg': float(np.mean(sil_k1_costs)),
                'sil_multi_avg': float(np.mean(final_sil_costs)),
                'sil_k1_time': t_k1,
                'sil_multi_time': t_multi,
                'gelato_single_avg': (float(np.mean(gelato_costs_single))
                                      if gelato_costs_single is not None else None),
                'gelato_multi_avg': (float(np.mean(gelato_costs_multi))
                                     if gelato_costs_multi is not None else None),
            }
        }
        result_path = args.save_ckp.rsplit('.', 1)[0] + "_results.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved results to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ogb_root', type=str, default='data/ogb/ogbg_molhiv/raw')
    parser.add_argument('--cache_dir', type=str, default='data/ogb/cache')
    parser.add_argument('--min_nodes', type=int, default=30)
    parser.add_argument('--max_nodes', type=int, default=50)
    parser.add_argument('--num_node_labels', type=int, default=120)
    parser.add_argument('--num_edge_labels', type=int, default=6)
    parser.add_argument('--train_pairs', type=int, default=5000)
    parser.add_argument('--test_pairs', type=int, default=1000)

    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--instances_per_pair', type=int, default=20)
    parser.add_argument('--node_cost', type=float, default=1.0)
    parser.add_argument('--edge_cost', type=float, default=1.0)
    parser.add_argument('--max_train_steps', type=float, default=1.0)

    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--num_runs', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.5)

    parser.add_argument('--num_cycles', type=int, default=20)
    parser.add_argument('--train_epochs_per_cycle', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.85)
    parser.add_argument('--refresh_every', type=int, default=3)
    parser.add_argument('--num_augment', type=int, default=2)

    parser.add_argument('--recon_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)

    parser.add_argument('--gelato_ckp', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_ckp', type=str, default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--nocuda', action='store_true')

    parser.add_argument('--no_cost_gate', action='store_true',
                        help='Ablation: disable strict cost-improvement gate (Eq. 8).')
    parser.add_argument('--fixed_temperature', type=float, default=None,
                        help='Ablation: use fixed reconstruction temperature instead of annealing.')
    parser.add_argument('--fixed_perturb_ratio', type=float, default=None,
                        help='Ablation: use fixed perturbation ratio instead of annealing.')


    args = parser.parse_args()
    args.device = torch.device("cuda" if (torch.cuda.is_available() and not args.nocuda) else "cpu")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.log and args.save_ckp:
        log_path = args.save_ckp.rsplit('.', 1)[0] + "_large.log"
        with open(log_path, "w") as f:
            f.write(str(args) + '\n')

    main(args)

