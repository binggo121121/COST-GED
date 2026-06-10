"""
SIL-GED: Self-Improved Learning for Graph Edit Distance
基于 GELATO 的自改进训练框架 (v4: 图对刷新 + 节点排列增强)
"""

import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.dataset import GraphMatchingDataset
from src.subproblem_dataset import make_instance, parallel_process
from src.model import LinkGNN
from src.utils import matching_cost, run_inference, training_step_link
from src.utils import normalized_mae, exact_hit_rate
from reconstruct import combined_reconstruct


# ============================================================
# Phase 0：Hungarian 初始化
# ============================================================

def hungarian_init(g1, g2, node_cost=1.0):
    n1 = g1.x.size(0)
    n2 = g2.x.size(0)
    N = max(n1, n2)
    cost_matrix = np.full((N, N), node_cost)
    for i in range(n1):
        for j in range(n2):
            if torch.equal(g1.x[i], g2.x[j]):
                cost_matrix[i, j] = 0.0
            else:
                cost_matrix[i, j] = node_cost
    for i in range(n1, N):
        for j in range(n2, N):
            cost_matrix[i, j] = 0.0
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matching = torch.full((n1,), -1, dtype=torch.long)
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            matching[r] = c
    return matching


# ============================================================
# 图对刷新：用模型推理初始化新图对的 matching
# ============================================================

def init_matchings_with_model(g1_list, g2_list, model, k=32, batch_size=32, node_cost=1.0):
    """
    为一批新图对生成初始 matching：
    - 如果模型可用，用模型推理（质量更高）
    - 失败时 fallback 到 Hungarian
    """
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
            except Exception:
                for idx in range(start, end):
                    matching_list.append(
                        hungarian_init(g1_list[idx], g2_list[idx], node_cost)
                    )
    return matching_list


def load_graph_pairs(dataset):
    """从 GraphMatchingDataset 提取 g1, g2, gt_matching 列表"""
    g1_list, g2_list, gt_list = [], [], []
    for data in dataset:
        g1 = Data(x=data.x_s, edge_index=data.edge_index_s.long(), edge_attr=data.edge_attr_s)
        g2 = Data(x=data.x_t, edge_index=data.edge_index_t.long(), edge_attr=data.edge_attr_t)
        g1_list.append(g1)
        g2_list.append(g2)
        gt_list.append(data.matching.long())
    return g1_list, g2_list, gt_list


# ============================================================
# 节点排列增强
# ============================================================

def augment_graph_pair(g1, g2, matching):
    """
    随机排列 g1 的节点顺序，同步调整 matching 和边索引。
    返回增强后的 (g1', g2, matching')，g2 不变。
    """
    n1 = g1.x.size(0)
    perm = torch.randperm(n1)
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(n1)

    new_x = g1.x[perm]
    new_edge_index = inv_perm[g1.edge_index]
    new_edge_attr = g1.edge_attr  # 边属性跟着边走，边索引变了但边本身不变

    # 重新映射 matching：new_matching[new_pos] = matching[old_pos]
    new_matching = torch.full_like(matching, -1)
    for new_i in range(n1):
        old_i = perm[new_i].item()
        new_matching[new_i] = matching[old_i]

    new_g1 = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    return new_g1, g2, new_matching


# ============================================================
# Phase B：构造训练数据（支持增强）
# ============================================================

def create_pseudo_label_dataset(g1_list, g2_list, matching_list,
                                num_instances_per_pair=20,
                                num_node_labels=None, num_edge_labels=None,
                                num_augment=0):
    """
    从改进后的匹配构造 GEDFM 训练子问题。
    
    Args:
        num_augment: 每个图对额外生成几个排列增强版本（0=不增强）
    """
    args_list = []

    for i in range(len(g1_list)):
        g1 = g1_list[i]
        g2 = g2_list[i]
        mu = matching_list[i]
        n_s = g1.x.size(0)

        # 原始 + 增强版本
        pairs_to_process = [(g1, g2, mu)]
        for _ in range(num_augment):
            aug_g1, aug_g2, aug_mu = augment_graph_pair(g1, g2, mu)
            pairs_to_process.append((aug_g1, aug_g2, aug_mu))

        for pg1, pg2, pmu in pairs_to_process:
            x_s_raw = pg1.x.argmax(dim=-1).to(torch.int8)
            x_t_raw = pg2.x.argmax(dim=-1).to(torch.int8)
            ea_s_raw = pg1.edge_attr.argmax(dim=-1).to(torch.int8) if pg1.edge_attr.dim() > 1 else pg1.edge_attr.to(torch.int8)
            ea_t_raw = pg2.edge_attr.argmax(dim=-1).to(torch.int8) if pg2.edge_attr.dim() > 1 else pg2.edge_attr.to(torch.int8)

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
            # 每个版本生成的子问题数 = num_instances_per_pair / (1 + num_augment)
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
# 主循环
# ============================================================

def sil_main(args):
    print("=" * 60)
    print("SIL-GED v4: Self-Improved Learning with Pair Refresh")
    print("=" * 60)

    # ---- 加载初始图对 ----
    train_pairs = GraphMatchingDataset(
        name=args.data, root=args.root,
        num_pairs=args.train_pairs, split='train', seed=args.seed
    )
    val_dataset_inf = GraphMatchingDataset(
        name=args.data, root=args.root,
        num_pairs=2000, split='val'
    )

    g1_list, g2_list, gt_matching_list = load_graph_pairs(train_pairs)

    num_node_labels = g1_list[0].x.size(-1)
    num_edge_labels = g1_list[0].edge_attr.size(-1) if g1_list[0].edge_attr.dim() > 1 else int(g1_list[0].edge_attr.max().item()) + 1

    print(f"Loaded {len(g1_list)} pairs, node_labels={num_node_labels}, edge_labels={num_edge_labels}")
    print(f"Settings: refresh_every={args.refresh_every}, num_augment={args.num_augment}")

    # ---- Phase 0: Hungarian ----
    print("\n--- Phase 0: Hungarian Initialization ---")
    matching_list = []
    init_costs, gt_costs = [], []
    for i in tqdm(range(len(g1_list)), desc="Hungarian init"):
        m = hungarian_init(g1_list[i], g2_list[i], node_cost=args.node_cost)
        matching_list.append(m)
        init_costs.append(matching_cost(g1_list[i], g2_list[i], m, args.node_cost, args.edge_cost))
        gt_costs.append(matching_cost(g1_list[i], g2_list[i], gt_matching_list[i], args.node_cost, args.edge_cost))

    print(f"Hungarian avg cost : {np.mean(init_costs):.2f}")
    print(f"Ground-truth avg cost: {np.mean(gt_costs):.2f}")

    # ---- 检测模型输入维度 ----
    probe = create_pseudo_label_dataset(
        g1_list[:2], g2_list[:2], matching_list[:2],
        num_instances_per_pair=1,
        num_node_labels=num_node_labels, num_edge_labels=num_edge_labels,
        num_augment=0
    )
    if len(probe) == 0:
        raise RuntimeError("无法生成 probe 数据，请检查数据格式")
    actual_node_dim = probe[0].x.shape[1]
    actual_edge_dim = probe[0].edge_attr.shape[1]
    print(f"Model input dims: node={actual_node_dim}, edge={actual_edge_dim}")

    # ---- 初始化模型 ----
    model = LinkGNN(actual_node_dim, actual_edge_dim, 128, args.layers, args.node_cost, args.edge_cost)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # ---- SIL 循环 ----
    best_nmae = 1e10
    best_ehr = 0.0
    patience_counter = 0
    best_model_state = None
    total_pairs_seen = len(g1_list)  # 统计总共见过多少不同的图对

    for cycle in range(args.num_cycles):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"SIL Cycle {cycle+1}/{args.num_cycles}  (lr={current_lr:.6f}, "
              f"patience={patience_counter}/{args.patience}, pairs_seen={total_pairs_seen})")
        print(f"{'='*60}")

        # ---- 图对刷新 ----
        if cycle > 0 and args.refresh_every > 0 and cycle % args.refresh_every == 0:
            print(f"\n  [Refresh] Resampling {args.train_pairs} new graph pairs (seed={args.seed + cycle})...")
            train_pairs_new = GraphMatchingDataset(
                name=args.data, root=args.root,
                num_pairs=args.train_pairs, split='train',
                seed=args.seed + cycle  # 不同 seed → 不同配对
            )
            g1_list, g2_list, gt_matching_list = load_graph_pairs(train_pairs_new)
            total_pairs_seen += len(g1_list)

            # 用当前模型为新图对生成初始 matching（比 Hungarian 更好）
            matching_list = init_matchings_with_model(
                g1_list, g2_list, model,
                k=args.k, batch_size=32, node_cost=args.node_cost
            )
            new_costs = [matching_cost(g1_list[i], g2_list[i], matching_list[i],
                                       args.node_cost, args.edge_cost)
                         for i in range(len(g1_list))]
            gt_costs_new = [matching_cost(g1_list[i], g2_list[i], gt_matching_list[i],
                                          args.node_cost, args.edge_cost)
                            for i in range(len(g1_list))]
            print(f"  [Refresh] Model init avg cost: {np.mean(new_costs):.2f} "
                  f"(GT: {np.mean(gt_costs_new):.2f})")

        # Phase A: Local Reconstruction
        print("\n[A] Local Reconstruction")
        improved_matchings, num_imp, cost_bef, cost_aft = combined_reconstruct(
            g1_list, g2_list, matching_list, model,
            k=args.k, batch_size=32,
            node_cost=args.node_cost, edge_cost=args.edge_cost,
            cycle_idx=cycle, num_cycles=args.num_cycles,
            no_cost_gate=args.no_cost_gate,
            fixed_temperature=args.fixed_temperature,
            fixed_perturb_ratio=args.fixed_perturb_ratio,
        )

        matching_list = improved_matchings
        print(f"  improvements={num_imp}, cost: {cost_bef:.2f} → {cost_aft:.2f}")

        # Phase B: Pseudo-label dataset
        print("\n[B] Building training data")
        pseudo_data = create_pseudo_label_dataset(
            g1_list, g2_list, matching_list,
            num_instances_per_pair=args.instances_per_pair,
            num_node_labels=num_node_labels, num_edge_labels=num_edge_labels,
            num_augment=args.num_augment
        )
        print(f"  {len(pseudo_data)} instances")
        if len(pseudo_data) == 0:
            print("  WARNING: 无训练数据，跳过本轮")
            continue
        pseudo_loader = DataLoader(pseudo_data, batch_size=256, shuffle=True, pin_memory=True)

        # Phase C: Train
        print("\n[C] Training GINE")
        model.train()
        for epoch in range(args.train_epochs_per_cycle):
            ep_loss, ep_acc = training_step_link(model, pseudo_loader, optimizer, args)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch+1}/{args.train_epochs_per_cycle}: loss={ep_loss:.5f}, acc={ep_acc:.4f}")

        scheduler.step()

        # Phase D: Evaluate
        print("\n[D] Evaluation")
        model.eval()
        costs, true_costs = run_inference(model, val_dataset_inf, k=args.k, batch_size=64, disable_tqdm=True)
        nmae = normalized_mae(costs, true_costs)
        ehr = exact_hit_rate(costs, true_costs)
        print(f"  nMAE={nmae:.5f}, EHR={ehr:.5f}")

        if nmae < best_nmae:
            best_nmae = nmae
            best_ehr = ehr
            patience_counter = 0
            best_model_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
            if args.save_ckp:
                torch.save(best_model_state, args.save_ckp)
                print(f"  ★ New best! Saved model (nMAE={nmae:.5f}, EHR={ehr:.5f})")
        else:
            patience_counter += 1
            print(f"  No improvement (best nMAE={best_nmae:.5f}, EHR={best_ehr:.5f})")

        if args.log and args.save_ckp:
            with open(args.save_ckp.rsplit('.', 1)[0] + "_sil.log", "a") as f:
                f.write(f"cycle={cycle+1} imp={num_imp} cost={cost_aft:.2f} "
                        f"nMAE={nmae:.5f} EHR={ehr:.5f} lr={current_lr:.6f} "
                        f"pairs_seen={total_pairs_seen} "
                        f"best_nMAE={best_nmae:.5f} best_EHR={best_ehr:.5f}\n")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered! No improvement for {args.patience} cycles.")
            print(f"  Restoring best model (nMAE={best_nmae:.5f}, EHR={best_ehr:.5f})")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                model.to(args.device)
            break

    print(f"\nDone. Best results: nMAE={best_nmae:.5f}, EHR={best_ehr:.5f}")
    print(f"Total unique graph pairs seen: {total_pairs_seen}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--train_pairs', type=int, default=None)
    parser.add_argument('--instances_per_pair', type=int, default=20)
    parser.add_argument('--node_cost', type=float, default=1.0)
    parser.add_argument('--edge_cost', type=float, default=1.0)
    parser.add_argument('--max_train_steps', type=float, default=1.0)
    parser.add_argument('--save_ckp', type=str, default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--nocuda', action='store_true')

    # SIL 参数
    parser.add_argument('--num_cycles', type=int, default=10)
    parser.add_argument('--destroy_k', type=int, default=5)
    parser.add_argument('--num_reconstructions', type=int, default=20)
    parser.add_argument('--train_epochs_per_cycle', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.85)

    # v4 新增参数
    parser.add_argument('--refresh_every', type=int, default=3,
                        help='每隔多少个 cycle 刷新图对（0=不刷新）')
    parser.add_argument('--num_augment', type=int, default=2,
                        help='每个图对的节点排列增强数量（0=不增强）')


    parser.add_argument('--no_cost_gate', action='store_true',
                        help='Ablation: disable strict cost-improvement gate (Eq. 8).')
    parser.add_argument('--fixed_temperature', type=float, default=None,
                        help='Ablation: use fixed reconstruction temperature instead of annealing.')
    parser.add_argument('--fixed_perturb_ratio', type=float, default=None,
                        help='Ablation: use fixed perturbation ratio instead of annealing.')


    args = parser.parse_args()
    args.device = torch.device("cuda" if (torch.cuda.is_available() and not args.nocuda) else "cpu")

    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.log and args.save_ckp:
        with open(args.save_ckp.rsplit('.', 1)[0] + "_sil.log", "w") as f:
            f.write(str(args) + '\n')

    sil_main(args)

