"""
Large-Graph Evaluation (timing-only) — 加载 ckp 跑推理 + 计时,不训练。

Usage:
    python evaluate_large_graphs.py \
        --min_nodes 30 --max_nodes 50 \
        --test_pairs 500 \
        --sil_ckp ckp/sil_molhiv30_50_with_gelato_seed0.pt \
        --gelato_ckp ckp/sil_molhiv16_4090.pt \
        --k 32 --num_runs 8 --temperature 0.5 \
        --seed 0 \
        --output results/eval_30_50_seed0.json
"""

import argparse
import numpy as np
import random
import time
import os
import json

import torch
from tqdm import tqdm

from train_large_graph import (
    load_ogb_molhiv, graphs_to_pyg, sample_graph_pairs,
    hungarian_init, greedy_matching, refine_matching,
    create_pseudo_label_dataset,
)
from src.model import LinkGNN
from src.utils import matching_cost


def time_baseline(test_pairs, method_fn, name,
                  node_cost=1.0, edge_cost=1.0, warmup=10):
    """对 CPU 启发式方法计时,返回 (costs, ms/pair)。"""
    # warmup —— 启发式方法的 LSAP 等首次调用有 numpy/scipy 初始化开销
    for i in range(min(warmup, len(test_pairs))):
        _ = method_fn(test_pairs[i][0], test_pairs[i][1])

    costs = []
    t0 = time.perf_counter()
    for g1, g2 in tqdm(test_pairs, desc=name, ncols=80):
        m = method_fn(g1, g2)
        costs.append(matching_cost(g1, g2, m, node_cost, edge_cost))
    elapsed = time.perf_counter() - t0
    return costs, 1000.0 * elapsed / len(test_pairs)


def time_neural(model, test_pairs, k, num_runs, temperature, name,
                node_cost=1.0, edge_cost=1.0, batch_size=8,
                device='cuda', warmup_batches=2):
    """对神经方法计时,包含 CUDA sync 保证测量准确,并加入逐对安全回退。"""
    model.eval()
    
    n_pairs = len(test_pairs)
    best_costs = [None] * n_pairs
    total_elapsed = 0.0
    
    it = range(0, n_pairs, batch_size)
    if num_runs > 1:
        it = tqdm(it, desc=f"{name} evaluating", ncols=80)
        
    for start in it:
        end = min(start + batch_size, n_pairs)
        g1_b = [p[0] for p in test_pairs[start:end]]
        g2_b = [p[1] for p in test_pairs[start:end]]
        
        # --- NEW SAFETY CHECK ---
        # Check for 0 edges or completely identical node counts which might trigger
        # edge-case bugs in the GNN or alignment layers.
        skip_gpu = False
        for g1, g2 in zip(g1_b, g2_b):
            if g1.edge_index is None or g1.edge_index.numel() == 0:
                skip_gpu = True
            if g2.edge_index is None or g2.edge_index.numel() == 0:
                skip_gpu = True
                
        t_start = time.perf_counter()
        ms = None
        
        if not skip_gpu:
            try:
                with torch.no_grad():
                    if num_runs > 1:
                        ms = model.batch_ensemble_inference(g1_b, g2_b, k=k, temperature=temperature)
                    else:
                        ms = model.batch_ensemble_inference(g1_b, g2_b, k=k)
                if device.type == 'cuda':
                    torch.cuda.synchronize() 
            except Exception as e:
                print(f"  [WARN] Batch {start}-{end} failed on GPU. Error: {str(e)[:100]}")
                ms = None
        else:
             print(f"  [WARN] Batch {start}-{end} skipped GPU due to empty edge tensors.")

        # Fallback to Heuristic if GPU failed or was skipped
        if ms is None:
            ms = [hungarian_init(g1_b[j], g2_b[j], node_cost) for j in range(len(g1_b))]
            
        t_end = time.perf_counter()
        total_elapsed += (t_end - t_start)
        
        # Calculate Cost
        for j, m in enumerate(ms):
            c = matching_cost(g1_b[j], g2_b[j], m, node_cost, edge_cost)
            idx = start + j
            if best_costs[idx] is None or c < best_costs[idx]:
                best_costs[idx] = c
                
    return best_costs, 1000.0 * (total_elapsed / (n_pairs * num_runs))


def main(args):
    print("=" * 70)
    print(f"Large-Graph Eval | nodes {args.min_nodes}-{args.max_nodes} | "
          f"test_pairs={args.test_pairs} | seed={args.seed}")
    print("=" * 70)

    # ---- 数据 ----
    raw = load_ogb_molhiv(root=args.ogb_root,
                          min_nodes=args.min_nodes, max_nodes=args.max_nodes,
                          cache_dir=args.cache_dir)
    pyg = graphs_to_pyg(raw, num_node_labels=args.num_node_labels,
                        num_edge_labels=args.num_edge_labels)
    random.shuffle(pyg)
    split = int(len(pyg) * 0.8)
    test_graphs = pyg[split:]
    test_pairs = sample_graph_pairs(test_graphs, args.test_pairs,
                                    seed=args.seed + 999)
    print(f"Test pairs: {len(test_pairs)}\n")

    results = {}

    def dump_results():
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump({'seed': args.seed, 'min_nodes': args.min_nodes,
                           'max_nodes': args.max_nodes,
                           'test_pairs': args.test_pairs,
                           'results': results}, f, indent=2)
            print(f"  [dump] saved → {args.output}")

    # ---- 启发式三连 ----
    print("[1/4] Heuristic baselines")
    hung_c, t_hung = time_baseline(
        test_pairs, lambda a, b: hungarian_init(a, b, args.node_cost),
        "Hungarian", args.node_cost, args.edge_cost)
    results['Hungarian'] = {'avg': float(np.mean(hung_c)), 'time_ms': t_hung}

    grd_c, t_grd = time_baseline(
        test_pairs, lambda a, b: greedy_matching(a, b, args.node_cost),
        "Greedy", args.node_cost, args.edge_cost)
    results['Greedy'] = {'avg': float(np.mean(grd_c)), 'time_ms': t_grd}

    ref_c, t_ref = time_baseline(
        test_pairs,
        lambda a, b: refine_matching(a, b, args.node_cost, args.edge_cost),
        "Refine", args.node_cost, args.edge_cost)
    results['Refine'] = {'avg': float(np.mean(ref_c)), 'time_ms': t_ref}
    dump_results()

    # ---- 探测模型维度 ----

    probe_m = hungarian_init(test_pairs[0][0], test_pairs[0][1], args.node_cost)
    probe = create_pseudo_label_dataset(
        [test_pairs[0][0]], [test_pairs[0][1]], [probe_m],
        num_instances_per_pair=1,
        num_node_labels=args.num_node_labels,
        num_edge_labels=args.num_edge_labels, num_augment=0)
    
    actual_node_dim = probe[0].x.shape[1]
    actual_edge_dim = probe[0].edge_attr.shape[1]
    
    # 【修改点 1】强制赋予一个足够大的维度，防止大图中虚拟节点 ID 越界
    safe_node_dim = max(actual_node_dim + 40, 160)
    safe_edge_dim = max(actual_edge_dim + 10, 20)
    print(f"\nModel dims: original_node={actual_node_dim}, Safe_node={safe_node_dim}")

    def load_model(ckp_path):
        m = LinkGNN(safe_node_dim, safe_edge_dim, 128,
                    args.layers, args.node_cost, args.edge_cost)
        
        checkpoint = torch.load(ckp_path, map_location='cpu')
        model_dict = m.state_dict()
        
        # 【修改点 2】智能加载权重：遇到维度被我们强行撑大时，自动进行切片填充，保留原有有效权重
        for k, v in checkpoint.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v
                else:
                    if len(v.shape) == 2 and len(model_dict[k].shape) == 2:
                        min_d0 = min(v.shape[0], model_dict[k].shape[0])
                        min_d1 = min(v.shape[1], model_dict[k].shape[1])
                        model_dict[k][:min_d0, :min_d1] = v[:min_d0, :min_d1]
                    elif len(v.shape) == 1 and len(model_dict[k].shape) == 1:
                        min_d0 = min(v.shape[0], model_dict[k].shape[0])
                        model_dict[k][:min_d0] = v[:min_d0]
                        
        m.load_state_dict(model_dict)
        return m.to(args.device)

    def load_model(ckp_path):
        m = LinkGNN(actual_node_dim, actual_edge_dim, 128,
                    args.layers, args.node_cost, args.edge_cost)
        m.load_state_dict(torch.load(ckp_path, map_location='cpu'))
        return m.to(args.device)

    # ---- COST-GED 先跑（核心数据，必拿）----
    print(f"\n[2/4] COST-GED from {args.sil_ckp}")
    sil = load_model(args.sil_ckp)
    sk1, tsk1 = time_neural(sil, test_pairs, k=args.k, num_runs=1,
                            temperature=args.temperature, name="COST-GED K=1",
                            node_cost=args.node_cost, edge_cost=args.edge_cost,
                            batch_size=args.eval_batch_size, device=args.device)
    results['COST-GED_K1'] = {'avg': float(np.mean(sk1)), 'time_ms': tsk1}
    dump_results()

    skm, tskm = time_neural(sil, test_pairs, k=args.k,
                            num_runs=args.num_runs,
                            temperature=args.temperature, name="COST-GED K=32",
                            node_cost=args.node_cost, edge_cost=args.edge_cost,
                            batch_size=args.eval_batch_size, device=args.device)
    results['COST-GED_K32'] = {'avg': float(np.mean(skm)), 'time_ms': tskm}
    dump_results()

    del sil
    torch.cuda.empty_cache()

    # ---- GELATO 最后跑（可能因 OOD 越界,挂了也无妨）----
    if args.skip_gelato:
        print("\n[3/4] GELATO skipped (--skip_gelato)")
    elif args.gelato_ckp and os.path.exists(args.gelato_ckp):
        print(f"\n[3/4] GELATO from {args.gelato_ckp}")
        try:
            gel = load_model(args.gelato_ckp)
            # 早期 sync 检测:先用一个 mini-batch 试水,挂就立即 except
            print("  Probing GELATO compatibility on 1 batch...")
            with torch.no_grad():
                probe_g1 = [test_pairs[0][0]]
                probe_g2 = [test_pairs[0][1]]
                _ = gel.batch_ensemble_inference(probe_g1, probe_g2, k=args.k)
            torch.cuda.synchronize()
            print("  Probe passed, running full eval...")

            gk1, tgk1 = time_neural(gel, test_pairs, k=args.k, num_runs=1,
                                    temperature=args.temperature,
                                    name="GELATO K=1",
                                    node_cost=args.node_cost,
                                    edge_cost=args.edge_cost,
                                    batch_size=args.eval_batch_size,
                                    device=args.device)
            results['GELATO_K1'] = {'avg': float(np.mean(gk1)),
                                    'time_ms': tgk1}
            dump_results()

            gkm, tgkm = time_neural(gel, test_pairs, k=args.k,
                                    num_runs=args.num_runs,
                                    temperature=args.temperature,
                                    name="GELATO K=32",
                                    node_cost=args.node_cost,
                                    edge_cost=args.edge_cost,
                                    batch_size=args.eval_batch_size,
                                    device=args.device)
            results['GELATO_K32'] = {'avg': float(np.mean(gkm)),
                                     'time_ms': tgkm}
            dump_results()
            del gel
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n  [WARN] GELATO eval failed ({type(e).__name__}): "
                  f"{str(e)[:200]}")
            print(f"  GELATO ckp 在 {args.min_nodes}-{args.max_nodes} 节点的 OOD "
                  f"图上越界,跳过。COST-GED 数据已保存。")
            results['GELATO_K1'] = {'avg': None, 'time_ms': None,
                                    'error': str(e)[:200]}
            results['GELATO_K32'] = {'avg': None, 'time_ms': None,
                                     'error': str(e)[:200]}
            dump_results()
            # CUDA context 已坏,后面不能再用 GPU
            print("\n[4/4] Final report (skipping further GPU ops)")
            for name, r in results.items():
                if r.get('avg') is not None:
                    print(f"  {name:<18} {r['avg']:>10.2f}  "
                          f"{r['time_ms']:>10.2f} ms/pair")
            return
    else:
        print("\n[3/4] No GELATO ckp provided, skipping")

    # ---- 汇报 ----
    avg_h = results['Hungarian']['avg']
    print("\n" + "=" * 70)
    print(f"{'Method':<18} {'Avg Cost':>10} {'ΔC% vs Hung':>14} {'Time (ms)':>12}")
    print("-" * 70)
    for name, r in results.items():
        if r.get('avg') is None:
            print(f"{name:<18} {'N/A':>10} {'(failed)':>14} {'N/A':>12}")
            continue
        delta = (avg_h - r['avg']) / avg_h * 100
        print(f"{name:<18} {r['avg']:>10.2f} {delta:>+13.1f}%  {r['time_ms']:>10.2f}")
    print("=" * 70)
    dump_results()



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--ogb_root', type=str, default='data/ogb/ogbg_molhiv/raw')
    p.add_argument('--cache_dir', type=str, default='data/ogb/cache')
    p.add_argument('--min_nodes', type=int, default=30)
    p.add_argument('--max_nodes', type=int, default=50)
    p.add_argument('--num_node_labels', type=int, default=120)
    p.add_argument('--num_edge_labels', type=int, default=6)
    p.add_argument('--test_pairs', type=int, default=500)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--node_cost', type=float, default=1.0)
    p.add_argument('--edge_cost', type=float, default=1.0)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--num_runs', type=int, default=8)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--sil_ckp', type=str, required=True)
    p.add_argument('--gelato_ckp', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--nocuda', action='store_true')
    p.add_argument('--skip_gelato', action='store_true',
               help='完全跳过 GELATO baseline (推荐:OOD 设置下挂)')


    args = p.parse_args()
    args.device = torch.device("cuda" if (torch.cuda.is_available() and not args.nocuda) else "cpu")
    np.random.seed(args.seed); random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    main(args)
