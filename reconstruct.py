"""
SIL Local Reconstruction 改进模块（含消融开关）

包含两种互补的重构策略：
1. 温度采样重构（Global Resample）：用 Gumbel-Max 从头采样，寻找全局更优解
2. 局部扰动重构（Local Perturbation）：在当前好解上做扰动+修复，精细搜索

消融开关（默认关闭，保持原行为）：
- no_cost_gate=True：去掉式 (8) 严格不等式，无条件接受最低代价候选
- fixed_temperature=float：关闭温度退火，全程用固定温度
- fixed_perturb_ratio=float：关闭扰动比例退火，全程用固定比例
"""

import torch
import numpy as np
from src.utils import matching_cost


# ============================================================
# 工具：候选接受规则（cost gate 消融的核心）
# ============================================================

def _accept(cost_new, cost_old, no_cost_gate=False):
    """
    伪标签接受规则。
    - no_cost_gate=False（默认）：式 (8) 严格不等式，仅当 new < old 接受
    - no_cost_gate=True（消融）：无条件接受候选（破坏命题 1 的单调性）
    """
    if no_cost_gate:
        return True
    return cost_new < cost_old


# ============================================================
# 策略 1：温度采样全局重构
# ============================================================

def global_resample(g1_list, g2_list, matching_list, model,
                    k=32, num_rounds=3, batch_size=32,
                    node_cost=1.0, edge_cost=1.0, temperature=1.0,
                    no_cost_gate=False):
    """
    从头推理 k 条路径，用 Gumbel-Max 采样。
    no_cost_gate=False（默认）：仅在 cost 严格降低时替换；
    no_cost_gate=True（消融）：直接用新候选替换当前 matching。
    """
    improved_matchings = [m.clone() for m in matching_list]
    nc, ec = node_cost, edge_cost
    total_improvements = 0

    model.eval()
    with torch.no_grad():
        for round_idx in range(num_rounds):
            for start in range(0, len(g1_list), batch_size):
                end = min(start + batch_size, len(g1_list))
                batch_g1 = g1_list[start:end]
                batch_g2 = g2_list[start:end]

                try:
                    new_matchings = model.batch_ensemble_inference(
                        batch_g1, batch_g2, k=k, temperature=temperature
                    )
                    for i, idx in enumerate(range(start, end)):
                        cost_old = matching_cost(
                            g1_list[idx], g2_list[idx],
                            improved_matchings[idx], nc, ec
                        )
                        cost_new = matching_cost(
                            g1_list[idx], g2_list[idx],
                            new_matchings[i], nc, ec
                        )
                        if _accept(cost_new, cost_old, no_cost_gate):
                            improved_matchings[idx] = new_matchings[i]
                            total_improvements += 1
                except Exception as e:
                    print(f"  Warning: batch {start}-{end} failed: {e}")
                    continue

    return improved_matchings, total_improvements


# ============================================================
# 策略 2：局部扰动重构
# ============================================================

def perturb_and_repair(g1_list, g2_list, matching_list, model,
                       k=32, num_rounds=3, batch_size=32,
                       node_cost=1.0, edge_cost=1.0,
                       perturb_ratio=0.3, temperature=0.5,
                       no_cost_gate=False):
    """
    在当前好解基础上做局部扰动 + 模型修复。
    no_cost_gate=True 时，每次扰动产生的 hybrid 都会替换当前 matching
    （而不是只在 cost 更低时替换）；这会破坏命题 1 的单调性，仅用于消融。
    """
    improved_matchings = [m.clone() for m in matching_list]
    nc, ec = node_cost, edge_cost
    total_improvements = 0

    model.eval()
    with torch.no_grad():
        for round_idx in range(num_rounds):
            for start in range(0, len(g1_list), batch_size):
                end = min(start + batch_size, len(g1_list))
                batch_g1 = g1_list[start:end]
                batch_g2 = g2_list[start:end]

                try:
                    new_matchings = model.batch_ensemble_inference(
                        batch_g1, batch_g2, k=k, temperature=temperature
                    )

                    for i, idx in enumerate(range(start, end)):
                        current_m = improved_matchings[idx]
                        new_m = new_matchings[i]
                        n_s = g1_list[idx].x.size(0)

                        matched_indices = (current_m >= 0).nonzero(as_tuple=True)[0]
                        if len(matched_indices) <= 1:
                            continue

                        num_destroy = max(1, int(len(matched_indices) * perturb_ratio))

                        cur_cost = matching_cost(
                            g1_list[idx], g2_list[idx], current_m, nc, ec
                        )
                        best_cost = cur_cost
                        best_matching = current_m

                        for _ in range(3):  # 每个图对 3 次不同扰动
                            destroy_idx = matched_indices[
                                torch.randperm(len(matched_indices))[:num_destroy]
                            ]
                            hybrid = current_m.clone()
                            for d_idx in destroy_idx:
                                hybrid[d_idx] = new_m[d_idx]
                            hybrid = resolve_conflicts(hybrid, n_s)

                            cost = matching_cost(
                                g1_list[idx], g2_list[idx], hybrid, nc, ec
                            )
                            # cost gate 仅在多次扰动之间选最优；
                            # 消融时则总是接受最后一次扰动结果（不和 current 比较）
                            if cost < best_cost:
                                best_cost = cost
                                best_matching = hybrid

                        # 是否替换 current_m：受 no_cost_gate 控制
                        if _accept(best_cost, cur_cost, no_cost_gate):
                            if not torch.equal(best_matching, current_m):
                                improved_matchings[idx] = best_matching
                                total_improvements += 1

                except Exception as e:
                    print(f"  Warning: batch {start}-{end} failed: {e}")
                    continue

    return improved_matchings, total_improvements


def resolve_conflicts(matching, n_s):
    """
    解决 matching 中的冲突：如果多个源节点映射到同一个目标节点，
    只保留第一个，其余设为 -1。
    """
    result = matching.clone()
    seen_targets = {}
    for i in range(n_s):
        t = result[i].item()
        if t < 0:
            continue
        if t in seen_targets:
            result[i] = -1
        else:
            seen_targets[t] = i
    return result


# ============================================================
# 组合策略：先全局探索，再局部精修
# ============================================================

def combined_reconstruct(g1_list, g2_list, matching_list, model,
                         k=32, batch_size=32,
                         node_cost=1.0, edge_cost=1.0,
                         temperature=1.0, cycle_idx=0, num_cycles=10,
                         no_cost_gate=False,
                         fixed_temperature=None,
                         fixed_perturb_ratio=None):
    """
    组合重构策略，根据训练阶段自动调整。

    新增消融参数（默认 None / False，保持原行为）：
        no_cost_gate (bool):       去掉严格 cost 接受规则
        fixed_temperature (float): 固定温度，关闭温度退火
        fixed_perturb_ratio (float): 固定扰动比例，关闭扰动比例退火
    """
    nc, ec = node_cost, edge_cost
    progress = cycle_idx / max(num_cycles - 1, 1)  # 0.0 → 1.0

    # 温度退火：1.5 → 0.3（默认）
    if fixed_temperature is None:
        temp = 1.5 - 1.2 * progress
    else:
        temp = float(fixed_temperature)

    # 扰动比例退火：0.5 → 0.15（默认）
    if fixed_perturb_ratio is None:
        perturb_ratio = 0.5 - 0.35 * progress
    else:
        perturb_ratio = float(fixed_perturb_ratio)

    # 全局采样轮数退火：3 → 1
    global_rounds = max(1, int(3 - 2 * progress))
    # 局部扰动轮数递增：1 → 3
    local_rounds = max(1, int(1 + 2 * progress))

    costs_before = [
        matching_cost(g1_list[i], g2_list[i], matching_list[i], nc, ec)
        for i in range(len(g1_list))
    ]

    current_matchings = [m.clone() for m in matching_list]
    total_imp = 0

    # 打印当前轮的设置（消融时方便检查）
    flags = []
    if no_cost_gate:
        flags.append("NO_COST_GATE")
    if fixed_temperature is not None:
        flags.append(f"FIXED_TEMP={fixed_temperature}")
    if fixed_perturb_ratio is not None:
        flags.append(f"FIXED_RHO={fixed_perturb_ratio}")
    flag_str = f" [{', '.join(flags)}]" if flags else ""

    # Phase A: 全局温度采样
    print(f"    Global resample: rounds={global_rounds}, temp={temp:.2f}{flag_str}")
    current_matchings, imp_a = global_resample(
        g1_list, g2_list, current_matchings, model,
        k=k, num_rounds=global_rounds, batch_size=batch_size,
        node_cost=nc, edge_cost=ec, temperature=temp,
        no_cost_gate=no_cost_gate
    )
    total_imp += imp_a

    # Phase B: 局部扰动精修
    print(f"    Local perturbation: rounds={local_rounds}, "
          f"perturb_ratio={perturb_ratio:.2f}, temp={temp*0.5:.2f}{flag_str}")
    current_matchings, imp_b = perturb_and_repair(
        g1_list, g2_list, current_matchings, model,
        k=k, num_rounds=local_rounds, batch_size=batch_size,
        node_cost=nc, edge_cost=ec,
        perturb_ratio=perturb_ratio, temperature=temp * 0.5,
        no_cost_gate=no_cost_gate
    )
    total_imp += imp_b

    costs_after = [
        matching_cost(g1_list[i], g2_list[i], current_matchings[i], nc, ec)
        for i in range(len(g1_list))
    ]

    return current_matchings, total_imp, np.mean(costs_before), np.mean(costs_after)
