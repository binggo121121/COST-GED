# COST-GED: Cost-verified Self-training for Graph Edit Distance

基于 GELATO 的自改进图编辑距离训练框架。不依赖精确 GED 标签，通过代价验证自训练持续优化图匹配模型。

---

## 环境

```bash
pip install -r requirements.txt
```

---

## 代码结构

```
├── src/
│   ├── model.py / model_large.py   # LinkGNN（model_large 向下兼容）
│   ├── dataset.py                  # 小图数据加载
│   ├── subproblem_dataset.py       # 子问题构造
│   └── utils.py                    # 代价/推理/指标
├── train.py                        # 小图训练
├── train_ablation.py               # 消融训练
├── reconstruct.py                  # 局部重构
├── train_large_graph.py            # 大图训练
├── evaluate_large_graphs.py        # 大图评估（计时）
├── test.py                         # 小图测试
├── hungarian_ged_standalone.py     # 独立匈牙利 GED
├── requirements.txt
└── .gitignore
```

数据放到 `data/`，模型保存到 `ckp/`。

---

## 训练

### 小图

```bash
python train.py \
    --data DATA --root data/ \
    --train_pairs 5000 --instances_per_pair 20 \
    --num_cycles CYCLES --train_epochs_per_cycle 10 \
    --k 32 --refresh_every 3 --num_augment 2 --patience 5 \
    --seed SEED --save_ckp ckp/model.pt --log
```

DATA: `aids | code2-22 | imdb-16 | linux | molhiv-16 | zinc-16`

| 数据集 | num_cycles |
|--------|-----------|
| aids, imdb-16, linux | 15 |
| code2-22, molhiv-16, zinc-16 | 25 |

### 大图

```bash
python train_large_graph.py \
    --ogb_root data/ogb/ogbg_molhiv/raw --cache_dir data/ogb/cache \
    --min_nodes 30 --max_nodes 50 \
    --num_node_labels 120 --num_edge_labels 6 \
    --train_pairs 5000 --test_pairs 1000 \
    --num_cycles 20 --instances_per_pair 20 --train_epochs_per_cycle 10 \
    --k 32 --num_runs 8 --temperature 0.5 \
    --refresh_every 3 --num_augment 2 --patience 5 \
    --recon_batch_size 8 --eval_batch_size 8 \
    --seed SEED --save_ckp ckp/model.pt --log
```

修改 `--min_nodes` / `--max_nodes` 切换节点范围。

---

## 测试 & 评估

```bash
# 小图
python test.py --data DATA --load_ckp ckp/model.pt \
    --k 32 --num_runs 8 --temperature 0.5

# 大图
python evaluate_large_graphs.py \
    --min_nodes 30 --max_nodes 50 --test_pairs 500 \
    --sil_ckp ckp/model.pt \
    --k 32 --num_runs 8 --temperature 0.5 --seed 0 \
    --output results/eval.json
```

---

## 消融实验

```bash
# Full（与 train.py 相同）
python train.py --data DATA ... --seed SEED --save_ckp ckp/full.pt --log

# w/o cost gate
python train.py --data DATA ... --no_cost_gate --seed SEED --save_ckp ckp/no_gate.pt --log

# w/o anneal
python train.py --data DATA ... \
    --fixed_temperature 0.5 --fixed_perturb_ratio 0.3 \
    --seed SEED --save_ckp ckp/no_anneal.pt --log
```
