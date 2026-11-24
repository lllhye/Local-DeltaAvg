# Local-DeltaAvg 代码交接包

**Directional Structurality: A Lightweight Proxy for Reasoning Data Quality**

## 内容

- `experiments/answer_masking/` 对比三种答案处理方式，生成校准分数和等预算入选清单。随包提供小样本 JSONL、CSV 和现稿汇总结果。
- `experiments/engineering_cost/` 测量代表性分片的打分速度，估算全池开销并生成 LaTeX 宏。
- `experiments/family_aware/` 包含两套 24 类分类表、零样本机制分配、覆盖感知选择及结果汇总。
- `experiments/reviewer_controls/` 生成目标不变性审计表，筛选上下文安全样本，比较单视图与多视图，并导出对照实验结果。
- `plots/plot_experiments.py` 生成扩刊稿的 11 张实验图和诊断图。
- `configs/public_pool_recipe.yaml` 记录公开数据池的版本、过滤规则以及 CPT、SFT 和评测配置。
- `results/manuscript_tables/` 收录现稿的结果宏，用于核对生成结果。各模块的 `results.json` 来自现稿汇总值。

各模块的 README 记录输入格式和命令。命令从本包根目录运行。四个表格生成器默认写到 `outputs/tables/`。输入字段 `proofarrow_score` 表示 Local-DeltaAvg 分数，可用 `--score-key` 读取其他分数字段。

## 快速检查

以下命令用 Python 标准库运行。第一条命令打印样本在三个变体下的上下文，第二条使用附带的示例分数生成小预算清单。

~~~bash
python3 experiments/answer_masking/run_answer_masking_scores.py \
  --input experiments/answer_masking/example_input.jsonl \
  --output outputs/example_scores.csv \
  --dry-run

python3 experiments/answer_masking/build_selection_manifests.py \
  --scores experiments/answer_masking/example_scores.csv \
  --null-scores experiments/answer_masking/example_null_scores.csv \
  --output-dir outputs/smoke_manifests \
  --budget 50 --min-null-bin 1
~~~

运行以下命令可按现稿汇总数据生成三份表格宏。

~~~bash
python3 experiments/answer_masking/make_results_tex.py \
  --results experiments/answer_masking/results.json
python3 experiments/family_aware/make_family_results_tex.py \
  --results experiments/family_aware/results.json
python3 experiments/reviewer_controls/make_reviewer_results_tex.py \
  --results experiments/reviewer_controls/results.json
~~~

## 绘图与模型打分

建议使用 Python 3.10 或更新版本。机制感知选择使用 NumPy，绘图还使用 Matplotlib。绘图脚本将 PDF 写入 `plots/`。

~~~bash
python3 -m pip install -r requirements.txt
python3 plots/plot_experiments.py
~~~

加载探针模型的打分、机制分配和速度测量命令还需要 PyTorch、Transformers 和模型文件。使用适合本机 CUDA、MPS 或 CPU 环境的 PyTorch。

~~~bash
python3 -m pip install -r requirements-model.txt
~~~

本包交付参考实现、示例及稿件结果。20B 全池复现还需要完整候选语料、逐条匹配的空对照数据、原运行清单、CPT/SFT 训练入口和评测产物。工程实测宏收在 `results/manuscript_tables/engineering_cost_results.tex`，分片测速脚本按所提供的分片与硬件重新估算。

许可见 `LICENSE`。
