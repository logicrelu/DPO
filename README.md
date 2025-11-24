# DPO

一个最小可运行的 Direct Preference Optimization（DPO）示例。

## 目录结构
- `train_dpo.py`：带详细中文注释的 DPO 训练脚本，默认保存权重到 `models/`。
- `data/preferences.jsonl`：示例偏好数据，每行包含 `prompt`、`chosen`、`rejected`。
- `models/`：训练输出目录（已放置 `.gitkeep` 以便提交）。

## 环境准备
```bash
pip install torch transformers datasets trl accelerate
```
> 如果显存有限，可使用 CPU 或更小的基础模型（脚本默认 `sshleifer/tiny-gpt2`）。

## 如何运行
```bash
python train_dpo.py \
  --model_name sshleifer/tiny-gpt2 \
  --data_path data/preferences.jsonl \
  --output_dir models/dpo-checkpoint \
  --num_epochs 1 \
  --batch_size 2 \
  --learning_rate 5e-5 \
  --max_length 512 \
  --beta 0.1
```
- 训练完成后，检查 `models/dpo-checkpoint/` 获取模型与分词器。
- 想要继续训练或换模型，只需更换命令行参数。

## 数据说明
偏好数据必须包含以下字段：
- `prompt`：给模型的问题或指令。
- `chosen`：偏好答案（正样本）。
- `rejected`：次优或错误答案（负样本）。

你可以在 `data/preferences.jsonl` 中按照同样格式追加更多行。

## 小贴士
- 数据越丰富，DPO 效果越好；但请从小规模开始，确保流程跑通。
- 如果想用中文模型，可将 `--model_name` 替换为对应的权重名称。
- 训练时若显存不足，可调小 `--batch_size` 或 `--max_length`。
