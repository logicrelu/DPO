"""简单的 DPO 训练脚本，带详细注释。

本示例演示如何用 TRL 的 ``DPOTrainer`` 在小型偏好数据上
进行 Direct Preference Optimization（直接偏好优化）。
训练好的权重会保存在 ``models/`` 目录下。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer


@dataclass
class DPOConfig:
    """用于收集训练配置的简单数据类。"""

    model_name: str
    data_path: Path
    output_dir: Path
    max_length: int = 512
    batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str | None = None
    trust_remote_code: bool = False


def parse_args() -> DPOConfig:
    """解析命令行参数，并返回 ``DPOConfig``。"""

    parser = argparse.ArgumentParser(description="Run a tiny DPO training job.")
    parser.add_argument(
        "--model_name",
        default="sshleifer/tiny-gpt2",
        help="基础因果语言模型的名称或路径（默认为小模型，便于快速演示）",
    )
    parser.add_argument(
        "--data_path",
        default=Path("data/preferences.jsonl"),
        type=Path,
        help="存放偏好数据的 JSONL 文件路径，包含 prompt/ chosen/ rejected 三列。",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("models/dpo-checkpoint"),
        type=Path,
        help="训练完成后保存权重的目录。",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="单个样本的最大 token 长度，超过将被截断。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="每个设备的训练批大小，示例数据较小建议保持 2。",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="训练轮数，示例数据很小，一轮即可看到效果。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="优化器学习率。",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO 损失中使用的 beta 温度参数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现。",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="是否使用 LoRA 适配器（推荐在大模型上节省显存，例如 Qwen 2.5 7B）。",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA 秩（r），值越大表示容量越大。",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA 缩放因子 alpha。",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout，用于缓解过拟合。",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help=(
            "逗号分隔的 LoRA 注入模块名称列表。"
            " 若留空则默认使用 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj。"
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="为需要自定义模型代码的仓库打开 trust_remote_code（例如部分 Qwen 版本）。",
    )
    args = parser.parse_args()
    return DPOConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        trust_remote_code=args.trust_remote_code,
    )


def prepare_dataset(data_path: Path) -> Dict:
    """加载 JSONL 偏好数据并返回 Hugging Face 数据集。

    数据文件应包含 ``prompt``、``chosen`` 与 ``rejected`` 三个字段：
    - prompt: 给模型的指令或问题。
    - chosen: 优选回答（正样本）。
    - rejected: 次优或错误回答（负样本）。
    """

    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    dataset = load_dataset("json", data_files={"train": str(data_path)}, split="train")
    # DPOTrainer 期望字段名正好是 prompt / chosen / rejected，所以无需额外处理。
    return dataset


def main() -> None:
    config = parse_args()

    # 1) 固定随机种子，保证可复现。
    torch.manual_seed(config.seed)

    # 2) 加载数据集。
    dataset = prepare_dataset(config.data_path)

    # 3) 加载基础模型与分词器。
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    # 一些小模型没有 pad_token，DPOTrainer 需要设置 pad_token 才能正常工作。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )

    # 可选：为大模型（如 Qwen 2.5 7B）启用 LoRA，以减少显存占用。
    if config.use_lora:
        target_modules = (
            [m.strip() for m in config.lora_target_modules.split(",")]
            if config.lora_target_modules
            else [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # 4) 配置训练超参。
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="no",
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
    )

    # 5) 创建 DPOTrainer。
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=config.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_length // 2,
    )

    # 6) 开始训练。
    dpo_trainer.train()

    # 7) 将模型与分词器保存到指定目录（默认为 models/ 下）。
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dpo_trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"模型已保存到: {config.output_dir}")


if __name__ == "__main__":
    main()
