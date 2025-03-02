import os
import math
import torch
import click
from transformers import LlamaForCausalLM
from transformers.trainer import Trainer
from torch.utils import _pytree as pytree
from peft.tuners import lora
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    get_peft_model)
import torch.nn as nn
import gc
from typing import List, Optional, Union, Dict, Any, cast


def recursive_getattr(obj, attr):
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj

def save_full_model(trainer: Trainer) -> None:
    if not isinstance(trainer.model, (PeftModelForCausalLM, PeftModelForSequenceClassification)):
        raise TypeError(
            f"Expected `PeftModelForCausalLM`, or "
            f"`PeftModelForSequenceClassification`, "
            f"but got {type(trainer.model)}")
    if not trainer.args.should_save:
        return

    state_dict = trainer.model.state_dict()
    file_name = os.path.join(
        trainer.args.output_dir,
        "full_model.pth")
    torch.save(state_dict, file_name)
    click.secho(f"Saved model state dict to {file_name}", fg="green")


def prepare_model_for_lora(
    model: LlamaForCausalLM,
    num_ranks: int,
    compression_ratio: float = 0.0,
    rank_ratio: float = 0.2,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = False,
) -> PeftModelForCausalLM:
        
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(f"Expected LlamaForCausalLM, but got {type(model)}")
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    click.secho(
        f"Applying LoRA with the following configurations:\n"
        f"\t -num_ranks: {num_ranks}\n"
        f"\t -lora_alpha: {lora_alpha}\n"
        f"\t -lora_dropout: {lora_dropout}\n"
        f"\t -target_modules: {target_modules}",
        fg="blue")

    # Create rank pattern based on weight sizes
    rank_pattern = {}
    if rank_ratio > 0.0:
        for name, module in model.named_modules():
            if any(target_module in name for target_module in target_modules):
                d_out, d_in = module.weight.size()
                rank = math.floor(rank_ratio * (1 - compression_ratio) * (d_out * d_in) / (d_out + d_in))
                rank_pattern[name] = rank
    else:
        print("This means you expect the rank to be fixes, so you passed ratio=-1. In that case unstructured_sparsity==compresstion_ratio")
    print("-" * 100)
    print("rank_pattern:", rank_pattern)
    print()
    peft_config = LoraConfig(
        rank_pattern=rank_pattern,
        r=num_ranks,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM")

    new_model = get_peft_model(model, peft_config)    
    new_model.print_trainable_parameters()
    if not isinstance(new_model, PeftModelForCausalLM):
        raise TypeError(f"Expected PeftModelForCausalLM, but got {type(new_model)}")
    return new_model