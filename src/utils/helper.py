import os
import click
import torch
import evaluate
import numpy as np
import copy
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from typing import Dict, Optional, Sequence
from datasets import load_dataset
import lm_eval

def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(
        result.get('acc_norm,none', result['acc,none']) for task, result in results.items() if 'mmlu' not in task
    )

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get('acc_norm,none', result['acc,none']) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)
