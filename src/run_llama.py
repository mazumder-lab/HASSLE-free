#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import gc
import copy
import logging
import math
import os
import sys
import click
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Dict, Any

import datasets
import evaluate
import torch
from datasets import (
    load_dataset,
    concatenate_datasets)

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from models import misc_utils
from models import lora_utils
from utils import helper
from utils import callback_utils
import time

from new_llama import llama_compressor_gd, get_loaders, llama_eval
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    lora_num_ranks: int = field(default=8)
    lora_dropout: float = field(default=0.05)
    lora_config: Optional[str] = field(default=None)
    lora_model_name: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class SparsityArguments:
    compression_ratio: float = field(
        default=0.0,
        metadata={"help": "Sparasity of pruning method"}
    )
    unstructured_sparsity: float = field(
        default=0.0,
        metadata={"help": "Sparasity of pruning method"}
    )
    prunen: int = field(
        default=0,
        metadata={"help": "N for N:M sparsity"}
    )
    prunem: int = field(
        default=0,
        metadata={"help": "M for N:M sparsity"}
    )
    rank_ratio: float = field(
        default=-1,
        metadata={"help": "ratio of #params that go to rank"}
    )
    seqlen: int = field(
        default=2048,
        metadata={"help": "regularization term"}
    )
    nsamples: int = field(
        default=128,
        metadata={"help": "regularization term"}
    )
    am_iters: int = field(
        default=80,
        metadata={"help": "number of iterations for alternating minimization"}
    )
    percdamp: float = field(
        default=0.01,
        metadata={"help": "regularization term"}
    )
    gd_lr_init: float = field(
        default=1e-3,
        metadata={"help": "learning_rate for lora components in the matrix decomposition"}
    )
    gd_iters: int = field(
        default=50,
        metadata={"help": "number of iterations for gradient descent within the alternating minimization procedure"}
    )
    hess_diag: bool = field(
        default=False,
        metadata={"help": "Incorporate more diagonal stuff in Hessian"}
    )
    hess_percdamp: float = field(
        default=0.015,
        metadata={"help": "regularization term"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, SparsityArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, sparsity_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, sparsity_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name == "c4":
        misc_utils.swarn(
            f"Using C4 dataset (`dataset_name` "
            f"= {data_args.dataset_name})",
            bg="yellow")
        raw_datasets = load_dataset(
            "allenai/c4",
            data_files={
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
        )
        _wikitext_dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test")
        # Hacks to be consistent with other works' preprocessing.
        wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset["text"])
                ],
            },
        )
        # Hacks to get around the `remove_columns` to be used later.
        wikitext_dataset = (
            wikitext_dataset  # type: ignore
            .add_column(
                name="timestamp",
                column=wikitext_dataset["text"])
            .add_column(
                name="url",
                column=wikitext_dataset["text"])
        )
        raw_datasets["wikitext"] = wikitext_dataset
    elif data_args.dataset_name == "c4-wiki-large":
        misc_utils.swarn(
            f"Using C4+WikiText2 dataset (`dataset_name` "
            f"= {data_args.dataset_name})",
            bg="yellow")
        raw_datasets = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={
                "train": [
                    "en/c4-train.00000-of-01024.json.gz",
                    "en/c4-train.00001-of-01024.json.gz"],
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
        )
        _wikitext_dataset_train = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train")
        _wikitext_dataset_eval = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test")
        # Hacks to be consistent with other works' preprocessing.
        wikitext_dataset_train = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset_train["text"])
                ],
            },
        )
        wikitext_dataset_eval = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset_eval["text"])
                ],
            },
        )
        # Hacks to get around the `remove_columns` to be used later.
        wikitext_dataset_train = (
            wikitext_dataset_train  # type: ignore
            .add_column(
                name="timestamp",
                column=[None for _ in range(len(wikitext_dataset_train["text"]))])
            .add_column(
                name="url",
                column=wikitext_dataset_train["text"])
        )
        wikitext_dataset_eval = (
            wikitext_dataset_eval  # type: ignore
            .add_column(
                name="timestamp",
                column=wikitext_dataset_eval["text"])
            .add_column(
                name="url",
                column=wikitext_dataset_eval["text"])
        )
        raw_datasets["train"] = concatenate_datasets([
            raw_datasets["train"],
            wikitext_dataset_train])
        raw_datasets["wikitext"] = wikitext_dataset_eval
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage)
        # https://github.com/huggingface/transformers/pull/24906
        if model.config.pretraining_tp != 1:
            raise NotImplementedError
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)        
        
        
        
        # Starting the Sparse + LR pruning procedure
        compression_ratio = sparsity_args.compression_ratio
        prunen = sparsity_args.prunen
        prunem = sparsity_args.prunem
        rank_ratio = sparsity_args.rank_ratio
        num_ranks=model_args.lora_num_ranks
        model.seqlen = sparsity_args.seqlen
        nsamples = sparsity_args.nsamples
        percdamp = sparsity_args.percdamp
        am_iters = sparsity_args.am_iters
        gd_lr_init = sparsity_args.gd_lr_init
        gd_iters = sparsity_args.gd_iters
        hess_diag = sparsity_args.hess_diag
        hess_percdamp = sparsity_args.hess_percdamp
        print("compression_ratio", compression_ratio)
        print("prunen", prunen)
        print("prunem", prunem)
        print("rank_ratio", rank_ratio)
        print("num_ranks", model_args.lora_num_ranks)
        print("model.seqlen", model.seqlen)
        print("nsamples", nsamples)
        print("percdamp", percdamp)
        print("am_iters", am_iters)
        print("gd_lr_init", gd_lr_init)
        print("gd_iters", gd_iters)
        print("hess_diag", hess_diag)
        print("hess_percdamp", hess_percdamp)
        model.eval()
                
        dataloader, testenc = get_loaders(
            "c4", nsamples=nsamples, seed=0, model=model, tokenizer=tokenizer, seqlen=model.seqlen
        )
        if model_args.lora_config in ["scale-sparsegpt-gd", "scale-alps-gd"]:
            click.secho(f"Debugging `{model_args.lora_config}`", bg="yellow")
            if prunen != 0:
                sparsity = -1
                if (compression_ratio != -1) and (1 - compression_ratio > prunen / prunem):
                    rank_ratio = (1 - prunen / prunem - compression_ratio) / (1 - compression_ratio)
                else:
                    rank_ratio = -1
            else:
                if rank_ratio > 0.0:
                    sparsity = compression_ratio + rank_ratio - compression_ratio * rank_ratio
                else:
                    sparsity = compression_ratio
            model.cpu()
            
        model = lora_utils.prepare_model_for_lora(
                        model=model,
                        num_ranks=num_ranks,
                        compression_ratio=compression_ratio,
                        rank_ratio=rank_ratio,
                        lora_dropout=model_args.lora_dropout,
                        use_gradient_checkpointing=training_args.gradient_checkpointing)

        if model_args.lora_config in ["scale-sparsegpt-gd", "scale-alps-gd"]:
            start_cpal_time = time.time()
            llama_compressor_gd(model, dataloader, dev="cuda", nsamples=nsamples, n_iters=am_iters, prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, max_iter=gd_iters, lr_init=gd_lr_init, pruning_method=model_args.lora_config, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
            try:
                torch.save(model.state_dict(), f"./{model_args.model_name_or_path}-{sparsity}-{prunen}-{prunem}-lr-{rank_ratio if rank_ratio > 0.0 else num_ranks}.pth")
            except:
                pass
            end_cpal_time = time.time()
            click.secho(f"Time taken for CPAL: {end_cpal_time - start_cpal_time}", bg="yellow")
            print("Time taken for CPAL: ", end_cpal_time - start_cpal_time)
            del dataloader
            torch.cuda.empty_cache()
            gc.collect()                
    else:
        click.secho(f"Full Finetuning", bg="yellow")


    if data_args.dataset_name in ["c4", "c4-wiki-large"]:
        eval_dataset_dict = {
            "c4": eval_dataset,
            "wikitext": lm_datasets["wikitext"]}
        misc_utils.swarn(f"Using evaluation data: {eval_dataset_dict}")
    else:
        eval_dataset_dict = eval_dataset

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset_dict if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=[callback_utils.SaveFullModelCallback],
    )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        model.eval()
        model.to("cuda")
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=training_args.per_device_eval_batch_size) 
        with torch.no_grad():
            zero_shot_tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "rte", "openbookqa", "boolq"]

            ### LM Eval Harness ###
            zs_results = lm_eval.simple_evaluate(hflm, tasks=zero_shot_tasks, num_fewshot=0, batch_size=training_args.per_device_eval_batch_size)[
                'results'
            ]
            print(zs_results)
            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in zs_results.items()}
            acc_avg = helper.calculate_avg_accuracy(zero_shot_tasks, zs_results)
            metric_vals['average_zero_shot'] = round(acc_avg, 4)
            print(metric_vals)
        trainer.log_metrics("oats_zs_results", metric_vals)
        trainer.save_metrics("oats_zs_results", metric_vals)
        
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, nsamples=nsamples, seed=0, model=model, tokenizer=tokenizer, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            dataset_ppl = llama_eval(model, testloader, dev="cuda")
            trainer.log_metrics(f"{dataset}_ppl", {f"{dataset}_ppl": dataset_ppl})
            trainer.save_metrics(f"{dataset}_ppl", {f"{dataset}_ppl": dataset_ppl})

        
    kwargs = {"model": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name


def _evaluation_post_processing(
    prefix: str,
    metrics: Dict[str, Any],
    eval_dataset: datasets.Dataset,
) -> None:

    metrics[f"eval_{prefix}samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics[f"eval_{prefix}loss"])
    except OverflowError:
        perplexity = float("inf")

    metrics[f"{prefix}perplexity"] = perplexity


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()


