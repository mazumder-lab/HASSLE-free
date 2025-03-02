import time

import torch
import torch.nn as nn
import random 
from transformers import AutoTokenizer, LlamaTokenizer
from datasets import load_dataset
from peft.tuners import lora
import numpy as np

from compress import LLM_AM_Compressor

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

def find_lora_layers(module, layers=lora.LoraLayer, name=''):
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_lora_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def llama_compressor_gd(model, dataloader, dev, nsamples=128, n_iters=20, prunen=2, prunem=4, sparsity=None, percdamp=0.01, hess_percdamp=0.05, max_iter=50, lr_init=1e-2, pruning_method="sparsegpt-gd", hess_diag=False):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    # because of lora components
    model = model.base_model.model

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to(dev)
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_lora_layers(layer)

        sequential = [list(full.keys())]
        gpts = {}

        for names in sequential:
            subset = {n: full[n] for n in names}

            for name in subset:
                gpts[name] = LLM_AM_Compressor(subset[name], name)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            start_XTX_time = time.time()
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                hidden_states = inps[j]
                position_ids = torch.arange(0, model.seqlen, device=dev).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
                outs[j] = layer(inps[j].unsqueeze(0), position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()
            print(f"Layer {i} -- name {name}: XTX construction time: ", time.time() - start_XTX_time)

            for name in subset:
                print(i, name)
                print("Pruning ...", flush=True)
                compressor = gpts[name]
                assert isinstance(compressor, LLM_AM_Compressor)
                start_layer_time = time.time()
                if pruning_method == "scale-sparsegpt-gd":
                    e = compressor.scale_sparsegpt_gd(n_iters=n_iters, prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, max_iter=max_iter, lr_init=lr_init, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
                elif pruning_method == "scale-alps-gd":
                    e = compressor.scale_alps_gd(n_iters=n_iters, prunen=prunen, prunem=prunem, sparsity=sparsity, percdamp=percdamp, max_iter=max_iter, lr_init=lr_init, hess_diag=hess_diag, hess_percdamp=hess_percdamp)
                
                end_layer_time = time.time()
                print(f"Layer {i}: {name} time: ", end_layer_time - start_layer_time)

                compressor.free()

        for j in range(nsamples):
            hidden_states = inps[j]
            position_ids = torch.arange(0, model.seqlen, device=dev).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
            outs[j] = layer(inps[j].unsqueeze(0), position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache



def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, tokenizer='', nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
    
    
    
@torch.no_grad()
def llama_eval(model, testenc, dev):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    # because of lora components
    model = model.base_model.model

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.to(dev)
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            hidden_states = inps[j]
            position_ids = torch.arange(0, model.seqlen, device=dev).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
            outs[j] = layer(inps[j].unsqueeze(0), position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl.item()

