"""
turboquant_experiments.py
=========================
A comprehensive, self-contained experiment pipeline for evaluating
Google's TurboQuant (ICLR 2026) on Gemma-2B across real-world tasks.

Experiments:
  E1 - Bit-depth tradeoff:       PolarQuant vs TurboQuant MSE/KL (2,3,4 bits)
  E2 - Distortion rate curve:    Theory vs practice across bit depths
  E3 - Layer sensitivity:        Which layers resist compression best
  E4 - Memory scaling:           KV cache footprint at 512→32768 tokens
  E5 - Latency/Throughput:       Prompt-length sweep with medians and variance
  E6 - Factual QA:               Hard accuracy + semantic similarity (20 Qs)
  E7 - Multi-task eval:          Factual / Reasoning / Coding / Summarisation
  E8 - RAG simulation:           Accuracy when KV cache holds retrieval context
  E9 - Attention entropy:        Does compression flatten attention distributions?
  E10- Mixed-bit precision:      Early layers 4-bit, later layers 3-bit hybrid

Requirements:
  pip install torch transformers tqdm matplotlib numpy

Usage:
  python turboquant_experiments.py

Outputs (in results/<run_timestamp>/):
  results.json          All numeric results
  config.json           Run configuration
  plots/                All matplotlib figures
  *.log                 Full experiment log

Author note:
  Synthetic distortion and layer-sensitivity studies still use k_proj weight
  quantization as a proxy, but generation experiments now use a live runtime
  KV-cache quantization prototype. For production, use fused Triton kernels
  that compress the live KV tensors during the forward pass.
"""

import os
import gc
import json
import math
import re
import random
import time
import sys
import logging
import datetime
import warnings
import inspect
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

class Config:
    # Model
    model_id       = "google/gemma-2b"
    load_dtype     = torch.float16       # float16 mirrors real serving
    device_map     = "auto"
    prefer_full_gpu = True
    low_cpu_mem_usage = True
    baseline_models = [
        {
            "label": "gemma2b_base",
            "model_id": "google/gemma-2b",
            "prompt_style": "plain",
        },
        {
            "label": "gemma2b_it",
            "model_id": "google/gemma-2b-it",
            "prompt_style": "chat",
        },
        {
            "label": "tinyllama_chat",
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt_style": "chat",
        },
    ]
    primary_model_label = "gemma2b_base"
    evaluator_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    evaluator_device = "cpu"
    evaluator_max_length = 256

    # Quantisation sweep
    bits_list      = [2, 3, 4]
    default_bits   = 3                   # paper's recommended operating point
    quantized_bits = 3
    quantization_mode = "runtime_kv"     # runtime_kv or k_proj_proxy

    # Synthetic tensor dimensions (mimics Gemma-2B attention head dims)
    seq_len        = 512
    head_dim       = 256                 # Gemma-2B key/value head dimension

    # Context lengths for memory scaling experiment
    context_lengths = [512, 1024, 4096, 8192, 16384, 32768]

    # Generation budget
    max_new_tokens = 40
    throughput_tokens = 80              # fixed decode steps per measured trial
    throughput_trials = 5
    throughput_warmup_trials = 1
    throughput_alternating_rounds = 2   # round 0: baseline->TQ, round 1: TQ->baseline
    throughput_prompt_token_targets = [64, 512, 1024]
    throughput_headline_prompt_target = 512
    throughput_benchmark_mode = "fixed_step_decode_sweep"
    text_eval_seeds = [0, 1, 2]
    text_eval_do_sample = True
    text_eval_temperature = 0.2
    text_eval_top_p = 0.9
    text_eval_repetition_penalty = 1.05

    # Sensitivity-ranked mixed-bit schedule
    mixed_top_layers = 0.25             # highest-MSE layers get 4-bit
    mixed_bottom_layers = 0.25          # lowest-MSE layers get 2-bit

    # Prompting / methodology
    eval_set_version = "v5_runtime_latency_throughput_sweep"
    proxy_note = (
        "Synthetic distortion and layer-sensitivity experiments still use k_proj "
        "weight quantization as a proxy, but downstream generation experiments now "
        "use a live KV-cache quantization prototype that quantizes cached keys and "
        "values between decoding steps. The prototype stores quantized values in "
        "int8 containers with per-vector scales, so it is behaviorally closer to a "
        "runtime cache path than the old weight proxy, but still not a packed or "
        "fused production kernel."
    )

    # Output
    output_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    fig_dpi        = 200
    resume_latest  = True
    progress_file  = "progress.json"

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────

def setup_logger(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, "experiment.log")
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(stream),
        ],
    )
    return logging.getLogger("turboquant")


def synthetic_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


# ─────────────────────────────────────────────
# TURBOQUANT CORE
# ─────────────────────────────────────────────
#
# def random_hadamard_rotation(x: torch.Tensor) -> torch.Tensor:
#     """
#     Fast Walsh-Hadamard rotation to spread outlier energy uniformly
#     across dimensions before quantisation (PolarQuant Stage 0).
#     Applied dimension-wise to pairs; full Hadamard needs power-of-2 dim.
#     We use a random sign flip + normalised random Gaussian as a practical
#     approximation that mirrors the paper's randomised Hadamard transform.
#     """
#     d = x.shape[-1]
#     torch.manual_seed(42)                         # deterministic rotation
#     R = torch.randn(d, d, device=x.device, dtype=x.dtype)
#     Q, _ = torch.linalg.qr(R)                    # orthonormal basis
#     return x @ Q.T

def random_hadamard_rotation(W):
        device = W.device

        # Step 1: create in float32
        R = torch.randn(W.shape[1], W.shape[1], device=device, dtype=torch.float32)

        # Step 2: QR in float32
        Q, _ = torch.linalg.qr(R)

        # Step 3: cast back to original dtype
        Q = Q.to(W.dtype)

        return W @ Q

def polarquant_encode(x: torch.Tensor, bits: int):
    """
    PolarQuant encoder (Zandieh et al., AISTATS 2026).
    Converts consecutive coordinate pairs to polar form and quantises
    the angle into 2^bits uniform levels.  Radius kept in float.
    """
    # Flatten to (..., D/2, 2)
    pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = pairs[..., 0], pairs[..., 1]

    r     = torch.sqrt(x1 ** 2 + x2 ** 2).clamp(min=1e-9)
    theta = torch.atan2(x2, x1)                  # ∈ [-π, π]

    levels  = 2 ** bits
    theta_q = ((theta + math.pi) / (2 * math.pi) * levels
               ).clamp(0, levels - 1).to(torch.int32)

    return theta_q, r


def polarquant_decode(theta_q: torch.Tensor, r: torch.Tensor, bits: int) -> torch.Tensor:
    """PolarQuant decoder — reconstructs Cartesian coordinates."""
    levels = 2 ** bits
    theta  = theta_q.float() / levels * 2 * math.pi - math.pi
    x1     = r * torch.cos(theta)
    x2     = r * torch.sin(theta)
    out    = torch.stack([x1, x2], dim=-1)
    return out.reshape(*x1.shape[:-1], x1.shape[-1] * 2)


def qjl_encode(residual: torch.Tensor, jl_dim: int):
    """
    Quantised Johnson-Lindenstrauss encoder (QJL, ICLR 2026 Stage 2).
    Projects residual through a random matrix; stores only the sign bit.
    Returns (signs, projection_matrix S).
    """
    d = residual.shape[-1]
    torch.manual_seed(99)
    S    = torch.randn(jl_dim, d, device=residual.device,
                       dtype=residual.dtype) / math.sqrt(jl_dim)
    proj = residual @ S.T                         # (..., jl_dim)
    signs = (proj >= 0).to(torch.float16)         # 1 bit per projection
    return signs, S


# def turboquant_apply(W: torch.Tensor, bits: int, use_rotation: bool = True):
#     """
#     Full TurboQuant pipeline:
#       1. (Optional) Random orthogonal rotation
#       2. PolarQuant encoding
#       3. QJL residual correction
#     Returns (W_hat, signs, S, W_rotated).
#     """
#     W_work = random_hadamard_rotation(W) if use_rotation else W
#
#     theta_q, r = polarquant_encode(W_work, bits)
#     W_hat      = polarquant_decode(theta_q, r, bits)
#
#     residual   = W_work - W_hat
#     jl_dim     = max(8, W.shape[-1] // 4)
#     signs, S   = qjl_encode(residual, jl_dim)
#
#     return W_hat, signs, S, W_work



def turboquant_apply(W, bits=3, use_rotation=True):
    if bits < 2:
        raise ValueError("turboquant_apply expects bits >= 2")

    orig_dtype = W.dtype
    device = W.device

    # FP32 working copy
    W_work = W.to(torch.float32)
    W_rotated = W_work

    if use_rotation:
        d = W_work.shape[1]
        torch.manual_seed(42)

        D = torch.sign(torch.randn(d, device=device))
        perm = torch.randperm(d, device=device)

        W_rotated = (W_work * D)[:, perm]

    # quantize
    qmax = (2 ** (bits - 1)) - 1
    scale = W_rotated.abs().max() / qmax + 1e-8
    W_q = torch.clamp((W_rotated / scale).round(), -qmax, qmax)

    W_hat = W_q * scale

    if use_rotation:
        inv_perm = torch.argsort(perm)
        W_hat = (W_hat[:, inv_perm]) * D

    return (
        W_hat.to(orig_dtype),
        None,
        None,
        W_rotated.to(orig_dtype),
    )


def apply_turboquant_to_k_proj_model(model, bits: int = 3):
    """Apply TurboQuant to each attention k_proj weight matrix in-place."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Expected a decoder model with model.layers")

    with torch.no_grad():
        for layer in model.model.layers:
            k_proj = layer.self_attn.k_proj
            W_hat, _, _, _ = turboquant_apply(k_proj.weight.data, bits=bits)
            k_proj.weight.data.copy_(W_hat.to(k_proj.weight.data.dtype))

    return model


class QuantizedTensorState:
    """Per-vector symmetric quantization state used for cached KV tensors."""

    def __init__(self, q_tensor, scale, orig_dtype, bits):
        self.q_tensor = q_tensor.contiguous()
        self.scale = scale.contiguous()
        self.orig_dtype = orig_dtype
        self.bits = bits
        self.shape = tuple(q_tensor.shape)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, bits: int):
        if tensor is None:
            return None
        if bits < 2:
            raise ValueError("Runtime KV quantization expects bits >= 2")

        work = tensor.detach().to(torch.float32)
        qmax = (2 ** (bits - 1)) - 1
        if qmax < 1:
            qmax = 1
        scale = work.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q_tensor = torch.clamp((work / scale).round(), -qmax, qmax).to(torch.int8)
        scale_dtype = tensor.dtype if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        return cls(q_tensor, scale.to(scale_dtype), tensor.dtype, bits)

    def dequantize(self) -> torch.Tensor:
        return (self.q_tensor.to(torch.float32) * self.scale.to(torch.float32)).to(self.orig_dtype)


class QuantizedKVCache:
    """
    Lightweight legacy-style cache wrapper so generation can keep a compressed
    cache object between decoding steps without needing custom kernels.
    """

    def __init__(self, layers, bits: int):
        self.layers = tuple(layers)
        self.bits = bits

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        for idx in range(len(self.layers)):
            yield self[idx]

    def __getitem__(self, idx):
        k_state, v_state = self.layers[idx]
        return k_state.dequantize(), v_state.dequantize()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers:
            return 0
        return int(self.layers[layer_idx][0].shape[-2])

    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length()

    def get_max_cache_shape(self):
        return None

    def to_legacy_cache(self):
        return tuple(self[idx] for idx in range(len(self.layers)))

    def reorder_cache(self, beam_idx):
        reordered = []
        for k_state, v_state in self.layers:
            reordered.append(
                (
                    QuantizedTensorState(
                        k_state.q_tensor.index_select(0, beam_idx),
                        k_state.scale.index_select(0, beam_idx),
                        k_state.orig_dtype,
                        k_state.bits,
                    ),
                    QuantizedTensorState(
                        v_state.q_tensor.index_select(0, beam_idx),
                        v_state.scale.index_select(0, beam_idx),
                        v_state.orig_dtype,
                        v_state.bits,
                    ),
                )
            )
        return QuantizedKVCache(reordered, self.bits)


def _coerce_cache_to_legacy(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, QuantizedKVCache):
        return past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache()
        except Exception:
            pass
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return tuple(zip(past_key_values.key_cache, past_key_values.value_cache))
    if isinstance(past_key_values, (list, tuple)):
        return tuple(past_key_values)
    return None


def quantize_past_key_values(past_key_values, bits: int):
    legacy_cache = _coerce_cache_to_legacy(past_key_values)
    if not legacy_cache:
        return past_key_values

    quantized_layers = []
    for layer in legacy_cache:
        if not isinstance(layer, (list, tuple)) or len(layer) < 2:
            return past_key_values
        key_cache, value_cache = layer[0], layer[1]
        if not isinstance(key_cache, torch.Tensor) or not isinstance(value_cache, torch.Tensor):
            return past_key_values
        quantized_layers.append(
            (
                QuantizedTensorState.from_tensor(key_cache, bits),
                QuantizedTensorState.from_tensor(value_cache, bits),
            )
        )
    return QuantizedKVCache(quantized_layers, bits)


def dequantize_past_key_values(past_key_values):
    if isinstance(past_key_values, QuantizedKVCache):
        return past_key_values.to_legacy_cache()
    return past_key_values


def quantize_model_output_past(outputs, bits: int):
    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
        outputs.past_key_values = quantize_past_key_values(outputs.past_key_values, bits)
    return outputs


def callable_signature(callable_obj):
    target = getattr(callable_obj, "__func__", callable_obj)
    return inspect.signature(target)


def apply_runtime_kv_quantization(model, bits: int = 3):
    """
    Patch model.forward so cached keys/values are quantized between decoding
    steps. This is a Python-level prototype: it preserves the generation API,
    but it still dequantizes on each reuse and does not use fused kernels.
    """
    if getattr(model, "_turboquant_runtime_kv_enabled", False):
        return model

    original_forward = model.forward
    original_prepare = getattr(model, "prepare_inputs_for_generation", None)

    def wrapped_forward(self, *args, **kwargs):
        if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
            kwargs["past_key_values"] = dequantize_past_key_values(kwargs["past_key_values"])
        outputs = original_forward(*args, **kwargs)
        return quantize_model_output_past(outputs, bits)

    def wrapped_prepare_inputs(self, *args, **kwargs):
        if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
            kwargs["past_key_values"] = dequantize_past_key_values(kwargs["past_key_values"])
        return original_prepare(*args, **kwargs)

    wrapped_forward.__signature__ = callable_signature(original_forward)
    model.forward = types.MethodType(wrapped_forward, model)
    if original_prepare is not None:
        wrapped_prepare_inputs.__signature__ = callable_signature(original_prepare)
        model.prepare_inputs_for_generation = types.MethodType(wrapped_prepare_inputs, model)
    model._turboquant_runtime_kv_enabled = True
    model._turboquant_quantization_mode = "runtime_kv"
    model._turboquant_bits = bits
    return model


def apply_turboquant_to_model(model, bits: int = 3, mode: str = "runtime_kv"):
    if mode == "runtime_kv":
        return apply_runtime_kv_quantization(model, bits=bits)
    if mode == "k_proj_proxy":
        return apply_turboquant_to_k_proj_model(model, bits=bits)
    raise ValueError(f"Unknown quantization mode: {mode}")
# ─────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────

def load_model(cfg: Config, model_id: str = None, apply_tq: bool = False, bits: int = 3):
    model_id = model_id or cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": cfg.load_dtype,
        "low_cpu_mem_usage": cfg.low_cpu_mem_usage,
    }
    if torch.cuda.is_available() and cfg.prefer_full_gpu:
        load_kwargs["device_map"] = {"": 0}
    elif cfg.device_map is not None:
        load_kwargs["device_map"] = cfg.device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except RuntimeError as exc:
        needs_fallback = (
            torch.cuda.is_available()
            and cfg.prefer_full_gpu
            and "out of memory" in str(exc).lower()
        )
        if not needs_fallback:
            raise
        torch.cuda.empty_cache()
        logging.getLogger("turboquant").warning(
            f"Full-GPU load failed for {model_id}; falling back to device_map={cfg.device_map!r}."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=cfg.device_map,
            torch_dtype=cfg.load_dtype,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        )
    model.eval()

    if apply_tq:
        model = apply_turboquant_to_model(
            model,
            bits=bits,
            mode=getattr(cfg, "quantization_mode", "runtime_kv"),
        )

    placement = model_placement_summary(model)
    mode_label = getattr(model, "_turboquant_quantization_mode", "baseline")
    logging.getLogger("turboquant").info(
        f"Loaded {model_id} ({mode_label if apply_tq else 'baseline'}) on {placement}"
    )
    if model_is_offloaded(model):
        logging.getLogger("turboquant").warning(
            f"{model_id} is split across devices ({placement}); generation speed may be distorted."
        )

    return model, tokenizer


def load_similarity_evaluator(cfg: Config):
    """Load a fixed external encoder used for all semantic similarity scoring."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.evaluator_model_id)
    model = AutoModel.from_pretrained(cfg.evaluator_model_id)
    model.to(resolve_device(cfg.evaluator_device))
    model.eval()
    return model, tokenizer


def model_device(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for key in ("model.embed_tokens", "transformer.wte", "model", ""):
            if key in device_map:
                return torch.device(normalize_device_spec(device_map[key]))
        first_device = next(iter(device_map.values()))
        return torch.device(normalize_device_spec(first_device))
    return next(model.parameters()).device


def normalize_device_spec(device_spec) -> str:
    if isinstance(device_spec, torch.device):
        return str(device_spec)
    if isinstance(device_spec, int):
        return f"cuda:{device_spec}"
    if isinstance(device_spec, str):
        return "cpu" if device_spec == "disk" else device_spec
    return str(device_spec)


def model_placement_summary(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        devices = sorted({normalize_device_spec(dev) for dev in device_map.values()})
        return ", ".join(devices)
    return str(next(model.parameters()).device)


def model_is_offloaded(model) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return False
    devices = {normalize_device_spec(dev) for dev in device_map.values()}
    return len(devices) > 1 or any(dev in {"cpu", "disk"} for dev in devices)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_apply_chat_template(tokenizer, messages) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return None
    return None


def format_prompt(task: str, payload: dict, tokenizer, prompt_style: str = "plain") -> str:
    system = (
        "You are a careful assistant. Follow the instruction exactly and stay concise."
    )

    if task in {"factual_qa", "factual"}:
        question = payload.get("question", payload.get("prompt", ""))
        user = (
            "Answer the question with only the shortest correct answer phrase.\n"
            f"Question: {question}\nAnswer:"
        )
    elif task == "rag_qa":
        user = (
            "Use only the provided context. Answer with only the shortest correct answer phrase.\n"
            f"Context: {payload['context']}\n\nQuestion: {payload['question']}\nAnswer:"
        )
    elif task == "reasoning":
        user = (
            "Solve the problem carefully. Give the final answer first, then one short justification.\n"
            f"Problem: {payload['prompt']}\nAnswer:"
        )
    elif task == "coding":
        user = (
            "Answer with concise Python code or one short explanatory sentence when code is not appropriate.\n"
            f"Task: {payload['prompt']}\nResponse:"
        )
    elif task == "summarisation":
        user = (
            "Write a single-sentence summary of the text.\n"
            f"Text: {payload['prompt']}\nSummary:"
        )
    else:
        user = (
            "Respond clearly and concisely.\n"
            f"Prompt: {payload['prompt']}\nResponse:"
        )

    if prompt_style == "chat":
        rendered = maybe_apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if rendered:
            return rendered

    return f"System: {system}\nUser: {user}\nAssistant:"


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    cfg: Config,
    seed: int = None,
    max_new_tokens: int = None,
    do_sample: bool = None,
):
    if seed is not None:
        set_random_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    max_new_tokens = max_new_tokens or cfg.max_new_tokens
    if do_sample is None:
        do_sample = cfg.text_eval_do_sample

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": cfg.text_eval_temperature,
                "top_p": cfg.text_eval_top_p,
                "repetition_penalty": cfg.text_eval_repetition_penalty,
            }
        )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = out[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    full_text = f"{prompt}{completion}"
    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": full_text,
    }


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-9)
    return pooled / denom


def embed_text(model, tokenizer, texts, max_length: int = 256) -> torch.Tensor:
    """Embed text with a fixed external encoder so scores are model-independent."""
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model_device(model))
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pool(outputs.last_hidden_state, inputs["attention_mask"])


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum().item()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9\+\#\.\-/\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_prompt_from_generation(prompt: str, decoded_text: str) -> str:
    idx = decoded_text.find(prompt)
    if idx != -1:
        return decoded_text[idx + len(prompt):].strip()
    return decoded_text.strip()


def extract_answer_text(prompt: str, decoded_text: str, max_words: int = 12) -> str:
    text = strip_prompt_from_generation(prompt, decoded_text) if prompt else decoded_text
    text = re.sub(r"^(answer|response)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    candidate = first_line or text.strip()
    candidate = re.split(r"(?<=[\.\!\?])\s+", candidate, maxsplit=1)[0]
    candidate = " ".join(candidate.split()[:max_words])
    return candidate.strip()


def canonical_answers(item: dict) -> list[str]:
    aliases = [item["a"], *item.get("aliases", [])]
    deduped = []
    seen = set()
    for alias in aliases:
        norm = normalize_text(alias)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(alias)
    return deduped


def strict_answer_accuracy(pred: str, item: dict, prompt: str) -> int:
    candidate = normalize_text(extract_answer_text(prompt, pred))
    if not candidate:
        return 0
    for alias in canonical_answers(item):
        alias_norm = normalize_text(alias)
        if candidate == alias_norm:
            return 1
        if re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", candidate):
            return 1
    return 0


def kv_memory_bytes(ctx_len: int, head_dim: int,
                    n_layers: int, n_heads: int,
                    bytes_per_val: float) -> float:
    """Theoretical KV cache memory in bytes."""
    return 2 * ctx_len * head_dim * n_layers * n_heads * bytes_per_val


def runtime_kv_prototype_bytes(
    ctx_len: int,
    head_dim: int,
    n_layers: int,
    n_heads: int,
    quantized_elem_bytes: int = 1,
    scale_elem_bytes: int = 2,
) -> int:
    """
    Storage used by the current runtime prototype:
    - int8 tensor for quantized K and V values
    - one scale value per cached token-vector for K and for V
    Does not include tiny Python object overhead; it reflects tensor storage.
    """
    q_bytes = 2 * ctx_len * head_dim * n_layers * n_heads * quantized_elem_bytes
    scale_bytes = 2 * ctx_len * n_layers * n_heads * scale_elem_bytes
    return q_bytes + scale_bytes


# ─────────────────────────────────────────────
# ATTENTION METRICS
# ─────────────────────────────────────────────

def attention_mse(q, k, k_hat) -> float:
    orig   = q @ k.T
    approx = q @ k_hat.T
    return ((orig - approx) ** 2).mean().item()


def attention_kl(q, k, k_hat) -> float:
    orig   = F.softmax(q @ k.T,     dim=-1).clamp_min(1e-9)
    approx = F.softmax(q @ k_hat.T, dim=-1).clamp_min(1e-9)
    return F.kl_div(approx.log(), orig, reduction="batchmean").item()


def attention_entropy(q, k) -> float:
    """Shannon entropy of the softmax attention distribution."""
    attn = F.softmax(q @ k.T, dim=-1)
    return -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()


# ─────────────────────────────────────────────
# E1 – BIT-DEPTH TRADEOFF
# ─────────────────────────────────────────────

def experiment_bit_tradeoff(cfg: Config, log) -> dict:
    """
    Compare PolarQuant-only vs full TurboQuant at 2/3/4 bits.
    Measures: attention MSE, attention KL divergence, entropy delta.
    Uses synthetic tensors matching Gemma-2B's head_dim.
    """
    log.info("E1 — Bit-depth tradeoff")
    results = {}
    device = synthetic_device()

    torch.manual_seed(0)
    q = torch.randn(cfg.seq_len, cfg.head_dim, device=device)
    k = torch.randn(cfg.seq_len, cfg.head_dim, device=device)

    for bits in cfg.bits_list:
        # ── PolarQuant only ───────────────────
        tq, r  = polarquant_encode(k, bits)
        k_pq   = polarquant_decode(tq, r, bits)
        pq_mse = attention_mse(q, k, k_pq)
        pq_kl  = attention_kl(q, k, k_pq)

        # ── Full TurboQuant ───────────────────
        k_hat, _, _, _ = turboquant_apply(k, bits)
        tq_mse = attention_mse(q, k, k_hat)
        tq_kl  = attention_kl(q, k, k_hat)

        # ── Entropy analysis ──────────────────
        ent_base = attention_entropy(q, k)
        ent_tq   = attention_entropy(q, k_hat)

        results[bits] = {
            "pq_mse":          pq_mse,
            "tq_mse":          tq_mse,
            "pq_kl":           pq_kl,
            "tq_kl":           tq_kl,
            "qjl_mse_gain_pct": round((pq_mse - tq_mse) / pq_mse * 100, 2),
            "entropy_base":    ent_base,
            "entropy_tq":      ent_tq,
            "entropy_delta":   ent_tq - ent_base,
        }
        log.info(f"  bits={bits}: PQ_MSE={pq_mse:.4f}  TQ_MSE={tq_mse:.4f}"
                 f"  PQ_KL={pq_kl:.4f}  TQ_KL={tq_kl:.4f}"
                 f"  entropy_delta={ent_tq - ent_base:.4f}")

    return results


# ─────────────────────────────────────────────
# E2 – DISTORTION RATE CURVE
# ─────────────────────────────────────────────

def experiment_distortion_curve(cfg: Config, log) -> dict:
    """
    Measure PolarQuant MSE across bits=1..6 on normalised unit vectors.
    Overlays the theoretical rate-distortion lower bound D ≥ 1/4^b.
    """
    log.info("E2 — Distortion rate curve (theory vs practice)")
    results = {}

    for b in range(1, 7):
        torch.manual_seed(1)
        x    = F.normalize(
            torch.randn(2000, cfg.head_dim, device=synthetic_device()), dim=-1
        )
        tq, r = polarquant_encode(x, b)
        x_hat = polarquant_decode(tq, r, b)
        mse   = ((x - x_hat) ** 2).mean().item()
        results[b] = {"mse": mse}

    anchor_mse = results[1]["mse"]
    for b in range(1, 7):
        ref = anchor_mse / (4 ** (b - 1))
        results[b]["reference_curve"] = ref
        results[b]["reference_gap_factor"] = round(results[b]["mse"] / ref, 3)
        log.info(
            f"  bits={b}: MSE={results[b]['mse']:.5f}  "
            f"ref={ref:.5f}  gap={results[b]['mse']/ref:.2f}x"
        )

    return results


# ─────────────────────────────────────────────
# E3 – LAYER SENSITIVITY
# ─────────────────────────────────────────────

def experiment_layer_sensitivity(model, cfg: Config, log) -> dict:
    """
    Compress k_proj of each layer at default_bits and record MSE.
    Reveals which layers are most vulnerable to quantisation error.
    """
    log.info("E3 — Layer sensitivity analysis")
    results = {}
    layers  = model.model.layers

    for i, layer in enumerate(layers):
        W          = layer.self_attn.k_proj.weight.data
        W_hat, _, _, _ = turboquant_apply(W, cfg.default_bits)
        mse        = ((W - W_hat.to(W.dtype)) ** 2).mean().item()
        results[f"layer_{i}"] = {
            "mse":      mse,
            "shape":    list(W.shape),
            "bits":     cfg.default_bits,
        }

    mse_vals = [v["mse"] for v in results.values()]
    ranked_layers = sorted(
        ((idx, vals["mse"]) for idx, vals in enumerate(results.values())),
        key=lambda x: x[1],
        reverse=True,
    )
    results["_summary"] = {
        "max_mse_layer": int(np.argmax(mse_vals)),
        "min_mse_layer": int(np.argmin(mse_vals)),
        "mean_mse":      float(np.mean(mse_vals)),
        "std_mse":       float(np.std(mse_vals)),
        "sensitivity_rank_desc": [idx for idx, _ in ranked_layers],
    }
    log.info(f"  Mean layer MSE: {np.mean(mse_vals):.5f} ± {np.std(mse_vals):.5f}")
    log.info(f"  Most sensitive layer: {np.argmax(mse_vals)}")
    return results


def build_sensitivity_ranked_schedule(layer_sensitivity: dict, cfg: Config) -> dict:
    layer_items = [
        (int(name.split("_")[1]), vals["mse"])
        for name, vals in layer_sensitivity.items()
        if name.startswith("layer_")
    ]
    if not layer_items:
        raise ValueError("Layer sensitivity results are required to build a mixed-bit schedule")

    layer_items.sort(key=lambda item: item[1], reverse=True)
    n_layers = len(layer_items)
    high_count = max(1, int(n_layers * cfg.mixed_top_layers))
    low_count = max(1, int(math.ceil(n_layers * cfg.mixed_bottom_layers)))

    if high_count + low_count >= n_layers:
        low_count = max(1, n_layers - high_count - 1)

    schedule = {idx: 3 for idx, _ in layer_items}
    for idx, _ in layer_items[:high_count]:
        schedule[idx] = 4
    for idx, _ in layer_items[-low_count:]:
        schedule[idx] = 2
    return schedule


# ─────────────────────────────────────────────
# E4 – MEMORY SCALING
# ─────────────────────────────────────────────

def experiment_memory_scaling(cfg: Config, log) -> dict:
    """
    Compare idealized KV-memory targets against the actual storage model used
    by the runtime prototype (int8 values + per-vector scales).
    """
    log.info("E4 — Memory scaling across context lengths")

    # Gemma-2B architecture constants
    N_LAYERS   = 18
    N_KV_HEADS = 1    # Gemma-2B uses MQA (1 KV head per group)
    HEAD_DIM   = 256

    BIT_BYTES = {
        "fp16":  2.0,
        "4bit":  0.5,
        "3bit":  0.375,
        "2bit":  0.25,
    }
    scale_elem_bytes = torch.tensor([], dtype=cfg.load_dtype).element_size()
    results = {}

    for ctx in cfg.context_lengths:
        row = {}
        for label, bpv in BIT_BYTES.items():
            mem_gb = (kv_memory_bytes(ctx, HEAD_DIM, N_LAYERS, N_KV_HEADS, bpv)
                      / 1e9)
            row[label + "_gb"] = round(mem_gb, 4)

        runtime_mem_gb = runtime_kv_prototype_bytes(
            ctx,
            HEAD_DIM,
            N_LAYERS,
            N_KV_HEADS,
            quantized_elem_bytes=1,
            scale_elem_bytes=scale_elem_bytes,
        ) / 1e9
        row["runtime_int8_scale_gb"] = round(runtime_mem_gb, 4)
        row["fp16_vs_3bit_ratio"] = round(row["fp16_gb"] / row["3bit_gb"], 2)
        row["fp16_vs_runtime_ratio"] = round(row["fp16_gb"] / row["runtime_int8_scale_gb"], 2)
        row["runtime_vs_ideal_3bit_ratio"] = round(
            row["runtime_int8_scale_gb"] / row["3bit_gb"], 2
        )
        results[ctx] = row
        log.info(
            f"  ctx={ctx:>6}: fp16={row['fp16_gb']:.3f} GB  "
            f"ideal_3bit={row['3bit_gb']:.3f} GB  "
            f"runtime_proto={row['runtime_int8_scale_gb']:.3f} GB  "
            f"ideal_ratio={row['fp16_vs_3bit_ratio']}x  "
            f"runtime_ratio={row['fp16_vs_runtime_ratio']}x"
        )

    return results


# ─────────────────────────────────────────────
# E5 – THROUGHPUT
# ─────────────────────────────────────────────

def _legacy_experiment_throughput(base_model, tq_model, tokenizer,
                          cfg: Config, log) -> dict:
    """
    Tokens/sec for baseline vs 3-bit TurboQuant on a warm GPU.
    Runs 3 trials each; reports mean ± std.
    """
    log.info("E5 — Throughput benchmark")
    prompt = (
        "Explain the transformer architecture and the role of "
        "self-attention in natural language processing."
    )

    def measure(model, n_trials=3):
        times = []
        for _ in range(n_trials):
            inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.throughput_tokens,
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_gen = out.shape[1] - inputs["input_ids"].shape[1]
            times.append(n_gen / elapsed)
        return float(np.mean(times)), float(np.std(times))

    base_mean, base_std = measure(base_model)
    tq_mean,   tq_std   = measure(tq_model)

    result = {
        "baseline_tps":    round(base_mean, 2),
        "baseline_tps_std":round(base_std,  2),
        "tq3_tps":         round(tq_mean, 2),
        "tq3_tps_std":     round(tq_std,  2),
        "tps_ratio":       round(tq_mean / base_mean, 3),
    }
    log.info(f"  Baseline:   {base_mean:.1f} ± {base_std:.1f} tok/s")
    log.info(f"  TQ-3bit:    {tq_mean:.1f}  ± {tq_std:.1f}  tok/s")
    log.info(f"  Ratio:      {result['tps_ratio']:.3f}")
    return result


# ─────────────────────────────────────────────
# E6 – FACTUAL QA  (20 questions)
# ─────────────────────────────────────────────

FACTUAL_QA = [
    # Geography
    {"q": "What is the capital of India?",        "a": "New Delhi"},
    {"q": "What is the capital of USA?",          "a": "Washington", "aliases": ["Washington DC", "Washington, D.C."]},
    {"q": "What is the capital of France?",       "a": "Paris"},
    {"q": "What is the largest ocean on Earth?",  "a": "Pacific"},
    {"q": "What is the longest river on Earth?",  "a": "Nile"},
    # Science & nature
    {"q": "Who discovered gravity?",              "a": "Newton", "aliases": ["Isaac Newton"]},
    {"q": "What is the chemical formula of water?","a": "H2O"},
    {"q": "What is the largest planet?",          "a": "Jupiter"},
    {"q": "What is the smallest planet?",         "a": "Mercury"},
    {"q": "What is the fastest land animal?",     "a": "Cheetah"},
    # Mathematics
    {"q": "What is 5 + 7?",                       "a": "12"},
    {"q": "What is the square root of 144?",      "a": "12"},
    {"q": "What is 9 multiplied by 8?",           "a": "72"},
    {"q": "What is 2 to the power of 10?",        "a": "1024"},
    # History & culture
    {"q": "Who wrote the Ramayana?",              "a": "Valmiki"},
    {"q": "Who was the first president of USA?",  "a": "Washington", "aliases": ["George Washington"]},
    {"q": "Who invented the telephone?",          "a": "Bell", "aliases": ["Alexander Graham Bell"]},
    # Technology
    {"q": "What does CPU stand for?",             "a": "Central Processing Unit", "aliases": ["CPU", "Central"]},
    {"q": "What does HTML stand for?",            "a": "Hypertext Markup Language", "aliases": ["HTML", "Hypertext"]},
    {"q": "What does RAM stand for?",             "a": "Random Access Memory", "aliases": ["RAM", "Random"]},
    # Additional geography & science
    {"q": "What is the capital of Japan?",        "a": "Tokyo"},
    {"q": "What is the capital of Australia?",    "a": "Canberra"},
    {"q": "What is the largest continent?",       "a": "Asia"},
    {"q": "What is the tallest mountain on Earth?", "a": "Mount Everest", "aliases": ["Everest"]},
    {"q": "What gas do plants absorb during photosynthesis?", "a": "Carbon dioxide", "aliases": ["CO2"]},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "What planet is known for its prominent rings?", "a": "Saturn"},
    {"q": "What is the hardest natural substance?", "a": "Diamond"},
    # Additional history & culture
    {"q": "Who painted the Mona Lisa?",           "a": "Leonardo da Vinci", "aliases": ["Da Vinci", "Leonardo"]},
    {"q": "Who wrote Hamlet?",                    "a": "Shakespeare", "aliases": ["William Shakespeare"]},
    {"q": "Who was the first person to walk on the Moon?", "a": "Armstrong", "aliases": ["Neil Armstrong"]},
    {"q": "In what year did World War II end?",   "a": "1945"},
    # Additional maths
    {"q": "What is the freezing point of water in Celsius?", "a": "0", "aliases": ["0 degrees celsius", "0 c"]},
    {"q": "What is the smallest prime number?",   "a": "2"},
    {"q": "What is 15 divided by 3?",             "a": "5"},
    {"q": "What is 12 multiplied by 12?",         "a": "144"},
    # Additional technology
    {"q": "What does GPU stand for?",             "a": "Graphics Processing Unit", "aliases": ["GPU", "Graphics"]},
    {"q": "What does USB stand for?",             "a": "Universal Serial Bus", "aliases": ["USB", "Universal"]},
    {"q": "What does PDF stand for?",             "a": "Portable Document Format", "aliases": ["PDF", "Portable"]},
    {"q": "What language is primarily spoken in Brazil?", "a": "Portuguese"},
]


def _legacy_experiment_factual_qa(model, tokenizer, evaluator_model, evaluator_tokenizer,
                          cfg: Config, log, label: str = "baseline") -> dict:
    """
    Strict answer accuracy + semantic similarity against ground truth
    using one fixed external encoder.
    """
    log.info(f"E6 — Factual QA [{label}]")
    acc_scores, sim_scores, per_q = [], [], []

    for item in tqdm(FACTUAL_QA, desc=f"factual_qa/{label}", leave=False):
        q, gt = item["q"], item["a"]
        inputs = tokenizer(q, return_tensors="pt").to(model_device(model))

        with torch.no_grad():
            out  = model.generate(**inputs,
                                   max_new_tokens=cfg.max_new_tokens,
                                   do_sample=False)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)

        answer_text = extract_answer_text(q, pred)
        acc = strict_answer_accuracy(pred, item, q)
        emb_gt, emb_pred = embed_text(
            evaluator_model,
            evaluator_tokenizer,
            [gt, answer_text or pred],
            max_length=cfg.evaluator_max_length,
        )
        sim = cosine_sim(emb_gt.unsqueeze(0), emb_pred.unsqueeze(0))

        acc_scores.append(acc)
        sim_scores.append(sim)
        per_q.append({
            "q": q,
            "gt": gt,
            "accepted_answers": canonical_answers(item),
            "pred": pred,
            "answer_text": answer_text,
            "accuracy": acc,
            "similarity": round(sim, 4),
        })

    result = {
        "accuracy":       round(float(np.mean(acc_scores)), 4),
        "avg_similarity": round(float(np.mean(sim_scores)), 4),
        "std_similarity": round(float(np.std(sim_scores)),  4),
        "n_questions":    len(FACTUAL_QA),
        "per_question":   per_q,
    }
    log.info(f"  Accuracy: {result['accuracy']:.2%}   "
             f"Similarity: {result['avg_similarity']:.4f} ± {result['std_similarity']:.4f}")
    return result


# ─────────────────────────────────────────────
# E7 – MULTI-TASK EVALUATION
# ─────────────────────────────────────────────

MULTI_TASK_PROMPTS = {
    "factual": [
        "Who invented the telephone?",
        "What is the boiling point of water in Celsius?",
        "What planet is known as the Red Planet?",
        "Who wrote Pride and Prejudice?",
        "What is the speed of light unit?",
    ],
    "reasoning": [
        "A train travels 60 km in 1 hour. How far in 3.5 hours?",
        "If all cats are animals and some animals are black, are all cats black?",
        "What comes next in the sequence: 2, 4, 8, 16?",
        "Why do shadows change length during the day?",
        "A shop sells apples at 3 for Rs.10. How much for 9 apples?",
    ],
    "coding": [
        "Write a Python function to compute factorial.",
        "How do you reverse a string in Python?",
        "Write a function to check if a number is prime in Python.",
        "Implement binary search in Python.",
        "Write Python code to find the largest element in a list.",
    ],
    "summarisation": [
        "Summarise: AI is transforming industries by automating tasks and improving decisions.",
        "Summarise: Climate change causes rising temperatures, melting ice, and extreme weather.",
        "Summarise: The internet enables global communication and information sharing at scale.",
        "Summarise: Exercise improves physical health and mental wellbeing.",
        "Summarise: Renewable energy sources reduce fossil fuel dependence and lower emissions.",
    ],
}


def _legacy_experiment_multi_task(base_model, tq_model, tokenizer,
                          evaluator_model, evaluator_tokenizer,
                          cfg: Config, log) -> dict:
    """
    For each task: generate responses from both models, measure semantic
    similarity. Identifies which task categories are most affected by compression.
    """
    log.info("E7 — Multi-task evaluation (baseline vs TQ-3bit)")
    results = {}

    for task, prompts in MULTI_TASK_PROMPTS.items():
        sims, examples = [], []
        for p in tqdm(prompts, desc=f"multi_task/{task}", leave=False):
            inputs = tokenizer(p, return_tensors="pt").to(model_device(base_model))

            with torch.no_grad():
                out_base = base_model.generate(**inputs,
                                               max_new_tokens=cfg.max_new_tokens,
                                               do_sample=False)
                out_tq   = tq_model.generate(**inputs,
                                              max_new_tokens=cfg.max_new_tokens,
                                              do_sample=False)

            text_base = tokenizer.decode(out_base[0], skip_special_tokens=True)
            text_tq   = tokenizer.decode(out_tq[0],   skip_special_tokens=True)
            cont_base = strip_prompt_from_generation(p, text_base)
            cont_tq   = strip_prompt_from_generation(p, text_tq)

            emb_base, emb_tq = embed_text(
                evaluator_model,
                evaluator_tokenizer,
                [cont_base or text_base, cont_tq or text_tq],
                max_length=cfg.evaluator_max_length,
            )
            sim = cosine_sim(emb_base.unsqueeze(0), emb_tq.unsqueeze(0))
            sims.append(sim)

            if len(examples) < 2:
                examples.append({
                    "prompt":     p,
                    "baseline":   text_base,
                    "turboquant": text_tq,
                    "baseline_continuation": cont_base,
                    "turboquant_continuation": cont_tq,
                    "similarity": round(sim, 4),
                })

        results[task] = {
            "avg_similarity": round(float(np.mean(sims)),  4),
            "std_similarity": round(float(np.std(sims)),   4),
            "min_similarity": round(float(np.min(sims)),   4),
            "examples":       examples,
        }
        log.info(f"  [{task}] similarity: {results[task]['avg_similarity']:.4f}"
                 f" ± {results[task]['std_similarity']:.4f}")

    return results


# ─────────────────────────────────────────────
# E8 – RAG SIMULATION
# ─────────────────────────────────────────────

RAG_CONTEXTS = [
    {
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
            "It was constructed between 1887 and 1889 as the centrepiece of the "
            "1889 World's Fair. It stands 330 metres tall and was designed by "
            "Gustave Eiffel. It is the most visited monument in the world."
        ),
        "q": "Who designed the Eiffel Tower?",
        "a": "Eiffel",
        "aliases": ["Gustave Eiffel"],
    },
    {
        "context": (
            "Python is a high-level, general-purpose programming language. "
            "It was created by Guido van Rossum and first released in 1991. "
            "Python emphasises code readability and uses indentation to delimit "
            "code blocks. It supports multiple programming paradigms."
        ),
        "q": "Who created Python?",
        "a": "Guido",
        "aliases": ["Guido van Rossum"],
    },
    {
        "context": (
            "The human heart has four chambers: the left atrium, right atrium, "
            "left ventricle, and right ventricle. The left ventricle pumps "
            "oxygenated blood to the body. An adult heart beats approximately "
            "60 to 100 times per minute at rest."
        ),
        "q": "How many chambers does the human heart have?",
        "a": "four",
        "aliases": ["4", "four chambers"],
    },
    {
        "context": (
            "The Apollo 11 mission landed the first humans on the Moon on "
            "July 20, 1969. Neil Armstrong became the first person to walk "
            "on the Moon, followed by Buzz Aldrin. Michael Collins orbited "
            "above in the command module Columbia."
        ),
        "q": "Who was the first person to walk on the Moon?",
        "a": "Armstrong",
        "aliases": ["Neil Armstrong"],
    },
    {
        "context": (
            "Photosynthesis is a process used by plants to convert sunlight "
            "into food. It takes place in the chloroplasts. Plants absorb "
            "carbon dioxide from the air and water from the soil. Oxygen is "
            "released as a by-product. The chemical equation is: "
            "6CO2 + 6H2O + light → C6H12O6 + 6O2."
        ),
        "q": "What gas is released as a by-product of photosynthesis?",
        "a": "Oxygen",
        "aliases": ["O2", "oxygen gas"],
    },
    {
        "context": (
            "New York City is the most populous city in the United States. "
            "It is composed of five boroughs and is often called the Big Apple. "
            "The city is a global centre for finance, media, art, and commerce."
        ),
        "q": "Which city is called the Big Apple?",
        "a": "New York City",
        "aliases": ["New York", "NYC"],
    },
    {
        "context": (
            "The Great Barrier Reef is the world's largest coral reef system. "
            "It is located in the Coral Sea, off the coast of Queensland in "
            "northeastern Australia, and stretches for more than 2,300 kilometres."
        ),
        "q": "Off the coast of which country is the Great Barrier Reef located?",
        "a": "Australia",
        "aliases": ["northeastern Australia"],
    },
    {
        "context": (
            "Mars has two small natural satellites named Phobos and Deimos. "
            "Both moons were discovered in 1877 by the astronomer Asaph Hall. "
            "Mars is often called the Red Planet because of its reddish appearance."
        ),
        "q": "How many natural satellites does Mars have?",
        "a": "two",
        "aliases": ["2"],
    },
    {
        "context": (
            "Insulin was first isolated in 1921 by Frederick Banting and Charles Best "
            "at the University of Toronto. Their work transformed the treatment of diabetes "
            "and led to the Nobel Prize in Physiology or Medicine for Banting in 1923."
        ),
        "q": "Who first isolated insulin with Charles Best?",
        "a": "Banting",
        "aliases": ["Frederick Banting"],
    },
    {
        "context": (
            "Mount Everest is Earth's highest mountain above sea level, reaching "
            "8,848.86 metres. It lies in the Mahalangur Himal sub-range of the Himalayas "
            "on the border between Nepal and the Tibet Autonomous Region of China."
        ),
        "q": "What is Earth's highest mountain above sea level?",
        "a": "Mount Everest",
        "aliases": ["Everest"],
    },
    {
        "context": (
            "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. "
            "It extends from the Arctic Ocean in the north to the Southern Ocean in the south "
            "and covers more than 30 percent of the Earth's surface."
        ),
        "q": "Which ocean is the largest on Earth?",
        "a": "Pacific",
        "aliases": ["Pacific Ocean"],
    },
    {
        "context": (
            "The World Wide Web was invented by Tim Berners-Lee while he was working at CERN. "
            "He proposed the idea in 1989 and built the first web browser and web server soon after. "
            "The web made it easy to navigate documents linked across the internet."
        ),
        "q": "Who invented the World Wide Web?",
        "a": "Berners-Lee",
        "aliases": ["Tim Berners-Lee"],
    },
]


def _legacy_experiment_rag_simulation(base_model, tq_model, tokenizer,
                               cfg: Config, log) -> dict:
    """
    Simulates RAG: inject a factual context paragraph before the question.
    Measures whether compressed model correctly extracts the answer.
    The KV cache holds a non-trivial context before the question token —
    a realistic proxy for compressed-cache retrieval-augmented generation.
    """
    log.info("E8 — RAG simulation (context-grounded factual retrieval)")
    base_results, tq_results, per_q = [], [], []

    for item in tqdm(RAG_CONTEXTS, desc="rag_simulation", leave=False):
        prompt = (
            f"Context: {item['context']}\n\n"
            f"Question: {item['q']}\n"
            f"Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device(base_model))

        with torch.no_grad():
            out_base = base_model.generate(**inputs,
                                           max_new_tokens=20,
                                           do_sample=False)
            out_tq   = tq_model.generate(**inputs,
                                          max_new_tokens=20,
                                          do_sample=False)

        text_base = tokenizer.decode(out_base[0], skip_special_tokens=True)
        text_tq   = tokenizer.decode(out_tq[0],   skip_special_tokens=True)

        answer_base = extract_answer_text(prompt, text_base)
        answer_tq   = extract_answer_text(prompt, text_tq)
        acc_base = strict_answer_accuracy(text_base, item, prompt)
        acc_tq   = strict_answer_accuracy(text_tq,   item, prompt)

        base_results.append(acc_base)
        tq_results.append(acc_tq)
        per_q.append({
            "question":       item["q"],
            "answer":         item["a"],
            "accepted_answers": canonical_answers(item),
            "baseline_pred":  text_base[-80:],
            "tq_pred":        text_tq[-80:],
            "baseline_answer_text": answer_base,
            "tq_answer_text": answer_tq,
            "baseline_acc":   acc_base,
            "tq_acc":         acc_tq,
        })

    return {
        "baseline_accuracy": round(float(np.mean(base_results)), 4),
        "tq_accuracy":       round(float(np.mean(tq_results)),   4),
        "accuracy_delta":    round(float(np.mean(tq_results)) -
                                   float(np.mean(base_results)), 4),
        "n_questions":       len(RAG_CONTEXTS),
        "per_question":      per_q,
    }


# ─────────────────────────────────────────────
# E9 – ATTENTION ENTROPY ANALYSIS
# ─────────────────────────────────────────────

def _experiment_throughput_reloaded_sweep(cfg: Config, log, model_id: str, prompt_style: str = "plain") -> dict:
    """
    Robust latency/throughput benchmark.

    The previous benchmark used one short prompt and one baseline->TQ order.
    This version sweeps prompt/cache lengths, alternates load order across
    rounds, reports medians and percentile bands, and keeps raw trial data so
    reruns can be compared without overreacting to one noisy measurement.
    """
    log.info("E5 - Latency/throughput benchmark sweep")

    prompt_targets = [
        int(target)
        for target in getattr(cfg, "throughput_prompt_token_targets", [64])
    ]
    decode_steps = int(getattr(cfg, "throughput_tokens", 80))
    rounds = max(1, int(getattr(cfg, "throughput_alternating_rounds", 1)))
    headline_target = int(
        getattr(cfg, "throughput_headline_prompt_target", prompt_targets[len(prompt_targets) // 2])
    )
    headline_key = str(min(prompt_targets, key=lambda t: abs(t - headline_target)))

    raw_trials = {
        "baseline": {str(target): [] for target in prompt_targets},
        "tq3": {str(target): [] for target in prompt_targets},
    }

    def aggregate_trials(trials):
        return {
            "prompt_tokens_median": round(float(np.median([t["prompt_tokens"] for t in trials])), 2),
            "decode_tps": rounded_summary(
                summarize_samples([t["decode_tps"] for t in trials]), digits=4
            ),
            "full_tps": rounded_summary(
                summarize_samples([t["full_tps"] for t in trials]), digits=4
            ),
            "prefill_ms": rounded_summary(
                summarize_samples([t["prefill_ms"] for t in trials]), digits=4
            ),
            "ttft_ms": rounded_summary(
                summarize_samples([t["ttft_ms"] for t in trials]), digits=4
            ),
            "decode_ms_per_token": rounded_summary(
                summarize_samples([t["decode_ms_per_token"] for t in trials]), digits=4
            ),
            "raw_trials": trials,
        }

    for round_idx in range(rounds):
        order = [("baseline", False), ("tq3", True)]
        if round_idx % 2 == 1:
            order.reverse()
        log.info(
            f"  Throughput round {round_idx + 1}/{rounds} "
            f"order={','.join(label for label, _ in order)}"
        )

        for variant_label, apply_tq in order:
            model = None
            tokenizer = None
            try:
                model, tokenizer = load_model(
                    cfg,
                    model_id=model_id,
                    apply_tq=apply_tq,
                    bits=cfg.quantized_bits,
                )
                if model_is_offloaded(model):
                    log.warning(
                        f"{variant_label} throughput model is offloaded; timing may be distorted."
                    )

                for prompt_spec in throughput_prompt_suite(tokenizer, prompt_style, cfg):
                    key = str(prompt_spec["target_tokens"])
                    stats = measure_loaded_model_throughput(
                        model,
                        tokenizer,
                        prompt_spec["prompt"],
                        cfg.throughput_trials,
                        cfg,
                        decode_steps=decode_steps,
                    )
                    for trial_idx, trial in enumerate(stats["raw_trials"]):
                        trial_record = dict(trial)
                        trial_record.update(
                            {
                                "round": round_idx,
                                "trial": trial_idx,
                                "variant": variant_label,
                                "target_tokens": prompt_spec["target_tokens"],
                            }
                        )
                        raw_trials[variant_label][key].append(trial_record)

                    log.info(
                        f"    {variant_label:8s} target>={prompt_spec['target_tokens']:4d} "
                        f"actual={stats['prompt_tokens']:4d}  "
                        f"median={stats['decode_tps_median']:.2f} tok/s  "
                        f"lat={stats['decode_ms_per_token_median']:.2f} ms/tok  "
                        f"prefill={stats['prefill_ms_median']:.1f} ms"
                    )
            finally:
                model_ref, tokenizer_ref = model, tokenizer
                model = None
                tokenizer = None
                _cleanup_loaded_models(model_ref, tokenizer_ref)

    per_prompt = {}
    ratios = []
    latency_ratios = []
    for target in prompt_targets:
        key = str(target)
        base = aggregate_trials(raw_trials["baseline"][key])
        tq3 = aggregate_trials(raw_trials["tq3"][key])
        base_tps = base["decode_tps"]["median"]
        tq_tps = tq3["decode_tps"]["median"]
        base_latency = base["decode_ms_per_token"]["median"]
        tq_latency = tq3["decode_ms_per_token"]["median"]
        tps_ratio = float(tq_tps / max(base_tps, 1e-9))
        latency_ratio = float(tq_latency / max(base_latency, 1e-9))
        ratios.append(tps_ratio)
        latency_ratios.append(latency_ratio)
        per_prompt[key] = {
            "target_tokens": target,
            "baseline": base,
            "tq3": tq3,
            "tps_ratio_median": round(tps_ratio, 4),
            "latency_ratio_median": round(latency_ratio, 4),
            "prefill_ratio_median": round(
                tq3["prefill_ms"]["median"] / max(base["prefill_ms"]["median"], 1e-9), 4
            ),
        }

    headline = per_prompt[headline_key]
    base_head = headline["baseline"]
    tq_head = headline["tq3"]
    consistency = {
        "all_prompt_tps_ratios_above_1": bool(all(r > 1.0 for r in ratios)),
        "all_prompt_tps_ratios_below_1": bool(all(r < 1.0 for r in ratios)),
        "ratio_range": [
            round(float(min(ratios)), 4),
            round(float(max(ratios)), 4),
        ],
        "ratio_cv_pct": round(
            float(np.std(ratios) / max(abs(np.mean(ratios)), 1e-12) * 100.0), 2
        ),
    }

    result = {
        "benchmark_mode": getattr(cfg, "throughput_benchmark_mode", "fixed_step_decode_sweep"),
        "decode_steps": decode_steps,
        "prompt_token_targets": prompt_targets,
        "headline_prompt_target": int(headline_key),
        "headline_prompt_tokens": base_head["prompt_tokens_median"],
        "measurement_trials_per_round": int(cfg.throughput_trials),
        "warmup_trials_per_round": int(getattr(cfg, "throughput_warmup_trials", 0)),
        "alternating_rounds": rounds,
        "baseline_tps": round(base_head["decode_tps"]["median"], 2),
        "baseline_tps_std": round(base_head["decode_tps"]["std"], 2),
        "baseline_tps_p10": round(base_head["decode_tps"]["p10"], 2),
        "baseline_tps_p90": round(base_head["decode_tps"]["p90"], 2),
        "baseline_full_tps": round(base_head["full_tps"]["median"], 2),
        "baseline_full_tps_std": round(base_head["full_tps"]["std"], 2),
        "baseline_prefill_ms": round(base_head["prefill_ms"]["median"], 2),
        "baseline_prefill_ms_std": round(base_head["prefill_ms"]["std"], 2),
        "baseline_decode_ms_per_token": round(base_head["decode_ms_per_token"]["median"], 4),
        "tq3_tps": round(tq_head["decode_tps"]["median"], 2),
        "tq3_tps_std": round(tq_head["decode_tps"]["std"], 2),
        "tq3_tps_p10": round(tq_head["decode_tps"]["p10"], 2),
        "tq3_tps_p90": round(tq_head["decode_tps"]["p90"], 2),
        "tq3_full_tps": round(tq_head["full_tps"]["median"], 2),
        "tq3_full_tps_std": round(tq_head["full_tps"]["std"], 2),
        "tq3_prefill_ms": round(tq_head["prefill_ms"]["median"], 2),
        "tq3_prefill_ms_std": round(tq_head["prefill_ms"]["std"], 2),
        "tq3_decode_ms_per_token": round(tq_head["decode_ms_per_token"]["median"], 4),
        "tps_ratio": round(headline["tps_ratio_median"], 3),
        "latency_ratio": round(headline["latency_ratio_median"], 3),
        "aggregate_tps_ratio_median": round(float(np.median(ratios)), 3),
        "aggregate_latency_ratio_median": round(float(np.median(latency_ratios)), 3),
        "consistency": consistency,
        "per_prompt": per_prompt,
    }

    for target in prompt_targets:
        row = per_prompt[str(target)]
        log.info(
            f"  target>={target:4d}: ratio={row['tps_ratio_median']:.3f}  "
            f"base={row['baseline']['decode_tps']['median']:.2f} tok/s  "
            f"tq={row['tq3']['decode_tps']['median']:.2f} tok/s  "
            f"lat_ratio={row['latency_ratio_median']:.3f}"
        )
    log.info(
        f"  Headline target>={headline_key}: ratio={result['tps_ratio']:.3f}  "
        f"aggregate_median_ratio={result['aggregate_tps_ratio_median']:.3f}  "
        f"ratio_range={consistency['ratio_range']}"
    )
    return result


def plot_all(results: dict, run_dir: str, cfg: Config):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    BLUE = "#2563EB"
    ORANGE = "#F97316"
    GREEN = "#16A34A"
    RED = "#DC2626"
    GREY = "#6B7280"

    primary = results["model_runs"][results["primary_model_label"]]

    def save(name):
        plt.savefig(os.path.join(plot_dir, name), dpi=cfg.fig_dpi, bbox_inches="tight")
        plt.close()

    def similarity_ylim(values):
        vmax = max(values)
        return 0.0, min(1.05, max(0.2, vmax + 0.08))

    def normalize_numeric_keyed_dict(d):
        normalized = {}
        for key, value in d.items():
            try:
                normalized[int(key)] = value
            except (TypeError, ValueError):
                normalized[key] = value
        return normalized

    if "bit_tradeoff" in results:
        bt = normalize_numeric_keyed_dict(results["bit_tradeoff"])
        bits = sorted(bt.keys())
        pq_mse = [bt[b]["pq_mse"] for b in bits]
        tq_mse = [bt[b]["tq_mse"] for b in bits]
        pq_kl = [bt[b]["pq_kl"] for b in bits]
        tq_kl = [bt[b]["tq_kl"] for b in bits]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(bits, pq_mse, "o-", color=ORANGE, label="PolarQuant")
        axes[0].plot(bits, tq_mse, "s-", color=BLUE, label="TurboQuant")
        axes[0].set_xlabel("Bits"); axes[0].set_ylabel("Attention MSE")
        axes[0].set_title("Attention MSE vs Bit Depth"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(bits, pq_kl, "o-", color=ORANGE, label="PolarQuant")
        axes[1].plot(bits, tq_kl, "s-", color=BLUE, label="TurboQuant")
        axes[1].set_xlabel("Bits"); axes[1].set_ylabel("KL Divergence")
        axes[1].set_title("Attention KL Divergence vs Bit Depth")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.suptitle("E1 - PolarQuant vs TurboQuant Bit-Depth Tradeoff", fontsize=13)
        plt.tight_layout(); save("E1_bit_tradeoff.png")

    if "distortion_curve" in results:
        dc = normalize_numeric_keyed_dict(results["distortion_curve"])
        bits = sorted(dc.keys())
        ref_key = "reference_curve" if "reference_curve" in dc[bits[0]] else "lower_bound"
        plt.figure(figsize=(7, 4))
        plt.semilogy(bits, [dc[b]["mse"] for b in bits], "o-", color=BLUE, label="PolarQuant (measured)")
        plt.semilogy(bits, [dc[b][ref_key] for b in bits], "--", color=GREY, label="Anchored 4^-b reference")
        plt.xlabel("Bits"); plt.ylabel("MSE (log scale)")
        plt.title("E2 - Rate-Distortion Trend")
        plt.legend(); plt.grid(alpha=0.3)
        save("E2_distortion_curve.png")

    if "layer_sensitivity" in primary:
        ls = {k: v for k, v in primary["layer_sensitivity"].items() if k.startswith("layer_")}
        lnames = sorted(ls.keys(), key=lambda x: int(x.split("_")[1]))
        mse_v = [ls[l]["mse"] for l in lnames]
        x = list(range(len(lnames)))

        plt.figure(figsize=(max(8, len(x) * 0.4), 4))
        plt.bar(x, mse_v, color=BLUE, alpha=0.7, width=0.8)
        plt.xticks(x[::2], [l.replace("layer_", "L") for l in lnames[::2]], rotation=45, fontsize=8)
        plt.xlabel("Transformer Layer"); plt.ylabel("Reconstruction MSE")
        plt.title(f"E3 - Layer Sensitivity (TQ at {cfg.default_bits} bits)")
        plt.axhline(np.mean(mse_v), color=RED, linestyle="--", label=f"Mean={np.mean(mse_v):.4f}")
        plt.legend(); plt.grid(axis="y", alpha=0.3)
        save("E3_layer_sensitivity.png")

    if "memory_scaling" in results:
        ms = normalize_numeric_keyed_dict(results["memory_scaling"])
        ctxs = sorted(ms.keys())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(ctxs, [ms[c]["fp16_gb"] for c in ctxs], "o-", color=RED, label="FP16 baseline")
        axes[0].plot(ctxs, [ms[c]["4bit_gb"] for c in ctxs], "s-", color=ORANGE, label="Ideal 4-bit target")
        axes[0].plot(ctxs, [ms[c]["3bit_gb"] for c in ctxs], "^-", color=BLUE, label="Ideal 3-bit target")
        axes[0].plot(ctxs, [ms[c]["2bit_gb"] for c in ctxs], "D-", color=GREEN, label="Ideal 2-bit target")
        axes[0].set_xlabel("Context Length (tokens)")
        axes[0].set_ylabel("KV Cache Memory (GB)")
        axes[0].set_title("Idealized KV Memory Targets")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].plot(ctxs, [ms[c]["fp16_gb"] for c in ctxs], "o-", color=RED, label="FP16 baseline")
        axes[1].plot(ctxs, [ms[c]["runtime_int8_scale_gb"] for c in ctxs], "s-", color=BLUE, label="Runtime prototype")
        axes[1].plot(ctxs, [ms[c]["3bit_gb"] for c in ctxs], "--", color=GREY, label="Ideal 3-bit target")
        axes[1].set_xlabel("Context Length (tokens)")
        axes[1].set_ylabel("KV Cache Memory (GB)")
        axes[1].set_title("Measured Prototype Storage Model")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        fig.suptitle("E4 - KV Cache Memory: Ideal Target vs Runtime Prototype", fontsize=13)
        plt.tight_layout(); save("E4_memory_scaling.png")

    base = primary["factual_qa_baseline"]
    tq3 = primary["factual_qa_tq3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Baseline", "TQ-3bit"]
    acc = [base["accuracy"], tq3["accuracy"]]
    sim = [base["avg_similarity"], tq3["avg_similarity"]]
    sim_sd = [base["std_similarity"], tq3["std_similarity"]]
    axes[0].bar(labels, acc, color=[BLUE, ORANGE], alpha=0.85)
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Strict Answer Accuracy")
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    axes[1].bar(labels, sim, color=[BLUE, ORANGE], alpha=0.85, yerr=sim_sd, capsize=5)
    sim_low, sim_high = similarity_ylim(sim)
    axes[1].set_ylim(sim_low, sim_high); axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("External Evaluator Similarity (mean +/- std)")
    for i, v in enumerate(sim):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    fig.suptitle("E6 - Factual QA: Baseline vs TurboQuant 3-bit", fontsize=13)
    plt.tight_layout(); save("E6_factual_qa.png")

    mt = primary["multi_task"]
    tasks = list(mt.keys())
    sims = [mt[t]["avg_similarity"] for t in tasks]
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    sims_c = sims + [sims[0]]
    angles_c = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_c, sims_c, "o-", color=BLUE, linewidth=2)
    ax.fill(angles_c, sims_c, alpha=0.2, color=BLUE)
    ax.set_xticks(angles)
    ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=11)
    radar_low, radar_high = similarity_ylim(sims)
    ax.set_ylim(radar_low, max(radar_high, radar_low + 0.2))
    ax.set_title("E7 - Multi-task Semantic Similarity\n(TQ-3bit vs Baseline)", pad=20, fontsize=12)
    save("E7_multitask_radar.png")

    rag = primary["rag_simulation"]
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Baseline", "TQ-3bit"], [rag["baseline_accuracy"], rag["tq_accuracy"]], color=[BLUE, ORANGE], alpha=0.85, width=0.4)
    plt.ylim(0, 1.2); plt.ylabel("Accuracy (RAG context)")
    plt.title("E8 - RAG Simulation: Context Grounded QA")
    for bar, v in zip(bars, [rag["baseline_accuracy"], rag["tq_accuracy"]]):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    save("E8_rag_simulation.png")

    mb = primary["mixed_bit"]
    pl = {k: v for k, v in mb["per_layer"].items()}
    lnames = sorted(pl.keys(), key=lambda x: int(x.split("_")[1]))
    u_mse = [pl[l]["uniform_mse"] for l in lnames]
    m_mse = [pl[l]["mixed_mse"] for l in lnames]
    x = list(range(len(lnames)))
    plt.figure(figsize=(max(8, len(x) * 0.4), 4))
    plt.plot(x, u_mse, "-", color=ORANGE, alpha=0.7, label="Uniform 3-bit")
    plt.plot(x, m_mse, "-", color=BLUE, alpha=0.7, label="Sensitivity-ranked mixed-bit")
    plt.fill_between(x, u_mse, m_mse, where=[u > m for u, m in zip(u_mse, m_mse)], alpha=0.15, color=GREEN, label="Improvement region")
    plt.xlabel("Layer Index"); plt.ylabel("MSE")
    plt.title(f"E10 - Sensitivity-ranked Mixed-bit vs Uniform 3-bit  (delta={mb['mse_improvement_pct']:.1f}%, eff={mb['effective_bits']:.2f} bits)")
    plt.legend(); plt.grid(alpha=0.3)
    save("E10_mixed_bit.png")

    fig = plt.figure(figsize=(15, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)
    suite_labels = list(results["model_runs"].keys())
    suite_runs = [results["model_runs"][label] for label in suite_labels]
    suite_tasks = list(suite_runs[0]["multi_task"].keys())
    x_pos = np.arange(len(suite_labels))
    width = 0.36

    ax0 = fig.add_subplot(gs[0, 0])
    ms_all = normalize_numeric_keyed_dict(results["memory_scaling"])
    if 16384 in ms_all:
        ms16 = ms_all[16384]
        ax0.bar(
            ["FP16", "Ideal 3-bit", "Runtime proto"],
            [ms16["fp16_gb"], ms16["3bit_gb"], ms16["runtime_int8_scale_gb"]],
            color=[RED, BLUE, GREEN],
            alpha=0.8,
        )
        ax0.set_title("KV Cache @ 16K ctx (GB)", fontsize=10)
        ax0.set_ylabel("GB")
        ax0.grid(axis="y", alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "16K context data unavailable", ha="center", va="center", fontsize=10)
        ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    factual_base_vals = [run["factual_qa_baseline"]["accuracy"] for run in suite_runs]
    factual_tq_vals = [run["factual_qa_tq3"]["accuracy"] for run in suite_runs]
    ax1.bar(x_pos - width / 2, factual_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax1.bar(x_pos + width / 2, factual_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Factual Accuracy by Model", fontsize=10)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    heatmap = np.array([[run["multi_task"][task]["avg_similarity"] for task in suite_tasks] for run in suite_runs])
    im = ax2.imshow(heatmap, cmap="Blues", vmin=0.0, vmax=max(0.6, float(np.max(heatmap))))
    ax2.set_xticks(range(len(suite_tasks)))
    ax2.set_xticklabels([task.capitalize() for task in suite_tasks], rotation=25, ha="right", fontsize=8)
    ax2.set_yticks(range(len(suite_labels)))
    ax2.set_yticklabels(suite_labels, fontsize=8)
    ax2.set_title("Multi-task Similarity by Model", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 0])
    rag_base_vals = [run["rag_simulation"]["baseline_accuracy"] for run in suite_runs]
    rag_tq_vals = [run["rag_simulation"]["tq_accuracy"] for run in suite_runs]
    ax3.bar(x_pos - width / 2, rag_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax3.bar(x_pos + width / 2, rag_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("RAG Accuracy by Model", fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    throughput_ratios = [run["throughput"]["tps_ratio"] for run in suite_runs]
    ax4.bar(suite_labels, throughput_ratios, color=GREEN, alpha=0.8)
    ax4.axhline(1.0, color=GREY, linestyle="--", linewidth=1)
    ax4.set_title("Headline Decode Throughput Ratio", fontsize=10)
    ax4.set_ylabel("Ratio")
    ax4.tick_params(axis="x", rotation=15, labelsize=8)
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    mixed_gains = [run["mixed_bit"]["mse_improvement_pct"] for run in suite_runs]
    ax5.bar(suite_labels, mixed_gains, color=BLUE, alpha=0.8)
    ax5.set_title("Mixed-bit MSE Improvement", fontsize=10)
    ax5.set_ylabel("Percent")
    ax5.tick_params(axis="x", rotation=15, labelsize=8)
    ax5.grid(axis="y", alpha=0.3)

    fig.suptitle("TurboQuant Suite Summary - Runtime Prototype Evaluation", fontsize=14, y=1.01)
    save("SUMMARY_dashboard.png")


def experiment_attention_entropy(cfg: Config, log) -> dict:
    """
    Measures how compression affects the sharpness of the attention
    distribution. High entropy = diffuse / flat attention.
    Low entropy = sharp / focused attention.
    A large entropy increase indicates the model is losing focus.
    Sweeps across bits and two vector regimes: random and structured.
    """
    log.info("E9 — Attention entropy analysis")
    results = {}

    for dist in ["random", "structured"]:
        results[dist] = {}
        for bits in cfg.bits_list:
            torch.manual_seed(2)
            device = synthetic_device()
            if dist == "random":
                q = torch.randn(256, cfg.head_dim, device=device)
                k = torch.randn(256, cfg.head_dim, device=device)
            else:
                # Structured: simulate realistic KV vectors with a few
                # dominant directions (closer to what real LLMs produce)
                dominant = torch.randn(8, cfg.head_dim, device=device)
                q = dominant[torch.randint(8, (256,), device=device)] + 0.1 * torch.randn(
                    256, cfg.head_dim, device=device
                )
                k = dominant[torch.randint(8, (256,), device=device)] + 0.1 * torch.randn(
                    256, cfg.head_dim, device=device
                )

            k_hat, _, _, _ = turboquant_apply(k, bits)
            k_hat = k_hat.to(k.dtype)

            ent_base = attention_entropy(q, k)
            ent_tq   = attention_entropy(q, k_hat)

            results[dist][bits] = {
                "entropy_base": round(ent_base, 4),
                "entropy_tq":   round(ent_tq,   4),
                "delta":        round(ent_tq - ent_base, 4),
                "delta_pct":    round((ent_tq - ent_base) / ent_base * 100, 2),
            }
            log.info(f"  [{dist}] bits={bits}: base={ent_base:.4f}  "
                     f"tq={ent_tq:.4f}  Δ={ent_tq-ent_base:+.4f}")

    return results


# ─────────────────────────────────────────────
# E10 – MIXED-BIT PRECISION STRATEGY
# ─────────────────────────────────────────────

def _legacy_experiment_mixed_bit(model, cfg: Config, log) -> dict:
    """
    Tests a mixed-precision strategy motivated by E3 (layer sensitivity):
      - Early layers (bottom 25%): 4-bit compression
      - Middle layers (25-75%):    3-bit compression
      - Late layers (top 25%):     2-bit compression

    Compares against uniform 3-bit on overall MSE and per-block statistics.
    """
    log.info("E10 — Mixed-bit precision strategy")
    layers   = model.model.layers
    n        = len(layers)
    cut1     = n // 4
    cut2     = 3 * n // 4

    uniform_mse, mixed_mse = [], []
    per_layer = {}

    for i, layer in enumerate(layers):
        W = layer.self_attn.k_proj.weight.data

        # Uniform 3-bit
        W_u, _, _, _ = turboquant_apply(W, 3)
        mse_u = ((W - W_u.to(W.dtype)) ** 2).mean().item()

        # Mixed-bit assignment
        if i < cut1:
            b_mix = 4    # early layers: more bits
        elif i < cut2:
            b_mix = 3    # middle
        else:
            b_mix = 2    # late layers: fewer bits (lower sensitivity)

        W_m, _, _, _ = turboquant_apply(W, b_mix)
        mse_m = ((W - W_m.to(W.dtype)) ** 2).mean().item()

        uniform_mse.append(mse_u)
        mixed_mse.append(mse_m)
        per_layer[f"layer_{i}"] = {
            "uniform_bits":  3,
            "mixed_bits":    b_mix,
            "uniform_mse":   round(mse_u, 6),
            "mixed_mse":     round(mse_m, 6),
            "improvement":   round((mse_u - mse_m) / mse_u * 100, 2),
        }

    # Effective bit rate of mixed strategy (weighted average)
    bit_assignments = [4 if i < cut1 else (3 if i < cut2 else 2) for i in range(n)]
    eff_bits = float(np.mean(bit_assignments))

    result = {
        "n_layers":              n,
        "uniform_3bit_mean_mse": round(float(np.mean(uniform_mse)), 6),
        "mixed_mean_mse":        round(float(np.mean(mixed_mse)),   6),
        "mse_improvement_pct":   round(
            (np.mean(uniform_mse) - np.mean(mixed_mse)) / np.mean(uniform_mse) * 100, 2
        ),
        "effective_bits":        round(eff_bits, 2),
        "per_layer":             per_layer,
    }
    log.info(f"  Uniform 3-bit MSE:  {result['uniform_3bit_mean_mse']:.6f}")
    log.info(f"  Mixed-bit MSE:      {result['mixed_mean_mse']:.6f}")
    log.info(f"  Improvement:        {result['mse_improvement_pct']:.1f}%")
    log.info(f"  Effective bit rate: {eff_bits:.2f} bits/coord")
    return result


def build_sensitivity_ranked_schedule(layer_sensitivity: dict, cfg: Config) -> dict:
    layer_items = [
        (int(name.split("_")[1]), vals["mse"])
        for name, vals in layer_sensitivity.items()
        if name.startswith("layer_")
    ]
    if not layer_items:
        raise ValueError("Layer sensitivity results are required to build a mixed-bit schedule")

    layer_items.sort(key=lambda item: item[1], reverse=True)
    n_layers = len(layer_items)
    high_count = max(1, int(n_layers * cfg.mixed_top_layers))
    low_count = max(1, int(math.ceil(n_layers * cfg.mixed_bottom_layers)))
    if high_count + low_count >= n_layers:
        low_count = max(1, n_layers - high_count - 1)

    schedule = {idx: 3 for idx, _ in layer_items}
    for idx, _ in layer_items[:high_count]:
        schedule[idx] = 4
    for idx, _ in layer_items[-low_count:]:
        schedule[idx] = 2
    return schedule


def experiment_mixed_bit(model, layer_sensitivity: dict, cfg: Config, log) -> dict:
    """
    Allocate 4/3/2-bit precision by measured layer sensitivity rather
    than by fixed early/middle/late position.
    """
    log.info("E10 â€” Mixed-bit precision strategy (sensitivity-ranked)")
    layers = model.model.layers
    n = len(layers)
    schedule = build_sensitivity_ranked_schedule(layer_sensitivity, cfg)

    uniform_mse, mixed_mse = [], []
    per_layer = {}
    ranked_layers = layer_sensitivity.get("_summary", {}).get("sensitivity_rank_desc", [])

    for i, layer in enumerate(layers):
        W = layer.self_attn.k_proj.weight.data

        W_u, _, _, _ = turboquant_apply(W, 3)
        mse_u = ((W - W_u.to(W.dtype)) ** 2).mean().item()

        b_mix = schedule[i]
        W_m, _, _, _ = turboquant_apply(W, b_mix)
        mse_m = ((W - W_m.to(W.dtype)) ** 2).mean().item()

        uniform_mse.append(mse_u)
        mixed_mse.append(mse_m)
        per_layer[f"layer_{i}"] = {
            "uniform_bits": 3,
            "mixed_bits": b_mix,
            "uniform_mse": round(mse_u, 6),
            "mixed_mse": round(mse_m, 6),
            "improvement": round((mse_u - mse_m) / max(mse_u, 1e-12) * 100, 2),
            "sensitivity_mse": round(layer_sensitivity[f"layer_{i}"]["mse"], 6),
            "sensitivity_rank": ranked_layers.index(i) if i in ranked_layers else None,
        }

    bit_assignments = [schedule[i] for i in range(n)]
    eff_bits = float(np.mean(bit_assignments))
    result = {
        "n_layers": n,
        "schedule_source": "layer_sensitivity_rank",
        "uniform_3bit_mean_mse": round(float(np.mean(uniform_mse)), 6),
        "mixed_mean_mse": round(float(np.mean(mixed_mse)), 6),
        "mse_improvement_pct": round(
            (np.mean(uniform_mse) - np.mean(mixed_mse)) / np.mean(uniform_mse) * 100, 2
        ),
        "effective_bits": round(eff_bits, 2),
        "high_precision_layers": sorted(idx for idx, bits in schedule.items() if bits == 4),
        "low_precision_layers": sorted(idx for idx, bits in schedule.items() if bits == 2),
        "bit_assignments": bit_assignments,
        "per_layer": per_layer,
    }
    log.info(f"  Uniform 3-bit MSE:  {result['uniform_3bit_mean_mse']:.6f}")
    log.info(f"  Mixed-bit MSE:      {result['mixed_mean_mse']:.6f}")
    log.info(f"  Improvement:        {result['mse_improvement_pct']:.1f}%")
    log.info(f"  Effective bit rate: {eff_bits:.2f} bits/coord")
    log.info(f"  4-bit layers:       {result['high_precision_layers']}")
    log.info(f"  2-bit layers:       {result['low_precision_layers']}")
    return result


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def _legacy_plot_all(results: dict, run_dir: str, cfg: Config):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    BLUE   = "#2563EB"
    ORANGE = "#F97316"
    GREEN  = "#16A34A"
    RED    = "#DC2626"
    GREY   = "#6B7280"

    def save(name):
        plt.savefig(os.path.join(plot_dir, name),
                    dpi=cfg.fig_dpi, bbox_inches="tight")
        plt.close()

    # ── Plot 1: Bit-depth tradeoff ────────────
    if "bit_tradeoff" in results:
        bt = results["bit_tradeoff"]
        bits = sorted(bt.keys())
        pq_mse = [bt[b]["pq_mse"] for b in bits]
        tq_mse = [bt[b]["tq_mse"] for b in bits]
        pq_kl  = [bt[b]["pq_kl"]  for b in bits]
        tq_kl  = [bt[b]["tq_kl"]  for b in bits]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(bits, pq_mse, "o-", color=ORANGE, label="PolarQuant")
        axes[0].plot(bits, tq_mse, "s-", color=BLUE,   label="TurboQuant")
        axes[0].set_xlabel("Bits"); axes[0].set_ylabel("Attention MSE")
        axes[0].set_title("Attention MSE vs Bit Depth"); axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(bits, pq_kl, "o-", color=ORANGE, label="PolarQuant")
        axes[1].plot(bits, tq_kl, "s-", color=BLUE,   label="TurboQuant")
        axes[1].set_xlabel("Bits"); axes[1].set_ylabel("KL Divergence")
        axes[1].set_title("Attention KL Divergence vs Bit Depth")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.suptitle("E1 — PolarQuant vs TurboQuant Bit-Depth Tradeoff", fontsize=13)
        plt.tight_layout(); save("E1_bit_tradeoff.png")

    # ── Plot 2: Distortion rate curve ─────────
    if "distortion_curve" in results:
        dc   = results["distortion_curve"]
        bits = sorted(dc.keys())
        mse  = [dc[b]["mse"]         for b in bits]
        lb   = [dc[b]["lower_bound"] for b in bits]

        plt.figure(figsize=(7, 4))
        plt.semilogy(bits, mse, "o-", color=BLUE,   label="PolarQuant (measured)")
        plt.semilogy(bits, lb,  "--", color=GREY,   label="Theoretical lower bound")
        plt.xlabel("Bits"); plt.ylabel("MSE (log scale)")
        plt.title("E2 — Rate-Distortion: Theory vs Practice")
        plt.legend(); plt.grid(alpha=0.3)
        save("E2_distortion_curve.png")

    # ── Plot 3: Layer sensitivity ─────────────
    if "layer_sensitivity" in results:
        ls  = {k: v for k, v in results["layer_sensitivity"].items()
               if k.startswith("layer_")}
        lnames = sorted(ls.keys(), key=lambda x: int(x.split("_")[1]))
        mse_v  = [ls[l]["mse"] for l in lnames]
        x      = list(range(len(lnames)))

        plt.figure(figsize=(max(8, len(x) * 0.4), 4))
        plt.bar(x, mse_v, color=BLUE, alpha=0.7, width=0.8)
        plt.xticks(x[::2], [l.replace("layer_", "L") for l in lnames[::2]],
                   rotation=45, fontsize=8)
        plt.xlabel("Transformer Layer"); plt.ylabel("Reconstruction MSE")
        plt.title(f"E3 — Layer Sensitivity (TQ at {cfg.default_bits} bits)")
        plt.axhline(np.mean(mse_v), color=RED, linestyle="--",
                    label=f"Mean={np.mean(mse_v):.4f}")
        plt.legend(); plt.grid(axis="y", alpha=0.3)
        save("E3_layer_sensitivity.png")

    # ── Plot 4: Memory scaling ────────────────
    if "memory_scaling" in results:
        ms   = results["memory_scaling"]
        ctxs = sorted(ms.keys())
        fp16 = [ms[c]["fp16_gb"] for c in ctxs]
        b4   = [ms[c]["4bit_gb"] for c in ctxs]
        b3   = [ms[c]["3bit_gb"] for c in ctxs]
        b2   = [ms[c]["2bit_gb"] for c in ctxs]

        plt.figure(figsize=(8, 4))
        plt.plot(ctxs, fp16, "o-", color=RED,    label="FP16 (baseline)")
        plt.plot(ctxs, b4,   "s-", color=ORANGE, label="4-bit TurboQuant")
        plt.plot(ctxs, b3,   "^-", color=BLUE,   label="3-bit TurboQuant")
        plt.plot(ctxs, b2,   "D-", color=GREEN,  label="2-bit TurboQuant")
        plt.xlabel("Context Length (tokens)"); plt.ylabel("KV Cache Memory (GB)")
        plt.title("E4 — KV Cache Memory Scaling (Gemma-2B, 18 layers)")
        plt.legend(); plt.grid(alpha=0.3)
        save("E4_memory_scaling.png")

    # ── Plot 5: Factual QA comparison ─────────
    if "factual_qa_baseline" in results and "factual_qa_tq3" in results:
        base = results["factual_qa_baseline"]
        tq3  = results["factual_qa_tq3"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = ["Baseline", "TQ-3bit"]
        acc    = [base["accuracy"],       tq3["accuracy"]]
        sim    = [base["avg_similarity"], tq3["avg_similarity"]]
        sim_sd = [base["std_similarity"], tq3["std_similarity"]]

        axes[0].bar(labels, acc, color=[BLUE, ORANGE], alpha=0.85)
        axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Strict Answer Accuracy")
        for i, v in enumerate(acc):
            axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)

        axes[1].bar(labels, sim, color=[BLUE, ORANGE], alpha=0.85,
                    yerr=sim_sd, capsize=5)
        axes[1].set_ylim(0.0, 1.05); axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("External Evaluator Similarity (mean ± std)")
        for i, v in enumerate(sim):
            axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)

        fig.suptitle("E6 — Factual QA: Baseline vs TurboQuant 3-bit", fontsize=13)
        plt.tight_layout(); save("E6_factual_qa.png")

    # ── Plot 6: Multi-task radar ───────────────
    if "multi_task" in results:
        mt    = results["multi_task"]
        tasks = list(mt.keys())
        sims  = [mt[t]["avg_similarity"] for t in tasks]

        angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
        sims_c = sims + [sims[0]]; angles_c = angles + [angles[0]]
        tasks_c = tasks + [tasks[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.plot(angles_c, sims_c, "o-", color=BLUE, linewidth=2)
        ax.fill(angles_c, sims_c, alpha=0.2, color=BLUE)
        ax.set_xticks(angles)
        ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=11)
        ax.set_ylim(0.7, 1.0)
        ax.set_title("E7 — Multi-task Semantic Similarity\n(TQ-3bit vs Baseline)",
                     pad=20, fontsize=12)
        save("E7_multitask_radar.png")

    # ── Plot 7: RAG accuracy ──────────────────
    if "rag_simulation" in results:
        rag = results["rag_simulation"]
        labels = ["Baseline", "TQ-3bit"]
        accs   = [rag["baseline_accuracy"], rag["tq_accuracy"]]

        plt.figure(figsize=(5, 4))
        bars = plt.bar(labels, accs, color=[BLUE, ORANGE], alpha=0.85, width=0.4)
        plt.ylim(0, 1.2); plt.ylabel("Accuracy (RAG context)")
        plt.title("E8 — RAG Simulation: Context Grounded QA")
        for bar, v in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                     f"{v:.0%}", ha="center", fontsize=12)
        plt.grid(axis="y", alpha=0.3)
        save("E8_rag_simulation.png")

    # ── Plot 8: Mixed-bit vs uniform ──────────
    if "mixed_bit" in results:
        mb = results["mixed_bit"]
        pl = {k: v for k, v in mb["per_layer"].items()}
        lnames = sorted(pl.keys(), key=lambda x: int(x.split("_")[1]))
        u_mse  = [pl[l]["uniform_mse"] for l in lnames]
        m_mse  = [pl[l]["mixed_mse"]   for l in lnames]
        x      = list(range(len(lnames)))

        plt.figure(figsize=(max(8, len(x) * 0.4), 4))
        plt.plot(x, u_mse, "-",  color=ORANGE, alpha=0.7, label="Uniform 3-bit")
        plt.plot(x, m_mse, "-",  color=BLUE,   alpha=0.7, label="Sensitivity-ranked mixed-bit")
        plt.fill_between(x, u_mse, m_mse,
                         where=[u > m for u, m in zip(u_mse, m_mse)],
                         alpha=0.15, color=GREEN, label="Improvement region")
        plt.xlabel("Layer Index"); plt.ylabel("MSE")
        plt.title(f"E10 — Sensitivity-ranked Mixed-bit vs Uniform 3-bit  "
                  f"(Δ={mb['mse_improvement_pct']:.1f}%,  "
                  f"eff={mb['effective_bits']:.2f} bits)")
        plt.legend(); plt.grid(alpha=0.3)
        save("E10_mixed_bit.png")

    # ── Plot 9: Summary dashboard ─────────────
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # Memory bar at 16K context
    ax0 = fig.add_subplot(gs[0, 0])
    if "memory_scaling" in results and 16384 in results["memory_scaling"]:
        ms16 = results["memory_scaling"][16384]
        mem_labels = ["FP16", "4-bit", "3-bit", "2-bit"]
        mem_vals   = [ms16["fp16_gb"], ms16["4bit_gb"],
                      ms16["3bit_gb"], ms16["2bit_gb"]]
        ax0.bar(mem_labels, mem_vals,
                color=[RED, ORANGE, BLUE, GREEN], alpha=0.8)
        ax0.set_title("KV Cache @ 16K ctx (GB)", fontsize=10)
        ax0.set_ylabel("GB"); ax0.grid(axis="y", alpha=0.3)

    # Factual accuracy delta
    ax1 = fig.add_subplot(gs[0, 1])
    if "factual_qa_baseline" in results and "factual_qa_tq3" in results:
        base_a = results["factual_qa_baseline"]["accuracy"]
        tq3_a  = results["factual_qa_tq3"]["accuracy"]
        ax1.bar(["Baseline", "TQ-3bit"], [base_a, tq3_a],
                color=[BLUE, ORANGE], alpha=0.8)
        ax1.set_ylim(0, 1.1); ax1.set_title("Factual Accuracy", fontsize=10)
        ax1.set_ylabel("Accuracy"); ax1.grid(axis="y", alpha=0.3)

    # Multi-task similarity bar
    ax2 = fig.add_subplot(gs[0, 2])
    if "multi_task" in results:
        mt     = results["multi_task"]
        tasks  = list(mt.keys())
        s_vals = [mt[t]["avg_similarity"] for t in tasks]
        s_errs = [mt[t]["std_similarity"] for t in tasks]
        ax2.bar(range(len(tasks)), s_vals, yerr=s_errs,
                color=BLUE, alpha=0.7, capsize=4)
        ax2.set_xticks(range(len(tasks)))
        ax2.set_xticklabels([t[:5] for t in tasks], fontsize=8)
        ax2.set_ylim(0.7, 1.05)
        ax2.set_title("Multi-task Similarity (TQ-3bit)", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

    # Distortion curve
    ax3 = fig.add_subplot(gs[1, 0])
    if "distortion_curve" in results:
        dc   = results["distortion_curve"]
        bits = sorted(dc.keys())
        ax3.semilogy(bits, [dc[b]["mse"]         for b in bits],
                     "o-", color=BLUE,   label="Measured")
        ax3.semilogy(bits, [dc[b]["lower_bound"] for b in bits],
                     "--", color=GREY,   label="Lower bound")
        ax3.set_title("Rate-Distortion Curve", fontsize=10)
        ax3.set_xlabel("Bits"); ax3.set_ylabel("MSE (log)")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # RAG
    ax4 = fig.add_subplot(gs[1, 1])
    if "rag_simulation" in results:
        rag = results["rag_simulation"]
        ax4.bar(["Baseline", "TQ-3bit"],
                [rag["baseline_accuracy"], rag["tq_accuracy"]],
                color=[BLUE, ORANGE], alpha=0.8)
        ax4.set_ylim(0, 1.2); ax4.set_title("RAG Accuracy", fontsize=10)
        ax4.grid(axis="y", alpha=0.3)

    # Mixed-bit improvement
    ax5 = fig.add_subplot(gs[1, 2])
    if "mixed_bit" in results:
        mb = results["mixed_bit"]
        ax5.bar(["Uniform 3-bit", "Mixed-bit"],
                [mb["uniform_3bit_mean_mse"], mb["mixed_mean_mse"]],
                color=[ORANGE, BLUE], alpha=0.8)
        ax5.set_title(
            f"Mixed-bit MSE\n(eff {mb['effective_bits']:.2f} bits)", fontsize=10
        )
        ax5.set_ylabel("Mean MSE"); ax5.grid(axis="y", alpha=0.3)

    fig.suptitle("TurboQuant Experiment Summary — Gemma-2B", fontsize=14, y=1.01)
    save("SUMMARY_dashboard.png")


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def _legacy_run_all():
    cfg = Config()

    # Set up output directory
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, f"turboquant_suite_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    log = setup_logger(run_dir)
    log.info("=" * 60)
    log.info("TurboQuant Experiment Pipeline — Gemma-2B")
    log.info(f"Output directory: {run_dir}")
    log.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log.info("=" * 60)

    results = {}

    # ── Algorithm-only experiments (no model load) ──
    log.info("\n[PHASE 1] Algorithm characterisation (synthetic tensors)")
    results["bit_tradeoff"]    = experiment_bit_tradeoff(cfg, log)
    results["distortion_curve"]= experiment_distortion_curve(cfg, log)
    results["attention_entropy"]= experiment_attention_entropy(cfg, log)
    results["memory_scaling"]  = experiment_memory_scaling(cfg, log)

    # ── Model-dependent experiments ──────────────
    log.info("\n[PHASE 2] Loading baseline model")
    base_model, tokenizer = load_model(cfg, apply_tq=False)
    evaluator_model, evaluator_tokenizer = load_similarity_evaluator(cfg)

    log.info("\n[PHASE 3] Layer sensitivity & mixed-bit (baseline model)")
    results["layer_sensitivity"] = experiment_layer_sensitivity(base_model, cfg, log)
    results["mixed_bit"]         = experiment_mixed_bit(
        base_model, results["layer_sensitivity"], cfg, log
    )

    log.info("\n[PHASE 4] Baseline task evaluation")
    results["factual_qa_baseline"] = experiment_factual_qa(
        base_model, tokenizer, evaluator_model, evaluator_tokenizer, cfg, log, label="baseline"
    )

    # ── Load TurboQuant 3-bit model ──────────────
    log.info("\n[PHASE 5] Loading TurboQuant 3-bit model")
    tq_model, _ = load_model(cfg, apply_tq=True, bits=3)

    results["factual_qa_tq3"] = experiment_factual_qa(
        tq_model, tokenizer, evaluator_model, evaluator_tokenizer, cfg, log, label="tq_3bit"
    )

    log.info("\n[PHASE 6] Throughput benchmark")
    results["throughput"] = experiment_throughput(
        base_model, tq_model, tokenizer, cfg, log
    )

    log.info("\n[PHASE 7] Multi-task evaluation")
    results["multi_task"] = experiment_multi_task(
        base_model, tq_model, tokenizer, evaluator_model, evaluator_tokenizer, cfg, log
    )

    log.info("\n[PHASE 8] RAG simulation")
    results["rag_simulation"] = experiment_rag_simulation(
        base_model, tq_model, tokenizer, cfg, log
    )

    # ── Save results ─────────────────────────────
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "model_id":       cfg.model_id,
            "evaluator_model_id": cfg.evaluator_model_id,
            "bits_list":      cfg.bits_list,
            "default_bits":   cfg.default_bits,
            "head_dim":       cfg.head_dim,
            "seq_len":        cfg.seq_len,
            "context_lengths":cfg.context_lengths,
            "max_new_tokens": cfg.max_new_tokens,
            "mixed_top_layers": cfg.mixed_top_layers,
            "mixed_bottom_layers": cfg.mixed_bottom_layers,
            "timestamp":      ts,
        }, f, indent=2)

    # ── Generate all plots ───────────────────────
    log.info("\n[PHASE 9] Generating plots")
    plot_all(results, run_dir, cfg)

    # ── Print summary ────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT SUMMARY")
    log.info("=" * 60)

    if "factual_qa_baseline" in results and "factual_qa_tq3" in results:
        b = results["factual_qa_baseline"]
        t = results["factual_qa_tq3"]
        log.info(f"Factual QA Accuracy:     baseline={b['accuracy']:.0%}  "
                 f"tq3={t['accuracy']:.0%}  "
                 f"delta={t['accuracy']-b['accuracy']:+.0%}")
        log.info(f"Factual QA Similarity:   baseline={b['avg_similarity']:.4f}  "
                 f"tq3={t['avg_similarity']:.4f}")

    if "rag_simulation" in results:
        r = results["rag_simulation"]
        log.info(f"RAG Accuracy:            baseline={r['baseline_accuracy']:.0%}  "
                 f"tq3={r['tq_accuracy']:.0%}  "
                 f"delta={r['accuracy_delta']:+.2%}")

    if "throughput" in results:
        tp = results["throughput"]
        log.info(f"Throughput:              baseline={tp['baseline_tps']} tok/s  "
                 f"tq3={tp['tq3_tps']} tok/s  ratio={tp['tps_ratio']}")

    if "memory_scaling" in results and 32768 in results["memory_scaling"]:
        ms = results["memory_scaling"][32768]
        log.info(
            f"Memory @ 32K ctx:        fp16={ms['fp16_gb']:.3f} GB  "
            f"ideal_3bit={ms['3bit_gb']:.3f} GB  "
            f"runtime_proto={ms['runtime_int8_scale_gb']:.3f} GB  "
            f"ideal_ratio={ms['fp16_vs_3bit_ratio']}x  "
            f"runtime_ratio={ms['fp16_vs_runtime_ratio']}x"
        )

    if "mixed_bit" in results:
        mb = results["mixed_bit"]
        log.info(f"Mixed-bit improvement:   {mb['mse_improvement_pct']:.1f}% MSE "
                 f"at {mb['effective_bits']:.2f} eff bits")

    log.info(f"\nAll outputs saved to: {run_dir}")
    log.info("=" * 60)

    return results, run_dir


def experiment_throughput(base_model, tq_model, tokenizer,
                          cfg: Config, log, prompt_style: str = "plain") -> dict:
    """Tokens/sec for baseline vs quantized model on a standard prompt."""
    log.info("E5 — Throughput benchmark")
    if model_is_offloaded(base_model) or model_is_offloaded(tq_model):
        log.warning(
            "Throughput models are offloaded or split across devices; timing may be distorted."
        )
    prompt = format_prompt(
        "general",
        {
            "prompt": (
                "Explain the transformer architecture and the role of "
                "self-attention in natural language processing."
            )
        },
        tokenizer,
        prompt_style=prompt_style,
    )

    def measure(model, n_trials):
        times = []
        for _ in range(n_trials):
            inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.throughput_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_gen = out.shape[1] - inputs["input_ids"].shape[1]
            times.append(n_gen / elapsed)
        return float(np.mean(times)), float(np.std(times))

    base_mean, base_std = measure(base_model, cfg.throughput_trials)
    tq_mean, tq_std = measure(tq_model, cfg.throughput_trials)

    result = {
        "baseline_tps": round(base_mean, 2),
        "baseline_tps_std": round(base_std, 2),
        "tq3_tps": round(tq_mean, 2),
        "tq3_tps_std": round(tq_std, 2),
        "tps_ratio": round(tq_mean / base_mean, 3),
    }
    log.info(f"  Baseline:   {base_mean:.1f} ± {base_std:.1f} tok/s")
    log.info(f"  TQ-3bit:    {tq_mean:.1f} ± {tq_std:.1f} tok/s")
    log.info(f"  Ratio:      {result['tps_ratio']:.3f}")
    return result


def throughput_prompt(tokenizer, prompt_style: str = "plain") -> str:
    return format_prompt(
        "general",
        {
            "prompt": (
                "Explain the transformer architecture and the role of "
                "self-attention in natural language processing."
            )
        },
        tokenizer,
        prompt_style=prompt_style,
    )


def _sync_cuda_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_samples(values) -> dict:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
            "cv_pct": None,
            "n": 0,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv_pct": float(std / max(abs(mean), 1e-12) * 100.0),
        "n": int(arr.size),
    }


def rounded_summary(summary: dict, digits: int = 4) -> dict:
    rounded = {}
    for key, value in summary.items():
        if isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), digits)
        else:
            rounded[key] = value
    return rounded


def make_target_token_prompt(tokenizer, prompt_style: str, target_tokens: int) -> dict:
    """
    Build a deterministic prompt that lands at or above the requested token count.
    We keep the text natural so attention/cache behavior is closer to serving than
    repeated single-token padding, while still being reproducible across reruns.
    """
    seed = (
        "Explain how transformer attention uses the key value cache during "
        "autoregressive inference, then connect that explanation to practical "
        "latency, throughput, and memory behavior in long context serving. "
    )
    repeats = max(1, int(math.ceil(target_tokens / 24)))
    for _ in range(12):
        payload = (seed * repeats).strip()
        prompt = format_prompt(
            "general",
            {"prompt": payload},
            tokenizer,
            prompt_style=prompt_style,
        )
        token_count = int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])
        if token_count >= target_tokens:
            return {
                "target_tokens": int(target_tokens),
                "prompt_tokens": token_count,
                "prompt": prompt,
            }
        repeats *= 2
    return {
        "target_tokens": int(target_tokens),
        "prompt_tokens": token_count,
        "prompt": prompt,
    }


def throughput_prompt_suite(tokenizer, prompt_style: str, cfg: Config):
    return [
        make_target_token_prompt(tokenizer, prompt_style, target)
        for target in getattr(cfg, "throughput_prompt_token_targets", [64])
    ]


def fixed_step_decode_trial(
    model,
    tokenizer,
    prompt: str,
    cfg: Config,
    decode_steps=None,
):
    """
    Deterministic fixed-step greedy decode benchmark.
    Measures prefill and decode separately and avoids `generate()` early-stop
    / sampling behavior so every model executes the same number of decode steps.
    """
    decode_steps = int(decode_steps or cfg.throughput_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    _sync_cuda_if_needed()
    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        _sync_cuda_if_needed()
        prefill_elapsed = time.perf_counter() - t0

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = attention_mask

        _sync_cuda_if_needed()
        t1 = time.perf_counter()
        for _ in range(decode_steps):
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    current_attention_mask.new_ones((current_attention_mask.shape[0], 1)),
                ],
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync_cuda_if_needed()
        decode_elapsed = time.perf_counter() - t1

    prefill_ms = prefill_elapsed * 1000.0
    decode_ms = decode_elapsed * 1000.0
    decode_ms_per_token = decode_ms / max(decode_steps, 1)
    return {
        "prefill_ms": prefill_ms,
        "ttft_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "decode_tps": decode_steps / max(decode_elapsed, 1e-9),
        "full_tps": decode_steps / max(prefill_elapsed + decode_elapsed, 1e-9),
        "decode_steps": decode_steps,
        "prompt_tokens": int(input_ids.shape[1]),
    }


def measure_loaded_model_throughput(
    model,
    tokenizer,
    prompt: str,
    n_trials: int,
    cfg: Config,
    decode_steps=None,
):
    for _ in range(getattr(cfg, "throughput_warmup_trials", 0)):
        fixed_step_decode_trial(model, tokenizer, prompt, cfg, decode_steps=decode_steps)

    trials = [
        fixed_step_decode_trial(model, tokenizer, prompt, cfg, decode_steps=decode_steps)
        for _ in range(n_trials)
    ]
    decode_tps = [trial["decode_tps"] for trial in trials]
    full_tps = [trial["full_tps"] for trial in trials]
    prefill_ms = [trial["prefill_ms"] for trial in trials]
    ttft_ms = [trial["ttft_ms"] for trial in trials]
    decode_ms_per_token = [trial["decode_ms_per_token"] for trial in trials]
    return {
        "decode_tps_mean": float(np.mean(decode_tps)),
        "decode_tps_std": float(np.std(decode_tps)),
        "decode_tps_median": float(np.median(decode_tps)),
        "decode_tps_p10": float(np.percentile(decode_tps, 10)),
        "decode_tps_p90": float(np.percentile(decode_tps, 90)),
        "full_tps_mean": float(np.mean(full_tps)),
        "full_tps_std": float(np.std(full_tps)),
        "full_tps_median": float(np.median(full_tps)),
        "prefill_ms_mean": float(np.mean(prefill_ms)),
        "prefill_ms_std": float(np.std(prefill_ms)),
        "prefill_ms_median": float(np.median(prefill_ms)),
        "ttft_ms_mean": float(np.mean(ttft_ms)),
        "ttft_ms_std": float(np.std(ttft_ms)),
        "ttft_ms_median": float(np.median(ttft_ms)),
        "decode_ms_per_token_mean": float(np.mean(decode_ms_per_token)),
        "decode_ms_per_token_std": float(np.std(decode_ms_per_token)),
        "decode_ms_per_token_median": float(np.median(decode_ms_per_token)),
        "decode_tps_summary": summarize_samples(decode_tps),
        "full_tps_summary": summarize_samples(full_tps),
        "prefill_ms_summary": summarize_samples(prefill_ms),
        "ttft_ms_summary": summarize_samples(ttft_ms),
        "decode_ms_per_token_summary": summarize_samples(decode_ms_per_token),
        "decode_steps": int(decode_steps or cfg.throughput_tokens),
        "prompt_tokens": trials[0]["prompt_tokens"] if trials else 0,
        "raw_trials": trials,
    }


def experiment_throughput_reloaded(cfg: Config, log, model_id: str, prompt_style: str = "plain") -> dict:
    """Benchmark baseline and TQ throughput with one model resident at a time."""
    log.info("E5 â€” Throughput benchmark")

    base_model = None
    tq_model = None
    tokenizer = None
    tq_tokenizer = None

    try:
        base_model, tokenizer = load_model(cfg, model_id=model_id, apply_tq=False)
        if model_is_offloaded(base_model):
            log.warning("Baseline throughput model is offloaded; timing may still be distorted.")
        base_mean, base_std = measure_loaded_model_throughput(
            base_model,
            tokenizer,
            throughput_prompt(tokenizer, prompt_style),
            cfg.throughput_trials,
            cfg,
        )
    finally:
        base_model_ref, tokenizer_ref = base_model, tokenizer
        base_model = None
        tokenizer = None
        _cleanup_loaded_models(base_model_ref, tokenizer_ref)

    try:
        tq_model, tq_tokenizer = load_model(
            cfg, model_id=model_id, apply_tq=True, bits=cfg.quantized_bits
        )
        if model_is_offloaded(tq_model):
            log.warning("Quantized throughput model is offloaded; timing may still be distorted.")
        tq_mean, tq_std = measure_loaded_model_throughput(
            tq_model,
            tq_tokenizer,
            throughput_prompt(tq_tokenizer, prompt_style),
            cfg.throughput_trials,
            cfg,
        )
    finally:
        tq_model_ref, tq_tokenizer_ref = tq_model, tq_tokenizer
        tq_model = None
        tq_tokenizer = None
        _cleanup_loaded_models(tq_model_ref, tq_tokenizer_ref)

    result = {
        "baseline_tps": round(base_mean, 2),
        "baseline_tps_std": round(base_std, 2),
        "tq3_tps": round(tq_mean, 2),
        "tq3_tps_std": round(tq_std, 2),
        "tps_ratio": round(tq_mean / base_mean, 3),
    }
    log.info(f"  Baseline:   {base_mean:.1f} Â± {base_std:.1f} tok/s")
    log.info(f"  TQ-3bit:    {tq_mean:.1f} Â± {tq_std:.1f} tok/s")
    log.info(f"  Ratio:      {result['tps_ratio']:.3f}")
    return result


def experiment_factual_qa(model, tokenizer, evaluator_model, evaluator_tokenizer,
                          cfg: Config, log, label: str = "baseline",
                          prompt_style: str = "plain") -> dict:
    """Strict answer accuracy plus external-evaluator similarity."""
    log.info(f"E6 — Factual QA [{label}]")
    acc_scores, sim_scores, per_q = [], [], []

    for item in tqdm(FACTUAL_QA, desc=f"factual_qa/{label}", leave=False):
        q, gt = item["q"], item["a"]
        prompt = format_prompt(
            "factual_qa",
            {"question": q},
            tokenizer,
            prompt_style=prompt_style,
        )

        seed_results = []
        for seed in cfg.text_eval_seeds:
            generation = generate_completion(
                model,
                tokenizer,
                prompt,
                cfg,
                seed=seed,
                max_new_tokens=cfg.max_new_tokens,
            )
            pred = generation["completion"]
            answer_text = extract_answer_text("", pred)
            acc = strict_answer_accuracy(pred, item, "")
            emb_gt, emb_pred = embed_text(
                evaluator_model,
                evaluator_tokenizer,
                [gt, answer_text or pred],
                max_length=cfg.evaluator_max_length,
            )
            sim = cosine_sim(emb_gt.unsqueeze(0), emb_pred.unsqueeze(0))
            acc_scores.append(acc)
            sim_scores.append(sim)
            seed_results.append(
                {
                    "seed": seed,
                    "prediction": pred,
                    "answer_text": answer_text,
                    "accuracy": acc,
                    "similarity": round(sim, 4),
                }
            )

        per_q.append(
            {
                "q": q,
                "gt": gt,
                "accepted_answers": canonical_answers(item),
                "accuracy": round(float(np.mean([r["accuracy"] for r in seed_results])), 4),
                "similarity": round(float(np.mean([r["similarity"] for r in seed_results])), 4),
                "seed_results": seed_results,
            }
        )

    result = {
        "accuracy": round(float(np.mean(acc_scores)), 4),
        "avg_similarity": round(float(np.mean(sim_scores)), 4),
        "std_similarity": round(float(np.std(sim_scores)), 4),
        "n_questions": len(FACTUAL_QA),
        "n_seeds": len(cfg.text_eval_seeds),
        "per_question": per_q,
    }
    log.info(
        f"  Accuracy: {result['accuracy']:.2%}   "
        f"Similarity: {result['avg_similarity']:.4f} ± {result['std_similarity']:.4f}"
    )
    return result


def experiment_multi_task(base_model, tq_model, tokenizer,
                          evaluator_model, evaluator_tokenizer,
                          cfg: Config, log, prompt_style: str = "plain") -> dict:
    """Compare seeded generations from baseline vs quantized models."""
    log.info("E7 — Multi-task evaluation (baseline vs TQ-3bit)")
    results = {}

    for task, prompts in MULTI_TASK_PROMPTS.items():
        sims, examples = [], []
        for prompt_text in tqdm(prompts, desc=f"multi_task/{task}", leave=False):
            prompt = format_prompt(
                task,
                {"prompt": prompt_text},
                tokenizer,
                prompt_style=prompt_style,
            )
            seed_results = []
            for seed in cfg.text_eval_seeds:
                out_base = generate_completion(
                    base_model, tokenizer, prompt, cfg, seed=seed, max_new_tokens=cfg.max_new_tokens
                )
                out_tq = generate_completion(
                    tq_model, tokenizer, prompt, cfg, seed=seed, max_new_tokens=cfg.max_new_tokens
                )
                emb_base, emb_tq = embed_text(
                    evaluator_model,
                    evaluator_tokenizer,
                    [
                        out_base["completion"] or out_base["full_text"],
                        out_tq["completion"] or out_tq["full_text"],
                    ],
                    max_length=cfg.evaluator_max_length,
                )
                sim = cosine_sim(emb_base.unsqueeze(0), emb_tq.unsqueeze(0))
                sims.append(sim)
                seed_results.append(
                    {
                        "seed": seed,
                        "baseline": out_base["completion"],
                        "turboquant": out_tq["completion"],
                        "similarity": round(sim, 4),
                    }
                )

            if len(examples) < 2:
                examples.append(
                    {
                        "prompt": prompt_text,
                        "similarity": round(float(np.mean([r["similarity"] for r in seed_results])), 4),
                        "seed_results": seed_results,
                    }
                )

        results[task] = {
            "avg_similarity": round(float(np.mean(sims)), 4),
            "std_similarity": round(float(np.std(sims)), 4),
            "min_similarity": round(float(np.min(sims)), 4),
            "examples": examples,
        }
        log.info(
            f"  [{task}] similarity: {results[task]['avg_similarity']:.4f}"
            f" ± {results[task]['std_similarity']:.4f}"
        )

    return results


def experiment_rag_simulation(base_model, tq_model, tokenizer,
                               cfg: Config, log, prompt_style: str = "plain") -> dict:
    """Evaluate context-grounded QA with multiple generation seeds."""
    log.info("E8 — RAG simulation (context-grounded factual retrieval)")
    base_results, tq_results, per_q = [], [], []

    for item in tqdm(RAG_CONTEXTS, desc="rag_simulation", leave=False):
        prompt = format_prompt(
            "rag_qa",
            {"context": item["context"], "question": item["q"]},
            tokenizer,
            prompt_style=prompt_style,
        )
        seed_results = []
        for seed in cfg.text_eval_seeds:
            out_base = generate_completion(
                base_model, tokenizer, prompt, cfg, seed=seed, max_new_tokens=20
            )
            out_tq = generate_completion(
                tq_model, tokenizer, prompt, cfg, seed=seed, max_new_tokens=20
            )
            answer_base = extract_answer_text("", out_base["completion"])
            answer_tq = extract_answer_text("", out_tq["completion"])
            acc_base = strict_answer_accuracy(out_base["completion"], item, "")
            acc_tq = strict_answer_accuracy(out_tq["completion"], item, "")
            base_results.append(acc_base)
            tq_results.append(acc_tq)
            seed_results.append(
                {
                    "seed": seed,
                    "baseline_pred": out_base["completion"],
                    "tq_pred": out_tq["completion"],
                    "baseline_answer_text": answer_base,
                    "tq_answer_text": answer_tq,
                    "baseline_acc": acc_base,
                    "tq_acc": acc_tq,
                }
            )

        per_q.append(
            {
                "question": item["q"],
                "answer": item["a"],
                "accepted_answers": canonical_answers(item),
                "baseline_acc": round(float(np.mean([r["baseline_acc"] for r in seed_results])), 4),
                "tq_acc": round(float(np.mean([r["tq_acc"] for r in seed_results])), 4),
                "seed_results": seed_results,
            }
        )

    return {
        "baseline_accuracy": round(float(np.mean(base_results)), 4),
        "tq_accuracy": round(float(np.mean(tq_results)), 4),
        "accuracy_delta": round(float(np.mean(tq_results)) - float(np.mean(base_results)), 4),
        "n_questions": len(RAG_CONTEXTS),
        "n_seeds": len(cfg.text_eval_seeds),
        "per_question": per_q,
    }


def _cleanup_loaded_models(*objects):
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pick_primary_label(cfg: Config) -> str:
    labels = [spec["label"] for spec in cfg.baseline_models]
    return cfg.primary_model_label if cfg.primary_model_label in labels else labels[0]


def _suite_summary(model_runs: dict) -> dict:
    summary = {}
    for label, run in model_runs.items():
        factual_base = run.get("factual_qa_baseline", {})
        factual_tq = run.get("factual_qa_tq3", {})
        rag = run.get("rag_simulation", {})
        throughput = run.get("throughput", {})
        mixed = run.get("mixed_bit", {})
        summary[label] = {
            "model_id": run.get("model_id"),
            "prompt_style": run.get("prompt_style"),
            "factual_baseline_accuracy": factual_base.get("accuracy"),
            "factual_tq_accuracy": factual_tq.get("accuracy"),
            "rag_baseline_accuracy": rag.get("baseline_accuracy"),
            "rag_tq_accuracy": rag.get("tq_accuracy"),
            "throughput_ratio": throughput.get("tps_ratio"),
            "aggregate_throughput_ratio": throughput.get("aggregate_tps_ratio_median"),
            "latency_ratio": throughput.get("latency_ratio"),
            "throughput_ratio_range": (throughput.get("consistency") or {}).get("ratio_range"),
            "mixed_bit_improvement_pct": mixed.get("mse_improvement_pct"),
        }
    return summary


def config_snapshot(cfg: Config, primary_model_label: str = None, timestamp: str = None) -> dict:
    return {
        "model_id": cfg.model_id,
        "baseline_models": cfg.baseline_models,
        "primary_model_label": primary_model_label or cfg.primary_model_label,
        "evaluator_model_id": cfg.evaluator_model_id,
        "eval_set_version": cfg.eval_set_version,
        "factual_qa_count": len(FACTUAL_QA),
        "rag_context_count": len(RAG_CONTEXTS),
        "bits_list": cfg.bits_list,
        "default_bits": cfg.default_bits,
        "quantized_bits": cfg.quantized_bits,
        "quantization_mode": cfg.quantization_mode,
        "head_dim": cfg.head_dim,
        "seq_len": cfg.seq_len,
        "context_lengths": cfg.context_lengths,
        "max_new_tokens": cfg.max_new_tokens,
        "text_eval_seeds": cfg.text_eval_seeds,
        "text_eval_do_sample": cfg.text_eval_do_sample,
        "throughput_benchmark_mode": cfg.throughput_benchmark_mode,
        "throughput_warmup_trials": cfg.throughput_warmup_trials,
        "throughput_trials": cfg.throughput_trials,
        "throughput_tokens": cfg.throughput_tokens,
        "throughput_alternating_rounds": cfg.throughput_alternating_rounds,
        "throughput_prompt_token_targets": cfg.throughput_prompt_token_targets,
        "throughput_headline_prompt_target": cfg.throughput_headline_prompt_target,
        "text_eval_temperature": cfg.text_eval_temperature,
        "text_eval_top_p": cfg.text_eval_top_p,
        "text_eval_repetition_penalty": cfg.text_eval_repetition_penalty,
        "mixed_top_layers": cfg.mixed_top_layers,
        "mixed_bottom_layers": cfg.mixed_bottom_layers,
        "proxy_note": cfg.proxy_note,
        "timestamp": timestamp,
    }


def read_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def resume_config_matches(saved_cfg: dict, cfg: Config) -> bool:
    if not saved_cfg:
        return False
    current = config_snapshot(cfg, primary_model_label=saved_cfg.get("primary_model_label"))
    keys = [k for k in current.keys() if k != "timestamp"]
    return all(saved_cfg.get(k) == current.get(k) for k in keys)


def model_result_complete(model_result: dict) -> bool:
    required = [
        "layer_sensitivity",
        "mixed_bit",
        "factual_qa_baseline",
        "factual_qa_tq3",
        "throughput",
        "multi_task",
        "rag_simulation",
    ]
    return all(key in model_result for key in required)


def _refresh_primary_views(results: dict, cfg: Config):
    model_runs = results.get("model_runs", {})
    if not model_runs:
        return
    primary_label = results.get("primary_model_label") or _pick_primary_label(cfg)
    if primary_label not in model_runs:
        primary_label = next(iter(model_runs))
    results["primary_model_label"] = primary_label
    results["model_suite_summary"] = _suite_summary(model_runs)
    primary = model_runs[primary_label]
    for key in [
        "layer_sensitivity",
        "mixed_bit",
        "factual_qa_baseline",
        "factual_qa_tq3",
        "throughput",
        "multi_task",
        "rag_simulation",
    ]:
        if key in primary:
            results[key] = primary[key]


def save_run_state(run_dir: str, cfg: Config, results: dict, progress: dict, ts: str):
    _refresh_primary_views(results, cfg)
    progress["last_saved"] = datetime.datetime.now().isoformat()
    write_json(os.path.join(run_dir, "results.json"), results)
    write_json(
        os.path.join(run_dir, "config.json"),
        config_snapshot(
            cfg,
            primary_model_label=results.get("primary_model_label") or _pick_primary_label(cfg),
            timestamp=ts,
        ),
    )
    write_json(os.path.join(run_dir, cfg.progress_file), progress)


def find_resume_run(cfg: Config):
    if not os.path.isdir(cfg.output_dir):
        return None
    run_dirs = [
        os.path.join(cfg.output_dir, name)
        for name in os.listdir(cfg.output_dir)
        if os.path.isdir(os.path.join(cfg.output_dir, name))
    ]
    run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for run_dir in run_dirs:
        saved_cfg = read_json_if_exists(os.path.join(run_dir, "config.json"))
        progress = read_json_if_exists(os.path.join(run_dir, cfg.progress_file)) or {}
        if resume_config_matches(saved_cfg, cfg) and progress.get("status") != "completed":
            return run_dir, saved_cfg, progress
    return None


def _legacy_plot_all_ascii_fix_pending(results: dict, run_dir: str, cfg: Config):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    BLUE   = "#2563EB"
    ORANGE = "#F97316"
    GREEN  = "#16A34A"
    RED    = "#DC2626"
    GREY   = "#6B7280"

    primary = results["model_runs"][results["primary_model_label"]]

    def save(name):
        plt.savefig(os.path.join(plot_dir, name), dpi=cfg.fig_dpi, bbox_inches="tight")
        plt.close()

    def similarity_ylim(values):
        vmin = min(values)
        vmax = max(values)
        low = 0.0
        high = min(1.05, max(0.2, vmax + 0.08))
        return low, high

    if "bit_tradeoff" in results:
        bt = results["bit_tradeoff"]
        bits = sorted(bt.keys())
        pq_mse = [bt[b]["pq_mse"] for b in bits]
        tq_mse = [bt[b]["tq_mse"] for b in bits]
        pq_kl = [bt[b]["pq_kl"] for b in bits]
        tq_kl = [bt[b]["tq_kl"] for b in bits]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(bits, pq_mse, "o-", color=ORANGE, label="PolarQuant")
        axes[0].plot(bits, tq_mse, "s-", color=BLUE, label="TurboQuant")
        axes[0].set_xlabel("Bits"); axes[0].set_ylabel("Attention MSE")
        axes[0].set_title("Attention MSE vs Bit Depth"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(bits, pq_kl, "o-", color=ORANGE, label="PolarQuant")
        axes[1].plot(bits, tq_kl, "s-", color=BLUE, label="TurboQuant")
        axes[1].set_xlabel("Bits"); axes[1].set_ylabel("KL Divergence")
        axes[1].set_title("Attention KL Divergence vs Bit Depth")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.suptitle("E1 — PolarQuant vs TurboQuant Bit-Depth Tradeoff", fontsize=13)
        plt.tight_layout(); save("E1_bit_tradeoff.png")

    if "distortion_curve" in results:
        dc = results["distortion_curve"]
        bits = sorted(dc.keys())
        plt.figure(figsize=(7, 4))
        plt.semilogy(bits, [dc[b]["mse"] for b in bits], "o-", color=BLUE, label="PolarQuant (measured)")
        ref_key = "reference_curve" if "reference_curve" in dc[bits[0]] else "lower_bound"
        plt.semilogy(bits, [dc[b][ref_key] for b in bits], "--", color=GREY, label="Anchored 4^-b reference")
        plt.xlabel("Bits"); plt.ylabel("MSE (log scale)")
        plt.title("E2 — Rate-Distortion: Theory vs Practice")
        plt.legend(); plt.grid(alpha=0.3)
        save("E2_distortion_curve.png")

    if "layer_sensitivity" in primary:
        ls = {k: v for k, v in primary["layer_sensitivity"].items() if k.startswith("layer_")}
        lnames = sorted(ls.keys(), key=lambda x: int(x.split("_")[1]))
        mse_v = [ls[l]["mse"] for l in lnames]
        x = list(range(len(lnames)))

        plt.figure(figsize=(max(8, len(x) * 0.4), 4))
        plt.bar(x, mse_v, color=BLUE, alpha=0.7, width=0.8)
        plt.xticks(x[::2], [l.replace("layer_", "L") for l in lnames[::2]], rotation=45, fontsize=8)
        plt.xlabel("Transformer Layer"); plt.ylabel("Reconstruction MSE")
        plt.title(f"E3 — Layer Sensitivity (TQ at {cfg.default_bits} bits)")
        plt.axhline(np.mean(mse_v), color=RED, linestyle="--", label=f"Mean={np.mean(mse_v):.4f}")
        plt.legend(); plt.grid(axis="y", alpha=0.3)
        save("E3_layer_sensitivity.png")

    if "memory_scaling" in results:
        ms = results["memory_scaling"]
        ctxs = sorted(ms.keys())
        plt.figure(figsize=(8, 4))
        plt.plot(ctxs, [ms[c]["fp16_gb"] for c in ctxs], "o-", color=RED, label="FP16 (baseline)")
        plt.plot(ctxs, [ms[c]["4bit_gb"] for c in ctxs], "s-", color=ORANGE, label="4-bit TurboQuant")
        plt.plot(ctxs, [ms[c]["3bit_gb"] for c in ctxs], "^-", color=BLUE, label="3-bit TurboQuant")
        plt.plot(ctxs, [ms[c]["2bit_gb"] for c in ctxs], "D-", color=GREEN, label="2-bit TurboQuant")
        plt.xlabel("Context Length (tokens)"); plt.ylabel("KV Cache Memory (GB)")
        plt.title("E4 — KV Cache Memory Scaling (proxy from k_proj quantization)")
        plt.legend(); plt.grid(alpha=0.3)
        save("E4_memory_scaling.png")

    base = primary["factual_qa_baseline"]
    tq3 = primary["factual_qa_tq3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Baseline", "TQ-3bit"]
    acc = [base["accuracy"], tq3["accuracy"]]
    sim = [base["avg_similarity"], tq3["avg_similarity"]]
    sim_sd = [base["std_similarity"], tq3["std_similarity"]]
    axes[0].bar(labels, acc, color=[BLUE, ORANGE], alpha=0.85)
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Strict Answer Accuracy")
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    axes[1].bar(labels, sim, color=[BLUE, ORANGE], alpha=0.85, yerr=sim_sd, capsize=5)
    sim_low, sim_high = similarity_ylim(sim)
    axes[1].set_ylim(sim_low, sim_high); axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("External Evaluator Similarity (mean ± std)")
    for i, v in enumerate(sim):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    fig.suptitle("E6 — Factual QA: Baseline vs TurboQuant 3-bit", fontsize=13)
    plt.tight_layout(); save("E6_factual_qa.png")

    mt = primary["multi_task"]
    tasks = list(mt.keys())
    sims = [mt[t]["avg_similarity"] for t in tasks]
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    sims_c = sims + [sims[0]]
    angles_c = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_c, sims_c, "o-", color=BLUE, linewidth=2)
    ax.fill(angles_c, sims_c, alpha=0.2, color=BLUE)
    ax.set_xticks(angles)
    ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=11)
    radar_low, radar_high = similarity_ylim(sims)
    ax.set_ylim(radar_low, max(radar_high, radar_low + 0.2))
    ax.set_title("E7 — Multi-task Semantic Similarity\n(TQ-3bit vs Baseline)", pad=20, fontsize=12)
    save("E7_multitask_radar.png")

    rag = primary["rag_simulation"]
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Baseline", "TQ-3bit"], [rag["baseline_accuracy"], rag["tq_accuracy"]], color=[BLUE, ORANGE], alpha=0.85, width=0.4)
    plt.ylim(0, 1.2); plt.ylabel("Accuracy (RAG context)")
    plt.title("E8 — RAG Simulation: Context Grounded QA")
    for bar, v in zip(bars, [rag["baseline_accuracy"], rag["tq_accuracy"]]):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    save("E8_rag_simulation.png")

    mb = primary["mixed_bit"]
    pl = {k: v for k, v in mb["per_layer"].items()}
    lnames = sorted(pl.keys(), key=lambda x: int(x.split("_")[1]))
    u_mse = [pl[l]["uniform_mse"] for l in lnames]
    m_mse = [pl[l]["mixed_mse"] for l in lnames]
    x = list(range(len(lnames)))
    plt.figure(figsize=(max(8, len(x) * 0.4), 4))
    plt.plot(x, u_mse, "-", color=ORANGE, alpha=0.7, label="Uniform 3-bit")
    plt.plot(x, m_mse, "-", color=BLUE, alpha=0.7, label="Sensitivity-ranked mixed-bit")
    plt.fill_between(x, u_mse, m_mse, where=[u > m for u, m in zip(u_mse, m_mse)], alpha=0.15, color=GREEN, label="Improvement region")
    plt.xlabel("Layer Index"); plt.ylabel("MSE")
    plt.title(f"E10 — Sensitivity-ranked Mixed-bit vs Uniform 3-bit  (Δ={mb['mse_improvement_pct']:.1f}%, eff={mb['effective_bits']:.2f} bits)")
    plt.legend(); plt.grid(alpha=0.3)
    save("E10_mixed_bit.png")

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    ax0 = fig.add_subplot(gs[0, 0])
    if 16384 in results["memory_scaling"]:
        ms16 = results["memory_scaling"][16384]
        ax0.bar(["FP16", "4-bit", "3-bit", "2-bit"], [ms16["fp16_gb"], ms16["4bit_gb"], ms16["3bit_gb"], ms16["2bit_gb"]], color=[RED, ORANGE, BLUE, GREEN], alpha=0.8)
        ax0.set_title("KV Cache @ 16K ctx (GB)", fontsize=10)
        ax0.set_ylabel("GB"); ax0.grid(axis="y", alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(["Baseline", "TQ-3bit"], [base["accuracy"], tq3["accuracy"]], color=[BLUE, ORANGE], alpha=0.8)
    ax1.set_ylim(0, 1.1); ax1.set_title("Factual Accuracy", fontsize=10)
    ax1.set_ylabel("Accuracy"); ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    s_vals = [mt[t]["avg_similarity"] for t in tasks]
    s_errs = [mt[t]["std_similarity"] for t in tasks]
    ax2.bar(range(len(tasks)), s_vals, yerr=s_errs, color=BLUE, alpha=0.7, capsize=4)
    ax2.set_xticks(range(len(tasks)))
    ax2.set_xticklabels([t[:5] for t in tasks], fontsize=8)
    low, high = similarity_ylim(s_vals)
    ax2.set_ylim(low, high)
    ax2.set_title("Multi-task Similarity (TQ-3bit)", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    dc = results["distortion_curve"]
    bits = sorted(dc.keys())
    ax3.semilogy(bits, [dc[b]["mse"] for b in bits], "o-", color=BLUE, label="Measured")
    ax3.semilogy(bits, [dc[b]["lower_bound"] for b in bits], "--", color=GREY, label="Lower bound")
    ax3.set_title("Rate-Distortion Curve", fontsize=10)
    ax3.set_xlabel("Bits"); ax3.set_ylabel("MSE (log)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(["Baseline", "TQ-3bit"], [rag["baseline_accuracy"], rag["tq_accuracy"]], color=[BLUE, ORANGE], alpha=0.8)
    ax4.set_ylim(0, 1.2); ax4.set_title("RAG Accuracy", fontsize=10)
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(["Uniform 3-bit", "Mixed-bit"], [mb["uniform_3bit_mean_mse"], mb["mixed_mean_mse"]], color=[ORANGE, BLUE], alpha=0.8)
    ax5.set_title(f"Mixed-bit MSE\n(eff {mb['effective_bits']:.2f} bits)", fontsize=10)
    ax5.set_ylabel("Mean MSE"); ax5.grid(axis="y", alpha=0.3)

    fig.suptitle("TurboQuant Experiment Summary — k_proj Proxy Evaluation", fontsize=14, y=1.01)
    save("SUMMARY_dashboard.png")


def plot_all(results: dict, run_dir: str, cfg: Config):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    BLUE = "#2563EB"
    ORANGE = "#F97316"
    GREEN = "#16A34A"
    RED = "#DC2626"
    GREY = "#6B7280"

    primary = results["model_runs"][results["primary_model_label"]]

    def save(name):
        plt.savefig(os.path.join(plot_dir, name), dpi=cfg.fig_dpi, bbox_inches="tight")
        plt.close()

    def similarity_ylim(values):
        vmax = max(values)
        return 0.0, min(1.05, max(0.2, vmax + 0.08))

    def normalize_numeric_keyed_dict(d):
        normalized = {}
        for key, value in d.items():
            try:
                normalized[int(key)] = value
            except (TypeError, ValueError):
                normalized[key] = value
        return normalized

    if "bit_tradeoff" in results:
        bt = normalize_numeric_keyed_dict(results["bit_tradeoff"])
        bits = sorted(bt.keys())
        pq_mse = [bt[b]["pq_mse"] for b in bits]
        tq_mse = [bt[b]["tq_mse"] for b in bits]
        pq_kl = [bt[b]["pq_kl"] for b in bits]
        tq_kl = [bt[b]["tq_kl"] for b in bits]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(bits, pq_mse, "o-", color=ORANGE, label="PolarQuant")
        axes[0].plot(bits, tq_mse, "s-", color=BLUE, label="TurboQuant")
        axes[0].set_xlabel("Bits"); axes[0].set_ylabel("Attention MSE")
        axes[0].set_title("Attention MSE vs Bit Depth"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(bits, pq_kl, "o-", color=ORANGE, label="PolarQuant")
        axes[1].plot(bits, tq_kl, "s-", color=BLUE, label="TurboQuant")
        axes[1].set_xlabel("Bits"); axes[1].set_ylabel("KL Divergence")
        axes[1].set_title("Attention KL Divergence vs Bit Depth")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.suptitle("E1 - PolarQuant vs TurboQuant Bit-Depth Tradeoff", fontsize=13)
        plt.tight_layout(); save("E1_bit_tradeoff.png")

    if "distortion_curve" in results:
        dc = normalize_numeric_keyed_dict(results["distortion_curve"])
        bits = sorted(dc.keys())
        ref_key = "reference_curve" if "reference_curve" in dc[bits[0]] else "lower_bound"
        plt.figure(figsize=(7, 4))
        plt.semilogy(bits, [dc[b]["mse"] for b in bits], "o-", color=BLUE, label="PolarQuant (measured)")
        plt.semilogy(bits, [dc[b][ref_key] for b in bits], "--", color=GREY, label="Anchored 4^-b reference")
        plt.xlabel("Bits"); plt.ylabel("MSE (log scale)")
        plt.title("E2 - Rate-Distortion Trend")
        plt.legend(); plt.grid(alpha=0.3)
        save("E2_distortion_curve.png")

    if "layer_sensitivity" in primary:
        ls = {k: v for k, v in primary["layer_sensitivity"].items() if k.startswith("layer_")}
        lnames = sorted(ls.keys(), key=lambda x: int(x.split("_")[1]))
        mse_v = [ls[l]["mse"] for l in lnames]
        x = list(range(len(lnames)))

        plt.figure(figsize=(max(8, len(x) * 0.4), 4))
        plt.bar(x, mse_v, color=BLUE, alpha=0.7, width=0.8)
        plt.xticks(x[::2], [l.replace("layer_", "L") for l in lnames[::2]], rotation=45, fontsize=8)
        plt.xlabel("Transformer Layer"); plt.ylabel("Reconstruction MSE")
        plt.title(f"E3 â€” Layer Sensitivity (TQ at {cfg.default_bits} bits)")
        plt.axhline(np.mean(mse_v), color=RED, linestyle="--", label=f"Mean={np.mean(mse_v):.4f}")
        plt.legend(); plt.grid(axis="y", alpha=0.3)
        save("E3_layer_sensitivity.png")

    if "memory_scaling" in results:
        ms = normalize_numeric_keyed_dict(results["memory_scaling"])
        ctxs = sorted(ms.keys())
        plt.figure(figsize=(8, 4))
        plt.plot(ctxs, [ms[c]["fp16_gb"] for c in ctxs], "o-", color=RED, label="FP16 (baseline)")
        plt.plot(ctxs, [ms[c]["4bit_gb"] for c in ctxs], "s-", color=ORANGE, label="4-bit TurboQuant")
        plt.plot(ctxs, [ms[c]["3bit_gb"] for c in ctxs], "^-", color=BLUE, label="3-bit TurboQuant")
        plt.plot(ctxs, [ms[c]["2bit_gb"] for c in ctxs], "D-", color=GREEN, label="2-bit TurboQuant")
        plt.xlabel("Context Length (tokens)"); plt.ylabel("KV Cache Memory (GB)")
        plt.title("E4 - KV Cache Memory Scaling (proxy from k_proj quantization)")
        plt.legend(); plt.grid(alpha=0.3)
        save("E4_memory_scaling.png")

    base = primary["factual_qa_baseline"]
    tq3 = primary["factual_qa_tq3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Baseline", "TQ-3bit"]
    acc = [base["accuracy"], tq3["accuracy"]]
    sim = [base["avg_similarity"], tq3["avg_similarity"]]
    sim_sd = [base["std_similarity"], tq3["std_similarity"]]
    axes[0].bar(labels, acc, color=[BLUE, ORANGE], alpha=0.85)
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Strict Answer Accuracy")
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    axes[1].bar(labels, sim, color=[BLUE, ORANGE], alpha=0.85, yerr=sim_sd, capsize=5)
    sim_low, sim_high = similarity_ylim(sim)
    axes[1].set_ylim(sim_low, sim_high); axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("External Evaluator Similarity (mean Â± std)")
    for i, v in enumerate(sim):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    fig.suptitle("E6 - Factual QA: Baseline vs TurboQuant 3-bit", fontsize=13)
    plt.tight_layout(); save("E6_factual_qa.png")

    mt = primary["multi_task"]
    tasks = list(mt.keys())
    sims = [mt[t]["avg_similarity"] for t in tasks]
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    sims_c = sims + [sims[0]]
    angles_c = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_c, sims_c, "o-", color=BLUE, linewidth=2)
    ax.fill(angles_c, sims_c, alpha=0.2, color=BLUE)
    ax.set_xticks(angles)
    ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=11)
    radar_low, radar_high = similarity_ylim(sims)
    ax.set_ylim(radar_low, max(radar_high, radar_low + 0.2))
    ax.set_title("E7 - Multi-task Semantic Similarity\n(TQ-3bit vs Baseline)", pad=20, fontsize=12)
    save("E7_multitask_radar.png")

    rag = primary["rag_simulation"]
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Baseline", "TQ-3bit"], [rag["baseline_accuracy"], rag["tq_accuracy"]], color=[BLUE, ORANGE], alpha=0.85, width=0.4)
    plt.ylim(0, 1.2); plt.ylabel("Accuracy (RAG context)")
    plt.title("E8 - RAG Simulation: Context Grounded QA")
    for bar, v in zip(bars, [rag["baseline_accuracy"], rag["tq_accuracy"]]):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    save("E8_rag_simulation.png")

    mb = primary["mixed_bit"]
    pl = {k: v for k, v in mb["per_layer"].items()}
    lnames = sorted(pl.keys(), key=lambda x: int(x.split("_")[1]))
    u_mse = [pl[l]["uniform_mse"] for l in lnames]
    m_mse = [pl[l]["mixed_mse"] for l in lnames]
    x = list(range(len(lnames)))
    plt.figure(figsize=(max(8, len(x) * 0.4), 4))
    plt.plot(x, u_mse, "-", color=ORANGE, alpha=0.7, label="Uniform 3-bit")
    plt.plot(x, m_mse, "-", color=BLUE, alpha=0.7, label="Sensitivity-ranked mixed-bit")
    plt.fill_between(x, u_mse, m_mse, where=[u > m for u, m in zip(u_mse, m_mse)], alpha=0.15, color=GREEN, label="Improvement region")
    plt.xlabel("Layer Index"); plt.ylabel("MSE")
    plt.title(f"E10 - Sensitivity-ranked Mixed-bit vs Uniform 3-bit  (delta={mb['mse_improvement_pct']:.1f}%, eff={mb['effective_bits']:.2f} bits)")
    plt.legend(); plt.grid(alpha=0.3)
    save("E10_mixed_bit.png")

    fig = plt.figure(figsize=(15, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)
    suite_labels = list(results["model_runs"].keys())
    suite_runs = [results["model_runs"][label] for label in suite_labels]
    suite_tasks = list(suite_runs[0]["multi_task"].keys())
    x_pos = np.arange(len(suite_labels))
    width = 0.36

    ax0 = fig.add_subplot(gs[0, 0])
    ms_all = normalize_numeric_keyed_dict(results["memory_scaling"])
    if 16384 in ms_all:
        ms16 = ms_all[16384]
        ax0.bar(["FP16", "4-bit", "3-bit", "2-bit"], [ms16["fp16_gb"], ms16["4bit_gb"], ms16["3bit_gb"], ms16["2bit_gb"]], color=[RED, ORANGE, BLUE, GREEN], alpha=0.8)
        ax0.set_title("KV Cache @ 16K ctx (GB)", fontsize=10)
        ax0.set_ylabel("GB")
        ax0.grid(axis="y", alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "16K context data unavailable", ha="center", va="center", fontsize=10)
        ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    factual_base_vals = [run["factual_qa_baseline"]["accuracy"] for run in suite_runs]
    factual_tq_vals = [run["factual_qa_tq3"]["accuracy"] for run in suite_runs]
    ax1.bar(x_pos - width / 2, factual_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax1.bar(x_pos + width / 2, factual_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Factual Accuracy by Model", fontsize=10)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    heatmap = np.array([[run["multi_task"][task]["avg_similarity"] for task in suite_tasks] for run in suite_runs])
    im = ax2.imshow(heatmap, cmap="Blues", vmin=0.0, vmax=max(0.6, float(np.max(heatmap))))
    ax2.set_xticks(range(len(suite_tasks)))
    ax2.set_xticklabels([task.capitalize() for task in suite_tasks], rotation=25, ha="right", fontsize=8)
    ax2.set_yticks(range(len(suite_labels)))
    ax2.set_yticklabels(suite_labels, fontsize=8)
    ax2.set_title("Multi-task Similarity by Model", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 0])
    rag_base_vals = [run["rag_simulation"]["baseline_accuracy"] for run in suite_runs]
    rag_tq_vals = [run["rag_simulation"]["tq_accuracy"] for run in suite_runs]
    ax3.bar(x_pos - width / 2, rag_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax3.bar(x_pos + width / 2, rag_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("RAG Accuracy by Model", fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    throughput_ratios = [run["throughput"]["tps_ratio"] for run in suite_runs]
    ax4.bar(suite_labels, throughput_ratios, color=GREEN, alpha=0.8)
    ax4.axhline(1.0, color=GREY, linestyle="--", linewidth=1)
    ax4.set_title("Throughput Ratio (TQ/Baseline)", fontsize=10)
    ax4.set_ylabel("Ratio")
    ax4.tick_params(axis="x", rotation=15, labelsize=8)
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    mixed_gains = [run["mixed_bit"]["mse_improvement_pct"] for run in suite_runs]
    ax5.bar(suite_labels, mixed_gains, color=BLUE, alpha=0.8)
    ax5.set_title("Mixed-bit MSE Improvement", fontsize=10)
    ax5.set_ylabel("Percent")
    ax5.tick_params(axis="x", rotation=15, labelsize=8)
    ax5.grid(axis="y", alpha=0.3)

    fig.suptitle("TurboQuant Suite Summary - k_proj Proxy Evaluation", fontsize=14, y=1.01)
    save("SUMMARY_dashboard.png")


def plot_all(results: dict, run_dir: str, cfg: Config):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    BLUE = "#2563EB"
    ORANGE = "#F97316"
    GREEN = "#16A34A"
    RED = "#DC2626"
    GREY = "#6B7280"

    primary = results["model_runs"][results["primary_model_label"]]

    def save(name):
        plt.savefig(os.path.join(plot_dir, name), dpi=cfg.fig_dpi, bbox_inches="tight")
        plt.close()

    def similarity_ylim(values):
        vmax = max(values)
        return 0.0, min(1.05, max(0.2, vmax + 0.08))

    def normalize_numeric_keyed_dict(d):
        normalized = {}
        for key, value in d.items():
            try:
                normalized[int(key)] = value
            except (TypeError, ValueError):
                normalized[key] = value
        return normalized

    if "bit_tradeoff" in results:
        bt = normalize_numeric_keyed_dict(results["bit_tradeoff"])
        bits = sorted(bt.keys())
        pq_mse = [bt[b]["pq_mse"] for b in bits]
        tq_mse = [bt[b]["tq_mse"] for b in bits]
        pq_kl = [bt[b]["pq_kl"] for b in bits]
        tq_kl = [bt[b]["tq_kl"] for b in bits]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(bits, pq_mse, "o-", color=ORANGE, label="PolarQuant")
        axes[0].plot(bits, tq_mse, "s-", color=BLUE, label="TurboQuant")
        axes[0].set_xlabel("Bits"); axes[0].set_ylabel("Attention MSE")
        axes[0].set_title("Attention MSE vs Bit Depth"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(bits, pq_kl, "o-", color=ORANGE, label="PolarQuant")
        axes[1].plot(bits, tq_kl, "s-", color=BLUE, label="TurboQuant")
        axes[1].set_xlabel("Bits"); axes[1].set_ylabel("KL Divergence")
        axes[1].set_title("Attention KL Divergence vs Bit Depth")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.suptitle("E1 - PolarQuant vs TurboQuant Bit-Depth Tradeoff", fontsize=13)
        plt.tight_layout(); save("E1_bit_tradeoff.png")

    if "distortion_curve" in results:
        dc = normalize_numeric_keyed_dict(results["distortion_curve"])
        bits = sorted(dc.keys())
        ref_key = "reference_curve" if "reference_curve" in dc[bits[0]] else "lower_bound"
        plt.figure(figsize=(7, 4))
        plt.semilogy(bits, [dc[b]["mse"] for b in bits], "o-", color=BLUE, label="PolarQuant (measured)")
        plt.semilogy(bits, [dc[b][ref_key] for b in bits], "--", color=GREY, label="Anchored 4^-b reference")
        plt.xlabel("Bits"); plt.ylabel("MSE (log scale)")
        plt.title("E2 - Rate-Distortion Trend")
        plt.legend(); plt.grid(alpha=0.3)
        save("E2_distortion_curve.png")

    if "layer_sensitivity" in primary:
        ls = {k: v for k, v in primary["layer_sensitivity"].items() if k.startswith("layer_")}
        lnames = sorted(ls.keys(), key=lambda x: int(x.split("_")[1]))
        mse_v = [ls[l]["mse"] for l in lnames]
        x = list(range(len(lnames)))

        plt.figure(figsize=(max(8, len(x) * 0.4), 4))
        plt.bar(x, mse_v, color=BLUE, alpha=0.7, width=0.8)
        plt.xticks(x[::2], [l.replace("layer_", "L") for l in lnames[::2]], rotation=45, fontsize=8)
        plt.xlabel("Transformer Layer"); plt.ylabel("Reconstruction MSE")
        plt.title(f"E3 - Layer Sensitivity (TQ at {cfg.default_bits} bits)")
        plt.axhline(np.mean(mse_v), color=RED, linestyle="--", label=f"Mean={np.mean(mse_v):.4f}")
        plt.legend(); plt.grid(axis="y", alpha=0.3)
        save("E3_layer_sensitivity.png")

    if "memory_scaling" in results:
        ms = normalize_numeric_keyed_dict(results["memory_scaling"])
        ctxs = sorted(ms.keys())
        plt.figure(figsize=(8, 4))
        plt.plot(ctxs, [ms[c]["fp16_gb"] for c in ctxs], "o-", color=RED, label="FP16 (baseline)")
        plt.plot(ctxs, [ms[c]["4bit_gb"] for c in ctxs], "s-", color=ORANGE, label="4-bit TurboQuant")
        plt.plot(ctxs, [ms[c]["3bit_gb"] for c in ctxs], "^-", color=BLUE, label="3-bit TurboQuant")
        plt.plot(ctxs, [ms[c]["2bit_gb"] for c in ctxs], "D-", color=GREEN, label="2-bit TurboQuant")
        plt.xlabel("Context Length (tokens)"); plt.ylabel("KV Cache Memory (GB)")
        plt.title("E4 - KV Cache Memory Scaling (proxy from k_proj quantization)")
        plt.legend(); plt.grid(alpha=0.3)
        save("E4_memory_scaling.png")

    base = primary["factual_qa_baseline"]
    tq3 = primary["factual_qa_tq3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Baseline", "TQ-3bit"]
    acc = [base["accuracy"], tq3["accuracy"]]
    sim = [base["avg_similarity"], tq3["avg_similarity"]]
    sim_sd = [base["std_similarity"], tq3["std_similarity"]]
    axes[0].bar(labels, acc, color=[BLUE, ORANGE], alpha=0.85)
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Strict Answer Accuracy")
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    axes[1].bar(labels, sim, color=[BLUE, ORANGE], alpha=0.85, yerr=sim_sd, capsize=5)
    sim_low, sim_high = similarity_ylim(sim)
    axes[1].set_ylim(sim_low, sim_high); axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("External Evaluator Similarity (mean +/- std)")
    for i, v in enumerate(sim):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    fig.suptitle("E6 - Factual QA: Baseline vs TurboQuant 3-bit", fontsize=13)
    plt.tight_layout(); save("E6_factual_qa.png")

    mt = primary["multi_task"]
    tasks = list(mt.keys())
    sims = [mt[t]["avg_similarity"] for t in tasks]
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    sims_c = sims + [sims[0]]
    angles_c = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles_c, sims_c, "o-", color=BLUE, linewidth=2)
    ax.fill(angles_c, sims_c, alpha=0.2, color=BLUE)
    ax.set_xticks(angles)
    ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=11)
    radar_low, radar_high = similarity_ylim(sims)
    ax.set_ylim(radar_low, max(radar_high, radar_low + 0.2))
    ax.set_title("E7 - Multi-task Semantic Similarity\n(TQ-3bit vs Baseline)", pad=20, fontsize=12)
    save("E7_multitask_radar.png")

    rag = primary["rag_simulation"]
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Baseline", "TQ-3bit"], [rag["baseline_accuracy"], rag["tq_accuracy"]], color=[BLUE, ORANGE], alpha=0.85, width=0.4)
    plt.ylim(0, 1.2); plt.ylabel("Accuracy (RAG context)")
    plt.title("E8 - RAG Simulation: Context Grounded QA")
    for bar, v in zip(bars, [rag["baseline_accuracy"], rag["tq_accuracy"]]):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.0%}", ha="center", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    save("E8_rag_simulation.png")

    mb = primary["mixed_bit"]
    pl = {k: v for k, v in mb["per_layer"].items()}
    lnames = sorted(pl.keys(), key=lambda x: int(x.split("_")[1]))
    u_mse = [pl[l]["uniform_mse"] for l in lnames]
    m_mse = [pl[l]["mixed_mse"] for l in lnames]
    x = list(range(len(lnames)))
    plt.figure(figsize=(max(8, len(x) * 0.4), 4))
    plt.plot(x, u_mse, "-", color=ORANGE, alpha=0.7, label="Uniform 3-bit")
    plt.plot(x, m_mse, "-", color=BLUE, alpha=0.7, label="Sensitivity-ranked mixed-bit")
    plt.fill_between(x, u_mse, m_mse, where=[u > m for u, m in zip(u_mse, m_mse)], alpha=0.15, color=GREEN, label="Improvement region")
    plt.xlabel("Layer Index"); plt.ylabel("MSE")
    plt.title(f"E10 - Sensitivity-ranked Mixed-bit vs Uniform 3-bit  (delta={mb['mse_improvement_pct']:.1f}%, eff={mb['effective_bits']:.2f} bits)")
    plt.legend(); plt.grid(alpha=0.3)
    save("E10_mixed_bit.png")

    fig = plt.figure(figsize=(15, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)
    suite_labels = list(results["model_runs"].keys())
    suite_runs = [results["model_runs"][label] for label in suite_labels]
    suite_tasks = list(suite_runs[0]["multi_task"].keys())
    x_pos = np.arange(len(suite_labels))
    width = 0.36

    ax0 = fig.add_subplot(gs[0, 0])
    ms_all = normalize_numeric_keyed_dict(results["memory_scaling"])
    if 16384 in ms_all:
        ms16 = ms_all[16384]
        ax0.bar(["FP16", "4-bit", "3-bit", "2-bit"], [ms16["fp16_gb"], ms16["4bit_gb"], ms16["3bit_gb"], ms16["2bit_gb"]], color=[RED, ORANGE, BLUE, GREEN], alpha=0.8)
        ax0.set_title("KV Cache @ 16K ctx (GB)", fontsize=10)
        ax0.set_ylabel("GB")
        ax0.grid(axis="y", alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "16K context data unavailable", ha="center", va="center", fontsize=10)
        ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    factual_base_vals = [run["factual_qa_baseline"]["accuracy"] for run in suite_runs]
    factual_tq_vals = [run["factual_qa_tq3"]["accuracy"] for run in suite_runs]
    ax1.bar(x_pos - width / 2, factual_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax1.bar(x_pos + width / 2, factual_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Factual Accuracy by Model", fontsize=10)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    heatmap = np.array([[run["multi_task"][task]["avg_similarity"] for task in suite_tasks] for run in suite_runs])
    im = ax2.imshow(heatmap, cmap="Blues", vmin=0.0, vmax=max(0.6, float(np.max(heatmap))))
    ax2.set_xticks(range(len(suite_tasks)))
    ax2.set_xticklabels([task.capitalize() for task in suite_tasks], rotation=25, ha="right", fontsize=8)
    ax2.set_yticks(range(len(suite_labels)))
    ax2.set_yticklabels(suite_labels, fontsize=8)
    ax2.set_title("Multi-task Similarity by Model", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 0])
    rag_base_vals = [run["rag_simulation"]["baseline_accuracy"] for run in suite_runs]
    rag_tq_vals = [run["rag_simulation"]["tq_accuracy"] for run in suite_runs]
    ax3.bar(x_pos - width / 2, rag_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax3.bar(x_pos + width / 2, rag_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("RAG Accuracy by Model", fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    throughput_ratios = [run["throughput"]["tps_ratio"] for run in suite_runs]
    ax4.bar(suite_labels, throughput_ratios, color=GREEN, alpha=0.8)
    ax4.axhline(1.0, color=GREY, linestyle="--", linewidth=1)
    ax4.set_title("Throughput Ratio (TQ/Baseline)", fontsize=10)
    ax4.set_ylabel("Ratio")
    ax4.tick_params(axis="x", rotation=15, labelsize=8)
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    mixed_gains = [run["mixed_bit"]["mse_improvement_pct"] for run in suite_runs]
    ax5.bar(suite_labels, mixed_gains, color=BLUE, alpha=0.8)
    ax5.set_title("Mixed-bit MSE Improvement", fontsize=10)
    ax5.set_ylabel("Percent")
    ax5.tick_params(axis="x", rotation=15, labelsize=8)
    ax5.grid(axis="y", alpha=0.3)

    fig.suptitle("TurboQuant Suite Summary - k_proj Proxy Evaluation", fontsize=14, y=1.01)
    save("SUMMARY_dashboard.png")


def experiment_attention_entropy(cfg: Config, log) -> dict:
    """Entropy drift under quantization across random and structured key distributions."""
    log.info("E9 - Attention entropy analysis")
    results = {"random": {}, "structured": {}}
    device = synthetic_device()

    torch.manual_seed(123)
    q = torch.randn(cfg.seq_len, cfg.head_dim, device=device)

    for dist in ["random", "structured"]:
        for bits in cfg.bits_list:
            if dist == "random":
                k = torch.randn(cfg.seq_len, cfg.head_dim, device=device)
            else:
                dominant = torch.randn(8, cfg.head_dim, device=device)
                k = dominant[torch.randint(8, (256,), device=device)] + 0.1 * torch.randn(
                    256, cfg.head_dim, device=device
                )

            k_hat, _, _, _ = turboquant_apply(k, bits)
            k_hat = k_hat.to(k.dtype)

            ent_base = attention_entropy(q, k)
            ent_tq = attention_entropy(q, k_hat)
            delta = ent_tq - ent_base

            results[dist][bits] = {
                "entropy_base": round(ent_base, 4),
                "entropy_tq": round(ent_tq, 4),
                "delta": round(delta, 4),
                "delta_pct": round(delta / ent_base * 100, 2),
            }
            log.info(
                f"  [{dist}] bits={bits}: base={ent_base:.4f}  "
                f"tq={ent_tq:.4f}  delta={delta:+.4f}"
            )

    return results


def _legacy_run_all_resumeless():
    cfg = Config()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, f"gemma2b_turboquant_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    log = setup_logger(run_dir)
    log.info("=" * 60)
    log.info("TurboQuant Experiment Pipeline — Multi-model Proxy Study")
    log.info(f"Output directory: {run_dir}")
    log.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log.info(f"Method note: {cfg.proxy_note}")
    log.info("=" * 60)

    results = {
        "proxy_note": cfg.proxy_note,
    }

    log.info("\n[PHASE 1] Algorithm characterisation (synthetic tensors)")
    results["bit_tradeoff"] = experiment_bit_tradeoff(cfg, log)
    results["distortion_curve"] = experiment_distortion_curve(cfg, log)
    results["attention_entropy"] = experiment_attention_entropy(cfg, log)
    results["memory_scaling"] = experiment_memory_scaling(cfg, log)

    log.info("\n[PHASE 2] Loading fixed external evaluator")
    evaluator_model, evaluator_tokenizer = load_similarity_evaluator(cfg)

    model_runs = {}
    for spec in cfg.baseline_models:
        label = spec["label"]
        model_id = spec["model_id"]
        prompt_style = spec.get("prompt_style", "plain")
        log.info(f"\n[MODEL] {label} ({model_id})")

        base_model, tokenizer = load_model(cfg, model_id=model_id, apply_tq=False)
        model_result = {
            "label": label,
            "model_id": model_id,
            "prompt_style": prompt_style,
        }

        model_result["layer_sensitivity"] = experiment_layer_sensitivity(base_model, cfg, log)
        model_result["mixed_bit"] = experiment_mixed_bit(
            base_model, model_result["layer_sensitivity"], cfg, log
        )
        model_result["factual_qa_baseline"] = experiment_factual_qa(
            base_model,
            tokenizer,
            evaluator_model,
            evaluator_tokenizer,
            cfg,
            log,
            label=f"{label}/baseline",
            prompt_style=prompt_style,
        )

        tq_model, _ = load_model(cfg, model_id=model_id, apply_tq=True, bits=cfg.quantized_bits)
        model_result["factual_qa_tq3"] = experiment_factual_qa(
            tq_model,
            tokenizer,
            evaluator_model,
            evaluator_tokenizer,
            cfg,
            log,
            label=f"{label}/tq_{cfg.quantized_bits}bit",
            prompt_style=prompt_style,
        )
        model_result["throughput"] = experiment_throughput(
            base_model, tq_model, tokenizer, cfg, log, prompt_style=prompt_style
        )
        model_result["multi_task"] = experiment_multi_task(
            base_model,
            tq_model,
            tokenizer,
            evaluator_model,
            evaluator_tokenizer,
            cfg,
            log,
            prompt_style=prompt_style,
        )
        model_result["rag_simulation"] = experiment_rag_simulation(
            base_model, tq_model, tokenizer, cfg, log, prompt_style=prompt_style
        )

        model_runs[label] = model_result
        _cleanup_loaded_models(base_model, tq_model, tokenizer)

    _cleanup_loaded_models(evaluator_model, evaluator_tokenizer)

    results["model_runs"] = model_runs
    results["model_suite_summary"] = _suite_summary(model_runs)
    results["primary_model_label"] = _pick_primary_label(cfg)

    primary = model_runs[results["primary_model_label"]]
    for key in [
        "layer_sensitivity",
        "mixed_bit",
        "factual_qa_baseline",
        "factual_qa_tq3",
        "throughput",
        "multi_task",
        "rag_simulation",
    ]:
        results[key] = primary[key]

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "model_id": cfg.model_id,
                "baseline_models": cfg.baseline_models,
                "primary_model_label": results["primary_model_label"],
                "evaluator_model_id": cfg.evaluator_model_id,
                "bits_list": cfg.bits_list,
                "default_bits": cfg.default_bits,
                "quantized_bits": cfg.quantized_bits,
                "head_dim": cfg.head_dim,
                "seq_len": cfg.seq_len,
                "context_lengths": cfg.context_lengths,
                "max_new_tokens": cfg.max_new_tokens,
                "text_eval_seeds": cfg.text_eval_seeds,
                "text_eval_do_sample": cfg.text_eval_do_sample,
                "text_eval_temperature": cfg.text_eval_temperature,
                "text_eval_top_p": cfg.text_eval_top_p,
                "text_eval_repetition_penalty": cfg.text_eval_repetition_penalty,
                "mixed_top_layers": cfg.mixed_top_layers,
                "mixed_bottom_layers": cfg.mixed_bottom_layers,
                "proxy_note": cfg.proxy_note,
                "timestamp": ts,
            },
            f,
            indent=2,
        )

    log.info("\n[PHASE 9] Generating plots")
    plot_all(results, run_dir, cfg)

    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT SUMMARY")
    log.info("=" * 60)
    log.info(f"Primary model for detailed plots: {results['primary_model_label']}")
    for label, summary in results["model_suite_summary"].items():
        log.info(
            f"{label}: factual {summary['factual_baseline_accuracy']:.0%}->{summary['factual_tq_accuracy']:.0%}  "
            f"rag {summary['rag_baseline_accuracy']:.0%}->{summary['rag_tq_accuracy']:.0%}  "
            f"tps_ratio={summary['throughput_ratio']:.3f}  "
            f"mixed_bit_gain={summary['mixed_bit_improvement_pct']:.1f}%"
        )
    log.info(f"\nAll outputs saved to: {run_dir}")
    log.info("=" * 60)
    return results, run_dir


def run_all():
    cfg = Config()
    resume_state = find_resume_run(cfg) if cfg.resume_latest else None

    if resume_state:
        run_dir, saved_cfg, progress = resume_state
        parts = os.path.basename(run_dir).rsplit("_", 2)
        fallback_ts = "_".join(parts[-2:]) if len(parts) >= 3 else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ts = saved_cfg.get("timestamp") or fallback_ts
        results = read_json_if_exists(os.path.join(run_dir, "results.json")) or {}
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(cfg.output_dir, f"gemma2b_turboquant_{ts}")
        progress = {}
        results = {}

    os.makedirs(run_dir, exist_ok=True)
    log = setup_logger(run_dir)
    log.info("=" * 60)
    log.info("TurboQuant Experiment Pipeline - Multi-model Proxy Study")
    log.info(f"Output directory: {run_dir}")
    log.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log.info(f"Method note: {cfg.proxy_note}")
    if resume_state:
        log.info("Resume mode: reusing latest incomplete run")
    log.info("=" * 60)

    results.setdefault("proxy_note", cfg.proxy_note)
    results.setdefault("model_runs", {})
    progress.setdefault("status", "running")
    progress.setdefault("phase1_complete", False)
    progress.setdefault("phase1_steps", [])
    progress.setdefault("models", {})
    progress["status"] = "running"
    save_run_state(run_dir, cfg, results, progress, ts)

    phase1_steps = [
        ("bit_tradeoff", experiment_bit_tradeoff),
        ("distortion_curve", experiment_distortion_curve),
        ("attention_entropy", experiment_attention_entropy),
        ("memory_scaling", experiment_memory_scaling),
    ]

    log.info("\n[PHASE 1] Algorithm characterisation (synthetic tensors)")
    completed_phase1 = set(progress.get("phase1_steps", []))
    for key, fn in phase1_steps:
        if key in results:
            completed_phase1.add(key)
            continue
        progress["current_stage"] = f"phase1:{key}"
        results[key] = fn(cfg, log)
        completed_phase1.add(key)
        progress["phase1_steps"] = sorted(completed_phase1)
        progress["phase1_complete"] = len(completed_phase1) == len(phase1_steps)
        save_run_state(run_dir, cfg, results, progress, ts)
    progress["phase1_steps"] = sorted(completed_phase1)
    progress["phase1_complete"] = len(completed_phase1) == len(phase1_steps)
    save_run_state(run_dir, cfg, results, progress, ts)

    log.info("\n[PHASE 2] Loading fixed external evaluator")
    progress["current_stage"] = "phase2:evaluator"
    save_run_state(run_dir, cfg, results, progress, ts)
    evaluator_model, evaluator_tokenizer = load_similarity_evaluator(cfg)

    required_model_keys = [
        "layer_sensitivity",
        "mixed_bit",
        "factual_qa_baseline",
        "factual_qa_tq3",
        "throughput",
        "multi_task",
        "rag_simulation",
    ]

    try:
        for spec in cfg.baseline_models:
            label = spec["label"]
            model_id = spec["model_id"]
            prompt_style = spec.get("prompt_style", "plain")

            model_result = results["model_runs"].setdefault(
                label,
                {
                    "label": label,
                    "model_id": model_id,
                    "prompt_style": prompt_style,
                },
            )
            model_result["label"] = label
            model_result["model_id"] = model_id
            model_result["prompt_style"] = prompt_style

            model_progress = progress["models"].setdefault(
                label,
                {"completed": False, "completed_keys": []},
            )
            model_progress["completed_keys"] = sorted(
                set(model_progress.get("completed_keys", []))
                | {key for key in required_model_keys if key in model_result}
            )
            model_progress["completed"] = model_result_complete(model_result)

            if model_progress["completed"]:
                log.info(f"\n[MODEL] {label} ({model_id}) - already complete, skipping")
                save_run_state(run_dir, cfg, results, progress, ts)
                continue

            log.info(f"\n[MODEL] {label} ({model_id})")
            progress["current_stage"] = f"model:{label}"
            save_run_state(run_dir, cfg, results, progress, ts)

            base_model = None
            tq_model = None
            tokenizer = None
            try:
                base_model, tokenizer = load_model(cfg, model_id=model_id, apply_tq=False)

                if "layer_sensitivity" not in model_result:
                    progress["current_stage"] = f"model:{label}:layer_sensitivity"
                    model_result["layer_sensitivity"] = experiment_layer_sensitivity(base_model, cfg, log)
                    model_progress["completed_keys"] = sorted(
                        set(model_progress["completed_keys"]) | {"layer_sensitivity"}
                    )
                    save_run_state(run_dir, cfg, results, progress, ts)

                if "mixed_bit" not in model_result:
                    progress["current_stage"] = f"model:{label}:mixed_bit"
                    model_result["mixed_bit"] = experiment_mixed_bit(
                        base_model,
                        model_result["layer_sensitivity"],
                        cfg,
                        log,
                    )
                    model_progress["completed_keys"] = sorted(
                        set(model_progress["completed_keys"]) | {"mixed_bit"}
                    )
                    save_run_state(run_dir, cfg, results, progress, ts)

                if "factual_qa_baseline" not in model_result:
                    progress["current_stage"] = f"model:{label}:factual_qa_baseline"
                    model_result["factual_qa_baseline"] = experiment_factual_qa(
                        base_model,
                        tokenizer,
                        evaluator_model,
                        evaluator_tokenizer,
                        cfg,
                        log,
                        label=f"{label}/baseline",
                        prompt_style=prompt_style,
                    )
                    model_progress["completed_keys"] = sorted(
                        set(model_progress["completed_keys"]) | {"factual_qa_baseline"}
                    )
                    save_run_state(run_dir, cfg, results, progress, ts)

                tq_needed = any(
                    key not in model_result
                    for key in ["factual_qa_tq3", "throughput", "multi_task", "rag_simulation"]
                )
                if tq_needed:
                    tq_model, _ = load_model(
                        cfg,
                        model_id=model_id,
                        apply_tq=True,
                        bits=cfg.quantized_bits,
                    )

                if "factual_qa_tq3" not in model_result:
                    progress["current_stage"] = f"model:{label}:factual_qa_tq3"
                    model_result["factual_qa_tq3"] = experiment_factual_qa(
                        tq_model,
                        tokenizer,
                        evaluator_model,
                        evaluator_tokenizer,
                        cfg,
                        log,
                        label=f"{label}/tq_{cfg.quantized_bits}bit",
                        prompt_style=prompt_style,
                    )
                    model_progress["completed_keys"] = sorted(
                        set(model_progress["completed_keys"]) | {"factual_qa_tq3"}
                    )
                    save_run_state(run_dir, cfg, results, progress, ts)

                if "throughput" not in model_result:
                    progress["current_stage"] = f"model:{label}:throughput"
                    pending_joint_eval = any(
                        key not in model_result for key in ["multi_task", "rag_simulation"]
                    )
                    base_model_ref, tq_model_ref, tokenizer_ref = base_model, tq_model, tokenizer
                    base_model = None
                    tq_model = None
                    tokenizer = None
                    _cleanup_loaded_models(base_model_ref, tq_model_ref, tokenizer_ref)
                    model_result["throughput"] = experiment_throughput_reloaded(
                        cfg, log, model_id=model_id, prompt_style=prompt_style
                    )
                    if pending_joint_eval:
                        base_model, tokenizer = load_model(
                            cfg, model_id=model_id, apply_tq=False
                        )
                        tq_model, _ = load_model(
                            cfg,
                            model_id=model_id,
                            apply_tq=True,
                            bits=cfg.quantized_bits,
                        )
                    model_progress["completed_keys"] = sorted(
                        set(model_progress["completed_keys"]) | {"throughput"}
                    )
                    save_run_state(run_dir, cfg, results, progress, ts)

                if "multi_task" not in model_result:
                    progress["current_stage"] = f"model:{label}:multi_task"
                    model_result["multi_task"] = experiment_multi_task(
                        base_model,
                        tq_model,
                        tokenizer,
                        evaluator_model,
                        evaluator_tokenizer,
                        cfg,
                        log,
                        prompt_style=prompt_style,
                    )
                    model_progress["completed_keys"] = sorted(
                        set(model_progress["completed_keys"]) | {"multi_task"}
                    )
                    save_run_state(run_dir, cfg, results, progress, ts)

                if "rag_simulation" not in model_result:
                    progress["current_stage"] = f"model:{label}:rag_simulation"
                    model_result["rag_simulation"] = experiment_rag_simulation(
                        base_model,
                        tq_model,
                        tokenizer,
                        cfg,
                        log,
                        prompt_style=prompt_style,
                    )
                    model_progress["completed_keys"] = sorted(
                        set(model_progress["completed_keys"]) | {"rag_simulation"}
                    )
                    save_run_state(run_dir, cfg, results, progress, ts)

                model_progress["completed"] = model_result_complete(model_result)
                model_progress["completed_keys"] = sorted(
                    set(model_progress["completed_keys"])
                    | {key for key in required_model_keys if key in model_result}
                )
                save_run_state(run_dir, cfg, results, progress, ts)
            finally:
                base_model_ref, tq_model_ref, tokenizer_ref = base_model, tq_model, tokenizer
                base_model = None
                tq_model = None
                tokenizer = None
                _cleanup_loaded_models(base_model_ref, tq_model_ref, tokenizer_ref)
    finally:
        evaluator_model_ref, evaluator_tokenizer_ref = evaluator_model, evaluator_tokenizer
        evaluator_model = None
        evaluator_tokenizer = None
        _cleanup_loaded_models(evaluator_model_ref, evaluator_tokenizer_ref)

    _refresh_primary_views(results, cfg)
    save_run_state(run_dir, cfg, results, progress, ts)

    progress["status"] = "plotting"
    progress["current_stage"] = "phase9:plotting"
    save_run_state(run_dir, cfg, results, progress, ts)
    log.info("\n[PHASE 9] Generating plots")
    plot_all(results, run_dir, cfg)

    progress["status"] = "completed"
    progress["current_stage"] = "completed"
    save_run_state(run_dir, cfg, results, progress, ts)

    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT SUMMARY")
    log.info("=" * 60)
    log.info(f"Primary model for detailed plots: {results['primary_model_label']}")
    for label, summary in results["model_suite_summary"].items():
        log.info(
            f"{label}: factual {summary['factual_baseline_accuracy']:.0%}->{summary['factual_tq_accuracy']:.0%}  "
            f"rag {summary['rag_baseline_accuracy']:.0%}->{summary['rag_tq_accuracy']:.0%}  "
            f"tps_ratio={summary['throughput_ratio']:.3f}  "
            f"mixed_bit_gain={summary['mixed_bit_improvement_pct']:.1f}%"
        )
    log.info(f"\nAll outputs saved to: {run_dir}")
    log.info("=" * 60)
    return results, run_dir


def experiment_throughput_reloaded(cfg: Config, log, model_id: str, prompt_style: str = "plain") -> dict:
    """Run the robust latency/throughput sweep used by the resumable pipeline."""
    return _experiment_throughput_reloaded_sweep(
        cfg,
        log,
        model_id=model_id,
        prompt_style=prompt_style,
    )

    """Benchmark baseline and TQ throughput with one model resident at a time."""
    log.info("E5 - Throughput benchmark")

    base_model = None
    tq_model = None
    tokenizer = None
    tq_tokenizer = None

    try:
        base_model, tokenizer = load_model(cfg, model_id=model_id, apply_tq=False)
        if model_is_offloaded(base_model):
            log.warning("Baseline throughput model is offloaded; timing may still be distorted.")
        base_stats = measure_loaded_model_throughput(
            base_model,
            tokenizer,
            throughput_prompt(tokenizer, prompt_style),
            cfg.throughput_trials,
            cfg,
        )
    finally:
        base_model_ref, tokenizer_ref = base_model, tokenizer
        base_model = None
        tokenizer = None
        _cleanup_loaded_models(base_model_ref, tokenizer_ref)

    try:
        tq_model, tq_tokenizer = load_model(
            cfg, model_id=model_id, apply_tq=True, bits=cfg.quantized_bits
        )
        if model_is_offloaded(tq_model):
            log.warning("Quantized throughput model is offloaded; timing may still be distorted.")
        tq_stats = measure_loaded_model_throughput(
            tq_model,
            tq_tokenizer,
            throughput_prompt(tq_tokenizer, prompt_style),
            cfg.throughput_trials,
            cfg,
        )
    finally:
        tq_model_ref, tq_tokenizer_ref = tq_model, tq_tokenizer
        tq_model = None
        tq_tokenizer = None
        _cleanup_loaded_models(tq_model_ref, tq_tokenizer_ref)

    result = {
        "benchmark_mode": getattr(cfg, "throughput_benchmark_mode", "fixed_step_decode"),
        "decode_steps": cfg.throughput_tokens,
        "prompt_tokens": base_stats["prompt_tokens"],
        "baseline_tps": round(base_stats["decode_tps_mean"], 2),
        "baseline_tps_std": round(base_stats["decode_tps_std"], 2),
        "baseline_full_tps": round(base_stats["full_tps_mean"], 2),
        "baseline_full_tps_std": round(base_stats["full_tps_std"], 2),
        "baseline_prefill_ms": round(base_stats["prefill_ms_mean"], 2),
        "baseline_prefill_ms_std": round(base_stats["prefill_ms_std"], 2),
        "tq3_tps": round(tq_stats["decode_tps_mean"], 2),
        "tq3_tps_std": round(tq_stats["decode_tps_std"], 2),
        "tq3_full_tps": round(tq_stats["full_tps_mean"], 2),
        "tq3_full_tps_std": round(tq_stats["full_tps_std"], 2),
        "tq3_prefill_ms": round(tq_stats["prefill_ms_mean"], 2),
        "tq3_prefill_ms_std": round(tq_stats["prefill_ms_std"], 2),
        "tps_ratio": round(
            tq_stats["decode_tps_mean"] / max(base_stats["decode_tps_mean"], 1e-9), 3
        ),
    }
    log.info(
        f"  Baseline decode: {base_stats['decode_tps_mean']:.1f} +/- "
        f"{base_stats['decode_tps_std']:.1f} tok/s  "
        f"(prefill {base_stats['prefill_ms_mean']:.1f} +/- "
        f"{base_stats['prefill_ms_std']:.1f} ms)"
    )
    log.info(
        f"  TQ-3bit decode:  {tq_stats['decode_tps_mean']:.1f} +/- "
        f"{tq_stats['decode_tps_std']:.1f} tok/s  "
        f"(prefill {tq_stats['prefill_ms_mean']:.1f} +/- "
        f"{tq_stats['prefill_ms_std']:.1f} ms)"
    )
    log.info(f"  Ratio:      {result['tps_ratio']:.3f}")
    return result


_previous_plot_all = plot_all


def plot_all(results: dict, run_dir: str, cfg: Config):
    _previous_plot_all(results, run_dir, cfg)

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    BLUE = "#2563EB"
    ORANGE = "#F97316"
    GREEN = "#16A34A"
    RED = "#DC2626"
    GREY = "#6B7280"

    def save(name):
        plt.savefig(os.path.join(plot_dir, name), dpi=cfg.fig_dpi, bbox_inches="tight")
        plt.close()

    def normalize_numeric_keyed_dict(d):
        normalized = {}
        for key, value in d.items():
            try:
                normalized[int(key)] = value
            except (TypeError, ValueError):
                normalized[key] = value
        return normalized

    ms = normalize_numeric_keyed_dict(results["memory_scaling"])
    ctxs = sorted(ms.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ctxs, [ms[c]["fp16_gb"] for c in ctxs], "o-", color=RED, label="FP16 baseline")
    axes[0].plot(ctxs, [ms[c]["4bit_gb"] for c in ctxs], "s-", color=ORANGE, label="Ideal 4-bit target")
    axes[0].plot(ctxs, [ms[c]["3bit_gb"] for c in ctxs], "^-", color=BLUE, label="Ideal 3-bit target")
    axes[0].plot(ctxs, [ms[c]["2bit_gb"] for c in ctxs], "D-", color=GREEN, label="Ideal 2-bit target")
    axes[0].set_xlabel("Context Length (tokens)")
    axes[0].set_ylabel("KV Cache Memory (GB)")
    axes[0].set_title("Idealized KV Memory Targets")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ctxs, [ms[c]["fp16_gb"] for c in ctxs], "o-", color=RED, label="FP16 baseline")
    axes[1].plot(ctxs, [ms[c]["runtime_int8_scale_gb"] for c in ctxs], "s-", color=BLUE, label="Runtime prototype")
    axes[1].plot(ctxs, [ms[c]["3bit_gb"] for c in ctxs], "--", color=GREY, label="Ideal 3-bit target")
    axes[1].set_xlabel("Context Length (tokens)")
    axes[1].set_ylabel("KV Cache Memory (GB)")
    axes[1].set_title("Measured Prototype Storage Model")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("E4 - KV Cache Memory: Ideal Target vs Runtime Prototype", fontsize=13)
    plt.tight_layout()
    save("E4_memory_scaling.png")

    primary_label = results.get("primary_model_label")
    primary_run = results.get("model_runs", {}).get(primary_label, {})
    primary_throughput = primary_run.get("throughput", results.get("throughput", {}))
    if primary_throughput.get("per_prompt"):
        per_prompt = normalize_numeric_keyed_dict(primary_throughput["per_prompt"])
        targets = sorted(k for k in per_prompt.keys() if isinstance(k, int))
        actual_tokens = [
            per_prompt[t]["baseline"]["prompt_tokens_median"]
            for t in targets
        ]
        ratio_vals = [per_prompt[t]["tps_ratio_median"] for t in targets]
        base_latency = [
            per_prompt[t]["baseline"]["decode_ms_per_token"]["median"]
            for t in targets
        ]
        tq_latency = [
            per_prompt[t]["tq3"]["decode_ms_per_token"]["median"]
            for t in targets
        ]
        base_prefill = [
            per_prompt[t]["baseline"]["prefill_ms"]["median"]
            for t in targets
        ]
        tq_prefill = [
            per_prompt[t]["tq3"]["prefill_ms"]["median"]
            for t in targets
        ]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(actual_tokens, ratio_vals, "o-", color=GREEN)
        axes[0].axhline(1.0, color=GREY, linestyle="--", linewidth=1)
        axes[0].set_xlabel("Prompt tokens")
        axes[0].set_ylabel("TQ / baseline")
        axes[0].set_title("Median Decode Throughput Ratio")
        axes[0].grid(alpha=0.3)

        axes[1].plot(actual_tokens, base_latency, "o-", color=BLUE, label="Baseline")
        axes[1].plot(actual_tokens, tq_latency, "s-", color=ORANGE, label="TQ-3bit")
        axes[1].set_xlabel("Prompt tokens")
        axes[1].set_ylabel("ms / decoded token")
        axes[1].set_title("Median Decode Latency")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        axes[2].plot(actual_tokens, base_prefill, "o-", color=BLUE, label="Baseline")
        axes[2].plot(actual_tokens, tq_prefill, "s-", color=ORANGE, label="TQ-3bit")
        axes[2].set_xlabel("Prompt tokens")
        axes[2].set_ylabel("ms")
        axes[2].set_title("Median Prefill / TTFT")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        fig.suptitle(
            f"E5 - Latency/Throughput Sweep ({primary_label})",
            fontsize=13,
        )
        plt.tight_layout()
        save("E5_latency_throughput.png")

    suite_labels = list(results["model_runs"].keys())
    suite_runs = [results["model_runs"][label] for label in suite_labels]
    suite_tasks = list(suite_runs[0]["multi_task"].keys())
    x_pos = np.arange(len(suite_labels))
    width = 0.36

    fig = plt.figure(figsize=(15, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)

    ax0 = fig.add_subplot(gs[0, 0])
    ms16 = ms[16384]
    ax0.bar(
        ["FP16", "Ideal 3-bit", "Runtime proto"],
        [ms16["fp16_gb"], ms16["3bit_gb"], ms16["runtime_int8_scale_gb"]],
        color=[RED, BLUE, GREEN],
        alpha=0.8,
    )
    ax0.set_title("KV Cache @ 16K ctx (GB)", fontsize=10)
    ax0.set_ylabel("GB")
    ax0.grid(axis="y", alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    factual_base_vals = [run["factual_qa_baseline"]["accuracy"] for run in suite_runs]
    factual_tq_vals = [run["factual_qa_tq3"]["accuracy"] for run in suite_runs]
    ax1.bar(x_pos - width / 2, factual_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax1.bar(x_pos + width / 2, factual_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Factual Accuracy by Model", fontsize=10)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    heatmap = np.array([[run["multi_task"][task]["avg_similarity"] for task in suite_tasks] for run in suite_runs])
    im = ax2.imshow(heatmap, cmap="Blues", vmin=0.0, vmax=max(0.6, float(np.max(heatmap))))
    ax2.set_xticks(range(len(suite_tasks)))
    ax2.set_xticklabels([task.capitalize() for task in suite_tasks], rotation=25, ha="right", fontsize=8)
    ax2.set_yticks(range(len(suite_labels)))
    ax2.set_yticklabels(suite_labels, fontsize=8)
    ax2.set_title("Multi-task Similarity by Model", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 0])
    rag_base_vals = [run["rag_simulation"]["baseline_accuracy"] for run in suite_runs]
    rag_tq_vals = [run["rag_simulation"]["tq_accuracy"] for run in suite_runs]
    ax3.bar(x_pos - width / 2, rag_base_vals, width, color=BLUE, alpha=0.8, label="Baseline")
    ax3.bar(x_pos + width / 2, rag_tq_vals, width, color=ORANGE, alpha=0.8, label="TQ-3bit")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("RAG Accuracy by Model", fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(suite_labels, rotation=15, ha="right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    throughput_ratios = [run["throughput"]["tps_ratio"] for run in suite_runs]
    ax4.bar(suite_labels, throughput_ratios, color=GREEN, alpha=0.8)
    ax4.axhline(1.0, color=GREY, linestyle="--", linewidth=1)
    ax4.set_title("Decode Throughput Ratio (TQ/Baseline)", fontsize=10)
    ax4.set_ylabel("Ratio")
    ax4.tick_params(axis="x", rotation=15, labelsize=8)
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    mixed_gains = [run["mixed_bit"]["mse_improvement_pct"] for run in suite_runs]
    ax5.bar(suite_labels, mixed_gains, color=BLUE, alpha=0.8)
    ax5.set_title("Mixed-bit MSE Improvement", fontsize=10)
    ax5.set_ylabel("Percent")
    ax5.tick_params(axis="x", rotation=15, labelsize=8)
    ax5.grid(axis="y", alpha=0.3)

    fig.suptitle("TurboQuant Suite Summary - Runtime Prototype Evaluation", fontsize=14, y=1.01)
    save("SUMMARY_dashboard.png")


if __name__ == "__main__":
    run_all()
