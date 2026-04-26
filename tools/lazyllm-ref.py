"""
Phase 0 Python reference oracle for LazyLLM (arXiv:2407.14057).

Monkey-patches HuggingFace LlamaModel.forward to implement LazyLLM
progressive token pruning, dumps per-layer importance scores, and evaluates
LongBench quality. Oracle output (scores) is saved as an .npz file for use in
C++ Phase 1 validation.

Algorithm overview (§3 of the paper):
  At each configured pruning-point layer l_prune:
    1. Extract attention weights from the last query token to all prior tokens.
    2. Average over heads to get per-token importance scores (Eq. 1).
    3. Optionally smooth with 1-D average pooling (§3.2).
    4. Drop the bottom (1 - keep_ratio) tokens; kept tokens retain original
       position IDs so RoPE is applied at the correct positions.
  Subsequent layers run only on the kept token set.
"""

import argparse
import json
import logging
import math
import re
import string
import sys
import types
from collections import Counter

# Suppress broken flash_attn import (undefined symbol in this environment).
# Must be done before any transformers import that loads qwen2/llama modules.
def _make_stub_module(name: str, symbols: list) -> "types.ModuleType":
    """Create a minimal stub module with no-op callables for given symbols."""
    import importlib.util  # noqa: PLC0415
    def _noop(*args, **kwargs):  # noqa: ANN002
        return None

    m = types.ModuleType(name)
    m.__file__    = f"<stub:{name}>"
    m.__package__ = name.split(".")[0]
    m.__path__    = []
    m.__loader__  = None
    m.__spec__    = importlib.util.spec_from_loader(name, loader=None)
    for sym in symbols:
        setattr(m, sym, _noop)
    return m


def _fix_broken_deps() -> None:
    """
    Stub broken native extensions before any transformers/model imports.
    In this environment:
      - flash_attn_2_cuda.so has an undefined C++ symbol.
      - torchvision is compiled against an older torch ABI (torchvision::nms missing).
    Both packages are pre-stubbed in sys.modules so transformers loads cleanly.
    """
    _FA_SYMS = [
        "flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache",
        "flash_attn_varlen_qkvpacked_func", "flash_attn_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func", "pad_input", "unpad_input",
        "apply_rotary_emb", "FlashAttentionKwargs",
    ]
    for mod_name in [
        "flash_attn", "flash_attn.flash_attn_interface", "flash_attn.bert_padding",
        "flash_attn.layers", "flash_attn.layers.rotary",
        "flash_attn.flash_attn_triton", "flash_attn_2_cuda",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub_module(mod_name, _FA_SYMS)

    # Stub torchvision and key submodules (ABI mismatch with installed torch).
    # InterpolationMode must be an enum-like object (accessed as InterpolationMode.NEAREST etc.)
    class _InterpolationMode:
        NEAREST = NEAREST_EXACT = BILINEAR = BICUBIC = LANCZOS = BOX = HAMMING = None
    _noop = lambda *a, **kw: None  # noqa: E731

    _TV_SYMS = ["Compose", "Resize", "ToTensor", "Normalize", "CenterCrop",
                "RandomHorizontalFlip", "RandomCrop", "RandomResizedCrop",
                "nms", "roi_align", "center_to_corners_format"]
    for mod_name in [
        "torchvision", "torchvision.transforms", "torchvision.transforms.functional",
        "torchvision.ops", "torchvision.datasets", "torchvision.models",
        "torchvision.io", "torchvision.utils", "torchvision._meta_registrations",
    ]:
        if mod_name not in sys.modules:
            m = _make_stub_module(mod_name, _TV_SYMS)
            m.InterpolationMode = _InterpolationMode  # proper enum-like object
            sys.modules[mod_name] = m


_fix_broken_deps()
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smoke prompts used for the oracle .npz save
# ---------------------------------------------------------------------------
SMOKE_PROMPTS = [
    "In 2024, OpenAI released GPT-4o. What does the 'o' in GPT-4o stand for?",
    "The capital of France is Paris. What river runs through Paris?",
    "Python was created by Guido van Rossum. In which year was Python first released?",
    "The Eiffel Tower is located in Paris, France. How tall is the Eiffel Tower in meters?",
    "Albert Einstein developed the theory of relativity. What is the famous equation from special relativity?",
]


# ---------------------------------------------------------------------------
# Core helper: attention-score extraction (paper Eq. 1)
# ---------------------------------------------------------------------------

def extract_layer_scores(attn_weights: torch.Tensor, last_pos: int) -> np.ndarray:
    """Return per-token importance scores from a single decoder layer.

    Args:
        attn_weights: Tensor of shape [batch, n_heads, n_tokens, n_tokens].
            Each entry [b, h, q, k] is the attention probability that query q
            attends to key k (post-softmax inside the model).
        last_pos: Index of the last (most recent) query token position.

    Returns:
        np.ndarray of shape [n_tokens] with mean-head attention score from the
        last token to every other token (batch dimension collapsed, batch=1).
    """
    # attn_weights: [1, n_heads, seq, seq]  (batch=1 assumed)
    # Select the row corresponding to the last query token
    row = attn_weights[:, :, last_pos, :]  # [1, n_heads, n_tokens]
    scores = row.mean(dim=1)               # [1, n_tokens]
    return scores[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Average pooling smoothing (§3.2)
# ---------------------------------------------------------------------------

def apply_avg_pool(scores: np.ndarray, pool_size: int) -> np.ndarray:
    """Smooth importance scores with 1-D average pooling.

    Args:
        scores: 1-D array of length n_tokens.
        pool_size: Kernel size.  padding = pool_size // 2  so output length
            equals input length.  If pool_size <= 1 the array is returned
            unchanged.

    Returns:
        Smoothed 1-D array of the same length.
    """
    if pool_size <= 1:
        return scores

    n = len(scores)
    t = torch.tensor(scores, dtype=torch.float32).view(1, 1, n)  # [1, 1, n]
    padding = pool_size // 2
    pooled = F.avg_pool1d(t, kernel_size=pool_size, stride=1, padding=padding)
    # avg_pool1d with even kernel and symmetric padding may add one extra element
    return pooled[0, 0, :n].numpy()


# ---------------------------------------------------------------------------
# Token pruning
# ---------------------------------------------------------------------------

def prune_tokens(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    scores: np.ndarray,
    keep_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Drop low-importance tokens from the hidden state tensor.

    The last token is always kept regardless of its score (it is the query
    position and must survive all pruning rounds).  Kept tokens are returned
    in their original (positional) order so downstream layers see a coherent
    sequence; the original position IDs are preserved so RoPE sees the correct
    offsets.

    Args:
        hidden_states: [batch, n_tokens, hidden_dim]
        position_ids:  [batch, n_tokens]  or  [1, n_tokens]
        scores:        np.ndarray of shape [n_tokens]
        keep_ratio:    Fraction of tokens to keep (0 < keep_ratio <= 1).

    Returns:
        (pruned_hidden_states, pruned_position_ids, kept_indices)
        where kept_indices is a sorted list of the original token positions
        that were retained.
    """
    n_tokens = hidden_states.shape[1]
    n_keep = max(1, math.ceil(n_tokens * keep_ratio))

    last_idx = n_tokens - 1

    # Build a score array; force the last token to have +inf so it is always
    # selected.
    score_arr = scores.copy().astype(float)
    score_arr[last_idx] = float("inf")

    # Top-n_keep indices by score (descending)
    top_indices = np.argsort(score_arr)[::-1][:n_keep].tolist()

    # Restore original positional order
    kept_indices = sorted(top_indices)

    idx_tensor = torch.tensor(kept_indices, dtype=torch.long, device=hidden_states.device)
    pruned_hidden = hidden_states[:, idx_tensor, :]       # [batch, n_keep, hidden_dim]
    pruned_pos_ids = position_ids[:, idx_tensor]          # [batch, n_keep]

    return pruned_hidden, pruned_pos_ids, kept_indices


# ---------------------------------------------------------------------------
# LazyLLM model wrapper
# ---------------------------------------------------------------------------

class LazyLLMModel:
    """Wraps a HuggingFace LlamaForCausalLM with LazyLLM progressive pruning.

    Pruning is applied at the *output* of the specified layers: after layer
    ``pruning_layers[i]`` the bottom ``1 - keep_ratios[i]`` tokens are
    discarded.  Subsequent layers run on the reduced sequence.

    Usage::

        lazy = LazyLLMModel(hf_model, pruning_layers=[8, 16], keep_ratios=[0.7, 0.5])
        out, all_scores = lazy.forward(input_ids, collect_scores=True)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pruning_layers: List[int],
        keep_ratios: List[float],
        pool_size: int = 13,
    ) -> None:
        if len(pruning_layers) != len(keep_ratios):
            raise ValueError("pruning_layers and keep_ratios must have the same length")

        self.model = model
        self.pruning_layers = pruning_layers
        self.keep_ratios = keep_ratios
        self.pool_size = pool_size

        # Map layer_index -> keep_ratio for O(1) lookup
        self._prune_map: Dict[int, float] = dict(zip(pruning_layers, keep_ratios))

    # ------------------------------------------------------------------
    # Internal forward that runs layer-by-layer with optional pruning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        collect_scores: bool = False,
    ) -> Tuple[torch.Tensor, Dict[int, np.ndarray]]:
        """Run the model with LazyLLM pruning.

        Args:
            input_ids:      [1, seq_len]
            collect_scores: If True, accumulate raw (pre-pool) attention scores
                            at each pruning-point layer.

        Returns:
            (logits [1, n_kept, vocab], scores_by_layer)
            where scores_by_layer maps layer_idx -> np.ndarray[n_tokens_at_that_point].
        """
        llama_model = self.model.model  # Qwen2Model / LlamaModel

        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # --- Embedding ---
        hidden_states = llama_model.embed_tokens(input_ids)  # [1, seq, hidden]

        # Build initial position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        scores_by_layer: Dict[int, np.ndarray] = {}

        # --- Layer-by-layer forward ---
        for layer_idx, decoder_layer in enumerate(llama_model.layers):
            current_seq_len = hidden_states.shape[1]
            last_pos = current_seq_len - 1

            # Compute rotary position embeddings for the current token set.
            # Newer transformers (4.46+) passes position_embeddings as (cos, sin)
            # to each decoder layer rather than computing them inside the layer.
            position_embeddings = None
            if hasattr(llama_model, "rotary_emb"):
                position_embeddings = llama_model.rotary_emb(hidden_states, position_ids)

            # Run this layer with attention output so we can extract scores
            need_attn = collect_scores and (layer_idx in self._prune_map)

            # Build kwargs defensively — older models don't have position_embeddings param
            layer_kwargs: Dict[str, object] = dict(
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=need_attn,
                use_cache=False,
            )
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_out = decoder_layer(hidden_states, **layer_kwargs)

            hidden_states = layer_out[0]  # [batch, seq, hidden]

            if need_attn:
                # layer_out[1] is attn_weights when output_attentions=True
                attn_weights = layer_out[1]  # [1, n_heads, seq, seq]

                raw_scores = extract_layer_scores(attn_weights, last_pos)
                smoothed = apply_avg_pool(raw_scores, self.pool_size)

                if collect_scores:
                    scores_by_layer[layer_idx] = raw_scores

                keep_ratio = self._prune_map[layer_idx]
                hidden_states, position_ids, kept = prune_tokens(
                    hidden_states, position_ids, smoothed, keep_ratio
                )
                logger.debug(
                    "Layer %d: pruned to %d / %d tokens (keep_ratio=%.2f)",
                    layer_idx,
                    len(kept),
                    current_seq_len,
                    keep_ratio,
                )

        # --- Final norm + LM head ---
        hidden_states = llama_model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        return logits, scores_by_layer

    # ------------------------------------------------------------------
    # Generation helper (greedy)
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Greedy generation using the pruned forward pass.

        Only the *last* logit position is used for each step; new tokens are
        appended to the *original* (unpruned) prefix so each step restarts
        with the full pruning pipeline.  This matches the paper's decoding
        scheme where pruning applies to the prefill phase only.
        """
        generated = input_ids.clone()
        eos_id = getattr(self.model.config, "eos_token_id", 2)

        for _ in range(max_new_tokens):
            logits, _ = self.forward(generated, collect_scores=False)
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        return generated


# ---------------------------------------------------------------------------
# LongBench evaluation utilities
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens (SQuAD-style)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    """Compute token-level F1 between prediction and the best gold answer.

    Follows the standard SQuAD evaluation metric: normalise both strings by
    lowercasing and removing punctuation, then compute token overlap F1.  The
    score against each gold answer is computed and the maximum is returned.
    """
    pred_tokens = _normalize_text(prediction)
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold)
        if not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue

        precision = n_common / len(pred_tokens)
        recall = n_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def evaluate_longbench(
    lazy_model: Optional[LazyLLMModel],
    hf_model: torch.nn.Module,
    tokenizer,
    dataset_dir: str,
    n_examples: int,
    max_new_tokens: int,
    baseline: bool = False,
) -> dict:
    """Evaluate on LongBench multi-doc QA examples.

    Args:
        lazy_model:     LazyLLMModel wrapper (None when baseline=True).
        hf_model:       Raw HuggingFace model (used for baseline).
        tokenizer:      HuggingFace tokenizer.
        dataset_dir:    Path to directory containing hotpotqa_e.jsonl.
        n_examples:     Number of examples to evaluate.
        max_new_tokens: Tokens to generate per example.
        baseline:       If True use hf_model.generate directly (no pruning).

    Returns:
        dict with keys "mean_f1", "per_example" (list of per-example dicts).
    """
    dataset_path = Path(dataset_dir) / "hotpotqa_e.jsonl"
    if not dataset_path.exists():
        # Fall back to any available JSONL file in the directory
        candidates = list(Path(dataset_dir).glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL files found in {dataset_dir}")
        dataset_path = candidates[0]
        logger.warning("hotpotqa_e.jsonl not found; using %s", dataset_path)

    examples = []
    with open(dataset_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    examples = examples[:n_examples]
    logger.info("Evaluating on %d examples from %s", len(examples), dataset_path)

    device = next(hf_model.parameters()).device
    per_example = []
    eos_id = getattr(hf_model.config, "eos_token_id", 2)

    for ex_idx, example in enumerate(examples):
        prompt_text = example.get("input", "")
        gold_answers = example.get("answers", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        if baseline:
            out_ids = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                eos_token_id=eos_id,
            )
            generated_ids = out_ids[0, input_ids.shape[1]:]
        else:
            out_ids = lazy_model.generate(input_ids, max_new_tokens=max_new_tokens)
            generated_ids = out_ids[0, input_ids.shape[1]:]

        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
        ex_f1 = f1_score(prediction, gold_answers)

        per_example.append({
            "_id": example.get("_id", ex_idx),
            "length": example.get("length", -1),
            "f1": ex_f1,
            "prediction": prediction,
            "gold": gold_answers,
        })

        logger.info(
            "[%d/%d] _id=%s  length=%s  f1=%.3f",
            ex_idx + 1,
            len(examples),
            example.get("_id", ex_idx),
            example.get("length", "?"),
            ex_f1,
        )

    mean_f1 = float(np.mean([e["f1"] for e in per_example])) if per_example else 0.0
    logger.info("Mean F1: %.4f over %d examples", mean_f1, len(per_example))

    return {"mean_f1": mean_f1, "per_example": per_example}


# ---------------------------------------------------------------------------
# Oracle score saving
# ---------------------------------------------------------------------------

def save_oracle_scores(
    scores_by_prompt: Dict[int, Dict[int, np.ndarray]],
    output_path: str,
) -> None:
    """Serialise per-layer attention scores to an .npz archive.

    The key schema is ``prompt{i}_layer{j}`` where ``i`` is the prompt index
    (into SMOKE_PROMPTS) and ``j`` is the layer index at which pruning was
    applied.

    Args:
        scores_by_prompt: {prompt_idx: {layer_idx: np.ndarray[n_tokens]}}
        output_path:      Destination .npz path.
    """
    arrays: Dict[str, np.ndarray] = {}
    for prompt_idx, layer_dict in scores_by_prompt.items():
        for layer_idx, arr in layer_dict.items():
            key = f"prompt{prompt_idx}_layer{layer_idx}"
            arrays[key] = arr

    np.savez(output_path, **arrays)
    logger.info("Saved oracle scores to %s", output_path)
    print(f"Saved oracle scores to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0 LazyLLM reference oracle — progressive token pruning for Llama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument(
        "--pruning-layers",
        nargs="+",
        type=int,
        default=[8, 16, 24],
        metavar="L",
        help="0-indexed decoder layer indices where pruning is applied",
    )
    parser.add_argument(
        "--keep-ratios",
        nargs="+",
        type=float,
        default=[0.7, 0.5, 0.3],
        metavar="R",
        help="Keep ratio per pruning point (must match --pruning-layers count)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=13,
        help="Avg-pool kernel size for score smoothing (§3.2); <=1 disables",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt text")
    parser.add_argument(
        "--dump-scores",
        type=int,
        default=None,
        metavar="PROMPT_ID",
        help="Tokenize --prompt, run forward with output_attentions, print scores as JSON",
    )
    parser.add_argument(
        "--save-oracle",
        type=str,
        default=None,
        metavar="PATH.npz",
        help="Run all SMOKE_PROMPTS and save per-layer scores to this .npz file",
    )
    parser.add_argument(
        "--eval-longbench",
        type=str,
        default=None,
        metavar="DIR",
        help="Evaluate on LongBench JSONL files in this directory",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=50,
        help="Number of LongBench examples to evaluate",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run without pruning (for comparison)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Tokens to generate during evaluation",
    )
    return parser


def _dtype_from_str(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def _load_model_and_tokenizer(args):
    """Load HuggingFace model + tokenizer according to CLI flags."""
    dtype = _dtype_from_str(args.dtype)
    logger.info("Loading tokenizer from %s", args.model)

    # Import model classes directly from their submodules to avoid the
    # transformers lazy-import system which fails when optional deps (Gemma3n,
    # torchvision, flash_attn) are broken in this environment.
    model_lower = args.model.lower()
    from transformers import AutoTokenizer  # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    logger.info("Loading model from %s  dtype=%s  device=%s", args.model, args.dtype, args.device)

    if "qwen" in model_lower:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: PLC0415
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config   # noqa: PLC0415
        model = Qwen2ForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=args.device,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    elif "llama" in model_lower or "meta-llama" in model_lower:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM  # noqa: PLC0415
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=args.device,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    else:
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM  # noqa: PLC0415
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=args.device,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    model.eval()
    return model, tokenizer


def _collect_scores_for_prompt(
    lazy_model: LazyLLMModel,
    tokenizer,
    prompt: str,
    device: str,
) -> Dict[int, np.ndarray]:
    """Tokenize prompt and collect per-layer attention scores."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    _, scores = lazy_model.forward(input_ids, collect_scores=True)
    return scores


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if len(args.pruning_layers) != len(args.keep_ratios):
        parser.error("--pruning-layers and --keep-ratios must have the same number of elements")

    model, tokenizer = _load_model_and_tokenizer(args)
    device = args.device

    if args.baseline:
        logger.info("Baseline mode: no pruning hooks installed")
        lazy_model = None
    else:
        lazy_model = LazyLLMModel(
            model,
            pruning_layers=args.pruning_layers,
            keep_ratios=args.keep_ratios,
            pool_size=args.pool_size,
        )

    # --- Dump scores for a single prompt to stdout as JSON ---
    if args.dump_scores is not None:
        if args.prompt is None:
            parser.error("--dump-scores requires --prompt")
        if lazy_model is None:
            parser.error("--dump-scores is incompatible with --baseline (no pruning)")

        logger.info("Collecting scores for prompt id %d", args.dump_scores)
        scores = _collect_scores_for_prompt(lazy_model, tokenizer, args.prompt, device)
        output = {
            "prompt_id": args.dump_scores,
            "prompt": args.prompt,
            "layer_scores": {
                str(layer_idx): arr.tolist() for layer_idx, arr in sorted(scores.items())
            },
        }
        print(json.dumps(output, indent=2))

    # --- Save oracle scores (.npz) for all smoke prompts ---
    if args.save_oracle is not None:
        if lazy_model is None:
            parser.error("--save-oracle is incompatible with --baseline")

        logger.info("Collecting scores for %d smoke prompts", len(SMOKE_PROMPTS))
        scores_by_prompt: Dict[int, Dict[int, np.ndarray]] = {}
        for prompt_idx, prompt_text in enumerate(SMOKE_PROMPTS):
            logger.info("  Smoke prompt %d: %.60s...", prompt_idx, prompt_text)
            scores_by_prompt[prompt_idx] = _collect_scores_for_prompt(
                lazy_model, tokenizer, prompt_text, device
            )

        # Ensure parent directory exists
        Path(args.save_oracle).parent.mkdir(parents=True, exist_ok=True)
        save_oracle_scores(scores_by_prompt, args.save_oracle)

    # --- LongBench evaluation ---
    if args.eval_longbench is not None:
        results = evaluate_longbench(
            lazy_model=lazy_model,
            hf_model=model,
            tokenizer=tokenizer,
            dataset_dir=args.eval_longbench,
            n_examples=args.n_examples,
            max_new_tokens=args.max_new_tokens,
            baseline=args.baseline,
        )
        summary = {
            "model": args.model,
            "pruning_layers": args.pruning_layers if not args.baseline else [],
            "keep_ratios": args.keep_ratios if not args.baseline else [],
            "pool_size": args.pool_size,
            "baseline": args.baseline,
            "n_examples": len(results["per_example"]),
            "mean_f1": results["mean_f1"],
        }
        print(json.dumps(summary, indent=2))

    if args.dump_scores is None and args.save_oracle is None and args.eval_longbench is None:
        parser.print_help()


if __name__ == "__main__":
    main()
