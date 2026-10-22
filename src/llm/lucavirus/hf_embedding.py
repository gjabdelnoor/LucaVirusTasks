"""Frozen embeddings from local Hugging Face LucaVirus checkpoints.

Implements the matrix/BOS portion of get_embedding.predict_embedding's contract.
The checkpoint's Python files execute via trust_remote_code: use trusted files.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def is_hf_checkpoint(path):
    if not isinstance(path, (str, os.PathLike)):
        return False
    config = Path(path).expanduser() / "config.json"
    if not config.is_file():
        return False
    with config.open() as stream:
        data = json.load(stream)
    return data.get("model_type") == "lucavirus" and "AutoModel" in data.get("auto_map", {})


def _rotary_dtype_hook(module, inputs, output):
    # The released HF rotary implementation uses float32 cos/sin even in fp16.
    # SDPA requires q, k and v to have the same dtype. Scope the fix to this model.
    return output[0].to(inputs[0].dtype), output[1].to(inputs[1].dtype)


@lru_cache(maxsize=1)
def load_hf_model(path, fp16=False):
    if not is_hf_checkpoint(path):
        raise ValueError("Expected a local Hugging Face LucaVirus checkpoint directory")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model, info = AutoModel.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        output_loading_info=True, low_cpu_mem_usage=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete or incompatible HF backbone weights: %s" % info)
    # A masked-LM checkpoint includes an unused lm_head when loaded as AutoModel.
    unexpected = [key for key in info.get("unexpected_keys", []) if not key.startswith("lm_head.")]
    if unexpected:
        raise ValueError("Unexpected HF checkpoint weights: %s" % unexpected)
    if fp16:
        for module in model.modules():
            if type(module).__name__ == "LucaVirusRotaryEmbedding":
                module.register_forward_hook(_rotary_dtype_hook)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def predict_embedding_hf(llm_dirpath, sample, trunc_type, embedding_type,
                         repr_layers=(-1,), truncation_seq_length=4094,
                         device=None, matrix_add_special_token=False, fp16=False):
    if "contacts" in embedding_type:
        raise ValueError("HF adapter does not implement the legacy contact head")
    want_matrix = "matrix" in embedding_type or "representations" in embedding_type
    want_vector = "vector" in embedding_type or "bos" in embedding_type
    if not (want_matrix or want_vector):
        raise ValueError("Choose matrix/representations or vector/bos embeddings")
    if trunc_type not in ("left", "right") or truncation_seq_length <= 0:
        raise ValueError("Use left/right truncation and a positive sequence length")
    if len(sample) == 2:
        _, seq = sample
        seq_type = "prot"
    elif len(sample) == 3:
        _, seq_type, seq = sample
    else:
        raise ValueError("Expected [id, sequence] or [id, seq_type, sequence]")
    if seq_type not in ("prot", "gene"):
        raise ValueError("seq_type must be prot or gene")
    # Same normalization as upstream clean_seq_luca; do not map protein A/C/G/T to genes.
    seq = "".join(char for char in seq.upper() if "A" <= char <= "Z")
    if not seq:
        raise ValueError("Sequence is empty after normalization")
    path = str(Path(llm_dirpath).expanduser().resolve())
    model, tokenizer = load_hf_model(path, fp16)
    if list(repr_layers) not in ([-1], [model.config.num_hidden_layers]):
        raise ValueError("HF adapter supports only the final normalized representation")
    prepend = int(tokenizer.prepend_bos)
    append = int(tokenizer.append_eos)
    limit = min(truncation_seq_length, model.config.max_position_embeddings - prepend - append)
    if limit <= 0:
        raise ValueError("Checkpoint has no room for sequence tokens")
    seq = seq[-limit:] if trunc_type == "left" else seq[:limit]
    inputs = tokenizer(seq, seq_type=seq_type, return_tensors="pt", add_special_tokens=True)
    if inputs["input_ids"].shape != (1, len(seq) + prepend + append):
        raise ValueError("Expected exactly one token per residue plus BOS/EOS")
    target = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    if fp16 and target.type == "cpu":
        raise ValueError("Use fp16=False for CPU embeddings")
    model.to(target)
    model.eval()
    inputs = {key: value.to(target) for key, value in inputs.items()}
    with torch.no_grad():
        hidden = model(**inputs, return_dict=True).last_hidden_state[0]
    result = {}
    if want_matrix:
        matrix = hidden if matrix_add_special_token else hidden[prepend:prepend + len(seq)]
        result["representations"] = matrix.float().cpu().numpy()
    if want_vector:
        if not prepend:
            raise ValueError("BOS embeddings require a tokenizer with prepend_bos=True")
        result["bos_representations"] = hidden[0].float().cpu().numpy()
    return (next(iter(result.values())) if len(result) == 1 else result), seq
