# Local Hugging Face LucaVirus checkpoints

This fork adds an HF checkpoint adapter to the upstream embedding-based binary
classification workflow. The LucaBase head, pooling, loss and trainer are unchanged.
The LucaVirus backbone remains frozen: this is head training, not LoRA or full
backbone fine-tuning.

## Install

Use Python 3.12 and install PyTorch appropriate for your CUDA runtime first. The
adapter was tested with PyTorch 2.6.0 / CUDA 12.4. Then, from the repository root:

```bash
python -m pip install -r requirements-hf.txt
```

These requirements cover the HF adapter and original protein binary-head imports;
they are not a replacement environment for every task in this repository.

## Use with the original binary recipe

Start with `src/training/lucavirus/DeepVirID/run_DeepVirID_prot_luca_virus_linear.sh`.
In your copy of its `python run.py` command, add:

```bash
--llm_hf_dir /absolute/path/to/LucaVirus-prot-step3.8M
```

Keep `--llm_type lucavirus`. The new option overrides the legacy checkpoint path
built from `--llm_dir`, time and step. Point to the complete local HF directory
(config, tokenizer files, custom Python files and weights). It loads locally only;
it executes that checkpoint's custom Python, so use a trusted checkpoint.
The resolved directory is stored in training arguments for later prediction.

The upstream launcher is an example: adjust its GPU, data/output paths and malformed
`--test_data_dir` argument before execution. This fork does not select data splits
or change its hyperparameters. Cluster-related records must remain in the same
split when preparing clustered datasets.

## What changed

- `src/llm/lucavirus/hf_embedding.py`: local AutoModel/AutoTokenizer loading, frozen
  final-layer matrices and BOS vectors in the existing NumPy return format.
- `src/llm/lucavirus/get_embedding.py`: dispatches HF directories to that adapter;
  original-format checkpoints retain their existing loader.
- `src/run.py`: adds `--llm_hf_dir`.

Protein tokenization explicitly uses `seq_type="prot"`; the HF tokenizer otherwise
defaults to genes. The adapter preserves the upstream A–Z sequence normalization,
left/right truncation, and optional BOS/EOS matrix rows. Matrices are float32 at
the head boundary. The fp16 option adds a model-local rotary output cast because
the released HF implementation otherwise mixes float32 q/k with float16 v.

Only the final normalized representation is supported. Contact predictions and
intermediate-layer requests raise errors. Per-call length is capped by the HF
configuration including special tokens. The existing encoder controls chunked
completion for longer sequences. Errors propagate rather than silently returning
incorrect embeddings. CPU inference requires fp16=False.

## Checks

From the repository root, small contract tests require no checkpoint:

```bash
python -m unittest discover -s tests -v
```

To also check real checkpoint loading, legacy/HF protein token IDs, parity with
direct HF output, padding masks, the original head's backward pass, frozen backbone
and a CUDA fp16 forward:

```bash
LUCAVIRUS_HF_DIR=/absolute/path/to/LucaVirus-prot-step3.8M python -m unittest discover -s tests -v
```

The tests use a short synthetic alphabet fixture, not training data. They do not
establish numerical equivalence with an original-format backbone checkpoint,
validate a full training run, or establish biological classification accuracy.
