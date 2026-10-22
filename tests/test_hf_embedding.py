"""Contract tests; real-checkpoint tests are opt-in via LUCAVIRUS_HF_DIR."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.llm.lucavirus import hf_embedding as hf


class FakeTokenizer:
    prepend_bos = True
    append_eos = True

    def __call__(self, seq, seq_type, **kwargs):
        assert seq_type == "prot"
        ids = torch.tensor([[2] + [ord(c) for c in seq] + [3]])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), token_type_ids=torch.ones_like(ids))


class FakeModel(torch.nn.Module):
    config = SimpleNamespace(num_hidden_layers=12, max_position_embeddings=6)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        assert not torch.is_grad_enabled()
        assert torch.all(attention_mask == 1) and torch.all(token_type_ids == 1)
        return SimpleNamespace(last_hidden_state=input_ids[..., None].float().repeat(1, 1, 3))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch.object(hf, "load_hf_model", return_value=(FakeModel(), FakeTokenizer()))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_matrix_truncation_normalization_and_specials(self):
        matrix, seq = hf.predict_embedding_hf("unused", ["id", "a-cdEF*"], "right", "matrix", device="cpu")
        self.assertEqual(seq, "ACDE")
        np.testing.assert_array_equal(matrix[:, 0], [65, 67, 68, 69])
        matrix, seq = hf.predict_embedding_hf("unused", ["id", "prot", "ACDEF"], "left", "matrix", device="cpu", matrix_add_special_token=True)
        self.assertEqual(seq, "CDEF")
        np.testing.assert_array_equal(matrix[:, 0], [2, 67, 68, 69, 70, 3])
        self.assertEqual(matrix.dtype, np.float32)

    def test_bos_and_combined_outputs(self):
        result, _ = hf.predict_embedding_hf("unused", ["id", "AC"], "right", "matrix_vector", device="cpu")
        self.assertEqual(result["representations"].shape, (2, 3))
        np.testing.assert_array_equal(result["bos_representations"], [2, 2, 2])

    def test_invalid_requests_fail_explicitly(self):
        for overrides in [dict(embedding_type="contacts"), dict(repr_layers=[1]),
                          dict(truncation_seq_length=0), dict(trunc_type="middle"),
                          dict(sample=["id", "---"]), dict(fp16=True)]:
            args = dict(llm_dirpath="unused", sample=["id", "AC"], trunc_type="right", embedding_type="matrix", device="cpu")
            args.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                hf.predict_embedding_hf(**args)


@unittest.skipUnless(os.environ.get("LUCAVIRUS_HF_DIR"), "Set LUCAVIRUS_HF_DIR to test real weights")
class CheckpointTests(unittest.TestCase):
    def test_checkpoint_and_stock_head(self):
        from src.llm.lucavirus.get_embedding import predict_embedding
        from src.llm.lucavirus.models.alphabet import Alphabet
        from src.common.luca_base import LucaBase
        from src.common.model_config import LucaConfig

        path = os.path.realpath(os.environ["LUCAVIRUS_HF_DIR"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq = "ACDEFGHIKLMNPQRSTVWYXBZUOJ"
        # Synthetic token fixture, not biological training data.
        matrix, processed = predict_embedding(path, ["fixture", "prot", seq], "right", "matrix", device=device, matrix_add_special_token=True)
        model, tokenizer = hf.load_hf_model(path, False)
        tokens = tokenizer(seq, seq_type="prot", return_tensors="pt")
        alphabet = Alphabet.from_predefined("gene_prot")
        expected_ids = [alphabet.cls_idx] + alphabet.encode(seq) + [alphabet.eos_idx]
        self.assertEqual(tokens["input_ids"][0].tolist(), expected_ids)
        self.assertTrue(torch.all(tokens["token_type_ids"] == 1))
        with torch.no_grad():
            direct = model(**{k: v.to(device) for k, v in tokens.items()}, return_dict=True).last_hidden_state[0].float().cpu().numpy()
        np.testing.assert_allclose(matrix, direct, rtol=1e-5, atol=1e-5)
        self.assertEqual(processed, seq)
        self.assertEqual(matrix.shape, (len(seq) + 2, 2560))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

        config = LucaConfig.from_json_file("config/luca_base/luca_base_config.json")
        config.embedding_input_size = 2560
        config.matrix_fc_size = [512]
        config.classifier_size = 128
        args = SimpleNamespace(input_type="matrix", output_mode="binary_class", fusion_type="concat",
            task_level_type="seq_level", prepend_bos=True, append_eos=True, loss_reduction="meanmean",
            matrix_encoder=False, matrix_pooling_type="value_attention", sigmoid=True, loss_type="bce",
            ignore_index=-100, classifier_activate_func="gelu", pos_weight=20.0, device=torch.device(device))
        head = LucaBase(config, args).to(device).eval()
        values = torch.tensor(matrix[None], device=device)
        mask = torch.ones(values.shape[:2], device=device, dtype=torch.long)
        loss, logits, probability = head(matrices=values, matrix_attention_masks=mask, labels=torch.ones((1, 1), device=device))
        self.assertEqual(tuple(logits.shape), (1, 1))
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertIsNotNone(head.classifier.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        padded = torch.cat([values, torch.ones((1, 3, 2560), device=device) * 100], dim=1)
        padded_mask = torch.cat([mask, torch.zeros((1, 3), device=device, dtype=torch.long)], dim=1)
        with torch.no_grad():
            padded_logits = head(matrices=padded, matrix_attention_masks=padded_mask)[0]
        torch.testing.assert_close(logits, padded_logits, atol=1e-5, rtol=1e-5)
        print("PASS: HF/legacy token IDs, direct HF parity, matrix shape, mask, stock head backward, frozen backbone", flush=True)
        del model, head, values, loss, logits, probability, padded, padded_logits
        hf.load_hf_model.cache_clear()
        if device == "cuda":
            torch.cuda.empty_cache()
            half, _ = predict_embedding(path, ["fixture", "prot", seq], "right", "matrix", device=device, fp16=True)
            self.assertEqual(half.shape, (len(seq), 2560))
            self.assertTrue(np.isfinite(half).all())
            self.assertEqual(half.dtype, np.float32)
            print("PASS: fp16 embedding forward", flush=True)


if __name__ == "__main__":
    unittest.main()
