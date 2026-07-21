# Copyright 2026 Upstage AI and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SolarOpen2 model."""

import unittest

import torch
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import SolarOpen2Config, SolarOpen2ForCausalLM, SolarOpen2Model


class SolarOpen2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = SolarOpen2Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=0,
        use_rope=False,
        use_qk_norm=False,
        use_gqa_gate=True,
        kda_allow_neg_eigval=True,
        hidden_act="silu",
    ):
        super().__init__(parent=parent, num_experts_per_tok=num_experts_per_tok)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_gqa_gate = use_gqa_gate
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.hidden_act = hidden_act
        # Interleave a full-attention (GQA) layer (0) and a Kimi-Delta linear-attention layer (1).
        self.gqa_layers = [0]
        self.linear_attn_config = {
            "short_conv_kernel_size": 4,
            "head_dim": self.head_dim,
            "num_heads": self.num_attention_heads,
            "num_kv_heads": None,
        }


@require_torch
class SolarOpen2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = SolarOpen2ModelTester
    model_split_percents = [0.5, 0.85, 0.9]  # it tries to offload everything with the default value

    @unittest.skip("SolarOpen2 hybrid linear-attention cache is not compatible with quantized cache yet.")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip("SolarOpen2 hybrid linear-attention cache is recurrent and not compatible with a static cache.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("SolarOpen2 hybrid linear-attention cache is recurrent and not compatible with a static cache.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("The recurrent linear-attention cache cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        self.skipTest(
            "SolarOpen2 defaults to NoPE (use_rope=False); RoPE scaling has no effect on the shipped config."
        )

    def _get_conv_state_shape(self, batch_size: int, config):
        linear_attn_config = config.linear_attn_config
        num_heads = linear_attn_config["num_heads"]
        head_dim = linear_attn_config["head_dim"]
        num_kv_heads = linear_attn_config.get("num_kv_heads") or num_heads
        conv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        return (batch_size, conv_dim, linear_attn_config["short_conv_kernel_size"])

    def _get_recurrent_state_shape(self, batch_size: int, config):
        linear_attn_config = config.linear_attn_config
        return (
            batch_size,
            linear_attn_config["num_heads"],
            linear_attn_config["head_dim"],
            linear_attn_config["head_dim"],
        )

    def test_attention_outputs(self):
        """Overwritten: SolarOpen2 alternates full-attention (GQA) and Kimi-Delta linear-attention layers,
        so only the full-attention layers emit attention weights."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            n_full = sum(layer == "full_attention" for layer in config.layer_types)
            self.assertEqual(len(attentions), n_full)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    def test_rope_parameters_partially_initialized(self):
        """SolarOpen2Config overrides the parent's default partial_rotary_factor to 1.0."""
        config = SolarOpen2Config(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 65536,
            }
        )
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)

    def test_kda_cached_decode_matches_full_forward(self):
        """The hybrid (KDA conv+recurrent state and GQA KV) cache must match a full forward."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs["input_ids"][:1, :6]
        model = SolarOpen2ForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            full = model(input_ids, use_cache=False).logits
            out = model(input_ids[:, :4], use_cache=True)
            cache = out.past_key_values
            incremental = [out.logits[:, -1]]
            for t in range(4, 6):
                out = model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                incremental.append(out.logits[:, -1])
        incremental = torch.stack(incremental, dim=1)
        torch.testing.assert_close(incremental, full[:, 3:6], rtol=1e-4, atol=1e-4)

    def test_layer_types_derivation(self):
        """gqa_layers (or gqa_interval) determines the per-layer attention pattern."""
        config = SolarOpen2Config(num_hidden_layers=6, gqa_layers=[0, 4])
        self.assertEqual(
            config.layer_types,
            [
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        )
        config = SolarOpen2Config(num_hidden_layers=6, gqa_layers=None, gqa_interval=3)
        self.assertEqual(config.layer_types.count("full_attention"), 2)

    def test_public_250b_config_defaults(self):
        config = SolarOpen2Config()
        self.assertEqual(config.max_position_embeddings, 1_048_576)
        self.assertEqual(config.n_routed_experts, 320)
        self.assertTrue(config.kda_allow_neg_eigval)
        self.assertFalse(any(key.startswith("hc_") for key in config.to_dict()))

    def test_hyper_connections_are_rejected(self):
        config = SolarOpen2Config(hc_rate=1)
        with self.assertRaisesRegex(NotImplementedError, "Hyper-Connections"):
            SolarOpen2Model(config)
