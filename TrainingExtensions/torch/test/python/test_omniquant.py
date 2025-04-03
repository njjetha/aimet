# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Test Omniquant functions. (Not include LET modules.) """
import contextlib
import copy
import os
from safetensors.numpy import save_file
import tempfile
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.lora.layer import Linear as LoraLinear
import pytest

from aimet_torch.omniquant import decoder_processor
from aimet_torch.omniquant import omniquant_optimizer

@contextlib.contextmanager
def add_custom_model_class_to_support_model_group(model_class, target_group_name):
    """
    Add model_class Class to decoder_processor.target_group_name.
    """
    target_group = getattr(decoder_processor, target_group_name)
    new_target_group = list(target_group)
    new_target_group.append(model_class)
    setattr(decoder_processor, target_group_name, tuple(new_target_group))

    yield

    setattr(decoder_processor, target_group_name, target_group)

class FakeLlamaModel(torch.nn.Module):
    def __init__(self, layer_num, seq_len, head_num, emb_dim):
        super().__init__()
        assert emb_dim % head_num == 0, "emb_dim need to be dividable by head_num."
        self.layers = torch.nn.ModuleList([FakeDecoderBlcok(seq_len, head_num, emb_dim) for _ in range(layer_num)])
        self.out_linear = torch.nn.Linear(emb_dim, 5)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_linear(x)
        return x

class FakeDecoderBlcok(torch.nn.Module):
    def __init__(self, seq_len, head_num, emb_dim):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(emb_dim)
        self.self_attn = FakeSelfAttn(seq_len, head_num, emb_dim)
        self.mlp = FakeMlp(emb_dim)
        self.post_attention_layernorm = torch.nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.mlp(x)
        x = self.post_attention_layernorm(x)
        return x

class FakeSelfAttn(torch.nn.Module):
    def __init__(self, seq_len, head_num, emb_dim):
        super().__init__()
        self.seq_len = seq_len
        self.head_num = head_num
        self.emb_dim = emb_dim
        self.q_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.k_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.v_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.o_proj = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        head_dim = self.emb_dim//self.head_num
        q = self.q_proj(x).reshape(-1, self.seq_len, self.head_num, head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(-1, self.seq_len, self.head_num, head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(x).reshape(-1, self.seq_len, self.head_num, head_dim).permute(0, 2, 1, 3)
        qk = torch.matmul(q, k)
        qkv = torch.matmul(qk, v).permute(0, 2, 1, 3).reshape(-1, self.seq_len, self.head_num*head_dim)
        out = self.o_proj(qkv)

        return out

class FakeMlp(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gate_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.up_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.down_proj = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        y0 = self.gate_proj(x)
        y1 = self.up_proj(x)
        y1 = self.down_proj(y1)
        return y0 + y1

class TestOmniquant:
    """ Test Omniquant. """
    def test_get_transformer_processor(self):
        """ Test get_transformer_processor returns correct TransformerProcessor and raise error. """
        layer_num = 5
        seq_len = 20
        head_num = 5
        emb_dim = 10
        dummy_input = torch.randn(1, seq_len, emb_dim)
        fake_llama_model = FakeLlamaModel(layer_num, seq_len, head_num, emb_dim)
        with torch.no_grad():
            # Make sure model is runnable.
            fake_llama_model(dummy_input)

        with add_custom_model_class_to_support_model_group(FakeLlamaModel, "LlamaModelGroup"):
            llama_processor = decoder_processor.get_transformer_processor(fake_llama_model)
            assert llama_processor.__name__ == "LlamaProcessor"

            decoder_list = llama_processor.get_decoder_list(fake_llama_model)
            assert len(decoder_list) == layer_num

            for _decoder_block in decoder_list:
                let_module_pair = llama_processor.get_let_module_pair(_decoder_block)
                assert len(let_module_pair) == 4 # Llama Model Group should have 4 let pairs.

        with pytest.raises(ValueError):
            decoder_processor.get_transformer_processor(fake_llama_model)

    def test_load_lora_model(self):
        """ Test omniquant_optimizer.update_lora_weights """
        layer_num = 2
        seq_len = 20
        head_num = 5
        emb_dim = 10
        dummy_input = torch.randn(1, seq_len, emb_dim)
        let_model = FakeLlamaModel(layer_num, seq_len, head_num, emb_dim)
        ori_model = copy.deepcopy(let_model)

        input_layernorm_scale = torch.nn.Parameter(torch.randn(emb_dim))
        mlp_scale = torch.nn.Parameter(torch.randn(emb_dim))
        input_layernorm_scale_2 = torch.nn.Parameter(torch.randn(emb_dim))
        mlp_scale_2 = torch.nn.Parameter(torch.randn(emb_dim))
        meta_data = {
            "layers.0.input_layernorm.prev": input_layernorm_scale,
            "layers.0.self_attn.q_proj.foll": input_layernorm_scale,
            "layers.0.self_attn.k_proj.foll": input_layernorm_scale,
            "layers.0.self_attn.v_proj.foll": input_layernorm_scale,
            "layers.0.mlp.up_proj.prev": mlp_scale,
            "layers.0.mlp.down_proj.foll": mlp_scale,
            "layers.1.input_layernorm.prev": input_layernorm_scale_2,
            "layers.1.self_attn.q_proj.foll": input_layernorm_scale_2,
            "layers.1.self_attn.k_proj.foll": input_layernorm_scale_2,
            "layers.1.self_attn.v_proj.foll": input_layernorm_scale_2,
            "layers.1.mlp.up_proj.prev": mlp_scale_2,
            "layers.1.mlp.down_proj.foll": mlp_scale_2,
        }

        # Apply meta data to let_model
        with torch.no_grad():
            for layer_name, scale in meta_data.items():
                module = let_model.get_submodule(layer_name[:-5])
                if layer_name.endswith("prev"):
                    if module.bias is not None:
                        module.bias.copy_(module.bias / scale)
                    new_weight = module.weight / (scale.reshape(-1, 1) if isinstance(module, torch.nn.Linear) else scale)
                    module.weight.copy_(new_weight)

                elif layer_name.endswith("foll"):
                    module.weight.copy_(module.weight* scale.reshape(1, -1))

        let_output = let_model(dummy_input)
        ori_output = ori_model(dummy_input)
        assert torch.allclose(let_output, ori_output, atol=1e-5)

        # Set model to lora model
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=2,
            bias="none",
            target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"]
        )
        peft_ori_model = get_peft_model(ori_model, lora_config)
        peft_let_model = get_peft_model(let_model, lora_config)
        peft_ori_model.eval() # .eval() to disable lora dropout.
        peft_let_model.eval() # .eval() to disable lora dropout.

        # Lora B default init to zero weight.
        # Init lora B weight to random weight.
        with torch.no_grad():
            for _, module in peft_ori_model.named_modules():
                if isinstance(module, LoraLinear):
                    for _, _lora_b in module.lora_B.items():
                        _lora_b.weight = torch.nn.Parameter(torch.randn(_lora_b.weight.shape))
        # Copy peft weight to let peft model.
        set_peft_model_state_dict(peft_let_model, get_peft_model_state_dict(peft_ori_model))

        # Run omniquant_optimizer.update_lora_weights
        with tempfile.TemporaryDirectory() as tempdir:
            metadata_path = os.path.join(tempdir, "./metadata.safetensor")
            save_file({k: v.data.numpy() for k, v in meta_data.items()}, metadata_path)
            omniquant_optimizer.update_lora_weights(peft_let_model, metadata_path)

        peft_let_output = peft_let_model(dummy_input)
        peft_ori_output = peft_ori_model(dummy_input)

        # peft model output should be different from base model output.
        assert not torch.allclose(let_output, peft_let_output, atol=1e-5)
        assert torch.allclose(peft_let_output, peft_ori_output, atol=1e-5)
