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
""" AdaScale implementation """

import contextlib
from typing import Callable, List

import torch
from torch.utils.data import DataLoader

from aimet_common.utils import AimetLogger
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn import QuantizedLinear, BaseQuantizationMixin, compute_param_encodings
from aimet_torch.v2.utils import default_forward_fn, remove_all_quantizers
from aimet_torch.experimental.adascale.adascale_quantizer import AdaScaleQuantizeDequantize
from aimet_torch.blockwise_sampler import BlockwiseSampler

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)

loss_fn = torch.nn.MSELoss()

# mapping of model and the corresponding adascale blocks type
model_to_block_mapping = {}

supported_modules: List = [QuantizedLinear]


class AdaScale:
    """
    AdaScale is PTQ technique which performs Knowledge Distillation on blocks of modules by using the FP32 output as its
    reference output. Adascale is based on FlexRound: https://arxiv.org/abs/2306.00317 but integrates LWC from Omniquant.

    The optimization is performed on a block-by-block basis by comparing the quantized output of the block with its FP32
    equivalent and by training the parameters (gamma, beta, s2, s3) which are temporarily introduced in every supported
    module.

    A block is defined as a non-leaf module which takes in one activation input tensor and outputs one activation tensor
    Currently only Linear layers are supported, and all the linears in a block are optimized at the same time.
    """

    @classmethod
    def apply_adascale(cls, qsim: QuantizationSimModel,
                       data_loader: DataLoader,
                       forward_fn: Callable = None,
                       num_batches: int = 1,
                       num_epochs: int = 1):
        """
        :param qsim: Quantization Sim model
        :param data_loader: DataLoader object to load the input data
        :param forward_fn: forward function to run the forward pass of the model
        :param num_batches: Number of batches
        :param num_epochs: Number of epochs to perform the adascale training

        """
        # pylint: disable=unused-variable, too-many-locals
        if not forward_fn:
            forward_fn = default_forward_fn

        compute_param_encodings(qsim.model)

        adascale_blocks = cls._get_blocks(qsim)

        # replace with adascale weight quantizer which introduces trainable params - beta, gamma, s2, s3
        cls._replace_with_adascale_weight_quantizers(adascale_blocks)

        sampler = BlockwiseSampler(qsim, adascale_blocks, data_loader, num_batches)
        qsim.model.requires_grad_(False)
        for block, fp_block_inputs, qt_block_inputs in sampler.sample():
            assert num_batches == len(qt_block_inputs)
            assert num_batches == len(fp_block_inputs)

            # only set adascale params to train mode
            trainable_params = cls._get_adascale_trainable_params(block)
            optimizer = torch.optim.Adam(trainable_params)
            cls._set_requires_grad(trainable_params, True)
            fp_out = []
            for batch_idx in range(num_batches):
                with remove_all_quantizers(block):
                    fp_out.append(block(fp_block_inputs[batch_idx][0]))

            for epoch in range(num_epochs):
                for batch_idx in range(num_batches):
                    with cls._remove_act_quantizers_in_block(block) and torch.set_grad_enabled(True):
                        quant_out = block(qt_block_inputs[batch_idx][0])
                        loss = loss_fn(quant_out, fp_out[batch_idx])
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

            cls._set_requires_grad(trainable_params, False)
        cls._fold_weights_and_replace_with_qdq(adascale_blocks)

    @staticmethod
    def _get_blocks(qsim: QuantizationSimModel):
        """ helper to get all the blocks in the model represented by model_to_block_mapping """
        target_type = model_to_block_mapping.get(type(qsim.model))
        target_modules = []
        if target_type is not None:
            target_modules = [m for m in qsim.model.modules() if type(m) == target_type] #pylint: disable=unidiomatic-typecheck
        return target_modules

    @staticmethod
    def _replace_with_adascale_weight_quantizers(adascale_blocks: List):
        """Replace all the weight quantizers in supported modules with adascale quantizers"""
        for block in adascale_blocks:
            for layer in block.modules():
                if isinstance(layer, tuple(supported_modules)):
                    layer.param_quantizers['weight'] = AdaScaleQuantizeDequantize(layer.param_quantizers['weight'],
                                                                                  layer.weight.shape)

    @classmethod
    def _fold_weights_and_replace_with_qdq(cls, adascale_blocks: List):
        """Replace adascale weight quantizers back with QDQ, and fold the params s2, s3 into weight"""
        for block in adascale_blocks:
            for layer in block.modules():
                if isinstance(layer, tuple(supported_modules)):
                    layer.weight.copy_(layer.weight / (torch.exp(layer.param_quantizers['weight'].s2) *
                                    torch.exp(layer.param_quantizers['weight'].s3)))
                    layer.param_quantizers['weight'] = layer.param_quantizers['weight'].get_qdq()

    @staticmethod
    def _remove_act_quantizers_in_block(non_leaf_module: torch.nn.Module):
        """ Remove all quantizers in the given block """
        exit_stack = contextlib.ExitStack()
        for module in non_leaf_module.modules():
            if isinstance(module, BaseQuantizationMixin):
                exit_stack.enter_context(module._remove_activation_quantizers())  # pylint: disable=protected-access
        return exit_stack

    @staticmethod
    def _get_adascale_trainable_params(non_leaf_module: torch.nn.Module) -> List:
        """ Get all the adascale scale params present in the non-leaf module """
        trainable_params = []
        for module in non_leaf_module.modules():
            if isinstance(module, tuple(supported_modules)) and type(
                    module.param_quantizers['weight'] == AdaScaleQuantizeDequantize):
                trainable_params.extend(module.param_quantizers['weight'].get_adascale_trainable_parameters())
        return trainable_params

    @staticmethod
    def _set_requires_grad(adascale_params: list, val: bool):
        """ Helper to update requires_grad to the input `val` for all the params in `adascale_params` """
        for p in adascale_params:
            p.requires_grad = val

apply_adascale = AdaScale.apply_adascale
