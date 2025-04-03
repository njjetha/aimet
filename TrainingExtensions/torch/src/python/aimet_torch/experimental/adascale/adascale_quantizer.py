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
# pylint: disable=redefined-builtin
""" Adascale quantizer """

from typing import Optional

import torch

from aimet_torch.v2.quantization import DequantizedTensor
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.affine.backends import torch_builtins

use_adascale_lwc: bool = True

class AdaScaleQuantizeDequantize(QuantizeDequantize):
    """Specialized class for AdaScale QDQ"""
    beta: torch.nn.Parameter
    gamma: torch.nn.Parameter
    s2: torch.nn.Parameter
    s3: torch.nn.Parameter

    def __init__(self, qdq: QuantizeDequantize, weight_shape: tuple):
        """
        Creates the AdaScale QDQ object. This quantizer should be substituted in place of Linear weight qdq object

        :param qdq: QuantizeDequantize object using which the Adascale object needs to be created
        :param weight_shape: Shape of the weight tensor
        """

        assert use_adascale_lwc, "Flexround QDQ is not yet implemented."
        assert qdq.symmetric is True, "Only symmetric quantization is supported"
        super().__init__(qdq.shape, qdq.bitwidth, qdq.symmetric, qdq.encoding_analyzer, qdq.block_size)

        self.register_parameter('beta', torch.nn.Parameter(torch.zeros(self.shape)))
        self.register_parameter('gamma', torch.nn.Parameter(torch.zeros(self.shape)))

        if qdq.block_size is not None:
            self.register_parameter('s2', torch.nn.Parameter(torch_builtins.reshape_tensor_for_blocks(
                torch.zeros(weight_shape), qdq.shape, self.block_size).squeeze(1)))
            self.register_parameter('s3', torch.zeros(self.shape).unsqueeze(-1))
        else:
            self.register_parameter('s2', torch.nn.Parameter(torch.zeros(weight_shape)))
            self.register_parameter('s3', torch.nn.Parameter(torch.zeros(self.shape)))

        self.set_range(qdq.min, qdq.max)

    def get_adascale_trainable_parameters(self):
        """Helper to query all the trainable parameters of AdaScale QDQ"""
        return [self.beta,
                self.gamma,
                self.s2,
                self.s3]

    def get_qdq(self) -> QuantizeDequantize:
        """
        Return the Quantized QDQ object for the sim object to be restored to original condition.
        S2, S3 are not used to create QDQ object. This needs to be folded into the weights before converting to QDQ.
        """
        q = QuantizeDequantize(self.shape, self.bitwidth, self.symmetric, self.encoding_analyzer, self.block_size)
        q.set_range(self.get_min(), self.get_max())
        return q

    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        """
        Performs QDQ on the input tensor based on the learnt parameters gamma, beta, s2, s3 and by using the
        parameters min and max tensors

        :param input: Input tensor to be QDQ
        :return: Dequantized tensor after applying AdaScale QDQ
        """
        # scale the input with s2 and s3
        input = input / (torch.exp(self.s2) * torch.exp(self.s3))
        return super().forward(input)

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        scale = (torch.exp(self.gamma) * self.max.to(dtype) - torch.exp(self.beta) * self.min.to(dtype)) / self._get_num_steps()
        return scale

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        dtype = dtype or torch.float32
        return torch.zeros_like(self.min, requires_grad=False, dtype=dtype)
