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
"""Adascale tests"""
import torch

from aimet_torch.experimental.adascale.adascale_quantizer import AdaScaleQuantizeDequantize
from aimet_torch.v2.quantization.affine import QuantizeDequantize

def test_adascale_compute_encodings():
    """
    Given:
    - Create QDQ module, store initial scale and create adascale equivalent with the QDQ module
    - Set Adascale params requires_grad to True
    When:
    - Train with random data
    - Save S2, S3
    Then:
    - S2, S3 Should not be zeros
    - Compare original scale with new scale
    """

    weight_shape = (1, 3, 224, 224)
    qdq_shape = (1, 3, 1, 1)
    torch.manual_seed(0)
    input_tensor = torch.rand(*weight_shape)

    torch.manual_seed(1)
    expected_tensor = torch.rand(*weight_shape)

    qdq = QuantizeDequantize(shape=qdq_shape, bitwidth=8, symmetric=True)

    with qdq.compute_encodings():
        _ = qdq(input_tensor)

    adascale_qdq = AdaScaleQuantizeDequantize(qdq, weight_shape)
    assert torch.equal(adascale_qdq.min, qdq.min)
    assert torch.equal(adascale_qdq.max, qdq.max)
    assert torch.equal(qdq(input_tensor), adascale_qdq(input_tensor))

    adascale_qdq.eval()
    adascale_params = adascale_qdq.get_adascale_trainable_parameters()
    for p in adascale_params:
        p.requires_grad = True


    prev_loss = None
    for epoch in range(5):
        optimizer = torch.optim.Adam(adascale_params)
        quant_out = adascale_qdq(input_tensor)
        loss = torch.nn.functional.mse_loss(expected_tensor, quant_out)
        assert prev_loss != loss
        prev_loss = loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    modified_q = adascale_qdq.get_qdq()
    adascale_out = adascale_qdq(input_tensor)
    input_with_s2_s3_folded = input_tensor / (torch.exp(adascale_qdq.s2) * torch.exp(adascale_qdq.s3))
    modified_out = modified_q(input_with_s2_s3_folded)

    assert torch.equal(adascale_qdq.get_max(), modified_q.get_max())
    assert torch.equal(adascale_qdq.get_min(), modified_q.get_min())
    assert torch.equal(adascale_qdq.get_scale(), modified_q.get_scale())
    assert torch.equal(adascale_qdq.get_offset(), modified_q.get_offset())

    assert torch.equal(modified_out, adascale_out)
