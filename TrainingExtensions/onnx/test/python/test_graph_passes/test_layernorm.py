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
from ..models.test_models import layernorm_model
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.graph_passes.pass_registry import apply_graph_passes
from .utils import assert_on_const_quantizers, assert_on_output_quantizers, get_dummy_qc_quantize_op_dict
import pytest

@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_layer_norm(elementwise_affine, bias):
    model = layernorm_model(elementwise_affine=elementwise_affine, bias=bias)
    graph = ConnectedGraph(model)
    qc_quantize_op_dict = get_dummy_qc_quantize_op_dict(graph)

    quantization_status = [q_op.enabled for q_op in list(qc_quantize_op_dict.values())]
    # Check if quantization is enabled for all ops
    assert all(quantization_status)
    apply_graph_passes(graph, qc_quantize_op_dict, ["LayerNormalization"])

    all_ops = graph.ordered_ops
    # Check if quantization is disabled for LayerNormalization intermediate op outputs
    assert_on_output_quantizers(all_ops[:-1], qc_quantize_op_dict)
    # Check if quantization is enabled for last op of LayerNormalization sub-graph
    assert_on_output_quantizers(all_ops[-1:], qc_quantize_op_dict, enabled=True)

    # Check if quantization is disabled for LayerNormalization sub-graph constant ops except layernorm.weight
    if elementwise_affine:
        layernorm_weight = all_ops[-2 if bias else -1]
        all_ops.remove(layernorm_weight)
        assert_on_const_quantizers([layernorm_weight], qc_quantize_op_dict, enabled=True)

    assert_on_const_quantizers(all_ops, qc_quantize_op_dict)

def test_layer_norm_intermediate():
    model = layernorm_model(include_add_ops=True)
    graph = ConnectedGraph(model)
    qc_quantize_op_dict = get_dummy_qc_quantize_op_dict(graph)

    quantization_status = [q_op.enabled for q_op in list(qc_quantize_op_dict.values())]
    # Check if quantization is enabled for all ops
    assert all(quantization_status)
    apply_graph_passes(graph, qc_quantize_op_dict, ["LayerNormalization"])

    all_ops = graph.ordered_ops
    # Check if quantization is disabled for LayerNormalization intermediate op outputs
    assert_on_output_quantizers(all_ops[1:-2], qc_quantize_op_dict)
    # Check if quantization is enabled for last op of LayerNormalization sub-graph
    assert_on_output_quantizers(all_ops[-2:-1], qc_quantize_op_dict, enabled=True)
    # Check if quantization is disabled for LayerNormalization sub-graph constant ops except layernorm.weight
    layernorm_weight = all_ops[-3]
    all_ops.remove(layernorm_weight)
    assert_on_const_quantizers(all_ops, qc_quantize_op_dict)
    assert_on_const_quantizers([layernorm_weight], qc_quantize_op_dict, enabled=True)
