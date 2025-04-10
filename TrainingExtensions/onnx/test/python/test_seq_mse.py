# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import json
import numpy as np
import os
from onnx.utils import Extractor
import logging
from aimet_common.utils import AimetLogger
from aimet_onnx.sequential_mse.dependency_graph import DependencyGraph, SUPPORTED_MODULES
from aimet_onnx.sequential_mse.seq_mse import SeqMseParams
from aimet_onnx.sequential_mse.seq_mse import SequentialMse
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

from .models.test_models import single_linear_layer_model
from .models.test_models import single_conv_layer_model
from .models.test_models import model_with_split
from .models.test_models import single_residual_model
from .models import models_for_tests
from .models.test_models_onnx import model_with_multiple_inputs
from .models.test_models_onnx import model_with_multiple_outputs

np.random.seed(0)
torch.manual_seed(42)
AimetLogger.set_level_for_all_areas(logging.DEBUG)

def unlabeled_data_loader(dummy_inputs):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            if isinstance(dummy_inputs, list):
                return [value[index] for value in self.data]
            else:
                return self.data[index]

        def __len__(self):
            if isinstance(dummy_inputs, list):
                return len(self.data[0])
            else:
                return len(self.data)

    if isinstance(dummy_inputs, list):
        dataset = MyDataset([[value[0, :] for _ in range(10)] for value in dummy_inputs])
    else:
        dataset = MyDataset([dummy_inputs[0, :] for _ in range(10)])

    return DataLoader(dataset)

def dummy_input_for_linear_layer():
    return torch.randn((1, 100, 100))


def dummy_input_for_conv_layer():
    return torch.randn((1, 5, 5, 5))


def dummy_input_for_dependency_graph():
    return torch.randn((1, 1, 10, 10))


def dummy_input_for_residual_model():
    return torch.randn((1, 3, 32, 32))


def dummy_input_for_model_with_multiple_input():
    return [torch.randn((1, 3, 32, 32)), torch.randn((1, 3, 32, 32))]


def get_single_linear_layer_model():
    return single_linear_layer_model()


def get_single_conv_layer_model():
    return single_conv_layer_model()


def get_model_with_split():
    return model_with_split()


def get_model_with_multiple_inputs():
    return model_with_multiple_inputs()


def get_model_with_multiple_outputs():
    return model_with_multiple_outputs()


def _get_config_file(is_symmetric: bool, strict_symmetric: bool, unsigned_symmetric:bool, pcq: bool) -> str:
    """ Temporary fix until the config file can be read from beq_config directory"""

    def get_bool_str(in_bool: bool) -> str:
        if in_bool:
            return "True"
        else:
            return "False"

    beq_per_channel_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric)
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric)
            },
            "strict_symmetric": get_bool_str(strict_symmetric),
            "unsigned_symmetric": get_bool_str(unsigned_symmetric),
            "per_channel_quantization": get_bool_str(pcq)
        },
        "params": {
            "bias": {
                "is_quantized": "True"
            }
        },
        "op_type": {
            "PRelu": {
                "params": {
                    "weight": {
                        "is_quantized": "False"
                    }
                }
            }
        },
        "supergroups": [
            {
                "op_list": [
                    "Gemm",
                    "PRelu"
                ]
            },
            {
                "op_list": [
                    "Gemm",
                    "Sigmoid"
                ]
            },
            {
                "op_list": [
                    "Conv",
                    "PRelu"
                ]
            },
            {
                "op_list": [
                    "Conv",
                    "Sigmoid"
                ]
            }
        ],
        "model_input": {
            "is_input_quantized": "True"
        },
        "model_output": {}
    }

    if not os.path.exists("data"):
        os.mkdir("data")
    file_name = './data/beq_per_channel_config.json'
    with open(file_name, 'w') as f:
        json.dump(beq_per_channel_config, f)

    return file_name


def _create_input_dict(input_names, cached_data):
    input_dict = {}

    if not input_names:
        return cached_data
    for input_name in input_names:
        input_dict[input_name] = cached_data[input_name]
    return input_dict


def _build_session(model):
    """
    Build and return onnxruntime inference session
    :param providers: providers to execute onnxruntime
    """
    from onnxruntime import SessionOptions, InferenceSession,GraphOptimizationLevel
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(
        path_or_bytes=model.SerializeToString(),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    return session

def data_loader(dummy_input):
    return [dummy_input for _ in range(10)]

@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_do_seq_mse_for_conv(param_bw, loss_fn, enable_pcq):
    model = single_conv_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=enable_pcq))
    seq_params = SeqMseParams(num_batches=1)
    seq_params.loss_fn = loss_fn
    dataloader = unlabeled_data_loader(dummy_input_for_conv_layer())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    conv_node = seq_mse.dependency_graph._name_to_node['/conv/Conv']
    seq_mse._run_seq_mse([conv_node])
    _, per_channel_max = seq_mse._get_min_max_from_weights(conv_node)
    if not enable_pcq:
        per_channel_max = max(per_channel_max)

    weight_name = seq_mse.dependency_graph.get_param_name(conv_node)
    quantize_op = seq_mse.sim.qc_quantize_op_dict[weight_name]
    encodings = quantize_op.get_encodings()
    encodings_max = [encoding.max for encoding in encodings]
    if param_bw == 31:
        assert np.all(np.isclose(encodings_max, per_channel_max))
    else:
        assert not np.all(np.isclose(encodings_max, per_channel_max))


@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_do_seq_mse_for_linear(param_bw, loss_fn, enable_pcq):
    model = get_single_linear_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    seq_params.loss_fn = loss_fn
    dataloader = unlabeled_data_loader(dummy_input_for_linear_layer())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    fc_node = seq_mse.dependency_graph._name_to_node['/fc/MatMul']
    seq_mse._run_seq_mse([fc_node])
    _, per_channel_max = seq_mse._get_min_max_from_weights(fc_node)
    weight_name = seq_mse.dependency_graph.get_param_name(fc_node)
    quantize_op = seq_mse.sim.qc_quantize_op_dict[weight_name]
    encodings = quantize_op.encodings
    encodings_max = [encoding.max for encoding in encodings]
    if param_bw == 31:
        assert np.all(np.isclose(encodings_max, per_channel_max))
    else:
        assert not np.all(np.isclose(encodings_max, per_channel_max))

@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_conv(param_bw, loss_fn, enable_pcq):
    model = get_single_conv_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    seq_params.loss_fn = loss_fn
    dataloader = unlabeled_data_loader(dummy_input_for_conv_layer())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.apply_seq_mse_algo()
    weight_quantizer = seq_mse.sim.qc_quantize_op_dict['conv.weight']
    assert weight_quantizer._is_encoding_frozen

@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_static_apply_seq_mse(param_bw, loss_fn, enable_pcq):
    model = get_single_conv_layer_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    seq_params.loss_fn = loss_fn
    dataloader = unlabeled_data_loader(dummy_input_for_conv_layer())
    SequentialMse.apply_seq_mse(model, sim, seq_params, dataloader)

@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_split(param_bw, loss_fn, enable_pcq):
    model = get_model_with_split()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    seq_params.loss_fn = loss_fn
    dataloader = unlabeled_data_loader(dummy_input_for_dependency_graph())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.apply_seq_mse_algo()

    weight_quantizer_conv_1 = seq_mse.sim.qc_quantize_op_dict['conv1.weight']
    weight_quantizer_conv_2 = seq_mse.sim.qc_quantize_op_dict['conv2.weight']
    weight_quantizer_conv_3 = seq_mse.sim.qc_quantize_op_dict['conv3.weight']

    assert weight_quantizer_conv_1.is_encoding_frozen()
    assert weight_quantizer_conv_2.is_encoding_frozen()
    assert weight_quantizer_conv_3.is_encoding_frozen()


def test_dependency_graph():
    model = get_model_with_split()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    dataloader = unlabeled_data_loader(dummy_input_for_dependency_graph())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].in_degree == 0
    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].out_degree == 2
    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].op_input_names == ['input']
    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].op_output_names == ['/conv1/Conv_output_0']

    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].out_degree == 0
    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].op_input_names == ['/conv1/Conv_output_0']
    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].op_output_names == ['/conv2/Conv_output_0']

    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].out_degree == 0
    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].op_input_names == ['/conv1/Conv_output_0']
    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].op_output_names == ['/conv3/Conv_output_0']


def test_residual_model_dependency_graph():
    model = single_residual_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    dataloader = unlabeled_data_loader(dummy_input_for_residual_model())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].in_degree == 0
    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].out_degree == 2
    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].op_input_names == ['input']
    assert seq_mse.dependency_graph._name_to_node['/conv1/Conv'].op_output_names == ['/conv1/Conv_output_0']

    assert seq_mse.dependency_graph._name_to_node['/conv4/Conv'].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv4/Conv'].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv4/Conv'].op_input_names == ['/maxpool/MaxPool_output_0']
    assert seq_mse.dependency_graph._name_to_node['/conv4/Conv'].op_output_names == ['/conv4/Conv_output_0']

    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].op_input_names == ['/maxpool/MaxPool_output_0']
    assert seq_mse.dependency_graph._name_to_node['/conv2/Conv'].op_output_names == ['/conv2/Conv_output_0']

    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].op_input_names == ['/relu2/Relu_output_0']
    assert seq_mse.dependency_graph._name_to_node['/conv3/Conv'].op_output_names == ['/conv3/Conv_output_0']

@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ['mse', 'l1', 'sqnr'])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_residual_model(param_bw, loss_fn, enable_pcq):
    model = single_residual_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=param_bw,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=enable_pcq))
    seq_params = SeqMseParams(num_batches=2)
    seq_params.loss_fn = loss_fn
    dataloader = unlabeled_data_loader(dummy_input_for_residual_model())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.apply_seq_mse_algo()

    for conn_graph_op in seq_mse.dependency_graph.conn_graph.ordered_ops:
        if conn_graph_op.type in SUPPORTED_MODULES:
            param_names = [param_name for param_name, (_, param_type) in conn_graph_op.parameters.items() if param_type == "weight"]
            assert len(param_names) == 1
            quantizer = seq_mse.sim.qc_quantize_op_dict[param_names[0]]
            assert quantizer.is_encoding_frozen()

def test_model_with_multiple_inputs_dependency_graph_utils():

    model = get_model_with_multiple_inputs()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    dataloader = unlabeled_data_loader(dummy_input_for_model_with_multiple_input())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    starting_ops_names = [op.name_op for op in seq_mse.dependency_graph.conn_graph.starting_ops]

    assert starting_ops_names == ["Conv1"]
    assert seq_mse.dependency_graph._op_names_with_model_inputs == ["Conv1", "ADD_0", "ADD_1"]

def test_model_with_multiple_outputs_value_info():

    model = get_model_with_multiple_outputs()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    dataloader = unlabeled_data_loader(dummy_input_for_model_with_multiple_input())
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    assert 'Conv1_Y' in seq_mse._sim_extractor.vimap

def test_concat_model():

    model = models_for_tests.concat_model()
    sim = QuantizationSimModel(model=copy.deepcopy(model),
                               quant_scheme=QuantScheme.post_training_tf,
                               default_activation_bw=8,
                               default_param_bw=4,
                               use_cuda=False,
                               config_file=_get_config_file(is_symmetric=True, strict_symmetric=False,
                                                            unsigned_symmetric=False, pcq=True))
    seq_params = SeqMseParams(num_batches=1)
    dataloader = unlabeled_data_loader([torch.randn((1, 3, 8, 8)), torch.randn((1, 3, 8, 8)), torch.randn((1, 3, 8, 8))])
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)
    seq_mse.apply_seq_mse_algo()
    for cg_op in seq_mse.dependency_graph.conn_graph.ordered_ops:
        if cg_op.type in SUPPORTED_MODULES:
            param_names = [param_name for param_name, (_, param_type) in cg_op.parameters.items() if param_type == "weight"]
            assert len(param_names) == 1
            quantizer = seq_mse.sim.qc_quantize_op_dict[param_names[0]]
            assert quantizer.is_encoding_frozen()


class TestDependencyGraph:

    @pytest.mark.parametrize("model, cached_data", [(models_for_tests.single_residual_model(),
                                                    {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}),
                                                    (models_for_tests.concat_model(),
                                                     {"input1": np.random.randn(1, 3, 8, 8).astype(np.float32),
                                                      "input2": np.random.randn(1, 3, 8, 8).astype(np.float32),
                                                      "input3": np.random.randn(1, 3, 8, 8).astype(np.float32)}),
                                                    (models_for_tests.multi_input_model(),
                                                     {"input1": np.random.randn(32, 1, 28, 28).astype(np.float32),
                                                      "input2": np.random.randn(32, 1, 28, 28).astype(np.float32)}),
                                                    (models_for_tests.mobilenetv2(),
                                                     {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}),
                                                    (models_for_tests.resnet18(),
                                                     {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)})
                                                    ])
    def test_dependency_graph(self, model, cached_data):
        """ Compare the one-shot and iterative outputs """
        dl = data_loader(tuple(cached_data.values()))
        dep_graph = DependencyGraph(model, dl, 1)

        sorted_nodes = dep_graph.get_topologically_sorted_nodes()
        model_outputs = [node.name for node in model.model.graph.output]

        session = _build_session(model.model)
        one_shot_output = session.run(None, input_feed=cached_data)[0]

        # Iterate over the topologically sorted nodes to gather intermediate outputs and
        # provide them as inputs to the subsequent subgraph.
        extractor = Extractor(model.model)
        for i in range(1, len(sorted_nodes)):
            subgraph_inp_names, subgraph_out_names = dep_graph.get_subgraph_inp_out_names(sorted_nodes[i])
            model_ = extractor.extract_model(subgraph_inp_names, subgraph_out_names)
            session = _build_session(model_)
            input_dict = _create_input_dict(subgraph_inp_names, cached_data)
            cached_data[subgraph_out_names[0]] = session.run(None, input_feed=input_dict)[0]

        # Final subgraph extraction and session run
        subgraph_inp_names = subgraph_out_names
        subgraph_out_names = model_outputs
        model_ = extractor.extract_model(subgraph_inp_names, subgraph_out_names)
        session = _build_session(model_)
        input_dict = _create_input_dict(subgraph_inp_names, cached_data)
        iterative_output = session.run(None, input_feed=input_dict)[0]

        assert np.all(iterative_output == one_shot_output)
