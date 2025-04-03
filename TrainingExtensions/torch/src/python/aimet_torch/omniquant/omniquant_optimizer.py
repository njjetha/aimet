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
""" Optimizer for Omniquant """

import contextlib
import os
from pathlib import Path
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.peft_model import PeftModel
from safetensors.numpy import save_file, load_file
import tempfile
import torch
from torch import nn
from typing import Union

from aimet_torch._base.quantsim import logger
from aimet_torch.utils import CachedDataset
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch._base.adaround.activation_sampler import get_block_inputs, get_block_outputs

from .decoder_processor import get_transformer_processor
from .omniquant_config import OmniquantConfig
from .let_modules import LETModule
from ._utils import _convert_sim_to_letsim, _convert_letsim_to_sim

OMNIQUANT_ARTIFACT_DIR = "./aimet_omniquant_artifact/"
OMNIQUANT_METADATA_SAFETENSOR_NAME = "aimet_omniquant_metadata.safetensor"

class Omniquant:
    """
    Omniquant for Post Training Quantization (PTQ)
    """
    @classmethod
    def apply_omniquant(cls, quant_sim: QuantizationSimModel, model: torch.nn.Module, omniquant_config: OmniquantConfig, dataloader,
                        output_path: str = OMNIQUANT_ARTIFACT_DIR) -> torch.nn.Module:
        """
        Returns model with with omniquant weight, and save metadata in safetensor format to output path. Metadata safetensor
        can be used in update_lora_weights to update lora adaptor weights for peft lora model.

        :param quant_sim: QuantizationSimModel object to optimize with Omniquant.
        :param model: Original fp32 model from which quant_sim was created.
        :param omniquant_config: Configuration for Omniquant optimization.
        :param dataloader: Dataloader used to train model.
        :param output_path: Path to save {layer_name: scale} metadata safetensor.
        :return: Model with Omniquant weights.
        """
        @contextlib.contextmanager
        def disable_dynamic_cache():
            # Disable dynamic_cache for LET blockwise training, and restore after optimization.
            quant_sim_use_cache_bool, model_use_cache_bool = quant_sim.model.config.use_cache, model.config.use_cache
            quant_sim.model.config.use_cache, model.config.use_cache = False, False
            try:
                yield
            finally:
                quant_sim.model.config.use_cache, model.config.use_cache = quant_sim_use_cache_bool, model_use_cache_bool
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)

        cls._validate_omniquant_config(omniquant_config)
        _convert_sim_to_letsim(quant_sim)
        with disable_dynamic_cache():
            quant_sim.model = cls._apply_omniquant(quant_sim, model, omniquant_config, dataloader, output_path)

        return quant_sim.model

    # pylint: disable=too-many-locals
    # pylint: disable=unused-variable
    # pylint: disable=unused-argument
    @classmethod
    def _apply_omniquant(cls, quant_sim: QuantizationSimModel, model: torch.nn.Module, omniquant_config: OmniquantConfig,
                         dataloader, output_path: str) -> torch.nn.Module:
        """
        Implemenatation to run omniquant optimization block by block. Return model with optimized weights.

        :param quant_sim: QuantizationSimModel object to optimize with Omniquant.
        :param model: Original fp32 model from which quant_sim was created.
        :param omniquant_config: Configuration for Omniquant optimization.
        :param dataloader: Dataloader used to train model.
        :param output_path: Path where to store artifacts.
        :return: Model with Omniquant weights.
        """
        transformer_processor = get_transformer_processor(model)
        fp_transformer_block_list = transformer_processor.get_decoder_list(model)
        qt_transformer_block_list = transformer_processor.get_decoder_list(quant_sim.model)
        device = model.device

        def calibration_wrapper(model, dataloader):
            for batch_id, batch in enumerate(dataloader):
                if batch_id < omniquant_config.num_batch:
                    batch = tuple(batch) # Dataloader returns List[torch.Tensor]
                    model(*batch)
                else:
                    break

        quant_sim.compute_encodings(calibration_wrapper, dataloader)

        with tempfile.TemporaryDirectory() as tempdir:
            cached_dir = os.path.join(tempdir, 'cached_dataset')
            cached_dataset = CachedDataset(dataloader, omniquant_config.num_batch, cached_dir)

            cached_fp_dataset, cached_qt_dataset = get_block_inputs(
                model, quant_sim, ".".join([transformer_processor.transformer_block_list_path, "0"]), cached_dataset, omniquant_config.cache_on_cpu,
                lambda model, input: model.forward(*input), omniquant_config.num_batch, cached_dir, incl_kwargs=True
            )

            for block_num, (fp_block, qt_block) in enumerate(zip(fp_transformer_block_list, qt_transformer_block_list)):
                qt_let_pair_list = transformer_processor.get_let_module_pair(qt_block)
                transformer_processor.init_let_params(qt_let_pair_list)

                for epoch in range(omniquant_config.num_epoch):
                    for batch_num in range(omniquant_config.num_batch):
                        fp_input, qt_input = cached_fp_dataset[batch_num], cached_qt_dataset[batch_num]
                        # Do block-wise training.
                        cls._block_wise_training(omniquant_config, fp_input, qt_input, fp_block, qt_block, qt_let_pair_list)

                # fold_let_params
                for module in model.modules():
                    if isinstance(module, LETModule):
                        module.fold_let_params()

                get_block_outputs(
                        fp_block, qt_block, False, cached_fp_dataset, cached_qt_dataset, omniquant_config.cache_on_cpu,
                        lambda decoder_block, *args, **kwargs: decoder_block(*args, **kwargs), device, cached_dir
                    )

        # pylint: disable=protected-access
        # pylint: disable=unnecessary-comprehension
        cls._dump_meta_data(quant_sim.model, output_path)

        with torch.no_grad():
            _convert_letsim_to_sim(quant_sim)

        # QDQ on models to fold quantizations into weight params.
        quant_sim._apply_qdq_to_model_parameters(quant_sim.model)

        # quantized module -> original module
        all_modules_in_original_model = [module for module in quant_sim.model.modules()]
        quant_sim._remove_quantization_wrappers(quant_sim.model, all_modules_in_original_model)

        return quant_sim.model

    @classmethod
    def _block_wise_training(cls, omniquant_config, fp_input, qt_input, fp_block, qt_block, qt_let_pair_list):
        """
        Run block-wise traing on LET parameters. Use fp_block output as ground truth and qt_block output as
        model output. Use omniquant_config.input_symmetry to choose input for fp and qt block.

        :param omniquant_config: Configuration for Omniquant optimization.
        :param fp_input: block output from previous block in fp model.
        :param qt_input: block output from previous block in qt model.
        :param fp_block: decoder block in fp model.
        :param qt_block: decoder block in qt model.
        :param qt_let_pair_list: let_pair_list in qt model. Use to get LET training parameters.
        """
        let_param = []
        for _pair in qt_let_pair_list:
            for q_module in _pair.prev + _pair.follow:
                let_param.extend([_v for _v in q_module.get_let_params().values() if isinstance(_v, nn.Parameter)])

        grouped_params = [
                {"params": let_param, "lr": omniquant_config.let_lr, "weight_decay":  0.},
            ]
        loss_fn = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.AdamW(grouped_params)

        def _process_block_input(_block_input):
            """ Unpack and detach input tensor. """
            _args, _kwargs = _block_input
            _args = [_arg.detach() for _arg in _args]
            _kwargs = {_k: _v.detach() if isinstance(_v, torch.Tensor) else _v for _k, _v in _kwargs.items()}
            return _args, _kwargs

        # Get target output (ground truth)
        target_input = fp_input if omniquant_config.input_symmetry.startswith("fp") else qt_input
        _args, _kwargs = _process_block_input(target_input)
        target_outputs = fp_block(*_args, **_kwargs)[0]

        # Get model output (prediction)
        model_input = fp_input if omniquant_config.input_symmetry.endswith("fp") else qt_input
        _args, _kwargs = _process_block_input(model_input)
        qt_output = qt_block(*_args, **_kwargs)[0]

        loss = loss_fn(qt_output, target_outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    @classmethod
    def _validate_omniquant_config(cls, omniquant_config: OmniquantConfig):
        """ Validate omniquant config """
        # input_symmetry should be one of qtqt, qtfp, fpqt, fpfp
        input_symmetry_error_msg = f"Expect omniquant_config.input_symmetry be one of qtqt, qtfp, fpqt, fpfp but got {omniquant_config.input_symmetry}."
        assert len(omniquant_config.input_symmetry) == 4, input_symmetry_error_msg
        assert omniquant_config.input_symmetry[:2] in ("qt", "fp"), input_symmetry_error_msg
        assert omniquant_config.input_symmetry[2:] in ("qt", "fp"), input_symmetry_error_msg

    @classmethod
    def _dump_meta_data(cls, model, output_path):
        """ Traverse quantized model, get LET scales, and dump {module_name: scale} dict to output path. """
        meta_data = {}
        for name, module in model.named_modules():
            if isinstance(module, LETModule):
                if module.prev_scale is not None:
                    meta_data[f"{name}.prev"] = module.prev_scale.data.numpy()
                if  module.foll_scale is not None:
                    meta_data[f"{name}.foll"] = module.foll_scale.data.numpy()

        save_file(meta_data, output_path/OMNIQUANT_METADATA_SAFETENSOR_NAME)
        logger.info("Aimet omniquant metadata saved at %s", (output_path/OMNIQUANT_METADATA_SAFETENSOR_NAME).absolute())

def _get_meta_dict(meta_data):
    """ Process meta_data to {module_name: {"foll"/"prev": scale}} dict. """
    meta_data_dict = {}
    def _get_name_suf(key):
        """ split module name and foll/prev suffix """
        split_key = key.split(".")
        return ".".join(split_key[:-1]), split_key[-1]

    for key, scale in meta_data.items():
        let_module_name, prev_foll = _get_name_suf(key)
        assert prev_foll in ("foll", "prev"), f"Expect metadata suffix = foll or prev, but got {prev_foll}"
        meta_data_dict[let_module_name] = {prev_foll: torch.from_numpy(scale)}

    return meta_data_dict

def update_lora_weights(peft_model: PeftModel, omniquant_metadata_path: Union[str, Path]):
    """
    Read omniquant metadata safetensor, and apply LET scale to LoraLinear's lora_A and lora_B.
    Lora_A = Lora_A*foll and Lora_B = Lora_B/prev.

    :param peft_model: Peft Lora model.
    :param omniquant_metadata_path: Path to omniquant metadata generated at the end of apply omniquant.
    """
    meta_data = load_file(omniquant_metadata_path)
    meta_data_dict = _get_meta_dict(meta_data)

    assert isinstance(peft_model, PeftModel), f"Expect peft_mdel class PeftModel, but got {peft_model.__class__}"
    hf_model = peft_model.base_model.model # PeftModel -> LoraModel (base_model) -> Transformer model (model)
    with torch.no_grad():
        for _module_name, _let_scale_dict in meta_data_dict.items():
            let_module = hf_model.get_submodule(_module_name)
            if isinstance(let_module, LoraLinear):
                prev = _let_scale_dict["prev"] if "prev" in _let_scale_dict else None
                foll = _let_scale_dict["foll"] if "foll" in _let_scale_dict else None

                if foll is not None:
                    # Lora_A weight [lora_dim, lin_in_dim]
                    for module in getattr(let_module, "lora_A").values():
                        new_weight = module.weight*(foll.unsqueeze(0))
                        module.weight.copy_(new_weight)

                if prev is not None:
                    # Lora_B weight [lin_out_dim, lora_dim]
                    for module in getattr(let_module, "lora_B").values():
                        new_weight = module.weight/(prev.unsqueeze(-1))
                        module.weight.copy_(new_weight)
