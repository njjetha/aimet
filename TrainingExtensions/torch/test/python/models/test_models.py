# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Models for use in unit testing """

# pylint: skip-file
from collections import namedtuple
from typing import Dict, List, Tuple, Optional
import math

import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn as nn
from torchvision.ops import roi_align

import aimet_torch._base.nn.modules.custom as aimet_modules



class ModelWithMatMul(torch.nn.Module):
    """
    Model with MatMul module
    """
    def __init__(self):
        super().__init__()
        self.act1 = nn.PReLU()
        self.act2 = nn.ReLU()
        self.matmul = aimet_modules.MatMul()

    def forward(self, *inputs):
        x = self.act1(inputs[0])
        y = self.act2(inputs[1])
        y = y.reshape(10, 4, 5)
        return self.matmul(x, y)

class ModelWithMatMul2(torch.nn.Module):
    """
    Model with MatMul module
    """
    def __init__(self):
        super().__init__()
        self.act1 = nn.PReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.Softmax()
        self.matmul = aimet_modules.MatMul()

    def forward(self, *inputs):
        x = self.act1(inputs[0])
        y = self.act3(inputs[1])
        y = self.act2(y)
        y = y.reshape(10, 4, 5)
        return self.matmul(x, y)

class ModelWithMatMul4(torch.nn.Module):
    """
    Model with MatMul module
    """
    def __init__(self):
        super().__init__()
        self.act1 = nn.ReLU()
        self.matmul = aimet_modules.MatMul()

    def forward(self, *inputs):
        x = self.act1(inputs[0])
        return self.matmul(x, inputs[1])

class ModelWithMatMul5(torch.nn.Module):
    """
    Model with MatMul module
    """
    def __init__(self):
        super().__init__()
        self.matmul = aimet_modules.MatMul()

    def forward(self, *inputs):
        return self.matmul(inputs[0], inputs[1])

class ModelWithMatMul6(torch.nn.Module):
    """
    Model with MatMul module
    """
    def __init__(self):
        super().__init__()
        self.act1 = nn.ReLU()
        self.permute = aimet_modules.Permute()
        self.matmul = aimet_modules.MatMul()

    def forward(self, *inputs):
        x = self.act1(inputs[1])
        x = self.permute(x, (1, 0))
        return self.matmul(inputs[0], x)

class ModelWithUnusedMatmul(ModelWithMatMul5):

    def __init__(self):
        super().__init__()
        self.unused_matmul = aimet_modules.MatMul()

class ModelWithGroupNorm(torch.nn.Module):
    """
    Model with GroupNorm module
    """
    def __init__(self):
        super().__init__()
        self.gn = torch.nn.GroupNorm(2, 6)
        self.gn_with_no_affine = torch.nn.GroupNorm(2, 6, affine=False)

    def forward(self, *inputs):
        return self.gn_with_no_affine(self.gn(inputs[0]))

class ModelWithEmbedding(torch.nn.Module):
    """
    Model with embedding module
    """
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def forward(self, *inputs):
        return self.embedding(inputs[0])

class SingleResidual(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidual, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # All layers above are same as ResNet
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)

        # The output of Conv2d layer above(conv3) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=2, bias=True)
        self.ada = nn.AdaptiveAvgPool2d(5)
        self.fc = nn.Linear(72, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SingleResidualWithAvgPool(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidualWithAvgPool, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # All layers above are same as ResNet
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)

        # The output of Conv2d layer above(conv3) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=2, bias=True)
        self.ada = nn.AvgPool2d(5)
        self.fc = nn.Linear(72, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SingleResidualWithModuleAdd(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidualWithModuleAdd, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # All layers above are same as ResNet
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)

        # The output of Conv2d layer above(conv3) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.add = aimet_modules.Add()
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=2, bias=True)
        self.ada = nn.AdaptiveAvgPool2d(5)
        self.fc = nn.Linear(72, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x = self.add(x, residual)
        x = self.relu3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiInput(nn.Module):
    """ A model with multiple inputs.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=3):
        super(MultiInput, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x1 = self.conv2(x1)
        x2 = self.conv3(inputs[1])
        x = x1 + x2
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DictInputModel(nn.Module):
    """ Model with dictionary as input. """
    def __init__(self, num_classes=3):
        super(DictInputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0]['inp_1'])
        x1 = self.conv2(x1)
        x2 = self.conv3(inputs[0]['inp_2'])
        x = x1 + x2
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConcatModel(nn.Module):
    """ A model with concat op.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(ConcatModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=2, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2)
        self.fc = nn.Linear(504, num_classes)

    def forward(self, *inputs):
        c1 = self.conv1(inputs[0])
        c2 = self.conv2(inputs[1])
        c3 = self.conv3(inputs[2])
        cat_inputs = [c1, c2, c3]
        x = torch.cat(cat_inputs, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModuleListModel(nn.Module):
    """ A model with modules defined using ModuleLists.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(ModuleListModel, self).__init__()
        self.mod_list = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),       # use 4th
            nn.ReLU(inplace=True),      # use 3rd
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),                # use 5th
            nn.ReLU(),      # dummy unused op
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False)        # use 1st
        ])
        self.seq_list = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2),                # use 6th
            nn.ReLU(),      # dummy unused op
            nn.BatchNorm2d(16),     # use 2nd
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.mod_list[4](inputs[0])
        x = self.seq_list[2](x)
        x = self.mod_list[1](x)
        x = self.mod_list[0](x)
        x = self.mod_list[2](x)
        x = self.seq_list[0](x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TinyModel(nn.Module):
    """ Use this model for unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2, bias=True)
        self.fc = nn.Linear(36, 12)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class QuantSimTinyModel(nn.Module):
    """ Use this model for quantsim_config unit testing purposes. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(QuantSimTinyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU6(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2, bias=True)
        self.fc = nn.Linear(36, 12)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithDropouts(nn.Module):
    """ Use this model for unit testing purposes. """

    def __init__(self):
        super(ModelWithDropouts, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=.4)
        self.dropout2 = nn.Dropout2d(p=.6)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithReusedNodes(nn.Module):
    """ Model that reuses a relu module. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithReusedNodes, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithFunctionalOps(nn.Module):
    """ Model that uses functional modules instead of nn.Modules. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithFunctionalOps, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = F.linear(x, torch.randn(10, 2592))
        return x


class SequentialModel(nn.Module):
    """ A model with modules defined using nn.Sequential.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(SequentialModel, self).__init__()
        self.seq_list = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.seq_list(inputs[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):
    """ A Simple Super Node Model used as building block in Hierarchical Model  """

    def __init__(self, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(64, 64, bias=False, **kwargs)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.bn = nn.BatchNorm2d(64, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        x = self.conv(inputs[0])
        x = self.dropout(x)
        x = self.bn(x)
        return self.relu(x)


class MultiConv2dModel(nn.Module):
    """ Sequential Model contains sequences of BasicConv2d Model  """

    def __init__(self):
        super(MultiConv2dModel, self).__init__()
        self.seq_list = nn.Sequential(
            BasicConv2d(kernel_size=3),
            BasicConv2d(kernel_size=1),
            BasicConv2d(kernel_size=3)
        )

    def forward(self, *inputs):
        return self.seq_list(inputs[0])


class NestedModel(nn.Module):
    """ Aggregation Model contains two instance of Tiny Model """

    def __init__(self):
        super(NestedModel, self).__init__()
        self.tm1 = TinyModel()
        self.tm2 = TinyModel()

    def forward(self, *inputs):
        c1 = self.tm1(inputs[0])
        c2 = self.tm2(inputs[1])
        cat_inputs = [c1, c2]
        x = torch.cat(cat_inputs, 1)
        return x


class HierarchicalModel(nn.Module):
    """ Aggregation Model contains multi-level of PyTorch Module
        Expected 5 inputs with shapes  in the following order:
            (1, 64, 32, 32)
            (1,  3, 32, 32)
            (1, 64, 32, 32)
            (1,  3, 32, 32)
            (1,  3,  8,  8) """

    def __init__(self):
        super(HierarchicalModel, self).__init__()
        self.conv1 = BasicConv2d(kernel_size=3)
        self.conv2 = BasicConv2d(kernel_size=3)
        self.multi_conv = MultiConv2dModel()
        self.nm1 = NestedModel()
        self.nm2 = NestedModel()
        self.sq = SequentialModel()

    def forward(self, *inputs):
        x = self.conv1((inputs[0]))
        x = x.narrow(1, 0, 3)
        c1 = self.nm1(x, inputs[1])
        x = self.conv2(inputs[2])
        x = self.multi_conv(x)
        x = x.narrow(1, 0, 3)
        c2 = self.nm2(x, inputs[3])
        c3 = self.sq(inputs[4])
        cat_inputs = [c1, c2, c3]
        x = torch.cat(cat_inputs, 1)
        return x


class PassThroughOpLastLayerModel(nn.Module):
    """ Model with PassThroughOp as last layer. Expect input shape (1, 3, 32, 32) """

    def __init__(self):
        super(PassThroughOpLastLayerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.passthrough = torch.nn.Identity()

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.passthrough(x)
        return x


class TransposedConvModel(torch.nn.Module):
    """
    Model with transposed conv2D
    """
    def __init__(self):
        super(TransposedConvModel, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3)
        self.bn2 = torch.nn.BatchNorm2d(10)

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return x


class TransposedConvModelWithoutBN(torch.nn.Module):
    """
    Model with transposed conv2D
    """
    def __init__(self):
        super(TransposedConvModelWithoutBN, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3)

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class TupleOutputModel(torch.nn.Module):
    """
    Model with Tuple of Tensors as output
    """
    def __init__(self):
        super(TupleOutputModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1)

    def forward(self, *inputs):
        c1 = self.conv1(inputs[0])
        c2 = self.conv2(inputs[0])
        c3 = self.conv3(inputs[0])
        return c1, c2, c3


class MultiOutputModel(torch.nn.Module):
    """
    Model with Tuple of Tensors as output
    """
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.layer = TupleOutputModel()
        self.conv1 = torch.nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 4, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x, y, z = self.layer(inputs[0])
        x1 = self.conv1(x)
        x2 = self.conv2(y)
        x3 = self.conv3(z)
        return torch.cat([x1, x2, x3], 1)


class ConfigurableTupleOutputModel(torch.nn.Module):
    """
    Model with Tuple of Tensors as output with configurable channels
    """
    def __init__(self, channels=(2, 4, 6)):
        super(ConfigurableTupleOutputModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1)

    def forward(self, *inputs):
        c1 = self.conv1(inputs[0])
        c2 = self.conv2(inputs[1])
        c3 = self.conv3(inputs[2])
        return c1, c2, c3


class SingleLayerRNNModel(nn.Module):
    """
    Model using torch.nn.RNN module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(SingleLayerRNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=3, hidden_size=5, num_layers=1)

    def forward(self, x, hx=None):
        return self.rnn(x, hx)


class SingleLayerBidirectionalLstmModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(SingleLayerBidirectionalLstmModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=1, bidirectional=True)

    def forward(self, x, hx=None):
        return self.lstm(x, hx)


class TwoLayerBidirectionalLSTMModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(TwoLayerBidirectionalLSTMModel, self).__init__()
        self.recurrent = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=2, bidirectional=True)

    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class TwoLayerBidirectionaRNNModel(nn.Module):
    """
    Model using torch.nn.RNN module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(TwoLayerBidirectionaRNNModel, self).__init__()
        self.recurrent = torch.nn.RNN(input_size=3, hidden_size=5, num_layers=2, bidirectional=True)

    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class TwoLayerBidirectionalGRUModel(nn.Module):
    """
    Model using torch.nn.GRU module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(TwoLayerBidirectionalGRUModel, self).__init__()
        self.recurrent = torch.nn.GRU(input_size=3, hidden_size=5, num_layers=2, bidirectional=True)

    def forward(self, x, hx=None):
        return self.recurrent(x, hx)


class MultiLayerRNNModel(nn.Module):
    """
    Model using torch.nn.RNN module with multiple layers
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(MultiLayerRNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=3, hidden_size=5, num_layers=2)

    def forward(self, x, hx=None):
        return self.rnn(x, hx)


class RNNCellModel(nn.Module):
    """
    Model using torch.nn.RNNCell module
    Expected input shape = (SEQ_LENGTH=10, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(RNNCellModel, self).__init__()
        self.rnn_cell = torch.nn.RNNCell(input_size=3, hidden_size=5)

    def forward(self, x, hx0=None):
        output = []
        for i in range(x.shape[0]):
            hx0 = self.rnn_cell(x[i], hx0)
            output.append(hx0)
        return tuple(output), hx0


class LSTMModel(nn.Module):
    """
    Model using torch.nn.LSTM module
    Expected input shape = (SEQ_LENGTH, BATCH_SIZE, INPUT_SIZE=3)
    """
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=1)

    def forward(self, x, hx_cx=None):
        return self.rnn(x, hx_cx)


class NestedSequentialModel(nn.Module):
    """
    Model using nested Sequential modules
    Expected input shape = (1, 3, 8, 8)
    """
    def __init__(self, num_classes=3):
        super(NestedSequentialModel, self).__init__()
        self.inner_seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16)
        )
        self.seq_list = nn.Sequential(
            self.inner_seq,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.seq_list(inputs[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithFunctionalReLU(nn.Module):
    """ Model that uses functional ReLU instead of nn.Modules. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x


class ModelWithDuplicateReLU(nn.Module):
    """ Model that uses single ReLU instances multiple times in the forward. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x


class ModelWithTwoInputs(nn.Module):

    def __init__(self):
        super(ModelWithTwoInputs, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x2 = self.relu1_b(self.maxpool1_b(self.conv1_b(x2)))
        x = x1 + x2
        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class ModelWithTransposeConv(nn.Module):

    def __init__(self):
        super(ModelWithTransposeConv, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1280, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x2 = self.relu1_b(self.maxpool1_b(self.conv1_b(x2)))
        x = x1 + x2
        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class SimpleConditional(torch.nn.Module):
    """
    Model using conditional paths
    Expected input shape = (1, 3)
    """
    def __init__(self):
        super(SimpleConditional, self).__init__()
        self.prelu1 = torch.nn.PReLU(init=.3)
        self.prelu2 = torch.nn.PReLU(init=.4)
        self.linear1 = torch.nn.Linear(3, 2)
        self.linear2 = torch.nn.Linear(3, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, _input, condition):
        if condition:
            x = self.linear1(_input)
            x = x.view(x.size(0), -1)
            x = self.prelu1(x)
            return x
        x = self.linear2(_input)
        x = self.prelu2(x)
        x = self.softmax(x)
        return x


class LinearAndLSTMModel(torch.nn.Module):
    def __init__(self):
        super(LinearAndLSTMModel, self).__init__()

        self.linear = torch.nn.Linear(10, 4)
        self.prelu = torch.nn.PReLU(init=.3)
        self.recurrent = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=2)

    def forward(self, x, h_and_c=None):
        x = self.linear(x)
        x = self.prelu(x)
        x = torch.unsqueeze(x, 1)
        return self.recurrent(x, h_and_c)



class RoiAlignPyTorch(torch.nn.Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoiAlignPyTorch, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_align(input=features,
                         boxes = rois,
                         output_size = [self.aligned_height, self.aligned_width],
                         spatial_scale = self.spatial_scale,
                         sampling_ratio = 0)

class RoiModel(torch.nn.Module):

    def __init__(self, height, width, scale):
        super(RoiModel, self).__init__()
        self.roi = RoiAlignPyTorch(height, width, scale)

    def forward(self, *inputs):
        return self.roi(*inputs)


class InputOutputDictModel(nn.Module):
    def __init__(self):
        super(InputOutputDictModel, self).__init__()
        self.mul1 = aimet_modules.Multiply()
        self.mul2 = aimet_modules.Multiply()
        self.mul3 = aimet_modules.Multiply()

    def forward(self, inputs: Dict[str, torch.Tensor]):
        ab = self.mul1(inputs['a'], inputs['b'])
        bc = self.mul2(inputs['b'], inputs['c'])
        ca = self.mul3(inputs['c'], inputs['a'])

        output_def = namedtuple('output_def', ['ab', 'bc', 'ca'])
        return output_def(ab, bc, ca)


class Float32AndInt64InputModel(nn.Module):
    """
    This model uses a list of Tensors as input. The input Tensor list contains both float32 and int63 tensors.
    """
    def __init__(self):
        super(Float32AndInt64InputModel, self).__init__()
        self.index_feature_map = 0
        self.index_x = 1
        self.index_y = 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.add = aimet_modules.Add()

    def forward(self, *inputs: List[torch.Tensor]):
        grid_x = inputs[self.index_x]
        grid_y = inputs[self.index_y]

        x = inputs[self.index_feature_map]
        x = self.conv1(x)
        x = self.bn1(x)
        f_00 = x[:, :, grid_x, grid_y]
        f_01 = x[:, :, grid_y, grid_x]
        return self.add(f_00, f_01)


class Conv3dModel(nn.Module):

    def __init__(self):
        super(Conv3dModel, self).__init__()

        self.conv1 = nn.Conv3d(3, 6, 3)
        self.bn1 = nn.BatchNorm3d(6)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(6, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.relu2 = nn.ReLU()

    def forward(self, inp):

        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class Conv3dModel1(nn.Module):

    def __init__(self):
        super(Conv3dModel1, self).__init__()

        self.conv1 = nn.Conv3d(3, 6, 3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(6)

        self.conv2 = nn.Conv3d(6, 8, 3)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(8)

    def forward(self, inp):

        out = self.conv1(inp)
        out = self.relu1(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        return out


class ModuleWithListInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape = aimet_modules.Reshape()
        self.conv = torch.nn.Conv2d(in_channels=1,
                                    out_channels=16,
                                    kernel_size=3)

    def forward(self, *inputs):
        # Module with list input (second argument)
        x = self.reshape(inputs[0], [-1, 1, 32, 128])
        x = self.conv(x)
        return x


class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(1000, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        return self.softmax(x)


class MultiplePReluModel(nn.Module):
    def __init__(self, num_parameters: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
        self.act1 = nn.PReLU(num_parameters=num_parameters)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
        self.act2 = nn.PReLU(num_parameters=num_parameters)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3)
        self.act3 = nn.PReLU(num_parameters=num_parameters)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        return x


class GroupedConvModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=2, bias=False)

    def forward(self, *inputs):
        return self.conv(inputs[0])


class CustomGroupedConvModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, bias=False)

    def forward(self, *inputs):
        input1, input2 = inputs
        output1, output2 = self.conv1(input1), self.conv2(input2)
        return torch.cat([output1, output2], dim=1)


class NestedSeqModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Softmax()))

    def forward(self, x):
        return self.m1(x)


class NestedSeqModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Sequential(torch.nn.Sequential(torch.nn.ReLU(), NestedSeqModule()))

    def forward(self, x):
        return self.m(x)


class NestedModelWithOverlappingNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = NestedSeqModule()

    def forward(self, x):
        return self.m(x)


class ModelWithModuleList(torch.nn.Module):

    def __init__(self):
        super(ModelWithModuleList, self).__init__()
        self.m = torch.nn.ModuleList([NestedSeqModule()])

    def forward(self, x):
        return self.m[0](x)


class ModelWithSplitModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.split = aimet_modules.Split()

    def forward(self, *inputs):
        return self.split(inputs[0], 2)


class ModelWithReluAfterSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_module = ModelWithSplitModule()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, *inputs):
        chunks = self.split_module(inputs[0])
        return self.relu1(chunks[0]), self.relu2(chunks[1]), self.relu3(chunks[2])


class ModuleList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(256, 256) for _ in range(3))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x)) if i < 3 - 1 else layer(x)
        return x


class ModelWithReusedInitializers(torch.nn.Module):
    def __init__(self, repetition):
        super(ModelWithReusedInitializers, self).__init__()
        self.modulelist = ModuleList()
        self.repetition = repetition

    def forward(self, x):
        for i in range(self.repetition):
            x = self.modulelist(x)
        return x


class TinyModelWithNoMathInvariantOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.mul1 = aimet_modules.Multiply()
        self.mul2 = aimet_modules.Multiply()
        self.add1 = aimet_modules.Add()

    def forward(self, x):
        conv_output = self.conv1(x)
        y = self.mul1(conv_output, 2)
        m = self.add1(y, 3)
        z = self.mul2(m, 5)
        return z


class ModelWithThreeLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 3072)
        self.linear3 = nn.Linear(3072, 768)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


class CustomFunctionalConv(nn.Module):
    """
    Custom module with functional conv
    Expected input shape: (1, 3, 16, 16)
    """
    def __init__(self):
        super().__init__()

        kernel_size = 5
        channels = 3
        sigma = 0.7
        stride = 2

        kernel = torch.zeros(kernel_size, kernel_size)
        mean_loc = int((kernel_size - 1) / 2)  # Because 0 indexed
        kernel[mean_loc, mean_loc] = 1
        kernel = torch.from_numpy(ndimage.gaussian_filter(kernel.numpy(), sigma=sigma))

        # Make a dwise conv out of the kernel
        # Weights of shape out_channels, in_channels/groups, k, k
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.channels, stride=self.stride)

class SmallLinearModel(nn.Module):
    def __init__(self):
        super(SmallLinearModel, self).__init__()
        self.linear = nn.Linear(3, 8)
        self.linear2 = nn.Linear(8, 3, bias=False)
        self.innerlinear = InnerLinear()
        self.prelu = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.groupnorm = nn.GroupNorm(2, 8)

    def forward(self, inp):
        x = self.linear(inp)
        x = self.prelu(x)
        x = self.innerlinear(x)
        x = self.groupnorm(x)
        x = self.prelu2(x)
        return self.linear2(x)

class InnerLinear(nn.Module):
    def __init__(self):
        super(InnerLinear, self).__init__()
        self.in_linear1 = nn.Linear(8, 16)
        self.linear_modlist = nn.ModuleList([torch.nn.Linear(16, 8)])

    def forward(self, inp):
        x = self.in_linear1(inp)
        x = self.linear_modlist[0](x)
        return x


class ModelWithMultiInputOps(torch.nn.Module):
    def __init__(self):
        super(ModelWithMultiInputOps, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer('gemm_weight', torch.ones([1, 8, 12, 24], dtype=torch.float32))
        self.matmul = aimet_modules.MatMul()
        self.min = aimet_modules.Minimum()
        self.register_buffer('constant', torch.ones([1, 8, 12, 12], dtype=torch.float32))
        self.concat = aimet_modules.Concat()
        self.register_buffer('conv_bias', torch.zeros([3], dtype=torch.float32))
        self.dynamic_conv = aimet_modules.DynamicConv2d(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
        self.fc = nn.Linear(24, 3)

    def forward(self, inp, weight):
        x1_conv = self.conv(inp)
        x1_bn = self.bn(x1_conv)
        x1 = self.relu(x1_bn)
        x2 = self.matmul(x1, self.gemm_weight)
        x3 = self.fc(x2)
        x4 = self.min(self.constant, x1)
        x5 = self.concat(x1_bn, self.constant, x1)
        x6 = self.dynamic_conv(x5, weight, self.conv_bias)
        return x3, x6+2, x4


class SmallMnist(nn.Module):
    def __init__(self):
        super(SmallMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(80, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.relu2(self.conv2_drop(x))
        x = x.view(-1, 80)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class ModelWithOneSplit(nn.Module):
    def __init__(self):
        super(ModelWithOneSplit, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        z = self.conv3(x)
        return z


class ModelWithMatMul3(nn.Module):
    def __init__(self):
        super(ModelWithMatMul3, self).__init__()
        self.matmul_1 = aimet_modules.MatMul()

    def forward(self, x, y):
        y = self.matmul_1(x, y)
        return y


class ModelWithFlatten(nn.Module):
    def __init__(self):
        super(ModelWithFlatten, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2,
                                      padding=2, bias=False)
        self.maxpool_1 = nn.MaxPool2d(2)
        self.relu_1 = torch.nn.ReLU(inplace=True)
        self.fc_1 = torch.nn.Linear(1, 10, bias=False)

    def forward(self, x):
        x = self.relu_1(self.maxpool_1(self.conv_1(x)))
        x = torch.reshape(x, shape=(1, 648))
        x = torch.transpose(x, 0, 1)
        x = self.fc_1(x)
        return x


class ModelWithSeveralDataMovementOps(nn.Module):
    def __init__(self):
        super(ModelWithSeveralDataMovementOps, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2,
                                      padding=2, bias=False)
        self.maxpool_1 = nn.MaxPool2d(2)
        self.fc_1 = torch.nn.Linear(648, 10, bias=False)
        self.fc_2 = torch.nn.Linear(10, 10, bias=False)
        self.fc_3 = torch.nn.Linear(9, 10, bias=False)

    def forward(self, x):
        y = self.maxpool_1(self.conv_1(x))
        # branch 1
        x = torch.reshape(y, shape=(1, 648))
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = torch.reshape(x, shape=(1, 10))
        x = torch.transpose(x, 0, 1)
        # branch 2
        y = self.fc_3(y)
        x = y + torch.transpose(x, 0, 1)
        return x


class ModelWithTwoInputsTwoOutputs(nn.Module):
    def __init__(self):
        super(ModelWithTwoInputsTwoOutputs, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x2 = self.relu1_b(self.maxpool1_b(self.conv1_b(x2)))
        return x1, x2


class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul_rr = aimet_modules.Multiply()
        self.mul_ii = aimet_modules.Multiply()
        self.sub = aimet_modules.Subtract()
        self.mul_ri = aimet_modules.Multiply()
        self.mul_ir = aimet_modules.Multiply()
        self.add = aimet_modules.Add()
        self.stack = aimet_modules.Concat(3)

    def forward(self, x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
        x = self.apply_rope_single(x, rope_vals)
        return x

    def apply_rope_single(self, x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
        '''
        Based on FacebookResearch's llama, provided by Carl
        '''
        rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
        rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

        x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
        x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

        x_prod_real = self.sub(self.mul_rr(x_real,rope_real), self.mul_ii(x_im, rope_im))
        x_prod_im = self.add(self.mul_ri(x_real,rope_im), self.mul_ir(x_im,rope_real))

        x = self.stack(x_prod_real,x_prod_im).view(*x.shape)
        return x


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.head_dim, bias=config.attention_bias) for _ in range(self.num_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.head_dim, bias=config.attention_bias) for _ in range(self.num_key_value_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.head_dim, bias=config.attention_bias) for _ in range(self.num_key_value_heads)])
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.q_rope = nn.ModuleList([RoPE() for _ in range(self.num_heads)])
        self.k_rope = nn.ModuleList([RoPE() for _ in range(self.num_key_value_heads)])
        self.k_cat = nn.ModuleList([aimet_modules.Concat(2) for _ in range(self.num_key_value_heads)])
        self.v_cat = nn.ModuleList([aimet_modules.Concat(2) for _ in range(self.num_key_value_heads)])
        self.mm_qk = nn.ModuleList([aimet_modules.MatMul() for _ in range(self.num_heads)])
        self.div = nn.ModuleList([aimet_modules.Divide() for _ in range(self.num_heads)])
        self.mask_add = nn.ModuleList([aimet_modules.Add() for _ in range(self.num_heads)])
        self.sm = nn.ModuleList([nn.Softmax(dim=-1) for _ in range(self.num_heads)])
        self.mm_qkv = nn.ModuleList([aimet_modules.MatMul() for _ in range(self.num_heads)])

    def repeat_kv(self, hidden_states, n_rep: int):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        if n_rep == 1:
            return hidden_states
        if isinstance(hidden_states, list):
            return [state for state in hidden_states for _ in range(n_rep)]
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def apply_rope_single(self, x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
        '''
        Based on FacebookResearch's llama, provided by Carl
        '''
        rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
        rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

        x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
        x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

        x_prod_real = x_real*rope_real - x_im * rope_im
        x_prod_im = x_real*rope_im + x_im*rope_real

        x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[Tuple[torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = [q_proj(hidden_states).view(bsz, q_len, 1, self.head_dim).transpose(1, 2) for q_proj in self.q_proj]
        key_states = [k_proj(hidden_states).view(bsz, q_len, 1, self.head_dim).transpose(1, 2) for k_proj in self.k_proj]
        value_states = [v_proj(hidden_states).view(bsz, q_len, 1, self.head_dim).transpose(1, 2) for v_proj in self.v_proj]

        rope_embedding = position_ids
        query_states = [rope(q, rope_embedding) for rope, q in zip(self.q_rope, query_states)]
        key_states = [rope(k, rope_embedding) for rope, k in zip(self.k_rope, key_states)]

        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_key = [past_key[:, i:i+1, :, :] for i in self.num_key_value_heads]
            past_value = [past_value[:, i:i+1, :, :] for i in self.num_key_value_heads]
            key_states = [cat(pk, k) for cat, pk, k in zip(self.k_cat, past_key, key_states)]
            value_states = [cat(pv, v) for cat, pv, v in zip(self.v_cat, past_value, value_states)]

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = [div(mm(q, k.transpose(2, 3)), math.sqrt(self.head_dim))
                        for mm, div, q, k in zip(self.mm_qk, self.div, query_states, key_states)]

        if attention_mask is not None:
            attn_weights = [self.mask_add[i](attn_weights[i], attention_mask)
                            for i in range(len(attn_weights)//2)] + \
                           [self.mask_add[i](attention_mask, attn_weights[i])
                            for i in range(len(attn_weights)//2, len(attn_weights))]

        # upcast attention to fp32
        attn_weights = [sm(aw).to(query_states[0].dtype) for sm, aw in zip(self.sm, attn_weights)]
        attn_output = [mm(aw, v) for mm, aw, v in zip(self.mm_qkv, attn_weights, value_states)]

        if attn_output[0].size() != (bsz, 1, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output[0]` should be of size {(bsz, 1, q_len, self.head_dim)}, but is"
                f" {attn_output[0].size()}"
            )

        attn_output = torch.cat(attn_output, dim=1)

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


