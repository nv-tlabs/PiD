# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils.ema import EMAModelTracker, PowerEMATracker, ema_scope


@pytest.fixture
def linear_model_instance():
    beta = 0.9
    # Fix the seed for model initialization
    torch.manual_seed(0)
    model = ImaginaireModel()
    model.net = torch.nn.Linear(10, 2)
    model.ema = EMAModelTracker(model, beta)
    return model


@pytest.fixture
def linear_model_no_bias_instance():
    beta = 0.9
    # Fix the seed for model initialization
    torch.manual_seed(0)
    model = ImaginaireModel()
    model.net = torch.nn.Linear(10, 2, bias=False)
    model.ema = EMAModelTracker(model, beta)
    return model


@pytest.mark.L0
def test_val_error(linear_model_instance):
    model = linear_model_instance
    torch.manual_seed(0)
    x_train = torch.rand((100, 10))
    y_train = torch.rand(100).round().long()
    x_val = torch.rand((100, 10))
    y_val = torch.rand(100).round().long()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for _ in range(2):
        logits = model.net(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.ema.update_average(model)

    model.eval()
    logits = model.net(x_val)
    loss_orig = torch.nn.functional.cross_entropy(logits, y_val)
    print(f"Original loss: {loss_orig}")

    with ema_scope(model, True):
        logits = model.net(x_val)
        loss_ema = torch.nn.functional.cross_entropy(logits, y_val)
    print(f"EMA loss: {loss_ema}")
    assert loss_ema < loss_orig, "EMA loss was not lower"

    logits = model.net(x_val)
    loss_orig2 = torch.nn.functional.cross_entropy(logits, y_val)
    assert torch.allclose(loss_orig, loss_orig2), "Restored model was not the same as stored model"


@pytest.mark.L0
@pytest.mark.parametrize("ema_tracker", [EMAModelTracker, PowerEMATracker])
def test_ema_update(linear_model_no_bias_instance, ema_tracker):
    model = linear_model_no_bias_instance
    with torch.no_grad():
        model.net.weight.fill_(0.0)

    if ema_tracker == EMAModelTracker:
        model.ema = ema_tracker(model, beta=0.9)
    elif ema_tracker == PowerEMATracker:
        model.ema = ema_tracker(model, s=0.1)
    else:
        raise ValueError(f"Unknown EMA tracker: {ema_tracker}. Please use EMAModelTracker or PowerEMATracker.")

    # Check that the ema weights were initialized correctly.
    ema_weight = model.ema.state_dict()["net-weight"]
    assert torch.all(ema_weight == 0.0), "EMA weights were not initialized correctly"

    with torch.no_grad():
        model.net.weight.fill_(1.0)

    # Iteration is used to compute beta in power ema, but is not used in regular ema.
    model.ema.update_average(model, iteration=1)

    # Check that the ema weights were updated correctly.
    ema_weight = model.ema.state_dict()["net-weight"]
    assert torch.allclose(ema_weight, torch.full(size=(1,), fill_value=(1.0 - model.ema.beta))), (
        "EMA update was incorrect"
    )

    # Check that the regular model weights were not changed from the ema update.
    assert torch.all(model.net.weight == 1.0), "EMA update shouldn't have changed the regular model weights"

    # Check that the ema weights were copied back to the model correctly.
    model.ema.copy_to(model)
    assert torch.allclose(model.net.weight, torch.full(size=(1,), fill_value=(1.0 - model.ema.beta))), (
        "EMA weights were copied to the model incorrectly"
    )


@pytest.mark.L1
@pytest.mark.parametrize("ema_tracker", [EMAModelTracker, PowerEMATracker])
def test_ema_update_torch_compile(linear_model_no_bias_instance, ema_tracker):
    model = linear_model_no_bias_instance
    with torch.no_grad():
        model.net.weight.fill_(0.0)

    if ema_tracker == EMAModelTracker:
        model.ema = ema_tracker(model, beta=0.9, torch_compile_buffer_renaming=True)
    elif ema_tracker == PowerEMATracker:
        model.ema = ema_tracker(model, s=0.1, torch_compile_buffer_renaming=True)
    else:
        raise ValueError(f"Unknown EMA tracker: {ema_tracker}. Please use EMAModelTracker or PowerEMATracker.")

    # Compilation should take place after EMA was created
    model.net = torch.compile(model.net)

    # Check that the ema weights were initialized correctly.
    ema_weight = model.ema.state_dict()["net-weight"]
    assert torch.all(ema_weight == 0.0), "EMA weights were not initialized correctly"

    with torch.no_grad():
        model.net.weight.fill_(1.0)

    # Iteration is used to compute beta in power ema, but is not used in regular ema.
    model.ema.update_average(model, iteration=1)

    # Check that the ema weights were updated correctly.
    ema_weight = model.ema.state_dict()["net-weight"]
    assert torch.allclose(ema_weight, torch.full(size=(1,), fill_value=(1.0 - model.ema.beta))), (
        "EMA update was incorrect"
    )

    # Check that the regular model weights were not changed from the ema update.
    assert torch.all(model.net.weight == 1.0), "EMA update shouldn't have changed the regular model weights"

    # Check that the ema weights were copied back to the model correctly.
    model.ema.copy_to(model)
    assert torch.allclose(model.net.weight, torch.full(size=(1,), fill_value=(1.0 - model.ema.beta))), (
        "EMA weights were copied to the model incorrectly"
    )
