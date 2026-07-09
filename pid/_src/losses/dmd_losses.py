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

"""
DMD (Distribution Matching Distillation) loss functions.

Reference: FastGen paper for DMD/DMD2 distillation methodology.
"""

import torch
import torch.nn.functional as F


def variational_score_distillation_loss(
    gen_data: torch.Tensor,
    teacher_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
) -> torch.Tensor:
    """
    VSD (Variational Score Distillation) loss in x0 space.

    This loss distills knowledge from a teacher model to a student model by using
    a fake score network to estimate the student's distribution. The gradient is
    computed as the difference between fake_score and teacher predictions, weighted
    by an adaptive factor.

    Args:
        gen_data: Student output (x0 prediction), shape (B, C, T, H, W)
        teacher_x0: Teacher's x0 prediction (detached), shape (B, C, T, H, W)
        fake_score_x0: Fake score's x0 prediction (detached), shape (B, C, T, H, W)

    Returns:
        VSD loss scalar
    """
    # Compute dimensions for mean reduction (all except batch)
    dims = tuple(range(1, teacher_x0.ndim))  # (1, 2, 3, 4) for 5D tensor

    # stop gradient. sg(x - grad)
    with torch.no_grad():
        # Compute adaptive weight and VSD gradient in float64 for numerical stability
        # (Cosmos Self-Forcing alignment: prevents NaN from small denominators)
        original_dtype = gen_data.dtype
        gen_f64 = gen_data.double()
        teacher_f64 = teacher_x0.double()
        fake_score_f64 = fake_score_x0.double()

        diff_abs_mean = (gen_f64 - teacher_f64).abs().mean(dim=dims, keepdim=True)
        w = 1.0 / (diff_abs_mean + 1e-6)

        # VSD gradient: direction from fake_score to teacher
        vsd_grad = (fake_score_f64 - teacher_f64) * w
        vsd_grad = torch.nan_to_num(vsd_grad, nan=0.0, posinf=0.0, neginf=0.0)

        # Pseudo target for gradient computation
        # grad(gen_data) = gen_data - pseudo_target
        pseudo_target = (gen_f64 - vsd_grad).detach()

    # MSE loss in float64 for numerical stability (matching reference dmd.py)
    loss = 0.5 * F.mse_loss(gen_data.double(), pseudo_target, reduction="mean")
    return loss


def denoising_score_matching_loss_flow(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    """
    DSM (Denoising Score Matching) loss for flow matching (velocity prediction).

    This is the standard MSE loss between predicted and target velocity fields,
    used to train the fake score network to match the student's distribution.

    Args:
        pred_velocity: Network's velocity prediction, shape (B, C, T, H, W)
        target_velocity: True velocity (epsilon - x0), shape (B, C, T, H, W)

    Returns:
        MSE loss scalar
    """
    return F.mse_loss(pred_velocity, target_velocity, reduction="mean")


def gan_loss_generator(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator GAN loss: encourage discriminator to classify fake as real.

    Non-saturating logistic GAN loss (softplus form).

    Args:
        fake_logits: Discriminator output on fake (student-generated) features, shape [B, num_heads]

    Returns:
        Scalar loss
    """
    return F.softplus(-fake_logits).mean()


def gan_loss_discriminator(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator GAN loss: classify real as real and fake as fake.

    Non-saturating logistic GAN loss (softplus form).

    Args:
        real_logits: Discriminator output on real (GT-perturbed) features, shape [B, num_heads]
        fake_logits: Discriminator output on fake (student-generated) features, shape [B, num_heads]

    Returns:
        Scalar loss
    """
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
