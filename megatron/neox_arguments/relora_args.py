# Copyright (c) 2023, EleutherAI and Vlad Lialin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

try:
    from .template import NeoXArgsTemplate
except ImportError:
    from template import NeoXArgsTemplate

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class NeoXArgsReLoRAConfig(NeoXArgsTemplate):
    """
    Args for ReLoRA config
    """

    relora: bool = False
    """boolean flag to enable ReLoRA"""

    relora_r: int = 128
    """r-parameter (rank of a single LoRA) for ReLoRA"""

    relora_alpha: float = 32
    """alpha (scaling factor for LoRA: W + 1/alpha * W_A @ W_B) for ReLoRA"""

    relora_dropout: float = 0.1
    """dropout probability for ReLoRA"""

    relora_trainable_scaling: bool = False
    """whether to make alpha trainable for ReLoRA. Not recommended for now, but might be fun to explore"""


@dataclass
class NeoXArgsReLoRATrainingConfig(NeoXArgsTemplate):
    relora_reset_frequency: int = 5000
    """How often (in upadte iterations) to perform .merge_and_reinit() on the model"""

    relora_reset_warmup_steps: int = 100
    """Short warmup after .merge_and_reinit() to avoid instability. Recommended: 10-100"""

    relora_optimizer_reset_amount: float = 0.9
    """How much to reset (magnitude prune) the optimizer after .merge_and_reinit(). Recommended: 0.9-0.99"""
