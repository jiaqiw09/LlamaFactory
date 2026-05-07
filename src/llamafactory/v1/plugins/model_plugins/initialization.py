# Copyright 2025 the LlamaFactory team.
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
from typing import Literal

import torch

from ...accelerator.helper import DeviceType
from ...accelerator.interface import DistributedInterface
from ...config.arg_utils import StrictConfigMixin
from ...utils.plugin import BasePlugin


@dataclass
class InitConfig(StrictConfigMixin):
    """Init-device strategy. Variants are kept in one dataclass until they
    actually grow divergent fields — splitting into per-variant classes ahead of
    that just adds maintenance cost."""

    name: Literal["init_on_meta", "init_on_rank0", "init_on_default"] = "init_on_default"


class InitPlugin(BasePlugin):
    def __call__(self) -> torch.device:
        return super().__call__()


# All three variants share the same config dataclass, hence the same ``config=InitConfig``.
@InitPlugin("init_on_meta", config=InitConfig).register()
def init_on_meta() -> torch.device:
    return torch.device(DeviceType.META.value)


@InitPlugin("init_on_rank0", config=InitConfig).register()
def init_on_rank0() -> torch.device:
    if DistributedInterface().get_rank() == 0:
        return torch.device(DeviceType.CPU.value)
    else:
        return torch.device(DeviceType.META.value)


@InitPlugin("init_on_default", config=InitConfig).register()
def init_on_default() -> torch.device:
    return DistributedInterface().current_device
