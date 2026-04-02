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

#todo dpo: implement the v1 DPO trainer entrypoint in this file.
#todo dpo: add run_dpo(), ref-model creation, concatenated forward, sequence log-prob reduction, and DPO loss.

from copy import deepcopy

import torch

from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..utils import logging
from ..utils.constants import IGNORE_INDEX
from ..utils.types import BatchInput, Tensor

logger = logging.get_logger(__name__)


def _log_npu_memory(stage: str) -> None:
    device = DistributedInterface().current_device
    if device.type != "npu" or not torch.npu.is_available():
        logger.info_rank0(f"[DPO memory] {stage}: current device is {device}, npu memory stats unavailable.")
        return

    allocated = torch.npu.memory_allocated(device) / 1024**3
    reserved = torch.npu.memory_reserved(device) / 1024**3
    max_allocated = torch.npu.max_memory_allocated(device) / 1024**3
    max_reserved = torch.npu.max_memory_reserved(device) / 1024**3
    logger.info_rank0(
        "[DPO memory] %s: allocated=%.2f GiB, reserved=%.2f GiB, max_allocated=%.2f GiB, max_reserved=%.2f GiB",
        stage,
        allocated,
        reserved,
        max_allocated,
        max_reserved,
    )

def _log_model_sharding_status(model: torch.nn.Module, stage: str) -> None:
    sample_name = None
    sample_param = None
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        if sample_param is None:
            sample_name = name
            sample_param = param

    if sample_param is None:
        logger.info_rank0("[DPO shard] %s: no parameters found on model.", stage)
        return

    import_error = None
    is_dtensor = False
    try:
        from torch.distributed._tensor import DTensor

        is_dtensor = isinstance(sample_param, DTensor)
    except Exception as exc:  # pragma: no cover - depends on torch runtime support
        import_error = repr(exc)
        is_dtensor = all(hasattr(sample_param, attr) for attr in ("device_mesh", "placements", "to_local"))

    if is_dtensor:
        local_shape = tuple(sample_param.to_local().shape) if hasattr(sample_param, "to_local") else None
        logger.info_rank0(
            "[DPO shard] %s: sharded=yes, sample_param=%s, total_params=%d, global_shape=%s, local_shape=%s, placements=%s, mesh=%s",
            stage,
            sample_name,
            total_params,
            tuple(sample_param.shape),
            local_shape,
            getattr(sample_param, "placements", None),
            getattr(sample_param, "device_mesh", None),
        )
    else:
        logger.info_rank0(
            "[DPO shard] %s: sharded=no, sample_param=%s, total_params=%d, param_type=%s%s",
            stage,
            sample_name,
            total_params,
            type(sample_param),
            f", dtensor_check_error={import_error}" if import_error is not None else "",
        )


class DPOTrainer(BaseTrainer):
    def __init__(self, args, model, ref_model, renderer, train_dataset, callbacks=None):
        self.ref_model = ref_model
        self._memory_debug_logged = False
        super().__init__(args, model, renderer, train_dataset, callbacks)
        self._prepare_ref_model()

    def _prepare_ref_model(self) -> None:
        dist_name = self.args.dist_config.name if self.args.dist_config is not None else None

        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        if dist_name == "fsdp2":
            from ..plugins.trainer_plugins.distributed.hub import DistributedPlugin

            logger.info_rank0("Preparing frozen DPO reference model with FSDP2 sharding.")
            self.ref_model = DistributedPlugin("fsdp2")(self.ref_model, self.args.dist_config)
        elif dist_name == "deepspeed":
            raise NotImplementedError(
                "DPO with a dedicated reference model is not implemented for DeepSpeed yet. "
                "The trainable policy model uses DeepSpeed runtime hooks, but the frozen reference model "
                "still needs a separate inference-time preparation path."
            )
        elif dist_name is None and DistributedInterface().get_world_size() > 1:
            logger.info_rank0(
                "DPO reference model remains fully replicated on each rank under DDP fallback. "
                "Use fsdp2 with init_on_meta to reduce duplicated reference-model memory."
            )

        self.ref_model.eval()
        _log_model_sharding_status(self.ref_model, "after ref model preparation")
        _log_npu_memory("after ref model preparation")


    def compute_loss(self, batch: BatchInput) -> Tensor:
        """Compute DPO loss for a batch of data."""

        # policy
        policy_logps = self.compute_segment_logps(self.model, batch)
        policy_chosen_logps = policy_logps["chosen_logps"]
        policy_rejected_logps = policy_logps["rejected_logps"]

        # ref-model
        with torch.no_grad():
            ref_logps = self.compute_segment_logps(self.ref_model, batch)
        ref_chosen_logps = ref_logps["chosen_logps"]
        ref_logps_rejected = ref_logps["rejected_logps"]

        # calculate log-ratio
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_logps_rejected

        # calculate DPO loss
        logits = self.args.dpo_beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -torch.nn.functional.logsigmoid(logits).mean()

        return loss

    def compute_segment_logps(
        self,
        model: torch.nn.Module,
        batch: BatchInput,
    ) -> dict[str, Tensor]:
        """Compute log probabilities for the chosen and rejected segments."""
        labels = batch["labels"].to(self.device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(self.device, non_blocking=True)

        model_inputs = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"labels", "loss_weights", "token_type_ids"}:
                continue
            model_inputs[key] = value.to(self.device, non_blocking=True)

        outputs = model(**model_inputs)
        logits = outputs.logits.float()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_token_type_ids = token_type_ids[..., 1:].contiguous()

        valid_mask = shift_labels != IGNORE_INDEX
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
        per_token_logps = torch.gather(
            shift_logits.log_softmax(dim=-1),
            dim=2,
            index=safe_labels.unsqueeze(2),
        ).squeeze(2)

        chosen_mask = (shift_token_type_ids == 1) & valid_mask
        rejected_mask = (shift_token_type_ids == 2) & valid_mask

        chosen_logps = (per_token_logps * chosen_mask).sum(dim=1)
        rejected_logps = (per_token_logps * rejected_mask).sum(dim=1)

        chosen_length = chosen_mask.sum(dim=-1)
        rejected_length = rejected_mask.sum(dim=-1)

        batch_size = chosen_logps.shape[0] // 2
        chosen_logps = chosen_logps[:batch_size]
        rejected_logps = rejected_logps[batch_size:]
        chosen_length = chosen_length[:batch_size]
        rejected_length = rejected_length[batch_size:]

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_length": chosen_length,
            "rejected_length": rejected_length,
        }


def run_dpo(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args.dist_config)
    train_dataset = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    _log_npu_memory("after policy ModelEngine init")
    if training_args.dpo_ref_model is None:
        raise ValueError("`dpo_ref_model` must be provided for the current minimal DPO implementation.")

    ref_model_args = deepcopy(model_args)
    ref_model_args.model = training_args.dpo_ref_model
    # Keep the first version simple: use an explicit reference model instead of sharing PEFT state.
    ref_model_args.peft_config = None
    ref_model = ModelEngine(ref_model_args, is_train=False).model
    ref_model.eval()
    _log_npu_memory("after ref_model ModelEngine init")
    for param in ref_model.parameters():
        param.requires_grad_(False)

    trainer = DPOTrainer(
        args=training_args,
        model=model_engine.model,
        ref_model=ref_model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.trainers.dpo_trainer \
        --model Qwen/Qwen3-0.6B \
        --template qwen3_nothink \
        --train_dataset data/v1_dpo_demo.yaml \
        --dpo_ref_model Qwen/Qwen3-0.6B
    """
    run_dpo()
