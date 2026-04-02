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
from ..utils.constants import IGNORE_INDEX
from ..utils.types import BatchInput, Tensor


class DPOTrainer(BaseTrainer):
    def __init__(self, args, model, ref_model, renderer, train_dataset, callbacks=None):
        self.ref_model = ref_model
        super().__init__(args, model, renderer, train_dataset, callbacks)

        dist_name = self.args.dist_config.name if self.args.dist_config is not None else None
        if dist_name in {"deepspeed", "fsdp2"}:
            raise NotImplementedError(
                "The current minimal DPO implementation only supports single-GPU/DDP-style runs. "
                "Reference-model runtime handling for deepspeed/fsdp2 is not implemented yet."
            )

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

    if training_args.dpo_ref_model is None:
        raise ValueError("`dpo_ref_model` must be provided for the current minimal DPO implementation.")

    ref_model_args = deepcopy(model_args)
    ref_model_args.model = training_args.dpo_ref_model
    # Keep the first version simple: use an explicit reference model instead of sharing PEFT state.
    ref_model_args.peft_config = None
    ref_model = ModelEngine(ref_model_args, is_train=False).model
    ref_model.eval()

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
