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


import torch

from ..accelerator.interface import DistributedInterface
from ..config import InputArgument, TrainingArguments, get_args
from ..core.base_trainer import BaseTrainer
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..plugins.model_plugins.chunk_loss import enable_sft_chunk_loss, sft_chunk_loss_context
from ..utils.types import BatchInput, HFModel, Tensor, TorchDataset


class SFTTrainer(BaseTrainer):
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        renderer,
        train_dataset: TorchDataset,
        callbacks=None,
    ) -> None:
        if args.chunk_loss_size is not None:
            cp_size = args.dist_config.get("cp_size", 1) if args.dist_config is not None else 1
            dist_name = args.dist_config.name if args.dist_config is not None else None
            if cp_size > 1:
                raise NotImplementedError("SFT chunk loss does not support context parallelism yet.")
            if dist_name not in (None, "fsdp2"):
                raise NotImplementedError("SFT chunk loss currently supports only single-process/DDP or FSDP2.")
            enable_sft_chunk_loss(model)

        super().__init__(args, model, renderer, train_dataset, callbacks)

    def compute_loss(self, batch: BatchInput) -> Tensor:
        if self.args.chunk_loss_size is not None:
            labels = batch["labels"].to(self.device, non_blocking=True)
            loss_weights = batch["loss_weights"].to(self.device, non_blocking=True)
            model_inputs = {
                key: value.to(self.device, non_blocking=True)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor) and key not in ("labels", "loss_weights")
            }
            with sft_chunk_loss_context(labels, loss_weights, self.args.chunk_loss_size):
                outputs = self.model(**model_inputs, use_cache=False, return_dict=True)

            loss = outputs.logits
            if loss.ndim != 0:
                raise RuntimeError("SFT chunk loss expected lm_head to return a scalar loss.")
            return loss

        shift_loss_weights = batch["loss_weights"].to(self.device, non_blocking=True)[..., 1:]
        log_probs = self.compute_log_probs(self.model, batch)
        loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)
        return loss


def run_sft(args: InputArgument = None):
    model_args, data_args, training_args, _ = get_args(args)
    DistributedInterface(training_args.dist_config)
    train_dataset = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    trainer = SFTTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.trainers.sft_trainer --model Qwen/Qwen3-0.6B --train_dataset data/v1_sft_demo.yaml
    """
    run_sft()
