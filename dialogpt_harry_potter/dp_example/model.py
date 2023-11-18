import re
import json
import math
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from overrides import overrides
from transformers import AutoModelWithLMHead, AutoConfig, AutoTokenizer

from deeppavlov import build_model
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.models.torch_model import TorchModel

logger = getLogger(__name__)


@register('dialogue_generator')
class DialogueGenerator(TorchModel):
    def __init__(self,
                 pretrained_transformer: str,
                 config_file: Optional[str] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: Optional[dict] = None,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-06,
                 **kwargs) -> None:

        if not optimizer_parameters:
            optimizer_parameters = {"lr": 0.01,
                                    "weight_decay": 0.01,
                                    "betas": (0.9, 0.999),
                                    "eps": 1e-6}
        self.clip_norm = clip_norm
        self.pretrained_transformer = pretrained_transformer
        self.config_file = config_file
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_transformer, do_lower_case=False)
        self.tokenizer.pad_token_id = 0

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         **kwargs)

    def train_on_batch(self, input_ids_batch, attention_mask_batch, target_ids_batch) -> Dict:
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        target_ids_batch = torch.LongTensor(target_ids_batch).to(self.device)
        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': target_ids_batch
        }

        self.optimizer.zero_grad()
        loss = self.model(**input_).loss
        if self.is_data_parallel:
            loss = loss.mean()
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    @property
    def is_data_parallel(self) -> bool:
        return isinstance(self.model, torch.nn.DataParallel)

    def __call__(self, input_ids_batch, attention_mask_batch, target_ids=None):
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        
        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
        }
        
        with torch.no_grad():
            if self.is_data_parallel:
                answer_ids_batch = self.model.module.generate(**input_, max_length=512)
            else:
                answer_ids_batch = self.model.generate(**input_, max_length=512)
        answers_batch = []
        for answer_ids in answer_ids_batch:
            answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
            answers_batch.append(answer)

        if target_ids is not None:
            input_['labels'] = target_ids
            loss = self.model(**input_).loss
            if self.is_data_parallel:
                loss = loss.mean()
            ppl = torch.exp(loss)
            ppl = [ppl.detach().cpu().numpy().tolist()]
            return answers_batch, ppl
        else:
            return answers_batch

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_transformer:
            logger.info(f"From pretrained {self.pretrained_transformer}.")
            config = AutoConfig.from_pretrained(self.pretrained_transformer,
                                                output_attentions=False,
                                                output_hidden_states=False)

            self.model = AutoModelWithLMHead.from_pretrained(self.pretrained_transformer, config=config)

        elif self.config_file and Path(self.config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.config_file)))
            self.model = AutoModelWithLMHead(config=self.config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            logger.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                logger.info(f"Load path {weights_path} exists.")
                logger.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                logger.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                model_state = checkpoint["model_state_dict"]
                optimizer_state = checkpoint["optimizer_state_dict"]

                # load a multi-gpu model on a single device
                if not self.is_data_parallel and "module." in list(model_state.keys())[0]:
                    tmp_model_state = {}
                    for key, value in model_state.items():
                        tmp_model_state[re.sub("module.", "", key)] = value
                    model_state = tmp_model_state

                strict_load_flag = bool([key for key in checkpoint["model_state_dict"].keys()
                                         if key.endswith("embeddings.position_ids")])
                self.model.load_state_dict(model_state, strict=strict_load_flag)
                self.optimizer.load_state_dict(optimizer_state)
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                logger.info(f"Init from scratch. Load path {weights_path} does not exist.")
