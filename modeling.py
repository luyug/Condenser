# Copyright 2021 Condenser Author All rights reserved.
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

import os
import warnings

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer

from arguments import DataTrainingArguments, ModelArguments, CoCondenserPreTrainingArguments
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)


class CondenserForPretraining(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = bert
        self.c_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def forward(self, model_input, labels):
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]

        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        loss = self.mlm_loss(hiddens, labels)
        if self.model_args.late_mlm:
            loss += lm_out.loss

        return loss


    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss


    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, train_args)
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments,
    ):
        hf_model = AutoModelForMaskedLM.from_config(config)
        model = cls(hf_model, model_args, data_args, train_args)

        return model

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('lm')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

class RobertaCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            roberta: RobertaModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = roberta
        self.c_head = nn.ModuleList(
            [RobertaLayer(roberta.config) for _ in range(model_args.n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)
        # self.mlm_head = BertOnlyMLMHead(bert.config)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.lm_head(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss

class CoCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            bert: BertModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: CoCondenserPreTrainingArguments
    ):
        super(CoCondenserForPretraining, self).__init__(bert, model_args, data_args, train_args)

        effective_bsz = train_args.per_device_train_batch_size * self._world_size() * 2
        target = torch.arange(effective_bsz, dtype=torch.long).view(-1, 2).flip([1]).flatten().contiguous()

        self.register_buffer(
            'co_target', target
        )

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def forward(self, model_input, labels, grad_cache: Tensor = None, chunk_offset: int = None):
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        if self.train_args.local_rank > -1 and grad_cache is None:
            co_cls_hiddens = self.gather_tensors(cls_hiddens.squeeze().contiguous())[0]
        else:
            co_cls_hiddens = cls_hiddens.squeeze()

        skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]
        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        loss = self.mlm_loss(hiddens, labels)
        if self.model_args.late_mlm:
            loss += lm_out.loss

        if grad_cache is None:
            co_loss = self.compute_contrastive_loss(co_cls_hiddens)
            return loss + co_loss
        else:
            loss = loss * (float(hiddens.size(0)) / self.train_args.per_device_train_batch_size)
            cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
            surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())
            return loss, surrogate

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    def compute_contrastive_loss(self, co_cls_hiddens):
        similarities = torch.matmul(co_cls_hiddens, co_cls_hiddens.transpose(0, 1))
        similarities.fill_diagonal_(float('-inf'))
        co_loss = F.cross_entropy(similarities, self.co_target) * self._world_size()
        return co_loss