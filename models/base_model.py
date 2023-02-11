from transformers import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Dict


class TextEncoderOnlyForMultiplyChoice(nn.Module):
    def __init__(self, tokenizer, config: Dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.distill = config["distill"]
        bert_config = BertConfig.from_json_file(config["bert_config"])

        self.text_encoder = BertModel.from_pretrained(config["text_encoder"],
                                                      config=bert_config,
                                                      add_pooling_layer=False)
        self.cls_head = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size,
                self.text_encoder.config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size,
                      config["multi_label_nums"]),
        )

        if self.distill:
            self.text_encoder_m = BertModel.from_pretrained(
                config["text_encoder"],
                config=bert_config,
                add_pooling_layer=False)
            self.cls_head_m = nn.Sequential(
                nn.Linear(
                    self.text_encoder.config.hidden_size,
                    self.text_encoder.config.hidden_size,
                ),
                nn.ReLU(),
                nn.Linear(self.text_encoder.config.hidden_size,
                          config["multi_label_nums"]),
            )

            self.model_pairs = [
                [self.text_encoder, self.text_encoder_m],
                [self.cls_head, self.cls_head_m],
            ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self,
                image: torch.Tensor,
                text: torch.Tensor,
                targets: Dict[str, torch.Tensor],
                alpha: Optional[float] = 0,
                train: Optional[bool] = True):

        # image is specificed but isn't used
        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   return_dict=True)
        prediction = self.cls_head(output.last_hidden_state[:, 0, :])

        if train:
            # 训练则返回loss，否则返回prediction
            if self.distill:
                # 如果动态蒸馏
                with torch.no_grad():
                    self._momentum_update()
                    output_m = self.text_encoder_m(
                        text.input_ids,
                        attention_mask=text.attention_mask,
                        return_dict=True,
                    )
                    prediction_m = self.cls_head_m(
                        output_m.last_hidden_state[:, 0, :])

                loss = ((1 - alpha) * F.binary_cross_entropy_with_logits(
                    prediction, targets["info_type"]) - alpha * torch.sum(
                        F.log_softmax(prediction, dim=1) *
                        F.softmax(prediction_m, dim=1),
                        dim=1,
                    ).mean())
            else:
                loss = F.binary_cross_entropy_with_logits(
                    prediction, targets["info_type"])
            return loss
        # valuatation or inference
        return {"info_type": prediction}

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum)
