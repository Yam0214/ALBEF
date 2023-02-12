from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Dict
from .task_head import MultiChoiceHead, LogisticRegressionHead


class AlbefForMultiTask(nn.Module):
    def __init__(
        self,
        tokenizer=None,
        config: Optional[Dict] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config["distill"]
        self.use_info_type_cls = config["use_info_type_cls"]
        self.use_priority_regression = config["use_priority_regression"]

        self.visual_encoder = VisionTransformer(
            img_size=config["image_res"],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        bert_config = BertConfig.from_json_file(config["bert_config"])
        self.text_encoder = BertModel.from_pretrained(config["text_encoder"],
                                                      config=bert_config,
                                                      add_pooling_layer=False)

        # multi-task cls head
        self.task_head_dict = {}
        if self.use_info_type_cls:
            self.info_cls_head = MultiChoiceHead(bert_config,
                                                 config["info_label_nums"])
            self.task_head_dict.update({"info_type_cls": self.info_cls_head})
        if self.use_priority_regression:
            self.priority_regression_head = LogisticRegressionHead(bert_config)
            self.task_head_dict.update(
                {"priority_regression": self.priority_regression_head})

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config["image_res"],
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            self.text_encoder_m = BertModel.from_pretrained(
                config["text_encoder"],
                config=bert_config,
                add_pooling_layer=False)

            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
            ]

            # multi task
            self.task_head_dict_m = {}
            if self.use_info_type_cls:
                self.info_cls_head_m = MultiChoiceHead(
                    bert_config, config["info_label_nums"])
                self.model_pairs.append(
                    [self.info_cls_head, self.info_cls_head_m])
                self.task_head_dict_m.update(
                    {"info_cls_type": self.info_cls_head_m})
            if self.use_priority_regression:
                self.priority_regression_head_m = LogisticRegressionHead(
                    bert_config)
                self.model_pairs.append([
                    self.priority_regression_head,
                    self.priority_regression_head_m
                ])
                self.task_head_dict_m.update(
                    {"priority_regression": self.priority_regression_head_m})

            self.copy_params()
            self.momentum = 0.995

    def forward(self,
                image: torch.Tensor,
                text: torch.Tensor,
                targets: Dict[str, torch.Tensor],
                alpha: Optional[float] = 0,
                train: Optional[bool] = True):
        # VisonTransformer, ViT -l 12
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        mtl_outputs_dict = {
            task: head(output.last_hidden_state[:, 0, :], labels=targets[task])
            for task, head in self.task_head_dict.items()
        }
        # 训练则返回loss，否则返回prediction
        if train:
            task_loss = {
                task: task_output.loss
                for task, task_output in mtl_outputs_dict.items()
            }
            if self.distill:
                # 如果动态蒸馏
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    output_m = self.text_encoder_m(
                        text.input_ids,
                        attention_mask=text.attention_mask,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    mtl_outputs_dict_m = {
                        task: head(output_m.last_hidden_state[:, 0, :])
                        for task, head in self.task_head_dict_m.items()
                    }

                distill_loss = {}
                for task in mtl_outputs_dict.keys():
                    logits = mtl_outputs_dict[task].logits
                    logits_m = mtl_outputs_dict_m[task].logits
                    distill_loss.update({
                        task:
                        -torch.sum(
                            F.log_softmax(logits, dim=1) *
                            F.softmax(logits_m, dim=1),
                            dim=1,
                        ).mean()
                    })

                # 合并 loss
                # loss = (1-alpha) * task_loss + alpha * distill_loss
                loss_dict = {}
                for task in mtl_outputs_dict.keys():
                    loss = (1 - alpha
                            ) * task_loss[task] + alpha * distill_loss[task]
                    loss_dict.update({task: loss})
            else:
                loss_dict = task_loss

            total_loss = torch.stack(list(loss_dict.values())).sum()
            return total_loss, loss_dict
        # valuatation or inference
        return {
            task: task_output.logits
            for task, task_output in mtl_outputs_dict.items()
        }

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
