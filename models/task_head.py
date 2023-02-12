from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (MultipleChoiceModelOutput,
                                           ModelOutput)


@dataclass
class RegressionModelOurput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class MultiChoiceHead(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.multi_choice_head = nn.Sequential(
            nn.Linear(
                config.hidden_size,
                config.hidden_size,
            ), nn.ReLU(), nn.Linear(config.hidden_size, num_labels))

    def forward(self,
                inputs: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        logits = self.multi_choice_head(inputs)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss.to(torch.float32)
            
        return MultipleChoiceModelOutput(loss=loss, logits=logits)


class LogisticRegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.regression_head = nn.Linear(config.hidden_size, 1)

    def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor]):
        logits = self.regression_head(inputs)
        loss = None
        if labels is not None:
            loss = F.mse_loss(logits, labels)
            loss = loss.to(torch.float32)

        return RegressionModelOurput(loss=loss, logits=logits)