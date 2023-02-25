import json

# from dataset.utils import pre_caption
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# information types
INFO_TYPE_CATEGORIES = [
    "Request-GoodsServices",
    "Request-InformationWanted",
    "Request-SearchAndRescue",
    "CallToAction-Donations",
    "CallToAction-MovePeople",
    "CallToAction-Volunteer",
    "Report-CleanUp",
    "Report-EmergingThreats",
    "Report-Factoid",
    "Report-FirstPartyObservation",
    "Report-Hashtags",
    "Report-Location",
    "Report-MultimediaShare",
    "Report-News",
    "Report-NewSubEvent",
    "Report-Official",
    "Report-OriginalEvent",
    "Report-ServiceAvailable",
    "Report-ThirdPartyObservation",
    "Report-Weather",
    "Other-Advice",
    "Other-ContextualInformation",
    "Other-Discussion",
    "Other-Irrelevant",
    "Other-Sentiment",
]

# priority categroies
PRIORITY_CATEGORIES_DICT = {"Critical": 1.0,
                            "High": 0.75, "Medium": 0.5, "Low": 0.25}


def map_pri(scores: np.ndarray) -> List:
    label_list = []
    for score in scores.reshape(-1):
        if score > 0.75:
            label_list.append("Critical")
        elif score > 0.5:
            label_list.append("High")
        elif score > 0.25:
            label_list.append("Medium")
        else:
            label_list.append("Low")
    return np.array(label_list)


class TRECISDataset(Dataset):
    """
    "image": "<post_id>",  # "" means no image provided
    "sentence": "<text>",
    "label": "<info_type_1>,<info_type_2>,..."
    """

    def __init__(self, ann_file, transform, image_root, max_words=128, max_imgs=1):
        # ann 即 data.json
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.image_root = Path(image_root)
        self.max_words = max_words
        self.img_size = transform.transforms[0].size
        self.max_imgs = max_imgs
        self.img_embed_size = int((self.img_size[0] // 16) ** 2 + 1)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path_list = list(self.image_root.glob(f"{ann['post_id']}_*.jpg"))
        if len(image_path_list) == 0:
            # 如果没有图，则生成一张0图
            image = torch.zeros((3,) + self.img_size)
            image_attention_mask = torch.zeros((self.img_embed_size,))
        elif self.max_imgs == 1:
            image_path = image_path_list[0]
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            image_attention_mask = torch.ones((self.img_embed_size,))
        else:
            raise ValueError(
                "Only support max_imgs == 1 just now but {} is specified.".format(
                    self.max_imgs
                )
            )

        text = ann["text"]
        if "info_type" in ann:
            info_type_label = self.convert_label_to_multi_tag(
                ann["info_type"], INFO_TYPE_CATEGORIES
            )
            priority_score = np.array(
                [PRIORITY_CATEGORIES_DICT[ann["priority"]]], dtype="float32"
            )
        else:
            info_type_label = None
            priority_score = None

        target = {
            "info_type_cls": info_type_label,
            "priority_regression": priority_score,
        }

        return image, text, target, image_attention_mask

    def convert_label_to_multi_tag(self, label_str: str, label_name_list: List[str]):
        label_list = label_str.split(",")
        multi_tag = [
            1.0 if label in label_list else 0.0 for label in label_name_list]
        return np.array(multi_tag, dtype="float32")

    def convert_label_to_one_hot(self, label_str: str, label_name_list: List[str]):
        one_hot = [1.0 if label_str ==
                   label else 0.0 for label in label_name_list]
        return np.array(one_hot, dtype="float32")
