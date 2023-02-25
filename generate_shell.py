import argparse
import itertools
from dataclasses import dataclass

from tqdm.auto import tqdm

# """
# python finetuned_trecis.py \
#   --config ./trecis/tiny.yaml \
#   --output_dir output/trecis \
#   --checkpoint ./provided_ckpt/ALBEF.pth \
#   --device 'cuda:0' \
#   --only_text_encoder \
#   --use_info_type_cls \
#   --use_priority_regression
# """

FINETUNED_TEMPLATE = """# {description}
python finetuned_trecis.py \\
  --config ./trecis/{config_name}.yaml \\
  --output_dir output/trecis/{runing_name} \\
  --evaluate \\
  --save_eval_label \\
  --checkpoint {ckpt_path} \\
  --early_stop 3 \\{text_encoder_flag}{info_type_flag}{priority_regression_flag}
  --device 'cuda:0' > finetuning_log/{runing_name}

"""

CKPT_PATH_DICT = {"albef": "./provided_ckpt/ALBEF.pth", "bert-base-uncased": "''"}


def fill_template(
    desc,
    runing_name,
    config_name="tiny",
    ckpt="albef",
    only_text_encoder=False,
    use_info_type_cls=True,
    use_priority_regression=True,
):
    template_dict = {
        "description": desc,
        "runing_name": runing_name,
        "config_name": config_name,
        "ckpt_path": CKPT_PATH_DICT.get(ckpt, ""),
        "text_encoder_flag": "\n  --only_text_encoder \\" if only_text_encoder else "",
        "info_type_flag": "\n  --use_info_type_cls \\" if use_info_type_cls else "",
        "priority_regression_flag": "\n  --use_priority_regression \\"
        if use_priority_regression
        else "",
    }

    return FINETUNED_TEMPLATE.format(**template_dict)


@dataclass
class RunConfig:
    only_text_encoder: bool = False
    use_itc: bool = True
    use_pr: bool = True

    def build(self):
        if self.only_text_encoder:
            self.ckpt = "bert-base-uncased"
        else:
            self.ckpt = "albef"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="tiny", type=str)

    args = parser.parse_args()
    with open(f"abalation_{args.config_name}.shell", "w", encoding="utf8") as file:
        for runs in tqdm(
            itertools.product([False, True], [False, True], [False, True])
        ):
            run_config = RunConfig(*runs)
            run_config.build()

            mtl = []
            if run_config.use_itc:
                mtl.append("itc")
            if run_config.use_pr:
                mtl.append("pr")
            if len(mtl) == 0:
                continue

            runing_name = (
                f"{args.config_name}_"
                + f"ONLYTEXT_{run_config.only_text_encoder}_"
                + f"CKPT_{run_config.ckpt.split('-')[0]}_"
                + f"TASK_{'_'.join(mtl)}"
            )
            desc = (
                f"{args.config_name}" + "/only_text_encoder"
                if run_config.only_text_encoder
                else "" + f"  ckpt({run_config.ckpt})  task({mtl})"
            )
            script = fill_template(
                desc,
                runing_name,
                args.config_name,
                ckpt=run_config.ckpt,
                only_text_encoder=run_config.only_text_encoder,
                use_info_type_cls=run_config.use_itc,
                use_priority_regression=run_config.use_pr,
            )

            file.write(script + "\n")
