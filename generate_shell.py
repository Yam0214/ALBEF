import itertools
import argparse
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

FINETUNED_TEMPLATE = """
# {description}
python finetuned_trecis.py \\
  --config ./trecis/{config_name}.yaml \\
  --output_dir output/trecis/{runing_name} \\
  --checkpoint {ckpt_path} \\
  --save_eval_label \\{text_encoder_flag}{info_type_flag}{priority_regression_flag}
  --device 'cuda:0' > finetuning_log/{runing_name}
"""

CKPT_PATH_DICT = {
    "albef": "./provided_ckpt/ALBEF.pth",
    "bert-base-uncased": "''"
}


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
        "priority_regression_flag": "\n  --use_priority_regression \\" if use_priority_regression else "",
    }

    return FINETUNED_TEMPLATE.format(**template_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="tiny", type=str)

    args = parser.parse_args()
    ckpt_choices = ["albef", "bert-base-uncased"]
    with open(f"abalation_{args.config_name}.shell", "w", encoding="utf8") as file:

        for runs in tqdm(
                itertools.product(ckpt_choices, [False, True], [False, True],
                                  [False, True])):
            mtl = []
            if runs[2]:
                mtl.append("itc")
            if runs[3]:
                mtl.append("pr")
            if len(mtl) == 0:
                continue

            runing_name = f"{args.config_name}_" \
                        + f"ONLYTEXT_{runs[1]}_" \
                        + f"CKPT_{runs[0].split('-')[0]}_" \
                        + f"TASK_{'_'.join(mtl)}"
            desc = f"{args.config_name}{'/only_text_encoder' if runs[1] else ''}  ckpt({runs[0]})  task({mtl})"
            script = fill_template(desc, runing_name, args.config_name, *runs)

            file.write(script + "\n")
