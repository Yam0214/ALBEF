import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_trecis import AlbefForMultiplyChoice
from models.base_model import TextEncoderOnlyForMultiplyChoice
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from dataset.trecis_dataset import INFO_TYPE_CATEGORIES
from scheduler import create_scheduler
from optim import create_optimizer


def train(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    config,
):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "loss", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (images, text, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in targets.items()
        }

        text_inputs = tokenizer(text, padding="longest",
                                return_tensors="pt").to(device)

        if epoch > 0 or not config["warm_up"]:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"] * min(1, i / len(data_loader))

        loss = model(images,
                     text_inputs,
                     targets=targets,
                     train=True,
                     alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.4f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config, split=False):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = "Evaluation:"
    print_freq = 50

    pred_class_collector = {}
    targets_collector = {}

    for images, text, targets in metric_logger.log_every(
            data_loader, print_freq, header):

        images = images.to(device, non_blocking=True)
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in targets.items()
        }

        text_inputs = tokenizer(text, padding="longest",
                                return_tensors="pt").to(device)

        prediction = model(images, text_inputs, targets=targets, train=False)

        for task_key, task_pred in prediction.items():
            if task_key.split("_")[-1] == "score":
                # skip priority score
                continue
            pred_class = (task_pred > 0.5).float()
            label_class = targets[task_key]
            # 多标签分类损失
            accuracy = (label_class == pred_class).sum().item() / np.prod(
                label_class.shape)

            # 收集每个info类别的标签和预测
            pred_class_collector.setdefault(task_key, [])
            pred_class_collector[task_key].extend(
                pred_class.cpu().numpy().tolist())
            targets_collector.setdefault(task_key, [])
            targets_collector[task_key].extend(
                label_class.cpu().numpy().tolist())
            # 记录 micro_acc
            metric_logger.meters[f"{task_key}_acc"].update(accuracy,
                                                           n=images.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    for task_key in pred_class_collector.keys():
        targets_list = targets_collector[task_key]
        pred_class_list = pred_class_collector[task_key]
        # 对每个 info_type 分别计算准确率、precision、recall、macro-f1
        f1_score, precision, recall, support = precision_recall_fscore_support(
            targets_list, pred_class_list, average=None)
        print(f"{task_key:=^80}")
        for info_type, f1, p, r, num in zip(INFO_TYPE_CATEGORIES, f1_score,
                                            precision, recall, support):
            print(f"{info_type:>30}\tf1:{f1:<.2%}\t"
                  f"precision:{p:<.2%}\trecall:{r:<.2%}\tsupport:{num:}")
        print(
            f"{'macro':>30}\tf1:{f1_score.mean():<.2%}\t"
            f"precision:{precision.mean():<.2%}\trecall:{recall.mean():<.2%}")
        print()
    return {
        k: "{:.4f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Dataset
    print("Creating dataset")
    datasets = create_dataset("trecis", config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks,
                                  global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        datasets,
        samplers,
        batch_size=[config["batch_size_train"]] +
        [config["batch_size_test"]] * 2,
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    # Model
    print("Creating model")
    if args.only_text_encoder:
        model = TextEncoderOnlyForMultiplyChoice(tokenizer=tokenizer,
                                                 config=config)
    else:
        model = AlbefForMultiplyChoice(tokenizer=tokenizer, config=config)

    # load ALBEF pretraining checkpoint if model contain vison encoder
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]

        if not args.only_text_encoder:
            # reshape positional embedding to accomodate
            # for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], model.visual_encoder)
            state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped

        if not args.evaluate:
            if config["distill"]:
                if not args.only_text_encoder:
                    m_pos_embed_reshaped = interpolate_pos_embed(
                        state_dict["visual_encoder_m.pos_embed"],
                        model.visual_encoder_m)
                    state_dict[
                        "visual_encoder_m.pos_embed"] = m_pos_embed_reshaped

            for key in list(state_dict.keys()):
                if "bert" in key:
                    new_key = key.replace("bert.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print("load checkpoint from %s" % args.checkpoint)
        print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(
                model,
                train_loader,
                optimizer,
                tokenizer,
                epoch,
                warmup_steps,
                device,
                lr_scheduler,
                config,
            )

        val_stats = evaluate(model, val_loader, tokenizer, device, config)
        # no label in test_data
        # test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():
            if args.evaluate:
                log_stats = {
                    **{f"val_{k}": v
                       for k, v in val_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                    "epoch": epoch,
                }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {
                    **{f"train_{k}": v
                       for k, v in train_stats.items()},
                    **{f"val_{k}": v
                       for k, v in val_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                    "epoch": epoch,
                }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(val_stats["info_type_acc"]) > best:
                    save_obj = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "config": config,
                        "epoch": epoch,
                    }
                    torch.save(
                        save_obj,
                        os.path.join(args.output_dir, "checkpoint_best.pth"))
                    best = float(val_stats["info_type_acc"])
                    best_epoch = epoch

        if args.evaluate:
            break
        lr_scheduler.step(epoch + warmup_steps + 1)
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

    print("End finetuning for `class {}` with '{}' pretraining checkpoint".
          format(
              model.__class__.__name__,
              args.checkpoint if args.checkpoint else args.text_encoder,
          ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./trecis/trecis.yaml")
    parser.add_argument("--output_dir", default="output/trecis")
    parser.add_argument("--checkpoint",
                        default="",
                        help="path to pretrained checkpoint")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size",
                        default=1,
                        type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist_url",
                        default="env://",
                        help="url used to set up distributed training")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument(
        "--only_text_encoder",
        action="store_true",
        help="set this param to load a text encoder for abalation experiment",
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    config["text_encoder"] = args.text_encoder
    config["multi_label_nums"] = len(INFO_TYPE_CATEGORIES)
    print("{:*^50}".format("load config"))
    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
