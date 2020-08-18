# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from rvai.base.cell import Parameters
from rvai.types import Float, Integer, String, Boolean, Enum

# Parameters
@dataclass
class DetrParameters(Parameters):
    # learning
    lr: Float = Parameters.field(
        default=Float(1e-4),
        name="Learning Rate", description="Learning rate (for the head)."
    )
    lr_backbone: Float = Parameters.field(
        default=Float(1e-5),
        name="Learning Rate Backbone", description="Learning rate (for the backbone)."
    )
    batch_size: Integer = Parameters.field(
        default=Float(2),
        name="Learning Rate Backbone", description="Learning rate (for the backbone)."
    )
    weight_decay: Float = Parameters.field(
        default=Float(1e-4),
        name="Weight Decay", description="Weight decay"
    )
    epochs: Integer = Parameters.field(
        default=Integer(300),
        name="Epochs", description="Epochs"
    )
    lr_drop: Integer = Parameters.field(
        default=Integer(200),
        name="Learning Rate Drop", description="Learning Rate Drop"
    )
    clip_max_norm: Float = Parameters.field(
        default=Float(0.1),
        name="Clip Max Norm", description="Gradient clipping max norm"
    )
    frozen_weights: Optional[String] = Parameters.field(
        default=None,
        name="Frozen weights", description="Path to the pretrained model. If set, only the mask head will be trained"
    )
    # model
    backbone: String = Parameters.field(
        default=String("resnet50"),
        name="Backbone", description="Name of the convolutional backbone to us"
    )
    dilation: Boolean = Parameters.field(
        default=Boolean(False),
        name="Dilation", description="If true, we replace stride with dilation in the last convolutional block (DC5)"
    )
    position_embedding: Enum[String] = Parameters.field(
        default=Enum[String](
            "sine", "learned", selected="sine"
        ),
        name="Poistion Embedding",
        description="Type of positional embedding to use on top of the image features"
    )
    # Transformer
    enc_layers: Integer = Parameters.field(
        default=Integer(6),
        name="Encoding layers", description="Number of encoding layers in the transformer"
    )
    dec_layers: Integer = Parameters.field(
        default=Integer(6),
        name="Decoding layers", description="Number of decoding layers in the transformer"
    )
    dim_feedforward: Integer = Parameters.field(
        default=Integer(2048),
        name="Dimensions feedforward", description="Intermediate size of the feedforward layers in the transformer blocks"
    )
    hidden_dim: Integer = Parameters.field(
        default=Integer(256),
        name="Hidden Dimension", description="Size of the embeddings (dimension of the transformer)"
    )
    dropout: Float = Parameters.field(
        default=Float(0.1),
        name="Dropout", description="Dropout applied in the transformer"
    )
    nheads: Integer = Parameters.field(
        default=Integer(8),
        name="Number of heads", description="Number of attention heads inside the transformer's attentions"
    )
    num_queries: Integer = Parameters.field(
        default=Integer(100),
        name="Number of queries", description="Number of query slots"
    )
    pre_norm: Boolean = Parameters.field(
        default=Boolean(False),
        name="Pre norm", description="Pre norm"
    )
    # Segmentation
    masks: Boolean = Parameters.field(
        default=Boolean(False),
        name="Masks", description="Train segmentation head if the flag is provided"
    )

    # Loss
    aux_loss: Boolean = Parameters.field(
        default=Boolean(True),
        name="Auxiliary loss", description="Enables auxiliary decoding losses (loss at each layer)"
    )
    # Matcher
    set_cost_class: Float = Parameters.field(
        default=Float(1.0),
        name="Set cost class", description="Class coefficient in the matching cost"
    )
    set_cost_bbox: Float = Parameters.field(
        default=Float(5.0),
        name="Set cost bounding box", description="L1 box coefficient in the matching cost"
    )
    set_cost_giou: Float = Parameters.field(
        default=Float(2.0),
        name="Set cost GIOU", description="giou box coefficient in the matching cost"
    )

    # Loss coefficients
    mask_loss_coef: Float = Parameters.field(
        default=Float(1.0),
        name="Mask loss coefficient", description="Mask loss coefficient"
    )
    dice_loss_coef: Float = Parameters.field(
        default=Float(1.0),
        name="Dice loss coefficient", description="Dice loss coefficient"
    )
    bbox_loss_coef: Float = Parameters.field(
        default=Float(5.0),
        name="Bounding box loss coefficient", description="Bounding box loss coefficient"
    )
    giou_loss_coef: Float = Parameters.field(
        default=Float(2.0),
        name="GIOU loss coefficient", description="GIOU loss coefficientt"
    )
    eos_coef: Float = Parameters.field(
        default=Float(0.1),
        name="Eos coefficient", description="Relative classification weight of the no-object class"
    )

    # dataset parameters
    dataset_file: String = Parameters.field(
        default=String("coco"),
        name="dataset_file", description="dataset_file"
    )
    coco_path: String = Parameters.field(
        default=String(""),
        name="coco_path", description="coco_path"
    )
    coco_panoptic_path: Optional[String] = Parameters.field(
        default=None,
        name="coco_panoptic_path", description="coco_panoptic_path"
    )
    remove_difficult: Boolean = Parameters.field(
        default=Boolean(False),
        name="remove_difficult", description="remove_difficult"
    )
    output_dir: String = Parameters.field(
        default=String(""),
        name="output_dir", description='path where to save, empty for no saving'
    )
    seed: Integer = Parameters.field(
        default=Integer(42),
        name="seed", description="seed"
    )
    resume: String = Parameters.field(
        default=String("https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"),
        name="resume", description='resume from checkpoint'
    )
    start_epoch: Integer = Parameters.field(
        default=Integer(0),
        name="start_epoch", description="start_epoch"
    )
    eval: Boolean = Parameters.field(
        default=Boolean(False),
        name="eval", description="eval"
    )

def main(args):
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=1)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=1)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    #args = DetrParameters(coco_path="/home/jovyan/work/coco")#train
    args = DetrParameters(aux_loss=False, eval=True, resume="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", coco_path="/home/jovyan/work/coco")#eval
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
