from dataclasses import dataclass
from typing import Optional, Tuple

from rvai.base.cell import CellMode, TrainableCell, cell
from rvai.base.context import (
    InferenceContext,
    ModelContext,
    ParameterContext,
    TestContext,
    TrainingContext,
)
from rvai.base.data import (
    Annotations,
    Class,
    Dataset,
    DatasetConfig,
    Example,
    Expertise,
    Inputs,
    Measurement,
    Measurements,
    Metrics,
    Outputs,
    Parameters,
    Samples,
    State,
    Tag,
)
from rvai.base.test import TestSession
from rvai.base.training import Model, ModelConfig, ModelPath, TrainingSession
from rvai.types import (
    Boolean,
    BoundingBox,
    Class,
    Classes,
    Dict,
    Enum,
    Float,
    Image,
    Integer,
    List,
    Point,
    String,
)

import numpy as np
import torch
import argparse
import datetime
import json
import random
import time
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler

#fix imports
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import datasets
import util.misc as utils
from datasets import get_coco_api_from_dataset, build_dataset_rvai
from engine import evaluate, train_one_epoch
from models import build_model

from .detr_helpers import get_model_dir, clear_folder

@dataclass
class DetrInputs(Inputs):
    image: Image = Inputs.field(name="Image", description="An image.")

@dataclass
class DetrOutputs(Outputs):
    boxes: List[BoundingBox] = Annotations.field(
        name="Bounding Boxes", description="A list of object bounding boxes.",
    )

@dataclass
class DetrSamples(DetrInputs, Samples):
    pass


@dataclass
class DetrAnnotations(DetrOutputs, Annotations):
    pass

@dataclass
class DetrParameters(Parameters):
    classes: Classes = Parameters.field(
        default_factory=lambda: Classes([]),
        name="Classes",
        description="Classes.",
    )

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
    #TODO check which backbones are possible
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
    resume: String = Parameters.field(
        default=String("https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"),
        name="resume", description='resume from checkpoint'
    )
    start_epoch: Integer = Parameters.field(
        default=Integer(0),
        name="start_epoch", description="start_epoch"
    )


@dataclass
class DetrMetrics(Metrics):
    mAP: Float = Metrics.field(
        name="Mean average precision", short_name="mAP", performance=True
    )


@dataclass
class ModelConfig:
    class_to_index: Dict[Class, int]
    index_to_class: Dict[int, Class]

@cell
class DetrCell(TrainableCell):

    @classmethod
    def load_model(
        cls,
        context: ModelContext,
        parameters: DetrParameters,
        model_path: Optional[ModelPath],
        dataset_config: Optional[DatasetConfig],
    ) -> Tuple[Model, ModelConfig]:
    #TODO load pretrained ...
        """Load a serialized model from disk."""
        if parameters.frozen_weights is not None:
            assert parameters.masks, "Frozen training is meant for segmentation only"

        model, criterion, postprocessors = build_model(parameters)

        model = (model, criterion, postprocessors)

        class_to_idx, idx_to_class = parameters.classes.class_index_mapping()


        return (
            model,
            ModelConfig(
                class_to_index=class_to_idx,
                index_to_class=idx_to_class,
            ),
        )


    @classmethod
    def train(
        cls,
        context: TrainingContext,
        parameters: DetrParameters,
        model: Model,
        model_config: ModelConfig,
        train_dataset: Dataset[DetrSamples, DetrAnnotations],
        validation_dataset: Dataset[DetrSamples, DetrAnnotations],
        dataset_config: Optional[DatasetConfig],
    ) -> TrainingSession[DetrMetrics]:
        """Train a predictive model on annotated data."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model, criterion, postprocessors = model
        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": parameters.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=parameters.lr,
                                    weight_decay=parameters.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, parameters.lr_drop)

        train_dataset = build_dataset_rvai(train_dataset, class_to_index=model_config.class_to_index, image_set='train', args=parameters)
        validation_dataset = build_dataset_rvai(validation_dataset, class_to_index=model_config.class_to_index, image_set='val', args=parameters)
        
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(validation_dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, parameters.batch_size, drop_last=True)

        data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=1)
        data_loader_val = DataLoader(validation_dataset, parameters.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=1)

        if parameters.dataset_file == "coco_panoptic":
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = datasets.coco.build("val", parameters)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(validation_dataset)

        if parameters.frozen_weights is not None:
            checkpoint = torch.load(parameters.frozen_weights, map_location='cpu')
            model_without_ddp.detr.load_state_dict(checkpoint['model'])

        output_dir = Path(get_model_dir())
        clear_folder(output_dir)
        if parameters.resume:
            if parameters.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    parameters.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(parameters.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                parameters.start_epoch = checkpoint['epoch'] + 1

        print("Start training")
        for epoch in range(parameters.start_epoch, parameters.epochs):
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                parameters.clip_max_norm)
            lr_scheduler.step()
            
            # save every epoch
            checkpoint_path = output_dir / f'checkpoint{epoch:04}.pth'
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': parameters,
                }, checkpoint_path)

            test_stats = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir, parameters.classes
            )

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        return checkpoint_path

    @classmethod
    def test(
        cls,
        context: TestContext,
        parameters: DetrParameters,
        model: Model,
        model_config: ModelConfig,
        test_dataset: Dataset[DetrSamples, DetrAnnotations],
        dataset_config: Optional[DatasetConfig],
    ) -> TestSession[DetrMetrics]:
        """Test the performance of a predictive model on new, unseen, data."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model, criterion, postprocessors = model
        model.to(device)

        model_without_ddp = model

        test_dataset = build_dataset_rvai(test_dataset, class_to_index=model_config.class_to_index, image_set='val', args=parameters)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        data_loader_test = DataLoader(test_dataset, parameters.batch_size, sampler=sampler_test,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=1)

        if parameters.dataset_file == "coco_panoptic":
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = datasets.coco.build("val", parameters)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(test_dataset)

        if parameters.frozen_weights is not None:
            checkpoint = torch.load(parameters.frozen_weights, map_location='cpu')
            model_without_ddp.detr.load_state_dict(checkpoint['model'])

        output_dir = Path(get_model_dir())
        if parameters.resume:
            if parameters.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    parameters.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(parameters.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])

        test_stats, mAP = evaluate(model, criterion, postprocessors,
                                                data_loader_test, base_ds, device, output_dir, parameters.classes)
        return DetrMetrics(mAP = Float(mAP))

    @classmethod
    def predict(
        cls,
        context: InferenceContext,
        parameters: DetrParameters,
        model: Model,
        model_config: ModelConfig,
        inputs: DetrInputs,
    ) -> DetrOutputs:
        """Make predictions about sampled input data using a predictive model."""
        # TODO: implement
        ...
