#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""
import os
import math
import torch
import slowfast as slowfast
from fvcore.common.registry import Registry

from . import vit_helper

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if isinstance(model, slowfast.models.video_model_builder.VisionTransformer):
        if cfg.VIT.IM_PRETRAINED:
            vit_helper.load_pretrained(
                model, cfg=cfg, num_classes=cfg.MODEL.NUM_CLASSES, 
                in_chans=cfg.VIT.CHANNELS, filter_fn=vit_helper._conv_filter, 
                strict=False
            )
            if hasattr(model, 'st_embed'):
                model.st_embed.data[:, 1:, :] = model.pos_embed.data[:, 1:, :].repeat(
                    1, cfg.VIT.TEMPORAL_RESOLUTION, 1)
                model.st_embed.data[:, 0, :] = model.pos_embed.data[:, 0, :]
            if hasattr(model, 'patch_embed_3d'):
                model.patch_embed_3d.proj.weight.data = torch.zeros_like(
                    model.patch_embed_3d.proj.weight.data)
                n = math.floor(model.patch_embed_3d.proj.weight.shape[2] / 2)
                model.patch_embed_3d.proj.weight.data[:, :, n, :, :] = model.patch_embed.proj.weight.data
                model.patch_embed_3d.proj.bias.data = model.patch_embed.proj.bias.data

    elif isinstance(model, slowfast.models.img_model_builder.VisionTransformer2D) and cfg.MODEL.BACKBONE!='StltBackbone':
        pretrained_file = { 'deitsmall8': 'dino_deitsmall8_pretrain.pth',
                        'deitsmall16': 'dino_deitsmall16_pretrain.pth',
                        'vitbase8': 'dino_vitbase8_pretrain.pth',
                        'vitbase16': 'dino_vitbase16_pretrain.pth',
                        'resnet50': 'dino_resnet50_pretrain.pth',
                        }
        pretrained_weights = cfg.VIT.PRETRAINED_WEIGHTS

        if cfg.VIT.IM_PRETRAINED and pretrained_weights in pretrained_file:
            state_dict = torch.load(
                os.path.join(
                    cfg.VIT.PRETRAINED_PATH,
                    pretrained_file[pretrained_weights],
                ),
                map_location="cpu",
                )
            model.backbone.load_state_dict(state_dict, strict=True)
            
            if hasattr(model.backbone, 'st_embed'):
                model.backbone.st_embed.data[:, 1:, :] = model.backbone.pos_embed.data[:, 1:, :].repeat(
                    1, cfg.VIT.TEMPORAL_RESOLUTION, 1)
                model.backbone.st_embed.data[:, 0, :] = model.backbone.pos_embed.data[:, 0, :]
            if hasattr(model.backbone, 'patch_embed_3d'):
                model.backbone.patch_embed_3d.proj.weight.data = torch.zeros_like(
                    model.backbone.patch_embed_3d.proj.weight.data)
                n = math.floor(model.backbone.patch_embed_3d.proj.weight.shape[2] / 2)
                model.backbone.patch_embed_3d.proj.weight.data[:, :, n, :, :] != model.backbone.patch_embed.proj.weight.data
                model.backbone.patch_embed_3d.proj.bias.data = model.backbone.patch_embed.proj.bias.data
                
        elif cfg.VIT.IM_PRETRAINED  and cfg.TRAIN.BACKBONE_PRETRAINED_WEIGHTS!='':
            vit_helper.load_pretrained(
                model.backbone, cfg=cfg, num_classes=cfg.MODEL.NUM_CLASSES, 
                in_chans=cfg.VIT.CHANNELS, filter_fn=vit_helper._conv_filter, 
                strict=True
            )

    elif cfg.MODEL.BACKBONE =='StltBackbone' and cfg.VIT.BOX_STLT_ENCODER.PRETRAINED_PATH!='':
        stlt = torch.load(cfg.VIT.BOX_STLT_ENCODER.PRETRAINED_PATH)
        if 'model_state' in stlt:
            stlt = stlt['model_state']
        stlt = {k.replace('stlt_backbone','backbone').replace('prediction_head','backbone.prediction_head'):v for k,v in stlt.items()}
        # if  cfg.VIT.BOX_STLT_ENCODER.OBJ_CLS_EMBED =='id':
        #     stlt = {k:v for k,v in stlt.items() if 'category_embeddings' not in k}
        
        print('WEIGHTS NOT LOADED FROM THE PRETRAINED STLT MODEL')
        print([k for k in stlt.keys() if k not in [*model.state_dict().keys()]])
        print('WEIGHTS MISSING IN THE PRETRAINED STLT MODEL')
        print([k for k in [*model.state_dict().keys()] if k not in stlt.keys()])
        model.load_state_dict(stlt,strict=False)

    if cfg.VIT.BOX_STLT_ENCODER.ENABLE and cfg.VIT.BOX_STLT_ENCODER.PRETRAINED_PATH != '':
        stlt = torch.load(cfg.VIT.BOX_STLT_ENCODER.PRETRAINED_PATH)
        if 'model_state' in stlt:
            stlt = stlt['model_state']
        if  cfg.VIT.BOX_STLT_ENCODER.OBJ_CLS_EMBED =='id':
            stlt = {k:v for k,v in stlt.items() if 'category_embeddings' not in k}
        stlt = {k.replace('stlt_','').replace('backbone.',''):v for k,v in stlt.items() if 'head' not in k and 'pre_logits' not in k}
        print('WEIGHTS NOT USED FROM THE PRETRAINED STLT MODEL')
        print([k for k in stlt.keys() if k not in [*model.stlt_backbone.state_dict().keys()]])
        print('WEIGHTS MISSING IN THE PRETRAINED STLT MODEL')
        print([k for k in [*model.stlt_backbone.state_dict().keys()] if k not in stlt.keys()])
        model.stlt_backbone.load_state_dict(stlt,strict=False)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=True
        )
    return model
