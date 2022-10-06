from pyexpat import features
from .build import MODEL_REGISTRY
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import random
import string
import torchvision
from .st_attn import Spatio_Temporal_Attn_Head
from timm.models.layers import trunc_normal_
from einops import rearrange
from .video_model_builder import VisionTransformer
from .transformers import MLP, positionalencoding1d
from .obj_query_module import Obj_Query_Decoder
from .stlt_box_encoder import StltBackbone
from .obj_traj_encoder import ObjTrajEncoder
from . import vit_helper
from functools import partial


@MODEL_REGISTRY.register()
class VisionTransformer2D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.embed_dim = cfg.VIT.EMBED_DIM
        self.head_act = cfg.VIT.HEAD_ACT
        self.head_dropout = cfg.VIT.HEAD_DROPOUT
        if "Epickitchens" in cfg.TRAIN.DATASET:
            self.num_classes = [97, 300]  
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        self.patch_size = cfg.VIT.PATCH_SIZE
        self.num_frames = cfg.DATA.NUM_FRAMES//cfg.VIT.PATCH_SIZE_TEMP
        self.layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.n_objects = cfg.VIT.OBJ_QUERY_MODULE.N_OBJECTS
        cls_multiplier =1 

        self.backbone = VisionTransformer(cfg)
        self.stlt_backbone = StltBackbone(object_cls_embed = cfg.VIT.BOX_STLT_ENCODER.OBJ_CLS_EMBED,
                                            add_frame_emb =cfg.VIT.BOX_STLT_ENCODER.ADD_FRAME_EMB,
                                            n_objs = cfg.VIT.BOX_STLT_ENCODER.N_OBJS,
                                            use_masks =(cfg.DATA.LOAD_MASK or cfg.DATA.LOAD_MASK_FROM_BOX),
                                            use_boxes = cfg.DATA.LOAD_BOX,
                                            use_flows = cfg.DATA.LOAD_FLOW,
                                            mask_sz =cfg.DATA.MASK_SIZE )
        
        self.obj_traj_encoder = ObjTrajEncoder(
            emb_dim =cfg.VIT.OBJ_QUERY_MODULE.EMB_DIM,
            enc_type = cfg.VIT.OBJ_TRAJ_ENCODER.ENC_TYPE,
            n_objects = cfg.VIT.OBJ_QUERY_MODULE.N_OBJECTS)                            
           
        self.obj_query_module = Obj_Query_Decoder(
                                num_layers=cfg.VIT.OBJ_QUERY_MODULE.N_LAYERS,
                                num_heads=cfg.VIT.OBJ_QUERY_MODULE.N_HEADS,
                                n_queries=cfg.VIT.OBJ_QUERY_MODULE.N_QUERIES,
                                n_objs=cfg.VIT.OBJ_QUERY_MODULE.N_OBJECTS,
                                emb_dim_attn=cfg.VIT.OBJ_QUERY_MODULE.EMB_DIM,
                                add_gt = cfg.VIT.OBJ_QUERY_MODULE.ADD_GT,
                                dropout= cfg.VIT.OBJ_QUERY_MODULE.DROP,
                                predict_box = cfg.VIT.OBJ_QUERY_MODULE.BOX_DECODER.ENABLE,
                                temp_attn = cfg.VIT.OBJ_QUERY_MODULE.TEMP_ATTN,
                                cls_token=cfg.VIT.OBJ_QUERY_MODULE.CLS_TOKEN,
                                attn_type = cfg.VIT.OBJ_QUERY_MODULE.ATTN_TYPE,
                                pre_norm = cfg.VIT.OBJ_QUERY_MODULE.PRE_NORM,
                                num_classes = cfg.VIT.OBJ_QUERY_MODULE.BOX_DECODER.NUM_CLASSES,
                                add_norm =  cfg.VIT.OBJ_QUERY_MODULE.ADD_NORM,
                                layers_from_bb = cfg.VIT.LAYERS_OUT, )
            
        self.st_reason_module = Spatio_Temporal_Attn_Head(
                    st_head_layers = cfg.VIT.ST_MODEL.N_LAYERS,
                    st_head_heads=  cfg.VIT.ST_MODEL.N_HEADS,
                    emb_dim_in = cfg.VIT.OBJ_QUERY_MODULE.EMB_DIM,
                    emb_dim_attn=cfg.VIT.ST_MODEL.EMB_DIM*cls_multiplier,
                    drop_rate = cfg.VIT.ST_MODEL.DROP,
                    attn_drop_rate = cfg.VIT.ST_MODEL.ATTN_DROPOUT,
                    drop_path_rate = cfg.VIT.ST_MODEL.DROP_PATH,
                    add_gt = cfg.VIT.ST_MODEL.ADD_GT,
                    visual_drop_rate= cfg.VIT.ST_MODEL.VISUAL_DROPOUT,
                    attn_type = cfg.VIT.ST_MODEL.ATTN_TYPE,
                    add_norm = cfg.VIT.ST_MODEL.ADD_NORM,
                    )
    
        self.final_cls = nn.Embedding(1, embedding_dim=768*2)
        self.emb_out = cfg.VIT.HEAD_DIM

        if cfg.VIT.USE_MLP:
            hidden_dim = self.emb_out
            print("Using TanH activation in MLP")
            act = nn.Tanh() 
   
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.emb_out, hidden_dim)),
                ('act', act),
            ]))
            self.pre_logits_0 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.emb_out, hidden_dim)),
                ('act', act),
            ]))

        else:
            self.pre_logits = nn.Identity()
        
        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)

        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d"%a, nn.Linear(self.emb_out*cls_multiplier, self.num_classes[i]))
                setattr(self, "head%d_0"%a, nn.Linear(self.emb_out*cls_multiplier, self.num_classes[i]))
        else:
            self.head = nn.Linear(self.emb_out, self.num_classes) 
            self.head_0 = nn.Linear(self.emb_out*2, self.num_classes)

    def forward(self, x, meta, backbone_feature_out=False,features_in=False):
        distance_mats = []
                
        if isinstance(x,list):
            x = x[0]

        boxes = meta['boxes'].to(x.dtype)
        B,C,T,H,W = x.shape

        #### Visual Encoder #### 
        cls_token , out, _, _, _ = self.backbone.forward(x,meta,return_features=True)
        T = T//self.cfg.VIT.PATCH_SIZE_TEMP
        # n is the number of intermediate self-attention layers that we read visual feature from
        # in our paper, we always read from the 6th layer, thus n = 1
        visual_out = rearrange(out[:,:,1:], 'b n (t h w) c ->  b t n h w c',
                h=H//self.patch_size, w=W//self.patch_size, t=T)
            
        #### Trajtectory Encoder #### 
        visual_features =  rearrange(visual_out,'b t n h w c -> b t n (h w) c')
        meta['boxes'] = meta["boxes"].to(visual_features.dtype)
        stlt_embs,stlt_obj_embs = self.stlt_backbone(meta)
        
        stlt_cls_token = stlt_embs[-1]
        cls_token = torch.cat([cls_token,stlt_cls_token],dim=-1)
        stlt_embs = rearrange(stlt_embs[:-1,...],
                                '(t k) b c -> b t k c',
                                t=self.num_frames, k=2)
        stlt_obj_embs = rearrange(stlt_obj_embs,
                                'b (t k) o c -> b t k o c',
                                t=self.num_frames, k=2)

        # average to downsample the temporal dimension, so that it is compatiable with the temporal dimension in visual encoder
        stlt_embs = stlt_embs.mean(2)[:,:,None,None,:]
        stlt_obj_embs = stlt_obj_embs.mean(2)[:,:,None,:,:]

        out = torch.cat([visual_features,stlt_embs,stlt_obj_embs],3)
        stlt_spatial = self.stlt_backbone.frames_embeddings.layout_embedding.category_box_embeddings

        # Initialize Object ID Embeddings
        id_embeds = stlt_spatial.category_embeddings.weight[1:]
        id_embeds = id_embeds[None,None,...].detach()

        #### Object Learner ####
        summaries = self.obj_query_module(
            out, 
            boxes,
            id_embeds=id_embeds,
            perm=None) # [n_q b c]

        #### Classification Module ####
        x = self.st_reason_module(
            summaries, 
            T=summaries.shape[1]//B, 
            B=B, 
            cls_token=self.final_cls.weight.repeat(B,1))

        # Generate object-level trajectory embeddings for auxiliary loss
        obj_gt_embeds = self.obj_traj_encoder(visual_out,boxes) # [B O t+1 C]
        for key_gt, obj_gt_emb in obj_gt_embeds.items():
            if obj_gt_emb.dim() == 3:
                obj_embeds = summaries.transpose(0,1)[:,:self.n_objects]
                obj_embeds = nn.functional.normalize(obj_embeds, dim=-1)
                obj_gt_emb = nn.functional.normalize(obj_gt_emb, dim=-1)
                distance_mat = torch.mm(obj_embeds.flatten(0,1),obj_gt_emb.flatten(0,1).t())/0.07 
            elif obj_gt_emb.dim() == 4:
                obj_embeds = summaries.transpose(0,1)[:,:self.n_objects]
                obj_embeds = nn.functional.normalize(obj_embeds, dim=-1)
                obj_gt_emb = nn.functional.normalize(obj_gt_emb, dim=-1)
                distance_mat = torch.einsum('nc,mxc->nmx', 
                    obj_embeds.flatten(0,1),obj_gt_emb.flatten(0,1))/0.07                 
            distance_mats.append([key_gt, distance_mat])

        out_cls_x = []
        features_out = x.copy()
        
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            output = []
            # assert self.emb_out* len(self.num_classes) == x[0].shape[-1]
            for head in range(len(self.num_classes)):
                x_out = getattr(self, "head%d_0"%head)(cls_token)
                if not self.training:
                    x_out = torch.nn.functional.softmax(x_out, dim=-1)
                output.append(x_out)
            out_cls_x.append(output)
        else:
            cls_x = self.pre_logits_0(cls_token])
            cls_x = self.head_drop(cls_x)
            cls_x = self.head_0(cls_x)
            if not self.training:
                cls_x = torch.nn.functional.softmax(cls_x, dim=-1)
            out_cls_x.append(cls_x)


        for layer_x in x[1:]:   
            if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
                output = []
                for head in range(len(self.num_classes)):
                    x_out = getattr(self, "head%d"%head)(layer_x)
                    if not self.training:
                        x_out = torch.nn.functional.softmax(x_out, dim=-1)
                    output.append(x_out)
                out_cls_x.append(output)
            else:
                cls_x = self.pre_logits(layer_x)
                cls_x = self.head_drop(cls_x)
                cls_x = self.head(cls_x)
                if not self.training:
                    cls_x = torch.nn.functional.softmax(cls_x, dim=-1)
                out_cls_x.append(cls_x)
    
        if distance_mats is not None:
            distance_mats = dict(distance_mats)

        return out_cls_x, distance_mats, features_out
