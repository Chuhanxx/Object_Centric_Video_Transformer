import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from .transformers import TransformerDecoder,TransformerDecoderLayer
from timm.models.layers import trunc_normal_
from . import vit_helper
from functools import partial

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Spatio_Temporal_Attn_Head(nn.Module):

    def __init__(
            self,
            emb_dim_in=768,
            emb_dim_attn=512,
            st_head_layers=4,
            st_head_heads=4,
            mlp_ratio=4,
            num_classes =174,
            qkv_bias=True,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0.2,
            visual_drop_rate =0.5,
            add_gt = True,
            attn_type='joint',
            add_norm = False):

        super(Spatio_Temporal_Attn_Head, self).__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.proj = nn.Linear(emb_dim_in,emb_dim_attn)
        self._init_weights(self.proj)
        self.norm = norm_layer(emb_dim_attn)
        self.dropout = nn.Dropout(p=0.5)
        self.action_classifier = nn.Linear(emb_dim_attn, num_classes)
        self.avgpool2d = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool3d = nn.AdaptiveMaxPool3d((1,1,None))
        self.cls_token = nn.Parameter(torch.zeros(1 ,1, emb_dim_attn))
        self.x_mlp =  nn.Linear(emb_dim_in,emb_dim_attn)

        self.visual_drop_rate = visual_drop_rate
        self.add_norm = add_norm


        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, st_head_layers)]
        self.self_attn_blocks =  nn.ModuleList([
                vit_helper.Block(
                    attn_type=attn_type, 
                    dim= emb_dim_attn, 
                    num_heads=st_head_heads,
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, 
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate, 
                    drop_path=dpr[i], 
                    norm_layer=norm_layer,
                )
                for i in range(st_head_layers)
            ])

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            trunc_normal_(m.weight, std=1e-3)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x,  T=8, B=4, cls_token=None ):
        n_q, BT, C = x.shape
        x = rearrange(x, 'q (b t) c -> b (t q) c', t=T , b=B)
        cls_features = []
        
        if cls_token is not None:
            cls_features.append(cls_token)
            cls_token = cls_token.unsqueeze(1) 
        else:
            cls_token = self.cls_token.repeat(B,1,1)

        x = self.x_mlp(x)
        x = torch.cat([cls_token,x],1)

        if self.add_norm:
            x = self.norm(x)
            
        # attention between different objects at the same time step
        for i, blk in enumerate(self.self_attn_blocks): 
            x = blk(x, seq_len=n_q, num_frames=T,layer_n=i) 
            cls_features.append(x[:,0,:])
            
        return cls_features


