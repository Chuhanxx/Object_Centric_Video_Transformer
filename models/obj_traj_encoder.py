import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from . import vit_helper
from functools import partial
import torchvision.ops as ops

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

class ObjTrajEncoder(nn.Module):

    def __init__(
            self,
            emb_dim=512,
            num_layers=2,
            num_heads=4,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0.2,
            use_cls_token=True,
            n_objects=5,
            enc_type = 'traj'):
        super(ObjTrajEncoder, self).__init__()
        
        self.n_objects = n_objects 
        self.roi_sz = (7,7)
        self.enc_type = enc_type
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(emb_dim)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1 ,1, emb_dim))
        else:
            self.cls_token = None
        self.box_proj = MLP(4,emb_dim,emb_dim,2)
        self.temp_embed_box = nn.Embedding(num_embeddings=32, embedding_dim=512)
        self.temp_embed_affin = nn.Embedding(num_embeddings=32, embedding_dim=512)
        self.box_pre_norm = nn.LayerNorm(512)
        self.affin_pre_norm = nn.LayerNorm(512)

        
        dpr = [x.item() for x in torch.linspace(
                0, drop_path_rate, num_layers)]
        if ('traj' in enc_type) or ('traj-shuffle' in enc_type):
            self.self_attn_blocks_taj =  nn.ModuleList([
                    vit_helper.Block(
                        attn_type='joint', 
                        dim= emb_dim, 
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=dpr[i], 
                        norm_layer=norm_layer,
                    )
                    for i in range(num_layers)
                ])


    def forward(self, x, boxes):
        B,T,O,C = boxes.shape
        outs = []
        if 'traj' in self.enc_type:
            boxes_in = rearrange(boxes, 'b t o c -> (b o) t c')
            boxes_in = self.box_proj(boxes_in)
            boxes_in = self.box_pre_norm(boxes_in) + self.temp_embed_box.weight[None,0:T]
            
            # add temporal emb then attention
            if self.cls_token is not None:
                cls_token = self.cls_token.repeat(B*O,1,1)
                boxes_in = torch.cat([cls_token,boxes_in],1)   
            for i, blk in enumerate(self.self_attn_blocks_taj): 
                boxes_in = blk(boxes_in) 
            out = rearrange(boxes_in,'(b o) t c -> b o t c', b =B, o=O)
            outs.append(['traj', out[:,:,0,:]])

        if 'traj-shuffle' in self.enc_type:
            boxes_in = rearrange(boxes, 'b t o c -> (b o) t c')
            boxes_in = self.box_proj(boxes_in)
            NUM_NEG = 3
            boxes_all = []

            if self.cls_token is not None:
                cls_token = self.cls_token.repeat(B*O,1,1)
                boxes_tmp = self.box_pre_norm(boxes_in) + self.temp_embed_box.weight[None,0:T]
                boxes_tmp = torch.cat([cls_token,boxes_tmp],1)   
            for i, blk in enumerate(self.self_attn_blocks_taj): 
                boxes_tmp = blk(boxes_tmp) 
            out = rearrange(boxes_tmp,'(b o) t c -> b o t c', b =B, o=O)
            boxes_all.append(out[:,:,0,:])
            
            for _ in range(NUM_NEG):
                # temporally shuffle the box, feed as negative
                if self.cls_token is not None:
                    cls_token = self.cls_token.repeat(B*O,1,1)
                    boxes_tmp = boxes_in[:,torch.randperm(boxes_in.shape[1], device=boxes_in.device),:]
                    boxes_tmp = self.box_pre_norm(boxes_tmp) + self.temp_embed_box.weight[None,0:T]
                    boxes_tmp = torch.cat([cls_token,boxes_tmp],1)
                else:
                    raise NotImplementedError
                
                for i, blk in enumerate(self.self_attn_blocks_taj): 
                    boxes_tmp = blk(boxes_tmp)
                out = rearrange(boxes_tmp,'(b o) t c -> b o t c', b =B, o=O)
                boxes_all.append(out[:,:,0,:])
                
            assert len(boxes_all) == 1 + NUM_NEG
            outs.append(['traj-shuffle', torch.stack(boxes_all, dim=2)])  # b o X c; X=1+NUM_NEG
            
        return dict(outs)

