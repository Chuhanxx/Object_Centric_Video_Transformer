import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from einops import rearrange
from .transformers import TransformerDecoder, TransformerDecoderLayer, MLP, positionalencoding1d
from timm.models.layers import trunc_normal_
import numpy as np
import torch.distributions as dist


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Obj_Query_Decoder(nn.Module):
    """Object Query Decoder with a cross-attention transformer,
    """

    def __init__(
            self,
            num_layers=4,
            num_heads=4,
            n_queries=4,
            emb_dim_in = 768,
            emb_dim_attn = 512,
            n_objs = 5,
            dropout=0,
            add_gt='none',
            predict_box=False,
            temp_attn=False,
            cls_token=0,
            attn_type='joint',
            pre_norm=False,
            num_classes=0,
            pred_vis=False,
            pred_state=False,
            add_norm=False,
            layers_from_bb =[12],):

        super(Obj_Query_Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_queries = n_queries 
        self.n_objs = n_objs
        self.add_gt = add_gt
        self.predict_box = predict_box
        self.cls_token = cls_token
        self.num_classes = num_classes
        self.temp_attn = temp_attn
        self.add_norm = add_norm
        self.n_layers = len(layers_from_bb)
        
        decoder_layer =  TransformerDecoderLayer(
                        emb_dim_attn, 
                        num_heads, 
                        dim_feedforward=512, 
                        normalize_before=pre_norm, 
                        dropout=dropout,
                        traj= (attn_type == 'trajectory'))

        if self.n_layers ==1:
            self.decoder = TransformerDecoder(
                            decoder_layer, 
                            num_layers,
                            norm =nn.LayerNorm(emb_dim_attn),
                            return_intermediate=False)
        else:
            self.decoder_layers = _get_clones(decoder_layer, num_layers)
            self.decoder_norm = nn.LayerNorm(emb_dim_attn)


        self.proj = MLP(emb_dim_in,emb_dim_in//2,emb_dim_attn,2)
        self.id_proj = MLP(emb_dim_in+emb_dim_attn,emb_dim_in//2,emb_dim_attn,2)
            
        self.queries = nn.Embedding(n_queries+self.cls_token,emb_dim_attn)
        trunc_normal_(self.queries.weight, std=.02)
        self.query_norm = nn.LayerNorm(emb_dim_attn)

        temporal_pos_emb =  positionalencoding1d(emb_dim_attn,32)
        self.register_buffer('temporal_pos_emb', temporal_pos_emb, persistent=False)

 
    def forward(self, x, boxes, id_embeds=None, perm=None):
        B,T,N,L,C = x.shape
        x = self.proj(x)
        
        x = rearrange(x, 'b t n l c -> b n (t l) c')
        t_pos_emb = self.temporal_pos_emb[None,None,:T]
        x += t_pos_emb.repeat(1,N,L,1)
            
        queries = self.queries.weight[None,:].repeat(x.shape[0],1,1) 
        
        if id_embeds is not None:
            id_embeds = id_embeds[0].repeat(B,1,1)
            concat_id_embeds = torch.cat([queries[:,:id_embeds.shape[1],:],id_embeds],-1)
            queries[:,:id_embeds.shape[1],:] = self.id_proj(concat_id_embeds)
            
        if self.add_norm:
            queries = self.query_norm(queries)

        decoder_out, _ = self.decoder(queries.transpose(0,1),
                                    x[:,0].transpose(0,1),
                                    seq_len = L,
                                    num_frames = T)

        return decoder_out





