import copy
from typing import List, Optional

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import matplotlib.pyplot as plt
from ..visualization.utils import plot_attn_map
from einops import rearrange
import pickle as cp

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        # self.layers[-1] = per_frame_decoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        seq_len = 196, 
        num_frames = 8,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for i in range(len(self.layers)):
            layer = self.layers[i]
            output, attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                counter =i,
                seq_len = seq_len,
                num_frames= num_frames
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, attn




class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
        activation="relu", normalize_before=False, traj=False, num_layers =4):
        super().__init__()
        self.traj = traj
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        if not self.traj:
            self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.cross_attn_image = TrajectoryCrossAttention(d_model, nhead,proj_drop=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        seq_len = 196, 
        num_frames = 8,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        counter: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2,attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        # print(counter,attn)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        if not self.traj:
            # if memory.shape[0] == seq_len*num_frames:
            #     memory  = rearrange(memory, '(t s) b c -> s (b t) c', t =num_frames)
            #     tgt = tgt.repeat_interleave(num_frames,dim=1)

            tgt2, attn = self.cross_attn_image(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,)
        else:
            tgt2, attn = self.cross_attn_image(
                query = tgt.transpose(0,1),
                memory = memory.transpose(0,1),
                seq_len = seq_len, 
                num_frames = num_frames,
                layer_n = counter)
            tgt2 = tgt2.transpose(0,1)

        # for i in range(1,5):plot_attn_map(attn.view(2,8,-1,14,14)[0,:,i].detach(),name='layer'+str(counter)+'_'+str(i),n_cols=4, n_rows=2)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt) 
        return tgt, attn

    def forward_pre(
        self,
        tgt,
        memory,
        seq_len = 196, 
        num_frames = 8,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = 0,
        counter: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        if not self.traj:
            tgt2, attn = self.cross_attn_image(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
        else:
            tgt2, attn = self.cross_attn_image(
                query = tgt2.transpose(0,1),
                memory = memory.transpose(0,1),
                seq_len = seq_len, 
                num_frames = num_frames,
                layer_n = counter)
            tgt2 = tgt2.transpose(0,1)
        # for i in range(6):plot_attn_map(attn[0,i].view(4,14,14),name='layer'+str(counter)+'_'+str(i),n_cols=2, n_rows=2)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn

    def forward(
        self,
        tgt,
        memory,
        seq_len = 196, 
        num_frames = 8,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        counter: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, 
                memory, 
                seq_len, 
                num_frames, 
                tgt_mask, 
                memory_mask, 
                text_memory_key_padding_mask,
                tgt_key_padding_mask, 
                memory_key_padding_mask, 
                pos, 
                query_pos, 
                counter
            )
        return self.forward_post(
            tgt,
            memory,
            seq_len, 
            num_frames,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            counter,
        )


class TrajectoryCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        


    def forward(self, query, memory, seq_len=196, num_frames=8, layer_n=0, approx='none'):
        B, N, C = memory.shape
        S = seq_len
        F = num_frames
        h = self.num_heads

        # project x to q, k, v vaalues
        q = self.q(query) # b, q, d
        k, v = self.kv(memory).chunk(2, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Using full attention
        q_dot_k = q @ k .transpose(-2, -1)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)

        space_attn = self.scale * q_dot_k
        attn = self.attn_drop(space_attn.softmax(dim=-1))
        # print(layer_n, attn[0,:,3,:].mean(0))
 
        # if layer_n ==3: 
        #     for i in range(5):
        #         plot_attn_map(attn[:,i,:,:196].mean(0).view(8,14,14),name='figs/obj_'+str(i),n_cols=4, n_rows=2)
        #         for j in range(4):
        #             plot_attn_map(attn[j,i,:,:196].view(8,14,14),name='figs/obj_'+str(i)+'_head'+str(j),n_cols=4, n_rows=2)
        #     with open('figs/attn.pkl','wb') as f:
        #         cp.dump(attn[:,:,:,:196].detach().cpu().numpy(),f)
        
        v = rearrange(v, 'b (f s) d -> b f s d',  f=F, s=S)
        x = torch.einsum('b q f s, b f s d -> b q f d', attn, v)

        # Temporal attention: new keys and values are the similarity-aggregated patch
        kv2 = rearrange(x, '(b h) q f d -> b q f (h d)', b=B)
        k2, v2 = self.proj_kv(kv2).chunk(2, dim=-1)
        k2, v2 = map(
            lambda t: rearrange(t, f'b q f (h d) -> b h q f d', f=F,  h=h), (k2, v2))

        q2 = self.proj_q(rearrange(q, '(b h) q d -> b q (h d)', h=h))
        q2 = rearrange(q2, f'b q (h d) -> b h q d', h=h)
        q2 *= self.scale

        attn2 = torch.einsum('b h q d, b h q f d -> b h q f', q2, k2)
        attn2 = attn2.softmax(dim=-1)
        q_out = torch.einsum('b h q f, b h s f d -> b h q d', attn2, v2)
        q_out = rearrange(q_out, f'b h q d -> b q (h d)')

        q_out = self.proj(q_out)
        attn = rearrange(attn,'(b h) q f h_w -> b h q f h_w', b = B).mean(1)
        return q_out, space_attn


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class TransformerDecoderLayer_SAfirst(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
        activation="relu", normalize_before=False, traj=False, num_layers =4):
        super().__init__()
        self.traj = traj
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        if not self.traj:
            self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.cross_attn_image = TrajectoryCrossAttention(d_model, nhead,proj_drop=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        seq_len = 196, 
        num_frames = 8,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        counter: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Cross attention to image
        tgt2, attn = self.cross_attn_image(
            query = tgt.transpose(0,1),
            memory = memory.transpose(0,1),
            seq_len = seq_len, 
            num_frames = num_frames,
            layer_n = counter)
        tgt2 = tgt2.transpose(0,1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Self attention
        tgt2,attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        # print(counter,attn)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt) 
        return tgt, attn

    def forward(
        self,
        tgt,
        memory,
        seq_len = 196, 
        num_frames = 8,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        counter: Optional[Tensor] = None,
    ):
       return self.forward_post(
            tgt,
            memory,
            seq_len, 
            num_frames,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            counter,
        )
