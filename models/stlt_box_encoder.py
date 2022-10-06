from typing import Dict
from unicodedata import category

import torch
from torch import nn
from torch.nn import functional as F
from .build import MODEL_REGISTRY
from timm.models.layers import trunc_normal_
from .vit_helper import PatchEmbedMask


class CategoryBoxEmbeddings(nn.Module):
    def __init__(self,
                 object_cls_embed,
                 n_objs=6,
                 use_masks=False,
                 use_boxes=True,
                 use_flows=False,
                 mask_sz = 224):
        super(CategoryBoxEmbeddings, self).__init__()
        hidden_size = 768

        self.max_num_objects = n_objs
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.object_cls_embed = object_cls_embed
        self.use_masks = use_masks
        self.use_boxes = use_boxes 
        self.use_flows = use_flows 

        frame_cls_token = torch.full((self.max_num_objects, 4), 0.0)
        frame_cls_token[0] = torch.tensor([0, 0, 1, 1])
        box_cls_token = torch.tensor([0,0,1,1])
        self.box_embedding = nn.Linear(4, hidden_size)

        self.register_buffer('box_cls_token', box_cls_token[None,None,None], persistent=False)
        self.register_buffer('frame_cls_token', frame_cls_token[None,None], persistent=False)

        linear_labels = torch.arange(self.max_num_objects)
        self.register_buffer('linear_labels', linear_labels[None,None,:], persistent=False)
        n_embeddings = n_objs

        self.category_embeddings = nn.Embedding(
            embedding_dim=hidden_size,
            num_embeddings=n_embeddings,
            padding_idx=0,
        )
        
    def forward(self, meta) -> torch.Tensor:
        B,T,O,_  = meta['boxes'].shape
        boxes_embeddings,boxes = self.embed_box(meta)

        box_labels = self.linear_labels[:O+1].repeat(B,T+1,1)    
        category_embeddings = self.category_embeddings(box_labels)
        embeddings = category_embeddings + boxes_embeddings
        if "scores" in meta:
            score_embeddings = self.score_embeddings(meta["scores"].unsqueeze(-1))
            embeddings += score_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, boxes, box_labels # [B,T,O,C]

    def embed_box(self,meta):
        boxes = meta['boxes']
        B,T,_,_  = boxes.shape
        # Add box cls token, [B,T,N,C] - > [B,T,N+1,C]
        boxes = torch.cat([self.box_cls_token.repeat(B,T,1,1),
                          boxes],-2)
        # Add video cls token, [B,T,N+1,C] - > [B,T+1,N+1,C]
        boxes = torch.cat([boxes,
                           self.frame_cls_token.repeat(B,1,1,1),],1)
        
        boxes_embeddings = self.box_embedding(boxes)
        return boxes_embeddings,boxes


class SpatialTransformer(nn.Module):
    def __init__(self,object_cls_embed,
                 n_objs=6,
                 use_masks=False,
                 use_boxes=True,
                 use_flows=False,
                 mask_sz = 224):
        super(SpatialTransformer, self).__init__()
        hidden_size =768
                
        self.category_box_embeddings = CategoryBoxEmbeddings(object_cls_embed,
                                                             n_objs=n_objs,
                                                             use_masks = use_masks,
                                                             use_boxes = use_boxes,
                                                             use_flows = use_flows,
                                                             mask_sz = mask_sz)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=4
        )
        self.obj_cls_embed = object_cls_embed


    def forward(self, meta):
        # [Batch size, Num. frames, Num. boxes, Hidden size]
        cb_embeddings, boxes, box_labels = self.category_box_embeddings(meta)
        num_frames, num_boxes, hidden_size = cb_embeddings.size()[1:]
        
        # [Batch size x Num. frames, Num. boxes, Hidden size]
        cb_embeddings = cb_embeddings.flatten(0, 1)
        # [Num. boxes, Batch size x Num. frames, Hidden size]
        cb_embeddings = cb_embeddings.transpose(0, 1)

        # [Batch size x Num. frames, Num. boxes]
        src_key_padding_mask_boxes = torch.zeros_like(
           box_labels, dtype=torch.bool
        )
        if self.obj_cls_embed == 'ag':
            src_key_padding_mask_boxes[torch.where(box_labels== 0)] = True

        # [Num. boxes, Batch size x Num. frames, Hidden size]
        layout_embeddings = self.transformer(
            src=cb_embeddings,
            src_key_padding_mask=src_key_padding_mask_boxes.flatten(0,1),
        )
        # [Batch size x Num. frames, Num. boxes, Hidden size]
        layout_embeddings = layout_embeddings.transpose(0, 1)
        # [Batch size, Num. frames, Num. boxes, Hidden_size]
        layout_embeddings = layout_embeddings.view(
            -1, num_frames, num_boxes, hidden_size
        )
        # [Batch size, Num. frames, Hidden size]
        layout_embeddings = layout_embeddings

        return layout_embeddings
    
    
class FramesEmbeddings(nn.Module):
    def __init__(self, object_cls_embed,
                 add_frame_emb=False,
                 n_objs=6,
                 use_masks=False,
                 use_boxes=True,
                 use_flows=False,
                 mask_sz = 224):
        super(FramesEmbeddings, self).__init__()
        self.layout_embedding = SpatialTransformer(object_cls_embed,
                                                   n_objs=n_objs,
                                                   use_masks=use_masks,
                                                   use_boxes = use_boxes,
                                                   use_flows = use_flows,
                                                   mask_sz = mask_sz)
        hidden_size = 768
        self.add_frame_emb = add_frame_emb
        self.position_embeddings = nn.Embedding(
            256, hidden_size
        )
        if add_frame_emb:
            self.frame_type_embedding = nn.Embedding(5, hidden_size, padding_idx=0)
            frame_types = torch.zeros(1,17).fill_(2).to(torch.int)
            frame_types[:,-1] = 4
            self.register_buffer("frame_types",frame_types)
            
        self.register_buffer(
            "position_ids", torch.arange(256).expand((1, -1))
            )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
    
    

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Batch size, Num. frames, Hidden size]
        layouts_embeddings = self.layout_embedding(batch)
        # [Batch size, Num. frames+1,  Hidden size]
        aggregated_layout_embeddings = layouts_embeddings[:, :, 0, :]
        # [Batch size, Num. frames, O, Hidden size]
        object_layout_embeddings =  layouts_embeddings[:, :-1, 1:, :]
       
        B,T,_ = aggregated_layout_embeddings.shape   
        position_embeddings = self.position_embeddings(
            self.position_ids[:, :T]
        )
        # Preparing everything together
        # Add position embeddings
        embeddings = aggregated_layout_embeddings + position_embeddings 
        if self.add_frame_emb:
            # Add frame type 
            frame_types_embeddings = self.frame_type_embedding(batch['frame_types'].to(embeddings.device))
            embeddings = embeddings + frame_types_embeddings
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings,object_layout_embeddings #[B,T,C], [B,T,O,C]
    
    
    
@MODEL_REGISTRY.register()
class StltBackbone(nn.Module):
    def __init__(self,object_cls_embed = 'id', 
                 add_frame_emb = False,
                 n_objs =6,
                 use_masks=False,
                 use_boxes=True,
                 use_flows=False,
                 mask_sz = 224):
        super(StltBackbone, self).__init__()
        
        hidden_size = 768
        self.frames_embeddings = FramesEmbeddings(object_cls_embed, 
                                                  add_frame_emb=add_frame_emb,
                                                  n_objs=n_objs,
                                                  use_masks = use_masks,
                                                  use_boxes = use_boxes,
                                                  use_flows = use_flows,
                                                  mask_sz = mask_sz)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=8
        )
        

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Batch size, Num. frames, Hidden size]
        frames_embeddings,object_layout_embeddings = self.frames_embeddings(batch)
        # [Num. frames, Batch size, Hidden size]
        frames_embeddings = frames_embeddings.transpose(0, 1)
        B = frames_embeddings.shape[1]
        # [Num. frames, Num. frames]
        
        src_key_padding_mask_frames = None
        causal_mask_frames = None
        # [Num. frames, Batch size, Hidden size]
        transformer_output = self.transformer(
            src=frames_embeddings,
            mask=causal_mask_frames,
            src_key_padding_mask=src_key_padding_mask_frames,
        )

        return transformer_output,object_layout_embeddings #[T,B,C], [B,T,O,C]
    


