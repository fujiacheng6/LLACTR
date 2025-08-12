# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

""" This is the implementation of the following paper:
    [WWW2021] FM2: Field-matrixed Factorization Machines for Recommender Systems
"""
import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, LR_Layer


class FmFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FmFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10, 
                 regularizer=None, 
                 field_interaction_type="matrixed",
                 **kwargs):
        super(FmFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs)
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.interact_dim = int(self.num_fields * (self.num_fields - 1) / 2)
        self.field_interaction_type = field_interaction_type

        if self.field_interaction_type == "vectorized":
            self.interaction_weight = nn.Parameter(torch.Tensor(self.interact_dim, embedding_dim)) # [C(N,2), D]
        elif self.field_interaction_type == "matrixed":
            self.interaction_weight = nn.Parameter(torch.Tensor(self.interact_dim, embedding_dim, embedding_dim)) # [C(N,2), D, D]
        nn.init.xavier_normal_(self.interaction_weight)

        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=False)
        self.upper_triange_mask = torch.triu(torch.ones(self.num_fields, self.num_fields - 1), 0).byte().bool().to(self.device)
        self.lower_triange_mask = torch.tril(torch.ones(self.num_fields, self.num_fields - 1), -1).byte().bool().to(self.device)
       
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)
        
        # new
        self.fc_4096_32 = nn.Linear(4096, 32)
        self.fm_mask = torch.triu(torch.ones(self.nlp_fields, self.nlp_fields), diagonal=1).bool().to(self.device)
        C_N_2 = int(self.nlp_fields * (self.nlp_fields-1) / 2)
        self.sim_weight = nn.Linear(C_N_2, C_N_2)

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X) # [B, N, D]

        field_wise_emb = feature_emb.unsqueeze(2).expand(-1, -1, self.num_fields - 1, -1) # [B, N, N-1, D]
        
        upper_tensor = torch.masked_select(field_wise_emb, self.upper_triange_mask.unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.embedding_dim) # [B, C(N,2), D]         
  
        if self.field_interaction_type == "vectorized":
            upper_tensor = upper_tensor * self.interaction_weight
        elif self.field_interaction_type == "matrixed":
            upper_tensor = torch.matmul(upper_tensor.unsqueeze(2), self.interaction_weight).squeeze(2) # [B, C(N,2), 1, D] * [C(N,2), D, D] --> [B, C(N,2), 1, D] 

        lower_tensor = torch.masked_select(field_wise_emb.transpose(1, 2), self.lower_triange_mask.t().unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.embedding_dim) # [B, C(N,2), D]

        y_pred = (upper_tensor * lower_tensor).flatten(start_dim=1).sum(dim=-1, keepdim=True) # [B, 1]

        y_pred += self.lr_layer(X)
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict    

    def forward_llm(self, inputs, lambda_loss=0, llm_emb2=None, lambda_llm=0, temperature=1, nlp_fields=0):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X) # [B, N, D]

        field_wise_emb = feature_emb.unsqueeze(2).expand(-1, -1, self.num_fields - 1, -1) # [B, N, N-1, D]        
        upper_tensor = torch.masked_select(field_wise_emb, self.upper_triange_mask.unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.embedding_dim) # [B, C(N,2), D] 
        if self.field_interaction_type == "vectorized":
            upper_tensor = upper_tensor * self.interaction_weight
        elif self.field_interaction_type == "matrixed":
            upper_tensor = torch.matmul(upper_tensor.unsqueeze(2), self.interaction_weight).squeeze(2) # [B, C(N,2), 1, D] * [C(N,2), D, D] --> [B, C(N,2), 1, D]        
        lower_tensor = torch.masked_select(field_wise_emb.transpose(1, 2), self.lower_triange_mask.t().unsqueeze(-1)) \
                            .view(-1, self.interact_dim, self.embedding_dim) # [B, C(N,2), D]        
        
        y_pred = (upper_tensor * lower_tensor).flatten(start_dim=1).sum(dim=-1, keepdim=True) # [B, C(N,2), D] -> [B, 1]
        y_pred += self.lr_layer(X)

        # field_fm
        llm_emb2_norm = llm_emb2 / torch.norm(llm_emb2, p=2, dim=-1, keepdim=True) # [N, D_llm]
        ip_llm = torch.matmul(llm_emb2_norm, llm_emb2_norm.transpose(0, 1)) # [N, N]
        ip_llm_masked = ip_llm[self.fm_mask] # [C(N, 2)]
        ip_llm_masked = self.sim_weight(ip_llm_masked) # [C(N, 2)]

        ip_ctr = torch.bmm(feature_emb, feature_emb.transpose(1, 2)) # [B, N, N]
        ctr_fm_mask = self.fm_mask.unsqueeze(0).repeat(feature_emb.size(0), 1, 1) # [B, N, N]
        ip_ctr_masked = ip_ctr[ctr_fm_mask].view(feature_emb.size(0),-1) # [B, C(N, 2)]

        y_pred_llm = (ip_ctr_masked * ip_llm_masked).sum(-1, keepdim=True)  # [B, 1]        

        y_pred = y_pred + lambda_llm * y_pred_llm # [B, 1] 
        y_pred = torch.sigmoid(y_pred) # [B, 1]
        loss = self.loss_with_reg(y_pred, y)

        # kl
        field_emb = self.fc_4096_32(llm_emb2) # [N, D]
        field_emb_resize = field_emb.unsqueeze(0).repeat(feature_emb.size(0), 1, 1) # [B, N, D]
        predict_probs = torch.softmax(feature_emb, dim=-1) # [B, N, D]
        target_probs = torch.softmax(field_emb_resize, dim=-1) # [B, N, D]
        kl_loss = nn.functional.kl_div(torch.log(predict_probs), target_probs, reduction='batchmean')
        fin_loss = loss + kl_loss * lambda_loss
        return_dict = {"loss": fin_loss, "y_pred": y_pred}       
        return return_dict

