# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v2, EmbeddingLayer_v3, InnerProductLayer_v2

class FwFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FwFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 regularizer=None, 
                 linear_type="FiLV", 
                 **kwargs):
        """ 
        linear_type: `LW`, `FeLV`, or `FiLV`
        """
        super(FwFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs) 
        interact_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2)
        self.interaction_weight_layer = nn.Linear(interact_dim, 1)
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields, output="dot_vector")
        self._linear_type = linear_type
        if linear_type == "LW":
            self.linear_weight_layer = EmbeddingLayer_v3(feature_map, 1)
        elif linear_type == "FeLV":
            self.linear_weight_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        elif linear_type == "FiLV":
            self.linear_weight_layer = nn.Linear(feature_map.num_fields * embedding_dim, 1, bias=False)
        else:
            raise NotImplementedError("linear_type={} is not supported.".format(linear_type))
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
        feature_emb = self.embedding_layer(X)
        interaction_vec = self.inner_product_layer(feature_emb)
        poly2_part = self.interaction_weight_layer(interaction_vec)
        if self._linear_type == "LW":
            linear_weights = self.linear_weight_layer(X)
            linear_part = linear_weights.sum(dim=1)
        elif self._linear_type == "FeLV":
            linear_weights = self.linear_weight_layer(X)
            linear_part = (feature_emb * linear_weights).sum((1, 2)).view(-1, 1)
        elif self._linear_type == "FiLV":
            linear_part = self.linear_weight_layer(feature_emb.flatten(start_dim=1))
        y_pred = poly2_part + linear_part # bias added in poly2_part
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict        
        
    def forward_llm(self, inputs, lambda_loss=0, llm_emb2=None, lambda_llm=0, temperature=1, nlp_fields=0):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        if self._linear_type == "LW":
            linear_weights = self.linear_weight_layer(X)
            linear_part = linear_weights.sum(dim=1)
        elif self._linear_type == "FeLV":
            linear_weights = self.linear_weight_layer(X)
            linear_part = (feature_emb * linear_weights).sum((1, 2)).view(-1, 1)
        elif self._linear_type == "FiLV":
            linear_part = self.linear_weight_layer(feature_emb.flatten(start_dim=1))               
        interaction_vec = self.inner_product_layer(feature_emb) # [B, C(N,2)]
        poly2_part = self.interaction_weight_layer(interaction_vec) # [B, 1]
        y_pred = poly2_part + linear_part # bias added in poly2_part

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

