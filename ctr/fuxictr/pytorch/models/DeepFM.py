# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, DNN_Layer, FM_Layer_v2 #, FM_Layer_v22, FM_Layer_v2_llm

class DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)", 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)

        self.fm_layer = FM_Layer_v2(feature_map, final_activation=None, use_bias=False)
        self.dnn = DNN_Layer(input_dim=embedding_dim * feature_map.num_fields,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             final_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True)

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
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs) # [B, fields] (float, not embedded), [B, 1]       
        feature_emb = self.embedding_layer(X) # [B, fields, dim]
        
        y_pred = self.fm_layer(X, feature_emb)
        y_pred += self.dnn(feature_emb.flatten(start_dim=1))

        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict

    def forward_llm(self, inputs, lambda_loss=0, llm_emb2=None, lambda_llm=0, temperature=1, nlp_fields=0):
        X, y = self.inputs_to_device(inputs) # [B, fields] (float, not embedded), [B, 1]       
        feature_emb = self.embedding_layer(X) # [B, fields, dim]
        y_pred = self.fm_layer(X, feature_emb)
        y_pred += self.dnn(feature_emb.flatten(start_dim=1))

        ### field_fm
        llm_emb2_norm = llm_emb2 / torch.norm(llm_emb2, p=2, dim=-1, keepdim=True) # [N, D_llm]
        ip_llm = torch.matmul(llm_emb2_norm, llm_emb2_norm.transpose(0, 1)) # [N, N]
        # print(ip_llm)
        # import pdb
        # pdb.set_trace()
        ip_llm_masked = ip_llm[self.fm_mask] # [C(N, 2)]
        ip_llm_masked = self.sim_weight(ip_llm_masked) # [C(N, 2)]

        ip_ctr = torch.bmm(feature_emb, feature_emb.transpose(1, 2)) # [B, N, N]
        ctr_fm_mask = self.fm_mask.unsqueeze(0).repeat(feature_emb.size(0), 1, 1) # [B, N, N]
        ip_ctr_masked = ip_ctr[ctr_fm_mask].view(feature_emb.size(0),-1) # [B, C(N, 2)]        
        y_pred_llm = (ip_ctr_masked * ip_llm_masked).sum(-1, keepdim=True)  # [B, 1]
        y_pred = y_pred + lambda_llm * y_pred_llm # [B, 1]

        y_pred = torch.sigmoid(y_pred) # [B, 1]
        loss = self.loss_with_reg(y_pred, y)

        ### kl
        field_emb = self.fc_4096_32(llm_emb2) # [N, D]
        field_emb_resize = field_emb.unsqueeze(0).repeat(feature_emb.size(0), 1, 1) # [B, N, D]
        predict_probs = torch.softmax(feature_emb, dim=-1) # [B, N, D]
        target_probs = torch.softmax(field_emb_resize, dim=-1) # [B, N, D]
        kl_loss = nn.functional.kl_div(torch.log(predict_probs), target_probs, reduction='batchmean')
        fin_loss = loss + kl_loss * lambda_loss
        return_dict = {"loss": fin_loss, "y_pred": y_pred}

        # return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict







