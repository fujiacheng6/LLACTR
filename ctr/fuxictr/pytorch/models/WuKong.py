# =========================================================================
# Copyright (C) 2024. XiaoLongtao. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
""" This model implements the paper: Zhang et al., Wukong: Towards a Scaling Law for 
    Large-Scale Recommendation, Arxiv 2024.
    [PDF] https://arxiv.org/abs/2403.02545
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from fuxictr.pytorch.models import BaseModel
# from fuxictr.pytorch.layers import EmbeddingLayer_v3, DNN_Layer
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, DNN_Layer

class WuKong(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="WuKong",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_layers=3,
                 compression_dim=40,
                 mlp_hidden_units=[32,32],
                 fmb_units=[32,32],
                 fmb_dim=40,
                 project_dim=8,
                 dropout_rate=0.2,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(WuKong, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        self.interaction_layers = nn.ModuleList([
            WuKongLayer(feature_map.num_fields, embedding_dim, project_dim, fmb_units, fmb_dim, compression_dim,dropout_rate) for _ in range(num_layers)
            ])
        self.final_mlp = DNN_Layer(input_dim=feature_map.num_fields*embedding_dim,
                                   output_dim=1,
                                   hidden_units=mlp_hidden_units,
                                   hidden_activations='relu',
                                   final_activation=None)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        # self.reset_parameters()
        # self.model_to_device()

        # new
        self.fc_4096_32 = nn.Linear(4096, 32)
        self.fm_mask = torch.triu(torch.ones(self.nlp_fields, self.nlp_fields), diagonal=1).bool().to(self.device)
        C_N_2 = int(self.nlp_fields * (self.nlp_fields-1) / 2)
        self.sim_weight = nn.Linear(C_N_2, C_N_2)

    def forward(self, inputs):
        # X = self.get_inputs(inputs)
        X, y = self.inputs_to_device(inputs) 
        feature_emb = self.embedding_layer(X)
        for layer in self.interaction_layers:
            feature_emb = layer(feature_emb)
        y_pred = self.final_mlp(feature_emb)
        # y_pred = self.final_activation(y_pred)
        y_pred = torch.sigmoid(y_pred)
        # return_dict = {"y_pred": y_pred}
        # return return_dict
        loss = self.loss_with_reg(y_pred, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict 
        
    def forward_llm(self, inputs, lambda_loss=0, llm_emb2=None, lambda_llm=0, temperature=1, nlp_fields=0):
        X, y = self.inputs_to_device(inputs) 
        feature_emb = self.embedding_layer(X)

        llm_emb2_norm = llm_emb2 / torch.norm(llm_emb2, p=2, dim=-1, keepdim=True) # [N, D_llm]
        ip_llm = torch.matmul(llm_emb2_norm, llm_emb2_norm.transpose(0, 1)) # [N, N]
        ip_llm_masked = ip_llm[self.fm_mask] # [C(N, 2)]
        # trick for magazine & game 
        ip_llm_adaptor = self.sim_weight(ip_llm_masked) # [C(N, 2)]
        y_pred_llm = ip_llm_adaptor.sum(-1) # [1]  

        # fm
        # ip_ctr = torch.bmm(feature_emb, feature_emb.transpose(1, 2)) # [B, N, N]
        # ctr_fm_mask = self.fm_mask.unsqueeze(0).repeat(feature_emb.size(0), 1, 1) # [B, N, N]
        # ip_ctr_masked = ip_ctr[ctr_fm_mask].view(feature_emb.size(0),-1) # [B, C(N, 2)]
        # y_pred_llm = (ip_ctr_masked * ip_llm_masked).sum(-1, keepdim=True)  # [B, 1]  

        
        # kl
        field_emb = self.fc_4096_32(llm_emb2) # [N, D]
        field_emb_resize = field_emb.unsqueeze(0).repeat(feature_emb.size(0), 1, 1) # [B, N, D]
        predict_probs = torch.softmax(feature_emb, dim=-1) # [B, N, D]
        target_probs = torch.softmax(field_emb_resize, dim=-1) # [B, N, D]
        kl_loss = nn.functional.kl_div(torch.log(predict_probs), target_probs, reduction='batchmean')

        for layer in self.interaction_layers:
            feature_emb = layer(feature_emb)
        y_pred = self.final_mlp(feature_emb)



        y_pred = y_pred + lambda_llm * y_pred_llm # [B, 1]
        y_pred = torch.sigmoid(y_pred) # [B, 1]
        loss = self.loss_with_reg(y_pred, y)

        fin_loss = loss + kl_loss * lambda_loss
        return_dict = {"loss": fin_loss, "y_pred": y_pred} 
        return return_dict 






class FactorizationMachineBlock(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, project_dim=8):
        super(FactorizationMachineBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.project_dim = project_dim
        self.num_features = num_features
        self.projection_matrix = nn.Parameter(torch.randn(self.num_features, self.project_dim))
    
    def forward(self, x):
        batch_size = x.size(0)
        x_fm = x.view(batch_size, self.num_features, self.embedding_dim)
        projected = torch.matmul(x_fm.transpose(1, 2), self.projection_matrix)
        fm_matrix = torch.matmul(x_fm, projected)
        return fm_matrix.view(batch_size, -1)


class FMB(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, fmb_units=[32,32], fmb_dim=40, project_dim=8):
        super(FMB, self).__init__()
        self.fm_block = FactorizationMachineBlock(num_features, embedding_dim, project_dim)
        self.layer_norm = nn.LayerNorm(num_features * project_dim)
        model_layers = [nn.Linear(num_features * project_dim, fmb_units[0]), nn.ReLU()]
        for i in range(1, len(fmb_units)):
            model_layers.append(nn.Linear(fmb_units[i-1], fmb_units[i]))
            model_layers.append(nn.ReLU())
        model_layers.append(nn.Linear(fmb_units[-1], fmb_dim))
        self.mlp = nn.Sequential(*model_layers)
    
    def forward(self, x):
        y = self.fm_block(x)
        y = self.layer_norm(y)
        y = self.mlp(y)
        y = F.relu(y)
        return y


class LinearCompressionBlock(nn.Module):
    """ Linear Compression Block (LCB) """
    def __init__(self, num_features=14, embedding_dim=16, compressed_dim=8,dropout_rate=0.2):
        super(LinearCompressionBlock, self).__init__()
        self.linear = nn.Linear(num_features * embedding_dim, compressed_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        return self.dropout(self.linear(x.view(x.size(0), -1)))


class WuKongLayer(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, project_dim=4, fmb_units=[40,40,40], fmb_dim=40, compressed_dim=40, dropout_rate=0.2):
        super(WuKongLayer, self).__init__()
        self.fmb = FMB(num_features, embedding_dim, fmb_units, fmb_dim, project_dim)
        self.lcb = LinearCompressionBlock(num_features, embedding_dim, compressed_dim, dropout_rate)
        self.layer_norm = nn.LayerNorm(num_features * embedding_dim)
        self.transform = nn.Linear(fmb_dim + compressed_dim, num_features*embedding_dim)
    
    def forward(self, x):
        fmb_out = self.fmb(x)
        lcb_out = self.lcb(x)
        concat_out = torch.cat([fmb_out, lcb_out], dim=1)
        concat_out = self.transform(concat_out)
        add_norm_out = self.layer_norm(concat_out+x.view(x.size(0), -1))
        return add_norm_out