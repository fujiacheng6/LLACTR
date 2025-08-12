    # Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch import nn
from .embedding import EmbeddingLayer
from .interaction import InnerProductLayer, InnerProductLayer_v2
from itertools import combinations

class LR_Layer(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(LR_Layer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.final_activation = final_activation
        # A trick for quick one-hot encoding in LR (dim=1)
        self.embedding_layer = EmbeddingLayer(feature_map, 1)

    def forward(self, X):
        # print("self.embedding_layer = ", self.embedding_layer)
        # (UserID): Embedding(5401, 1, padding_idx=5400)
        # (Gender): Embedding(3, 1, padding_idx=2)
        # (Age): Embedding(8, 1, padding_idx=7)
        # (Occupation): Embedding(22, 1, padding_idx=21)
        # (MovieID): Embedding(3663, 1, padding_idx=3662)
        # (Title): Embedding(3663, 1, padding_idx=3662)
        # (Genres): Embedding(302, 1, padding_idx=301)
        embed_weights = self.embedding_layer(X) # list: [fields: [B, 1]]  <-- [B, fields]
        output = torch.stack(embed_weights).sum(dim=0)

        if self.bias is not None:
            output += self.bias
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output

class FM_Layer(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(FM_Layer, self).__init__()
        self.inner_product_layer = InnerProductLayer(output="sum")
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=use_bias)
        self.final_activation = final_activation
    def forward(self, X, feature_emb_list):
        lr_out = self.lr_layer(X)
        dot_out = self.inner_product_layer(feature_emb_list)
        output = dot_out + lr_out
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output

class FM_Layer_v2(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(FM_Layer_v2, self).__init__()
        self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields, output="sum")
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=use_bias)
        self.final_activation = final_activation
    def forward(self, X, feature_emb):
        lr_out = self.lr_layer(X)  
        dot_sum = self.inner_product_layer(feature_emb)
        output = dot_sum + lr_out # [B, 1]
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output

class FM_Layer_v2_ours(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(FM_Layer_v2, self).__init__()
        self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields, output="sum")
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=use_bias)
        self.final_activation = final_activation
    def forward(self, X, feature_emb):
        lr_out = self.lr_layer(X) # [B, 1] 
        dot_sum = self.inner_product_layer(feature_emb) # [B, 1]
        output = dot_sum + lr_out # [B, 1]
        return output

class FM_Layer_v2_llm(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(FM_Layer_v2_llm, self).__init__()
        self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields+1, output="sum")
        self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=use_bias)
        self.final_activation = final_activation
    def forward(self, X, feature_emb):
        lr_out = self.lr_layer(X) # [B, 1]  
        dot_sum = self.inner_product_layer(feature_emb)# [B, 1]
        output = dot_sum + lr_out # [B, 1]
        return output



class LR_Layer2(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(LR_Layer2, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        self.final_activation = final_activation
        # A trick for quick one-hot encoding in LR
        self.embedding_layer = EmbeddingLayer(feature_map, 1)

    def forward(self, X, llm_emb_1, lambda_llm = 0.1):
        embed_weights = self.embedding_layer(X)
        # print(f"embed_weights: {type(embed_weights)}, {len(embed_weights)}, {embed_weights[0].shape}") # embed_weights: <class 'list'>, 24, torch.Size([10000, 1])
        
        # embed_weights.append(llm_emb_1)
        # print(f"embed_weights: {type(embed_weights)}, {len(embed_weights)}, {embed_weights[0].shape}") # embed_weights: <class 'list'>, 25, torch.Size([10000, 1])
        
        lambda_llm = 0.1

        nlp_field_bool = [0, 1, 1, 1, 0, 1, 1, 0]
        nlp_field_bool_tensor = torch.tensor(nlp_field_bool)
        indices = torch.nonzero(nlp_field_bool_tensor).squeeze()

        embed_weights_tensor = torch.stack(embed_weights)
        j = 0
        for idx in indices:
            embed_weights_tensor[idx] = embed_weights_tensor[idx] + lambda_llm * llm_emb_1[:, j, :]
            j += 1
        embed_weights = list(embed_weights_tensor.unbind(dim=0))        

        output = torch.stack(embed_weights).sum(dim=0)
        if self.bias is not None:
            output += self.bias
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output


        # import pdb
        # pdb.set_trace()

                
        # nonzero_indices = torch.nonzero(ip)
        # ip_cn2 = ip[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].view(ip.size(0), -1) # [B, C(N, 2)]
        # ip_llm_cn2 = ip_llm_adaptor.unsqueeze(0).repeat(ip.size(0), 1, 1)[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].view(ip.size(0), -1) # [B, C(N, 2)]

        # ip_cn2_norm = ip_cn2 / torch.norm(ip_cn2, p=2, dim=-1, keepdim=True)
        # ip_llm_cn2_norm = ip_llm_cn2 / torch.norm(ip_llm_cn2, p=2, dim=-1, keepdim=True)
        # sim_metric = torch.matmul(ip_cn2_norm, ip_llm_cn2_norm.transpose(0, 1)) # (BXD) * (BXD) -> BXB
        # pos_score = torch.exp(torch.diag(sim_metric) / temperature)
        # neg_score = torch.sum(torch.exp(sim_metric / temperature), dim=-1)
        # loss_infoNCE = -torch.mean(torch.log(pos_score / neg_score))
        # fm_loss = loss_infoNCE * lambda_llm



                 
        ##### distribution constraint: (ctr fm <-> llm fm)       
        # nonzero_indices = torch.nonzero(ip)
        # ip_cn2 = ip[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].view(ip.size(0), -1) # [B, C(N, 2)]
        # ip_llm_cn2 = ip_llm.unsqueeze(0).repeat(ip.size(0), 1, 1)
        # ip_llm_cn2 = ip_llm_cn2[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].view(ip.size(0), -1)
        # ip_llm_adaptor_cn2 = ip_llm_adaptor.unsqueeze(0).repeat(ip.size(0), 1, 1)[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].view(ip.size(0), -1)
        # ctr_emb = feature_emb # [B, N, D]
        # llm_emb = llm_emb2.unsqueeze(0).repeat(ctr_emb.size(0), 1, 1) # [B, N, D_llm1]
        # llm_emb_adaptor = self.weight_4096_32(llm_emb) # [B, N, D]

        ### mse_loss (ip_llm, ip)
        # criterion = nn.MSELoss()
        # mse_loss = criterion(ip_llm_cn2, ip_cn2)        
        # llm_loss = mse_loss * lambda_llm


        ### kl_div (ip_llm, ip)
        # predict_probs = torch.softmax(ip_cn2, dim=-1)
        # target_probs = torch.softmax(ip_llm_cn2, dim=-1)
        # kl_loss = nn.functional.kl_div(torch.log(predict_probs), target_probs, reduction='batchmean')


        ### infonce_loss (ip_llm, ip)
        # sim_metric = torch.matmul(ip_cn2, ip_llm_adaptor_cn2.transpose(0, 1)) # (BXD) * (BXD) -> BXB
        # pos_score = torch.exp(torch.diag(sim_metric) / temperature)
        # neg_score = torch.sum(torch.exp(sim_metric / temperature), dim=-1)
        # loss_infoNCE = -torch.mean(torch.log(pos_score / neg_score))
        # llm_loss = loss_infoNCE  * lambda_llm





        ### prompt
        # indices = self.gift[:5] # [TOP-K, 2]
        # rank_weight = torch.exp(- (torch.arange(1, len(indices)+1) / 1.0).float() ) # [TOP-K]  
        # for i, idx in enumerate(indices):
        #     dot_sum = dot_sum + ip[:, idx[0], idx[1]].unsqueeze(-1) * rank_weight[i]  * temperature
      
        ## mask to get a ranking list, whose length is n*(n-1)/2
        # if i!=-1 and j!=-1:
        #     ip_ij = (feature_emb[:,i,:] * feature_emb[:,j,:]).sum(dim=-1, keepdim=True)
        #     dot_sum = dot_sum + ip_ij







# class FM_Layer_v2_llm(nn.Module):
#     def __init__(self, feature_map, final_activation=None, use_bias=True, nlp_field=None):
#         super(FM_Layer_v2_llm, self).__init__()
#         self.inner_product_layer = InnerProductLayer_v2(feature_map.num_fields, output="sum")
#         self.inner_product_layer_llm = InnerProductLayer_v2(nlp_field, output="sum")
#         self.lr_layer = LR_Layer(feature_map, final_activation=None, use_bias=use_bias)
#         self.final_activation = final_activation

#         self.nlp_field = nlp_field
        
#         self.linear = nn.Linear(self.nlp_field*4096, 1)
        
#     def forward(self, X, feature_emb, llm_emb, lambda_llm):
#         self.mask = torch.tril(torch.ones(self.nlp_field, self.nlp_field), diagonal=-1).unsqueeze(0).to(feature_emb.device)

#         lr_out_org = self.lr_layer(X)
#         # llm_emb_flaten = llm_emb.view(llm_emb.size(0), -1)
#         # lr_out_llm = self.linear(llm_emb_flaten)
#         # lr_out = lambda_llm * lr_out_llm + (1-lambda_llm) * lr_out_org
#         lr_out = lr_out_org

#         dot_sum = self.inner_product_layer(feature_emb)
#         dot_sum_llm = self.inner_product_layer_llm(llm_emb)

#         # save id
#         indices_to_keep = [i for i in range(self.nlp_field) if i not in [0, 4]]
#         feature_emb_woid = feature_emb[:, indices_to_keep, :]
#         dot_sum_woid = self.inner_product_layer(feature_emb_woid)
#         print("dot_sum_woid=", dot_sum_woid) # [B, 1]

#         # cos similarity
#         dot_product = torch.bmm(llm_emb, llm_emb.transpose(1, 2))
#         norms = torch.norm(llm_emb, dim=2, keepdim=True)
#         norms = torch.bmm(norms, norms.transpose(1, 2)) 
#         cos_sim = dot_product / norms

#         mask = self.mask.expand(cos_sim.size(0), self.nlp_field, self.nlp_field) 
#         dot_sum_llm_cos = torch.sum(cos_sim * mask, dim=(1, 2), keepdim=True).squeeze(-1)

#         dot_sum = (dot_sum - dot_sum_woid) + lambda_llm * dot_sum_llm + (1-lambda_llm) * dot_sum_woid
#         output = dot_sum + lr_out

#         print("self.final_activation=", self.final_activation)
        
#         if self.final_activation is not None:
#             output = self.final_activation(output)
#         return output



class FM_Layer_v22(nn.Module):
    def __init__(self, feature_map, final_activation=None, use_bias=True):
        super(FM_Layer_v22, self).__init__()
        self.inner_product_layer2 = InnerProductLayer_v2(feature_map.num_fields+1, output="sum")
        self.lr_layer2 = LR_Layer2(feature_map, final_activation=None, use_bias=use_bias)
        self.final_activation = final_activation

    def forward(self, X, feature_emb, llm_emb_1):
        lr_out = self.lr_layer2(X, llm_emb_1)
        dot_sum = self.inner_product_layer2(feature_emb)
        output = dot_sum + lr_out
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output

