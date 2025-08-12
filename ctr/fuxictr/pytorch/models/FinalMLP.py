import torch
from torch import nn
# from fuxictr.pytorch.models import BaseModel
# from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from .base_model import BaseModel
from ..layers import EmbeddingLayer_v3, DNN_Layer 

class FinalMLP(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FinalMLP",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 mlp1_hidden_units=[64, 64, 64],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[64, 64, 64],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0,
                 mlp2_batch_norm=False,
                 use_fs=True,
                 fs_hidden_units=[64],
                 fs1_context=[],
                 fs2_context=[],
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FinalMLP, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.embedding_layer = EmbeddingLayer_v3(feature_map, embedding_dim)
        feature_dim = embedding_dim * feature_map.num_fields
        self.mlp1 = DNN_Layer(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              final_activation=None,
                              dropout_rates=mlp1_dropout,
                              batch_norm=mlp1_batch_norm)
        self.mlp2 = DNN_Layer(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              final_activation=None,
                              dropout_rates=mlp2_dropout, 
                              batch_norm=mlp2_batch_norm)
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(feature_map, 
                                              feature_dim, 
                                              embedding_dim, 
                                              fs_hidden_units, 
                                              fs1_context,
                                              fs2_context)
        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        # self.reset_parameters()
        # self.model_to_device()

        # new
        self.fc_4096_32 = nn.Linear(4096, 32)
        self.fm_mask = torch.triu(torch.ones(self.nlp_fields, self.nlp_fields), diagonal=1).bool().to(self.device)
        C_N_2 = int(self.nlp_fields * (self.nlp_fields-1) / 2)
        self.sim_weight = nn.Linear(C_N_2, C_N_2)

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        # X = self.get_inputs(inputs)
        X, y = self.inputs_to_device(inputs)
        flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(X, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        # y_pred = self.final_activation(y_pred)
        # return_dict = {"y_pred": y_pred}
        y_pred = torch.sigmoid(y_pred) # [B, 1]
        loss = self.loss_with_reg(y_pred, y)  
        return_dict = {"loss": loss, "y_pred": y_pred}      
        return return_dict

    def forward_llm(self, inputs, lambda_loss=0, llm_emb2=None, lambda_llm=0, temperature=1, nlp_fields=0):
        X, y = self.inputs_to_device(inputs)
        # flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        feature_emb = self.embedding_layer(X)
        flat_emb = feature_emb.flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(X, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        
        llm_emb2_norm = llm_emb2 / torch.norm(llm_emb2, p=2, dim=-1, keepdim=True) # [N, D_llm]
        ip_llm = torch.matmul(llm_emb2_norm, llm_emb2_norm.transpose(0, 1)) # [N, N]
        ip_llm_masked = ip_llm[self.fm_mask] # [C(N, 2)]
        # trick for most dataset (except magazine)
        ip_llm_masked = self.sim_weight(ip_llm_masked) # [C(N, 2)]
        y_pred_llm = ip_llm_masked.sum(-1) # [1]  

        # fm
        # ip_ctr = torch.bmm(feature_emb, feature_emb.transpose(1, 2)) # [B, N, N]
        # ctr_fm_mask = self.fm_mask.unsqueeze(0).repeat(feature_emb.size(0), 1, 1) # [B, N, N]
        # ip_ctr_masked = ip_ctr[ctr_fm_mask].view(feature_emb.size(0),-1) # [B, C(N, 2)]
        # y_pred_llm = (ip_ctr_masked * ip_llm_masked).sum(-1, keepdim=True)  # [B, 1]   

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







class FeatureSelection(nn.Module):
    def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], 
                 fs1_context=[], fs2_context=[]):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_emb = EmbeddingLayer_v3(feature_map, embedding_dim,
                                                required_feature_columns=fs1_context)
        self.fs2_context = fs2_context
        if len(fs2_context) == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_emb = EmbeddingLayer_v3(feature_map, embedding_dim,
                                                required_feature_columns=fs2_context)
        self.fs1_gate = DNN_Layer(input_dim=embedding_dim * max(1, len(fs1_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  dropout_rates=0,
                                  final_activation="Sigmoid",
                                  batch_norm=False)
        self.fs2_gate = DNN_Layer(input_dim=embedding_dim * max(1, len(fs2_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  dropout_rates=0,
                                  final_activation="Sigmoid",
                                  batch_norm=False)

    def forward(self, X, flat_emb):
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output
