# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from torch.utils import data
import torch

class Dataset(data.Dataset):
    def __init__(self, darray):
        self.darray = darray        
    def __getitem__(self, index):
        X = self.darray[index, 0:-1]
        y = self.darray[index, -1]
        return X, y    
    def __len__(self):
        return self.darray.shape[0]
class DataGenerator(data.DataLoader):
    def __init__(self, data_array, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        self.dataset = Dataset(data_array)
        super(DataGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    def __len__(self):
        return int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))

## idx
class Dataset_idx(data.Dataset):
    def __init__(self, darray, idx_array):
        self.darray = darray 
        self.idx_array = idx_array      
    def __getitem__(self, index):
        X = self.darray[index, 0:-1]
        y = self.darray[index, -1]
        idx = self.idx_array[index]
        return X, y, idx    
    def __len__(self):
        return self.darray.shape[0]
class DataGenerator_idx(data.DataLoader):
    def __init__(self, data_array, idx_array, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        self.dataset = Dataset_idx(data_array, idx_array)
        super(DataGenerator_idx, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    def __len__(self):
        return int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))



class Dataset3(data.Dataset):
    def __init__(self, darray, nlp_fields=0):
        self.darray = darray
        self.nlp_fields = nlp_fields

    def __getitem__(self, index):
        nlp_fields = self.nlp_fields
        label_position = -(1+4096*nlp_fields)

        X = self.darray[index, 0:label_position]
        y = self.darray[index, label_position]
        
        llm_emb_np = self.darray[index, (label_position+1):]
        llm_emb = torch.tensor(llm_emb_np, dtype=torch.float32)
        llm_emb = llm_emb.view(-1, 4096)
        return X, y, llm_emb

    def __len__(self):
        return self.darray.shape[0]

class DataGenerator3(data.DataLoader):
    def __init__(self, data_array, batch_size=32, shuffle=False, num_workers=1, nlp_fields=0, **kwargs):
        self.nlp_fields = nlp_fields
        self.dataset = Dataset3(data_array, self.nlp_fields)
        super(DataGenerator3, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    def __len__(self):
        return int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))





class Dataset4(data.Dataset):
    def __init__(self, darray):
        self.darray = darray
        
    def __getitem__(self, index):
        X = self.darray[index, 0:-1]
        y = self.darray[index, -1]
        global_index = index  # 记录全局索引
        return X, y, global_index
    
    def __len__(self):
        return self.darray.shape[0]

class DataGenerator4(data.DataLoader):
    def __init__(self, data_array, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        self.dataset = Dataset4(data_array)
        self.batch_size = batch_size  # 添加batch_size属性
        super(DataGenerator4, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    
    def __iter__(self):
            batch_data = [[], [], []]
            for idx, (data, target, global_index) in enumerate(self.dataset):
                data = torch.from_numpy(data)  # 将 NumPy 数组转换为 PyTorch 张量
                target = torch.tensor(target)  # 将 Python 数字转换为 PyTorch 张量
                global_index = torch.tensor(global_index)  # 将 Python 数字转换为 PyTorch 张量
                
                batch_data[0].append(data)
                batch_data[1].append(target)
                batch_data[2].append(global_index)
                
                if len(batch_data[0]) == self.batch_size:
                    yield [torch.stack(batch_data[0]), torch.stack(batch_data[1]), torch.stack(batch_data[2])]
                    batch_data = [[], [], []]
            
            if len(batch_data[0]) > 0:
                yield [torch.stack(batch_data[0]), torch.stack(batch_data[1]), torch.stack(batch_data[2])]
        
    def __len__(self):
        return int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))



class Dataset5(data.Dataset):
    def __init__(self, darray, nlp_fields=0):
        self.darray = darray
        self.nlp_fields = nlp_fields
    def __getitem__(self, index):
        nlp_fields = self.nlp_fields
        label_position = -(1+nlp_fields)

        X = self.darray[index, 0:label_position]
        y = self.darray[index, label_position]
        
        idx_seq = self.darray[index, (label_position+1):]
        idx_seq = torch.tensor(idx_seq, dtype=torch.float32)
        return X, y, idx_seq
    def __len__(self):
        return self.darray.shape[0]

class DataGenerator5(data.DataLoader):
    def __init__(self, data_array, batch_size=32, shuffle=False, num_workers=1, nlp_fields=0, **kwargs):        
        self.nlp_fields = nlp_fields
        self.dataset = Dataset5(data_array, nlp_fields=nlp_fields)
        super(DataGenerator5, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)

    def __len__(self):
        return int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))



class Dataset_fm(data.Dataset):
    def __init__(self, darray, nlp_fields=0):
        self.darray = darray
        self.nlp_fields = nlp_fields

    def __getitem__(self, index):

        label_position = -2

        X = self.darray[index, 0:label_position]
        y = self.darray[index, label_position]
        
        llm_dot_sum = self.darray[index, -1]
        llm_dot_sum = torch.tensor(llm_dot_sum, dtype=torch.float32)
        llm_dot_sum = llm_dot_sum.unsqueeze(dim=-1)
        return X, y, llm_dot_sum

    def __len__(self):
        return self.darray.shape[0]

class DataGenerator_fm(data.DataLoader):
    def __init__(self, data_array, batch_size=32, shuffle=False, num_workers=1, nlp_fields=0, **kwargs):
        
        self.nlp_fields = nlp_fields
        self.dataset = Dataset_fm(data_array, nlp_fields=nlp_fields)
        super(DataGenerator_fm, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)

    def __len__(self):
        return int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))











        