# ---------------------------------------------------------------------------------------------------------------------------
# LLM data -- instruction
# ---------------------------------------------------------------------------------------------------------------------------
import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import os
random.seed(42)

def csv_to_json(input_path, output_path, sample_data=None, sample_feature=1000):
    folder_path = os.path.dirname(output_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data = pd.read_csv(input_path)
    # sample data
    if sample_data is not None:
        data = data.sample(n=sample_data, random_state=42).reset_index(drop=True)
        data.to_csv(output_path[:-5] + ".csv", index=False)
    # remove the rating column
    fields = [col for col in data.columns if col not in ['rating']]
    features = [data[field].unique() for field in fields]
    print("fields:", fields)
    # cutoff when feature_num > sample_feature
    dictionary = dict(zip(fields, features))
    for key, value in dictionary.items():
        if len(value) > sample_feature:
            dictionary[key] = random.sample(list(value), sample_feature)
        print(f"{key}: {len(dictionary[key])}")
    # use reverse_dictionary to generate instrution
    reverse_dictionary = {}
    for k, v in dictionary.items():
        for value in v:
            reverse_dictionary[value] = k
    reverse_list = list(reverse_dictionary.items())
    # instruction
    json_list = []
    for feature, field in reverse_list:
        # instruction
        instruction = f"Given a feature, please answer which field it belongs to. Please select the field name from the candidate set."
        # input
        feature_input = f"The feature is: {feature}."
        candidate_set = f"The candidate set is: ['parent_asin', 'user_id', 'timestamp', 'main_category', 'title', 'average_rating', 'rating_number', 'features', 'description', 'price', 'store', 'categories', 'details']."
        input_str = f"{feature_input}\n{candidate_set}\n"
        # output
        target_str = f"{field}"
        # final prompt
        json_list.append({
            "instruction": instruction,
            "input": input_str,
            "output": target_str,
        })
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)
        print("data num = ", len(json_list))
        print("-"*120)

csv_to_json('./train.csv', './instruction/train.json', sample_feature=1000)
csv_to_json('./valid.csv', './instruction/valid.json', sample_feature=100)
csv_to_json('./test.csv', './instruction/test.json', sample_feature=100)

"""
fields: ['parent_asin', 'user_id', 'timestamp', 'main_category', 'title', 'average_rating', 'rating_number', 'features', 'description', 'price', 'store', 'categories', 'details']
parent_asin: 1000
user_id: 1000
timestamp: 1000
main_category: 13
title: 773
average_rating: 30
rating_number: 506
features: 501
description: 378
price: 26
store: 391
categories: 41
details: 865
data num =  6492
------------------------------------------------------------------------------------------------------------------------
fields: ['parent_asin', 'user_id', 'timestamp', 'main_category', 'title', 'average_rating', 'rating_number', 'features', 'description', 'price', 'store', 'categories', 'details']
parent_asin: 100
user_id: 100
timestamp: 100
main_category: 5
title: 100
average_rating: 21
rating_number: 100
features: 100
description: 100
price: 26
store: 100
categories: 34
details: 100
data num =  983
------------------------------------------------------------------------------------------------------------------------
fields: ['parent_asin', 'user_id', 'timestamp', 'main_category', 'title', 'average_rating', 'rating_number', 'features', 'description', 'price', 'store', 'categories', 'details']
parent_asin: 100
user_id: 100
timestamp: 100
main_category: 8
title: 100
average_rating: 25
rating_number: 100
features: 100
description: 100
price: 26
store: 100
categories: 35
details: 100
data num =  987
"""