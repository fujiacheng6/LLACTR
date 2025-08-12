# ---------------------------------------------------------------------------------------------------------------------------
# pre process
# ---------------------------------------------------------------------------------------------------------------------------
import json
from tqdm import tqdm

meta_path = "./meta_Video_Games.jsonl"
review_path = "./Video_Games.jsonl"
print(meta_path)
print(review_path)

""" 
Item Metadata:
    main_category, title, average_rating, rating_number, features, description, price, images, videos, store, categories, details, parent_asin, bought_together
We use: (11)
    main_category, title, average_rating, rating_number, features, 
    description, price, store, categories, details, parent_asin

Reviews:
    rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase, helpful_vote
We use: (4)
    rating, parent_asin, user_id, timestamp
"""

# Load meta_data into item dictionary with 'parent_asin' as the key
meta_data_dict = {}
with open(meta_path, 'r') as meta_file:
    for line in tqdm(meta_file):
        meta_item = json.loads(line)
        meta_data_dict[meta_item['parent_asin']] = meta_item
print("meta data num:", len(meta_data_dict))

# Fill review data by searching “parent_asin” in meta_data
processed_list = []
user_set = set()
item_set = set()
field_list = [
    'parent_asin', 'user_id', 'timestamp',
    'main_category', 'title', 'average_rating', 'rating_number', 'features', 
    'description', 'price', 'store', 'categories', 'details',
]
feature_counts = {field: {} for field in field_list}

with open(review_path, 'r') as review_file:
    for line in tqdm(review_file):
        data = json.loads(line)
        # rating 1/0
        if data.get('rating') >= 4:
            rating_modified = 1
        else:
            rating_modified = 0
        # get review data
        row = [rating_modified, data.get('parent_asin'), data.get('user_id'), data.get('timestamp')]
        # get full data: 
        meta_item = meta_data_dict.get(data.get('parent_asin'))
        row.extend([meta_item.get('main_category'), meta_item.get('title'), meta_item.get('average_rating'), meta_item.get('rating_number'), meta_item.get('features'),
                    meta_item.get('description'), meta_item.get('price'), meta_item.get('store'), meta_item.get('categories'), meta_item.get('details')])
        processed_list.append(row)
        item_set.add(row[1])
        user_set.add(row[2])        
        for i, field in enumerate(field_list):
            feature = row[i+1]     
            feature = str(feature)
            if feature is None:
                feature = "null"
            if feature in feature_counts[field]:
                feature_counts[field][feature] += 1
            else:
                feature_counts[field][feature] = 1

print("data num:", len(processed_list))
print("user num:", len(user_set))
print("item num:", len(item_set))
print("sparsity:", len(processed_list) / (len(user_set) * len(item_set)))

for field, counts in feature_counts.items():
    total_features = sum(counts.values())
    print(f"{field}: {total_features}")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for feature, count in sorted_counts[:3]:
        print(f"   {feature}: {count}")
total_feature_count = sum(sum(counts.values()) for counts in feature_counts.values())
print("Total number of features:", total_feature_count)

"""
137269it [00:07, 18261.20it/s]
meta data num: 137269
4624615it [03:51, 20002.09it/s]
data num: 4624615
user num: 2766656
item num: 137249
sparsity: 1.2178988807984922e-05
parent_asin: 4624615
   B01N3ASPNV: 18105
   B0BN942894: 17310
   B077GG9D5D: 15594
user_id: 4624615
   AHJRJCJMK3XVV4BSPBRAHIYEODWA: 664
   AGMWACNMAG74AXBF7IJ22IOZSZPA: 596
   AGIBXD3LM6HNDWWRTIOJHB5EKNFA: 469
timestamp: 4624615
   1613240849000: 11
   1645462688476: 11
   1460556837000: 10
main_category: 4624615
   Video Games: 2880571
   Computers: 871348
   All Electronics: 541182
title: 4624615
   amFilm Tempered Glass Screen Protector for Nintendo Switch 2017 (2-Pack): 18105
   BENGOO Stereo Pro Gaming Headset for PS4, PC, Xbox One Controller, Noise Cancelling Over Ear Headphones with Mic, LED Light, Bass Surround, Soft Memory Earmuffs for Laptop Mac Wii Accessory Kits: 17310
   DualShock 4 Wireless Controller for PlayStation 4 - Jet Black: 15594
average_rating: 4624615
   4.6: 615991
   4.5: 605414
   4.4: 539273
rating_number: 4624615
   110368: 18105
   19: 17410
   2766: 17310
features: 4624615
   []: 443497
   ['Brand New in box. The product ships with all relevant accessories']: 38981
   ['Specifically designed for the 6.2-inch Nintendo Switch only, NOT for 7-inch Nintendo Switch OLED', 'Ultra-clear High Definition with 99.9% transparency to allow an optimal, natural viewing experience', 'Ultra thin-0.3mm thickness is reliable and resilient, and promises full compatibility with touchscreen sensitivity', 'Highly durable, and scratch resistant - surface hardness 9H and topped with oleophobic coating to reduce fingerprints.', 'Includes: 2x GLASS Screen Protector, Wet Wipes, Micro-Fiber Cleaning Cloth, Squeeze Card, Easy Installation Use Guide, Hinge Stickers']: 18105
description: 4624615
   []: 1347041
   ["The DualShock 4 Wireless Controller features familiar controls, and incorporates several innovative features to usher in a new era of interactive experiences. Its definitive analog sticks and trigger buttons have been improved for greater feel and sensitivity. A multi touch, clickable touch pad expands gameplay possibilities, while the incorporated light bar in conjunction with the PlayStation Camera allows for easy player identification and screen adjustment when playing with friends in the same room. The addition of the Share button makes utilizing the social capabilities of the PlayStation 4 as easy as the push of a button. The DualShock 4 Wireless Controller is more than a controller; it's your physical connection to a new era of gaming."]: 15594
   ['From the Manufacturer', 'PlayStation Plus connects you with the best online community of gamers. As the premium membership service for PS4, your subscription grants you access to the fastest multiplayer network, includes free games every month, and provides exclusive sales and benefits. With millions of gamers logging on every day to play, the best time to join PlayStation Plus is today!']: 13853
price: 4624615
   None: 1396611
   19.99: 101550
   29.99: 85706
store: 4624615
   Nintendo: 339229
   Electronic Arts: 211770
   PlayStation: 183502
categories: 4624615
   ['Video Games', 'PC', 'Games']: 319665
   ['Video Games', 'PC', 'Accessories', 'Headsets']: 244129
   ['Video Games', 'PC', 'Accessories', 'Gaming Mice']: 211915
details: 4624615
   {}: 88510
   {'Product Dimensions': '7.09"L x 3.94"W', 'Item Weight': '3.52 ounces', 'Item model number': 'FBA_Nintendo Switch 2017', 'Best Sellers Rank': {'Video Games': 89, 'Nintendo Switch Screen Protectors': 3}, 'Is Discontinued By Manufacturer': 'No', 'Date First Available': 'February 2, 2017', 'Manufacturer': 'amFilm', 'Brand': 'amFilm', 'Compatible Devices': 'Video Game', 'Material': 'Tempered Glass', 'Item Hardness': '9H', 'Compatible Phone Models': 'Nintendo Switch Oled', 'Special Feature': '9 H Surface Hardness, Scratch Resistant', 'Finish Type': 'Glossy', 'Unit Count': '2.0 Count', 'Screen Size': '6.2 Inches'}: 18105
   {'Brand': 'BENGOO', 'Series': 'G9000', 'Item model number': 'G9000', 'Hardware Platform': 'Gaming Console, PC, Nintendo 64', 'Item Weight': '15.5 ounces', 'Package Dimensions': '8.35 x 7.68 x 3.7 inches', 'Color': 'Red', 'Manufacturer': 'BENGOO', 'Date First Available': 'November 17, 2019', 'Best Sellers Rank': {'Video Games': 1481, 'Mac Game Headsets': 10, 'Xbox One Headsets': 23, 'PlayStation 4 Headsets': 35}, 'Model Name': 'G9000', 'Form Factor': 'Over Ear', 'Connectivity Technology': 'Wired'}: 17310
Total number of features: 60119995

train_data num: 3699692
valid_data num: 462461
test_data num: 462462
"""
# ---------------------------------------------------------------------------------------------------------------------------
# CTR data -- Split: 8:1:1
# ---------------------------------------------------------------------------------------------------------------------------
import csv
from multiprocessing import Pool
import os

keys = [
    'rating', 'parent_asin', 'user_id', 'timestamp', 
    'main_category', 'title', 'average_rating', 'rating_number', 'features', 
    'description', 'price', 'store', 'categories', 'details',
]

def write_csv(filename, header, data):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"{os.path.basename(filename)} num:", len(data))


sorted_data = sorted(processed_list, key=lambda x: int(x[3]))
total_samples = len(sorted_data)
train_samples = int(0.8 * total_samples)
valid_samples = int(0.1 * total_samples)

train_data = sorted_data[:train_samples]
valid_data = sorted_data[train_samples:train_samples + valid_samples]
test_data = sorted_data[train_samples + valid_samples:]


with Pool(processes=8) as pool:  
    pool.starmap(write_csv, [
        ('train.csv', keys, train_data),
        ('valid.csv', keys, valid_data),
        ('test.csv', keys, test_data)
    ])

