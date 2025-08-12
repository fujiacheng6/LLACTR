# ---------------------------------------------------------------------------------------------------------------------------
# pre process
# ---------------------------------------------------------------------------------------------------------------------------
import json
from tqdm import tqdm

meta_path = "./meta_Digital_Music.jsonl"
review_path = "./Digital_Music.jsonl"
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
70537it [00:01, 40653.20it/s]
meta data num: 70537
130434it [00:03, 35727.62it/s]
data num: 130434
user num: 100952
item num: 70511
sparsity: 1.8323946352798762e-05
parent_asin: 130434
   B00003CXKT: 398
   5559166928: 319
   B001EJH4SW: 215
user_id: 130434
   AGAFM74L2RIJ5O36NNYH4Z5ISQNQ: 341
   AEDFM4VDH2MKYVBKGYVTU6R5L5FQ: 182
   AH3FC6V3IUJIN2Y7BCZ7DN3IMMJQ: 175
timestamp: 130434
   1628619966868: 7
   1408310605000: 6
   1512250261736: 6
main_category: 130434
   Digital Music: 130434
title: 130434
   ProdName: 521
   Greatest Hits: 432
   Test Big Data 1737: 398
average_rating: 130434
   5.0: 34929
   4.8: 18493
   4.7: 17734
rating_number: 130434
   1: 20857
   2: 12035
   3: 7851
features: 130434
   []: 130325
   ['Hanteo registered store. All sales count to charts.', '100% official sealed album']: 17
   ['A TRUSTED VINYL CLEANING SOLUTION: Inspired by the most popular vinyl record cleaning kit from the 1970s, GrooveWasher’s G2 High Tech Record Cleaning Fluid is safe for your beloved vinyl record collection. Thousands of seasoned audiophiles have relied on G2 cleaning fluid as their choice record cleaner for nearly a decade. You can trust us with your valuable vinyl record collection too!', 'FORMULATED TO SAFELY CLEAN VINYL RECORDS: G2 Cleaning Fluid is formulated with just five ingredients that are all hydrophilic and safe for PVC vinyl. State of the art detergents dissolve and suspend fingerprints and other oily grime. Special emulsifiers suspend dirt and dust particles in the fluid to be easily wiped away with one of GrooveWasher’s microfiber cleaning pads. Find yours in our GrooveWasher Vinyl Record Cleaning Kit.', 'TRIPLE TREATED PURIFIED WATER: The purified water carrier solution for all G2 Cleaning Fluid is triple treated. This water is laboratory grade, carbon filtered, double deionized and UV light treated to ensure every spray is clean and anti-static on contact. You won’t have to worry about any existing static build up on the record after cleaning your vinyl record collection with this professional grade record cleaner. The only thing you’ll have is crisp, clear, optimal sound.', 'WETTER THAN WATER: With some record cleaning solutions, you are getting water and cleaner. With Groove Washer’s G2 Cleaning Fluid, you are getting a solution wetter than water! Each bottle contains a super wetting agent that reduces the fluid’s surface tension to dive deep to the bottom of the microgrooves. When you spray G2 Fluid on a record you’ll see the difference as it spreads down into the grooves for a deep clean.', 'CLEAN RECORDS WITHOUT RINSING: Remove the extra step required by other record cleaners. G2 Record Cleaning Fluid is a no-rinse required vinyl record cleaner that leaves no residue behind! Try it. Compare it with your current favorite record cleaning fluid. If you are not completely satisfied, return it to us for a refund.']: 9
description: 130434
   []: 64738
   ['CD ALBUM']: 1530
   ['CD']: 551
price: 130434
   None: 48942
   19.99: 1429
   24.99: 976
store: 130434
   Format: Audio CD: 18433
   None: 6424
   BTS   Format: Audio CD: 1176
categories: 130434
   []: 130396
   ['Digital Music', 'Country']: 20
   ['Digital Music', 'Music By Price', '$5.00 to $5.99']: 8
details: 130434
   {}: 4282
   {'Date First Available': 'July 31, 2017'}: 489
   {'Date First Available': 'January 29, 2016'}: 403
Total number of features: 1695642
train_data num: 104347
valid_data num: 13043
test_data num: 13044
"""
# ---------------------------------------------------------------------------------------------------------------------------
# CTR data -- Split: 8:1:1
# ---------------------------------------------------------------------------------------------------------------------------
import csv

keys = [
    'rating', 'parent_asin', 'user_id', 'timestamp', 
    'main_category', 'title', 'average_rating', 'rating_number', 'features', 
    'description', 'price', 'store', 'categories', 'details',
]
sorted_data = sorted(processed_list, key=lambda x: int(x[3]))
total_samples = len(sorted_data)
train_samples = int(0.8 * total_samples)
valid_samples = int(0.1 * total_samples)

train_data = sorted_data[:train_samples]
valid_data = sorted_data[train_samples:train_samples + valid_samples]
test_data = sorted_data[train_samples + valid_samples:]

with open('train.csv', 'w', newline='', encoding='utf-8') as trainfile:
    writer = csv.writer(trainfile)
    writer.writerow(keys)
    for row in train_data:
        writer.writerow(row)
print("train_data num:", len(train_data)) 

with open('valid.csv', 'w', newline='', encoding='utf-8') as validfile:
    writer = csv.writer(validfile)
    writer.writerow(keys)
    for row in valid_data:
        writer.writerow(row)
print("valid_data num:", len(valid_data))

with open('test.csv', 'w', newline='', encoding='utf-8') as testfile:
    writer = csv.writer(testfile)
    writer.writerow(keys)
    for row in test_data:
        writer.writerow(row)
print("test_data num:", len(test_data))





