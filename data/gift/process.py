# ---------------------------------------------------------------------------------------------------------------------------
# pre process
# ---------------------------------------------------------------------------------------------------------------------------
import json
from tqdm import tqdm

meta_path = "./meta_Gift_Cards.jsonl"
review_path = "./Gift_Cards.jsonl"
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
1137it [00:00, 58198.77it/s]
meta data num: 1137
152410it [00:03, 48022.53it/s]
data num: 152410
user num: 132732
item num: 1137
sparsity: 0.001009897646300382
parent_asin: 152410
   B00IX1I3G6: 36863
   B00ADR2LV6: 14912
   B077N4CNVJ: 6940
user_id: 152410
   AF4OCZTPFHXZKUTAGP6CAOCMWPAA: 55
   AHCSNG7KC3OXVLJBK3GQPB6NKNUQ: 32
   AHQP6NAQN7KH5FDNWW6OIZPLR6GQ: 27
timestamp: 152410
   1578363788140: 7
   1410793403000: 6
   1410793340000: 6
main_category: 152410
   Gift Cards: 103210
   None: 48814
   Software: 125
title: 152410
   Amazon Reload: 43505
   Amazon.com Gift Card in a Holiday Gift Box (Various Designs): 14912
   Amazon.com Gift Card in Various Gift Boxes: 6940
average_rating: 152410
   4.9: 72446
   4.7: 52388
   4.8: 20751
rating_number: 152410
   143309: 36863
   185606: 14912
   104005: 6940
features: 152410
   ['Add funds to your Amazon Gift Card balance, then use your balance to manage how much you spend while shopping.', 'Save up for a purchase by adding funds to your Gift Card balance.', 'Checkout faster when you reload to your Gift Card balance in advance.', 'Set up Auto-Reload to automatically reload to your Gift Card Balance on a particular date, week, month or when your balance gets low.', 'Reloaded funds never expire and have no fees.', 'Amazon Reload purchases are not refundable or redeemable for cash, except as required by law.']: 43505
   ['Gift Card is affixed inside a box', 'Gift amount may not be printed on Gift Cards', 'You can customize the gift amount as desired, for example $36, $54, $72, etc.', 'Gift Card has no fees and no expiration date', 'Gift Card is redeemable towards millions of items storewide at Amazon.com', 'Scan and redeem any Gift Card with a mobile or tablet device via the Amazon App', 'Free One-Day Shipping (where available)', 'Customized gift message, if chosen at check-out, only appears on packing slip and not on the actual gift card or carrier']: 14912
   ['Gift Card is affixed inside a gift box', 'Gift amount may not be printed on Gift Cards', 'Gift Card has no fees and no expiration date', 'No returns and no refunds on Gift Cards', 'Gift Card is redeemable towards millions of items storewide at Amazon.com', 'Scan and redeem any Gift Card with a mobile or tablet device via the Amazon App', 'Free One-Day Shipping (where available)', 'Customized gift message, if chosen at check-out, only appears on packing slip and not on the actual gift card or carrier']: 11646
description: 152410
   []: 86023
   ["Amazon.com Gift Cards are the perfect way to give them exactly what they're hoping for - even if you don't know what it is. Amazon.com Gift Cards are redeemable for millions of items across Amazon.com. Item delivered is a single physical Amazon.com Gift Card nested inside or with a free gift accessory."]: 30458
   ["Amazon.com Gift Cards are the perfect way to give them exactly what they're hoping for - even if you don't know what it is. Amazon.com Gift Cards are redeemable for millions of items across Amazon.com. Item delivered is a single physical Amazon.com Gift Card nested inside a tin Gift Box."]: 5124
price: 152410
   None: 69020
   25.0: 38289
   50.0: 13457
store: 152410
   Amazon: 116569
   Visa: 6017
   Starbucks: 2333
categories: 152410
   ['Gift Cards', 'Gift Card Recipients', 'For Him']: 66891
   ['Gift Cards', 'Occasions', 'Chanukah']: 14912
   ['Gift Cards', 'Gift Card Categories', 'Restaurants']: 12164
details: 152410
   {'Item model number': 'VariableDenomination', 'Date First Available': 'April 13, 2021', 'Manufacturer': 'Amazon'}: 43505
   {'Package Dimensions': '8.19 x 4.41 x 1.3 inches; 1.45 Ounces', 'Item model number': 'VariableDenomination', 'Date First Available': 'September 16, 2020', 'Manufacturer': 'Amazon'}: 14912
   {'Package Dimensions': '4.65 x 3.7 x 0.67 inches; 1.76 Ounces', 'Item model number': 'VariableDenomination', 'Date First Available': 'October 25, 2020', 'Manufacturer': 'Amazon'}: 6940
Total number of features: 1981330

valid.csv num: 15241
test.csv num: 15241
train.csv num: 121928
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


