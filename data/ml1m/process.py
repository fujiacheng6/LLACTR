import csv
import re
import random
from collections import defaultdict

ratings = './ml-1m/ratings.dat'
users = './ml-1m/users.dat'
movies = './ml-1m/movies.dat'

# Load user info
user_info = {}
gender_map = {'F': 'female', 'M': 'male'}
age_map = {'1': 'Under 18', '18': '18-24', '25': '25-34', '35': '35-44', '45': '45-49', '50': '50-55', '56': '56+'}
occupation_map = {
    '0': 'other or not specified', '1': 'academic/educator', '2': 'artist', '3': 'clerical/admin',
    '4': 'college/grad student', '5': 'customer service', '6': 'doctor/health care', '7': 'executive/managerial',
    '8': 'farmer', '9': 'homemaker', '10': 'K-12 student', '11': 'lawyer', '12': 'programmer',
    '13': 'retired', '14': 'sales/marketing', '15': 'scientist', '16': 'self-employed',
    '17': 'technician/engineer', '18': 'tradesman/craftsman', '19': 'unemployed', '20': 'writer'
}

with open(users, 'r') as infile:
    for line in infile:
        user_id, gender, age, occupation, _ = re.split('::', line)
        user_info[user_id] = {'Gender': gender_map.get(gender, ''), 'Age': age_map.get(age, ''), 'Occupation': occupation_map.get(occupation, '')}

# Load movie info
movie_info = {}

with open(movies, 'r', encoding='latin-1') as infile:
    for line in infile:
        movie_id, title, genres = re.match(r"(\d+)::(.*?)::(.*)", line).groups()
        movie_info[movie_id] = {'Title': title, 'Genres': genres}

# Load ratings data and enrich with user and movie info
data = []
with open(ratings, 'r') as infile:
    for line in infile:
        user_id, movie_id, rating, timestamp = line.strip().split('::')
        rating = int(rating)
        if rating > 3:
            rating = 1
        elif rating == 3: # drop
            continue            
        else:
            rating = 0
        user = user_info.get(user_id, {})
        movie = movie_info.get(movie_id, {})
        data.append({'UserID': user_id, 'Gender': user.get('Gender', ''), 'Age': user.get('Age', ''), 'Occupation': user.get('Occupation', ''),
                     'MovieID': movie_id, 'Title': movie.get('Title', ''), 'Genres': movie.get('Genres', ''),
                     'Rating': rating, 'Timestamp': timestamp})

# Split data
data = sorted(data, key=lambda x: int(x['Timestamp']))
total_samples = len(data)
train_samples = int(0.8 * total_samples)
valid_samples = int(0.1 * total_samples)
train_data = data[:train_samples]
valid_data = data[train_samples:train_samples + valid_samples]
test_data = data[train_samples + valid_samples:]

# Write to train.csv, valid.csv, test.csv
def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

write_to_csv(train_data, 'train.csv')
write_to_csv(valid_data, 'valid.csv')
write_to_csv(test_data, 'test.csv')

# Feature statistics
feature_counts = defaultdict(lambda: defaultdict(int))

for entry in data:
    for key, value in entry.items():
        feature_counts[key][value] += 1

for field, counts in feature_counts.items():
    total_features = sum(counts.values())
    print(f"{field}: {total_features}")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for feature, count in sorted_counts[:3]:
        print(f"   {feature}: {count}")

total_feature_count = sum(sum(counts.values()) for counts in feature_counts.values())
print("Total number of features:", total_feature_count)

# Additional statistics
unique_users = set()
unique_movies = set()
for entry in data:
    unique_users.add(entry['UserID'])
    unique_movies.add(entry['MovieID'])
user_num = len(unique_users)
item_num = len(unique_movies)
sparsity = total_samples / (user_num * item_num)
train_data_num = len(train_data)
valid_data_num = len(valid_data)
test_data_num = len(test_data)

print("all_data_num=", total_samples)
print("user_num=", user_num)
print("item_num=", item_num)
print("sparsity=", sparsity)
print("train_data_num=", train_data_num)
print("valid_data_num=", valid_data_num)
print("test_data_num=", test_data_num)


"""
UserID: 739012
   4169: 1556
   4277: 1463
   1680: 1443
Gender: 739012
   male: 555538
   female: 183474
Age: 739012
   25-34: 291269
   35-44: 146013
   18-24: 135935
Occupation: 739012
   college/grad student: 97913
   other or not specified: 95005
   executive/managerial: 77371
MovieID: 739012
   2858: 3070
   260: 2703
   1196: 2615
Title: 739012
   American Beauty (1999): 3070
   Star Wars: Episode IV - A New Hope (1977): 2703
   Star Wars: Episode V - The Empire Strikes Back (1980): 2615
Genres: 739012
   Comedy: 84912
   Drama: 84730
   Comedy|Drama: 31842
Rating: 739012
   1: 575281
   0: 163731
Timestamp: 739012
   975528402: 25
   975528243: 24
   975280276: 23
Total number of features: 6651108
all_data_num= 739012
user_num= 6040
item_num= 3668
sparsity= 0.033356864812554614
train_data_num= 591209
valid_data_num= 73901
test_data_num= 73902
"""
