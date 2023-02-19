import glob, os
import re
import pandas as pd
from preprocess.utils.util import connect, get_users_wtih_exisiting_total_tweet_count



"""
for file in glob.glob(csv_location, recursive = True):

    if count > 10:
        break
"""

#friend_mapping = {'1000033827849424897': ['1000150382197268483'], '1006591676427710465': ['1000150382197268483'], '1010358022214946816': ['1000150382197268483'], '1017485754996535296': ['1000150382197268483'], '1018953252870721536': ['1000150382197268483'], '1037148991283191808': ['1000150382197268483'], '1038462769379598336': ['1000150382197268483'], '1040000736271237120': ['1000150382197268483'], '1047269324124299265': ['1000150382197268483'], '1051388559003963392': ['1000150382197268483'], '1052132799187169280': ['1000150382197268483']}


def collect_friends_tweet_count(client, config, group_type):
    collections, csv_dir, csv_location = connect(client, config, group_type, 'friends')
    count = 0

    # collection
    users_collection = collections.users
    friends_collection = collections.friends
    friend_tweets_collection = collections.friend_tweets
    """
    from datetime import datetime
    from pymongo import UpdateOne

    d = []
    while True:
        d.clear()
        cursor = friend_tweets_collection.find({'created_at': {'$exists': False}}, {'timestamp': 1}, limit = 10000)

        count += 1
        print(count)
        for row in cursor:
            _id, timestamp = row['_id'], row['timestamp']
            datetime = datetime.fromtimestamp(timestamp)
            c = UpdateOne(
                    {'_id': _id},
                    {'$set': {'created_at': datetime}, '$unset': {'timestamp': 1}}
            )
            d.append(c)
        friend_tweets_collection.bulk_write(d)
    exit()
    """
    friend_mapping = {}


    cursor = friends_collection.find({'$expr': {'$ne': ['$tweet_count', '$total_tweets']}})
    incomplete_friend_ids = {}
    for doc in cursor:
        if 'user_id' in doc and 'total_tweets' in doc:
            incomplete_friend_ids[doc['user_id']] = doc['total_tweets']
    incomplete_friend_size = len(incomplete_friend_ids)

    # Make sure every csv contains the right file
    # One time only

    for file in glob.glob(csv_location, recursive = True):
        m = re.search(r'([0-9]+)\/([0-9]+)\.csv', file)
        if m != None:
            user_id, friend_id = m.groups()
            user_id = int(user_id)
            friend_id = int(friend_id)
            count += 1
            if friend_id not in friend_mapping:
                friend_mapping[friend_id] = []
            friend_mapping[friend_id].append(user_id)
    count = 0
    total = len(friend_mapping)

    users_wtih_exisiting_total_tweet_count = get_users_wtih_exisiting_total_tweet_count(friends_collection)

    for friend_id, user_ids in friend_mapping.items():
        count += 1
        tweet_ids = set()
        if friend_id in users_wtih_exisiting_total_tweet_count:
            continue
        username = None
        for user_id in user_ids:
            file_path = f'{csv_dir}/{user_id}/{friend_id}.csv'
            df = pd.read_csv(file_path, index_col = None, header = 0)
            # Makre sure every row is pertaining to the user.
            df = df[df['user_id'] == friend_id]
            unique_ids = df['id'].unique()
            if username == None and len(df.head(1)) != 0:
                username = str(df.head(1)['username'].values[0])
            tweet_ids.update(unique_ids)
        friend_tweet_count = len(tweet_ids)
        if username != None:
            friends_collection.update_one({'user_id': friend_id}, {'$set': {'total_tweets': friend_tweet_count, 'username': username}}, upsert = True)
            for user_id in user_ids:
                users_collection.update_one({'user_id': user_id}, {'$addToSet': {'friends': friend_id}}, upsert = True)
        print(count, total, friend_id, username, friend_tweet_count)
