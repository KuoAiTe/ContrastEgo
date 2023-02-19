import glob, os
import pandas as pd
from pathlib import Path
from preprocess.utils.util import connect, get_userid_from_file_name, process_user, get_users_wtih_inconsistent_tweet_count


def friends_push_to_mongo(client, config, group_type):
    collections, csv_dir, csv_location = connect(client, config, group_type, 'friends')

    # collection
    friends_collection = collections.friends
    friend_tweets_collection = collections.friend_tweets


    incomplete_friend_ids, incomplete_friend_size = get_users_wtih_inconsistent_tweet_count(friends_collection)

    count = 0
    for file in glob.glob(csv_location, recursive = True):
        if "_tweet_prior_diagnose" in os.path.basename(file):
            continue

        friend_id = get_userid_from_file_name(file)
        if friend_id not in incomplete_friend_ids:
            continue
        if incomplete_friend_ids[friend_id] == 0:
            friends_collection.update_one({'user_id': friend_id}, {'$set': {'tweet_count': 0}})
            print('empty', friend_id)
            continue

        usecols = ['id', 'created_at', 'user_id', 'tweet', 'mentions', 'hashtags', 'quote_url', 'reply_to']

        df = pd.read_csv(file, usecols = usecols)
        print("Inserting Friend: ", friend_id)
        documents = set(friend_tweets_collection.find({'user_id': friend_id}, {'id': 1}).distinct('id'))
        user_tweets = []

        df = df[~df['id'].isin(documents)].drop_duplicates(subset=['id'])
        for row in df.to_dict('records'):
            if type(row['tweet']) != str:
                row['tweet'] = ''
            row = process_user(row)
            user_tweets.append(row)
        count += 1
        if len(user_tweets) > 0:
            result = friend_tweets_collection.insert_many(user_tweets)
        total_document_count = friend_tweets_collection.count_documents({'user_id': friend_id})
        friends_collection.update_one({'user_id': friend_id}, {'$set': {'tweet_count': total_document_count}}, upsert = True)
        print(f"# {count:06} / {incomplete_friend_size:06} -> Inserted - {friend_id:022}: total -> {incomplete_friend_ids[friend_id]:06} / {total_document_count:06}, #new inserted-> {len(user_tweets):06}")
