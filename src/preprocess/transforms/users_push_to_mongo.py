import glob, os
import pandas as pd
from pathlib import Path
from preprocess.utils.util import connect, get_userid_from_file_name, get_users_wtih_inconsistent_tweet_count, get_users_wtih_exisiting_total_tweet_count, process_user

def collect_user_and_tweet_count(client, config, group_type):
    collections, csv_dir, csv_location = connect(client, config, group_type, 'users')
    user_collection = collections.users
    user_tweets_collection = collections.user_tweets
    users_wtih_total_tweets_exists = get_users_wtih_exisiting_total_tweet_count(user_collection)
    for file in sorted(glob.glob(csv_location, recursive = True)):
        if "_tweet_prior_diagnose" in os.path.basename(file):
            # print("SKIPPING: ", os.path.basename(file))
            continue
        else:
            # print("PUSHING: ", os.path.basename(file))
            # df = pandas.read_csv(file)
            # tweets_json = json.loads(df.to_json(orient="index"))

            user_id = get_userid_from_file_name(file)
            usecols = ['id', 'conversation_id', 'created_at', 'user_id', 'username', 'name', 'tweet', 'mentions', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'link', 'quote_url', 'reply_to']

            df = pd.read_csv(file, usecols = usecols)
            df = df[df['user_id'] == user_id].drop_duplicates(subset=['id'])
            if user_id in users_wtih_total_tweets_exists:

                print("collect_user_and_tweet_count", "User ", user_id, " is already in mongo. SKIPPING.")
                continue

            total_tweets = len(df.index)
            username = None
            if total_tweets > 0:
                username = df['username'].unique()[0]
            if username != None:
                document_count = int(user_tweets_collection.count_documents({'user_id': user_id}))
                print(f"Collect User {user_id} -> tweet count {document_count} / {total_tweets}")
                user_result = user_collection.update_one(
                    filter = {
                        'user_id': user_id,
                    },
                    update = {
                        "$set": {
                            'username': username,
                            'tweet_count': document_count,
                            'total_tweets': total_tweets,
                        }
                    },
                    upsert= True
                )

def users_push_to_mongo(client, config, group_type):
    collections, csv_dir, csv_location = connect(client, config, group_type, 'users')
    # collection
    user_collection = collections.users
    user_tweets_collection = collections.user_tweets
    users_wtih_inconsistent_tweet_count, users_wtih_inconsistent_tweet_count_size = get_users_wtih_inconsistent_tweet_count(user_collection)
    count = 0
    for file in sorted(glob.glob(csv_location, recursive = True)):
        if "_tweet_prior_diagnose" in os.path.basename(file):
            # print("SKIPPING: ", os.path.basename(file))
            continue
        else:
            count += 1
            user_id = get_userid_from_file_name(file)


            usecols = ['id', 'conversation_id', 'created_at', 'user_id', 'username', 'name', 'tweet', 'mentions', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'link', 'quote_url', 'reply_to']

            df = pd.read_csv(file, usecols = usecols)
            df = df[df['user_id'] == user_id].drop_duplicates(subset=['id'])
            if user_id not in users_wtih_inconsistent_tweet_count:
                continue

            documents = set(user_tweets_collection.find({'user_id': user_id}, {'id': 1}).distinct('id'))

            print("Inserting User: ", user_id)
            user_tweets = []
            df = df[~df['id'].isin(documents)]
            df = df.dropna(subset=['tweet'])
            for row in df.to_dict('records'):
                row = process_user(row)
                user_tweets.append(row)
            print(count, users_wtih_inconsistent_tweet_count_size)
            if len(user_tweets) > 0:
                result = user_tweets_collection.insert_many(user_tweets)
            
            document_count = int(user_tweets_collection.count_documents({'user_id': user_id}))
            if result:
                # print(f"One user: {result.inserted_id}")
                print(f'Inserted User: {user_id}, {len(user_tweets)} / {document_count} / total -> {users_wtih_inconsistent_tweet_count[user_id]:06} ')
                user_result = user_collection.update_one({'user_id': user_id}, {'$set': {'tweet_count': document_count}}, upsert = True)
            else:
                print("Failed inserting: ", user_id)
