import glob, os
import pandas as pd
from preprocess.utils.util import connect, process_user

def users_final_tweet_push_to_mongo(client, config, group_type):
    collections, csv_dir, csv_location = connect(client, config, group_type, 'users_final_tweet')
    user_tweets_collection = collections.user_tweet_prior_diagnose
    for file in sorted(glob.glob(csv_location, recursive = True)):

        if not "_tweet_prior_diagnose" in os.path.basename(file):
            # print("SKIPPING: ", os.path.basename(file))
            continue
        else:
            # print("PUSHING: ", os.path.basename(file))
            # df = pandas.read_csv(file)
            # tweets_json = json.loads(df.to_json(orient="index"))

            user_id = int(os.path.splitext(os.path.basename(file.replace('_tweet_prior_diagnose', '')))[0])

            user_exist = user_tweets_collection.count_documents({'user_id': user_id}, limit = 1)
            usecols = ['id', 'conversation_id', 'created_at', 'user_id', 'username', 'name', 'tweet', 'mentions', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'link', 'quote_url', 'reply_to']
            df = pd.read_csv(file, usecols = usecols)
            if user_exist:
                print("User ", user_id, " is already in mongo. SKIPPING.")
                continue

            print("Inserting: ",user_id)

            user_tweets = []

            for row in df.to_dict('records'):
                row = process_user(row)
                user_tweets.append(row)
                break # we only want the first row (latest tweet in case there are more)
            result = user_tweets_collection.insert_many(user_tweets)
            if result:
                # print(f"One user: {result.inserted_id}")
                print("Inserted: ",user_id)
            else:
                print("Failed inserting: ", user_id)
