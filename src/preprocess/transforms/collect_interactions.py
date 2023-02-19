from preprocess.utils.util import connect
from preprocess.utils.util import get_mentions_by_user, get_replies_by_user, get_quotes_by_user
from preprocess.utils.util import get_mentions_by_friends, get_replies_by_friends, get_quotes_by_friends

def collect_interactions(client, config, group_type):
    helper, csv_dir, csv_location = connect(client, config, group_type, 'friends')

    # collection
    user_tweets_collection = helper.user_tweets
    friends_collection = helper.friends
    friend_tweets_collection = helper.friend_tweets
    user_interactions_collection = helper.user_interactions

    users = helper.find_users_if_friends_exists(return_list = True)
    i = 0
    for user in users:
        i += 1
        user_id = user['user_id']
        friends = user['friends']
        username = user['username']
        j = 0
        last_tweet_timestamp = helper.find_user_diagnosis_time(user_id)
        for friend_id in friends:
            j += 1
            friend_username = friends_collection.find_one({'user_id': friend_id}, {'username': 1})
            if friend_username == None: continue
            doc = user_interactions_collection.find_one({'user_id': user_id, 'friend_id': friend_id})
            if doc != None: continue

            friend_username = friend_username['username']
            u1 = get_mentions_by_user(friend_tweets_collection, user_id, friend_id, last_tweet_timestamp)
            u1 = get_mentions_by_user(user_tweets_collection, user_id, friend_id, last_tweet_timestamp)
            u2 = get_replies_by_user(user_tweets_collection, user_id, friend_id, last_tweet_timestamp)
            u3 = get_quotes_by_user(user_tweets_collection, user_id, friend_username, last_tweet_timestamp)
            f1 = get_mentions_by_friends(friend_tweets_collection, user_id, friend_id, last_tweet_timestamp)
            f2 = get_replies_by_friends(friend_tweets_collection, user_id, friend_id, last_tweet_timestamp)
            f3 = get_quotes_by_friends(friend_tweets_collection, user_id, username, friend_id, last_tweet_timestamp)
            u1, u2, u3 = list(u1), list(u2), list(u3)
            f1, f2, f3 = list(f1), list(f2), list(f3)
            u1 = [tweet['id'] for tweet in u1]
            u2 = [tweet['id'] for tweet in u2]
            u3 = [tweet['id'] for tweet in u3]
            f1 = [tweet['id'] for tweet in f1]
            f2 = [tweet['id'] for tweet in f2]
            f3 = [tweet['id'] for tweet in f3]
            

            num_interactions = len(u1) + len(u2) + len(u3) + len(f1) + len(f2) + len(f3)
            
            update = {}
            if num_interactions > 0:
                print(i, user_id, friend_id, num_interactions)
                update['total_interactions'] = num_interactions
                if len(u1) > 0: update['mentions'] = u1
                if len(u2) > 0: update['replies'] = u2
                if len(u3) > 0: update['quotes'] = u3
                if len(f1) > 0: update['mentioned'] = f1
                if len(f2) > 0: update['replied'] = f2
                if len(f3) > 0: update['quoted'] = f3
                print(f'{i}-{j} ({num_interactions}) {len(u1)} {len(u2)} {len(u3)} {len(f1)} {len(f2)} {len(f3)}')
            
            user_interactions_collection.update_one({'user_id': user_id, 'friend_id': friend_id}, {'$set': update}, upsert = True)
    