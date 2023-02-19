from preprocess.utils.util import connect
def create_indices_for_collections(client, config, group_type):
    collections, csv_dir, csv_location = connect(client, config, group_type)

    # collection
    user_collection = collections.users
    user_collection.create_index('user_id', unique = True)
    user_collection.create_index('username', unique = True)

    user_tweets_collection = collections.user_tweets
    user_tweets_collection.create_index('id', unique = True)
    user_tweets_collection.create_index('user_id', unique = False)
    user_tweets_collection.create_index('created_at', unique = False)
    user_tweets_collection.create_index('marked', unique = False)
    user_tweets_collection.create_index( [("user_id", 1), ("marked", 1)], unique = False)

    friends_collection = collections.friends
    #friends_collection.find({'username': {'$exists': True}})
    friends_collection.create_index('user_id', unique = True)
    friends_collection.create_index('username', unique = True)

    friend_tweets_collection = collections.friend_tweets
    friend_tweets_collection.create_index('id', unique = True)
    friend_tweets_collection.create_index('user_id', unique = False)
    friend_tweets_collection.create_index('created_at', unique = False)
    friend_tweets_collection.create_index('marked', unique = False)
    friend_tweets_collection.create_index( [("user_id", 1), ("marked", 1)], unique = False)

    user_interactions_collection = collections.user_interactions
    user_interactions_collection.create_index('user_id', unique = False)
    user_interactions_collection.create_index('friend_id', unique = False)
    user_interactions_collection.create_index( [("user_id", 1), ("friend_id", 1)], unique = True)

    period_interaction_cache = collections.period_interaction_cache
    friends_collection.create_index('user_id', unique = False)
    friends_collection.create_index('friend_id', unique = False)
    user_interactions_collection.create_index( [("user_id", 1), ("friend_id", 1), ('period_start_date', 1), ('period_end_date', 1)], unique = True)
