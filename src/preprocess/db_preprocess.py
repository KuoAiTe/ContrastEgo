if __name__ == "__main__":
    from preprocess.transforms.users_push_to_mongo import collect_user_and_tweet_count, users_push_to_mongo
    from preprocess.transforms.friends_push_to_mongo import friends_push_to_mongo
    from preprocess.transforms.users_final_tweet_push_to_mongo import users_final_tweet_push_to_mongo
    from preprocess.transforms.collect_interactions import collect_interactions
    from preprocess.transforms.collect_friends_tweet_count import collect_friends_tweet_count
    from preprocess.transforms.create_indices_for_collections import create_indices_for_collections
    from preprocess.configs.config import config
    from preprocess.utils.util import get_mongo_client

    client = get_mongo_client(config)

    for group_type in config.group_types:
        #create_indices_for_collections(client, config, group_type)
        continue
        collect_user_and_tweet_count(client, config, group_type)
        users_push_to_mongo(client, config, group_type)
        collect_friends_tweet_count(client, config, group_type)
        friends_push_to_mongo(client, config, group_type)
        users_final_tweet_push_to_mongo(client, config, group_type)
        collect_interactions(client, config, group_type)