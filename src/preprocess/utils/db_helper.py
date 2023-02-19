import pytz
import dateutil
import ast
import os
import numpy as np
import pandas as pd
import emoji
from cleantext import clean
import csv
import re
import contractions
from pathlib import Path
from subprocess import check_output
from datetime import datetime

class MongoHelper:
    def __init__(self, db, config):
      self.users = db[config.collection_names.users]
      self.user_tweets = db[config.collection_names.user_tweets]
      self.friends = db[config.collection_names.friends]
      self.friend_tweets = db[config.collection_names.friend_tweets]
      self.user_tweet_prior_diagnose = db[config.collection_names.user_tweet_prior_diagnose]
      self.stats_mentions = db[config.collection_names.stats_mentions]
      self.stats_replies = db[config.collection_names.stats_replies]
      self.stats_quote_tweets = db[config.collection_names.stats_quote_tweets]
      self.user_interactions = db[config.collection_names.user_interactions]
      self.period_interaction_cache = db[config.collection_names.period_interaction_cache]
    def find_start_and_end_dates(self):
        earliest_post_in_user_tweets = self.user_tweets.find_one(sort=[("created_at", 1)])
        latest_post_in_user_tweets = self.user_tweets.find_one(sort=[("created_at", -1)])
        start_date = earliest_post_in_user_tweets['created_at'].replace(hour = 0, minute = 0, second = 0)
        end_date = latest_post_in_user_tweets['created_at'].replace(hour = 23, minute = 59, second = 59)
        return start_date, end_date

    def find_users_if_interaction_exists(self, return_list = True, **kwargs):
        result = self.user_interactions.find({
            'total_interactions': {
                '$exists': True
            }
        })
        if return_list: result = list(result)
        return result
    def find_all_friends(self, return_list = True, **kwargs):
        result = self.friends.find({})
        if return_list: result = list(result)
        return result
    def find_username_by_id(self, user_id):
        result = self.users.find_one({'user_id': user_id}, {'username'})
        return None if result == None else result['username']
    def find_friend_username_by_id(self, friend_id):
        result = self.friends.find_one({'user_id': friend_id}, {'username'})
        return None if result == None else result['username']

    def find_user_by_user_id(self, user_id):
        return self.users.find_one({'user_id': user_id})
    def find_users_if_friends_exists(self, return_list = True, **kwargs):
        result = self.users.find({'friends': {'$exists': True}}, **kwargs)
        if return_list: result = list(result)
        return result

    def find_user_diagnosis_time(self, user_id):
        user_doc = self.user_tweets.find_one({'user_id': user_id})
        if user_doc == None: return None
        diagnosis_time = user_doc['created_at']
        return diagnosis_time
    
    def find_user_tweets_by_user_id(self, user_id, return_list = True, **kwargs):
        result = self.user_tweets.find({'user_id': user_id})
        if return_list: result = list(result)
        return result
    def find_tweet_by_id(self, tweet_id):
        user_doc = self.user_tweets.find_one({'id': tweet_id})
        if user_doc == None: 
            user_doc = self.friend_tweets.find_one({'id': tweet_id})
        if user_doc == None: return None
        return user_doc
    
    def if_user_has_no_tweet_between_dates(self, user_id, start_date, end_date, **kwargs):
        key = f'{start_date}-{end_date}'
        result = self.users.find_one({'user_id': user_id, key: {'$exists': True, '$eq': False}})
        return True if result != None else False
    def if_friend_has_no_tweet_between_dates(self, friend_id, start_date, end_date, **kwargs):
        key = f'{start_date}-{end_date}'
        result = self.friends.find_one({'user_id': friend_id, key: {'$exists': True, '$eq': False}})
        return True if result != None else False
    def get_embedding_column_name(self, model):
        model_name = model.rsplit('/', 1)[-1]
        return f'embeddings_{model_name}'
    def find_user_tweet_count_by_period(self, user_id, period_start_date, period_end_date):
        column_name = f'{period_start_date}-{period_end_date}'
        result = self.users.find_one({'user_id': user_id})
        assert column_name in result, "count first!"
        return result[column_name]
    def find_friend_tweet_count_by_period(self, friend_id, period_start_date, period_end_date):
        column_name = f'{period_start_date}-{period_end_date}'
        result = self.friends.find_one({'user_id': friend_id})
        assert column_name in result, "count first!"
        return result[column_name]
    def update_user_tweet_count_by_period(self, user_id, period_start_date, period_end_date):
        user_tweets_in_period = self.find_user_tweets_between_dates(user_id, period_start_date, period_end_date, limit = 999999)
        tweet_count_in_period = len(user_tweets_in_period)
        column_name = f'{period_start_date}-{period_end_date}'
        result = self.users.update_one({'user_id': user_id}, {'$set': {column_name: tweet_count_in_period}})
        return tweet_count_in_period
    def update_friend_tweet_count_by_period(self, friend_id, period_start_date, period_end_date):
        friend_tweets_in_period = self.find_friend_tweets_between_dates(friend_id, period_start_date, period_end_date, limit = 999999)
        tweet_count_in_period = len(friend_tweets_in_period)
        column_name = f'{period_start_date}-{period_end_date}'
        print(f'{friend_id} {period_start_date} {period_end_date} {tweet_count_in_period}')
        result = self.friends.update_one({'user_id': friend_id}, {'$set': {column_name: tweet_count_in_period}})
        return tweet_count_in_period
    def update_tweet_interaction(self, user_id, friend_id, tweet_id_by_period, tweet_count_by_period, periods_in_months):
        doc = self.user_interactions.find_one({'user_id': user_id, 'friend_id': friend_id})
        id = doc['_id']
        column_name = f'slice_data_{periods_in_months}'
        
        result = self.user_interactions.update_one({'_id': id}, {'$set': {column_name: {'data': tweet_id_by_period, 'count': tweet_count_by_period}}})
    def find_period_interaction_cache(self, user_id, friend_id, period_start_date, period_end_date):

        result = self.period_interaction_cache.find_one({
            'user_id': user_id,
            'friend_id': friend_id,
            'period_start_date': period_start_date,
            'period_end_date': period_end_date
        })
        return result
    def update_period_interaction_cache(self, user_id, friend_id, period_start_date, period_end_date, mention_count, reply_count, quote_count, total_count):
        result = self.period_interaction_cache.update_one({
            'user_id': user_id,
            'friend_id': friend_id,
            'period_start_date': period_start_date,
            'period_end_date': period_end_date
        },{
            '$set': {
                'mention_count': mention_count,
                'reply_count': reply_count,
                'quote_count': quote_count,
                'total_count': total_count,
            }
        }, upsert = True)
    def update_tweet_embeddings(self, model, group, tweet_id, tweet_embedding):
        column_name = self.get_embedding_column_name(model)
        if group == 'user':
            result = self.user_tweets.update_one({'id': tweet_id}, {'$set': {column_name: tweet_embedding}})
        elif group == 'friend':
            result = self.friend_tweets.update_one({'id': tweet_id}, {'$set': {column_name: tweet_embedding}})
    def find_tweet_embeddings(self, model, group, tweet_id):
        column_name = self.get_embedding_column_name(model)
        if group == 'user':
            result = self.user_tweets.find_one({'id': tweet_id, column_name: {'$exists': True}}, {column_name: 1})
            return result[column_name] if result != None else None
        elif group == 'friend':
            result = self.friend_tweets.find_one({'id': tweet_id, column_name: {'$exists': True}}, {column_name: 1})
            return result[column_name] if result != None else None
        return None
    def find_user_tweets_between_dates(self, user_id, start_date, end_date, min_num_tweets, update = True, **kwargs):
        # only tweets before diagnosis
        user_diagnosis_time = self.find_user_diagnosis_time(user_id)
        if end_date > user_diagnosis_time: 
            end_date = user_diagnosis_time
        pipeline = [
            {
                "$match": {
                    'user_id': user_id,
                    'created_at': {'$gt': start_date, '$lt': end_date},
                    'marked': {
                        '$exists': False
                    },
                },
            },
            {
                "$limit": min_num_tweets
            }

        ]
        result = self.user_tweets.aggregate(pipeline)
        result = list(result)
        return result
    def mark_user_tweet(self, tweet_id):
        result = self.user_tweets.update_one({'id': tweet_id}, {'$set': {'marked': True}})
    def mark_friend_tweet(self, tweet_id):
        result = self.friend_tweets.update_one({'id': tweet_id}, {'$set': {'marked': True}})
    def find_user_tweets_between_dates_marked(self, user_id, start_date, end_date, min_num_tweets, update = True, **kwargs):
            # only tweets before diagnosis
            user_diagnosis_time = self.find_user_diagnosis_time(user_id)
            if end_date > user_diagnosis_time: 
                end_date = user_diagnosis_time
            pipeline = [
                {
                    "$match": {
                        'user_id': user_id,
                        'created_at': {'$gt': start_date, '$lt': end_date},
                        'marked': True,
                    },
                },

            ]
            result = self.user_tweets.aggregate(pipeline)
            result = list(result)[:min_num_tweets]
            return result
    def find_friend_tweets_between_dates(self, friend_id, start_date, end_date, min_num_tweets, **kwargs):
        key = f'{start_date}-{end_date}'
        pipeline = [
            {
                "$match": {
                    'user_id': friend_id,
                    'created_at': {'$gt': start_date, '$lt': end_date},
                    'marked': {
                        '$exists': False
                    },
                },
            },
            {
                "$limit": min_num_tweets
            }

        ]
        result = self.friend_tweets.aggregate(pipeline)
        result = list(result)
        return result
    
    def find_friend_tweets_between_dates_marked(self, friend_id, start_date, end_date, min_num_tweets, **kwargs):
        key = f'{start_date}-{end_date}'
        pipeline = [
            {
                "$match": {
                    'user_id': friend_id,
                    'created_at': {'$gt': start_date, '$lt': end_date},
                    'marked': True,
                },
            },

        ]
        result = self.friend_tweets.aggregate(pipeline)
        result = list(result)[:min_num_tweets]
        return result

    def find_friend_tweets_mention_user_between_dates(self, friend_id, user_id, start_date, end_date, return_list = True, **kwargs):
        result = self.friend_tweets.find(
            {
                'user_id': friend_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                'mentions': {
                    '$all': [user_id],
                    '$exists': True
                },
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'mentions': 5,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result

    def find_friend_tweets_reply_to_user_between_dates(self, friend_id, user_id, start_date, end_date, return_list = True, **kwargs):
        result = self.friend_tweets.find(
            {
                'user_id': friend_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                'reply_to': {
                    '$all': [user_id],
                    '$exists': True
                },
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'reply_to': 5,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result
    def find_friend_tweets_quote_user_between_dates(self, friend_id, username, start_date, end_date, return_list = True, **kwargs):
        result = self.friend_tweets.find(
            {
                'user_id': friend_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                'quote_tweet_username': username,
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'quote_tweet_username': 5,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result

    def find_user_interact_friend_between_dates(self, user_id, friend_id, friend_username, start_date, end_date, return_list = True, **kwargs):
        result = self.user_tweets.find(
            {
                'user_id': user_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                '$or': [
                    {
                        'mentions': {
                            '$all': [friend_id],
                            '$exists': True
                        }
                    },
                    {
                        'reply_to': {
                            '$all': [friend_id],
                            '$exists': True
                        }
                    },
                    {'quote_tweet_username': friend_username},
                ],
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'mentions': 5,
                'reply_to': 6,
                'quote_tweet_username': 7,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result

    def find_friend_interact_user_between_dates(self, friend_id, user_id, username, start_date, end_date, return_list = True, **kwargs):
        result = self.friend_tweets.find(
            {
                'user_id': friend_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                '$or': [
                    {
                        'mentions': {
                            '$all': [user_id],
                            '$exists': True
                        }
                    },
                    {
                        'reply_to': {
                            '$all': [user_id],
                            '$exists': True
                        }
                    },
                    {'quote_tweet_username': username},
                ],
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'mentions': 5,
                'reply_to': 6,
                'quote_tweet_username': 7,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result
    def find_user_tweets_mention_friend_between_dates(self, user_id, friend_id, start_date, end_date, return_list = True, **kwargs):
        result = self.user_tweets.find(
            {
                'user_id': user_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                'mentions': {
                    '$all': [friend_id],
                    '$exists': True
                },
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'mentions': 5,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result

    def find_user_tweets_reply_to_friend_between_dates(self, user_id, friend_id, start_date, end_date, return_list = True, **kwargs):
        result = self.user_tweets.find(
            {
                'user_id': user_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                'reply_to': {
                    '$all': [friend_id],
                    '$exists': True
                },
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'reply_to': 5,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result
    def find_user_tweets_quote_friend_between_dates(self, user_id, friend_username, start_date, end_date, return_list = True, **kwargs):
        result = self.user_tweets.find(
            {
                'user_id': user_id,
                'created_at': {'$gt': start_date, '$lt': end_date},
                'quote_tweet_username': friend_username,
            }, {
                'id': 1,
                'user_id': 2,
                'created_at': 3,
                'tweet': 4,
                'quote_tweet_username': 5,
            }, **kwargs
        )
        if return_list: result = list(result)
        return result
