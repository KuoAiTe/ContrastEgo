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
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
from preprocess.utils.db_helper import MongoHelper

def get_mongo_client(config):
    client = MongoClient(host = config.raw_db.host, port = config.raw_db.port)
    return client
def connect(client, config, group_type, data = None):
    #Use helper instead

    db = client[config.raw_db[group_type]]
    if data == 'users':
        csv_dir = None
        csv_location = str(Path(config.raw_db.csv_dir) / config.users[group_type].csv_location)
    elif data == 'friends':
        csv_dir = str(Path(config.raw_db.csv_dir) / config.friends[group_type].csv_dir)
        csv_location = str(Path(config.raw_db.csv_dir) / config.friends[group_type].csv_location)
    elif data == 'users_final_tweet':
        csv_dir = None
        csv_location = str(Path(config.raw_db.csv_dir) / config.users_final_tweet[group_type].csv_location)
    else:
        csv_dir = None
        csv_location = None
    return MongoHelper(db, config), csv_dir, csv_location

def pdt_to_utc(datestring):
    TZINFOS = {
    'PDT': pytz.timezone('US/Pacific'),
    'PST': pytz.timezone('US/Pacific')
    }
    datestring = datestring.replace("Pacific Daylight Time", "PDT")
    datestring = datestring.replace("Pacific Standard Time", "PST")
    # Parse the string using dateutil
    datetime_in_pdt = dateutil.parser.parse(datestring, tzinfos= TZINFOS)
    # t is now a PDT datetime; convert it to UTC
    datetime_in_utc = datetime_in_pdt.astimezone(pytz.utc)
    # Let's convert it to a naive datetime object
    return datetime_in_utc.replace(tzinfo = None)

def write_completed(file, message):
  with open(file, "a", encoding="utf8") as file_object:
    # Append at the end of file
    file_object.write(message + "\n")

def get_userid_from_file_name(file):
    return int(os.path.splitext(os.path.basename(file))[0])

def get_user_diagnose_tweet_timestamp(users_location, user_id):
  read_filename = os.path.join(users_location, f"{user_id}_tweet_prior_diagnose.csv")
  with open(read_filename, 'r') as f:
    rows = list(csv.reader(f))
    last_tweet_datetime = pdt_to_utc(str(rows[1][2]))
    return datetime.timestamp(last_tweet_datetime)

def get_all_user_friends(friends_location, user_file):
    all_friends = []
    user_id = get_userid_from_file_name(user_file)
    for root, dirs, files in os.walk(os.path.join(friends_location, user_id), topdown=True):
        for file in files: #for every file in the user dir
            if(file.lower().endswith(".csv")): # file is csv
                all_friends.append(int(os.path.splitext(file)[0])) # add the friend to the list
    return all_friends

def get_all_user_friends_files(friends_location, user_file):
    all_friends = []
    user_id = str(get_userid_from_file_name(user_file))
    for root, dirs, files in os.walk(os.path.join(friends_location, user_id), topdown=True):
        for file in files: #for every file in the user dir
            if(file.lower().endswith(".csv")): # file is csv
                all_friends.append(os.path.join(root, file)) # add the friend to the list
    return all_friends

# MENTIONS
def get_user_id_by_username(user_col, friend_col, username):
    result = friend_col.find_one({"username": username}, {'user_id': 1}, limit = 1)
    if result != None:
        return result['user_id']
    result = user_col.find_one({"username": username}, {'user_id': 1}, limit = 1)
    if result != None:
        return result['user_id']
    return None

def get_mentions_by_user(tweets_collection, user_id, friend_id, last_tweet_date_before_diagnosis):
    return tweets_collection.find(
        {
            'user_id': user_id,
            'created_at': {
                '$lt': last_tweet_date_before_diagnosis
            },
            'mentions': {
                '$all': [friend_id],
                '$exists': True
            },
        }, {
            'id': 1,
            'user_id': 2,
            'mentions': 3,
            'created_at': 4,
        }
    )


def get_mentions_by_friends(tweets_collection, user_id, friend_id, last_tweet_date_before_diagnosis):
    return tweets_collection.find(
        {
            'user_id': friend_id,
            'created_at': {
                '$lt': last_tweet_date_before_diagnosis
            },
            'mentions': {
                '$all': [user_id],
                '$exists': True
            },
        }, {
            'id': 1,
            'user_id': 2,
            'mentions': 3,
            'group_id': 4,
            'created_at': 5,
        }
    )


def get_replies_by_user(tweets_collection, user_id, friend_id, last_tweet_date_before_diagnosis):
    return tweets_collection.find(
        {
            'user_id': user_id,
            'created_at': {
                '$lt': last_tweet_date_before_diagnosis
            },
            'reply_to': {
                '$all': [friend_id],
                '$exists': True
            },
        }, {
            'id': 1,
            'user_id': 2,
            'reply_to': 3,
            'created_at': 4,
        }
    )

def get_replies_by_friends(tweets_collection, user_id, friend_id, last_tweet_date_before_diagnosis):
    return tweets_collection.find(
        {
            'user_id': friend_id,
            'created_at': {
                '$lt': last_tweet_date_before_diagnosis
            },
            'reply_to': {
                '$all': [user_id],
                '$exists': True
            },
        }, {
            'id': 1,
            'user_id': 2,
            'reply_to': 3,
            'created_at': 4,
        }
    )


def get_quotes_by_user(tweets_collection, user_id, friend_username, last_tweet_date_before_diagnosis):
    return tweets_collection.find(
        {
            'user_id': user_id,
            'quote_tweet_username': friend_username,
            'created_at': {
                '$lt': last_tweet_date_before_diagnosis
            },
        }, {
            'id': 1,
            'user_id': 2,
            'quote_tweet_username': 3,
            'quote_tweet_id': 4,
            'created_at': 5,
        }
    )

def get_quotes_by_friends(tweets_collection, user_id, username, friend_id, last_tweet_date_before_diagnosis):
    return tweets_collection.find(
        {
            'user_id': friend_id,
            'quote_tweet_username': username,
            'created_at': {
                '$lt': last_tweet_date_before_diagnosis
            },
        }, {
            'id': 1,
            'user_id': 2,
            'quote_tweet_username': 3,
            'quote_tweet_id': 4,
            'created_at': 5,
        }
    )

def get_users_wtih_exisiting_total_tweet_count(col):
    cursor = col.find({'total_tweets': {'$exists': True}}, {'user_id'})
    result = set()
    for row in cursor:
        result.add(row['user_id'])
    return result

def get_users_wtih_consistent_tweet_count(col):
    inconsistent_friend_ids = {}
    cursor = col.find({'tweet_count': {'$exists': True}, 'total_tweets': {'$exists': True}, '$expr': {'$eq': ['$tweet_count', '$total_tweets']}})
    for doc in cursor:
        if 'user_id' in doc and 'total_tweets' in doc:
            inconsistent_friend_ids[doc['user_id']] = doc['total_tweets']
    return (inconsistent_friend_ids, len(inconsistent_friend_ids))

def get_users_wtih_inconsistent_tweet_count(col):
    inconsistent_friend_ids = {}
    cursor = col.find({'$and': [
        {'total_tweets': {'$exists': True}},
        {'$expr': {'$ne': ['$tweet_count', '$total_tweets']}}
        ]
    })
    for doc in cursor:
        print(doc)
        if 'user_id' in doc and 'total_tweets' in doc:
            inconsistent_friend_ids[doc['user_id']] = doc['total_tweets']
    return (inconsistent_friend_ids, len(inconsistent_friend_ids))

def process_user(row):
    date = pdt_to_utc(row["created_at"])
    row['created_at'] = date

    row["mentions"] = ast.literal_eval(row["mentions"])
    if len(row["mentions"]) > 0:
        # hide screename and name
        row["mentions"] = list(map(lambda x: int(x["id"]), row["mentions"]))
    else:
        del row["mentions"]


    row["hashtags"] = ast.literal_eval(row["hashtags"])
    if len(row["hashtags"]) == 0:
        del row["hashtags"]

    row["reply_to"] = ast.literal_eval(row["reply_to"])

    # delete reply if none
    if len(row["reply_to"]) > 0:
        # hide screename and name
        row["reply_to"] = list(map(lambda x: int(x["id"]), row["reply_to"]))
    else:
        del row["reply_to"]

    if pd.isnull(row["quote_url"]):
        del row["quote_url"]
    elif type(row["quote_url"]) == str and len(row["quote_url"]) > 5:
        p = re.compile('https:\/\/twitter.com\/([a-zA-Z0-9_]+)\/status\/([0-9]+)')
        m = p.findall(row["quote_url"])
        if len(m) > 0:
            row["quote_tweet_username"], row["quote_tweet_id"] = m[0]
            row["quote_tweet_id"] = int(row["quote_tweet_id"])
        del row["quote_url"]

    #row['timestamp'] = int(datetime.timestamp(date))
    #row['group_id'] = int(((date.year - 2000) * 12 + date.month) / time_interval_in_month)
    # not necessary
    row.pop("username", None)
    row.pop("name", None)
    row.pop("conversation_id", None)
    row.pop("link", None)
    row.pop("replies_count", None)
    row.pop("retweets_count", None)
    row.pop("likes_count", None)
    #row.pop("created_at", None) #use timestamp instead

    #row["tweet"] =
    #clean_tweet(row["tweet"])
    return row

def remove_emoji(string):
    return emoji.get_emoji_regexp().sub(u'', string)

def remove_mentions(input_text):
    '''
    Function to remove mentions, preceded by @, in a Pandas Series

    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series
    '''
    return re.sub(r'@\w+', '', input_text)
def clean_tweet(input_text):
    #input_text = remove_mentions(input_text)
    #input_text = remove_emoji(input_text)

    new_text = []
    for t in input_text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    input_text = " ".join(new_text)

    return clean(input_text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=True,         # replace all phone numbers with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=True,                # replace all digits with a special token
        no_currency_symbols=True,      # replace all currency symbols with a special token
        no_punct=True,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="",
        replace_with_currency_symbol="",
        lang="en"                       # set to 'de' for German special handling
    )
