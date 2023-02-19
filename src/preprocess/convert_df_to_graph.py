import re
import os
import glob
from tkinter import E
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
from ast import literal_eval
group_types = ["depressed_group", "control_group", ]
#group_types = "depressed_group",  "control_group"]

columns = ['user_id', 'group', 'period_id', 'tweet_id', 'created_at', 'mention_count', 'reply_count', 'quote_count', 'total_count', 'embeddings']
embedding_column_name = 'embeddings_twitter-roberta-base-jun2022'

max_num_friends_list = [5, 3, 1]#1, 2, 3, 4, 5, 6, 7, 8, 9, 10]#1, 3, 5]#5, 10, 15, 20]
periods_in_months_list = [12, 9, 6, 3]
num_tweets_per_period_list = [5, 3, 1]#, 10, 20]
dates = [
    datetime(year = 2017, month = 1, day = 1, hour = 0, minute = 0, second = 0),
    datetime(year = 2017, month = 7, day = 1, hour = 0, minute = 0, second = 0),
    datetime(year = 2018, month = 1, day = 1, hour = 0, minute = 0, second = 0),
    datetime(year = 2018, month = 7, day = 1, hour = 0, minute = 0, second = 0),
    datetime(year = 2019, month = 1, day = 1, hour = 0, minute = 0, second = 0),
    datetime(year = 2019, month = 7, day = 1, hour = 0, minute = 0, second = 0),
]
friend_selected = {}
refer_base_dir = 'ut_5_p_3_version3'

STRICT_MODE_NUM_TWEETS_EVERY_PERIOD = True
MIN_PERIOD_LENGTH = 2
PERIOD_LENGTH = 6
NUM_PERIODS_BEFORE_DIAGNOSIS = 0
INTERPOLATION = True

def save_graphs(graphs, pretrained = False):
    period_id = 0
    for _, G in graphs:
        if pretrained:
            path = f'data/{embedding_column_name}/{refer_base_dir}/ut_{num_tweets_per_period}_mnf_{max_num_friends}_p_{periods_in_months}/pretrained/{group_type}/{user_id}/{period_id}.pickle'
        else:
            path = f'data/{embedding_column_name}/{refer_base_dir}/ut_{num_tweets_per_period}_mnf_{max_num_friends}_p_{periods_in_months}/{group_type}/{user_id}/{period_id}.pickle'
                    
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.DEFAULT_PROTOCOL)
            print(f'{group_type} [{counter}/{size}] {user_id} - #{period_id}({_}) {G}')
        period_id += 1

def period_id_relabel(tweet_date):
    start_date = datetime(year = 2006, month = 1, day = 1, hour = 0, minute = 0, second = 0)
    period_end_date = datetime(year = 2022, month = 1, day = 1, hour = 0, minute = 0, second = 0)
    date = datetime.strptime(tweet_date, '%Y-%m-%d %H:%M:%S')
    period_id = 0
    while start_date <= period_end_date:
        end_date = start_date + relativedelta(months = periods_in_months)
        if date >= start_date and date < end_date:
            break
        # ending
        period_id += 1
        start_date = end_date
    return period_id
for periods_in_months in periods_in_months_list:
    for num_tweets_per_period in num_tweets_per_period_list:
        for max_num_friends in max_num_friends_list:
            stats_periods = []
            stats_num_friends = []
            stats_num_tweets_per_user = []
            stats_num_tweets_per_friends = []
            count_dict = {'num_users': 0, 'reply_count': 0, 'quote_count': 0, 'mention_count': 0, 'total_count': 0}
            for group_type in group_types:
                base_path = f'df_data/{embedding_column_name}/{refer_base_dir}/{group_type}/*csv'
                user_node_indices = {}
                user_node_indices_counter = 0
                friend_node_indices = {}
                friend_node_indices_counter = 0
                counter = 0
                file_counter = 0
                size = len(sorted(glob.glob(base_path, recursive = True)))
                period_length_distribution = {}
                for file in sorted(glob.glob(base_path, recursive = True)):
                    file_counter += 1
                    m = re.search(r'([0-9]+)\.csv', file)
                    assert m != None, "m == None"
                    user_id = int(m.groups()[0])
                    if user_id not in user_node_indices:
                        user_node_indices[user_id] = user_node_indices_counter
                        user_node_indices_counter += 1
                    
                    user_node_id = f'user_{user_node_indices[user_id]}'
                    df = pd.read_csv(file, header = 0, index_col = 0)
                    df['period_id'] = df['created_at'].apply(period_id_relabel)
                    #print(df[['created_at', 'period_id']])
                    df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep = ' '))
                    disqualified = False
                    #

                    period_indices = sorted(df['period_id'].unique())
                    period_length = len(period_indices)
                    print(period_indices, period_length)
                    if INTERPOLATION and period_length < MIN_PERIOD_LENGTH:
                        print(f'{user_id} has insufficient tweets in periods. {period_length} | {PERIOD_LENGTH}')
                        continue
                    elif not INTERPOLATION and period_length < PERIOD_LENGTH:
                        print(f'{user_id} has insufficient tweets in periods. {period_length} | {PERIOD_LENGTH}')
                        continue
                    #print(period_indices, len(period_indices))
                    if period_length not in period_length_distribution:
                        period_length_distribution[period_length] = 0
                    period_length_distribution[period_length] += 1
                    all_exists = os.path.isdir(f'/home/aite/med/process/data/embeddings_twitter-roberta-base-jun2022/ut_5_p_3_version3/depressed_group/{user_id}')
                    #print(file, period_indices, all_exists, period_length_distribution)
                    graphs = []
                    
                    for friend_id in df['user_id'].unique():
                        temp_df = df[df['user_id'] == friend_id].head(1)
                        for key in count_dict.keys():
                            if key != 'num_users':
                                count_dict[key] += temp_df[key].values[0]
                        count_dict['num_users'] += 1
                        
                    for period_id in period_indices:
                        G = nx.MultiDiGraph()
                        period_df = df[df['period_id'] == period_id]
                        is_user = period_df['user_id'] == user_id
                        #user_df = period_df[is_user].sample(n = num_tweets_per_period)
                        user_df = period_df[is_user][:num_tweets_per_period]
                        if STRICT_MODE_NUM_TWEETS_EVERY_PERIOD:  
                            if len(user_df) < num_tweets_per_period:
                                continue
                        friend_df = period_df[~is_user]
                        if len(friend_df) == 0:
                            continue
                        friend_df['interaction_count'] = friend_df[['reply_count', 'mention_count', 'quote_count']].apply(lambda x: int(x['reply_count'] > 0) + int(x['mention_count'] > 0) + int(x['quote_count'] > 0), axis = 1)
                    
                        user_embeddings = np.float32(np.mean(user_df['embeddings'].values))

                        G.add_node(user_node_id)
                        nx.set_node_attributes(G, {user_node_id: {'features': user_embeddings, 'label': group_type}})

                        key = f'{file}-{period_id}'
                        if key not in friend_selected:
                            friend_rank_df = friend_df.drop_duplicates(subset=['user_id']).sort_values(['interaction_count', 'total_count'], ascending = False)
                            friend_indices = []
                            for friend_id, group_df in friend_df.groupby('user_id'):
                                friend_tweets = group_df['embeddings'].values[:num_tweets_per_period]
                                if len(friend_tweets) != num_tweets_per_period:
                                    continue
                                friend_indices.append(friend_id)
                            friend_selected[key] = friend_indices
                        friend_indices = friend_selected[key]
                        #friend_indices = friend_indices
                        friend_df = friend_df[friend_df['user_id'].isin(friend_indices)]
                        
                        friend_count = 0
                        friend_data = []
                        for friend_id, group_df in friend_df.groupby('user_id'):
                            #print(friend_id, group_df)
                            first_row = group_df.head(1)
                            friend_tweets = group_df['embeddings'].values[:num_tweets_per_period]
                            if STRICT_MODE_NUM_TWEETS_EVERY_PERIOD:
                                if len(friend_tweets) != num_tweets_per_period:
                                    continue
                            friend_embeddings = np.float32(np.mean(friend_tweets))
                            #friend_embeddings = np.float32(np.mean(group_df['embeddings'].sample(n = num_tweets_per_period).values))
                            mention_count = first_row['mention_count'].values[0]
                            reply_count = first_row['reply_count'].values[0]
                            quote_count = first_row['quote_count'].values[0]
                            if friend_id not in friend_node_indices:
                                friend_node_indices[friend_id] = friend_node_indices_counter
                                friend_node_indices_counter += 1
                            friend_node_id = f'friend_{friend_node_indices[friend_id]}'
                            friend_data.append(
                                {
                                    'friend_node_id': friend_node_id,
                                    'friend_embeddings': friend_embeddings,
                                    'mention_count': mention_count,
                                    'reply_count': reply_count,
                                    'quote_count': quote_count,
                                }
                            )
                            friend_count += 1

                            #print(mention_count, reply_count, quote_count, friend_id)
                        if friend_count > max_num_friends:
                            friend_data = friend_data[:max_num_friends]
                            for row in friend_data:
                                friend_node_id = row['friend_node_id']
                                G.add_node(friend_node_id)
                                nx.set_node_attributes(G, {friend_node_id: {'features': row['friend_embeddings'], 'label': 'friend'}})
                                G.add_edge(user_node_id, friend_node_id, label = 'mention', weight = row['mention_count'])
                                G.add_edge(user_node_id, friend_node_id, label = 'reply', weight = row['reply_count'])
                                G.add_edge(user_node_id, friend_node_id, label = 'quote', weight = row['quote_count'])

                            graphs.append([period_id, G])
                        #print(period_id, user_df, friend_df)
                    #user_df = 
                    #print(df)
                    if STRICT_MODE_NUM_TWEETS_EVERY_PERIOD and disqualified:
                        continue
                    
                    if INTERPOLATION and len(graphs) < MIN_PERIOD_LENGTH + NUM_PERIODS_BEFORE_DIAGNOSIS:
                        print(f'INTERPOLATION: {INTERPOLATION} | {user_id} has insufficient tweets in periods. {len(graphs)} | {MIN_PERIOD_LENGTH}')
                        continue
                    elif not INTERPOLATION and len(graphs) < PERIOD_LENGTH + NUM_PERIODS_BEFORE_DIAGNOSIS:
                        print(f'INTERPOLATION: {INTERPOLATION} | {user_id} has insufficient tweets in periods. {len(graphs)} | {PERIOD_LENGTH}')
                        continue

                    
                    graphs = sorted(graphs, key = lambda x: x[0])
                    #pretrained_graphs = graphs[:-PERIOD_LENGTH - NUM_PERIODS_BEFORE_DIAGNOSIS]
                    train_graphs = graphs[-PERIOD_LENGTH:]#[-PERIOD_LENGTH - NUM_PERIODS_BEFORE_DIAGNOSIS: - NUM_PERIODS_BEFORE_DIAGNOSIS]
                    difference = PERIOD_LENGTH - len(train_graphs)
                    print(train_graphs, train_graphs[0])
                    if difference > 0:
                        train_graphs = [train_graphs[0] for i in range(difference)] + train_graphs
                    print(len(train_graphs))
                    #print('pretrain', list(map(lambda x:x[0], pretrained_graphs)))
                    print('train', list(map(lambda x:x[0], train_graphs)))
                    #save_graphs(pretrained_graphs, pretrained = True)
                    save_graphs(train_graphs, pretrained = False)
                    print(f'{group_type} (ut: {num_tweets_per_period:02}, mnf: {max_num_friends:02}, p: {periods_in_months:02}) | #{counter:03}, total: {file_counter}/{size}  - {user_id}')
                    counter += 1
