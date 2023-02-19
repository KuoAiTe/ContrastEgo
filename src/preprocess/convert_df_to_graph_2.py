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

columns = ['user_id', 'group', 'period_id', 'tweet_id', 'created_at', 'mention_count', 'reply_count', 'quote_count', 'total_count', 'embeddings']
NLP_MODEL = 'embeddings_twitter-roberta-base-jun2022'

friend_selected = {}
MIN_PERIOD_LENGTH = 3
PERIOD_LENGTH = 10
NUM_PERIODS_BEFORE_DIAGNOSIS = 0
INTERPOLATION = False
NUM_FRIENDS_LIST = [8]#2, 3, 4, 5, 6, 7, 8]#2, 4, 6, 8]
ANY_NUM_FRIEND = True
ANY_NUM_TWEET = True

DATASET_LIST = [
    'ut_10_p_9_mnf20_version3',
    'ut_10_p_6_mnf20_version3',
    'ut_10_p_3_mnf20_version3',
]
def save_graphs(graphs, save_dir):
    period_id = 0
    graphs = [G for _, G in graphs]
    save_dir.mkdir(parents=True, exist_ok=True)
    for G in graphs:
        path = save_dir / f'{period_id}.pickle'
        with open(path, 'wb') as f:
            pickle.dump(G, f, pickle.DEFAULT_PROTOCOL)
            print(f'{group_type} [{counter}/{size}] {user_id} - #{period_id}() {G}')
        period_id += 1



count_dict = {'num_users': 0, 'reply_count': 0, 'quote_count': 0, 'mention_count': 0, 'total_count': 0}
for refer_base_dir in DATASET_LIST: 
    m = re.search(r'ut_([0-9]+)_p_([0-9]+)_mnf([0-9]+)', refer_base_dir)
    num_tweets_per_period, period_interval_in_months, max_num_friends = [int(_) for _ in m.groups()]

    for num_tweets_per_period in [10]:
        ANY_NUM_TWEET = False
        for num_friend in NUM_FRIENDS_LIST:
            for group_type in group_types:
                base_path = f'df_data/{NLP_MODEL}/{refer_base_dir}/{group_type}/*csv'
                user_node_indices = {}
                user_node_indices_counter = 0
                friend_node_indices = {}
                friend_node_indices_counter = 0
                counter = 0
                file_counter = 0
                size = len(sorted(glob.glob(base_path, recursive = True)))
                period_length_distribution = {}
                t = False
                for file in sorted(glob.glob(base_path, recursive = True)):
                    file_counter += 1
                    m = re.search(r'([0-9]+)\.csv', file)
                    assert m != None, "m == None"
                    user_id = int(m.groups()[0])
                    if user_id not in user_node_indices:
                        user_node_indices[user_id] = user_node_indices_counter
                        user_node_indices_counter += 1
                    
                    user_node_id = f'user_{user_node_indices[user_id]}'
                    
                    df = pd.read_csv(file, header = 0, index_col = 0, engine='python')
                    df.dropna(subset=['embeddings'], inplace=True)
                    df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep = ' '))
                    #
                    
                    period_indices = sorted(df['period_id'].unique())
                    period_length = len(period_indices)
                    print(period_indices, period_length)
                    #if INTERPOLATION and period_length < MIN_PERIOD_LENGTH:
                    #    print(f'1. {user_id} has insufficient tweets in periods. {period_length} | {PERIOD_LENGTH}')
                    #    continue
                    #elif not INTERPOLATION and period_length < PERIOD_LENGTH:
                    #    print(f'2. {user_id} has insufficient tweets in periods. {period_length} | {PERIOD_LENGTH}')
                    #    continue
                    #print(period_indices, len(period_indices))
                    if period_length not in period_length_distribution:
                        period_length_distribution[period_length] = 0
                    period_length_distribution[period_length] += 1
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
                        if len(period_df[is_user]) > num_tweets_per_period:
                           user_df = period_df[is_user].sample(n = num_tweets_per_period)
                        else:
                            user_df = period_df[is_user]
                        #user_df = period_df[is_user][:num_tweets_per_period]
                        if len(user_df) == 0 or (not ANY_NUM_TWEET and len(user_df) < num_tweets_per_period):
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
                            if not ANY_NUM_TWEET and len(friend_tweets) != num_tweets_per_period:
                                continue
                            friend_embeddings = np.float32(np.mean(friend_tweets))
                            #friend_embeddings = np.float32(np.mean(group_df['embeddings'].sample(n = NUM_TWEETS_PER_PERIOD).values))
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
                        if (ANY_NUM_FRIEND and friend_count > 0) or friend_count >= num_friend:
                            if friend_count >= num_friend:
                                friend_data = friend_data[:num_friend]
                            for row in friend_data:
                                friend_node_id = row['friend_node_id']
                                G.add_node(friend_node_id)
                                nx.set_node_attributes(G, {friend_node_id: {'features': row['friend_embeddings'], 'label': 'friend'}})
                                if type(row['friend_embeddings']) != np.ndarray:
                                    print(row['friend_embeddings'].shape, type(row['friend_embeddings']))
                                    print(row['friend_embeddings'])
                                    exit()
                                G.add_edge(user_node_id, friend_node_id, label = 'mention', weight = row['mention_count'])
                                G.add_edge(user_node_id, friend_node_id, label = 'reply', weight = row['reply_count'])
                                G.add_edge(user_node_id, friend_node_id, label = 'quote', weight = row['quote_count'])
                            
                            G.graph['user_id'] = user_id
                            graphs.append([period_id, G])
                        #print(period_id, user_df, friend_df)
                    #user_df = 
                    #print(df)
                    """
                    if INTERPOLATION and len(graphs) < MIN_PERIOD_LENGTH + NUM_PERIODS_BEFORE_DIAGNOSIS:
                        print(f'3. INTERPOLATION: {INTERPOLATION} | {user_id} has insufficient tweets in periods. {len(graphs)} | {MIN_PERIOD_LENGTH}')
                        continue
                    elif not INTERPOLATION and len(graphs) < PERIOD_LENGTH + NUM_PERIODS_BEFORE_DIAGNOSIS:
                        print(f'4. INTERPOLATION: {INTERPOLATION} | {user_id} has insufficient tweets in periods. {len(graphs)} | {PERIOD_LENGTH}')
                        continue
                    """

                    
                    graphs = sorted(graphs, key = lambda x: x[0])
                    #pretrained_graphs = graphs[:-PERIOD_LENGTH - NUM_PERIODS_BEFORE_DIAGNOSIS]
                    train_graphs = graphs[-PERIOD_LENGTH:]#[-PERIOD_LENGTH - NUM_PERIODS_BEFORE_DIAGNOSIS: - NUM_PERIODS_BEFORE_DIAGNOSIS]
                    difference = PERIOD_LENGTH - len(train_graphs)
                    #print(train_graphs, train_graphs[0])
                    print(len(graphs), len(train_graphs))
                    if len(train_graphs) == 1 or len(train_graphs) < MIN_PERIOD_LENGTH:
                        continue
                    """
                    if difference > 0:
                        interpolated_graphs = []
                        for i in range(difference):
                            _, G = train_graphs[0]
                            G = G.copy()
                            G.graph['interpolation'] = True
                            interpolated_graphs.append((-1, G))
                        train_graphs = interpolated_graphs + train_graphs
                        continue
                    """
                    print(len(train_graphs))
                    #print('pretrain', list(map(lambda x:x[0], pretrained_graphs)))
                    print('train', list(map(lambda x:x[0], train_graphs)), )
                    #save_dir = Path(f'data/{NLP_MODEL}/ut_{num_tweets_per_period}_mnf_{num_friend}_p_{period_interval_in_months}_l_{PERIOD_LENGTH}/{group_type}/{user_id}')
                    save_dir = Path(f'data/{NLP_MODEL}/test/{group_type}/{user_id}00{num_tweets_per_period}11{num_friend}22{period_interval_in_months}33{PERIOD_LENGTH}')
                    save_graphs(train_graphs, save_dir = save_dir)
                    print(save_dir)
                    print(f'{group_type} (ut: {num_tweets_per_period:02}, mnf: {num_friend:02}, p: {period_interval_in_months:02}) | #{counter:03}, total: {file_counter}/{size}  - {user_id}')
                    counter += 1
