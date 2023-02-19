
"""
model = "embeddings_twitter-roberta-base-jun2022"
control_users = load_graph('control_group', 0,  model, max_num_friends, periods_in_months)
depressed_users = load_graph('depressed_group', 1, model, max_num_friends, periods_in_months)

exit()
"""
import copy
import pathlib
import numpy as np
from datetime import datetime
from mental.utils.utilities import load_data, get_model, get_train_test_data
from mental.utils.trainer import Trainer
from mental.utils.dataclass import BaselineModel, DatasetInfo
from mental.utils.logger import Logger

from sklearn.model_selection import KFold, StratifiedKFold
from itertools import product

dataset_name = 'version8'
dataset_location = pathlib.Path(__file__).parent.resolve()

logger = Logger('playground.csv')
tweet_processing_model_name_from_huggingface = "embeddings_twitter-roberta-base-jun2022"


#mentalnet
#mentalnetdysat
baselines = [
    #BaselineModel.MLP,
    #BaselineModel.UGformer,
    #BaselineModel.GCN,
    #BaselineModel.GAT,
    #BaselineModel.GraphSAGE,
    #BaselineModel.EvolveGCN,
    #BaselineModel.DySAT,
    #BaselineModel.MentalNet,
    #BaselineModel.MentalNet_Original,
    #BaselineModel.MentalNetNaive,
    #BaselineModel.MentalNet_SAGE,
    #BaselineModel.MentalNet_GAT,
    #BaselineModel.MentalNet_GAT2,
    #BaselineModel.MentalNetDySAT_SimSiam,
    #BaselineModel.MentalNet_DySAT,
    BaselineModel.MentalPlus,
    #BaselineModel.MentalPlus_NO_POSITION,
    #BaselineModel.MentalPlus_NO_POSITION_CLS_POOLING,
    #BaselineModel.MentalPlus_CLS_POOLING,
    #BaselineModel.MentalPlus_USE_NODE,
    #BaselineModel.MentalPlus_USE_GRAPH,
    #BaselineModel.MentalPlus_USE_GNN,
    #BaselineModel.MentalPlus_NO_HGNN,
    #BaselineModel.MentalPlus_NO_SUPERVISED,
    #BaselineModel.MentalPlus_Without_Transformer,
    #BaselineModel.MentalPlus_NO_GRAPH_AGGREGATION,
    #BaselineModel.MentalPlus_SimSiam,
    #BaselineModel.MentalPlus_GCN,
    #BaselineModel.MentalPlus_GAT,
    #BaselineModel.MentalPlus_SAGE,
]
#dataset_info -> current location
num_tweets_per_period_list = map(str, [5])
max_num_friends_list = map(str, [4])
periods_in_months_list = map(str, [ 6, 12])
period_length_list = map(str, [10])


random_state = 5
dataset_list = list(
    map(
        lambda x:
        DatasetInfo(
            tweet_processing_model_name = tweet_processing_model_name_from_huggingface,
            num_tweets_per_period = x[0],
            max_num_friends = x[1],
            periods_in_months = x[2],
            period_length = x[3],
            dataset_location = dataset_location,
            dataset_name = dataset_name,
            random_state = random_state),
        product(num_tweets_per_period_list, max_num_friends_list, periods_in_months_list, period_length_list)))

train_test_split = 0.3
for i in range(100):
    for dataset_info in dataset_list:
        data = load_data(dataset_info)
        kf = KFold(n_splits = 5, random_state = random_state, shuffle = True)
        round = int(datetime.timestamp(datetime.now()))
        for fold, (train_index, test_index) in enumerate(kf.split(data)): 
            train_data = data[train_index]
            test_data = data[test_index]
            for baseline in baselines:
                model, args, model_args = get_model(baseline, dataset_info)
                args.train_test_split = train_test_split
                logger.set(baseline.__name__, dataset_info, round, fold, args)
                _train_data = model.prepare_data(copy.deepcopy(train_data))
                #_validation_data = model.prepare_data(copy.deepcopy(train_data))
                _test_data = model.prepare_data(copy.deepcopy(test_data))

                trainer = Trainer(_train_data, _validation_data, _test_data, logger, args)
                model = trainer.train(model)
                #model = trainer.train(model, None)
                #try:
                #except Exception as e:
                #    print(e)
                #    with open('error.log', 'a+') as writer:
                #        writer.write(f'{baseline.__name__}, {dataset_info.random_state},{dataset_info}\n')
                