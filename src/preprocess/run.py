from datetime import datetime
import mailcap
import sys
import inspect
import pkgutil
from preprocess.transforms.collect_period_data import collect_period_data, collect_period_data_all
from preprocess.configs.config import config
from preprocess.utils.util import get_mongo_client

client = get_mongo_client(config)
group_types = ["depressed_group", "control_group"]
num_tweets_per_period_list = [10, 5, 3, 1]
periods_in_months_list = [3, 6, 9, 12]
num_friend_in_period_list = [20]#, 4, 6, 2]
start_date = datetime(year = 2007, month = 1, day = 1, hour = 0, minute = 0, second = 0)
end_date = datetime(year = 2022, month = 1, day = 1, hour = 0, minute = 0, second = 0)

for num_tweets_per_period in num_tweets_per_period_list:
    for periods_in_months in periods_in_months_list:
        for num_friend_in_period in num_friend_in_period_list:
            for group_type in group_types:
                """
                collect_period_data(
                    client = client,
                    config = config,
                    group_type = group_type,
                    start_date = start_date,
                    end_date = end_date,
                    num_tweets_per_period = num_tweets_per_period,
                    periods_in_months = periods_in_months
                )
                """

                collect_period_data_all(
                    client = client,
                    config = config,
                    group_type = group_type,
                    start_date = start_date,
                    end_date = end_date,
                    num_tweets_per_period = num_tweets_per_period,
                    num_friend_in_period = num_friend_in_period,
                    periods_in_months = periods_in_months
                )
