import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, file_path = 'result.csv'):
        self.columns = [
            'executed_at',
            'random_state',
            'round',
            'fold',
            'dataset_name',
            'ntpp',
            'mnf',
            'pim',
            'period_length',
            'model',
            'f1_depressed','recall_depressed', 'precision_depressed', 'acc_depressed',
            'f1_control', 'recall_control', 'precision_control', 'acc_control',
            'auc_roc',
            'train/test', 'num_depressed', 'num_control',
            'lr',
            'batch_size',
            'labels',
            'predictions',
            'epoch',
            'dataset_location',
            ]
        self.file_path = file_path
    def set(self, baseline, dataset_info, round, fold, args):
        self.baseline = baseline
        self.dataset_info = dataset_info
        self.args = args
        self.fold = fold
        self.round = round
        Path(self.file_path).touch(exist_ok = True)
        self.df = pd.read_csv(self.file_path, names = self.columns)

    def log(self, executed_at, epoch, r):
        df = self.df
        d = self.dataset_info
        a = self.args
        indices = (df['executed_at'] == executed_at) & (df['model'] == self.baseline) & (df['ntpp'] == d.num_tweets_per_period) & (df['mnf'] == d.max_num_friends) & (df['pim'] == d.periods_in_months)
        result = {
            'executed_at': executed_at,
            'dataset_location': f'{d.dataset_location}',
            'dataset_name': f'{d.dataset_name}',
            'random_state': f'{d.random_state}',
            'round': self.round,
            'fold': self.fold,
            'ntpp': d.num_tweets_per_period,
            'mnf': d.max_num_friends,
            'pim': d.periods_in_months,
            'period_length': d.period_length,
            'model': self.baseline,
            'epoch': epoch,
            'f1_depressed': f'{r.f1_depressed:.3f}',
            'recall_depressed': f'{r.recall_depressed:.3f}',
            'precision_depressed': f'{r.precision_depressed:.3f}',
            'acc_depressed': f'{r.acc_depressed:.3f}',
            'f1_control': f'{r.f1_control:.3f}',
            'recall_control': f'{r.recall_control:.3f}',
            'precision_control': f'{r.precision_control:.3f}',
            'acc_control': f'{r.acc_control:.3f}',
            'auc_roc': f'{r.auc_roc_macro:.3f}',
            'train/test': f'{a.train_test_split}',
            'num_depressed': f'{r.num_depressed}',
            'num_control': f'{r.num_control}',
            'lr': f'{a.learning_rate}',
            'batch_size': f'{a.train_batch_size}',
            'labels': np.array2string(r.labels, separator = ''),
            'predictions': np.array2string(r.predictions, separator = ''),
        }
        
        if len(df[indices].index) == 0:
            self.df = self.df.append(result, ignore_index = True)
        else:
            for key, value in result.items():
                self.df.loc[indices, key] = value
        self.df.to_csv(self.file_path, header = False)
