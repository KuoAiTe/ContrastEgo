import pandas as pd
import re
import numpy as np
import glob
from pathlib import Path

from mental.utils.utilities import compute_metrics_from_results
from mental.utils.logger import Logger

columns = Logger().columns
file_path = './results/v8/*.csv'
files = glob.glob(file_path)
df = pd.concat((pd.read_csv(f, names = columns) for f in files), ignore_index=True)
df = df.sort_values(by=['model', 'random_state','ntpp', 'mnf', 'pim', 'period_length', 'fold'])
df.reset_index(drop=True, inplace = True)
#df.to_csv(file_path, header = None)
metrics_for_depressed = ['f1_depressed', 'recall_depressed', 'precision_depressed']
metrics_for_control = ['f1_control', 'recall_control', 'precision_control', 'auc_roc']
metrics = metrics_for_depressed + metrics_for_control
column_mapping = {
    'f1_depressed': 'f1_score',
    'recall_depressed': 'recall',
    'precision_depressed': 'precision',
    'f1_control': 'f1_score',
    'recall_control': 'recall',
    'precision_control': 'precision',
    'auc_roc': 'auc_roc'
}

NAME_MAPPINGS = {
    'GATWrapper': 'GAT',
    'GCNWrapper': 'GCN',
    'GraphSAGEWrapper': 'GraphSAGE',
    'MentalNet_Original': 'MentalNet',
    'MentalPlus': 'ContrastEgo',
    'MentalPlus_BASE': 'ContastEgo-C',
    'MentalPlus_NO_HGNN': 'ContastEgo-H',
    'MentalPlus_GRAPH': 'ContastEgo-G',
    'MentalPlus_USE_NODE': 'ContastEgo-UN',
    'MentalPlus_USE_GRAPH': 'ContastEgo-UG',
    'MentalPlus_Without_Transformer': 'ContastEgo-T',
    'MentalPlus_NO_SUPERVISED': 'ContastEgo-S',
    'MentalPlus_NO_GRAPH_AGGREGATION': 'ContastEgo-GA',
    'MentalPlus_CON': 'ContrastEgo',
    'MentalPlus_CON_BASE': 'ContastEgo-C',
    'MentalPlus_CON_NO_HGNN': 'ContastEgo-H',
    'MentalPlus_CON_GRAPH': 'ContastEgo-G',
    'MentalPlus_CON_USE_NODE': 'ContastEgo-UN',
    'MentalPlus_CON_USE_GRAPH': 'ContastEgo-UG',
    'MentalPlus_CON_Without_Transformer': 'ContastEgo-T',
    'MentalPlus_CON_NO_SUPERVISED': 'ContastEgo-S',
    'MentalPlus_CON_NO_GRAPH_AGGREGATION': 'ContastEgo-GA',
}

ORDER = [
    'GCNWrapper',
    'GATWrapper',
    'GraphSAGEWrapper',
    'DySAT',
    'MentalNet_Original',
    'MentalPlus',
    'MentalPlus_CLS_POOLING',
    'MentalPlus_MEAN_POOLING',
    'MentalPlus_NO_POSITION',
    'MentalPlus_NO_POSITION_CLS_POOLING',
    'MentalPlus_NO_HGNN',
    'MentalPlus_USE_GNN',
    'MentalPlus_USE_GRAPH',
    'MentalPlus_USE_NODE',
    'MentalPlus_NO_SUPERVISED',
    'MentalPlus_NO_GRAPH_AGGREGATION',
    'MentalPlus_Without_Transformer',
    'MentalPlus_CON',
    'MentalPlus_CON_BASE',
    'MentalPlus_CON_NO_HGNN',
    'MentalPlus_CON_GRAPH',
    'MentalPlus_CON_USE_NODE',
    'MentalPlus_CON_USE_GRAPH',
    'MentalPlus_CON_Without_Ttransformer',
    'MentalPlus_CON_BASE_Without_Ttransformer',
    'MentalPlus_CON_NO_GRAPH_AGGREGATION',
    'MentalPlus_CON_BASE_NO_GRAPH_AGGREGATION',
    'MentalPlus_CON_BASE_GNN_NO_GRAPH_AGGREGATION',
    'MentalPlus_CON_GNN_NO_GRAPH_AGGREGATION',
]

NUM_TWEETS_PER_PERIOD_LIST = [1, 3, 5]
NUM_FRIEND_LIST = [2, 4, 8]
PERIOD_IN_MONTHS_LIST = [3, 6, 12]

# Custom
ALLOWED_RANDOM_STATE = 5
ALLOWED_TWEETS = [1, 3, 5]
ALLOWED_FRIENDS = [2, 4, 8]
ALLOWED_MONTHS = [3, 6, 12]

ABALATION_STUDY = False
print(df['model'].unique())
if ABALATION_STUDY:
    DISALLOWED_VARIANTS = [_ for _ in df['model'].unique() if not _.startswith('MentalPlus')]
    DISALLOWED_VARIANTS = DISALLOWED_VARIANTS + [_ for _ in df['model'].unique() if 'USE' in _]
    
else:
    DISALLOWED_VARIANTS = [_ for _ in df['model'].unique() if _.startswith('MentalPlus_') or _.startswith('MentalPlus_CON_')]
    DISALLOWED_VARIANTS.remove('MentalPlus_CON')
df = df[~df['model'].isin(DISALLOWED_VARIANTS)]

def get_performance(df):
    performance = {}
    #'time', 
    for (model, random_state), group_df in df.groupby(['model', 'random_state']):
        if random_state != ALLOWED_RANDOM_STATE: continue
        labels = []
        predictions = []
        metric_results = {}
        for (round, round_df) in group_df.groupby('round'):
            if len(round_df['fold']) % 5 != 0:
                continue

            for txt in round_df.labels.values:
                txt = re.sub('[^0-9]', '', txt)
                for _ in txt:
                    labels.append(int(_))
            for txt in round_df.predictions.values:
                txt = re.sub('[^0-9]', '', txt)
                for _ in txt:
                    predictions.append(int(_))
        if len(labels) == 0: continue
        assert(len(labels) == len(predictions))
        
        labels = np.array(labels)
        predictions = np.array(predictions)
        result = compute_metrics_from_results(labels, predictions)
        metric_result = {
            'f1_depressed': result.f1_depressed,
            'recall_depressed': result.recall_depressed,
            'precision_depressed': result.precision_depressed,
            'f1_control': result.f1_control,
            'recall_control': result.recall_control,
            'precision_control': result.precision_control,
            'auc_roc': result.auc_roc_macro,
        }
        metric_result = group_df[metrics].mean().to_dict()
        
        if (model, random_state) not in performance or metric_result['auc_roc'] > performance[(model, random_state)]['auc_roc']:
            performance[(model, random_state)] = metric_result
    return performance

def print_line(model, model_data, columns):
    segments = []
    for data, best_data in model_data:
        segment = []
        for column in columns:
            if type(data[column]) == np.float64 or type(data[column]) == float:

                if np.round(data[column], 2) == np.round(best_data[column], 2):
                    segment.append('\\textbf{%.2f}' % data[column])
                else:
                    segment.append(f'{data[column]:.2f}')
            else:
                segment.append(data[column])
        segment = '&'.join(segment)
        segments.append(segment)
    line = model.replace("_", "\_") + '&' + '&'.join(segments)
    return line


def get_header(experimental_variable_list, experimental_name, caption, label = ''):
    column_size = len(experimental_variable_list)
    first_line = ['']
    for i in range(column_size):
        first_line.append("\\multicolumn{7}{c}{%d %s}" % (experimental_variable_list[i], experimental_name))
    first_line = '&'.join(first_line)


    second_line = ['']
    for i in range(column_size):
        second_line.append("\\multicolumn{3}{c}{Healthy (0)} &  \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{Depessed (1)}")
    second_line = '&'.join(second_line)

    third_line = ['']
    for i in range(column_size):
        third_line.append("P&R&F$\_{1}$&AUC&P&R&F$\_{1}$")
    third_line = '&'.join(third_line)
    caption = caption.replace("_", '\_')
    header = "\\begin{table*}[]\n\\caption{" + caption + "\label{table:" + label + "}}\n\\bgroup\n\\resizebox{\\linewidth}{!}{%\n  \\begin{tabular}{"
    header += 'l'
    for _ in range(column_size):
        header += 'lllllll|'
    header += "}\n"
    lines = [first_line, second_line, third_line]
    for line in lines:
        header += f'    {line}\\\\\n'
    return header

def get_footer():
    return '\n \\end{tabular}\n}\n\\egroup\n\\end{table*}\n\n\n'
    
def generate_table(data, settings, experimental_variable_list, experimental_name = '',  caption = '', label = ''):
    columns = ['precision_control', 'recall_control', 'f1_control', 'auc_roc', 'precision_depressed', 'recall_depressed', 'f1_depressed']
    experimental_variable_list = experimental_variable_list
    lines = []
    best_performance = {}
    for key, model_info in settings.items():
        model, random_state = model_info['model'], model_info['random_state']
        model_data = []
        for experimental_variable in experimental_variable_list:
            experimental_data = data[experimental_variable]
            exp_data = {m:'-' for m in metrics}
            if experimental_variable not in best_performance:
                best_data = {column: -1 for column in columns}
                for key, value in experimental_data.items():
                    for column in columns:
                        if value[column] > best_data[column]:
                            best_data[column] = value[column]
                best_performance[experimental_variable] = best_data
            else:
                best_data = best_performance[experimental_variable]
            if (model, random_state) in experimental_data:
                exp_data = experimental_data[(model, random_state)]
            model_data.append((exp_data, best_data))
        setting_name = NAME_MAPPINGS[model] if model in NAME_MAPPINGS else f'{model}'
        line = print_line(setting_name, model_data, columns)
        lines.append('    ' + line)
    body = '\\\\\n'.join(lines)
    header = get_header(experimental_variable_list, experimental_name, caption, label)
    footer = get_footer()
    return f'{header}{body}{footer}'

def get_settings(df):
    settings = {}
    for i in df[['model', 'random_state']].values.tolist():
        key = f'{i[0]},{i[1]}'
        settings[key] = {
            'model': i[0],
            'random_state': i[1],
        }
    settings = dict(sorted(settings.items(), key=lambda x: ORDER.index(x[1]['model'])))

    return settings
def plot_by_num_friend(
    df,
    num_tweet_per_period,
    period_in_months,
):
    df = df[(df['ntpp'] == num_tweet_per_period) & (df['pim'] == period_in_months)]
    num_friends_list = df['mnf'].unique()
    num_friends_list = [_ for _ in num_friends_list if _ in ALLOWED_FRIENDS]
    data = {}
    for num_friend in num_friends_list:
        _ = df[df['mnf'] == num_friend]
        data[num_friend] = get_performance(_)
    caption = f'num_tweet_per_period: {num_tweet_per_period}, period_in_months:{period_in_months}'
    #df['setting'] = df[['time', 'model', 'random_state']]
    label = f'vary_friend_t_{num_tweet_per_period}_p_{period_in_months}'
    table = generate_table(data, get_settings(df), num_friends_list, experimental_name = 'friend', caption = caption, label = label)
    return table

def plot_by_num_tweet(
    df,
    num_friend,
    period_in_months,
):
    df = df[(df['mnf'] == num_friend) & (df['pim'] == period_in_months)]
    num_tweet_per_period_list = df['ntpp'].unique()
    num_tweet_per_period_list = [_ for _ in num_tweet_per_period_list if _ in ALLOWED_TWEETS]

    data = {}
    for num_tweet_per_period in num_tweet_per_period_list:
        _ = df[df['ntpp'] == num_tweet_per_period]
        data[num_tweet_per_period] = get_performance(_)
    caption = f'num_friend: {num_friend}, period_in_months:{period_in_months}'
    label = f'vary_tweet_f_{num_friend}_p_{period_in_months}'
    table = generate_table(data, get_settings(df), num_tweet_per_period_list, experimental_name = 'tweet', caption = caption, label = label)

    
    return table


def plot_by_period(
    df,
    num_tweet_per_period,
    num_friend,
):
    df = df[(df['ntpp'] == num_tweet_per_period) & (df['mnf'] == num_friend)]
    period_in_months_list = df['pim'].unique()
    period_in_months_list = [_ for _ in period_in_months_list if _ in ALLOWED_MONTHS]
    data = {}
    for period_in_months in period_in_months_list:
        _ = df[df['pim'] == period_in_months]
        data[period_in_months] = get_performance(_)
    caption = f'num_tweet_per_period: {num_tweet_per_period}, num_friend:{num_friend}'
    label = f'vary_period_t_{num_tweet_per_period}_f_{num_friend}'
    table = generate_table(data, get_settings(df), period_in_months_list, experimental_name = 'months', caption = caption, label = label)
    return table

def plot_all_by_num_friend():
    # Vary by num_friend
    tables = []
    for num_tweet_per_period in NUM_TWEETS_PER_PERIOD_LIST:
        for period_in_months in PERIOD_IN_MONTHS_LIST:
            table = plot_by_num_friend(df, num_tweet_per_period, period_in_months)
            tables.append(table)
    return tables
def plot_all_by_num_tweet():
    tables = []
    for num_friend in NUM_FRIEND_LIST:
        for period_in_months in PERIOD_IN_MONTHS_LIST:
            table = plot_by_num_tweet(df, num_friend, period_in_months)
            tables.append(table)
    return tables

def plot_all_by_period():
    # Vary by period
    tables = []
    for num_tweet_per_period in NUM_TWEETS_PER_PERIOD_LIST:
        for num_friend in NUM_FRIEND_LIST:
            table = plot_by_period(df, num_tweet_per_period, num_friend)
            tables.append(table)
    return tables

def write_tables(filename, tables):
    filepath = Path(f"tables/{filename}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding ="utf-8") as f:
        for table in tables:
            f.write(table)
table_friend = plot_all_by_num_friend()
table_tweet = plot_all_by_num_tweet()
table_period = plot_all_by_period()
write_tables('num_friend.txt', table_friend)
write_tables('num_tweet.txt', table_tweet)
write_tables('period.txt', table_period)
tables = table_friend + table_tweet + table_period
write_tables('total.txt', tables)

exit()
for key, group_df in df.groupby(['ntpp', 'mnf', 'pim']):
    num_tweets_per_period, max_num_friends, periods_in_months = key
    models = group_df['model'].unique()
    sep = " " * 4
    print('num_tweets_per_period:', num_tweets_per_period)
    print('max_num_friends', max_num_friends)
    print('periods_in_months', periods_in_months)
    for model in models:
        model_df = group_df[group_df['model'] == model][metrics].max(axis = 0)
        part1 = []
        part2 = []

        for metric in metrics_for_depressed:
            title = f'{column_mapping[metric]}'
            score = f'{model_df[metric]:.3f}'.rjust(len(title))
            part1.append(title)
            part2.append(score)
        part1 = sep.join(part1)
        part2 = sep.join(part2)

        part3 = []
        part4 = []
        for metric in metrics_for_control:
            title = f'{column_mapping[metric]}'
            score = f'{model_df[metric]:.3f}'.rjust(len(title))
            part3.append(title)
            part4.append(score)

        part3 = sep.join(part3)
        part4 = sep.join(part4)

        group_sep = " " * 4
        line1 = f'{part1}{group_sep}{part3}'
        line2 = f'{part2}{group_sep}{part4}'
        model = model.center(len(line1), '-')
        depressed_headline = 'Depressed'.center(len(part1), '-')
        healthy_headline = 'Healthy'.center(len(part3), '-')

        print(f'{model}\n{depressed_headline}{group_sep}{healthy_headline}\n{line1}\n{line2}\n')
