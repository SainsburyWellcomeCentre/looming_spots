import pandas as pd
import numpy as np

file_path = '/home/slenzi/Downloads/updated_loom_sheet_format.csv'


def load_df(file_path=file_path):
    df = pd.DataFrame.from_csv(file_path)
    return df


def get_mouse_ids_in_experiment(df, experiment_key):
    exp_df = get_experiment_subset_df(df, experiment_key)
    mouse_ids = get_mouse_ids(exp_df)
    return list(mouse_ids)


def get_mouse_ids(df, ignore_future_experiments=True):
    df = df[df['result'] != 'tbd']
    mouse_ids = np.array(df['mouse_id'])
    mouse_ids = np.unique(mouse_ids)
    return mouse_ids


def get_experiment_subset_df(df, experiment_key):
    exp_df = df[df['experiment'] == experiment_key]
    return exp_df


def get_incomplete_experiments(df):
    unfinished_experiment_df = df[df['result'] == 'tbd']
    filtered_unfinished_experiment_df = unfinished_experiment_df[['mouse_id', 'experiment', 'test_date']]
    return filtered_unfinished_experiment_df


def query_df(df, key, value):
    subset_df = df[df[key] == value]
    return subset_df


def query_df_exclude(df, key, value):
    subset_df = df[df[key] != value]
    return subset_df


def get_result_from_mouse_ids(mouse_ids, df):
    for mid in mouse_ids:
        print(df[df['mouse_id'] == mid][['mouse_id', 'test_type', 'context', 'delay', 'result']])



# TODO: function that takes a dictionary and returns the entries with all criteria met