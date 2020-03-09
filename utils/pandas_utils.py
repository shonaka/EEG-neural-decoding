import pandas as pd


def show_table_with_stats(df):
    temp = df
    temp.loc['mean'] = temp.mean()
    temp.loc['min'] = temp.min()
    temp.loc['max'] = temp.max()
    temp.loc['std'] = temp.std()
    return temp
