import pandas as pd

def dataframe_describe(df):
    import pandas as pd
    df_length = df.shape[0]
    df_column_types = df.dtypes
    df_column_samples_count = df.apply(lambda x: x.count())

    df_column_uniques = df.apply(lambda x: x.unique())
    df_column_uniques_count = df.apply(lambda x: x.unique().shape[0])

    df_column_NA_count = df.apply(lambda x: x.isnull().sum())
    df_column_NA_ratio = df.apply(lambda x: x.isnull().sum() / df_length *100)

    df_column_mean = df.mean()
    df_column_std = df.std()
    df_column_assymetry = df.skew()
    df_column_peakedness = df.kurt()

    columns= {'length':df_length,
             'types':df_column_types,
             'samples_count':df_column_samples_count,
             'uniques':df_column_uniques,
             'uniques_count':df_column_uniques_count,
             'uniques_count':df_column_uniques_count,
             'NA_count':df_column_NA_count,
             'NA_ratio':df_column_NA_ratio,
             'mean':df_column_mean,
             'std':df_column_std,
             'assymetry':df_column_assymetry,
             'peakedness':df_column_peakedness}
             
    return pd.DataFrame(data=columns)