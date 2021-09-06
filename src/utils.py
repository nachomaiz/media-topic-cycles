import numpy as np
import pandas as pd

# from operator import attrgetter, implemented custom version to handle missing values
def attrgetter(*items):
    if len(items) == 1:
        attr = items[0]
        def g(obj):
            return resolve_attr(obj, attr)
    else:
        def g(obj):
            return tuple(resolve_attr(obj, attr) for attr in items)
    return g

def resolve_attr(obj, attr, default = np.nan):
    for name in attr.split("."):
        try:
            obj = getattr(obj, name)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return obj

def merge_retweet_full_text(data: pd.DataFrame) -> pd.DataFrame:
    """Replaces truncated retweet text in full_text with the true full text"""
    data = data.copy()
    data['is_retweet'] = ~pd.isna(data['retweeted_status.full_text'])
    data.loc[~data['retweeted_status.full_text'].isna(),'full_text'] = data.loc[~pd.isna(data['retweeted_status.full_text']),'retweeted_status.full_text']
    return data