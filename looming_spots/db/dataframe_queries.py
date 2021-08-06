import operator

import numpy as np


def query_df(df, key, value):
    subset_df = df[df[key] == value]
    return subset_df


def query_df_exclude(df, key, value):
    subset_df = df[df[key] != value]
    return subset_df


def comparison_from_str(a, cmp_str, b):
    """
    :param a: a subset dataframe
    :param cmp_str: a comparison string (e.g. == something)
    :param b:
    :return:
    """

    comparators = get_comparator_functions()

    b = cast_to_same_type(b, a)
    if isinstance(b, list):
        return a.isin(b)
    return comparators[cmp_str](a, b)


def cast_to_same_type(b, a):
    """
    Converts the query key from a string to the appropriate type in the data series
    assumes all elements of data series are of the same type
    :param b:
    :param a:
    :return:
    """

    a_type_function = type(a.iloc()[0])
    if b.startswith("["):
        b = b.strip("[]").split(",")
        return [a_type_function(b_element.strip()) for b_element in b]

    return a_type_function(b)


def get_comparator_functions():
    comparators = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "and": operator.and_,
        "or": operator.or_,
        "in": operator.contains,
    }

    return comparators


def filter_df(db, filter_dict):

    f_df = db.copy()

    for k, v in filter_dict.items():

        if k not in f_df.keys():
            raise ValueError(k)

        components = v.split(" ", 1)
        cmp_str = components[0]
        cmp_val = components[-1]
        query_result = comparison_from_str(f_df[k], cmp_str, cmp_val)
        n_results = np.count_nonzero(query_result)

        if n_results == 0:
            return []

        f_df = f_df[query_result]

    return f_df
