from pandas import Series
import pandas_nlp


def register_pandas_nlp_if_not_available():
    if not hasattr(Series, "nlp"):
        pandas_nlp.register()
