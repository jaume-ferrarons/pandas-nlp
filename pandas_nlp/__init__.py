import pkg_resources


__version__ = pkg_resources.get_distribution("pandas_nlp").version

from .series_accessor import register
