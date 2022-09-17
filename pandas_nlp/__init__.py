import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("pandas_nlp").version
except pkg_resources.DistributionNotFound:
    __version__ = "dev"

from .series_accessor import NLPAccessor
