from appdirs import user_cache_dir
from pathlib import Path
import requests
import os

APP_NAME = "pandas_nlp"


def get_remote_file(
    prefix: str, name: str, url: str, force_reload: bool = False
) -> Path:
    cache_path = Path(user_cache_dir(APP_NAME))
    file_path = cache_path / prefix / name
    os.makedirs(file_path.parent, exist_ok=True)
    if not file_path.exists() or force_reload:
        response = requests.get(url)
        response.raise_for_status()
        with file_path.open("wb") as fp:
            fp.write(response.content)
    return file_path
