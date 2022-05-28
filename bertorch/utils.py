# coding=utf-8

import json
import os
import requests
import tempfile
from filelock import FileLock
from functools import partial
from hashlib import sha256
from tqdm.auto import tqdm
from typing import BinaryIO, Optional

VOCAB_NAME = "vocab.txt"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
SPECIAL_TOKENS_MAP_NAME = "special_tokens_map.json"

_default_endpoint = "https://huggingface.co"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", _default_endpoint)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"


def hf_bucket_url(model_id: str, filename: str):
    revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


def http_get(url: str, temp_file: BinaryIO):
    """
    Download remote file.
    """
    r = requests.get(url, stream=True)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        desc="Downloading",
        disable=False
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()
    
    
def cached_path(url: str, filename: str, cache_dir: str, etag_timeout: int = 10):
    """
    Given a URL, download the file and cache it, and return the path to the cached file.
    """
    url_to_download = url
    etag = None

    try:
        r = requests.head(url, allow_redirects=False, timeout=etag_timeout)
        r.raise_for_status()
        etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
        if etag is None:
            raise OSError(
                "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
            )
        if 300 <= r.status_code <= 399:
            url_to_download = r.headers["Location"]
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        # Actually raise for those subclasses of ConnectionError
        raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Otherwise, our Internet connection is down.
        # etag is None
        pass
    
    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        return cache_path
    
    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path):
            # Even if returning early like here, the lock will be released.
            return cache_path
        
        temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
        
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            http_get(url_to_download, temp_file)
        
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".url"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path
