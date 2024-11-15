import os
import shutil
from tempfile import TemporaryDirectory
from collections import OrderedDict

def clean_directory(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) and '.keep' not in item:
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


class TempDirCache:
    def __init__(self, max_cached_dirs, cache_parent='./model_cache'):
        self._max_cached_dirs = max_cached_dirs
        assert self._max_cached_dirs > 0
        self._cache_parent = cache_parent
        clean_directory(self._cache_parent)
        self._cache = OrderedDict()

    def get_temp_dir(self, key):
        if key not in self._cache:
            print(f'Adding {key} to cache')
            self._cache[key] = TemporaryDirectory(dir=self._cache_parent)

        # maintain LRU ordering and size
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cached_dirs:
            k, temp_dir = self._cache.popitem(last=False)
            print(f'Removing {k} from cache')
            temp_dir.cleanup()

        return self._cache[key].name

