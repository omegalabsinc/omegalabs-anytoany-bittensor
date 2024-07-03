from tempfile import TemporaryDirectory
from collections import OrderedDict

class TempDirCache:
    def __init__(self, max_cached_dirs):
        self._max_cached_dirs = max_cached_dirs
        assert self._max_cached_dirs > 0
        self._cache = OrderedDict()

    def get_temp_dir(self, key):
        if key not in self._cache:
            print(f'Adding {key} to cache')
            self._cache[key] = TemporaryDirectory()

        # maintain LRU ordering and size
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cached_dirs:
            k, _ = self._cache.popitem(last=False)
            print(f'Removing {k} from cache')

        return self._cache[key].name

