import pathlib
import numpy as np
import random
import time


class OfflineDataSequential:
    """Offline data which processes episodes sequentially"""

    def __init__(self, input_dir, reload_interval=300):
        self.input_dir = pathlib.Path(input_dir)
        self.reload_interval = reload_interval
        self._reload_files()

    def _reload_files(self):
        self._files = list(sorted(self.input_dir.glob('*.npz')))
        self._last_reload = time.time()
        print(f'Found data: {len(self._files)} episodes in {str(self.input_dir)}')

    def _should_reload_files(self):
        return time.time() - self._last_reload > self.reload_interval

    def iterate(self, batch_length, batch_size):
        # Parallel iteration over (batch_size) iterators
        # Iterates forever

        iters = [self._iter_single(batch_length) for _ in range(batch_size)]
        for batches in zip(*iters):
            batch = {}
            for key in batches[0]:
                batch[key] = np.stack([b[key] for b in batches]).swapaxes(0, 1)
            yield batch

    def _iter_single(self, batch_length):
        # Iterates "single thread" forever
        # TODO: join files so we don't miss the last step indicating done

        is_first = True
        for file in self._iter_shuffled_files():
            for batch in self._iter_file(file, batch_length, skip_random=is_first):
                yield batch
            is_first = False

    def _iter_file(self, file, batch_length, skip_random=False):
        try:
            with file.open('rb') as f:
                fdata = np.load(f)
                data = {key: fdata[key] for key in fdata}
        except Exception as e:
            print('Error reading file - skipping')
            print(e)
            return

        n = data['image'].shape[0]
        data['reset'] = np.zeros(n, bool)
        data['reset'][0] = True  # Indicate episode start

        i_start = 0
        if skip_random:
            i_start = np.random.randint(n - batch_length)

        for i in range(i_start, n - batch_length + 1, batch_length):
            # TODO: should return last shorter batch
            j = i + batch_length
            batch = {key: data[key][i:j] for key in data}
            yield batch

    def _iter_shuffled_files(self):
        while True:
            i = random.randint(0, len(self._files) - 1)
            f = self._files[i]
            if not f.exists() or self._should_reload_files():
                self._reload_files()
            else:
                yield self._files[i]


class OfflineDataRandom:
    """Offline data with random sampling from middle of episodes"""

    def __init__(self, input_dir):
        input_dir = pathlib.Path(input_dir)
        self._files = list(sorted(input_dir.glob('*.npz')))
        print(f'Offline data: {len(self._files)} episodes in {str(input_dir)}, loading...')
        self._data = self._load_data()

    def _load_data(self):
        all_data = []
        for path in self._files:
            with path.open('rb') as f:
                fdata = np.load(f)
                data = {key: fdata[key] for key in fdata}
                all_data.append(data)
        return all_data

    def iterate(self, batch_length, batch_size):
        while True:
            yield self._sample_batch(batch_length, batch_size)

    def _sample_batch(self, batch_length, batch_size):
        sequences = [self._sample_single(batch_length) for _ in range(batch_size)]
        batch = {}
        for key in sequences[0]:
            batch[key] = np.stack([b[key] for b in sequences]).swapaxes(0, 1)
        return batch

    def _sample_single(self, batch_length):
        i_episode = np.random.randint(len(self._data))
        data = self._data[i_episode]

        # TODO: this sampling undersamples episode starts and ends
        n = data['image'].shape[0]
        i = np.random.randint(n - batch_length)
        j = i + batch_length
        batch = {key: data[key][i:j] for key in data}
        return batch
