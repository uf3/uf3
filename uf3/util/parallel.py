import time
import warnings

import numpy as np
import pandas as pd
from concurrent import futures

try:
    import tqdm
    ProgressBar = tqdm.tqdm

    def progress_iter(iterable, **kwargs):
        tqdm.tqdm._instances.clear()
        return tqdm.tqdm(iterable, **kwargs)
except ImportError:  # optional import
    tqdm = None

    def progress_iter(iterable, **kwargs):
        return iterable

    class ProgressBar:
        """Makeshift progress counter."""
        def __init__(self, total=None):
            if total is None:
                self.total = "?"
                self.pad = 6
            else:
                self.total = str(total)
                self.pad = len(self.total)
            self.counter = 0

        def update(self, i):
            self.counter += i
            print(str(self.counter).ljust(self.pad),
                  "/",
                  self.total.ljust(self.pad), end='\r')

        def close(self):
            pass

try:
    from dask import distributed
    USE_DASK = True
except ImportError:
    distributed = None
    USE_DASK = False


def wait_progress(future_list, timeout=None):
    """
    Visualize progress while waiting for futures to complete.

    Args:
        future_list (list)
        timeout (float): seconds to wait before timing out.
    """
    t0 = time.time()

    n = len(future_list)
    future_progress = ProgressBar(total=n)
    last = 0

    if isinstance(future_list[0], futures.Future):
        wait = futures.wait
    elif isinstance(future_list[0], distributed.Future):
        wait = distributed.wait
    else:
        raise ValueError("Unsupported future type.")

    while True:
        if timeout is not None and ((time.time() - t0) > timeout):
            warnings.warn("Timeout reached: {} seconds.".format(timeout),
                          RuntimeWarning)
            break
        done, remaining = wait(future_list,
                               return_when='FIRST_COMPLETED')
        done = len(done)
        remain = len(remaining)
        future_progress.update(done - last)
        last = int(done)
        if remain < 1:
            break
    future_progress.close()


def split_zip(n_batches, *args):
    """
    General splitting function for one or more list-like objects.

    Args:
        n_batches (int): number of desired batches.
        *args: one or more list-like objects of equal length.

    Returns:
        batches (list): list of batched data. Resulting batches
            may not be equal in size.
    """
    batches = []
    for arg in args:
        idxs = np.arange(len(arg))
        n_batches_ = min(n_batches, len(idxs))
        split_idxs = np.array_split(idxs, n_batches_)
        batches.append([[arg[idx] for idx in batch_idxs]
                        for batch_idxs in split_idxs])
    return batches


def split_dataframe(df, n_batches):
    """
    Split dataframe using np.array_split.

    Args:
        df (pd.DataFrame)
        n_batches (int): number of desired batches.

    Returns:
        batches (list): list of batched data. Resulting batches
            may not be equal in size.
    """
    batches = []
    idxs = np.arange(len(df.index))
    n_batches = min(n_batches, len(idxs))

    split_idxs = np.array_split(idxs, n_batches)
    for batch_idxs in split_idxs:
        batches.append(df.iloc[batch_idxs])
    return batches


def batch_submit(func, batches, client, *args, **kwargs):
    """
    Args:
        func: function to call using client.
        batches (list): list of input(s) per batch,
            e.g. from split_dataframe or split_zip
        client (concurrent.futures.Executor, dask.distributed.Client)
        *args: arguments to pass to func.
        **kwargs: keyword arguments to pass to func.

    Returns:
        future_list (list)
    """
    future_list = []
    for batch in batches:
        future = client.submit(func, batch, *args, **kwargs)
        future_list.append(future)
    return future_list


def gather_and_merge(future_list, client=None, cancel=False):
    """
    Args:
        future_list (list): list of futures.
        client (concurrent.futures.Executor, dask.distributed.Client)
        cancel (bool): whether to cancel futures after gathering.

    Returns:
        result: merged result, if datatype is supported, or list of results.
    """
    wait_progress(future_list)
    try:  # more efficient but not implemented in concurrent.futures
        item_list = client.gather(future_list)
    except AttributeError:
        item_list = [future.result() for future in future_list]
    # merge according to data type
    if isinstance(item_list[0], dict):
        result = {}
        for sub_dict in item_list:
            result.update(sub_dict)
    elif isinstance(item_list[0], np.ndarray):
        try:
            result = np.concatenate(item_list)
        except ValueError:
            warnings.warn("Unable to merge gathered futures.", RuntimeWarning)
            result = item_list
    elif isinstance(item_list[0], pd.DataFrame):
        result = pd.concat(item_list, join='inner', copy=False)
    else:
        warnings.warn("Unable to merge gathered futures.", RuntimeWarning)
        result = item_list
    if cancel:
        try:  # more efficient but not implemented in concurrent.futures
            client.cancel(future_list)
        except:
            for future in future_list:
                future.cancel()
    return result
