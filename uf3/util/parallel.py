"""
This module provides functions for splitting/merging collections,
handling parallel tasks, and managing progress indicators.
"""
import datetime
import time
import warnings
import numpy as np
import pandas as pd
from concurrent import futures
from tqdm.auto import tqdm

try:
    from dask import distributed
    USE_DASK = True
except ImportError:
    from concurrent import futures as distributed
    USE_DASK = False


class ProgressText:
    """Makeshift progress counter."""
    def __init__(self, iterable=None, total=None):
        self.iterable = iterable
        self.start_time = datetime.datetime.now()
        start_str = self.start_time.strftime("%H:%M:%S")
        print(start_str, "START")

        try:
            self.total = len(iterable)
            self.pad = len(str(self.total))
        except (TypeError, AttributeError):
            if total is None:
                self.total = "?"
                self.pad = 6
            else:
                self.total = total
                self.pad = len(str(self.total))
        self.counter = 0

    def __iter__(self):
        iterable = self.iterable
        try:
            for obj in iterable:
                yield obj
                self.update(1)
        finally:
            self.close()

    def update(self, i=1):
        self.counter += i
        elapsed = datetime.datetime.now() - self.start_time
        if self.total != "?":
            progress = f"{100 * self.counter / self.total:.2f}%"
        else:
            progress = ""
        print(f"{elapsed}    "
              f"{self.counter:>{self.pad}} / "
              f"{self.total:>{self.pad}}    {progress}")

    def close(self):
        end_time = datetime.datetime.now()
        print(end_time.strftime("%H:%M:%S"), "END")
        pass


def progress_iter(iterable=None,
                  style="bar",
                  total=None,
                  leave=True,
                  ):
    if style == "bar" or style is True:
        return tqdm(iterable, total=total, leave=leave,)
    elif style == "text" or style == "str" or style == str:
        return ProgressText(iterable, total=total)
    else:
        return iterable


def wait_progress(future_list, timeout=None, leave=True, style="bar"):
    """
    Visualize progress while waiting for futures to complete.

    Args:
        future_list (list)
        timeout (float): seconds to wait before timing out.
    """
    t0 = time.time()

    n = len(future_list)
    future_progress = progress_iter(total=n, leave=leave, style=style)
    last = 0

    wait = select_wait_function(future_list)

    while True:
        if timeout is not None and ((time.time() - t0) > timeout):
            warnings.warn("Timeout reached: {} seconds.".format(timeout),
                          RuntimeWarning)
            break
        done, remaining = wait(future_list,
                               return_when='FIRST_COMPLETED')
        done = len(done)
        remain = len(remaining)
        if done > last:
            future_progress.update(done - last)
            last = int(done)
        if remain < 1:
            break
    future_progress.close()


def select_wait_function(future_list):
    if isinstance(future_list[0], futures.Future):
        wait = futures.wait
    elif USE_DASK and isinstance(future_list[0], distributed.Future):
        wait = distributed.wait
    else:
        raise ValueError("Unsupported future type.")
    return wait


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


def gather_and_merge(future_list,
                     client=None,
                     cancel=False,
                     progress="bar",
                     asynchronous=True):
    """
    Args:
        future_list (list): list of futures.
        client (concurrent.futures.Executor, dask.distributed.Client)
        cancel (bool): whether to cancel futures after gathering.

    Returns:
        result: merged result, if datatype is supported, or list of results.
    """

    if asynchronous:
        item_list = []
        pbar = progress_iter(total=len(future_list), style=progress)
        if USE_DASK and isinstance(client, distributed.Client):
            for future, result in distributed.as_completed(
                    future_list, with_results=True):
                item_list.append(result)
                client.cancel(future)
                if progress:
                    pbar.update()
        else:
            for future in futures.as_completed(future_list):
                item_list.append(future.result())
                if progress:
                    pbar.update()
        if progress:
            pbar.close()
    else:
        if progress:
            wait_progress(future_list)
        else:
            wait = select_wait_function(future_list)
            wait(future_list)
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
        except AttributeError:
            for future in future_list:
                future.cancel()
    return result
