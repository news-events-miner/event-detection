import datetime as dt
from csv import DictReader
from typing import Optional, Callable, List


class CsvDataset:
    """
    Dataloader that yields batches from csv using tumbling time window
    """

    def __init__(self, fname: str):
        self.fname = fname

    def __call__(self,
                 batch_size: int = 0,
                 time_col: str = "date",
                 time_window: Optional[dt.timedelta] = None,
                 columns: Optional[List[str]] = None,
                 time_format: Optional[str] = None,
                 drop_last: bool = True,
                 text_column: str = "text",
                 max_size: int = 0,
                 filter_func: Optional[Callable[dict, bool]] = None):
        batch = []
        start_time = None

        assert (time_window is not None and time_col is not None
                and time_format is not None) or batch_size != 0

        if filter_func is None:
            # Use a stub
            filter_func = (lambda _: True)

        with open(self.fname, "r") as fd:
            to_yield = False
            reader = DictReader(fd)

            for line in reader:
                cur_time = dt.datetime.strptime(line[time_col],
                                                time_format).date()

                if start_time is None:
                    batch = []
                    start_time = cur_time

                if len(batch) > 0:
                    window_edge = time_window is not None and\
                            cur_time > start_time + time_window
                    batch_end = batch_size != 0 and len(batch) == batch_size
                    limit = max_size > 0 and len(batch) == max_size

                    # Split for readability
                    if window_edge or batch_end or limit:
                        to_yield = True

                if to_yield:
                    start_time = None
                    to_yield = False
                    yield batch
                else:
                    # Filter out duplicates
                    if len(batch) > 0:
                        last_text = batch[-1][text_column]
                        cur_text = line[text_column]

                        strings_equal = last_text == cur_text
                    else:
                        strings_equal = False

                    if len(batch) > 0 and not strings_equal or len(batch) == 0:
                        if filter_func(line):
                            batch.append({key: line[key] for key in columns})

        if len(batch) > 0 and not drop_last:
            yield batch
        raise StopIteration
