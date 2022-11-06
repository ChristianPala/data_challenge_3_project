# Auxiliary library to time the execution of a function or a method:
# Libraries:
import datetime
import time
from functools import wraps
from typing import Callable, Any


# Functions:
def measure_time(func: Callable) -> Callable:
    @wraps(func)
    def timeit_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {str(datetime.timedelta(seconds=total_time))} to complete')
        return result

    return timeit_wrapper
