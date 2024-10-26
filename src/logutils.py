import functools
import inspect
import logging
import sys
import typing
from typing import Any

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] <%(filename)s:%(lineno)s - %(funcName)s()> %(message)s"
WRAPPER_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def create_wrapper_logger():
    logger = logging.getLogger("wrapper")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(WRAPPER_LOG_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


wrapper_logger: logging.Logger = create_wrapper_logger()


def logio(truncate_at: int = 100) -> typing.Callable:
    def logio_decorator(func) -> typing.Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def truncate(obj: Any, length: int = truncate_at) -> str:
                rep: str = repr(obj)
                if length < 3:
                    return rep
                return f"{rep[:length-3]}..." if len(rep) > length else rep

            logger = wrapper_logger
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            all_args = dict(zip(param_names, args))
            all_args.update(kwargs)
            allargs_repr = [f"{k}={truncate(v)}" for k, v in all_args.items()]
            signature = ", ".join(allargs_repr)
            logger.debug(f">>> {func.__name__}({signature})")

            result = func(*args, **kwargs)
            logger.debug(f"<<< {func.__name__}: {truncate(result)}")

            return result
        return wrapper
    return logio_decorator

