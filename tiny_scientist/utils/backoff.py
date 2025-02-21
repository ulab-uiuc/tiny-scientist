import time
from typing import Any, Callable, Dict

import backoff
import requests


def on_backoff(details: Dict[str, Any]) -> None:
    """Standard callback for backoff decorator."""
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

def with_retry(func: Callable) -> Callable:
    """Decorator to add backoff retry logic to a function."""
    return backoff.on_exception(
        backoff.expo,
        requests.exceptions.HTTPError,
        on_backoff=on_backoff
    )(func)
