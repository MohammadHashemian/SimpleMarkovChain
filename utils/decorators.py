from typing import Callable, Any, Dict
from functools import wraps
import warnings


def deprecated(reason: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def with_context(**context_factories: Callable[[], Any]):
    """
    Decorator to inject persistent context into a function.

    Each context_factory is called once at decoration time,
    and its result is injected into the function as a keyword argument.
    """

    def decorator(func: Callable):
        # Initialize context once (per decorated function instance)
        context: Dict[str, Any] = {
            name: factory() for name, factory in context_factories.items()
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject context into kwargs (without overwriting explicit args)
            for key, value in context.items():
                kwargs.setdefault(key, value)
            return func(*args, **kwargs)

        return wrapper

    return decorator
