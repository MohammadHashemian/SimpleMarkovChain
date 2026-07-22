import warnings
import zlib
from collections.abc import Callable
from functools import wraps
from typing import Any


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


def stable_hash(*parts: Any, modulus: int = 2**32) -> int:
    """Stable cross-process hash suitable for seeding per-scenario RNGs.

    Python's built-in ``hash()`` is randomized per process (PYTHONHASHSEED),
    so deriving a per-scenario seed via ``hash((seed, scenario.name))``
    produces a *different* value in every Python process — making the
    pipeline not actually reproducible from the env seed alone.

    This helper hashes the UTF-8 concatenation of ``parts`` with the
    platform-independent CRC-32 (via zlib), so the result is identical
    across runs, platforms, and Python versions.
    """
    buf = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(buf) % modulus


def with_context(**context_factories: Callable[[], Any]):
    """
    Decorator to inject persistent context into a function.

    Each context_factory is called once at decoration time,
    and its result is injected into the function as a keyword argument.
    """

    def decorator(func: Callable):
        # Initialize context once (per decorated function instance)
        context: dict[str, Any] = {
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
