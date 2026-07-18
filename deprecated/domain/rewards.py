from utils.decorators import deprecated, with_context


@deprecated("use make_pettersson_score for mutable closure version")
@with_context(hemarthrosis_cache=lambda: {"count": 0})
def pettersson_score(step: int, state: str, **kwargs) -> int:
    const = kwargs["const"]
    cache = kwargs["hemarthrosis_cache"]
    if state == "hemarthrosis":
        k = kwargs.get("event_count", 0)
        cache["count"] += k
    score = int(cache["count"] / const["conversion_factor"])
    return min(int(score), 79)
