import math

import numpy as np
from sklearn.preprocessing import normalize


def normalize_to_sum_to_one(array: list[float] | np.ndarray) -> np.ndarray:
    if isinstance(array, list):
        array = np.array(array)
    if len(array.shape) > 1:
        raise ValueError("Array have more than 1-Dimension")
    if np.any(array < 0):
        raise ValueError("Array contains negative values")
    normalized = normalize(array.reshape(-1, 1), norm="l1", axis=0).ravel()
    return normalized


def count_bleeds_conditional_prob(state: str, **kwargs) -> int:
    def conditional_probs(k: int, lam: float):
        return (lam**k) / (math.factorial(k) * (math.exp(lam) - 1))

    lambda_value = (
        kwargs.get("lambda_bleeding")
        if state.lower() == "bleeding"
        else (
            kwargs.get("lambda_joint_bleeding")
            if state.lower() == "joint_bleeding"
            else 0
        )
    )
    number_of_bleeds = 1
    if lambda_value != 0:
        if not isinstance(lambda_value, float):
            raise TypeError("No valid lambda value provided.")
        events_probs = [conditional_probs(k, lambda_value) for k in range(1, 5, 1)]
        events_probs = [p / sum(events_probs) for p in events_probs]
        number_of_bleeds = np.random.choice([i for i in range(1, 5, 1)], p=events_probs)
    return number_of_bleeds
