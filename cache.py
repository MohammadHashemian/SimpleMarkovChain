from pathlib import Path
from src.utils.logger import get_logger
import pickle

PROJECT_ROOT = Path(__file__).parents[0]
logger = get_logger()


def ensure_cache_dir():
    """Ensure the cache directory exists."""
    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_cache(
    cache_path: Path, expected_n_samples: int, expected_steps: int
) -> tuple | None:
    """Load cached results if valid, otherwise return None."""
    try:
        with open(cache_path, "rb") as file:
            cache_data = pickle.load(file)
        # Validate cache contents
        inputs, results, metadata = cache_data
        if (
            metadata.get("n_samples") != expected_n_samples
            or metadata.get("num_steps") != expected_steps
        ):
            logger.warning(
                f"Cache at {cache_path} has mismatched parameters. Ignoring."
            )
            return None
        return inputs, results
    except Exception as e:
        logger.error(f"Failed to load cache from {cache_path}: {e}")
        return None


def save_cache(cache_path: Path, inputs, results, n_samples: int, num_steps: int):
    """Save simulation results to cache with metadata."""
    metadata = {"n_samples": n_samples, "num_steps": num_steps}
    with open(cache_path, "wb") as file:
        pickle.dump((inputs, results, metadata), file)
    logger.info(f"Cached results at: {cache_path}")
