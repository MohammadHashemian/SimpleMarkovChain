from pathlib import Path

import pandas as pd


def store(df: pd.DataFrame, path: Path, override: bool = False) -> None:
    """Store the given DataFrame to a CSV file at the specified path."""

    path = Path(path)  # type: ignore # ensure Path object

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # File existence check (correct way)
    if path.exists() and not override:
        print(f"File '{path}' already exists. Set override=True to overwrite.")
        return

    df.to_csv(path, index=False)

    print(f"DataFrame successfully stored at '{path}'.")
