from pathlib import Path

import pandas as pd

from utils.logging import setup_root_logger


def show(
    df: pd.DataFrame,
    caption: str | None = None,
    format: dict | None = None,
    store: bool = False,
    options: dict = {},
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        caption (str | None, optional): _description_. Defaults to None.
        format (dict | None, optional): _description_. Defaults to None.
        store (bool, optional): _description_. Defaults to False.
        options (dict | None, optional): _description_. Defaults to None.
    """
    style = df.style.set_table_attributes('style="font-size:12px; table-layout:fixed;"')

    if caption:
        style = style.set_caption(caption)

    if format:
        style = style.format(format)

    from IPython.display import display

    display(style)
    logger = setup_root_logger()
    try:
        if store:
            storage = options.get("storage", None)
            override = options.get("override", False)
            if storage:
                path: Path = storage.get("excel_writer", None)
                if not path:
                    logger.warning(
                        "Excel writer path not provided. DataFrame not stored."
                    )
                    return
                if path.exists() and not override:
                    logger.warning(
                        f"File {path} already exists. DataFrame not stored to avoid overwriting."
                    )
                    return
                df.to_excel(**storage)
                logger.info(f"DataFrame stored successfully at {path}")
            else:
                logger.warning("Storage options not provided. DataFrame not stored.")
    except Exception as e:
        logger.error(f"Error storing DataFrame: {e}")
