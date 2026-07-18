def get_project_root():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return root
