import pandas as pd

def load_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                path,
                engine="python",       # more tolerant
                on_bad_lines="skip",   # or "skip"
                encoding=enc
            )
        except UnicodeDecodeError:
            continue

    return pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )

