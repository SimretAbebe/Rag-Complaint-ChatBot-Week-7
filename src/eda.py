from typing import Tuple
import pandas as pd


def summary_statistics(df: pd.DataFrame) -> dict:
    """Return concise EDA summary stats used for stakeholder report.

    Keys: original_rows, after_product_filter, after_removing_empty_narrative,
    product_counts, narrative_wordcount_stats, pct_with_narrative
    """
    original_rows = int(df.get("_original_total", len(df)))
    product_counts = df["product_category"].value_counts().to_dict()

    word_counts = df["cleaned_narrative"].dropna().apply(lambda x: len(str(x).split()))
    narrative_stats = word_counts.describe().to_dict() if not word_counts.empty else {}

    pct_with_narrative = (
        len(df[df["cleaned_narrative"].notna()]) / len(df) * 100 if len(df) > 0 else 0
    )

    return {
        "original_rows": original_rows,
        "after_filter_rows": len(df),
        "product_counts": product_counts,
        "narrative_wordcount_stats": narrative_stats,
        "pct_with_narrative": pct_with_narrative,
    }
