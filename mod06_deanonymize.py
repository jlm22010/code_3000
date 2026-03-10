import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_cols = ["age", "zip3", "gender"]

    # Keep only quasi identifier groups that occur exactly once in each dataset.
    anon_unique = anon_df[
        anon_df.groupby(quasi_cols)["anon_id"].transform("size") == 1
    ]
    aux_unique = aux_df[
        aux_df.groupby(quasi_cols)["name"].transform("size") == 1
    ]

    matches = anon_unique.merge(
        aux_unique,
        on=quasi_cols,
        how="inner",
        suffixes=("_anon", "_aux"),
    )

    return matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)
