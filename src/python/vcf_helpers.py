import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import sys
from pathlib import Path
import math
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(str(Path().resolve().parents[2] / "src" / "python"))
from paths import models_path

def compare_distributions(train_df, df):
    common_cols = sorted(set(train_df.columns) & set(df.columns))
    
    cols_to_plot = [col for col in common_cols if col not in ["CHROM", "POS", "REF", "ALT", "GOLDEN"]]
    
    # Determine subplot grid size
    n_cols = 3
    n_rows = math.ceil(len(cols_to_plot) / n_cols)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    
    # Plot each feature
    for i, col in enumerate(cols_to_plot):
        ax = axes.flat[i]
        sns.kdeplot(train_df[col], label='Train', fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(df[col], label='Prediction', fill=True, alpha=0.5, ax=ax)
        ax.set_title(col)
        ax.legend()
    
    # Hide unused subplots
    for j in range(len(cols_to_plot), len(axes.flat)):
        axes.flat[j].set_visible(False)
    
    # Layout and title
    plt.tight_layout()
    plt.suptitle("Feature Distribution Comparison: Train vs Prediction", fontsize=16, y=1.02)
    plt.show()

def determine_tumor_sample_index(sample_names):
    # Identify tumor sample
    tumor_sample_index = None
    tumor_sample_name = None
    
    if len(sample_names) == 1:
        tumor_sample_index = 0
        tumor_sample_name = sample_names[0]
    else:
        for idx, name in enumerate(sample_names):
            if "-T" in name or "tumor" in name.lower() or "tumour" in name.lower():
                tumor_sample_index = idx
                tumor_sample_name = name
                break
    
        if tumor_sample_index is None:
            raise ValueError("Could not identify tumor sample. Make sure tumor sample ID contains '-T' or 'tumor'.")
    
    print(f"Using tumor sample: {tumor_sample_name}")

    return tumor_sample_index

def normalize_vcf_features(df):
    # --- Feature Groups ---
    log1p_cols = [
        "DP", "AD_ref", "AD_alt", "FAD_ref", "FAD_alt",
        "F1R2_ref", "F1R2_alt", "F2R1_ref", "F2R1_alt",
        "SB_ref_fwd", "SB_ref_rev", "SB_alt_fwd", "SB_alt_rev",
        "AS_SB_ref_fwd", "AS_SB_ref_rev", "AS_SB_alt_fwd", "AS_SB_alt_rev"
    ]
    other_cols = ["MBQ_ref", "MBQ_alt", "MMQ_ref", "MMQ_alt", "AF", "NALOD", "MFRL"]

    # --- Apply log1p to selected fields ---
    for col in log1p_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                bad_values = df[df[col] < 0][col]
                print(f"[Warning] Negative values in field '{col}':")
                print(bad_values.head(10))
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])

    # --- Combine all columns to scale ---
    all_scaler_cols = [col for col in log1p_cols + other_cols if col in df.columns]

    # --- Fit and apply RobustScaler ---
    robust_scaler = RobustScaler()
    df[all_scaler_cols] = robust_scaler.fit_transform(df[all_scaler_cols])

    # --- Save the single scaler ---
    joblib.dump(robust_scaler, models_path / "robust_scaler.pkl")

    return df


def normalize_vcf_features_new(df):
    # --- Feature Groups ---
    log1p_cols = [
        "DP", "AD_ref", "AD_alt", "FAD_ref", "FAD_alt",
        "F1R2_ref", "F1R2_alt", "F2R1_ref", "F2R1_alt",
        "SB_ref_fwd", "SB_ref_rev", "SB_alt_fwd", "SB_alt_rev",
        "AS_SB_ref_fwd", "AS_SB_ref_rev", "AS_SB_alt_fwd", "AS_SB_alt_rev",
        "ECNT", "NCount", "AS_UNIQ_ALT_READ_COUNT",
        "MBQ_ref", "MBQ_alt", "PL_min", "PL_std",
        "MFRL_ref", "MFRL_alt", "MMQ_ref", "MMQ_alt"
    ] 

    passthrough_cols = [
        "AF", "POPAF", "NALOD", "NLOD", "TLOD", "CONTQ", "GERMQ",
        "MPOS", "OCM", "ROQ", "GQ", "STRQ", "STRANDQ", "SEQQ", "PON"
    ]

    # --- Derived features from arrays (optional preprocessing step) ---
    if "MBQ" in df.columns:
        df["MBQ_mean"] = df["MBQ"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else np.nan)
        df["MBQ_std"] = df["MBQ"].apply(lambda x: np.std(x) if isinstance(x, (list, np.ndarray)) else np.nan)

    if "MFRL" in df.columns:
        df["MFRL_mean"] = df["MFRL"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else np.nan)
        df["MFRL_std"] = df["MFRL"].apply(lambda x: np.std(x) if isinstance(x, (list, np.ndarray)) else np.nan)

    array_agg_cols = ["MBQ_mean", "MBQ_std", "MFRL_mean", "MFRL_std"]

    # --- Apply log1p transform ---
    for col in log1p_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                bad_values = df[df[col] < 0][col]
                print(f"[Warning] Negative values in field '{col}':")
                print(bad_values.head(10))
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])

    # --- Combine all columns to scale ---
    all_scaler_cols = [col for col in (log1p_cols + passthrough_cols + array_agg_cols) if col in df.columns]

    # --- Impute missing values ---
    # df[all_scaler_cols] = df[all_scaler_cols].fillna(-1)

    # --- Fit and apply single scaler (StandardScaler recommended) ---
    scaler = StandardScaler()
    df[all_scaler_cols] = scaler.fit_transform(df[all_scaler_cols])

    # --- Save the scaler ---
    joblib.dump(scaler, models_path / "standard_scaler.pkl")

    return df


def normalize_vcf_features_for_prediction(df, robust_scaler_path: str):
    # --- Feature Groups ---
    log1p_cols = [
        "DP", "AD_ref", "AD_alt", "FAD_ref", "FAD_alt",
        "F1R2_ref", "F1R2_alt", "F2R1_ref", "F2R1_alt",
        "SB_ref_fwd", "SB_ref_rev", "SB_alt_fwd", "SB_alt_rev",
        "AS_SB_ref_fwd", "AS_SB_ref_rev", "AS_SB_alt_fwd", "AS_SB_alt_rev"
    ]
    other_cols = ["MBQ_ref", "MBQ_alt", "MMQ_ref", "MMQ_alt", "AF", "NALOD", "MFRL"]

    # --- Apply log1p to relevant fields ---
    for col in log1p_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])

    # --- Combine all columns to scale ---
    all_scaler_cols = [col for col in log1p_cols + other_cols if col in df.columns]

    # --- Load and apply RobustScaler ---
    robust_scaler = joblib.load(robust_scaler_path)
    df[all_scaler_cols] = robust_scaler.transform(df[all_scaler_cols])

    return df

def normalize_vcf_features_for_prediction_new(df):
    # --- Reuse same transformation logic ---
    log1p_cols = [
        "DP", "AD_ref", "AD_alt", "FAD_ref", "FAD_alt",
        "F1R2_ref", "F1R2_alt", "F2R1_ref", "F2R1_alt",
        "SB_ref_fwd", "SB_ref_rev", "SB_alt_fwd", "SB_alt_rev",
        "AS_SB_ref_fwd", "AS_SB_ref_rev", "AS_SB_alt_fwd", "AS_SB_alt_rev",
        "ECNT", "NCount", "AS_UNIQ_ALT_READ_COUNT",
        "MBQ_ref", "MBQ_alt", "PL_min", "PL_std",
        "MFRL_ref", "MFRL_alt", "MMQ_ref", "MMQ_alt"
    ] 

    passthrough_cols = [
        "AF", "POPAF", "NALOD", "NLOD", "TLOD", "CONTQ", "GERMQ",
        "MPOS", "OCM", "ROQ", "GQ", "STRQ", "STRANDQ", "SEQQ", "PON"
    ]

    # --- Derived features from arrays ---
    if "MBQ" in df.columns:
        df["MBQ_mean"] = df["MBQ"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else np.nan)
        df["MBQ_std"] = df["MBQ"].apply(lambda x: np.std(x) if isinstance(x, (list, np.ndarray)) else np.nan)

    if "MFRL" in df.columns:
        df["MFRL_mean"] = df["MFRL"].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else np.nan)
        df["MFRL_std"] = df["MFRL"].apply(lambda x: np.std(x) if isinstance(x, (list, np.ndarray)) else np.nan)

    array_agg_cols = ["MBQ_mean", "MBQ_std", "MFRL_mean", "MFRL_std"]

    # --- Apply log1p to count fields ---    
    for col in log1p_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])

    # --- Collect all features to scale ---
    all_scaler_cols = [col for col in (log1p_cols + passthrough_cols + array_agg_cols) if col in df.columns]

    # --- Handle missing values ---
    df[all_scaler_cols] = df[all_scaler_cols].fillna(-1)

    # --- Load the saved scaler and transform ---
    scaler_path = models_path / "standard_scaler.pkl"
    scaler = joblib.load(scaler_path)

    df[all_scaler_cols] = scaler.transform(df[all_scaler_cols])

    return df

def cleanup_dataframe(df):
    # --- Drop all-null columns ---
    before_cols = set(df.columns)
    df.dropna(axis=1, how="all", inplace=True)
    after_null_cols = set(df.columns)
    dropped_null_cols = sorted(list(before_cols - after_null_cols))

    for col in df.columns:
        sample_val = df[col].dropna().iloc[0]
        if isinstance(sample_val, (list, np.ndarray)):
            print(f"Column '{col}' contains array-like data: {type(sample_val)}")

    # --- Drop single-value columns (no variance) ---
    protected_cols = {"CHROM"}  # Add other key columns here if needed
    single_value_cols = [col for col in df.columns if col not in protected_cols and df[col].nunique(dropna=False) <= 1]
    df.drop(columns=single_value_cols, inplace=True)

    # Track dropped columns
    dropped_columns = dropped_null_cols + single_value_cols
    if dropped_columns:
        print("\n Dropped empty or excluded columns:")
        for col in dropped_columns:
            print(f" - {col}")
    else:
        print("\n No empty columns were removed.")

    return df

def extract_variant_features(vcf, info_fields, format_fields, split_info_ref_alt, split_ref_alt_fields, tumor_sample_index):
    import numpy as np  # just in case

    def unpack_singleton_array(val):
        if isinstance(val, np.ndarray) and val.size == 1:
            return val.item()
        return val

    variant_data = []

    for variant in vcf:
        # Skip uncalled genotypes ("./.") for the specified sample
        gt = variant.genotypes[tumor_sample_index]  # e.g. [allele1, allele2, phased?]
        a1, a2 = gt[0], gt[1]
        if a1 is None or a2 is None or a1 < 0 or a2 < 0:
            continue
        
        alts = variant.ALT
        genotype_alleles = {a1, a2}
        for alt_index, alt_allele in enumerate(alts):
            alt_num = alt_index + 1
            if alt_num not in genotype_alleles:
                continue
            
            record = {
                "CHROM": variant.CHROM,
                "POS": variant.POS,
                "REF": variant.REF,
                "ALT": alt_allele,
            }

            # INFO fields
            for field in info_fields:
                value = variant.INFO.get(field)
                if field == "AS_SB_TABLE" and isinstance(value, str):
                    try:
                        parts = value.split("|")
                        if len(parts) > alt_index + 1:
                            ref_vals = parts[0].split(",")
                            alt_vals = parts[alt_index + 1].split(",")
                            record["AS_SB_ref_fwd"] = int(ref_vals[0]) if len(ref_vals) > 0 else None
                            record["AS_SB_ref_rev"] = int(ref_vals[1]) if len(ref_vals) > 1 else None
                            record["AS_SB_alt_fwd"] = int(alt_vals[0]) if len(alt_vals) > 0 else None
                            record["AS_SB_alt_rev"] = int(alt_vals[1]) if len(alt_vals) > 1 else None
                        else:
                            record["AS_SB_ref_fwd"] = record["AS_SB_ref_rev"] = record["AS_SB_alt_fwd"] = record["AS_SB_alt_rev"] = None
                    except Exception:
                        record["AS_SB_ref_fwd"] = record["AS_SB_ref_rev"] = record["AS_SB_alt_fwd"] = record["AS_SB_alt_rev"] = None

                elif isinstance(value, (list, tuple, np.ndarray)):
                    if field in split_info_ref_alt and len(value) >= 2:
                        record[f"{field}_ref"] = value[0]
                        record[f"{field}_alt"] = value[alt_index + 1] if len(value) > alt_index + 1 else None
                    else:
                        record[field] = value[alt_index] if alt_index < len(value) else None
                else:
                    record[field] = value

            # FORMAT fields
            for field in format_fields:
                try:
                    values = variant.format(field)
                    if values is not None:
                        value = values[tumor_sample_index]
                        value = unpack_singleton_array(value)

                        if isinstance(value, (list, tuple, np.ndarray)):
                            if field in split_ref_alt_fields:
                                if field == "AD" or field == "FAD":
                                    record[f"{field}_ref"] = value[0] if len(value) > 0 else None
                                    record[f"{field}_alt"] = value[alt_index + 1] if len(value) > alt_index + 1 else None
                                elif field == "AF":
                                    record[field] = value[alt_index] if len(value) > alt_index else None
                                elif field in {"F1R2", "F2R1"}:
                                    record[f"{field}_ref"] = value[0] if len(value) > 0 else None
                                    record[f"{field}_alt"] = value[alt_index + 1] if len(value) > alt_index + 1 else None
                            elif field == "SB" and len(value) == 4:
                                sb_labels = ["ref_fwd", "ref_rev", "alt_fwd", "alt_rev"]
                                for i, v in enumerate(value):
                                    record[f"{field}_{sb_labels[i]}"] = v
                            else:
                                record[field] = value  # keep full array if not handled explicitly
                        else:
                            record[field] = value
                    else:
                        record[field] = None
                except (KeyError, IndexError):
                    record[field] = None

            variant_data.append(record)

    return pd.DataFrame(variant_data)
