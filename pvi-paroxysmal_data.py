import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Read the carto ECG metadata CSV file
carto_metadata = pd.read_csv("carto_ecg_metadata.csv", parse_dates=["acquisition_date"])

# Display basic information about the dataset
print("Dataset shape:", carto_metadata.shape)

print("\nColumn names:")
print(carto_metadata.columns.tolist())

print("\nData types:")
print(carto_metadata.dtypes)
carto_metadata.head()

# Display the results
print("Sinus rhythm detection results:")
print(f"Total records: {len(carto_metadata)}")
print(f"Records with sinus rhythm: {carto_metadata['sinus_rhythm'].sum()}")
print(f"Records without sinus rhythm: {(carto_metadata['sinus_rhythm'] == 0).sum()}")

# Read the codebook file
codebook = pd.read_csv("ecg_codebook.csv")

# Display basic information about the codebook
print("Codebook shape:", codebook.shape)
print("\nCodebook columns:", codebook.columns.tolist())
print("\nFirst few rows of codebook:")
print(codebook.head())

# Create ecg_id column from filename by removing .xml suffix
carto_metadata["ecg_id"] = carto_metadata["filename"].str.replace(
    ".xml", "", regex=False
)

# Merge the dataframes on ecg_id to add MRN
carto_metadata_with_mrn = carto_metadata.merge(
    codebook[["ecg_id", "mrn"]], on="ecg_id", how="left"
)

# Display the results
print(f"\nOriginal metadata shape: {carto_metadata.shape}")
print(f"Merged metadata shape: {carto_metadata_with_mrn.shape}")
print(f"Records with MRN: {carto_metadata_with_mrn['mrn'].notna().sum()}")
print(f"Records without MRN: {carto_metadata_with_mrn['mrn'].isna().sum()}")

# Drop the ecg_id column from the merged dataframe
carto_metadata_with_mrn = carto_metadata_with_mrn.drop("ecg_id", axis=1)


carto_metadata_with_mrn.to_csv("carto_ecg_metadata_FULL.csv", index=False)

# ------------------------------------------------------------------------------------------------


# Configuration - Easy to edit cutoff period
PRE_ECG_CUTOFF_YEARS = 1  # Change this value to adjust the cutoff period

# Load the data
data = pd.read_csv("subset.csv")

# Display the first few rows of the data
data.head()


ecg_data = pd.read_csv("carto_ecg_metadata_FULL.csv")
ecg_data.head()

# Count unique MRNs in the data
unique_mrns = data["MRN"].nunique()
print(f"Number of unique MRNs in data: {unique_mrns}")

# Count unique MRNs in the data
unique_mrns = ecg_data["mrn"].nunique()
print(f"Number of unique MRNs in ECGs: {unique_mrns}")

# Get unique MRNs from both datasets
data_mrns = set(data["MRN"].unique())
ecg_mrns = set(ecg_data["mrn"].unique())

# Show some statistics
print(f"MRNs only in data: {len(data_mrns - ecg_mrns)}")
print(f"MRNs only in ECGs: {len(ecg_mrns - data_mrns)}")
print(f"MRNs in both datasets: {len(data_mrns.intersection(ecg_mrns))}")

# Filter ECG data to only include MRNs that are in the main data
filtered_ecg_data = ecg_data[ecg_data["mrn"].isin(data["MRN"])]

print(f"Original ECG data rows: {len(ecg_data)}")
print(f"Filtered ECG data rows: {len(filtered_ecg_data)}")
print(
    f"Number of unique MRNs in filtered ECG data: {filtered_ecg_data['mrn'].nunique()}"
)

# Convert date columns to datetime
print("\nProcessing dates...")
data["Procedure date"] = pd.to_datetime(data["Procedure date"])
ecg_data["acquisition_date"] = pd.to_datetime(ecg_data["acquisition_date"])

# Create a mapping from MRN to procedure date
mrn_to_procedure_date = dict(zip(data["MRN"], data["Procedure date"]))


# # Function to determine if ECG is pre-procedure
# def is_pre_ecg(row):
#     mrn = row["mrn"]
#     ecg_date = row["acquisition_date"]

#     # Get procedure date for this MRN
#     procedure_date = mrn_to_procedure_date.get(mrn)

#     if procedure_date is None:
#         return 0  # No procedure date found

#     # Calculate time difference
#     time_diff = procedure_date - ecg_date

#     # Check if ECG is before procedure date and within cutoff period
#     if time_diff.days >= 0 and time_diff.days <= (PRE_ECG_CUTOFF_YEARS * 365):
#         return 1
#     else:
#         return 0


# # Apply the function to create pre_ecg column
# print("Creating pre_ecg column...")
# ecg_data["pre_ecg"] = ecg_data.apply(is_pre_ecg, axis=1)

# # For each MRN, find the earliest ECG before the procedure and mark it as pre_ecg=1
# # But only if it's within the cutoff period, OR if there are no ECGs within the cutoff period
# print("Identifying earliest ECGs for each MRN...")
# for mrn in ecg_data["mrn"].unique():
#     if mrn in mrn_to_procedure_date:
#         procedure_date = mrn_to_procedure_date[mrn]

#         # Get all ECGs for this MRN that are before the procedure date
#         mrn_ecgs = ecg_data[
#             (ecg_data["mrn"] == mrn) & (ecg_data["acquisition_date"] < procedure_date)
#         ]

#         if not mrn_ecgs.empty:
#             # Check if any ECGs are already marked as pre_ecg=1 (within cutoff period)
#             ecgs_within_cutoff = mrn_ecgs[mrn_ecgs["pre_ecg"] == 1]

#             if ecgs_within_cutoff.empty:
#                 # No ECGs within cutoff period, so mark the earliest ECG as pre_ecg=1
#                 earliest_ecg_idx = mrn_ecgs["acquisition_date"].idxmin()
#                 ecg_data.loc[earliest_ecg_idx, "pre_ecg"] = 1

# # Show results
# print(f"\nResults:")
# print(f"Total ECGs: {len(ecg_data)}")
# print(f"ECGs marked as pre_ecg=1: {ecg_data['pre_ecg'].sum()}")
# print(
#     f"Percentage of ECGs marked as pre_ecg: {(ecg_data['pre_ecg'].sum() / len(ecg_data)) * 100:.1f}%"
# )

# # Show breakdown by MRN
# pre_ecg_by_mrn = ecg_data.groupby("mrn")["pre_ecg"].sum()
# print(f"\nMRNs with at least one pre_ecg: {(pre_ecg_by_mrn > 0).sum()}")
# print(f"Average pre_ecgs per MRN: {pre_ecg_by_mrn.mean():.2f}")

# Merge with main data to add recurrence information
print("\nMerging with main data to add recurrence columns...")

# Clean up column names for the recurrence data
recurrence_columns = {
    "AF recurrence after blanking period (60 days)?": "af_recurrence",
    "Days till AF recurrence": "days_till_af_recurrence",
    "AT recurrence after blanking period (60 days)?": "at_recurrence",
    "Days till AT recurrence": "days_till_at_recurrence",
    "AF/AT Recurrence": "af_at_recurrence",
    "Days till AF/AT Recurrence": "days_till_af_at_recurrence",
}

# Create a subset of the main data with just MRN, procedure date, and recurrence columns
recurrence_data = data[
    ["MRN", "Procedure date"] + list(recurrence_columns.keys())
].copy()

# Rename columns to cleaner names
recurrence_data = recurrence_data.rename(columns=recurrence_columns)
recurrence_data = recurrence_data.rename(
    columns={"MRN": "mrn", "Procedure date": "procedure_date"}
)

# Merge with ECG data
ecg_data_final = ecg_data.merge(recurrence_data, on="mrn", how="left")

print(f"Final ECG data shape: {ecg_data_final.shape}")
print(f"Columns in final data: {list(ecg_data_final.columns)}")

# Save the updated ECG data
ecg_data_final.to_csv("carto_ecg_metadata_FULL.csv", index=False)
