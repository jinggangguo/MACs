embeddedCode Python:

# -*- coding: utf-8 -*-
"""
Rewritten in a traditional (no function definitions) style.
This script reads input CSV files, performs all the data transformations,
loops over alpha values for both carbon and value‐added calculations,
runs regressions, enriches the final model results, produces plots, and
exports the final outputs to CSV and Excel files.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================================================================
# 1) Read & Transform SAM Details (IxISamDetails_ModelId.csv)
# =============================================================================
file_sam = "IxISamDetails_ModelId.csv"
df = pd.read_csv(file_sam)

# Drop unneeded columns
df = df.loc[:, ~df.columns.isin([' PayingDescription', ' ReceivingDescription',
                                   ' TransferCode', ' TransferDescription'])]

# Rename columns (strip extra spaces if needed)
df.rename(columns={' Value': 'Value', ' ReceivingCode': 'ReceivingCode'}, inplace=True)

# Reorder columns
columns_titles = ["ReceivingCode", "PayingCode", "Value"]
df = df.reindex(columns=columns_titles)

# Pivot to wide format and fill missing values with 0
wide_df = df.pivot_table(index='ReceivingCode',
                         columns='PayingCode',
                         values='Value',
                         aggfunc='first').fillna(0)

# Normalize columns (each column divided by its column sum)
wide_df = wide_df.divide(wide_df.sum(axis=0), axis=1)

# Subset: only keep rows and columns with index <= 5000
wide_df = wide_df.loc[wide_df.index <= 5000, wide_df.columns <= 5000]

# Build the (I - wide_df) inverse
I = np.identity(wide_df.shape[0])
result = I - wide_df
result_inverse = np.linalg.inv(result)
subset_wide_df = wide_df.copy()
result_inverse_df = pd.DataFrame(result_inverse, columns=wide_df.columns, index=wide_df.index)

# Re-map the index and columns to be 1..546
index_mapping = {i: i for i in range(1, 547)}
subset_wide_df.rename(index=index_mapping, columns=index_mapping, inplace=True)
result_inverse_df.rename(index=index_mapping, columns=index_mapping, inplace=True)

# Expand matrices to full shape 1..546 x 1..546, filling missing entries with 0
subset_wide_df = subset_wide_df.reindex(index=range(1, 547), columns=range(1, 547), fill_value=0)
result_inverse_df = result_inverse_df.reindex(index=range(1, 547), columns=range(1, 547), fill_value=0)

# =============================================================================
# 2) Read & Transform Environment Data (environment_US_546.csv)
# =============================================================================
file_env = "environment_US_546.csv"
df1 = pd.read_csv(file_env)

# Keep the first, second, and last columns and rename them
df1 = df1.iloc[:, [0, 1, -1]]
df1.columns = ['Environment Industry', 'Carbon_output', 'Emission_coefficient']

# Clean up the "Environment Industry" column: split by '-' and strip spaces
df1['Environment Industry'] = (df1['Environment Industry'].astype(str)
                                 .str.split('-').str[0].str.strip())

# Reindex the dataframe for indices 0..546 (fill missing rows with 0)
index_range = range(0, 547)
df1_filled = df1.reindex(index=index_range, fill_value=0)

# Convert "Environment Industry" to integer (ignoring errors)
df1_filled['Environment Industry'] = df1_filled['Environment Industry'].astype(int, errors='ignore')

# Identify missing industries (should be 1..546)
all_values = set(range(1, 547))
unique_values = set(df1_filled['Environment Industry'])
missing_values = all_values - unique_values
print("Missing environment industries:", missing_values)

# Add rows for missing environment industries with zeros
missing_values_df = pd.DataFrame({
    'Environment Industry': list(missing_values),
    'Carbon_output': 0,
    'Emission_coefficient': 0
})
df1_filled = pd.concat([df1_filled, missing_values_df], ignore_index=True)
df1_filled.sort_values(by='Environment Industry', inplace=True)
df1_filled.reset_index(drop=True, inplace=True)

# Remove any rows where "Environment Industry" is zero and fill remaining NaN with 0.05
df1_filled = df1_filled[df1_filled['Environment Industry'] != 0]
df1_filled.reset_index(drop=True, inplace=True)
df1_filled.fillna(0.05, inplace=True)
print("df1_filled after cleaning:\n", df1_filled)

# Extract carbon emission and carbon output series
carbonemission = df1_filled['Emission_coefficient'].copy()
carbon_output = df1_filled['Carbon_output'].copy()

# Build the diagonal matrix from the emission coefficients.
# (First, reassign the column names for clarity, then build a 546x546 diagonal DataFrame.)
df1.columns = ['Naics3', 'Carbon_output_temp', 'Emission_coefficient']
df1.set_index('Naics3', inplace=True)
df_diag = pd.DataFrame(np.zeros((546, 546)), index=range(1, 547), columns=range(1, 547))
np.fill_diagonal(df_diag.values, df1['Emission_coefficient'])

# =============================================================================
# 3) Read & Transform Industry Summary Data (industry_output_outlay_summary.csv)
# =============================================================================
file_industry = "industry_output_outlay_summary.csv"
df_ind = pd.read_csv(file_industry)
if not df_ind.empty:
    df_ind = df_ind.drop(df_ind.index[-1])  # drop the last row

# Clean and convert "Total Output"
total_output = df_ind[['Total Output']].copy()
total_output['Total Output'] = total_output['Total Output'].replace('[\\$,]', '', regex=True).astype(float)

# Clean the 3rd column: take the string before '-' and strip spaces
df_ind.iloc[:, 2] = df_ind.iloc[:, 2].str.split('-').str[0].str.strip()
# Clean the last column: remove dollar signs and commas then convert to float
df_ind.iloc[:, -1] = df_ind.iloc[:, -1].replace({'\$': '', ',': ''}, regex=True).astype(float)
df_ind.dropna(inplace=True)

# Keep just the 3rd and last columns and rename them
df_ind_summary = df_ind.iloc[:, [2, -1]]
df_ind_summary.columns = ['Extracted_Column_3', 'Value']
df_ind_summary.set_index('Extracted_Column_3', inplace=True)

# Use this summary as the carbon data (and make a copy for final demand)
df2_carbon = df_ind_summary.copy()
Final_demand = df2_carbon.copy()

# =============================================================================
# 4) Big Alpha Loop for the "Carbon" Portion
# =============================================================================
# Prepare arrays for computations (all matrices are 546x546 or 546x1)
diag_array = df_diag.values            # (546 x 546)
L_inv = result_inverse_df.values       # (546 x 546)
f2 = df2_carbon.values                 # (546 x 1)
A = subset_wide_df.values              # (546 x 546)

# (The following multiplication replicates the original “flow” but its result is not used.)
_ = diag_array @ L_inv @ f2

# Set up alpha values and empty lists to collect results
alpha_vals = np.arange(0.01, 1.01, 0.03)
row_idx_list = []
alpha_list = []
delE_list = []

n_sectors = f2.shape[0]
for i in range(n_sectors):
    # Create one-hot vector for row i
    ek21 = np.zeros(n_sectors)
    ek21[i] = 1

    # Get the i-th row of A, and zero out the diagonal element
    bk_row = A[i, :].copy()
    bk_row[i] = 0

    # Pre-calculate some quantities
    L_inv_ek21 = L_inv @ ek21
    L_inv_f2 = L_inv @ f2
    Ec_L = diag_array @ L_inv
    Ec_L_f2 = Ec_L @ f2

    for alpha in alpha_vals:
        # Scale the one-hot vector and L_inv by alpha
        alpha_ek21 = alpha * ek21
        alpha_L_inv = alpha * L_inv

        # Reshape for proper matrix multiplication
        alpha_ek = alpha_ek21.reshape(-1, 1)   # (n_sectors x 1)
        bk_2d = bk_row.reshape(1, -1)           # (1 x n_sectors)

        # Multiply alpha-scaled one-hot with bk row
        alpha_ek_bk = alpha_ek @ bk_2d          # (n_sectors x n_sectors)

        # Compute additional partial products
        alpha_L_inv_ek21 = alpha_L_inv @ ek21    # (n_sectors,)
        alpha_L_inv_ek21_2d = alpha_L_inv_ek21.reshape(-1, 1)
        alpha_L_inv_ek21_bk = alpha_L_inv_ek21_2d @ bk_2d  # (n_sectors x n_sectors)
        alpha_L_inv_ek21_bk_L_new = alpha_L_inv_ek21_bk @ L_inv  # (n_sectors x n_sectors)

        alpha_bk = alpha * bk_2d                 # (1 x n_sectors)
        alpha_bk_L = alpha_bk @ L_inv            # (1 x n_sectors)
        alpha_bk_L_ek21 = alpha_bk_L @ ek21        # scalar

        one_divide_DENOM = 1.0 / (1.0 + alpha_bk_L_ek21)
        NOM_One_divide_DENOM = alpha_L_inv_ek21_bk_L_new * one_divide_DENOM

        L_bar_new = L_inv - NOM_One_divide_DENOM
        L_bar_new_L_delL = L_bar_new - L_inv

        diagEc_21_Lbar = diag_array @ L_bar_new

        # Replicate fbar update for the current row i
        fbar = f2.copy()
        fbar[i] = (1 - alpha) * fbar[i]
        delf_fbar_f = fbar - f2

        Ec_21_del_L_Lbar = diag_array @ L_bar_new_L_delL
        Ec_21_del_L_fbar = Ec_21_del_L_Lbar @ fbar
        Ec_21_del_L_f_21 = Ec_21_del_L_Lbar @ f2

        Ec_21_del_L_fbar_Plus_Ec_21_del_L_f_21 = 0.5 * (Ec_21_del_L_fbar + Ec_21_del_L_f_21)
        Ec_L_21_Ec_L_bar = Ec_L + diagEc_21_Lbar
        Ec_L_21_Ec_L_bar_delf = (Ec_L_21_Ec_L_bar @ delf_fbar_f) / 2.0

        delE_2nd_3rd = Ec_21_del_L_fbar_Plus_Ec_21_del_L_f_21 + Ec_L_21_Ec_L_bar_delf

        # Calculate the share (scalar) and record results
        delE_share = delE_2nd_3rd.sum() / Ec_L_f2.sum()
        row_idx_list.append(i)
        alpha_list.append(alpha)
        delE_list.append(delE_share)

# Build the DataFrame of results and pivot it so that rows are by original index and columns by alpha
carbon_results_df = pd.DataFrame({
    'Row_Index': row_idx_list,
    'Alpha': alpha_list,
    'DelE_Share': delE_list
})
carbon_results_df = carbon_results_df.pivot(index='Row_Index', columns='Alpha', values='DelE_Share')
carbon_results_df.reset_index(inplace=True)

# =============================================================================
# 5) Merge Carbon Results with Industry Names for Labeling
# =============================================================================
# Read the industry file (again) and drop the last row
df_name = pd.read_csv(file_industry)
if not df_name.empty:
    df_name = df_name.drop(df_name.index[-1])

# Keep columns 1 and 2; then create a new column "Naics" by splitting the 2nd column
df_name = df_name.iloc[:, [1, 2]]
df_name['Naics'] = df_name.iloc[:, 1].str.split('-').str[0].str.strip()

# Rearrange so that the first column is "Naics" and then the original column
df_name = df_name.iloc[:, [-1, -2]]
df_name['Row_Index'] = range(1, 547)
df_name = df_name.iloc[:, [2, 1]]

# Merge the industry names with the pivoted carbon results
Carbon = pd.merge(df_name, carbon_results_df, on='Row_Index', how='left')
Carbon.drop(columns='Row_Index', inplace=True)

# Rename columns: if a column name is a float, round it to 3 decimals
Carbon.columns = [round(c, 3) if isinstance(c, float) else c for c in Carbon.columns]
Carbon.to_csv('carbon_results.csv', index=False)

# =============================================================================
# 6) Prepare the Value-Added Portion Data
# =============================================================================
# Read the industry file again
df_va = pd.read_csv(file_industry)
if not df_va.empty:
    df_va = df_va.drop(df_va.index[-1])
# Process the 3rd column: split by '-' and strip
df_va.iloc[:, 2] = df_va.iloc[:, 2].str.split('-').str[0].str.strip()
# Process columns 5 through 11: remove $ signs, commas, parentheses and convert to float
df_va.iloc[:, 5:12] = df_va.iloc[:, 5:12].replace({'\$': '', ',': '', '\(': '', '\)': ''},
                                                    regex=True).astype(float)
df_va.dropna(inplace=True)
# Keep columns 3, 8, and 9 (i.e. indices 2, 7, 8)
df_va = df_va.iloc[:, [2, 7, 8]]
# Compute the value‐added coefficient as (column at index 7) divided by (column at index 8)
df_va['value_add_coeff'] = df_va.iloc[:, 1] / df_va.iloc[:, 2]
df_va['value_add_coeff'] = df_va['value_add_coeff'].fillna(0)
df_va = df_va.iloc[:, [0, -1]]
# Remove any row where the first column equals '93'
df_va = df_va[df_va['Display Description'] != '93']
# Rename columns and set index
df_va.columns = ['Naics3', 'Value_add_coeff']
df_va.set_index('Naics3', inplace=True)

# Build a diagonal matrix for value‐added coefficients (size based on df_va)
df3_diag = pd.DataFrame(np.zeros((len(df_va), len(df_va))),
                        index=df_va.index, columns=df_va.index)
np.fill_diagonal(df3_diag.values, df_va['Value_add_coeff'])
df3_diag.fillna(0, inplace=True)

# Read df2 for value‐added (from industry file)
df2_value = pd.read_csv(file_industry)
if not df2_value.empty:
    df2_value = df2_value.drop(df2_value.index[-1])
df2_value.iloc[:, 2] = df2_value.iloc[:, 2].str.split('-').str[0].str.strip()
df2_value.iloc[:, -1] = df2_value.iloc[:, -1].replace({'\$': '', ',': ''}, regex=True).astype(float)
df2_value.dropna(inplace=True)
df2_value = df2_value.iloc[:, [2, -1]]
df2_value.columns = ['Extracted_Column_3', 'Value']
df2_value.set_index('Extracted_Column_3', inplace=True)

# =============================================================================
# 7) Big Alpha Loop for the "Value-Added" Portion
# =============================================================================
# Prepare arrays: using the value-added diagonal and df2_value; note that the inverse matrix
# and subset_wide_df remain the same.
diag_array_va = df3_diag.values        # from value-added coefficients
L_inv_va = result_inverse_df.values      # same as before (546 x 546)
f2_va = df2_value.values                 # value column from df2_value
A_va = subset_wide_df.values             # same as before

# (Replicate a similar initial multiplication as before)
_ = diag_array_va @ L_inv_va @ f2_va

alpha_vals = np.arange(0.01, 1.01, 0.03)
row_idx_list_va = []
alpha_list_va = []
delE_list_va = []
n_sectors_va = f2_va.shape[0]

for i in range(n_sectors_va):
    ek21 = np.zeros(n_sectors_va)
    ek21[i] = 1

    bk_row = A_va[i, :].copy()
    bk_row[i] = 0

    L_inv_ek21 = L_inv_va @ ek21
    L_inv_f2 = L_inv_va @ f2_va
    Ec_L = diag_array_va @ L_inv_va
    Ec_L_f2 = Ec_L @ f2_va

    for alpha in alpha_vals:
        alpha_ek21 = alpha * ek21
        alpha_L_inv = alpha * L_inv_va

        alpha_ek = alpha_ek21.reshape(-1, 1)
        bk_2d = bk_row.reshape(1, -1)
        alpha_ek_bk = alpha_ek @ bk_2d

        alpha_L_inv_ek21 = alpha_L_inv @ ek21
        alpha_L_inv_ek21_2d = alpha_L_inv_ek21.reshape(-1, 1)
        alpha_L_inv_ek21_bk = alpha_L_inv_ek21_2d @ bk_2d
        alpha_L_inv_ek21_bk_L_new = alpha_L_inv_ek21_bk @ L_inv_va

        alpha_bk = alpha * bk_2d
        alpha_bk_L = alpha_bk @ L_inv_va
        alpha_bk_L_ek21 = alpha_bk_L @ ek21
        one_divide_DENOM = 1.0 / (1.0 + alpha_bk_L_ek21)

        NOM_One_divide_DENOM = alpha_L_inv_ek21_bk_L_new * one_divide_DENOM
        L_bar_new = L_inv_va - NOM_One_divide_DENOM
        L_bar_new_L_delL = L_bar_new - L_inv_va

        diagEc_21_Lbar = diag_array_va @ L_bar_new

        fbar = f2_va.copy()
        fbar[i] = (1 - alpha) * fbar[i]
        delf_fbar_f = fbar - f2_va

        Ec_21_del_L_Lbar = diag_array_va @ L_bar_new_L_delL
        Ec_21_del_L_fbar = Ec_21_del_L_Lbar @ fbar
        Ec_21_del_L_f_21 = Ec_21_del_L_Lbar @ f2_va

        Ec_21_del_L_fbar_Plus_Ec_21_del_L_f_21 = 0.5 * (Ec_21_del_L_fbar + Ec_21_del_L_f_21)
        Ec_L_21_Ec_L_bar = Ec_L + diagEc_21_Lbar
        Ec_L_21_Ec_L_bar_delf = (Ec_L_21_Ec_L_bar @ delf_fbar_f) / 2.0

        delE_2nd_3rd = Ec_21_del_L_fbar_Plus_Ec_21_del_L_f_21 + Ec_L_21_Ec_L_bar_delf
        delE_share = delE_2nd_3rd.sum() / Ec_L_f2.sum()

        row_idx_list_va.append(i)
        alpha_list_va.append(alpha)
        delE_list_va.append(delE_share)

value_added_results = pd.DataFrame({
    'Row_Index': row_idx_list_va,
    'Alpha': alpha_list_va,
    'DelE_Share': delE_list_va
})
value_added_results_df = value_added_results.pivot(index='Row_Index', columns='Alpha', values='DelE_Share')
value_added_results_df.reset_index(inplace=True)

# =============================================================================
# 8) Merge Value-Added Results with Industry Names for Labeling
# =============================================================================
df_name_va = pd.read_csv(file_industry)
if not df_name_va.empty:
    df_name_va = df_name_va.drop(df_name_va.index[-1])
df_name_va = df_name_va.iloc[:, [1, 2]]
df_name_va['Naics'] = df_name_va.iloc[:, 1].str.split('-').str[0].str.strip()
df_name_va = df_name_va.iloc[:, [-1, -2]]
df_name_va['Row_Index'] = range(1, 547)
df_name_va = df_name_va.iloc[:, [2, 1]]

Value_added = pd.merge(df_name_va, value_added_results_df, on='Row_Index', how='left')
Value_added.drop(columns='Row_Index', inplace=True)
# Rename columns: if a column name is a float, round it to 3 decimals
Value_added.columns = [round(c, 3) if isinstance(c, float) else c for c in Value_added.columns]
Value_added.to_csv('value_added_results.csv', index=False)

# =============================================================================
# 9) Regression on Carbon Data (Transposed)
# =============================================================================
Carbon_transposed = Carbon.T
X_values_dict = {}

# Run an OLS regression on each column (for columns 0..544)
for col in Carbon_transposed.columns[:545]:
    Carbon_transposed[col] = pd.to_numeric(Carbon_transposed[col], errors='coerce')
    X = pd.to_numeric(Carbon_transposed.index, errors='coerce').dropna()
    Y = Carbon_transposed[col].dropna()
    if len(X) < 2 or len(Y) < 2:
        X_values_dict[col] = np.nan
        continue
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    print(f"Regression results for {col}:")
    print(model.summary())
    if len(model.params) >= 2:
        const = model.params[0]
        slope = model.params[1]
        if slope != 0:
            X_val = (0.001 - const) / slope
            X_values_dict[col] = X_val
            print(f"For {col}, when Y=0.001, X={X_val}")
        else:
            X_values_dict[col] = np.nan
    else:
        X_values_dict[col] = np.nan

# =============================================================================
# 10) Regression on Value-Added Data (Transposed)
# =============================================================================
Value_added_transposed = Value_added.T
model_results_df_raw = pd.DataFrame(columns=['Column', 'Constant', 'Coefficient', 'X1_value', 'Y_Calculated'])
Y_calculated_dict = {}

for col in Value_added_transposed.columns[:545]:
    Value_added_transposed[col] = pd.to_numeric(Value_added_transposed[col], errors='coerce')
    X = pd.to_numeric(Value_added_transposed.index, errors='coerce').dropna()
    Y = Value_added_transposed[col].dropna()
    if len(X) < 2 or len(Y) < 2:
        model_results_df_raw = model_results_df_raw.append({
            'Column': col,
            'Constant': np.nan,
            'Coefficient': np.nan,
            'X1_value': X_values_dict.get(col, np.nan),
            'Y_Calculated': np.nan
        }, ignore_index=True)
        continue
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    print(f"Regression results for {col}:")
    print(model.summary())
    if len(model.params) < 2:
        cst, slope = np.nan, np.nan
    else:
        cst, slope = model.params[0], model.params[1]
    X1_val = X_values_dict.get(col, np.nan)
    if pd.isna(X1_val) or pd.isna(slope) or slope == 0:
        Y_calc = np.nan
    else:
        Y_calc = cst + slope * X1_val
    model_results_df_raw = model_results_df_raw.append({
        'Column': col,
        'Constant': cst,
        'Coefficient': slope,
        'X1_value': X1_val,
        'Y_Calculated': Y_calc
    }, ignore_index=True)
    Y_calculated_dict[col] = Y_calc

# =============================================================================
# 11) Enrich Final Regression Results with Additional Data
# =============================================================================
df_name_final = pd.read_csv(file_industry)
if not df_name_final.empty:
    df_name_final = df_name_final.drop(df_name_final.index[-1])

# Use the "Display Description" column to extract sector labels
labels = df_name_final['Display Description'].str.split('-').str[0]

final_records = []
for _, row in model_results_df_raw.iterrows():
    col_str = row['Column']
    try:
        col_idx = int(float(col_str))
    except:
        col_idx = np.nan

    if pd.isna(col_idx) or col_idx >= len(labels):
        sector_name = np.nan
        last_column = np.nan
        final_demand_val = np.nan
        total_output_val = np.nan
        carbon_val = np.nan
        carbon_output_val = np.nan
    else:
        sector_name = labels.iloc[col_idx]
        last_column = df_name_final['Display Description'].iloc[col_idx]
        final_demand_val = Final_demand['Value'].iloc[col_idx]
        total_output_val = total_output['Total Output'].iloc[col_idx]
        carbon_val = carbonemission.iloc[col_idx] if col_idx < len(carbonemission) else np.nan
        carbon_output_val = carbon_output.iloc[col_idx] if col_idx < len(carbon_output) else np.nan

    rec = row.to_dict()
    rec.update({
        'Sector_name': sector_name,
        'Last_Column': last_column,
        'Final_demand': final_demand_val,
        'Total_output': total_output_val,
        'Carbon_entensity': carbon_val,
        'Carbon_output': carbon_output_val
    })
    final_records.append(rec)

final_model_results_df = pd.DataFrame(final_records)
column_order = ['Column', 'Constant', 'Coefficient', 'X1_value', 'Last_Column',
                'Sector_name', 'Final_demand', 'Total_output', 'Y_Calculated',
                'Carbon_entensity', 'Carbon_output']
column_order = [c for c in column_order if c in final_model_results_df.columns]
final_model_results_df = final_model_results_df[column_order]

# Multiply Y_Calculated by 1,000,000 as in the original code and save to CSV
final_model_results_df['Y_Calculated'] = final_model_results_df['Y_Calculated'] * 1000000
final_model_results_df.to_csv('Test_2021_results.csv', index=False)
print(type(carbonemission))

# =============================================================================
# 12) Plotting Routines
# =============================================================================
# Scatter plot: Carbon entensity vs. Carbon mitigation price
plt.figure()
plt.scatter(final_model_results_df['Carbon_entensity'], final_model_results_df['Y_Calculated'],
            alpha=0.5, s=8)
plt.xlabel('Carbon entensity level')
plt.ylabel('Carbon mitigation price')
plt.xlim(-0.01, 1)
plt.ylim(0, 0.02)
plt.savefig('Scatter_LA_2018.png', bbox_inches='tight')

# Bar plot: Filtered ranked bar plot for values < 0.05
filtered_data = final_model_results_df[final_model_results_df['Y_Calculated'] < 0.05]
sorted_data = filtered_data.sort_values(by='Y_Calculated', ascending=False)
Y_calculated_sorted = sorted_data['Y_Calculated']
last_column_values_sorted = sorted_data['Last_Column']
labels_plot = last_column_values_sorted.str.split('-').str[0]
plt.figure(figsize=(10, 20))
plt.barh(range(len(Y_calculated_sorted), 0, -1), Y_calculated_sorted, tick_label=labels_plot)
plt.xlabel('Carbon mitgitaion cost')
plt.ylabel('Sector')
plt.title('Mitigation costs (Filtered for values < 0.05)')
plt.savefig('foo_LA.png', bbox_inches='tight')

# =============================================================================
# 13) Export Final Data to Excel
# =============================================================================
# Clean Carbon_output column (remove commas and convert to float)
final_model_results_df['Carbon_output'] = final_model_results_df['Carbon_output'].astype(str) \
    .str.replace(',', '').astype(float)
with pd.ExcelWriter('ToGAMS_LA.xlsx') as writer:
    subset_wide_df.to_excel(writer, sheet_name='A', index=True)
    final_model_results_df.to_excel(writer, sheet_name='FCC', index=False)

print("All done.")

endEmbeddedCode