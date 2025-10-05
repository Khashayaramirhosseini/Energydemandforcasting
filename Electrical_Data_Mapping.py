# Reload libraries after code state reset
import pandas as pd

# Load the Excel file with all device sheets
excel_path = "/mnt/data/electrical_data_by_device_with_datetime (1).xlsx"
xls = pd.read_excel(excel_path, sheet_name=None)

# Updated parameter mapping with correct labels
parameter_mapping = {
    1: "voltage_phase_1",
    2: "voltage_phase_2",
    3: "voltage_phase_3",
    4: "current_phase_1",
    5: "current_phase_2",
    6: "current_phase_3",
    7: "power_phase_1",
    8: "power_phase_2",
    9: "power_phase_3",
    10: "total_system_power",
    11: "total_import_kwh",
    12: "total_export_kwh",
    13: "total_kwh",
    14: "l1_import_kwh",
    15: "l2_import_kwh",
    16: "l3_import_kwh",
    17: "l1_export_kwh",
    18: "l2_export_kwh",
    19: "l3_export_kwh",
    20: "l1_total_kwh",
    21: "l2_total_kwh",
    22: "l3_total_kwh"
}

# Process and pivot each sheet using new mapping
processed_sheets = {}
for sheet_name, df in xls.items():
    df['reading_date'] = pd.to_datetime(df['reading_date'])
    df['parameter_name'] = df['parameter'].map(parameter_mapping)
    pivoted = df.pivot_table(
        index='reading_date',
        columns='parameter_name',
        values='reading',
        aggfunc='mean'
    ).reset_index()
    processed_sheets[sheet_name] = pivoted

# Save the updated output
output_path = "/mnt/data/electrical_data_pivoted_correct_mapping.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for sheet_name, df in processed_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

output_path
