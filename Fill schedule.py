import pandas as pd
import openpyxl

# Load the schedule from the CSV file
csv_file_path = "C:\\Users\\ashur\\Desktop\\WSC\\schedule_output.csv"
schedule_df = pd.read_csv(csv_file_path)

# Load the Excel file, keeping macros intact
excel_file_path =  "C:\\Users\\ashur\\Desktop\\WSC\\sc_tst_shifts_Autimation_Test.xlsm"
workbook = openpyxl.load_workbook(excel_file_path, keep_vba=True)
sheet = workbook["Schedule"]

# Define where to start pasting in the Excel sheet for each block of 3 columns from the DataFrame
ranges_to_fill = [("E4:G34", 0), ("I4:K34", 3), ("M4:O34", 6), ("Q4:S34", 9)]

# Loop through each range and start_col pair
for cell_range, start_col in ranges_to_fill:
    cells = sheet[cell_range]
    for row in range(31):  # Assuming 30 rows to fill
        for col in range(3):  # M, E, N for each range
            if start_col + col < len(schedule_df.columns) - 1:  # Skipping 'Day' column, hence -1
                cell_value = schedule_df.iloc[row, start_col + col + 1]  # +1 to skip 'Day' column
                cells[row][col].value = cell_value

# Save the workbook with a new name
workbook.save("C:\\Users\\ashur\\Desktop\\WSC\\sc_tst_shifts_Autimation_Test-Final.xlsm")