import pandas as pd
import numpy as np  # Import numpy for NaN handling

def read_worker_requests_from_excel(file_path):
    worker_requests = []

    for i in range(1, 18):  # Employee tabs from E01 to E17
        tab_name = f'E{str(i).zfill(2)}'
        try:
            df = pd.read_excel(file_path, sheet_name=tab_name, header=None)
        except Exception as e:
            print(f"Failed to read sheet {tab_name} from {file_path}: {e}")
            continue

        emp_num = int(df.iloc[0, 0])  # Employee number from A1
        shift_range = df.iloc[2:34, 4:7]  # Shift availability from E3:G34
        days = df.iloc[3:34, 0]  # Day values from A4:A34

        for day, (morning, evening, night) in zip(days, shift_range.values):
            if pd.isnull(day):  # Skip rows where day is NaN
                continue
            corrected_day = int(day) - 1  # Correct the day by subtracting 1
            # If employee can't work any shifts, append a single request with shift 0
            if morning == 1 and evening == 1 and night == 1:
                worker_requests.append((emp_num, 0, corrected_day, -10))
            else:
                # Check each shift for restrictions and output accordingly
                if morning == 1:
                    worker_requests.append((emp_num, 1, corrected_day, 4))  # Morning shift restriction
                if evening == 1:
                    worker_requests.append((emp_num, 2, corrected_day, 4))  # Evening shift restriction
                if night == 1:
                    worker_requests.append((emp_num, 3, corrected_day, 4))  # Night shift restriction

    return worker_requests

def main():
    file_path = "C:\\Users\\ashur\\Desktop\\WSC\\sc_tst_shifts_Autimation_Test.xlsm"
    output_path = "C:\\Users\\ashur\\Desktop\\WSC\\requests.txt"  # Specify your desired output path
    worker_requests = read_worker_requests_from_excel(file_path)

    formatted_requests = ",\n".join(map(str, worker_requests))

    # Write to a file instead of printing
    with open(output_path, 'w') as f:
        f.write(f"[{formatted_requests}]")

if __name__ == "__main__":
    main()
