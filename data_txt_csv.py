import os
import csv

# Path to the EMG data file
filepath = 'emg+data+for+gestures/EMG_data_for_gestures-master/01/1_raw_data_13-12_22.03.16.txt'

# Output file path
output_csv = 'emg_data_1.csv'

# Check if the file exists
if os.path.isfile(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

        # Skip first line if it's a header or metadata
        data_lines = lines[1:]  # start from second row

        # Prepare to collect valid rows
        valid_rows = []

        for line in data_lines:
            row = line.strip()
            columns = row.split()  # or use .split(',') if CSV

            if len(columns) == 10:
                valid_rows.append(columns)
            else:
                print(f"Skipping row with {len(columns)} columns: {row}")

        if valid_rows:
            headers = ['Time (ms)'] + [f'Channel {i}' for i in range(1, 9)] + ['Class']
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(valid_rows)

            print(f"{len(valid_rows)} rows successfully written to '{output_csv}'.")
        else:
            print("No valid data rows found.")
else:
    print("File does not exist.")
