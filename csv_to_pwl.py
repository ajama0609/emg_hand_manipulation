import csv

def convert_emg_csv_to_pwl(csv_filename, pwl_filename, channel_name="Channel 1"):
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)  # uses header row
        data = []

        for row in reader:
            try:
                time_ms = float(row["Time (ms)"])
                time_sec = time_ms / 1000.0
                value = float(row[channel_name])
                data.append((time_sec, value))
            except (ValueError, KeyError):
                continue  # skip bad rows or missing values

    with open(pwl_filename, 'w') as pwl_file:
        for time, value in data:
            pwl_file.write(f"{time} {value}\n")

    print(f"PWL file written: {pwl_filename}")

# Example usage:
convert_emg_csv_to_pwl("emg_data_1.csv", "channel1.pwl", channel_name="Channel 1")
