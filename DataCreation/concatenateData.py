import os
import pandas as pd

def merge_processed_csv(input_folder, output_file):
    dataframes = []

    # Loop through all files and select only those ending with 'processed.csv'
    for file in os.listdir(input_folder):
        if file.endswith("processed.csv"):
            file_path = os.path.join(input_folder, file)
            print(f"Reading {file_path}")
            df = pd.read_csv(file_path)
            dataframes.append(df)

    if not dataframes:
        print("No 'processed.csv' files found in the folder.")
        return

    # Merge all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save final merged CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as {output_file}")


if __name__ == "__main__":
    input_folder = "csv_files"   # folder containing your processed CSVs
    output_file = "merged_processed_output.csv"
    merge_processed_csv(input_folder, output_file)
