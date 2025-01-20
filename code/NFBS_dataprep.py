import tarfile  
import os  
  
# Define the path to the .tar.gz file  
tar_path = '../data/NFBS_Dataset.tar.gz'  
# Define the extraction directory  
extraction_path = '../data'  
  
# Create the extraction directory if it doesn't exist  
if not os.path.exists(extraction_path):  
    os.makedirs(extraction_path)  
  
# Open the tar file  
with tarfile.open(tar_path, 'r:gz') as tar:  
    # Extract all contents to the specified directory  
    tar.extractall(path=extraction_path)  
  
print(f"Extraction complete. Files are extracted to: {extraction_path}")  

import shutil  
  
# Define the source and destination directories  
source_dir = '../data/NFBS_Dataset/'  
destination_dir = '../data/NFBS_Dataset_Flat/'  
  
# Create the destination directory if it doesn't exist  
if not os.path.exists(destination_dir):  
    os.makedirs(destination_dir)  
  
# Walk through the source directory  
for root, dirs, files in os.walk(source_dir):  
    for file in files:  
        # Construct full file path  
        file_path = os.path.join(root, file)  
        # Move file to the destination directory  
        shutil.move(file_path, os.path.join(destination_dir, file))  
  
print(f"Flattening complete. All files are in: {destination_dir}")

import gzip  
   
# Define the directory containing the .gz files  
directory = '../data/NFBS_Dataset_Flat/'  
  
# Iterate over all files in the directory  
for filename in os.listdir(directory):  
    if filename.endswith('.gz'):  
        # Construct the full file path  
        file_path = os.path.join(directory, filename)  
        # Define the output file path (removing the .gz extension)  
        output_file_path = os.path.join(directory, filename[:-3])  
  
        # Open the .gz file and extract its contents  
        with gzip.open(file_path, 'rb') as f_in:  
            with open(output_file_path, 'wb') as f_out:  
                shutil.copyfileobj(f_in, f_out)  
  
        # Optionally, remove the original .gz file after extraction  
        os.remove(file_path)  
  
print(f"Extraction complete. All .gz files in {directory} have been extracted.")  

import os
import zipfile
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import re
  
# Initialize a list to store file details  
file_details = []  
processed_dir = '../data/NFBS_Dataset_Flat'  
  
# Define a regex pattern to extract key parts of the filename  
# Assuming filenames like sub-A00028185_ses-NFB3_T1w.nii, sub-A00028185_ses-NFB3_T1w_brain.nii, sub-A00028185_ses-NFB3_T1w_brainmask.nii  
pattern = re.compile(  
    r"sub-(?P<SubjectID>A\d+)_ses-(?P<SessionID>\w+)_T1w(_(?P<ScanDetail>.+))?\.nii"  
)  
  
# Loop through the directory and collect file details  
for root, dirs, files in os.walk(processed_dir):  
    for file_name in files:  
        if file_name.endswith('.nii'):  # Check if it's an .nii file  
            match = pattern.match(file_name)  
            if match:  
                subject_id = match.group('SubjectID')  
                session_id = match.group('SessionID')  
                scan_detail = match.group('ScanDetail') if match.group('ScanDetail') else 'T1w'  
  
                normalized_file_path = os.path.join(root, file_name).replace('\\', '/')  
  
                file_details.append({  
                    'Subject ID': subject_id,  
                    'Session ID': session_id,  
                    'Scan Detail': scan_detail,  
                    'File Path': normalized_file_path  
                })  
            else:  
                print(f"File does not match expected structure: {file_name}")  
  
# Convert the list to a DataFrame  
summary_df = pd.DataFrame(file_details)  
  
filtered_df = summary_df[summary_df['Scan Detail'] == 'brain']  
  
# Delete files where Scan Detail is not 'brain'  
for index, row in summary_df.iterrows():  
    if row['Scan Detail'] != 'brain':  
        os.remove(row['File Path'])  
  
# If there are any 'brain' files, get the first one  
if not filtered_df.empty:  
    first_nii_file = filtered_df['File Path'].iloc[0]  
    print(f"First .nii file with Scan Detail 'brain': {first_nii_file}")  
else:  
    print("No .nii file with Scan Detail 'brain' found.")  
  
# Save or display the summary of remaining files  
summary_file = '../data/NFBS_detailed_summary.csv'  
filtered_df.to_csv(summary_file, index=False)  
print(f"Detailed summary created: {summary_file}") 