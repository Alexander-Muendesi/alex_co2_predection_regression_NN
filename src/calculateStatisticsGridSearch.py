import csv
import numpy as np

# Input and output file paths
input_file = 'grid_combined.csv'
output_file = 'output.csv'

# Create dictionaries to store results for each (parameter1, parameter2) combination
data_dict = {}
average_dict = {}
std_dev_dict = {}

# Read the input CSV file
with open(input_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row if present

    for row in csvreader:
        parameter1, parameter2, result = map(float, row)
        key = (parameter1, parameter2)

        if key in data_dict:
            data_dict[key].append(result)
        else:
            data_dict[key] = [result]

# Calculate averages and standard deviations and store in dictionaries
for key, values in data_dict.items():
    average = np.mean(values)
    std_dev = np.std(values)
    average_dict[key] = average
    std_dev_dict[key] = std_dev

# Find the smallest average value(s)
min_average = min(average_dict.values())
min_average_keys = [key for key, value in average_dict.items() if value == min_average]


# Write the results to the output CSV file
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['parameter1', 'parameter2', 'average', 'standard deviation'])

    for key in data_dict.keys():
        csvwriter.writerow([key[0], key[1], average_dict[key], std_dev_dict[key]])

print("Results have been written to", output_file)

# Print the line(s) with the smallest average
print("Line(s) with the smallest average:")
for key in min_average_keys:
    print("Parameter1:", key[0], "Parameter2:", key[1], "Average:", average_dict[key])
