import statistics
import csv

# Initialize lists to store data for each item
item_data = [[] for _ in range(15)]

# Open the input file for reading
with open('perceptron_30_test_runs.csv', 'r') as file:
    data = file.read().split('\n\n')  # Split runs by blank lines

# Process each run
for run_data in data:
    run_lines = run_data.strip().split('\n')
    run_values = [float(line) for line in run_lines]

    # Append run values to the respective item list
    for i, value in enumerate(run_values):
        item_data[i].append(value)

# Calculate metrics for each item
results = []
for i, item_values in enumerate(item_data, start=1):
    item_average = statistics.mean(item_values)
    item_std_dev = statistics.stdev(item_values)
    item_highest = max(item_values)
    item_lowest = min(item_values)
    
    results.append({
        'Item': i,
        'Average': item_average,
        'Standard_Deviation': item_std_dev,
        'Highest': item_highest,
        'Lowest': item_lowest
    })

# Write the results to a CSV file
with open('output_file.csv', 'w', newline='') as csv_file:
    fieldnames = ['Item', 'Average', 'Standard_Deviation', 'Highest', 'Lowest']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(results)
