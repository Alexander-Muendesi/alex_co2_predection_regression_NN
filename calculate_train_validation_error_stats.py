import csv
import statistics

# Initialize variables to calculate average and store values
values = []
total = 0

# Read the CSV file and extract real numbers
with open('perceptron_validation_mse.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if row:  # Check if the row is not empty
            value = float(row[0])
            values.append(value)
            total += value
            # print(value)

# Calculate and print the average and standard deviation
if values:
    average = total / len(values)
    std_deviation = statistics.stdev(values)
    print(f"Average: {average}")
    print(f"Standard Deviation: {std_deviation}")
else:
    print("No data found in the file.")
