import statistics

# Initialize variables to calculate average and store values
values = []

# Read the file and extract values
with open('perceptron/ASGD_perceptron_runs.txt', 'r') as file:
    for line in file:
        try:
            value = float(line.strip())
            values.append(value)
        except ValueError:
            print(f"Skipping line: {line.strip()} (Not a valid float)")

# Calculate and print the average and standard deviation
if values:
    average = statistics.mean(values)
    std_deviation = statistics.stdev(values)
    print(f"Average: {average}")
    print(f"Standard Deviation: {std_deviation}")
else:
    print("No valid data found in the file.")
