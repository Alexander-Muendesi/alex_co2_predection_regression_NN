import statistics

# Initialize variables to calculate average and store values
values = []
total = 0

# Read the file and extract values after the colon
with open('ASGD_run.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        if len(parts) == 2:
            value = float(parts[1])
            values.append(value)
            total += value
            print(value)

# Calculate and print the average and standard deviation
if values:
    average = total / len(values)
    std_deviation = statistics.stdev(values)
    print(f"Average: {average}")
    print(f"Standard Deviation: {std_deviation}")
else:
    print("No data found in the file.")
