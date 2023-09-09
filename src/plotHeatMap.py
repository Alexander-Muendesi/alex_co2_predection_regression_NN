import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Output file path (the one containing parameter1, parameter2, average)
output_file = 'output.csv'

# Read the output CSV file into a pandas DataFrame
df = pd.read_csv(output_file)

# Define custom ranges for x-axis and y-axis
x_axis_ranges = [0, 20, 40, 60, 80, 100, 120, 140, 160]  # Adjust these ranges as needed
y_axis_ranges = [0.00001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01]  # Adjust these ranges as needed

# Create empty dataframes to store aggregated values
aggregated_data = pd.DataFrame(index=y_axis_ranges[1:], columns=x_axis_ranges[1:])

# Iterate through the ranges and aggregate average values
for x_start, x_end in zip(x_axis_ranges[:-1], x_axis_ranges[1:]):
    for y_start, y_end in zip(y_axis_ranges[:-1], y_axis_ranges[1:]):
        subset = df[(df['parameter1'] >= x_start) & (df['parameter1'] < x_end) & (df['parameter2'] >= y_start) & (df['parameter2'] < y_end)]
        average = subset['average'].mean()
        aggregated_data.at[y_end, x_end] = average

# Convert the aggregated data to a numeric format
aggregated_data = aggregated_data.apply(pd.to_numeric)

# Create the heatmap using seaborn with adjusted linewidths and linecolor
plt.figure(figsize=(10, 6))
sns.heatmap(aggregated_data, cmap='viridis', annot=True, fmt='.5f', cbar=True, linewidths=0.5, linecolor='black')

plt.xlabel('Batch size')
plt.ylabel('Learning Rate')
plt.title('Heatmap of Aggregated Average Generalization Values')

# Show the plot
plt.show()
