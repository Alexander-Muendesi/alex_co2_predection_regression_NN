import matplotlib.pyplot as plt

# Read data from the file
with open('perceptron_thailand_prediction.csv', 'r') as file:
    lines = file.readlines()

# Initialize lists to store data
x_values = []
y_values1 = []
y_values2 = []
current_y_values = y_values1

# Process the lines and store data
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace
    if not line:  # Check for a blank line to switch to the next set of data
        current_y_values = y_values2
        continue

    item = float(line)  # Convert the line to a float (assuming it contains numerical data)
    current_y_values.append(item)

# Generate x-values based on the item count (1 to 20)
x_values = list(range(1, 21))

# Create a line graph for the first set of data (y_values1)
plt.plot(x_values, y_values1, label='NN', color='blue')

# Create a line graph for the second set of data (y_values2)
plt.plot(x_values, y_values2, label='Actual', color='red')

plt.xlabel('Years')
plt.ylabel('CO2 prediction')
plt.title('NN predictions vs actual CO2 values for Thailand')
plt.legend()  # Add a legend to distinguish between the two sets of data

# Show the plot
plt.show()
