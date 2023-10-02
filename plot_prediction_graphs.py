import matplotlib.pyplot as plt

# Initialize lists to store data for each country
south_africa_data = [338614.7477165,338528.60815073334,338442.4647574,338356.3208278667,338270.18239413336]
sri_lanka_data = [35179.26790813333,35093.12651563333,35006.98505163333,34920.8434671,34834.70167193333]
thailand_data = [243944.32598973333,243858.1839278,243772.04127356666,243685.89911383332,243599.7570648]

# Create x-axis values (years)
years = list(range(2021, 2026))

# Plot data for South Africa in blue
# plt.plot(years, south_africa_data, label='South Africa', color='blue')

# Plot data for Sri Lanka in green
# plt.plot(years, sri_lanka_data, label='Sri Lanka', color='green')

# Plot data for Thailand in red
plt.plot(years, thailand_data, label='Thailand', color='red')

# Set labels and legend
plt.xlabel('Year')
plt.ylabel('Average CO2 Levels')
plt.title('Average CO2 Levels Over Time')
plt.legend()

# Show the plot
plt.show()
