import pandas as pd
import numpy as np
from scipy.stats import zscore

# For instances that have missing values, those input neurons will simply not activate for now. Those instances are set to NaN
# The Entity column has been removed as it is of little use. Instead will use latitude and longitude to indetift country
# To detect outliers the z-score is calculated. Data points whose z score is out of 3 standard deviations are considered outliers
class DataReader:
    def __init__(self, random_number_generator):
        self.filename = "./src/dataset/data.csv"
        self.data = None                            # DataFrame representing the entire file
        self.prediction_data = None                 # DataFrame representing the prediction dataset
        self.test_data = None                       # DataFrame representing the test dataset
        self.train_data = None                      # DataFrame representing the training dataset
        self.validation_data = None                 # DataFrame representing the validation dataset
        self.random_number_generator = random_number_generator  # seeded random number generator
        self.training_indexs = {}                   # keeps track of what indexes where used for the training dataset
        self.validation_indexs = {}                 # keeps track of what indexes where used for the validation dataset
        self.test_indexs = {}                       # Keeps track of what indexes where used for the test dataset
        self.train_mean_values = None               # Keeps the mean values for the training dataset
        self.train_std_dev_values = None            # Keeps track of the standard deviation values

    def readFile(self):
        data = pd.read_csv(self.filename)
        header_rows = list(data.columns) 
        header_rows = header_rows[1:]#select all header names except the country
        data.replace('', pd.NA, inplace=True)
        data[header_rows] = data[header_rows].apply(pd.to_numeric, errors="coerce")
        self.data = data

        # remove outliers from the dataset
        self.remove_outliers()

        # generate the various datasets and place them in a DataFrame
        self.generate_prediction_data()
        self.generate_training_dataset()
        self.generate_validation_dataset()
        self.generate_test_dataset()

        # normalize the data
        self.normalize_training_set()
        self.normalize_validation_set()
        self.normalize_test_set()
        self.normalize_prediction_set()

    def normalize_prediction_set(self) : 
        nan_locations = self.prediction_data.isna()
        self.prediction_data.fillna(0)
        normalized_data = (self.prediction_data - self.train_mean_values) / self.train_std_dev_values
        normalized_data[nan_locations] = np.nan
        self.prediction_data = normalized_data

    def normalize_test_set(self):
        nan_locations = self.test_data.isna()
        self.test_data.fillna(0)
        normalized_data = (self.test_data - self.train_mean_values) / self.train_std_dev_values
        normalized_data[nan_locations] = np.nan
        self.test_data = normalized_data

    def normalize_validation_set(self):
        nan_locations = self.validation_data.isna()            # Keep track of where the NaN values are
        self.validation_data.fillna(0)                         # Temporarily feel the NaN values with 0
        normalized_data = (self.validation_data - self.train_mean_values) / self.train_std_dev_values
        normalized_data[nan_locations] = np.nan                # Restore the NaN values
        self.validation_data = normalized_data

    def normalize_training_set(self):
        mean_values = self.train_data.mean()
        std_dev_values = self.train_data.std()

        normalizd_data = (self.train_data - mean_values) / std_dev_values
        self.train_data = normalizd_data

        self.train_mean_values = mean_values
        self.train_std_dev_values = std_dev_values

    # removes data points whose z score is more than 3 standard deviations 
    def remove_outliers(self) :
        data = self.data.copy()
        del data["Entity"]
        z_scores = data.apply(lambda column: zscore(column.dropna()), axis=0)       # calculate the z scores for each data point, excluding NaN values

        threshold = 3
        anomalies = (z_scores > threshold) | (z_scores < -threshold)

        rows_without_anomalies = ~anomalies.any(axis=1)
        indexes_without_anomalies = data.index[rows_without_anomalies]              
        filtered_data = self.data.loc[indexes_without_anomalies]                    # keep only the rows that do not have anomalies in them
        self.data = filtered_data


    # generates the test dataset
    def generate_test_dataset(self) :
        upperbound = self.data.shape[0]
        max_size = int(0.2 * self.data.shape[0])
        data = []

        while len(data) < max_size :
            index = self.random_number_generator.randrange(upperbound)
            if index not in self.training_indexs and index not in self.validation_indexs and index not in self.test_indexs :
                self.test_indexs[index] = 'test'
                data.append(self.data.iloc[index].copy())

        data = pd.DataFrame(data, columns=self.data.columns)
        del data["Entity"]
        self.test_data = data
    
    #generates a validation set which consists of 10 % of the data
    def generate_validation_dataset(self) :
        upperbound = self.data.shape[0]
        max_size = int(0.1 * self.data.shape[0])
        data = []

        while len(data) < max_size : 
            index = self.random_number_generator.randrange(upperbound)
            if(index not in self.training_indexs and index not in self.validation_indexs) : 
                self.validation_indexs[index] = 'v'
                data.append(self.data.iloc[index].copy())

        data = pd.DataFrame(data, columns=self.data.columns)
        del data["Entity"]
        self.validation_data = data

    #generates a training set which consists of 70 % if the data
    def generate_training_dataset(self) :
        upperbound = self.data.shape[0]
        max_size = int(0.7 * self.data.shape[0])
        data = []

        #randomly fill data with elements to make dataset
        while len(data) < max_size:
            index = self.random_number_generator.randrange(upperbound)
            if(index not in self.training_indexs) :
                self.training_indexs[index] = 't'
                data.append(self.data.iloc[index].copy())

        data = pd.DataFrame(data, columns=self.data.columns)
        del data["Entity"]
        self.train_data = data    

    
    def generate_prediction_data(self):
        data = []#stores the countries last row with the years from [2021,2025]
        south_africa_rows = self.data[self.data["Entity"] == "South Africa"]
        spain_rows = self.data[self.data["Entity"] == "Spain"]
        sweeden_rows = self.data[self.data["Entity"] == "Sweden"]

        counter = int(2021)
        while counter <= 2025:
            last_item = south_africa_rows.iloc[-1].copy()
            last_item["Year"] = counter
            data.append(last_item)
            counter += 1

        counter = int(2021)
        while counter <= 2025 : 
            last_item = spain_rows.iloc[-1].copy()
            last_item["Year"] = counter
            data.append(last_item)
            counter += 1
            
        counter = int(2021)
        while counter <= 2025 :
            last_item = sweeden_rows.iloc[-1].copy()
            last_item["Year"] = counter
            data.append(last_item)
            counter += 1
        
        #convert the list to a data frame
        data = pd.DataFrame(data, columns=south_africa_rows.columns)
        data.reset_index(drop=True,inplace=True)#reset the indexes in the data frame so first element has index of 0 again

        #TODO migh have to add "del data["Entity"]" here but not sure. Keep this in mind
        self.prediction_data = data
    
    def get_train_data(self):
        return self.train_data
    
    def get_validation_data(self):
        return self.validation_data
    
    def get_test_data(self):
        return self.test_data
    
    def get_prediction_data(self):
        return self.prediction_data

#Arbitrary notes
#iloc[] can be used to access a specific row by providing it an index
#inplace=True means modify the current data frame. inplace=False means return a modified copy.
# pd.set_option('display.max_columns', None) to prevent truncation of columns
