import random
import pandas as pd

#For instances that have missing values, those input neurons will simply not activate for now
class DataReader:
    def __init__(self, random_number_generator):
        self.filename = "./src/dataset/data.csv"
        self.data = None
        self.prediction_data = None
        self.test_data = None
        self.train_data = None
        self.validation_data = None
        self.random_number_generator = random_number_generator# seeded random number generator
        self.training_indexs = {}
        self.validation_indexs = {}

    def readFile(self):
        data = pd.read_csv(self.filename)
        header_rows = list(data.columns) 
        header_rows = header_rows[1:]#select all header names except the country
        data.replace('', pd.NA, inplace=True)
        data[header_rows] = data[header_rows].apply(pd.to_numeric, errors="coerce")
        self.data = data
        # print(self.data)
        self.generate_prediction_data()
        self.generate_training_dataset()
        self.generate_validation_set()
    
    #generates a validation set which consists of 10 % of the data
    def generate_validation_set(self) :
        upperbound = self.data.shape[0]
        max_size = int(0.1 * self.data.shape[0])
        data = []

        while len(data) < max_size : 
            index = self.random_number_generator.randrange(upperbound)
            if(index not in self.training_indexs and index not in self.validation_indexs) : 
                self.validation_indexs[index] = 'v'
                data.append(self.data.iloc[index].copy())

        data = pd.DataFrame(data, columns=self.data.columns)
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
        self.prediction_data = data

#Arbitrary notes
#iloc[] can be used to access a specific row by providing it an index
#inplace=True means modify the current data frame. inplace=False means return a modified copy.
# pd.set_option('display.max_columns', None) to prevent truncation of columns
