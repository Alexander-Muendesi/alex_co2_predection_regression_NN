import pandas as pd

#For instances that have missing values, those input neurons will simply not activate for now
class DataReader:
    def __init__(self):
        self.filename = "./src/dataset/data.csv"
        self.data = None
        self.prediction_data = None

    def readFile(self):
        data = pd.read_csv(self.filename)
        header_rows = list(data.columns) 
        header_rows = header_rows[1:]#select all header names except the country
        data.replace('', pd.NA, inplace=True)
        data[header_rows] = data[header_rows].apply(pd.to_numeric, errors="coerce")
        self.data = data
        # print(self.data)
        self.generate_prediction_data()

    
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
        print(data)

#Arbitrary notes
#iloc[] can be used to access a specific row by providing it an index
#inplace=True means modify the current data frame. inplace=False means return a modified copy.
# pd.set_option('display.max_columns', None) to prevent truncation of columns
