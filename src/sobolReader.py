import pandas as pd

class SobolReader:
    def __init__(self):
        data = pd.read_csv("master_Sobol_numbers.txt")
        data.drop(0)
        del data["Unnamed: 20"]
        self.data = data
        self.index = 0
        
    def getRow(self):
        row = self.data.iloc[self.index]
        self.index += 1

        if self.index == len(self.data):
            raise Exception
        
        return row