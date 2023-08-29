from dataReader import DataReader
import random

random_generator = random.Random(0)
datareader = DataReader(random_generator)
datareader.readFile()