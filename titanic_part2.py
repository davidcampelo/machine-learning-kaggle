# Import the Pandas library
import pandas as pd
def log(msg):
	print("****************************************************************************\n* "+msg)


# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
# print train["Embarked"].describe()

# # Convert the Embarked classes to integer form
# S = 0
# C = 1
# Q = 2
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# #Print the Sex and Embarked columns
print(train[['Sex', 'Embarked']])