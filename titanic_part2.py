# Import the Pandas library
import pandas as pd
import numpy as np
from sklearn import tree
def log(msg):
	print("****************************************************************************\n* "+msg)


# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# Convert the male and female groups to integer form
# male = 0
# female = 1
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
# Impute the Embarked variable with the most common value (S)
train["Embarked"] = train["Embarked"].fillna("S")

# # Convert the Embarked classes to integer form
# S = 0
# C = 1
# Q = 2
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# Set the median value for Age if none is provided
train["Age"] = train["Age"].fillna(train["Age"].median())



log("Now let's do some ML...")

# Print the train data to see the available features

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

log("Show the importance of the included features - PClass, Sex, Age, Fare")
print(my_tree_one.feature_importances_)
log("Show the score of the included features - PClass, Sex, Age, Fare")
print(my_tree_one.score(features_one, target))