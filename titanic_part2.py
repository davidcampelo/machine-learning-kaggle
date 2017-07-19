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

print test
# Convert the male and female groups to integer form
# male = 0
# female = 1
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
# Impute the Embarked variable with the most common value (S)
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

# # Convert the Embarked classes to integer form
# S = 0
# C = 1
# Q = 2
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# Set the median value for Age if none is provided
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())



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

# Impute the missing value with the median
test.Fare[152] = test["Fare"].median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

log(" Writing solution to a csv file with the name my_solution.csv")
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

# #Print the score of the new decison tree
log("Show the importance of the included features - PClass, Sex, Age, Fare, SibSp, Parch, Embarked")
print(my_tree_two.feature_importances_)
log("Show the score of the included features - PClass, Sex, Age, Fare, SibSp, Parch, Embarked")
print(my_tree_two.score(features_two, target))



