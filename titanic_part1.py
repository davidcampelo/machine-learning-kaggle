	# Import the Pandas library
import pandas as pd
def log(msg):
	print("****************************************************************************\n* "+msg)


# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)
	
log("Print the `head` of the train and test dataframes ")
print train.head() 
print test.head() 

log("Show the train dataframe structure")
print train.describe() 

log("Show the shape (dimensions) of train dataframe")
print train.shape

log("Show absolute numbers of male/female")
print train['Sex'].value_counts()

log("Show the percentages of male/female")
print train['Sex'].value_counts(normalize=True)

log("Show absolute numbers of survivors/non-survivors")
print train['Survived'].value_counts()

log("Show the percentages of survivors/non-survivors")
print train['Survived'].value_counts(normalize = True)

log("Show absolute numbers of female survivors/non-survivors")
print train['Survived'][train['Sex'] == 'female'].value_counts()

log("Show the percentages of male survivors/non-survivors")
print train['Survived'][train['Sex'] == 'male'].value_counts(normalize=True)

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
# XXX BUG: Some lines still have NaN value as the Age column was empty

log("Print normalized Survival Rates for passengers under 18")
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

log("Print normalized Survival Rates for passengers 18 or older")
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


log("Create a copy of the test dataframe and create a Survived column. This column will be used to compare the prediction assertiveness.")
# Create a copy of test: test_one
test_one = test.copy()
# Initialize a Survived column to 0
test_one['Survived'] = 0
# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one['Survived'][test_one['Sex'] == 'female'] = 1
print(test_one['Survived'].value_counts())
