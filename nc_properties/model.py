import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def read_to_df():
# upload data by creating a Pandas-dataframe
    header_row=['id','address','area','building_type', 'neighborhood', 'number_bathroom', 'number_fireplace', 'owner1', 'owner2', 'total_value_property', 'year_build', 'json_blob', 'last_sale_date', 'heat', 'heat_fuel', 'foundation', 'num_half_bathroom', 'story', 'ext_wall', 'land_value', 'building_value', 'features_value'] 
    df = pd.read_csv('input/property.csv', names=header_row)
    # rearrange columns 
    cols1 = []
    df.columns.get_loc("total_value_property")
    cols = df.columns.tolist()
    cols[0] 
    cols1 = cols[0:1] + cols[10:] + cols[1:10]
    df = df[cols1]
    return df
    
    # check on data type of total_value_property withing a dataframe, it's a series of strings
    # type(df.iloc[22])
    
# call to create a DataFrame
df = read_to_df()

# preprocess 'total_value_property' to a series of integers
df['total_value_property'].replace(to_replace="\$([0-9,\.]+).*", value=r"\1", regex=True, inplace=True)
df['total_value_property'].replace(to_replace="[, ]+", value=r"", regex=True,inplace=True)
df['total_value_property'].astype(int)
df['total_value_property'].head()

# preprocess 'area' SqFt: series of str -> series of int, here account for NaNs
df['area'].replace(to_replace="[, ]+", value=r"", regex=True,inplace=True)
df.loc[df['area'].notnull(), 'area'] = df.loc[df['area'].notnull(), 'area'].apply(int)

# explore 'neighborhood' values
df.neighborhood.unique()
df.neighborhood.nunique(dropna=True) #111 out of 400 unique values 

# count IDs per each unique neighborhood
df.groupby('neighborhood')['id'].nunique()

# visualize 10 neighborhoods with the highest number of single family properties registered from the 400-sample size
neighbor = pd.DataFrame(df.neighborhood.value_counts().head(n=10))
sns.set(style="whitegrid")
ax = sns.barplot(x=neighbor.index, y=neighbor.neighborhood).set_title("10 neighborhoods with the highest number of registered properties in the sample.")
plt.xlabel("Neighborhood")
plt.ylabel("N of properties registered")
plt.savefig("output/10neigborhoods.png")
plt.show()

### Build machine learning model to predict total assessed value of a property given its parameters

# decide what columns not needed for analysis
# drop 'id', 'json blob', 'land_value', 'building_value', 'features_value', 'address', 'building_type', 'owner1', 'owner2'
# note: building / land / feature values are dropped to avoid multicolinearity 
dataset = df.copy(deep = True)
dataset.drop(['id', 'json_blob', 'land_value', 'building_value', 'features_value', 'address', 'building_type', 'owner1', 'owner2'], axis=1, inplace=True)

# parse dates of last sale str -> date type
pd.to_datetime(dataset['last_sale_date'])

# define independent and dependent valiables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 12].values

# encode 'neighborhood' values
labelencoder = LabelEncoder()
dataset.columns.get_loc("neighborhood")
X[:, 9] = labelencoder.fit_transform(X[:, 9])

# check if 'neighborhood' variable encoded propely
from collections import Counter
sum(Counter(X[:, 9]).values()) # counts the elements' frequency and sums it = 400
len(set(X[:, 9])) #sum of unique values for encoded 'neighborhood' variable = 111

# encode 'heat' variable
dataset.columns.get_loc("heat")
X[:, 2] = labelencoder.fit_transform(X[:, 2].astype(str))

# check encoding
# X[:, 1:3]

# double check 'last_sale_date' type
pd.to_datetime(X[:, 1]) # seems doesn't convert to 'consumable' format, need to encode

# encode 'last_sale_date' variable
dataset.columns.get_loc("heat")
X[:, 1] = labelencoder.fit_transform(X[:, 1].astype(str))

# encode 'heat_fuel' variable
dataset.columns.get_loc("heat")
X[:, 3] = labelencoder.fit_transform(X[:, 3].astype(str))

# encode 'foundation' variable
dataset.columns.get_loc("foundation")
X[:, 4] = labelencoder.fit_transform(X[:, 4].astype(str))

# encode 'story' variable
dataset.columns.get_loc("story")
X[:, 6] = labelencoder.fit_transform(X[:, 6].astype(str))

# encode 'ext_wall' variable
dataset.columns.get_loc("ext_wall")
X[:, 7] = labelencoder.fit_transform(X[:, 7].astype(str))

# take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:13])
X[:, 0:13] = imputer.transform(X[:, 0:13])

# convert y dependent variable to consumable format for modelling
y = list(map(int, y))

# split the dataset (or our X and y lists) into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# check nothing is missing
X_train.shape
X_test.shape

# import modules
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures #not sure
from sklearn.svm  import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# cross-validation (K-fold) - evaluate different model performance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# Multilinear regression 
lin_regressor = LinearRegression()
scoring = 'r2' # (coefficient of determination) regression score function, best score = 1, worst = 0, can be negative
score = cross_val_score(lin_regressor, X_train, y_train, cv=k_fold, scoring=scoring)
round(np.mean(score),5)
# resulting score = 0.72774


# SVR model requires feature scaling of data

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
sc_y = StandardScaler()
y_train_SVR = sc_y.fit_transform(np.reshape(y_train, (-1, 1)))

# SVR model
svr_regressor = SVR(kernel = 'rbf')
scoring = 'r2' # (coefficient of determination) regression score function, best score = 1, worst = 0, can be negative
score = cross_val_score(svr_regressor, X_train_SVR, y_train_SVR, cv=k_fold, scoring=scoring)
round(np.mean(score),5)  
# resulting score = 0.78338


# Decision Tree regression
tree_regressor = DecisionTreeRegressor(random_state = 0)
scoring = 'r2' # (coefficient of determination) regression score function, best score = 1, worst = 0, can be negative
score = cross_val_score(tree_regressor, X_train, y_train, cv=k_fold, scoring=scoring)
round(np.mean(score),5)
# resulting score 0.72591

# Random Forest regression
rand_regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
scoring = 'r2' # (coefficient of determination) regression score function, best score = 1, worst = 0, can be negative
score = cross_val_score(rand_regressor, X_train, y_train, cv=k_fold, scoring=scoring)
round(np.mean(score),5)
# resulting score 0.82237


# Polynomial regression
poly_reg = PolynomialFeatures(degree = 1)  # Playing with degree shows that the relation between variables is not exponential 
lin_reg = LinearRegression() 
X_poly = poly_reg.fit_transform(X_train)
scoring = 'r2' # (coefficient of determination) regression score function, best score = 1, worst = 0, can be negative
score = cross_val_score(lin_reg, X_poly, y_train, cv=k_fold, scoring=scoring)
round(np.mean(score),5)
# resulting score 0.72774 -because degree = 1 and it's only sensible degree value, mean score is equaled to the mean of score for multilinear regression

# Best score was achieved with Random Forest algorithm
# which is average of predictions of different trees in the forest
rand_regressor.fit(X_train, y_train)
y_pred = rand_regressor.predict(X_test).astype(int)

compare = pd.DataFrame({'actual total assessed value': y_test, 'predicted total assessed value': y_pred})
compare.to_csv('output/compare model outcome.csv', index=False, index_label=False)

