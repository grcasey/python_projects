# North Carolina Single Family Residential Properties

## Problem: 
Scrap data from the Mecklenburg County website (https://property.spatialest.com/nc/mecklenburg/)
on single family residential properties, write to db, explore and tell a story.

## My solution:
#### 1. Data appropriation 
Data harvesting involved a web-scraping from https://property.spatialest.com/nc/mecklenburg/
Results on the page are filtered for Single Family Residential properties
I analyzed the javascript activity and Ajax-calls on the page, and was able to take use of the same API calls for scraping structured JSON data.
The scrapping functions are stored in the separate folder and file under the name 'scraper'

#### 2. Database table set-up, writing data
To form a table in my DB I call my library 'scraper.py' to crawl on the website and return data
The idea for schema is to store all meaningful data potentially useful for further db selections and analysis.
I also write a JSON blob to the table to store initial piece of data that has been utilized for scrapping

#### 3. Reading data for further analysis
Data is also written to CSV-file to allow for alternative access to data in case of not having access to paramaters of db

#### 4. Exploration of data, preprocessing
Trying different approaches to filtering information on the Mecklenburg County website, it's appeared that I can only access a sample of data that is limited to 400 records
The data stored in db and in csv-file. Can be access from the both places.
I read data and create Pandas-DataFrame.
Further steps include exploration, cleaning, preprossessing, handling missing values.
While exploring, I visualized 10 neighborhoods with the highest number of listed properties from a sample. 
The figure is stored to PNG-file in the output folder.

#### 5. Modeling - Building a Machine Learning model to predict assessed value of a property.
My sample is limited to 400 records. I further hypothesize that parameters for each single property that I appropriated can predict Total Assessed Value.
I drop some columns that are not meaningful for multivariable regression model I consider to build.

#### Parameters I account for (independent variables):
Year Built
Last Sale Date
Heat system
Heat Fuel
Foundation
Number of Half Bathrooms
Number of Full Bathrooms
Story 
External walls (material)
Area (SqFt)
Neighborhood

#### Dependent variable (what I am trying to predict):
Total Assessed Value of a property

The sample size is divided to train and test sets.

#### Cross-validation
I used cross-validation to decide on regressor that gives me the best value of the regression score value. 
The regression models I considered were: 
Multilinear Regression, 
Support Vector Regression,
Polynomial Regression,
Decision Tree,
Random Forest.

K-fold cross-validation method shows that Random Forest Regression approach delivers the highest regression score value (0.82).
I further run Random Forest regressor to get predicted outcomes for my dependent variable.
The resulting predictions and actual dependent values are written to the resulting CSV-file 'compare model outcomes.csv'

```
﻿# Random Forest regression
rand_regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
scoring = 'r2' # (coefficient of determination) regression score function, best score = 1, worst = 0, can be negative
score = cross_val_score(rand_regressor, X_train, y_train, cv=k_fold, scoring=scoring)
round(np.mean(score),5)
# resulting score 0.82237
```

```
compare = pd.DataFrame({'actual total assessed value': y_test, 'predicted total assessed value': y_pred})
compare.to_csv('output/compare model outcome.csv', index=False, index_label=False)
```

#### Conclusion:
The data analysis showed a well established ground for predictive modeling. 
As I can see, there are enough parameters for the algorithm to predict total assessed value for properties with high regression scoring value.
For any other sample of single family residential properties in Mecklenburg County, the model will predict the total assessed value with a high degree of accuracy. 



