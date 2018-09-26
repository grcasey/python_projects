# North Carolina Single Family Residential Properties.

## Problem: 
Scrap data from the Mecklenburg County website (https://property.spatialest.com/nc/mecklenburg/)
on single family residential properties, write to db, explore and tell a story.

## My solution:
#### 1. Data appropriation.
Data harvesting involved creating a custom made web-scraper for https://property.spatialest.com/nc/mecklenburg/
Results on the page are filtered for Single Family Residential properties.

For the scraper I analyzed the javascript activity and Ajax-calls on the page, and was able to take use of the same API-calls for scraping structured JSON data. The reusable scrapping functions are stored in the separate folder and file under the name 'scraper'.

#### 2. Database table set-up, writing data.
For simplicity I choose to use my existing AWS MySQL database and create a new schema for the property data.

For the schema I used a sample selection of relevant fields of meaningful data from the scraped JSON.
I also put the original JSON blob in a separate field for future use.

#### 3. Reading data for further analysis.
Data is also written to CSV-file to allow for alternative access to data in case of not having access to db.

#### 4. Exploration of data, preprocessing.
Limitations: Trying different approaches of filtering information on the Mecklenburg County website, it's appeared that I can only access a sample of data that is limited to 400 records per search. 
The JSON data is parsed and stored in db and in a CSV-file. 
I read the data back out again from CSV-file (but also can be read from db) and create Pandas-DataFrame.

Further steps include exploration, cleaning, preprocessing, handling of missing values.
While exploring, I visualized 10 neighborhoods with the highest number of listed properties from a sample. 
The figure is stored to PNG-file in the output folder.

#### 5. Modeling - Building a Machine Learning model to predict total assessed value of a property.
My sample is limited to 400 records. I further hypothesize that parameters for each single property that I appropriated can predict Total Assessed Value.

I drop some columns that are not meaningful for multivariable regression model. The sample size is divided to train and test sets.

#### Parameters I account for (independent variables):
Year Built,
Last Sale Date,
Heat system,
Heat Fuel,
Foundation,
Number of Half Bathrooms,
Number of Full Bathrooms,
Story, 
External walls (material),
Area (SqFt),
Neighborhood,

#### Dependent variable (what I am trying to predict):
Total Assessed Value of a property


#### Cross-validation.
I used cross-validation to decide on regressor that gives me the best value of the regression score value. 
The regression models I considered were: 
Multilinear Regression, 
Support Vector Regression,
Polynomial Regression,
Decision Tree,
Random Forest.

K-fold cross-validation method shows that Random Forest Regression approach delivers the highest regression score value (0.82).
I further run Random Forest regressor to get predicted outcomes for my dependent variable.
The resulting predictions and actual dependent values are written to the resulting CSV-file 'compare model outcomes.csv'.

```
# Random Forest regression
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

#### First ten predicted values in comparison with actual total assessed values of listed properties.
```
actual total assessed value	predicted total assessed value
119400	119443
250100	295399
118400	115158
160500	160863
90300	70357
206900	173242
361300	286189
117600	123418
76200	63884
141300	146337
```


