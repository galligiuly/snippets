# Machine Learning - Supervised Learning

Supervised Learning requires Labelled Training Data.

Then a relationship between input and output is built. The results is a:
- Regressor: output is a Number
- Classifier: output is a Class


### Cross Validation

The dataset is split into n random parts. Is just the process of splitting your data into multiple pairs of training and test sets.

In sklearn you can use cross-validation with any classifier/regressor you'd like.


```python
# Load the library
from sklearn.model_selection import cross_val_score
# We calculate the metric for several subsets (determine by cv)
# With cv=5, we will have 5 results from 5 training/test
cross_val_score(reg,X,y,cv=5,scoring="neg_mean_squared_error")
````

Testing Parameters: GridSearchCV
We could try to find the best parameters by testing all of the combinations of them. We test a GRID of parameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
reg_test = GridSearchCV(KNeighborsRegressor(),
                       param_grid={"n_neighbors":np.arange(3,50)})
# Fit will test all of the combinations
reg_test.fit(X,y)

# Best estimator and best parameters
reg_test.best_score_
reg_test.best_estimator_
reg_test.best_params_
```

Randomized Grid Search

```python
from sklearn.model_selection import RandomizedSearchCV
reg = RandomizedSearchCV(DecisionTreeRegressor(),
                  param_distributions={"max_depth":np.arange(2,8),
                              "min_samples_leaf":[10,30,50,100]},
                  cv=5,
                  scoring="neg_mean_absolute_error",
                  n_iter=5      )
reg.fit(X,y)
````
 
 ## Regression
 
Models:

● Linear Regression
● k neighbor Regressor
● Decision Tree

Metrics:

● RMSE
● MAE and MAPE
● Correlation and Bias

### Linear Regression with Sklearn

Preparing data

```python
# Input
X = df[['TotalSF']] # pandas DataFrame
# Label
y = df["SalePrice"] # pandas Series
```


#### Load the library
```python
from sklearn.linear_model import LinearRegression
```
#### Create an instance of the model
```python
reg = LinearRegression()
```
#### Fit the regressor
```python
reg.fit(X,y)
```
#### Do predictions
```python
reg.predict([[2540],[3500],[4000]])
````

Train and Test split

```python
# Load the library
from sklearn.model_selection import train_test_split
# Create 2 groups each with input and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
# Fit only with training data
reg.fit(X_train,y_train)
```

Metric MAE

```python
# Load the scorer
from sklearn.metrics import mean_absolute_error
# Use against predictions
mean_absolute_error(reg.predict(X_test),y_test)
````

Metric MAPE (not in skalern)

```python
np.mean(np.abs(reg.predict(X_test)-y_test)/y_test)
````

### k Nearest Neighbors

Parameters:
**k**: Number of neighbors
**weight**: Way to combine the label of the nearest point
    Uniform: All the same
    Distance: Weighted Average per distance
    Custom: Weighted Average provided by user
**partition**: Way to partition the training dataset (ball_tree, kd_tree, brute)

```python
# Load the library
from sklearn.neighbors import KNeighborsRegressor
# Create an instance
regk = KNeighborsRegressor(n_neighbors=2)
# Fit the data
regk.fit(X,y)
````

Metric RMSE (penalizes more higt values of error)

```python 
# Load the scorer
from sklearn.metrics import mean_squared_error
# Use against predictions (we must calculate the square root of the MSE)
np.sqrt(mean_squared_error(reg.predict(X_test),y_test))
````

### Decision Tree

A decision tree is a structure that includes a root node, branches, and leaf nodes. Each internal node denotes a test on an attribute, each branch denotes the outcome of a test, and each leaf node holds a class label. The topmost node in the tree is the root node.
 
 
● Start at the training dataset
● For each feature:
    ○ Split in 2 partitions
    ○ Calculate the purity/homogeneity gain
● Keep the feature split with the best gain
● Repeat for the 2 new partitions
 
Parameters:
Max_depth: Number of Splits
Min_samples_leaf: Minimum number of observations per leaf

```python
# Load the library
from sklearn.tree import DecisionTreeRegressor
# Create an instance
regd = DecisionTreeRegressor(max_depth=3)
# Fit the data
regd.fit(X,y)
````

Metric: Correlation (measures the correlation between the predictions and the real value).

```python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]
# Custom Scorer
from sklearn.metrics import make_scorer
def corr(pred,y_test):
      return np.corrcoef(pred,y_test)[0][1]
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(corr))
```

Metric: Bias (the average of errors).

```python
# Direct Calculation
np.mean(reg.predict(X_test)-y_test)
# Custom Scorer
from sklearn.metrics import make_scorer
def bias(pred,y_test):
return np.mean(pred-y_test)
# Put the scorer in cross_val_score cross_val_score(reg,X,y,cv=5,scoring=make_scorer(bias))
```

Drawing the decision tree

```python
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
```


### Random Forest

```python
# Load the library
from sklearn.ensemble import RandomForestRegressor
# Create an instance
clf = RandomForestRegressor(max_depth=4)
# Fit the data
clf.fit(X,y)
```


### Gradient Boosting Tree

```python
# Load the library
from sklearn.ensemble import GradientBoostingRegressor
# Create an instance
clf = GradientBoostingRegressor(max_depth=4)
# Fit the data
clf.fit(X,y)
```


## Classification


### Logistic regression

```python
# Load the library
from sklearn.linear_model import LogisticRegression
# Create an instance of the classifier
clf=LogisticRegression()
# Fit the data
clf.fit(X,y)
```

Metric: Accuracy

#### With Metrics

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test,clf.predict(X_test))
```
#### Cross Validation
```python
cross_val_score(clf,X,y,scoring="accuracy")
````

###  k nearest neighbor

```python
# Load the library
from sklearn.neighbors import KNeighborsClassifier
# Create an instance
regk = KNeighborsClassifier(n_neighbors=2)
# Fit the data
regk.fit(X,y)
```

Metric: Precision and Recall

```python
# Metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
precision_score(y_test,clf.predict(X_test))
classification_report(y_test,clf.predict(X_test))
# Cross Validation
cross_val_score(clf,X,y,scoring="precision")
cross_val_score(clf,X,y,scoring="recall")
```

###  Support Vector Machine


Parameters:

    C: Sum of Error Margins
    kernel:
    linear: line of separation
    rbf: circle of separation
        Additional param gamma: Inverse of the radius
    poly: curved line of separation
        Additional param degree: Degree of the polynome



```python
# Load the library
from sklearn.svm import SVC
# Create an instance of the classifier
clf = SVC(kernel="linear",C=10)
# Fit the data
clf.fit(X,y)
```

###  Decision Tree

```python
# Import library
from sklearn.tree import DecisionTreeClassifier
# Create instance
clf = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)
# Fit the data
clf.fit(X,y)
```

### Random Forest

```python
# Load the library
from sklearn.ensemble import RandomForestClassifier
# Create an instance
clf = RandomForestClassifier(max_depth=4)
# Fit the data
clf.fit(X,y)
```

### Gradient Boosting Tree

```python
# Load the library
from sklearn.ensemble import GradientBoostingClassifier
# Create an instance
clf = GradientBoostingClassifier(max_depth=4)
# Fit the data
clf.fit(X,y)
```

### Train-test split

```python
# Load the library
from sklearn.model_selection import train_test_split
# Create 2 groups each with input and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
# Fit only with training data
reg.fit(X_train,y_train)
```

## Metrics

### Regression

### MAE

```python
# Load the scorer
from sklearn.metrics import mean_absolute_error
# Use against predictions
mean_absolute_error(reg.predict(X_test),y_test)
```

### MAPE

```python
np.mean(np.abs(reg.predict(X_test)-y_test)/y_test)
````

### RSME

```python
# Load the scorer
from sklearn.metrics import mean_squared_error
# Use against predictions (we must calculate the square root of the MSE)
np.sqrt(mean_squared_error(reg.predict(X_test),y_test))
````

### Correlation

```python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]
# Custom Scorer
from sklearn.metrics import make_scorer
def corr(pred,y_test):
return np.corrcoef(pred,y_test)[0][1]
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(corr))
````


### Bias

```python
# Direct Calculation
np.mean(reg.predict(X_test)-y_test)
# Custom Scorer
from sklearn.metrics import make_scorer
def bias(pred,y_test):
return np.mean(pred-y_test)
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(bias))
````

## Classification

### Accuracy

```python
# With Metrics
from sklearn.metrics import accuracy_score
accuracy_score(y_test,clf.predict(X_test))
# Cross Validation
cross_val_score(clf,X,y,scoring="accuracy")
````

### Precision and Recall

```python
# Metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
precision_score(y_test,clf.predict(X_test))
classification_report(y_test,clf.predict(X_test))
# Cross Validation
cross_val_score(clf,X,y,scoring="precision")
cross_val_score(clf,X,y,scoring="recall")
````


### ROC curve

```python
# Load the library
from sklearn.metrics import roc_curve
# We chose the target
target_pos = 1 # Or 0 for the other class
fp,tp,_ = roc_curve(y_test,pred[:,target_pos])
plt.plot(fp,tp)
````

### AUC

```python
# Metrics
from sklearn.metrics import roc_curve, auc
fp,tp,_ = roc_curve(y_test,pred[:,1])
auc(fp,tp)
# Cross Validation
cross_val_score(clf,X,y,scoring="roc_auc")
````




















