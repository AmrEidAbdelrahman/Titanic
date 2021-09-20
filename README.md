# Titanic
build a ML model to predict who will survive on titanic.

```python
import numpy as np
import matplotlib as plt
import pandas as pd
import re
import sklearn
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, LabelEncoder , StandardScaler
from sklearn import svm , tree
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

```


```python
#load the data and convert it to dataFrame
titanic = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_org = test.copy()
PassengerId = test['PassengerId']
combine = [titanic,test]
#fill all empty with NAN
titanic = titanic.fillna(np.nan)
titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(titanic.isnull().sum())
print('='*100)
print(test.isnull().sum())
```

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    ====================================================================================================
    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64
    


```python
test.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>3</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>male</td>
      <td>38.5</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/O.Q. 3101262</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>3</td>
      <td>Ware, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>359309</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>3</td>
      <td>Peter, Master. Michael J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
#fill the data with empty Embarked , Age from train set
titanic.loc[titanic["Embarked"].isnull(),"Embarked"] = 'S'

for dataset in [titanic,test]:
    avg_age = dataset['Age'].mean()
    std_age = dataset['Age'].std()
    number_of_null_age = dataset['Age'].isnull().sum()
    random_age_list = np.random.randint(avg_age - std_age , avg_age + std_age , size=number_of_null_age)
    dataset.loc[dataset['Age'].isnull() , 'Age'] = random_age_list
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['categoricalAge'] = pd.cut(dataset['Age'] , 5)
    

```


```python
titanic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>categoricalAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>(16.0, 32.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>(32.0, 48.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>(16.0, 32.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>(32.0, 48.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>(32.0, 48.0]</td>
    </tr>
  </tbody>
</table>
</div>




```python
#fill the data with empty Fare , Age from test set
Fare_mean = np.mean(test["Fare"])
test.loc[test["Fare"].isnull(),"Fare"] = Fare_mean

```


```python
print(titanic.isnull().sum())
print('='*100)
print(test.isnull().sum())
```

    PassengerId         0
    Survived            0
    Pclass              0
    Name                0
    Sex                 0
    Age                 0
    SibSp               0
    Parch               0
    Ticket              0
    Fare                0
    Cabin             687
    Embarked            0
    categoricalAge      0
    dtype: int64
    ====================================================================================================
    PassengerId         0
    Pclass              0
    Name                0
    Sex                 0
    Age                 0
    SibSp               0
    Parch               0
    Ticket              0
    Fare                0
    Cabin             327
    Embarked            0
    categoricalAge      0
    dtype: int64
    


```python
for dataset in [titanic,test]:
    dataset['Family_size'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['categoricalFare'] = pd.qcut(dataset['Fare'],4)
    
for dataset in [titanic,test]:
    dataset['Isalone'] = 0
    dataset.loc[dataset['Family_size'] == 1 , 'Isalone'] = 1
    
```


```python
for dataset in [titanic,test]:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
```


```python
for dataset in [titanic,test]:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

```


```python
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in [titanic,test]:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in [titanic,test]:
    dataset['Namelen'] = dataset['Name'].apply(len)
    
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    factorized_sex = pd.factorize(dataset["Title"])[0]
    dataset["Title"] = factorized_sex
```


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>categoricalAge</th>
      <th>Family_size</th>
      <th>categoricalFare</th>
      <th>Isalone</th>
      <th>Title</th>
      <th>Namelen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>0</td>
      <td>NaN</td>
      <td>Q</td>
      <td>(30.4, 45.6]</td>
      <td>1</td>
      <td>(-0.001, 7.896]</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>0</td>
      <td>NaN</td>
      <td>S</td>
      <td>(45.6, 60.8]</td>
      <td>2</td>
      <td>(-0.001, 7.896]</td>
      <td>0</td>
      <td>1</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>1</td>
      <td>NaN</td>
      <td>Q</td>
      <td>(60.8, 76.0]</td>
      <td>1</td>
      <td>(7.896, 14.454]</td>
      <td>1</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>1</td>
      <td>NaN</td>
      <td>S</td>
      <td>(15.2, 30.4]</td>
      <td>1</td>
      <td>(7.896, 14.454]</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>1</td>
      <td>NaN</td>
      <td>S</td>
      <td>(15.2, 30.4]</td>
      <td>3</td>
      <td>(7.896, 14.454]</td>
      <td>0</td>
      <td>1</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
factorized_sex = pd.factorize(titanic["Sex"])[0]
titanic["Sex"] = factorized_sex
factorized_sex = pd.factorize(titanic["Embarked"])[0]
titanic["Embarked"] = factorized_sex

factorized_sex = pd.factorize(test["Sex"])[0]
test["Sex"] = factorized_sex
factorized_sex = pd.factorize(test["Embarked"])[0]
test["Embarked"] = factorized_sex
```


```python
for dataset in [titanic,test]:
    dataset['hasCabin'] = 1
    dataset.loc[dataset['Cabin'].isnull() , 'hasCabin' ] = 0
```


```python
# #code categorical data
# label = LabelEncoder()
# for dataset in [titanic,test]:    
#     dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
#     dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
#     dataset['Title_Code'] = label.fit_transform(dataset['Title'])
#     dataset['AgeBin_Code'] = label.fit_transform(dataset['categoricalAge'])
#     dataset['FareBin_Code'] = label.fit_transform(dataset['categoricalFare'])
    

```


```python
# #define y variable aka target/outcome
# Target = ['Survived']

# #define x variables for original features aka feature selection
# data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'Family_size', 'Isalone'] #pretty name/values for charts
# data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
# data1_xy =  Target + data1_x
# print('Original X Y: ', data1_xy, '\n')


# #define x variables for original w/bin features to remove continuous variables
# data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
# data1_xy_bin = Target + data1_x_bin
# print('Bin X Y: ', data1_xy_bin, '\n')


# #define x and y variables for dummy features original
# data1_dummy = pd.get_dummies(titanic[data1_x])
# data1_dummy = pd.get_dummies(test[data1_x])
# data1_x_dummy = data1_dummy.columns.tolist()
# data1_xy_dummy = Target + data1_x_dummy
# print('Dummy X Y: ', data1_xy_dummy, '\n')



# data1_dummy.head()
```


```python
drop_elements = ['PassengerId' , 'Name' , 'SibSp' , 'Cabin' , 'Ticket' , 'Sex' , 'Embarked']
titanic = titanic.drop(drop_elements , axis=1)
titanic = titanic.drop(['categoricalAge' , 'categoricalFare'] , axis=1)

test = test.drop(drop_elements , axis=1)
# titanic = titanic.drop(['categoricalAge' , 'categoricalFare'] , axis=1)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Family_size</th>
      <th>Isalone</th>
      <th>Title</th>
      <th>Namelen</th>
      <th>hasCabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>51</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>44</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_target = titanic["Survived"].values
titanic = titanic.drop(['Survived'],axis = 1)
x_feature = titanic

```


```python

x_train , x_valid , y_train , y_valid = train_test_split(x_feature,y_target,test_size = 0.25)
```


```python
reg = LogisticRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_valid)
acc_reg = round(reg.score(x_valid,y_valid),2)
acc_reg
```

    c:\users\amrei\appdata\local\programs\python\python39\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    0.8




```python
svmc = svm.SVC()
svmc.fit(x_train,y_train)
y_pred = svmc.predict(x_valid)
acc_svmc = round(svmc.score(x_valid , y_valid)*100,2)
acc_svmc
```




    71.75




```python
dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train,y_train)
y_pred2 = dtree.predict(x_valid)
acc_dtree = round(dtree.score(x_valid,y_valid)*100 ,2)
acc_dtree
```




    79.82




```python
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
Y_pred = knn.predict(x_valid)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn
```




    86.53




```python
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_valid)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest
```




    96.71




```python
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_valid)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd
```




    70.81




```python
# Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
Y_pred = perceptron.predict(x_valid)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron
```




    80.24




```python
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train,y_train)
Y_pred = gaussian.predict(x_valid)
acc_gaussian = round(gaussian.score(x_train,y_train) * 100, 2)
acc_gaussian
```




    74.7




```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN',
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Decision Tree'],
    'Score': [acc_svmc, acc_knn, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd,  acc_dtree]})
models.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Support Vector Machines</td>
      <td>71.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>86.53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>96.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Naive Bayes</td>
      <td>74.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Perceptron</td>
      <td>80.24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Stochastic Gradient Decent</td>
      <td>70.81</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Decision Tree</td>
      <td>79.82</td>
    </tr>
  </tbody>
</table>
</div>




```python
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_feature,y_target)

```




    RandomForestClassifier()




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>categoricalAge</th>
      <th>Family_size</th>
      <th>categoricalFare</th>
      <th>Isalone</th>
      <th>Title</th>
      <th>Namelen</th>
      <th>hasCabin</th>
      <th>Sex_Code</th>
      <th>Embarked_Code</th>
      <th>Title_Code</th>
      <th>AgeBin_Code</th>
      <th>FareBin_Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>(30.4, 45.6]</td>
      <td>1</td>
      <td>(-0.001, 7.896]</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>(45.6, 60.8]</td>
      <td>2</td>
      <td>(-0.001, 7.896]</td>
      <td>0</td>
      <td>1</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>(60.8, 76.0]</td>
      <td>1</td>
      <td>(7.896, 14.454]</td>
      <td>1</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>(15.2, 30.4]</td>
      <td>1</td>
      <td>(7.896, 14.454]</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>(15.2, 30.4]</td>
      <td>3</td>
      <td>(7.896, 14.454]</td>
      <td>0</td>
      <td>1</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions = random_forest.predict(test)
```


```python
output = pd.DataFrame({'PassengerId': test_org.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
```

    Your submission was successfully saved!
    


```python

```
