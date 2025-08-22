<H3>NAME - HARSHITHA V</H3>
<H3>REGISTER NO - 212223230074</H3>
<H1 ALIGN =CENTER>EX. NO.1 Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### STEP 1: Importing the libraries
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
```
### STEP 2: Importing the dataset
```
dataset = pd.read_csv("Churn_Modelling.csv")
print("Original Data:")
print(dataset.head())
```
### STEP 3: Taking care of missing data
```
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
```
### STEP 4: Encoding categorical data
```
for col in dataset.select_dtypes(include=['object']).columns:
    dataset[col] = LabelEncoder().fit_transform(dataset[col])

print("\nAfter Handling Missing Values & Encoding:")
print(dataset.head())
```
### STEP 5: Normalizing the data
```
scaler = StandardScaler()
dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
print("\nNormalized Data:")
print(dataset.head())
```
### STEP 6: Splitting the data into test and train
```
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\nTrain and Test Shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

```


## OUTPUT:

### Importing the dataset

<img width="665" height="416" alt="image" src="https://github.com/user-attachments/assets/4c4430fb-8326-459d-9fa8-770dadc45ff6" />

### Encoding categorical data

<img width="663" height="425" alt="image" src="https://github.com/user-attachments/assets/9e00a571-21dd-41ba-a23f-60291a4c8f26" />

### Normalizing the data

<img width="685" height="420" alt="image" src="https://github.com/user-attachments/assets/ab62131a-1e74-4e79-899b-ae0c2643ea27" />

### Splitting the data into test and train

<img width="680" height="56" alt="image" src="https://github.com/user-attachments/assets/9e415766-6833-4890-a23f-a1d66e5c9529" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


