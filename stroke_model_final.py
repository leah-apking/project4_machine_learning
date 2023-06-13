# -*- coding: utf-8 -*-
"""stroke_model_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V5qsxm2yOwD9yzSygo9CClfdB9m9Tl2e
"""

# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib

import os

spark_version = 'spark-3.4.0'
os.environ['SPARK_VERSION']=spark_version

# Install Spark and Java
!apt-get update
!apt-get install openjdk-11-jdk-headless -qq > /dev/null
!wget -q http://www.apache.org/dist/spark/$SPARK_VERSION/$SPARK_VERSION-bin-hadoop3.tgz
!tar xf $SPARK_VERSION-bin-hadoop3.tgz
!pip install -q findspark

# Set Environment Variables
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["SPARK_HOME"] = f"/content/{spark_version}-bin-hadoop3"

# Start a SparkSession
import findspark
findspark.init()

"""## **Import Dataset from AWS with Spark**"""

# Import packages
from pyspark.sql import SparkSession
import time

# Create a SparkSession
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# Read in the healthcare-dataset-stroke-data.csv via AWS into Spark DataFrame
from pyspark import SparkFiles
url = "https://project4-06052023.s3.us-east-2.amazonaws.com/healthcare-dataset-stroke-data.csv"
spark.sparkContext.addFile(url)
stroke_data = spark.read.csv(SparkFiles.get("healthcare-dataset-stroke-data.csv"), sep=",", header=True, inferSchema=True)
stroke_data.show()

"""## **Preproccessing**"""

# Print Spark dataframe schema (Note: all schema except 'bmi' inferred correctly)
stroke_data.printSchema

# Convert Spark dataframe to Pandas df
stroke_data_df = stroke_data.toPandas()

# Drop the non-beneficial ID column.
stroke_df = stroke_data_df.drop(columns={'id'})
stroke_df.info()

# Convert 'bmi' to float (Note: 'coerce' converts 'N/A' values to NaN)
stroke_df['bmi'] = pd.to_numeric(stroke_df['bmi'], errors ='coerce')
stroke_df.info()

# Drop rows containing NaN
stroke_df = stroke_df.dropna()
stroke_df

# Convert categorical data to numeric with `pd.get_dummies`
encoded_stroke_data = pd.get_dummies(stroke_df)
encoded_stroke_data

# Split our preprocessed data into our features and target arrays
y = encoded_stroke_data["stroke"]
X = encoded_stroke_data.drop(["stroke"], axis=1)

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Check the balance of our target values
y.value_counts()

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

"""## **Resample Data with RandomOverSampler**"""

!pip install imblearn

# Instantiate the random oversampler model
ros = RandomOverSampler()

# Fit the original training data to the random_oversampler model
X_R, y_R = ros.fit_resample(X_train, y_train)

# Count the distinct values of the resampled labels data
y_R.value_counts()

# Scale the resampled data
X_train_scaled_R = X_scaler.transform(X_R)

"""## **Logistic Regression Model with RandomOverSampler**"""

# Logistic Regression model with RandomOverSampler
# Instantiate the Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=200)

# Fit the model using the resampled training data
model = logistic_regression_model.fit(X_train_scaled_R, y_R)

# Make a prediction using the testing data
predictions = logistic_regression_model.predict(X_test_scaled)

# Print the balanced_accuracy score of the model 
balanced_accuracy_score(y_test, predictions)

# Generate a confusion matrix for the model
matrix = confusion_matrix(y_test, predictions)
print(matrix)

# Print the classification report for the model
report = classification_report(y_test, predictions)
print(report)

# Save model
filename = 'stroke_model_LR.h5'
joblib.dump(model, filename)

# Check that model was saved correctly
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test_scaled, y_test)
print(result)