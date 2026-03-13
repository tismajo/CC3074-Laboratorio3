# ============================================================
# Exploratory Data Analysis - Ames Housing Dataset
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"]=(10,6)

DATA_PATH="data"
TRAIN_PATH=os.path.join(DATA_PATH,"train.csv")
TEST_PATH=os.path.join(DATA_PATH,"test.csv")

def load_data():
    print("Loading datasets...")
    train=pd.read_csv(TRAIN_PATH)
    test=pd.read_csv(TEST_PATH)
    print("Train shape:",train.shape)
    print("Test shape:",test.shape)
    return train,test

def dataset_overview(df):
    print("\n==============================")
    print("Dataset Overview")
    print("==============================")
    print("\nFirst rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe())
    print("\nDescribe categorical:")
    print(df.describe(include=["object"]))

def missing_values(df):
    print("\n==============================")
    print("Missing Values")
    print("==============================")
    missing=df.isnull().sum()
    missing=missing[missing>0]
    missing_percent=(missing/len(df))*100
    missing_df=pd.DataFrame({"Missing Values":missing,"Percentage":missing_percent}).sort_values(by="Missing Values",ascending=False)
    print(missing_df)
    plt.figure(figsize=(12,6))
    sns.barplot(x=missing_df.index,y=missing_df["Percentage"])
    plt.xticks(rotation=90)
    plt.title("Missing Values Percentage")
    plt.ylabel("Percentage")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()

def target_analysis(train):
    print("\n==============================")
    print("Target Variable Analysis")
    print("==============================")
    print(train["SalePrice"].describe())
    plt.figure()
    sns.histplot(train["SalePrice"],kde=True)
    plt.title("SalePrice Distribution")
    plt.show()
    plt.figure()
    sns.histplot(np.log1p(train["SalePrice"]),kde=True)
    plt.title("Log(SalePrice) Distribution")
    plt.show()

def correlation_analysis(train):
    print("\n==============================")
    print("Correlation Analysis")
    print("==============================")
    numeric=train.select_dtypes(include=[np.number])
    corr=numeric.corr()
    corr_target=corr["SalePrice"].sort_values(ascending=False)
    print("\nTop correlations with SalePrice:")
    print(corr_target.head(15))
    print("\nLowest correlations:")
    print(corr_target.tail(15))
    plt.figure(figsize=(12,10))
    sns.heatmap(corr,cmap="coolwarm",vmax=0.8)
    plt.title("Correlation Heatmap")
    plt.show()

def top_features_analysis(train):
    numeric=train.select_dtypes(include=[np.number])
    corr=numeric.corr()["SalePrice"].sort_values(ascending=False)
    top_features=corr.index[1:6]
    print("\nTop correlated features:",list(top_features))
    for feature in top_features:
        plt.figure()
        sns.scatterplot(x=train[feature],y=train["SalePrice"])
        plt.title(f"{feature} vs SalePrice")
        plt.show()

def categorical_analysis(train):
    categorical=train.select_dtypes(include=["object"]).columns
    print("\n==============================")
    print("Categorical Features")
    print("==============================")
    print("Total categorical:",len(categorical))
    for col in categorical[:10]:
        plt.figure()
        sns.boxplot(x=train[col],y=train["SalePrice"])
        plt.xticks(rotation=45)
        plt.title(f"{col} vs SalePrice")
        plt.show()

def outlier_analysis(train):
    print("\n==============================")
    print("Outlier Analysis")
    print("==============================")
    plt.figure()
    sns.scatterplot(x=train["GrLivArea"],y=train["SalePrice"])
    plt.title("GrLivArea vs SalePrice")
    plt.show()
    plt.figure()
    sns.boxplot(x=train["SalePrice"])
    plt.title("SalePrice Boxplot")
    plt.show()

train,test=load_data()
dataset_overview(train)
missing_values(train)
target_analysis(train)
correlation_analysis(train)
top_features_analysis(train)
categorical_analysis(train)
outlier_analysis(train)
