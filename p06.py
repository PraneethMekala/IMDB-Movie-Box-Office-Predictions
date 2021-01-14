# p06.py Movie Box Office Prediction
# updated 12/10/2019 by Praneeth Mekala
#
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import json
import sys
import csv
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
%matplotlib inline

pd.set_option('display.width', 800)
os.getcwd()
os.chdir('/Users/praneethmekala/Desktop/Data Science bootcamp/Week 6/Project 06/')

train = pd.read_csv("movie_train.csv")
test = pd.read_csv("movie_test.csv")

train.shape, test.shape
train.info()
test.info()
train.describe()
train.isna().sum()
test.isna().sum()

# Correlation chart
sns.jointplot(x="popularity", y="revenue", ratio=6, data=train, color="black")
plt.show()

sns.jointplot(x="budget", y="revenue", data=train, ratio=4, color="g", kind="reg")
plt.show()

sns.jointplot("runtime", "revenue", data=train, ratio=4, color="b")
plt.show()

# Distribution chart
sns.distplot(train.revenue)

train.revenue.describe()

# Create a new variable in the data
train['logRevenue'] = np.log1p(train['revenue'])
sns.distplot(train['logRevenue'] )

train['logBudget'] = np.log1p(train['budget'])
sns.distplot(train['logBudget'] )

# Since only last two digits of year are provided, this is the correct way of getting the year.
train[['release_month','release_day','release_year']] = train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)
# Some rows have 4 digits of year instead of 2, that's why I am applying (train['release_year'] < 100) this condition
train.loc[ (train['release_year'] <= 19) & (train['release_year'] < 100), "release_year"] += 2000
train.loc[ (train['release_year'] > 19)  & (train['release_year'] < 100), "release_year"] += 1900

# Powerful Time-stamp conversion
releaseDate = pd.to_datetime(train['release_date']) 
train['release_dayofweek'] = releaseDate.dt.dayofweek
train['release_quarter'] = releaseDate.dt.quarter

plt.figure(figsize=(10,6))
sns.countplot(train['release_year'].sort_values())
plt.title("Movie Release Count by Year",fontsize=20)
loc, labels = plt.xticks()
plt.xticks(fontsize=8,rotation=90)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(train['release_month'].sort_values())
plt.title("Movie Release Month Count",fontsize=20)
loc, labels = plt.xticks()
loc, labels = loc, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.xticks(loc, labels,fontsize=15)
plt.show()

train['meanRevenueByMonth'] = train.groupby("release_month")["revenue"].aggregate('mean')
train['meanRevenueByMonth'].plot(figsize=(15,10),color="b")
plt.xlabel("Release Month")
plt.ylabel("Revenue")
plt.title("Movie Mean Revenue Release Month",fontsize=20)
plt.xlim(1,12)
plt.show()

# Alternative way
group = train['revenue'].groupby(train['release_month'])
by_month = group.mean()
by_month.plot(kind='bar', figsize=(12, 6),color="b");
plt.xlabel("Release Month")
plt.ylabel("Revenue")
plt.title("Movie Mean Revenue By Release Month",fontsize=20)
plt.show()

train['meanRevenueByYear'] = train.groupby("release_year")["revenue"].aggregate('mean')
train['meanRevenueByYear'].plot(figsize=(10,6),color="g")
plt.xticks(np.arange(1920,2018,10))
plt.xlabel("Release Year")
plt.ylabel("Revenue")
plt.title("Movie Mean Revenue By Year",fontsize=10)
plt.xlim(1921,2017)
plt.show()

# Alternative way
group = train['revenue'].groupby(train['release_year'])
by_year = group.mean()
by_year.plot(kind='bar', figsize=(20, 8),color="g");
plt.xlabel("Release Year")
plt.ylabel("Revenue")
plt.title("Movie Mean Revenue By Release Year",fontsize=20)
plt.show()

def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

train = train

# Lambda and Map functions
# Map applies a function to all the value in an input list
def add(x,y):
    return x+y

add(2,3)

add1 = lambda x,y: x+y
add1(3,4)

input = [1,2,3,4,5,6]
squared = list(map(lambda x: x**2, input))
squared

train['genres'] = train['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

# get-dummies is a one-hot encoding method
genres = train.genres.str.get_dummies(sep=',')
train = pd.concat([train, genres], axis=1, sort=False)

# shape[0] shows the number of rows (n)
print("Action Genres Movie           ", train[train.Action == 1].shape[0])
print("Adventure Genres Movie        ", train[train.Adventure == 1].shape[0])
print("Animation Genres Movie        ", train[train.Animation == 1].shape[0])
print("Comedy Genres Movie           ", train[train.Comedy == 1].shape[0])
print("Crime Genres Movie            ", train[train.Crime == 1].shape[0])
print("Documentary Genres Movie      ", train[train.Documentary == 1].shape[0])
print("Drama Genres Movie            ", train[train.Drama == 1].shape[0])
print("Family Genres Movie           ", train[train.Family == 1].shape[0])
print("Fantasy Genres Movie          ", train[train.Fantasy == 1].shape[0])
print("Foreign Genres Movie          ", train[train.Foreign == 1].shape[0])
print("History Genres Movie          ", train[train.History == 1].shape[0])
print("Music Genres Movie            ", train[train.Music == 1].shape[0])
print("Mystery Genres Movie          ", train[train.Mystery == 1].shape[0])
print("Romance Genres Movie          ", train[train.Romance == 1].shape[0])
print("Science Fiction Genres Movie  ", train[train['Science Fiction'] == 1].shape[0])
print("TV Movie Genres Movie         ", train[train['TV Movie'] == 1].shape[0])
print("Thriller Genres Movie         ", train[train.Thriller == 1].shape[0])
print("War Genres Movie              ", train[train.War == 1].shape[0])
print("Western Genres Movie          ", train[train.Western == 1].shape[0])

plt.figure(figsize=(20,15))
sns.countplot(train['original_language'].sort_values())
plt.title("Original Language Count",fontsize=20)
plt.show()

train1 = train[['budget','popularity','runtime','release_year','release_month','release_dayofweek','revenue']]
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(train1.corr(), annot=True)
plt.show()

train.info()

# Linear regression
y = train['logRevenue']
x = train.drop(['id', 'belongs_to_collection','genres','homepage','revenue','logRevenue','budget','imdb_id','original_language',
                'original_title','overview','poster_path','production_companies','production_countries','release_date',
                'spoken_languages','status','tagline','title','Keywords', 'cast','crew','meanRevenueByMonth', 'meanRevenueByYear'], axis=1)
    
x = sm.add_constant(x)
results = sm.OLS(y, x, missing='drop').fit()

print(results.summary())





