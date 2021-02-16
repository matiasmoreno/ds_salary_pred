# -*- coding: utf-8 -*-
"""
Created on Jan 9 2021

author: matiasmoreno
"""

# Data

# First we need to get some data. 
# Kaggle is a realy good scource from data of any type, 
# also the community is large, we can find tutorials, 
# competitions and los of datasets to work.

# Common libraries
import pandas as pd
import numpy as np

# Read data from file 'salary_data/train.csv' 
# (in the same directory that your python process is based
df = pd.read_csv('train.csv')

# Drop index column, inplace means that the function modifies the actual df

df.drop(["Unnamed: 0"], inplace=True, axis = 1 )

# Average Salary

df["salary_avg"] = df["salary"].apply(lambda x: (int(x.split("to")[0]) + int(x.split("to")[1])) / 2 )

# Average years of experience required

df["exp_avg"] = df["experience"].apply(lambda x: x[:-4])
df["exp_avg"] = df["exp_avg"].apply(lambda x: (int(x.split("-")[0]) + int(x.split("-")[1])) / 2 )

# Length of job description
df["description_length"] = df["job_description"].apply(lambda x: 0 if isinstance(x, float) else len(x)) 

# Extract some location and normalize
df["location"].value_counts()
locations = df["location"]

def location_normalize(loc):
    switch(loc):
        
    if 'bengaluru' in loc.lower():
        return 'bengaluru'
    if 'navi mumbai' in loc.lower()

# Extrack key tools from job_description

# Extract Tittle of job (Senior, Junior, analyst, engineer)

# Extract some key skills
