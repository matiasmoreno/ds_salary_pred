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

# Length of key skills
df["key_skills_length"] = df["key_skills"].apply(lambda x: 0 if isinstance(x, float) else len(x)) 

# Extract some location and normalize
#df["location"].value_counts()
#df["location"].describe

def location_normalize(loc):
    if 'bengaluru' in loc.lower():
        return 'bengaluru'
    elif 'navi' in loc.lower():
        return 'navi'
    elif 'mumbai' in loc.lower():
        return 'mumbai'
    elif 'navi' in loc.lower():
        return 'navi'
    elif 'urgaon' in loc.lower():
        return 'gurgaon'
    elif 'pune' in loc.lower():
        return 'pune'
    elif 'hyderabad' in loc.lower():
        return 'Hyderabad'
    elif 'delhi' in loc.lower():
        return 'delhi'
    else:
        return 'na'
    
df["location_abbreviation"] = df["location"].apply(lambda x: location_normalize(x))

# Extrack key tools from job_description

# Extract Tittle of job (Senior, Junior, analyst, engineer)
df['jd_aux'] = df['job_desig'].apply(lambda x: '' if isinstance(x, float) else x)
#senior
df['jd_senior'] = df['jd_aux'].apply(lambda x: 1 if 'senior' in x.lower() else 0)
#manager
df['jd_manager'] = df['jd_aux'].apply(lambda x: 1 if 'manager' in x.lower() else 0)
#junior
df['jd_junior'] = df['jd_aux'].apply(lambda x: 1 if 'junior' in x.lower() else 0)
#architect
df['jd_architect'] = df['jd_aux'].apply(lambda x: 1 if 'architect' in x.lower() else 0)
#engineer
df['jd_engineer'] = df['jd_aux'].apply(lambda x: 1 if 'engineer' in x.lower() else 0)
#analyst
df['jd_analyst'] = df['jd_aux'].apply(lambda x: 1 if 'analyst' in x.lower() else 0)
#developer
df['jd_developer'] = df['jd_aux'].apply(lambda x: 1 if 'developer' in x.lower() else 0)

df.drop(['jd_aux'], inplace=True, axis = 1 )


# Extract some key skills
df['skill_aux'] = df['key_skills'].apply(lambda x: '' if isinstance(x, float) else x)
#python
df['sk_python'] = df['skill_aux'].apply(lambda x: 1 if 'python' in x.lower() else 0)
# sas
df['sk_r'] = df['skill_aux'].apply(lambda x: 1 if 'sas' in x.lower() else 0)
# Time Series
df['sk_time_series'] = df['skill_aux'].apply(lambda x: 1 if 'time series' in x.lower() else 0)
# Regression
df['sk_regression'] = df['skill_aux'].apply(lambda x: 1 if 'regression' in x.lower() else 0)
# Neural Networks
df['sk_NN'] = df['skill_aux'].apply(lambda x: 1 if 'neural networks' in x.lower() else 0)
# Machine learning
df['sk_ML'] = df['skill_aux'].apply(lambda x: 1 if 'machine learning' in x.lower() else 0)
# spark
df['sk_spark'] = df['skill_aux'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
# hadoop
df['sk_hadoop'] = df['skill_aux'].apply(lambda x: 1 if 'hadoop' in x.lower() else 0)
# SQL
df['sk_sql'] = df['skill_aux'].apply(lambda x: 1 if 'sql' in x.lower() else 0)

df.drop(['skill_aux'], inplace=True, axis = 1 )

df.to_csv('salary_data_cleaned.csv', index = False)

pd.read_csv('salary_data_cleaned.csv')

print('Data Cleaning Finished')
