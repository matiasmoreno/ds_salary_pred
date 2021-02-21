# Data Scientist salary prediction

# Data Science Salary Estimator: Project Overview 
* Created a tool that estimates data science salaries (MAE ~ $ 5K) to help data scientists negotiate their income when they get a job.
* Engineered features from the text of each job description to quantify the value companies put on python, excel, SQL, hadoop and spark. 
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model. 
* Built a client facing API using flask 

## Code and Resources Used 
**Python Version:** 3.9 
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn,, flask, json, pickle  
**For Web Framework Requirements:** anaconda virtual env .yaml included
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## DataSet
 * Data Scientis Salary from India, extracted from [Kaggle](https://www.kaggle.com/jaiganeshnagidi/data-scientist-salary/)

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made a new column for company city 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights tables. 

![alt text](https://github.com/matiasmoreno/ds_salary_pred/edit/master/images/earn_by_tittle.PNG "Title Histogram")
![alt text](https://github.com/matiasmoreno/ds_salary_pred/edit/master/images/skill_words.png "Common Skll words")
![alt text](https://github.com/matiasmoreno/ds_salary_pred/edit/master/images/correlations.png "Correlations")
![alt text](https://github.com/matiasmoreno/ds_salary_pred/edit/master/images/earn_by_param.png "Earnings by parameters")

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 4.83
*	**Linear Regression**: MAE = 5.15
*	**Lasso Regression**: MAE = 5.23

## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 

## Resources
 * Data Science Project from Scratch - [Playlist](https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t)



