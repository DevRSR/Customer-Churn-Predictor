# Customer-Churn-Predictor
This is a machine learning project that uses Iranian Telecom company dataset from UCI to build a model that not only predict customers that will churn the company but provide a business oversight and solution to the problem.
## Problem Statement
An Iranian telecom company lost an estimated $58,950 (Calculated as the sum of customer value across all churned customers) to customer churn over 12 months, which is approximately 4.2% of annual revenue. The retention team has a $15,000 budget to intervene with at-risk customers. Assuming the cost per retention offer is 20% ($94) of avg. customer value ($470), this budget supports flagging roughly 159 customers. Given this constraint, we cap the false-positive rate at 5% (≈143 customers) which fits within budget while preserving capacity to catch true churners, hence the goal of this project is to build a that maximaze recall of high-value at-risk customers within this budget ceiling.

Note: Currency is not provided in the dataset
## Data
[Iranian Telcom Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
<!-- The Iranian Telecom dataset was obtained from UCI irvine machine learning repository, collected over period of 12-months. This data contain 3150 rows and 14 columns.  -->

## Dataset Columns details:
- Anonymous Customer ID
- Call Failures: number of call failures
- Complains: binary (0: No complaint, 1: complaint)
- Subscription Length: total months of subscription
- Charge Amount: Ordinal attribute (0: lowest amount, 9: highest amount)
- Seconds of Use: total seconds of calls
- Frequency of use: total number of calls
- Frequency of SMS: total number of text messages
- Distinct Called Numbers: total number of distinct phone calls
- Age Group: ordinal attribute (1: younger age, 5: older age)
- Tariff Plan: binary (1: Pay as you go, 2: contractual)
- Status: binary (1: active, 2: non-active)
- Churn: binary (1: churn, 0: non-churn) - Class label
- Customer Value: The calculated value of customer

## Approach (EDA → Features → Models → Evaluation)
Libraries used for this project are: 
- Pandas
- matplotlib
- seaborn
- sklearn
- shap
- imblearn
- numpy

## EDA
Before the EDA an data integrity check was done on the dataset which led to the discovery of 300 duplicates and none null values. After the removal of the duplicates the rows of the dataset reduce to 2850 rows from 3150 rows.
### 1. Churn Distribution
> ![Churn distribution](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Customer%20churn%20class%20distribution.png)
> 15.6% of customers churn the telecom company

### 2. Call Failure distribution by churn status
> ![Call Failure distribution by churn status](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Call%20Failures%20by%20Churn%20Status.png)
recording a call failure of more than 20 can contribute to customer churn

### 3. Complaints distribution by churn status
> ![Complaints distribution by churn status](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Customer%20Churn%20by%20Number%20of%20Complaints.png)

> About 40% of customers that churn make complains

### 4. Call Failure with complaints
> ![Distribution of call failure by complains](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Call%20Failures%20by%20Complaining%20Status.png)
> This shows that complaints are mostly about call failure.
### 5. Subcription Length distribution by churn status
> ![Subscription Length by churn status](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Subscription%20Length%20by%20Churn%20class.png)
> The more the length of subscription of a customer the less likely they churn.

### 6. Distribution of charge amount based on churn status
> ![Distribution of charge Amount by churn](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Charge%20Amount%20by%20Churn%20Status.png)
> This shows that charge amount has negligible effect on customer churn

### 7. Distribution of customer churn class by Distinct Called Numbers
> ![Distribution of churn class by distincr called number](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Distinct%20Called%20Numbers%20by%20Churn%20Status.png)
> This shows that the more unique number a customer call the less they are likely to churn

### 8. Distribution of customer churn class by Frequency of use 
> ! [Distribution of churn class by FOU](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Frequency%20of%20use%20by%20Churn%20Status.png)
> The more customers use the telcom company services the less likely they leave.

### 9. Distribution of customer status based on churn class
> ![Distribution of churn class by customer status](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Churn%20by%20Status.png)

> The more customer are active on a line the less likely the churn.

### 10 Distribution of customer value
> ![Distribution of churn by customer value](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Distribution%20of%20Churn%20by%20Customer%20Value.png)

> Majority of customer having less than or equal to $500 are more likely to churn.

## Features 
No additional feature was addes to the dataset as the existing features satisfy the requirement of the project.

## Models
### Logistic Regression
Logistic regression using class weight to handle class imbalance produce the following metrics
> ![Confusion matrics of the model](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/log_reg_model_without_smote.png)
- Accuracy : 90% 
- precision: (churn:78%, non-churn:90%)
- recall: (churn: 48%, non-churn: 98%)
Remark: Even though accuracy look impressive and false positive rate (12/(12+468) = 2.5%) falls below our offer budget. The model have a very low recall, missing out on half of customers that actually churn, which will cost the business of its most valuable customers.

Logistic regression using SMOTE to handle class imbalance produce the following metrics
> ![Confusion matrics of log with smote model](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/log_reg_model_with_smote.png)
- Accuracy : 81% 
- precision: (churn:45%, non-churn:97%)
- recall: (churn: 84%, non-churn: 81%)
Remark: SMOTE did increase the recall of churn customers from 48% to 84%, which means we rarely missed out on churn customers. however, this come at the detriment of FPR which increase from 2.5% to 19%, which means that we will exhaust all our retention offer on customers that won't churn but wrongly flagged.

### Random Forest Classifier
Random Forest Classifier without resampling of class produce the following metrics
> ![Confusion matrics of rfc without smote model](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Rfc_model_without_smote.png)
- Accuracy: 96%
- precision: (churn: 94% , non-churn: 97%)
- recall: (churn: 83%, non-churn: 99%)
Remark: This is a sophisticated model with an accuracy of 96% and FPR (5/ (5+476) ~ 1%) which means that only (5 x $94 = $470) out of $15000 will be spent on wrongly accused customers. Also the recall of customer that churn is 83% which means that out of ten customers that churn only 2 excaped being flagged by the model.

Random Forest Classifier using SMOTE to handle class imbalance produce the following metrics:
> ![Confusion matrics of rfc with smote model](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Rfc_model_with_smote.png)
- Accuracy: 96%
- precision: (churn: 82% , non-churn: 99%)
- recall: (churn: 94%, non-churn: 96%)
Remark: This is a sophisticated model with an accuracy of 96% same as the one without resampling. However, it make an improvement to the rate of recall from 83% to 94% which means out of ten customers that churn the model rarely miss one customer but the FPR increase from 1% to (18 / (463+18) ~ 3.7%) hence, (18 x $94 = $1692) will be spent on wrongly accused customers.

### XGBoost Classifier
XGBoost classifier with class imbalance produce the following metrics:
> ![Confusion matrics of xgb without smote model](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Xgb_model_without_smote.png)
- Accuracy: 96%
- precision: (churn: 89% , non-churn: 98%)
- recall: (churn: 87%, non-churn: 98%)
Remark: The accuracy of this model is 96% same as majority of the models, with FPR of (10 / (10+471) ~ 2.1%) taking only (10 x $94 = $940) of our retention budget. The precision of the model for churn is 89% and recall is 87%.

XGBoost classifier with class imbalance produce the following metrics:
> ![Confusion matrics of xgb with smote model](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/New%20xgb%20confusion%20matrix.png)
- Accuracy: 96%
- precision: (churn: 81% , non-churn: 99%)
- recall: (churn: 93%, non-churn: 96%)
Remark: This is one of the most sophisticated model with accuracy of 96% and FPR of (16 / (16 + 465) ~ 4%) which means a scenario of max FPR will almost stretch our retention budget to its limit (in context: (3.3% x total customer(2850) ~ 95) x $94 ~ $8,900) but still give room for actual customer that churn. And its recall is impressive at 93% rate and precision at 84% rate.

### Best Model
The best model for this project is XGBoost classifier(with SMOTE) that maximize recall to 93% which make it rarely miss out customer that might churn even though its FPR is high at 3.3%, the model satisfy the need of the company in keeping its valuable customers. 

## Key Findings
> ![Feature Importance of the best model](https://github.com/DevRSR/Customer-Churn-Predictor/blob/main/outputs/Xgb_feature_importance.png)
### Finding 1: Poor Complaints Feedback make customers churn
About 45% of customers that churned the company had complaint filed to the company about their service quality.

### Finding 2: Customer with contractual tariff rarely churn
Over 90% of the customers that churn have a tariff plan of pay-as-you-go as those on contractual tariff rarely churn the company. Suggesting that service provide by the company maybe poor but they can't leave because of their long term subscription.

### Finding 3: Customers Value matters
Almost all customers (About 98%) that leave the company have value less than $500 which account of more than one-fifth of all customers having less than $500 in the dataset, costing the company about $54,000 which is over 90% of total loss revenue to customer churn. Suggesting that churn are more common on customers segment that make up of about 70% of the company customer base.

### Finding 4: Inactivity contribute to churn rate
Close to 50% of customers that are inactive churn the company accounting for about 70% of all customer that churn the company cost the company more than 50% ($35,235.00) of total revenue loss.

### Finding 5: Call failure contribute to churn rate
Majority of those that complained have issue with call failure and Eight in ten customers that logged a complaint to the company eventually leave the company, which cost the company about $32,400 in revenue.
## Business Recommendation
- **Tariff incentives**: Pay-As-You-Go tariff plan is what the majority of telecom customers uses for their service. incentivizing subscription will not only retain customers but also increase the revenue of the telecom company by 3% per annum.
- **Improvement of services**: Improving services and building infrastructure that will call quality will reduce customer complains by over 55% thereby increasing the company revenue by over $30000 which are lost through customer churn.
- **Improvement of technical support**: Having Good technical support that act promptly and efficient to customers needs will not only retain the customers but also make the company to be at tip of the lips of their customer which will increase their customer base and subcequently increase their revenue.
- **Welcome back bonus**: Data has shown that inactive customers rarely come back to use the service rather they churn the telecom company leading to a loss of $35000 in revenue. Introducing a welcome back bonus will not only reduce revenue loss but also increase the revenue due to increase in service used.
