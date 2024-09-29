Overview
This project analyzes airline passenger satisfaction using machine learning techniques. The goal is to predict passenger satisfaction based on various factors such as in-flight service, seat comfort, cleanliness, and more. This analysis provides valuable insights for airlines to improve customer experience and overall satisfaction.

Table of Contents
Project Motivation
Data
Methodology
Models
Results
Conclusion
Future Work
License
Project Motivation
In an increasingly competitive aviation industry, customer satisfaction is a key differentiator. By understanding the factors that influence passenger satisfaction, airlines can make targeted improvements. This project aims to predict customer satisfaction levels, enabling airlines to better serve their passengers.

Data
The dataset used for this project contains the following features:

Flight distance
Inflight service rating
Seat comfort rating
Food and drink satisfaction
Onboard entertainment
Cleanliness
Baggage handling
Customer Type
Satisfaction (Target Variable)
The data is preprocessed, including handling missing values, encoding categorical variables, and scaling numerical features.


Methodology
Exploratory Data Analysis (EDA): Conducted to understand the distribution of data, identify relationships, and detect any anomalies.
Data Preprocessing: Handling missing values, scaling numerical features, and encoding categorical variables.
Feature Selection: Mutual Information and Chi-square tests were used to select the most relevant features.
Model Training and Evaluation:
Split the data into training and testing sets.
Trained several classification models.
Evaluated model performance using metrics such as accuracy, precision, recall, and F1 score.
Models
The following models were trained:

Logistic Regression
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)
Decision Trees
Ensemble Models: Bagging and Boosting (XGBoost)
Hyperparameter tuning was performed using Grid Search.

Results
The best-performing model was the XGBoost classifier, which achieved:

Accuracy: 96%

Visualization of feature importance helped in interpreting which factors most influence passenger satisfaction.

Conclusion
The project successfully identified key factors affecting airline passenger satisfaction. Airlines can use these insights to improve customer experience and enhance satisfaction.

Future Work
Incorporating more advanced algorithms, such as neural networks.
Extending the project to include time series analysis for forecasting satisfaction trends.
Improving the deployment to include a more interactive user interface.
