# Import yahoo finance
import yfinance as yf
# Import pandas to use dataframes
import pandas as pd
# Import random to set seed
import random
import numpy as np

# Get the tickers from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_df = pd.read_html(url, header=0)[0]
sp500_tickers = sp500_df["Symbol"].tolist()
sp500_tickers = [t.replace('.', '-') if '.' in t else t for t in sp500_tickers]

# Initialize empty lists to fill with tickers data
companies = []
esg_score = []
esg_cat = []
esg_cat = []
ebitda =[]
roe = []
marketcap = []
industry = []
sector = []
companies = []

# Cycle to fill the lists with ESG and financial data
for i in sp500_tickers:
    ticker = yf.Ticker(i)
    try:
       if ticker.sustainability is not None and ticker.info is not None:
           esg_data = ticker.sustainability
           financial_data = ticker.info
           esg_score.append(esg_data.loc["totalEsg"][0])
           esg_cat.append(esg_data.loc["esgPerformance"][0])
           ebitda.append(financial_data.get("ebitda", None))
           roe.append(financial_data.get("returnOnEquity", None))
           marketcap.append(financial_data.get("marketCap", None))
           industry.append(financial_data.get("industry"))
           sector.append(financial_data.get("sector", None))
           companies.append(financial_data.get("symbol", None))
    except:
        continue

# Convert the lists into a dataframe
esg_df = pd.DataFrame({"company": companies,
                       "esg_score": esg_score,
                       "esg_cat": esg_cat,
                       "ebitda": ebitda,
                       "roe": roe,
                       "marketcap": marketcap,
                       "industry": industry,
                       "sector": sector})
# Delete mising values
esg_df.dropna(inplace = True)

"""# **Data Preprocessing**"""

# Transform categorical columns from text to category
esg_df["esg_cat"] = esg_df["esg_cat"].astype("category")
esg_df["industry"] = esg_df["industry"].astype("category")
esg_df["sector"] = esg_df["sector"].astype("category")

# Get dummies for the independent categorical variables
esg_df_encoded = pd.get_dummies(esg_df,
                                columns=["industry", "sector"],
                                drop_first=False)

# Get X and y variables for regression and classification(2)
X = esg_df_encoded.iloc[: , 3 : ]
y = esg_df_encoded["esg_score"]
X2 = esg_df_encoded.iloc[: , 3 : ]
y2 = esg_df_encoded["esg_cat"]

# Set seed
random.seed(123)

# Import and use the train test split for regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state = 123,
                                                    test_size = 0.35)

# Do the split for classification
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,
                                                    y2,
                                                    random_state = 123,
                                                    test_size = 0.35)

# Create empty dataframes to store accuracy

reg_accuracy = pd.DataFrame(columns = ["model", "accuracy"])
class_accuracy = pd.DataFrame(columns = ["model", "accuracy"])

"""# **Random Forest**

Random Forest Regression Training
"""

#Import Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Get the best parameters for this model
params_rfr = {"n_estimators": range(100, 1000, 50),
              "max_features": range(10, 100, 5),
                "max_depth": range(1, 10)}

grid_rfr = RandomizedSearchCV(estimator = RandomForestRegressor(),
                          param_distributions = params_rfr,
                          cv = 5,
                          n_iter = 10,
                          random_state = 123,
                          scoring = "neg_mean_absolute_percentage_error",
                          n_jobs = -1)

grid_rfr.fit(X_train, y_train)

# Save the best parameters
best_rfr_score = -grid_rfr.best_score_
best_rfr_estimators = grid_rfr.best_estimator_.n_estimators
best_rfr_features = grid_rfr.best_estimator_.max_features
best_rfr_depth = grid_rfr.best_estimator_.max_depth

"""Random Forest Regression Test"""

# Run the best model
rfr_model = RandomForestRegressor(n_estimators = best_rfr_estimators,
                                  oob_score = True,
                                  max_features = best_rfr_features,
                                  random_state = 123,
                                  max_depth = best_rfr_depth)

# Fit the model with the training data
rfr_model.fit(X_train, y_train)

# Store predictions on the test set
y_pred_rfr = rfr_model.predict(X_test)

#Import and compute MAPE
from sklearn.metrics import mean_absolute_percentage_error
rfr_accuracy = mean_absolute_percentage_error(y_test, y_pred_rfr)

# Add the results to the df
rfr_row = {"model": "Random Forest", "accuracy": rfr_accuracy}
reg_accuracy = pd.concat([reg_accuracy, pd.DataFrame([rfr_row])], 
                         ignore_index=True)

"""Random Forest Regression Feature Importance"""

# Get feature importances and column names to prepare the df
rfr_features = rfr_model.feature_importances_
feature_names = X_train.columns

rfr_features_df = pd.DataFrame({"feature": feature_names,
                                "importance": rfr_features})

# Order and filter for the top 10
top_rfr_features_df = rfr_features_df.sort_values("importance", 
                                                  ascending = False).head(10)

# Import plot libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Create the barplot
sns.barplot(data = top_rfr_features_df, x = "importance", y = "feature")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Regression Feature Importance")
plt.show()

"""Random Forest Regression Residuals Histogram"""

# Obtain residuals
rfr_resid = y_test - y_pred_rfr

# Plot residuals
sns.histplot(rfr_resid, bins = 30, kde = True)
plt.axvline(0, color = "red", linestyle = "--")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Random Forest Regression Residuals Histogram")
plt.show()


"""Random Forest Classification Training"""

#Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Get the best parameters for this model
params_rfc = {"n_estimators": range(100, 1000, 50),
          "max_features": range(10, 100, 5),
          "max_depth": range(1, 10)}

strat_cv_rfc = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

grid_rfc = RandomizedSearchCV(estimator = RandomForestClassifier(),
                          param_distributions = params_rfc,
                          cv = strat_cv_rfc,
                          n_iter = 10,
                          random_state = 123,
                          scoring = "accuracy",
                          n_jobs = -1)

grid_rfc.fit(X2_train, y2_train)

# Save the best parameters
best_rfc_score = grid_rfc.best_score_
best_rfc_estimators = grid_rfc.best_estimator_.n_estimators
best_rfc_features = grid_rfc.best_estimator_.max_features
best_rfc_depth = grid_rfc.best_estimator_.max_depth

"""Random Forest Classificacion Test"""

# Run model with best parameters
rfc_model = RandomForestClassifier(n_estimators = best_rfc_estimators,
                                   oob_score = True,
                                   max_features = best_rfc_features,
                                   max_depth = best_rfc_depth,
                                   random_state = 123)

# Fit the model with the training data
rfc_model.fit(X2_train, y2_train)

# Store predictions on the test set
y_pred_rfc = rfc_model.predict(X2_test)

# Import and compute accuracy
from sklearn.metrics import accuracy_score
rfc_accuracy = accuracy_score(y2_test, y_pred_rfc)

# Save the results in the df
rfc_row = {"model": "Random Forest",
           "accuracy": rfc_accuracy}

class_accuracy = pd.concat([class_accuracy, pd.DataFrame([rfc_row])], 
                           ignore_index = True)

"""Random Forest Classification Feature Importance"""

# Get feature importances and create the df
rfc_features = rfc_model.feature_importances_

rfc_features_df = pd.DataFrame({"feature": feature_names,
                                "importance": rfc_features})

# Sort and filter for top 10 features
top_rfc_features_df = rfc_features_df.sort_values("importance", 
                                                  ascending = False).head(10)

# Create the bar plot
sns.barplot(data = top_rfc_features_df, x = "importance", y = "feature")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Classification Feature Importance")
plt.show()

"""Random Forest Classification Confusion Matrix"""

# Import confusion matrix
from sklearn.metrics import confusion_matrix

# Get the normalized confusion matrix
rfc_cm = confusion_matrix(y2_test, y_pred_rfc)
rfc_cm_normalized = rfc_cm.astype('float') / rfc_cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))

# Plot as a heatmap
sns.heatmap(rfc_cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=np.unique(y_pred_rfc), yticklabels=np.unique(y2_test))
plt.title("Random Forest Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

"""# **XGBOOST**

XGBoost Regression Training
"""

# Import model
import xgboost as xgb


# Get the best parameters for this model
params_xgr = {"n_estimators": range(100, 1000, 50),
              "max_depth": range(1, 10),
              "learning_rate": np.arange(0.01, 0.1, 0.01)}

grid_xgr = RandomizedSearchCV(estimator = xgb.XGBRegressor(),
                              param_distributions = params_xgr,
                              cv = 5,
                              n_iter = 10,
                              random_state = 123,
                              scoring = "neg_mean_absolute_percentage_error",
                              n_jobs = -1)

grid_xgr.fit(X_train, y_train)

# Save the best parameters
best_xgr_score = -grid_xgr.best_score_
best_xgr_estimators = grid_xgr.best_estimator_.n_estimators
best_xgr_depth = grid_xgr.best_estimator_.max_depth
best_xgr_lr = grid_xgr.best_estimator_.learning_rate

"""XGBoost Regression Test"""

# Run the model with the best parameters
xgr_model = xgb.XGBRegressor(n_estimators = best_xgr_estimators,
                             max_depth = best_xgr_depth,
                             learning_rate = best_xgr_lr,
                             random_state = 123)

xgr_model.fit(X_train, y_train)

y_pred_xgr = xgr_model.predict(X_test)

xgr_accuracy = mean_absolute_percentage_error(y_test, y_pred_xgr)

# Save the results in the df
xgr_row = {"model": "XGBoost", "accuracy": xgr_accuracy}
reg_accuracy = pd.concat([reg_accuracy, pd.DataFrame([xgr_row])], 
                         ignore_index=True)

"""XGBoost Regression Feature Importance"""

# Get importances to create the df
xgr_features = xgr_model.feature_importances_

xgr_features_df = pd.DataFrame({"feature": feature_names,
                                "importance": xgr_features})

# Sort and filter to get the top 10
top_xgr_features_df = xgr_features_df.sort_values("importance", 
                                                  ascending = False).head(10)

# Plot as a barchart
sns.barplot(data = top_xgr_features_df, x = "importance", y = "feature")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("XGBoost Regression Feature Importance")
plt.show()

"""XGBoost Regression Residuals Histogram"""

# Get the residuals
xgr_resid = y_test - y_pred_xgr

# Plot the histogram
sns.histplot(xgr_resid, bins = 30, kde = True)
plt.axvline(0, color = "red", linestyle = "--")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("XGBoost Regression Residuals Histogram")
plt.show()

"""XGBoost Classification Training"""

# Import Label Encoder to treat the categorical output
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y2_train_encoded = le.fit_transform(y2_train)

# Get the best parameters for this model
params_xgc = {"n_estimators": range(100, 1000, 50),
              "max_depth": range(1, 10),
              "learning_rate": np.arange(0.01, 0.1, 0.01)}

strat_cv_xgc = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

grid_xgc = RandomizedSearchCV(estimator = xgb.XGBClassifier(),
                              param_distributions = params_xgr,
                              cv = strat_cv_xgc,
                              n_iter = 10,
                              random_state = 123,
                              scoring = "accuracy",
                              n_jobs = -1)

grid_xgc.fit(X2_train, y2_train_encoded)

# Save the best parameters
best_xgc_score = grid_xgc.best_score_
best_xgc_estimators = grid_xgc.best_estimator_.n_estimators
best_xgc_depth = grid_xgc.best_estimator_.max_depth
best_xgc_lr = grid_xgc.best_estimator_.learning_rate

"""XGBoost Classification Test"""

# Run the model with the best parameters
xgc_model = xgb.XGBClassifier(n_estimators = best_xgc_estimators,
                             max_depth = best_xgc_depth,
                             learning_rate = best_xgc_lr,
                             random_state = 123)

xgc_model.fit(X2_train, y2_train_encoded)

y2_test_encoded = le.transform(y2_test)

y_pred_xgc = xgc_model.predict(X2_test)

xgc_accuracy = accuracy_score(y2_test_encoded, y_pred_xgc)

# Save the results in the df
xgc_row = {"model": "XGBoost", "accuracy": xgc_accuracy}
class_accuracy = pd.concat([class_accuracy, pd.DataFrame([xgc_row])], 
                           ignore_index=True)

"""XGBoost Classification Feature Importance"""

# Get the importance and create the df
xgc_features = xgc_model.feature_importances_

xgc_features_df = pd.DataFrame({"feature": feature_names,
                                "importance": xgc_features})

# Sort and filter to het the top 10 features
top_xgc_features_df = xgc_features_df.sort_values("importance", 
                                                  ascending = False).head(10)

# Display as a barplot
sns.barplot(data = top_xgc_features_df, x = "importance", y = "feature")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("XGBoost Classification Feature Importance")
plt.show()

"""XGBoost Classification Confusion Matrix"""

# Get the confusion matrix with proportions
xgc_cm = confusion_matrix(y2_test_encoded, y_pred_xgc)
xgc_cm_normalized = xgc_cm.astype('float') / xgc_cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))

# Plot as a heatmap
sns.heatmap(xgc_cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=np.unique(y2_test), yticklabels=np.unique(y2_test))
plt.title("XGBoost Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

"""# **Support Vector Machine**

SVM Regression Training
"""

# Import the regressor and the scaler to normalize the data
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler_svr = StandardScaler().fit(X_train)
X_train_scaled = scaler_svr.transform(X_train)

# Get the best parameters for this model
params_svr = {"kernel": ["rbf", "poly"],
              "C": [0.1, 1, 10],
              "gamma": ["scale", "auto"]
}

grid_svr = RandomizedSearchCV(estimator = SVR(),
                          param_distributions = params_svr,
                          cv = 5,
                          n_iter = 10,
                          random_state = 123,
                          scoring = "neg_mean_absolute_percentage_error",
                          n_jobs = -1)

grid_svr.fit(X_train_scaled, y_train)

# Save the output
best_svr_score = -grid_svr.best_score_
best_svr_kernel = grid_svr.best_estimator_.kernel
best_svr_C = grid_svr.best_estimator_.C
best_svr_gamma = grid_svr.best_estimator_.gamma

"""SVM Regression Test"""

# Fit the best model for regression
svr_model = SVR(kernel = best_svr_kernel,
                C = best_svr_C,
                gamma = best_svr_gamma)

svr_model.fit(X_train_scaled, y_train)

# Scale the test data
X_test_scaled = scaler_svr.transform(X_test)

y_pred_svr = svr_model.predict(X_test_scaled)

# Get accuracy and add it to the df
svr_accuracy = mean_absolute_percentage_error(y_test, y_pred_svr)

# Save the results in the df
svr_row = {"model": "Support Vector Machine", "accuracy": svr_accuracy}
reg_accuracy = pd.concat([reg_accuracy, pd.DataFrame([svr_row])], 
                         ignore_index=True)

"""SVM Regression Feature Importance"""

# Import function to get importance and apply it
from sklearn.inspection import permutation_importance
svr_result = permutation_importance(svr_model, 
                                    X_test_scaled, 
                                    y_test, n_repeats=10, 
                                    random_state=42)
svr_features = svr_result.importances_mean

svr_features_df = pd.DataFrame({"feature": feature_names,
                                "importance": svr_features})

# Sort and filter the df to get top 10 features
top_svr_features_df = svr_features_df.sort_values("importance", 
                                                  ascending = False).head(10)

# Display as a barplot
sns.barplot(data = top_svr_features_df, x = "importance", y = "feature")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Support Vector Machine Regression Feature Importance")
plt.show()


"""SVM Regression Residuals Histogram"""

# Get residuals
svr_resid = y_test - y_pred_svr

# Plot the residuals
sns.histplot(svr_resid, bins = 30, kde = True)
plt.axvline(0, color = "red", linestyle = "--")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Support Vector Machine Regression Residuals Histogram")
plt.show()

"""SVM Classification Training"""

# Import the model
from sklearn.svm import SVC

# Scale the data
scaler_svc = StandardScaler().fit(X2_train)
X2_train_scaled = scaler_svc.fit_transform(X2_train)

# Get the best parameters for this model

params_svc = {"kernel": ["rbf", "poly"],
              "C": [0.1, 1, 10],
              "gamma": ["scale", "auto"]
}


strat_cv_svc = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

grid_svc = RandomizedSearchCV(estimator = SVC(),
                          param_distributions = params_svc,
                          cv = strat_cv_svc,
                          n_iter = 10,
                          random_state = 123,
                          scoring = "accuracy",
                          n_jobs = -1)

grid_svc.fit(X2_train_scaled, y2_train)

# Save the best parameters
best_svc_score = grid_svc.best_score_
best_svc_kernel = grid_svc.best_estimator_.kernel
best_svc_C = grid_svc.best_estimator_.C
best_svc_gamma = grid_svc.best_estimator_.gamma

"""SVM Classification Test"""

# Run the model with the best parameters
svc_model = SVC(kernel = best_svc_kernel,
                C = best_svc_C,
                gamma = best_svc_gamma,
                random_state = 123)

svc_model.fit(X2_train_scaled, y2_train)

# Scale the test data
X2_test_scaled = scaler_svc.transform(X2_test)

y2_pred_svc = svc_model.predict(X2_test_scaled)

svc_accuracy = accuracy_score(y2_test, y2_pred_svc)

# Save the results in the df
svc_row = {"model": "Support Vector Machine", "accuracy": svc_accuracy}
class_accuracy = pd.concat([class_accuracy, pd.DataFrame([svc_row])], 
                           ignore_index=True)

"""SVM Classification Feature Importance"""

# Get feature importances and create df
svc_result = permutation_importance(svc_model, 
                                    X2_test_scaled, 
                                    y2_test, 
                                    n_repeats=10, 
                                    random_state=42)
svc_features = svc_result.importances_mean

svc_features_df = pd.DataFrame({"feature": feature_names,
                                "importance": svc_features})

# Sort and filter for the top 10
top_svc_features_df = svc_features_df.sort_values("importance", 
                                                  ascending = False).head(10)

# Display as a barplot
sns.barplot(data = top_svc_features_df, x = "importance", y = "feature")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Support Vector Machine Classification Feature Importance")
plt.show()

"""SVM Classification Confusion Matrix"""

# Get the normalized confusion matrix
svc_cm = confusion_matrix(y2_test, y2_pred_svc)
svc_cm_normalized = svc_cm.astype('float') / svc_cm.sum(axis=1)[:, np.newaxis]

# Plot as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(svc_cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=np.unique(y2_test), yticklabels=np.unique(y2_test))
plt.title("Support Vector Machine Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

"""# **Naive Models**

Naive Regression
"""

# Import naive regressor
from sklearn.dummy import DummyRegressor

# Model with the mean of the training set as prediction
dr_model = DummyRegressor(strategy = "mean")

# Fit the model with the training data
dr_model.fit(X_train, y_train)

# Store predictions on the test set
y_pred_dr = dr_model.predict(X_test)

# Get MAPE
dr_accuracy = mean_absolute_percentage_error(y_test, y_pred_dr)

# Add the results to the df
dr_row = {"model": "Naive", "accuracy": dr_accuracy}
reg_accuracy = pd.concat([reg_accuracy, pd.DataFrame([dr_row])], 
                         ignore_index=True)

"""Naive Classification"""

# Import naive classifier
from sklearn.dummy import DummyClassifier

# Model with the most frequent value on the training set as prediction
dc_model = DummyClassifier(strategy = "most_frequent")

# Fit the model to the training set
dc_model.fit(X2_train, y2_train)

# Store predictions on the test set
y_pred_dc = dc_model.predict(X2_test)

# Get accuracy
dc_accuracy = accuracy_score(y2_test, y_pred_dc)

# Add the results to the df
dc_row = {"model": "Naive", "accuracy": dc_accuracy}
class_accuracy = pd.concat([class_accuracy, pd.DataFrame([dc_row])], 
                           ignore_index=True)
