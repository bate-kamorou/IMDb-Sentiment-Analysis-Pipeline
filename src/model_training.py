from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# instantiate the model
model = MultinomialNB()

# load the training  data
X_train = pd.read_csv("NPL_Sentiment_Analysis/data/processed/X_train_5000.csv")
y_train = pd.read_csv("NPL_Sentiment_Analysis/data/processed/y_train_5000.csv")
y_train = np.ravel(y_train)

# load the test data
X_test = pd.read_csv("NPL_Sentiment_Analysis/data/processed/X_test_5000.csv")
y_test = pd.read_csv("NPL_Sentiment_Analysis/data/processed/y_test_5000.csv")
y_test = np.ravel(y_test)


# train the model
model.fit(X_train, y_train)

# make prediction 
y_preds = model.predict(X_test)

# model accuracy
acc_score = accuracy_score(y_test, y_preds)
print(f"Model accuracy score is : {acc_score:2f} ")

# classification report '
cls_report  = classification_report(y_test, y_preds)
print("Model classification report is :", cls_report )

# confusion matrix
conf_mat = confusion_matrix(y_test, y_preds)
print("Model confusion matrix is : ", conf_mat)

# save the base model
joblib.dump(model, "NPL_Sentiment_Analysis/models/NB_senti_analysis_base_model.joblib")
print("model saved ...")

print(model.feature_names_in_)



