from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# load training and testing data
X_train = pd.read_csv("NPL_Sentiment_Analysis/data/processed/X_train_50_000.csv")
X_test = pd.read_csv("NPL_Sentiment_Analysis/data/processed/X_test_50_000.csv")
y_train = pd.read_csv("NPL_Sentiment_Analysis/data/processed/y_train_50_000.csv")
y_test = pd.read_csv("NPL_Sentiment_Analysis/data/processed/y_test_50_000.csv")

# flatten the labels y 
y_test =  np.ravel(y_test)
y_train=  np.ravel(y_train)


# instantiate the model
linear_model = LogisticRegression(max_iter=1000)

# fit the model
linear_model.fit(X_train, y_train)

# make predictions
model_preds = linear_model.predict(X_test)

# model accuracy
model_acc = accuracy_score(y_test, model_preds)
print("linear model accuracy is: ", model_acc)

# classification report 
cls_report = classification_report(y_test, model_preds)
print("Linear model classification report is: \n", cls_report )

# confustion matrix
conf_mat  =  confusion_matrix(y_test, model_preds)
print("linear model's confusion matrix \n" , conf_mat)


if model_acc > .84 :
    # save the linear model
    joblib.dump(linear_model,"NPL_Sentiment_Analysis/models/linear_model.joblib")
    print("new champion saved")
else :
    print("the naive bayes model is best ")

# plot the model's confusion points

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, cmap="Blues", fmt="d", annot=True, 
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"]
            )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix of the linear model")
plt.show()