from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np 
import joblib


# load the data set 
df  = pd.read_csv("NPL_Sentiment_Analysis/data/processed/5000-cleaned-reviews.csv")

# split into X and y 
X = df["cleaned_review"]
y = df["sentiment"].apply(lambda x : 1 if x == "positive" else 0)

# build the pipeline 
pipeline = Pipeline([("TF-idf vectorizer", TfidfVectorizer(max_features=2500)), 
                     ("linear_regression", LogisticRegression(max_iter=1000))])

# fit the pipline
pipeline.fit(X, y)


# save the vectorizer and trained model
joblib.dump(pipeline, "NPL_Sentiment_Analysis/models/sentiment_analysis_pipeline.joblib")

# test the model by making a prediction
test_text = ["This was an amazing experience, I loved every second!"]
prediction = pipeline.predict(test_text)

# print the model prediction
print(f"Prediction {'positive' if prediction[0] == 1 else 'negative'}")




