# imports
from shutil import rmtree
from tempfile import mkdtemp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import  classification_report, confusion_matrix
import joblib


# load the dataset 
df = pd.read_csv("NPL_Sentiment_Analysis/data/processed/50000-cleaned-reviews.csv")

# split into X and y 
X = df["cleaned_review"].values.astype("U")
y = df["sentiment"].apply(lambda x : 1 if x == "positive" else 0)

# split into train and test sets
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=.2 , random_state=43)

# caching to reduce memory foorprint 
cache_dir = mkdtemp()

# build the pipeline with memory cache to reduce memory footprint
model_pipeline = Pipeline([ 
    ("TF-IDF vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("linear_regressor", LogisticRegression(max_iter=1000)),
    ], memory=cache_dir)

# training the vectorizer and the model
model_pipeline.fit(X_train, y_train)
print("model training on the 50k reviews completed")

# make predictions with the model
model_preds = model_pipeline.predict(X_test)

# evalutate the model
# accuracy
model_acc = model_pipeline.score(X_test, y_test)
print(f"Final model accuracy is of : {model_acc:2f}")

# classification report
cls_report  = classification_report(y_test, model_preds)
print("Classification report \n", cls_report)

# confusion matrix
conf_mat = confusion_matrix(y_test, model_preds)
print("model's confuison matrix: \n", conf_mat)

# save the finale model
joblib.dump(model_pipeline,"NPL_Sentiment_Analysis/models/final_pipeline_v1.joblib")
print("finale pipeline sucessfully saved ...")

# remove the cached dir 
rmtree(cache_dir)
