from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# load the dataset
df = pd.read_csv("NPL_Sentiment_Analysis/data/processed/50000-cleaned-reviews.csv")

# instantiate the vectorizer
vectorizer = TfidfVectorizer(max_features=2500)

# vectorize the cleaned reviews 

X =  vectorizer.fit_transform(df["cleaned_review"]).toarray() # type: ignore 

# one hot encode the sentiment column
y = df["sentiment"].apply(lambda x : 1 if x == "positive" else 0 ).to_numpy()

print(f"Feature matrix X shape is {X.shape} ")
print(f"First 10 numbers in the first review is {X[0][:10]} ")

# save the vectorizer to models
joblib.dump(vectorizer, "NPL_Sentiment_Analysis/models/tfidf_50_000_vectorizer.joblib")

# split in to training and test sets
X_train_50_000, X_test_50_000, y_train_50_000, y_test_50_000 = train_test_split(X, y, test_size=0.2, random_state=43)


# save the 5000  training and test set for later use 
pd.DataFrame(X_train_50_000).to_csv("NPL_Sentiment_Analysis/data/processed/X_train_50_000.csv", index=False)
pd.DataFrame(y_train_50_000).to_csv("NPL_Sentiment_Analysis/data/processed/y_train_50_000.csv", index=False)
pd.DataFrame(X_test_50_000).to_csv("NPL_Sentiment_Analysis/data/processed/X_test_50_000.csv", index=False)
pd.DataFrame(y_test_50_000).to_csv("NPL_Sentiment_Analysis/data/processed/y_test_50_000.csv", index=False)
