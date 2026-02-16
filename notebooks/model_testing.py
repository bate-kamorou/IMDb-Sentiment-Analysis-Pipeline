import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np


from text_processor import TextCleaner # type: ignore
import joblib
import pandas as pd

# load the vectorizer
loaded_vectorizer = joblib.load("NPL_Sentiment_Analysis/models/tfidf_5000_vectorizer.joblib")

# load the model
loaded_model = joblib.load("NPL_Sentiment_Analysis/models/NB_senti_analysis_base_model.joblib")

def manual_testing(text):
    # process the text
    textcleaner = TextCleaner()
    cleaned_text = textcleaner.clean_text(text)
    

    # vectorize the data
    vectorized_text = loaded_vectorizer.transform([cleaned_text])
    vectorized_text = np.reshape(vectorized_text, shape=(1 , -1))

    # convert to a dense data set, then a dataframe with columns names sets
    # features_names = loaded_model.feature_names_in_
    # features_names = range(0, 2500)
    # vectorized_df = pd.DataFrame(vectorized_text.toarray(), columns=features_names)


    # use the model to make a prediction
    prediction = loaded_model.predict([vectorized_text])

    return  "positive" if prediction == 1 else "negative"

    



# Try these:
test_1 = "This movie was an absolute masterpiece! The acting was 10/10."
test_2 = "I wasted two hours of my life. The plot made no sense and the ending was terrible."
sarcastic_text  = "Oh great, another sequel that nobody asked for."

# run the test 
print("text 1", manual_testing(test_1))
print("text 2 ", manual_testing(test_2))
print("sarcastic text", manual_testing(sarcastic_text))