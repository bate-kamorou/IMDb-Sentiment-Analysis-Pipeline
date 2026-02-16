import streamlit as st

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from  text_processor import TextCleaner
import joblib

# page configuration 
st.set_page_config(page_title="IMDB AI Sentiment Analysis", page_icon="🎦")


st.title(" 🎬 AI Movies Review Sentiment Analysis", text_alignment="center")


# simulate an actual movie review
list_of_movies = ["Avatar", "Fast and Furious", "Avengers", "Black widow", "Aquaman"]

selected_movie = st.radio("Choose a movie to write a review for :", options=list_of_movies)

input_text  = st.text_area(label="## write you impression on the movie here", placeholder="The cinematography was brilliant, but the plot was lacking...")

# instantiate the TextCleaner class
test_cleaner = TextCleaner()

# load the pipeline function
@st.cache_resource
def load_pipeline(model_path):
    return joblib.load(model_path)

# load the pipeline

model_path = Path(__file__).parent.parent / "models" / "final_pipeline_v1.joblib"
pipeline = load_pipeline(model_path)




# add a button 
if st.button(label="Analyze review sentiment", type="primary"):

    if input_text:

        # clean the text
        cleaned_text = test_cleaner.clean_text(input_text)

        # make prediction with the model
        prediction = pipeline.predict([cleaned_text])
        predict_proba = pipeline.predict_proba([cleaned_text])

        st.divider()

        if prediction  == 1 :
            st.success(f"###  This user's Review Result for {selected_movie}  is POSITIVE.  With a score of {predict_proba[0][1]:.2f}%")
            st.balloons()
        else :
            st.error(f"### This user's Review Result for {selected_movie} is NEGATIVE.  With a score of {predict_proba[0][0]:.2f}%")
        
        st.subheader("💡key influencers")
        st.write("model focused on these words:")
        st.info(f"_{cleaned_text}_")
    else  :
        st.warning("Please enter a review first")
