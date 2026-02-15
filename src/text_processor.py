import pandas as pd 
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as wnl
from bs4 import BeautifulSoup

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Dowmload necessary NLTK tools
# nltk.download("stopwords")
# nltk.download("wordnet")

class TextCleaner():
    """
    Clean a given text by removing stop words and converting the word to it's root

    **Methods**:
        clean_text(): parse and clean the text
    """

    def __init__(self):
        self.stopwords = stopwords.words("english")
        self.lemmetizier = wnl()

    def clean_text(self, input_text:str) :
        """Parser and clean a given text by removing tags, pounctuation and none text characters
           convert the words using lemmetization

           **Args**:
                input_text: text to be parsed and convert
        """
        # parse and remove html tags from the text
        texts = BeautifulSoup(input_text, "html.parser").get_text()

        # turn all the text to lowwercase
        texts = texts.lower()

        # remove punctuation marks  non text characters and split the text
        texts = re.sub(r"[^a-zA-Z]", " ", texts)
        
        # split the text into words 
        texts = texts.split(" ")

        # filter and remove stop words 
        cleaned_words = [word for word in texts if word not in self.stopwords]

        # lemmetization convert words to their roots
        lemmetized_words = [self.lemmetizier.lemmatize(word) for word in cleaned_words]

        return " ".join(lemmetized_words)


# preprocessing
if __name__ == "__main__" :
    
    

    def data_cut(path:str, stop:int) -> pd.DataFrame :
        """
        load a dataset, clean a specified number of samples of it and save it 

        Args: 
            path: path to the dataset to be cleaned
            stop: number of samples to be cleaned
        """

        # load the dataset with pandas
        df = pd.read_csv(path)

        # instantiate the text cleaner class
        text_cleaner = TextCleaner()

        # apply the cleaning to reviews
        df["cleaned_review"] = df["review"][:stop].apply(text_cleaner.clean_text)

        # make the labels and the reviwes have the same lenght
        df["sentiment"] = df["sentiment"][:stop]
        # drop the raw data cloumn
        df = df.drop(columns=["review"])

        df = df.dropna(subset=['cleaned_review'])
        df = df[df['cleaned_review'].str.strip() != ""]

        #save the processed reviews 
        df.to_csv(f"NPL_Sentiment_Analysis/data/processed/{stop}-cleaned-reviews.csv", index=False)

        return df
    
    # data 
    path = "NPL_Sentiment_Analysis/data/raw/IMDB Dataset.csv"

    df = data_cut(path, 50000)



    text_ = df["cleaned_review"].str.cat(sep=" ")

    # size of the wordcloud
    plt.rcParams["figure.figsize"] = [10, 10]

    # create a worldcloud of the most used words in the cleaned dataset
    wordcloud = WordCloud(max_font_size=50, max_words= 50, background_color="white", colormap="flag").generate(text_)
    
    # # plot the wordcloud
    plt.plot()
    plt.imshow(wordcloud ,interpolation="bilinear")
    plt.axis("off")
    plt.title("Most used words in the processed reviews")
    plt.show()

