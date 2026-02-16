import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# load the cleaned dataser
df = pd.read_csv("NPL_Sentiment_Analysis/data/processed/1000_cleaned_reviews.csv")

# print(df.head(2))
df = df[:1000]
# groupe the poistive reviews  from the cleaned_review column in one bloc of text  
sorting = df["sentiment"] == "negative"
negative_review = df[sorting]
negative_review_cleaned = negative_review["cleaned_review"]
# print(df[sorting])
negative_text = negative_review_cleaned.str.cat(sep=" ")

# plot the wordcloud of the poistive reviews
plt.rcParams["figure.figsize"]= [10, 10]

nr_word_cloud = WordCloud(max_font_size=50, max_words=50, background_color="white", colormap="flag").generate(negative_text)
plt.plot()
plt.imshow(nr_word_cloud, interpolation="bilinear")
plt.axis("off")
plt.title("negative reviews most used words")
plt.show()
