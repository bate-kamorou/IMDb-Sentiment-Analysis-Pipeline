import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# load the cleaned dataser
df = pd.read_csv("NPL_Sentiment_Analysis/data/processed/1000_cleaned_reviews.csv")

# print(df.head(2))
df = df[:1000]
# groupe the poistive reviews  from the cleaned_review column in one bloc of text  
sorting = df["sentiment"] == "positive"
postive_review = df[sorting]
postive_review_cleaned = postive_review["cleaned_review"]
# print(df[sorting])
positive_text = postive_review_cleaned.str.cat(sep=" ")
# plot the wordcloud of the poistive reviews
plt.rcParams["figure.figsize"]= [10, 10]

pr_word_cloud = WordCloud(max_font_size=50, max_words=50, background_color="white", colormap="flag").generate(positive_text)
plt.plot()
plt.imshow(pr_word_cloud, interpolation="bilinear")
plt.axis("off")
plt.title("positive reviwes most used words")
plt.show()