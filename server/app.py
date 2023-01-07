from flask import Flask, request
# for converting texts->numbers for model
import re #regex
# natural language tool kit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle,os
app = Flask(__name__)


def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # we just want words without special characters and digits
    stemmed_content = stemmed_content.lower() # make it  all lower
    stemmed_content = stemmed_content.split() # covert string to arr for the below line
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    # remove stop words like is and at these are insignificant words that are used for sentense formatation
    # they are not good keywords
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


@app.route('/check-fake-news', methods=["POST"])
def add_guide():
    title = request.json['title']
    author = request.json['author']
    body = request.json['body']
    new_data = [body+" "+author+" "+title]
    new_data = [stemming(text) for text in new_data]
    absolute_path = os.path.abspath(".")
    full_path2 = absolute_path +"/model/vectorizer.pkl"
    with open(full_path2, "rb") as f:
        vectorizer = pickle.load(f)
    new_data = vectorizer.transform(new_data)
    full_path = absolute_path +"/model/fake-news-model.pickle"
    fp1 = open(full_path,"rb")
    __model = pickle.load(fp1)
    fp1.close()
    predictions =  __model.predict(new_data)
    print(predictions)
    if predictions[0]==1 :
        return "real" 
    else :
        return "fake"


if __name__ == "__main__" :
    app.run(debug=True,port=8000) # dev