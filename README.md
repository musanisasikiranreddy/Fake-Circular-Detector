# Fake-Circular-Detector
import pandas as pd
import numpy as np
import string
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# Read data
f_path = '/content/Fake.csv'
r_path = '/content/True.csv'
data_fake = pd.read_csv(f_path)
data_true = pd.read_csv(r_path)
# Assign classes and perform manual testing data manipulation
data_fake["class"] = 0
data_true['class'] = 1
data_fake_manual_testing = data_fake.tail(10).copy()
for i in range(23480, 23470, -1):
  data_fake.drop([i], axis=0, inplace=True)
  data_true_manual_testing = data_true.tail(10).copy()
for i in range(21416, 23405, -1):
  data_true.drop([i], axis=0, inplace=True)
  data_fake_manual_testing['class'] = 0
  data_true_manual_testing['class'] = 1
# Merge data
data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge = data_merge.drop(['title', 'subject', 'date'], axis=1)
data_merge = data_merge.sample(frac=1).reset_index(drop=True)
# Text processing function
def wordopt(text):
  text = text.lower()
  text = re.sub(r'\[.*?\]', '', text)
  text = re.sub(r"\\W", " ", text)
  text = re.sub(r'https?://\S+|www\.\S+', '', text)
  text = re.sub(r'<.*?>+', '', text)
  text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub(r'\n', '', text)
  text = re.sub(r'\w*\d\w*', '', text)
  return text
data_merge['text'] = data_merge['text'].apply(wordopt)
# Train-test split
x = data_merge['text']
y = data_merge['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
# Model training
LR = LogisticRegression()
DT = DecisionTreeClassifier()
LR.fit(xv_train, y_train)
DT.fit(xv_train, y_train)
# Output label function
def output_label(n):
  if n == 0:
    return "Fake News"
  elif n == 1:
    return "Not a Fake News"
# Manual testing function
def manual_testing(news):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame(testing_news)
  new_def_test["text"] = new_def_test["text"].apply(wordopt)
  new_x_test = new_def_test["text"]
  new_xv_test = vectorization.transform(new_x_test)
  pred_lr = LR.predict(new_xv_test)
  pred_dt = DT.predict(new_xv_test)
  return (
          output_label(pred_lr[0]),
          output_label(pred_dt[0]),
         )
# Streamlit app
22
def main():
  st.title('FAKE CIRCULAR DETECTOR')
  input_text = st.text_input("Enter the news article")
  if st.button("Check for Fake News"):
    st.text("Classifying...")
# Process the input text
    input_text_processed = wordopt(input_text)
    new_x_test = pd.Series(input_text_processed)
    new_xv_test = vectorization.transform(new_x_test)
# Make predictions
    pred_lr = LR.predict(new_xv_test)
    pred_dt = DT.predict(new_xv_test)
    st.write("- LR Prediction: {}".format(output_label(pred_lr[0])))
    st.write("- DT Prediction: {}".format(output_label(pred_dt[0])))
  if __name__ == '__main__':
    main()
