# -*- coding: utf-8 -*-
"""NLP HW 2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CSCjmoM7Pd7Ux5jLeGSJMu28m8lQBgSe
"""

#Q1
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords


#reads everything all at once
p= open("/positive-words.txt", "r")

positive = p.read()
#Another method using full location
n = open("/negative-words.txt", "r", encoding="latin-1")

negative = n.read()

def gen_senti(text,p,n):
  #lets first clean our text for special characters and uppercase characters
  text = text.lower()
  #print(text)
  tokens = ''.join([char for char in text if char not in string.punctuation])
  tokens=text.split()
  #print(tokens)
   # Remove special characters and punctuation

  #print(tokens)
  #import the positive and negative lexicon dictionaries
  #stop_words = set(stopwords.words('english'))

 # Initialize variables to count positive and negative words
  pc, nc = 0, 0

    # Calculate the sentiment score
  for token in tokens:
        if token in p:
            pc += 1

        elif token in n:
            nc += 1

    # Check if there are any positive and negative words
  if pc == 0 and nc == 0:
        # No positive or negative words found, ignore and return None
        return None

    # Calculate the sentiment score (normalized between -1 and 1)
  S = (pc - nc) / (pc + nc)
  print(S)
  return S

#Let's test out function on a sample string

# Read the contents of the lexicon files
with open("/positive-words.txt", "r",encoding="latin-1") as n_file:
    negative = n_file.read()

with open("/negative-words.txt", "r",encoding="latin-1") as n_file:
    negative = n_file.read()

# Example usage for negative score of -1
text = "badly bad"
sentiment = gen_senti(text, positive, negative)

# Example usage for positive score of 1
text = "greatly very"
sentiment = gen_senti(text, positive, negative)


# Example usage for neutral score, expected score of 0
text = "bad very"
sentiment = gen_senti(text, positive, negative)
text = "good very bad"
sentiment = gen_senti(text, positive, negative)

"""Please create a
function called gen_senti that Tokenizes arbitrary text and compares each token with
the positive and negative lexicons of each dictionary and outputs the sentiment score, S.
Positive and negative words, pw and nw, count as a score of 1 and -1 respectively for
each word matched. The total count for pw and nw are pc and nc, respectively. Each
message sentiment, S, is normalized between -1 and 1. Any text that does not any
positive AND negative words would have to be ignored, and not scored. (60 points)
"""

import pickle

#lets load our pickle file the data

# Specify the path and file name of the Pickle file
file_name = "/the_data.pk"

# Open the Pickle file for reading in binary mode ('rb')
with open(file_name, 'rb') as file:
  loaded_data = pickle.load(file)

# Now, 'loaded_data' contains the deserialized Python object
print("Loaded data:", loaded_data)

#We've successfully loaded our pickle file, and can now proceed to
##Using the dataframe, body column, from lecture, the_data, apply this function to each
#corpus and add a column called “simple_senti” (15 points)

#Lets isolate body column
body_column=loaded_data['body']

#Now, we can apply our defined funciton
with open("/positive-words.txt", "r") as p_file:
    positive = p_file.read()

with open("/negative-words.txt", "r",encoding="latin-1") as n_file:
    negative = n_file.read()


#Create a new column "simple_senti" by applying the gen_senti function to each corpus
loaded_data['simple_senti'] = body_column.apply(lambda x: gen_senti(x, positive, negative))
print(loaded_data['simple_senti'])

print(loaded_data['simple_senti'])

#Q3
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Assuming you've already loaded the dataframe "the_data"

# Apply VADER sentiment analysis to the "body" column and store the "compound" scores in a new column "vader"
loaded_data['vader'] = loaded_data['body'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
print(loaded_data['vader'])

#Q4
#Compute the mean, median and standard_deviations of both sentiment measures,
#“simple_senti” and “vader” (10 points)

# Compute the mean
mean_simple_senti = loaded_data['simple_senti'].mean()
mean_vader = loaded_data['vader'].mean()

# Compute the median
median_simple_senti = loaded_data['simple_senti'].median()
median_vader = loaded_data['vader'].median()

# Compute the standard deviation
std_simple_senti = loaded_data['simple_senti'].std()
std_vader = loaded_data['vader'].std()

# Print or use these values as needed
print("Mean of simple_senti:", mean_simple_senti)
print("Median of simple_senti:", median_simple_senti)
print("Standard Deviation of simple_senti:", std_simple_senti)

print("Mean of vader:", mean_vader)
print("Median of vader:", median_vader)
print("Standard Deviation of vader:", std_vader)

from google.colab import drive
drive.mount('/content/drive')