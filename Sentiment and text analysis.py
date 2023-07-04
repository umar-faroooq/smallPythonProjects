#!/usr/bin/env python
# coding: utf-8

# # Installing the libraries

# In[1]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install textblob')
get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')
get_ipython().system('pip install imblearn')
get_ipython().system('pip install svgling')


# In[2]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# # Importing the libraries

# In[3]:


import os
import sys
import string

arr = os.listdir()
arr


# In[4]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[5]:


#Importing libraries

#Basic libraries
import pandas as pd 
import numpy as np 

#NLTK libraries
import nltk
import re
import string
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.porter import PorterStemmer

from nltk.corpus import treebank
nltk.download('treebank')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import tag
from nltk import chunk
from nltk.corpus import treebank_chunk
from nltk import tokenize
from sklearn import preprocessing 

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
get_ipython().run_line_magic('matplotlib', 'inline')

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Other miscellaneous libraries
import svgling
from scipy import interp
from itertools import cycle
import cufflinks as cf
from collections import defaultdict
from collections import Counter


# In[6]:


pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[ ]:





# # Importing data

# In[7]:


table = pd.read_excel('extracted_table.xlsx')
table.head()


# In[8]:


summaries = pd.read_excel('extracted_summary.xlsx')
summaries.head(6)


# In[ ]:





# # Data Cleaning
# 
# 1. Removing the punctuations
# 2. General nltk stop words contains words like not,hasn't,would'nt which actually conveys a negative sentiment, hence can't remove them. So dropping the stop words which doesn't have any negative sentiment.

# In[9]:


clean_table=table.copy()

def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    #text = re.sub('\w*\d\w*', '', text)
    return text

clean_table['clean_sentences']=clean_table['text_sentences'].apply(lambda x:review_cleaning(x))


# In[10]:


clean_table.head()


# In[11]:


stop_words= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each', 
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above', 
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't", 
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from', 
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs', 
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all']

clean_table['processed_sentences'] = clean_table['clean_sentences'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[12]:


clean_table.head()


# In[ ]:





# # Feature Engineering
# 
# 1. Polarity: Textblob for figuring out the rate of sentiment. It is between [-1,1] where -1 is negative and 1 is positive polarity
# 
# 2. Review length: length of the review which includes each letters and spaces
# 
# 3. Word length: This measures how many words are there in review

# In[13]:


clean_table['polarity'] = clean_table['processed_sentences'].map(lambda text: TextBlob(text).sentiment.polarity)
clean_table['sentence_len'] = clean_table['clean_sentences'].astype(str).apply(len)
clean_table['word_count'] = clean_table['clean_sentences'].apply(lambda x: len(str(x).split()))


# In[14]:


clean_table.head()


# In[15]:


clean_table.head()


# In[ ]:





# # Analysing the data

# In[16]:


cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[17]:


## Sentiment polarity distribution

clean_table['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')


# In[18]:


#Positive sentences

clean_table[clean_table['polarity'] > 0.3]


# In[19]:


#Negative sentences

clean_table[clean_table['polarity'] < -0.1]


# In[20]:


def f(row):
    
    if row['polarity'] > -0.1 and row['polarity'] < 0.1:
        val = 'Neutral'
    elif row['polarity'] < -0.1:
        val = 'Negative'
    elif row['polarity'] > 0.1:
        val = 'Positive'
    else:
        val = -1
    return val


# In[21]:


#Applying the function in our new column

clean_table['sentiment'] = clean_table.apply(f, axis=1)
clean_table.head()


# In[ ]:





# # N-gram analysis

# An n-gram is a collection of n successive items in a text document that may include words, numbers, symbols, and punctuation. N-gram models are useful in many text analytics applications where sequences of words are relevant, such as in sentiment analysis, text classification, and text generation.

# In[22]:


#Filtering data
review_pos = clean_table[clean_table["sentiment"]=='Positive'].dropna()
review_neu = clean_table[clean_table["sentiment"]=='Neutral'].dropna()
review_neg = clean_table[clean_table["sentiment"]=='Negative'].dropna()

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace


# ## Monogram analysis
# 
# one word in reviews based on sentiments

# In[23]:


## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in review_pos["processed_sentences"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in review_neu["processed_sentences"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in review_neg["processed_sentences"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')

# Creating two subplots
fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                          "Frequent words of negative reviews"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
iplot(fig, filename='word-plots')


# ## Bigram analysis
# 
# most frequent two words in reviews based on sentiments

# In[24]:


## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in review_pos["processed_sentences"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in review_neu["processed_sentences"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in review_neg["processed_sentences"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(25), 'brown')

# Creating two subplots
fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,
                          subplot_titles=["Bigram plots of Positive reviews", 
                                          "Bigram plots of Neutral reviews",
                                          "Bigram plots of Negative reviews"
                                          ])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace2, 3, 1)


fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Bigram Plots")
iplot(fig, filename='word-plots')


# ## Trigram analysis
# 
# most frequent three words in reviews based on sentiments

# In[25]:


## Get the bar chart from positive reviews ##
for sent in review_pos["processed_sentences"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in review_neu["processed_sentences"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in review_neg["processed_sentences"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')

# Creating two subplots
fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04, horizontal_spacing=0.05,
                          subplot_titles=["Tri-gram plots of Positive reviews", 
                                          "Tri-gram plots of Neutral reviews",
                                          "Tri-gram plots of Negative reviews"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")
iplot(fig, filename='word-plots')


# In[ ]:





# # Visualizations: Wordcloud

# A word cloud (also known as a tag cloud) is a visual representation of words. Cloud creators are used to highlight popular words and phrases based on frequency and relevance. They provide you with quick and simple visual insights that can lead to more in-depth analyses.

# In[26]:


## All reviews

text = clean_table["processed_sentences"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[27]:


## Positive reviews

text = review_pos["processed_sentences"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[28]:


## Neutral reviews

text = review_neu["processed_sentences"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[29]:


## Negative reviews

text = review_neg["processed_sentences"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = stop_words).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:





# # Visualizations: Correlations

# A correlation heatmap is a heatmap that shows a 2D correlation matrix between two discrete dimensions, using colored cells to represent data from usually a monochromatic scale. The values of the first dimension appear as the rows of the table while of the second dimension as a column.

# In[30]:


clean_table.corr()


# In[31]:


# plotting correlation heatmap
dataplot = sns.heatmap(clean_table.corr(), cmap="YlGnBu", annot=True, vmin =-1, vmax = 1)
# displaying heatmap
plt.show()


# In[32]:


review_pos.corr()


# In[33]:


# plotting correlation heatmap
dataplot = sns.heatmap(review_pos.corr(), cmap="YlGnBu", annot=True, vmin =-1, vmax = 1)
# displaying heatmap
plt.show()


# In[34]:


review_neu.corr()


# In[35]:


# plotting correlation heatmap
dataplot = sns.heatmap(review_neu.corr(), cmap="YlGnBu", annot=True, vmin =-1, vmax = 1)
# displaying heatmap
plt.show()


# In[36]:


review_neg.corr()


# In[37]:


# plotting correlation heatmap
dataplot = sns.heatmap(review_neg.corr(), cmap="YlGnBu", annot=True, vmin =-1, vmax = 1)
# displaying heatmap
plt.show()


# In[ ]:





# # POS Tagging

# POS tagging involved labeling the words with their respective part of speech ie. noun, adjective, verb, etc.

# In[38]:


#Generating the tokens  of words
clean_table['tokens'] = [0]*len(clean_table)
for i in range(0,len(clean_table)):
    words = nltk.word_tokenize(clean_table["clean_sentences"][i])
    #Filtering the stopwords 
    words = [word for word in words if word not in stop_words]
    clean_table['tokens'][i] =  words


# In[39]:


clean_table.head()


# In[40]:


clean_table['POS'] = [0]*len(clean_table)
for i in range(0, len(clean_table)):
    POS = []
    for word in clean_table['tokens'][i]:
        POS.append(nltk.pos_tag([word]))
    clean_table['POS'][i] =  POS


# In[41]:


clean_table.head()


# ![POS-Tags.png](attachment:POS-Tags.png)

# In[42]:


# Visualising tree

sent = tokenize.word_tokenize(clean_table["clean_sentences"][37])
tagged_sent = tag.pos_tag(sent)
tree = chunk.ne_chunk(tagged_sent)
tree
#treebank_chunk.tagged_sents()[0]
#treebank_chunk.chunked_sents()[0]


# In[ ]:





# # Named Entity Recognition

# Named-entity recognition is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

# In[43]:


clean_table['NER'] = [0]*len(clean_table)

for i in range(0,len(clean_table)):
    doc = nlp(clean_table["text_sentences"][i])
 
    entity_list = []
    for ent in doc.ents:
        entity = [(ent.text, ent.label_)]
        entity_list.append(entity)

    clean_table['NER'][i] =  entity_list


# In[44]:


clean_table.head()


# In[45]:


clean_table.to_excel('analysed_sentences.xlsx')


# In[ ]:




