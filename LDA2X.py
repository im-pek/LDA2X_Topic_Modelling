import pandas as pd
from helper import *
# ! pip install pandas nltk gensim pyldavis
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
import string

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 

def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    return two

df = pd.read_csv(open("abstracts for 'automat'.csv", errors='ignore'))
df=df.astype(str)

text = df.applymap(cleaning)['paperAbstract']
text_list = [i.split() for i in text]


#add log for recording the model fitting data while training

from time import time
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                   filename='running.log',filemode='w')


#build dictionary

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(text_list)
dictionary.save('dictionary.dict')


#build corpus

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)


#Running LDA Model

start = time()
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=1, random_state=1, iterations=3) #'random_state=1' un-randomises LDA
#print 'used: {:.2f}s'.format(time()-start)

#print(ldamodel.print_topics(num_topics=2, num_words=4))

for i in ldamodel.print_topics():
    for j in i:
        print (j) ###########################
    

#save model for future use
    
ldamodel.save('topic.model')


#load saved model

from gensim.models import LdaModel
loading = LdaModel.load('topic.model')

#print(loading.print_topics(num_topics=2, num_words=4))


#PLOTTING

import pyLDAvis.gensim
import gensim
#pyLDAvis.enable_notebook()

d = gensim.corpora.Dictionary.load('dictionary.dict')
c = gensim.corpora.MmCorpus('corpus.mm')
lda = gensim.models.LdaModel.load('topic.model')

data = pyLDAvis.gensim.prepare(lda, c, d)


text_file = open('LDA-Corex-Word2Vec.txt', 'w')

text_file.write(str(data))

text_file.close()

pyLDAvis.save_html(data,'LDA-Corex-Word2Vec Visualisation of LDA.html')


#Plot words importance

import gensim
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
lda = gensim.models.LdaModel.load('topic.model')

fiz=plt.figure(figsize=(15,30))
for i in range(10):
    df=pd.DataFrame(lda.show_topic(i), columns=['term','prob']).set_index('term')
    df=df.sort_values('prob')
    
    plt.subplot(5,2,i+1)
    plt.title('topic '+str(i+1))
    sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Reds_d')
    plt.xlabel('probability')

plt.show()

print ('\n')

top_words=[]

print ('\n')
print ('LDA TOP TOPICS:')
print ('\n')

for index, topic in lda.show_topics(formatted=False, num_words= 30):
    print ('Topic {} \nWords: {}'.format(index+1, [w[0] for w in topic]))
    within_topic=[]
    for item in topic:
        within_topic.append(item[0])
    top_words.append(within_topic) 
#    print (top_words) #list view of top_words
    print ('\n')
    


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct

vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df=0.01)
vector=vectorizer.fit_transform(top_words).todense()
vocab=vectorizer.vocabulary_

numpy_array=np.array(vector, dtype=int)

# Sparse matrices are also supported
X = ss.csr_matrix(numpy_array)

#print (X.shape[0])

# WORD LABELS for each column can be provided to the model
all_vocabs=list(vocab.keys())


# DOCUMENT LABELS for each row can be provided
topics = np.arange(10)

seed = 1 #CHANGE THE SEED HERE

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=10, seed=seed)#, n_words=30)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=all_vocabs, docs=topics)

topics = topic_model.get_topics()

print ('Corex Topics:')
for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n+1)+': '+','.join(words)
    print(topic_str)

print ('\n')

print ('Corex Top Documents:')
top_docs = topic_model.get_top_docs()
for topic_n, topic_docs in enumerate(top_docs):
    docs,probs = zip(*topic_docs)
    topic_str = str(topic_n+1)+': '+ ''.join(str(docs))
    print(topic_str)