# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
 
########################## Importation du fichier ########################

# la partie de l'analyse pour le fichier clinton c'est de 2015-2016
 
import os
os.chdir("C:/Users/souk/Desktop/ENSP/Textométrie/Projet_Texto")
chemin = "clintontwt.txt"
chemin
 
############################

#Remplir l'espace de noms interactif à partir de numpy et matplotlib
import pandas as pd
%pylab inline
import nltk

f=open(chemin,"r")  
textebrut = f.read()
textebrut = textebrut.lower() # Transformer tout le texte en minuscule
import re    
textebrut = re.sub(r"[@()'-: ; __ /’$'–?=!\n] +"," ",textebrut)
textebrut


############################### Apporter des modifications ########################

# supprimer les mots qui sont en plus 

textebrut= textebrut.replace(' s ', '')
textebrut= textebrut.replace(' y ', ' ')
textebrut= textebrut.replace(' i ', ' ')
textebrut= textebrut.replace(' t ', ' ')
textebrut= textebrut.replace(' a ', ' ')
textebrut= textebrut.replace(' on ', ' ')
textebrut= textebrut.replace(' ta ', ' ')
textebrut= textebrut.replace(' on ', ' ')
textebrut= textebrut.replace(' ! ', ' ')
textebrut= textebrut.replace(' : ', ' ')

textebrut

##################### Tokinization ###############################

#diviser les paragraphes du textes en mots 

import nltk

from nltk.tokenize import sent_tokenize
tokenized_text=sent_tokenize(textebrut)
print(tokenized_text)

from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(textebrut)
print(tokenized_word)


####################  Présentation de stopwords ###################

# récupérer les stopwords 
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words=set(stopwords.words("english"))
print(stop_words)

# Eliminer les stoppwords 

filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)

############################### Stemming  ############################

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


################################ #######################################
#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
word = "better"
print("Lemmatized Word:",lem.lemmatize(word,'v'))
print("Stemmed Word:",stem.stem(word))

#########################################

# transformer des données textuelles 

tokenizer = nltk.RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(textebrut)
list(nltk.bigrams(tokens))


############################################# Visualisation TSNE #################################
# visualisation TSNE ( ? c'est ici que j'ai l'erreur)

import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models import word2vec

# transformer notre liste en liste des listes 

texts = [[text for text in doc.split()] for doc in filtered_sent]
texts
model = word2vec.Word2Vec(texts, vector_size=100, window=20, min_count=200, workers=4)
m=model.wv['hillaryclinton']
model.wv.similar_by_word('hillaryclinton',topn=100)

# afficher le nuage de mots 

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    
    for word in model.wv.index_to_key :
        tokens.append(model.wv[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5, random_state=23)
    new_values = tsne_model.fit_transform(texts)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                                                                                                                 xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)

#.........................#

####################################################################################
####################################################################################

###################### Frequency Distribution#########################

from nltk.probability import FreqDist
fdist = FreqDist(filtered_sent)
print(fdist)

# Frequency Distribution Plot

import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False) 
plt.show()


#################################  Pos tagging #########################################
# pos tagging

pos.count('NP')  # nom propre 
pos.count('JJ')
pos.count('JJS')
pos.count('JJR')
pos.count('NNP')
pos.count('VBN')
pos.count("SYM")
pos.count("VVP")
pos.count('RB')
pos.count('VBZ')
pos.count('VVD')
pos.count('VBD')

vp=pos.count('')
print (vp)



##############################  Analyse des sentiments ###################
# Analyse des sentiments 

import vaderSentiment
nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
 ])

# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
# function to print sentiments
# of the sentence.
def sentiment_scores(textebrut):
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(textebrut)
     
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
 
    print("Sentence Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        print("Positive")
 
    elif sentiment_dict['compound'] <= - 0.05 :
        print("Negative")
 
    else :
        print("Neutral")

 
####################### WORDCLOUD #############################################################


from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd


stopwords=set(STOPWORDS)
stopwords.add("S")

def generate_better_wordcloud(data, title):
    cloud = WordCloud(scale=3,
                      colormap='RdYlGn',
                      background_color='white',
                      stopwords=stopwords,
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
    
# Use the function with the rome_corpus and our mask to create word cloud     
generate_better_wordcloud(textebrut, 'Hillary Clinton 2016')


################################################# une autre méthode ###############################
# code du prof # 

 ####Séparation des étiquettes et mise au format####
 
import treetaggerwrapper
tagger=treetaggerwrapper.TreeTagger(TAGLANG='en')

u=textebrut.split(' ')
u
tags=tagger.tag_text(u)
tags

import nltk
graphies=[]
lemmes=[]
pos=[]
for n in range(0,len(tags)):
    mot=tags[n]
    graphies.append(mot.split('\t')[0])
    lemmes.append(mot.split('\t')[2])
    pos.append(mot.split('\t')[1])
    
#conversion de listes en objets nltk

type(lemmes)
type(pos)
type(graphies)

lemmes=nltk.Text(lemmes)
graphies=nltk.Text(graphies)
pos=nltk.Text(pos)

type(lemmes)
type(pos)
type(graphies)

#### Fonctions NLTK de base ####

#ensemble des formes

set(lemmes)
set(pos)
set(graphies)
 
sorted(set(lemmes))
len(sorted(set(lemmes)))
voc=len(set(lemmes)) # ensemble des caractéristique  
mots=len(lemmes) # ensemble des mots utilisé 
mots
# paramètre de dispersion 
disp=float(voc)/float(mots)
print (voc, mots, disp)

lemmes.count('hillaryclinton')
lemmes.count('donaldtrump')
lemmes.count('america')
lemmes.count('we')
lemmes.count('make')
lemmes.count('great')
graphies.count('hillaryclinton')
graphies.count('donaldtrump')

#boucle écriture condensée
V=sorted(set(lemmes))
mots_longs=[w for w in V if len(w)>10]
print (mots_longs)

# création d'une fonction personnelle
def richesse_lexicale(text):
    voc=len(sorted(set(text)))
    mots=len(text)
    return float(voc)/float(mots)    

a=richesse_lexicale(lemmes)
b=richesse_lexicale(graphies)
print (a,b) 



#concordances
graphies.concordance('hillaryclinton')
graphies.dispersion_plot(['hillaryclinton','crooked','campaign','fake','democrat', 'america'])
lemmes.concordance("hillaryclinton", lines=10)
concordances_hillary=lemmes.concordance_list("hillaryclinton")
for i in concordances_hillary:
    print (i.line)

#co-occurences
from nltk import bigrams
list(bigrams(graphies))
Motav=sorted(set([a for (a,b) in bigrams(graphies) if b=='hillaryclinton'])) # mots récupérer avant le discours sur Hillaryclinton
Motap=sorted(set([b for (a,b) in bigrams(graphies) if a=='hillaryclinton']))
av = ' '.join(Motav)  # fusion de l'ensemble des textes
ap = ' '.join(Motap)  # fusion de l'ensemble des textes

generate_better_wordcloud(av, 'Word cloud des mots avant Hillary Clinton') # les mots prononcé avant de parler de hillary
generate_better_wordcloud(ap, 'Word cloud des mots après Hillary Clinton')

################### détection des types de mots ######################

Adj=[] # Liste des adjectifs
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'JJ' :
        Adj.append(mot.split('\t')[0])

# les verbes en présent     
verbP=[] # Liste des verbes 
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'VBP' :
        verbP.append(mot.split('\t')[0])
        
noms=[] # Liste des mots
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'NN' :
        noms.append(mot.split('\t')[0])
        

        
############################## Présentation du graphique bar ####################

# visualisation de l'ensmeble des mots dans notre corpus 
#########création d'un dataframe en foction de setmmed_word ##############"
from nltk.stem.snowball import SnowballStemmer
# The Snowball Stemmer requires that you pass a language parameter
df = pd.DataFrame(stemmed_words)
df = df[0].value_counts()
df

#df
#df['freq'] = df.groupby(0)[0].transform('count')
#df['freq'] = df.groupby(0)[0].transform('count')
#df.sort_values(by = ('freq'), ascending=False)

#This will give frequencies of our words

from nltk.probability import FreqDist

freqdoctor = FreqDist()
for words in df:
    freqdoctor[words] += 1
freqdoctor

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#This is a simple plot that shows the top 20 words being used
#df.plot(20)
df = df[:20,]
plt.figure(figsize=(10,5))
sns.barplot(df.values, df.index, alpha=0.8)
plt.title('Top Words Overall')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()      
################### VISUALISATION TOP DES ADJECTIFS  ###################

dif = pd.DataFrame(Adj) #¶ créer un dataframe qui contient que les adj 
dif = dif[0].value_counts()

from nltk.probability import FreqDist
freqdoctor = FreqDist()

for words in dif:
    freqdoctor[words] += 1
freqdoctor

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#This is a simple plot that shows the top 20 words being used
#df.plot(20)

dif = dif[:20,] # sélectionner que les 20 prmiers aadj 
plt.figure(figsize=(10,5))
sns.barplot(dif.values, dif.index, alpha=0.8)
plt.title('Top Adjectif Overall')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()


################# déterminer les mots fréquents  ######################
fd = nltk.FreqDist(filtered_sent)
 #déterminer les mots fréquent dans sample
fd.most_common(5) #  get a list of tuples containing each word and how many times it appears in your text
fd.tabulate(5)
lower_fd = nltk.FreqDist([w.lower() for w in fd])
lower_fd 

#...........................................#


####################### shape word cloud ###########################


import sys
from os import path
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt



def create_wordCloud(text):
    img_mask = np.array(Image.open("C:/Users/souka/Documents/Matser ENSP/Textométrie/Projet_Texto/Trump.jpg")) #use an image mask

    stopwords = set(STOPWORDS) #words to not include, like "it, the, ..., etc."
    wc = WordCloud(background_color = "white", max_words=200, mask=img_mask, stopwords=stopwords, contour_width=3)
    wc.generate(textebrut)
   
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")


##########################################################################################################

############   campagne 2020 #############################################################################

#########################################################################################################


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:46:11 2021

@author: dasou
"""

import nltk
import os
import re

################### Importation de la base de donnée ######################                
os.chdir("C:/Users/dasou/Desktop/Cours M1 ENSP/M2/Textométrie/Projet_Texto") # on fixe le répertoire de travail
chemin = "Biden.txt"           
f=open(chemin,"r")   
tweets = f.read() 
tweets

############## nettoyage de la base de donnée ####################
tweets = re.sub(r"[@),.('-:’$%'–;?=!\n] +"," ",tweets) 

tweets=tweets.upper()
tweets=tweets.replace(' IS ', ' ')
tweets=tweets.replace(' S ', '')
tweets=tweets.replace(' OF ', ' ')
tweets=tweets.replace(' AND ', ' ')
tweets=tweets.replace(' M ', ' ')
tweets=tweets.replace(' Y ', ' ')
tweets=tweets.replace(' THEY ', ' ')
tweets=tweets.replace(' IN ', ' ')
tweets=tweets.replace(' ON ', ' ')
tweets=tweets.replace(' I ', ' ')
tweets=tweets.replace(' THAT ', ' ')
tweets=tweets.replace(' WE ', ' ')
tweets=tweets.replace(' YOU ', ' ')
tweets=tweets.replace(' A ', ' ')
tweets=tweets.replace(' TO ', ' ')
tweets=tweets.replace(' FOR ', ' ')
tweets=tweets.replace(' ARE ', ' ')
tweets=tweets.replace(' IT ', ' ')
tweets=tweets.replace(' WILL ', ' ')
tweets=tweets.replace(' BE ', ' ')
tweets=tweets.replace(' HAVE ', ' ')
tweets=tweets.replace(' THIS ', ' ')
tweets=tweets.replace(' OUR ', ' ')
tweets=tweets.replace(' & ', ' ')
tweets=tweets.replace(' DO ', ' ')
tweets=tweets.replace(' ALL ', ' ')
tweets=tweets.replace(' BY ', ' ')
tweets=tweets.replace(' WAS ', ' ')
tweets=tweets.replace(' WITH ', ' ')
tweets=tweets.replace(' WE ', ' ')
tweets=tweets.replace(' SO ', ' ')
tweets=tweets.replace('’', ' ')
tweets=tweets.replace(' HAS ', ' ')
tweets=tweets.replace(' UP ', ' ')
tweets=tweets.replace(' NOT ', ' ')
tweets=tweets.replace(' AT ', ' ')
tweets=tweets.replace(' AS ', ' ')
tweets=tweets.replace(' IF ', ' ')
tweets=tweets.replace(' NO ', ' ')
tweets=tweets.replace(' OR ', ' ')
tweets=tweets.replace(' THE ', ' ')
tweets=tweets.replace(' HE ', ' ')
tweets=tweets.replace(' T ', ' ')


##################### Tokinization #######################


from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(tweets)
print(tokenized_word)

from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words=set(stopwords.words("english"))
print(stop_words)

# éliminer les stoppwords 
filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)

#################### la distribution de fréquence ######################

from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)

fdist.most_common(10)
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()



########################## Présentation de nuages de mots #######################


from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd


stopwords=set(STOPWORDS)
stopwords.add("S")

def generate_better_wordcloud(data, title):
    cloud = WordCloud(scale=3,
                      colormap='RdYlGn',
                      background_color='white',
                      stopwords=stopwords,
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
    
# Use the function with the rome_corpus and our mask to create word cloud     
generate_better_wordcloud(tweets, 'Donald Trump 2020')


########################### code du prof ##################

import treetaggerwrapper
tagger=treetaggerwrapper.TreeTagger(TAGLANG='en')

u=tweets.split(' ')

tags=tagger.tag_text(u)

####Séparation des étiquettes et mise au format####
graphies=[]
lemmes=[]
pos=[]
for n in range(0,len(tags)):
    mot=tags[n]
    graphies.append(mot.split('\t')[0])
    lemmes.append(mot.split('\t')[2])
    pos.append(mot.split('\t')[1])

#conversion de listes en objets nltk

type(lemmes)
type(pos)
type(graphies)

lemmes=nltk.Text(lemmes)
graphies=nltk.Text(graphies)
pos=nltk.Text(pos)

type(lemmes)
type(pos)
type(graphies)


#### Fonctions NLTK de base ####

#ensemble des formes

set(lemmes)
set(pos)
set(graphies)
 
sorted(set(lemmes))
len(sorted(set(lemmes)))
voc=len(set(lemmes))
mots=len(lemmes)

# paramètre de dispersion
disp=float(voc)/float(mots)
print (voc, mots, disp)

lemmes.count('great')
lemmes.count('AMERICA')
lemmes.count('we')
lemmes.count('big')
lemmes.count('Fake')
lemmes.count('JoeBiden')
lemmes.count('Biden')
graphies.count('CORRUPT')
graphies.count('AMERICA')
graphies.count('AMERICA')
graphies.count('AMERICA')

verbpp = pos.count('VVP')
vp=pos.count('JJ')
print (vp)
print(verbpp)
################### détecter les longues mots ###################
#boucle écriture condensée
V=sorted(set(lemmes))
mots_longs=[w for w in V if len(w)>4]
print (mots_longs)


# création d'une fonction personnelle
def richesse_lexicale(text):
    voc=len(sorted(set(text)))
    mots=len(text)
    return float(voc)/float(mots)    

a=richesse_lexicale(lemmes)
b=richesse_lexicale(graphies)
print (a,b) 



#concordances

graphies.concordance('JOEBIDEN')
graphies = graphies.replace('CORRUPTION', 'CORRUPT')
graphies.dispersion_plot(['JOEBIDEN','CORRUPT','ELECTION','FAKE','DEMOCRAT', 'AMERICA'])
lemmes.concordance("JOEBIDEN", lines=10)
concordances_biden=lemmes.concordance_list("JOEBIDEN")
for i in concordances_biden:
    print (i.line)

#co-occurences
from nltk import bigrams
list(bigrams(graphies))
Motav=sorted(set([a for (a,b) in bigrams(graphies) if b=='JOEBIDEN'])) # mots récupérer avant le discours sur Hillaryclinton
Motap=sorted(set([b for (a,b) in bigrams(graphies) if a=='JOEBIDEN']))
av = ' '.join(Motav)  # fusion de l'ensemble des textes
av = av.replace('CORRUPTION', 'CORRUPT')
ap = ' '.join(Motap)  # fusion de l'ensemble des textes

generate_better_wordcloud(av, 'Word cloud des mots avant JOE BIDEN') # les mots prononcé avant de parler de hillary
generate_better_wordcloud(ap, 'Word cloud des mots après JOE BIDEN')


###############################  Détection des types de mots ###########################

tweets=tweets.upper()

import treetaggerwrapper
tagger=treetaggerwrapper.TreeTagger(TAGLANG='en')

u=tweets.split(' ')

tags=tagger.tag_text(u)

Adj=[] # Liste des adjectifs
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'JJ' :
        Adj.append(mot.split('\t')[0])


 #################### Liste des verbes au présent###################
Verb_pre=[]
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'VBP' :
        Verb_pre.append(mot.split('\t')[0])
        

############### Liste des verbes au passé####################
Verb_pas=[] 
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'VBD' :
        Verb_pas.append(mot.split('\t')[0])
        
        
Verb=[] 
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'VVP' :
        Verb.append(mot.split('\t')[0])
        
################### Liste des noms###################
Nom=[] 
for n in range(0,len(tags)):
    mot=tags[n]
    if mot.split('\t')[1] == 'NN' :
        Nom.append(mot.split('\t')[0])
   
daf = pd.DataFrame(Verb) #¶ créer un dataframe qui contient que les adj 
daf = daf[0].value_counts()

df = pd.DataFrame(Verb_pre) #¶ créer un dataframe qui contient que les adj 
df = df[0].value_counts()
############################## Présentation du graphique bar ####################


################### VISUALISATION TOP DES ADJECTIFS  ###################

dif = pd.DataFrame(Adj) #¶ créer un dataframe qui contient que les adj 
dif = dif[0].value_counts()

from nltk.probability import FreqDist
freqdoctor = FreqDist()

for words in dif:
    freqdoctor[words] += 1
freqdoctor

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#This is a simple plot that shows the top 20 words being used
#df.plot(20)

dif = dif[:20,] # sélectionner que les 20 prmiers aadj 
plt.figure(figsize=(10,5))
sns.barplot(dif.values, dif.index, alpha=0.8)
plt.title('Top Adjectif Overall')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()


################# déterminer les mots fréquents  ######################
fd = nltk.FreqDist(filtered_sent)
 #déterminer les mots fréquent dans sample
fd.most_common(5) #  get a list of tuples containing each word and how many times it appears in your text
fd.tabulate(5)
lower_fd = nltk.FreqDist([w.lower() for w in fd])
lower_fd 

##############################  Analyse des sentiments ###################
# Analyse des sentiments 
import vaderSentiment
nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
 ])

# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
# function to print sentiments
# of the sentence.
def sentiment_scores(tweets):
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(tweets)
     
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
 
    print("Sentence Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        print("Positive")
 
    elif sentiment_dict['compound'] <= - 0.05 :
        print("Negative")
 
    else :
        print("Neutral")

###########################################################














