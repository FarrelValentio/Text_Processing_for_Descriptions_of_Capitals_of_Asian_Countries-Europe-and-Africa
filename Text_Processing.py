# -*- coding: utf-8 -*-
!pip install geopandas

!pip install Unidecode

"""# Importing Library"""

# Data Structures
import numpy  as np
import pandas as pd
import geopandas as gpd
import json

# Corpus Processing
import re
import nltk.corpus
from unidecode                        import unidecode
from nltk.tokenize                    import word_tokenize
from nltk                             import SnowballStemmer
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.preprocessing            import normalize

# K-Means
from sklearn import cluster

# Visualization and Analysis
import matplotlib.pyplot  as plt
import matplotlib.cm      as cm
import seaborn            as sns
from sklearn.metrics                  import silhouette_samples, silhouette_score
from wordcloud                        import WordCloud

# Map Viz
import folium
import branca.colormap as cm
from branca.element import Figure

data = pd.read_csv('capital_city (1).csv', encoding='utf-8')
data.columns = map(str.lower, data.columns)
#Filter data
continents = ['Europe', 'Africa', 'Asia']
data = data.loc[data['continent'].isin(continents)]
data.head(10)

language='English'
corpus = data['teritory'].tolist()
corpus[18][0:447]

"""# Corpus Processing"""

# removes a list of words (ie. stopwords) from a tokenized list.
def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

# applies stemming to a list of tokenized words
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

# removes any words composed of less than 2 or more than 21 letters
def twoLetters(listOfTokens):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= 2 or len(token) >= 21:
            twoLetterWord.append(token)
    return twoLetterWord

def processCorpus(corpus, language):
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = SnowballStemmer(language)
    countries_list = [line.rstrip('\n') for line in open('countries.txt')] # Load .txt file line by line
    nationalities_list = [line.rstrip('\n') for line in open('nationalities.txt')] # Load .txt file line by line
    other_words = [line.rstrip('\n') for line in open('stopwords_scrapmaker.txt')] # Load .txt file line by line

    for document in corpus:
        index = corpus.index(document)
        corpus[index] = corpus[index].replace(u'\ufffd', '8')   # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace(',', '')          # Removes commas
        corpus[index] = corpus[index].rstrip('\n')              # Removes line breaks
        corpus[index] = corpus[index].casefold()                # Makes all letters lowercase

        corpus[index] = re.sub('\W_',' ', corpus[index])        # removes specials characters and leaves only words
        corpus[index] = re.sub("\S*\d\S*"," ", corpus[index])   # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub("\S*@\S*\s?"," ", corpus[index]) # removes emails and mentions (words with @)
        corpus[index] = re.sub(r'http\S+', '', corpus[index])   # removes URLs with http
        corpus[index] = re.sub(r'www\S+', '', corpus[index])    # removes URLs with www

        listOfTokens = word_tokenize(corpus[index])
        twoLetterWord = twoLetters(listOfTokens)

        listOfTokens = removeWords(listOfTokens, stopwords)
        listOfTokens = removeWords(listOfTokens, twoLetterWord)
        listOfTokens = removeWords(listOfTokens, countries_list)
        listOfTokens = removeWords(listOfTokens, nationalities_list)
        listOfTokens = removeWords(listOfTokens, other_words)

        listOfTokens = applyStemming(listOfTokens, param_stemmer)
        listOfTokens = removeWords(listOfTokens, other_words)

        corpus[index]   = " ".join(listOfTokens)
        corpus[index] = unidecode(corpus[index])

    return corpus

import nltk
nltk.download('punkt')
nltk.download('stopwords')

language = 'english'
corpus = processCorpus(corpus, language)
corpus[18][0:460]

"""###Statistical Weighting of Words"""

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
tf_idf = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())
final_df = tf_idf
print("{} rows".format(final_df.shape[0]))
final_df.T.nlargest(5, 0)

# first 5 words with highest weight on document 0:
final_df.T.nlargest(5, 0)

"""#K-Means"""

def run_KMeans(max_k, data):
    max_k += 1
    kmeans_results = dict()
    for k in range(2 , max_k):
        kmeans = cluster.KMeans(n_clusters = k
                               , init = 'k-means++'
                               , n_init = 10
                               , tol = 0.0001
                               , random_state = 1)

        kmeans_results.update( {k : kmeans.fit(data)} )

    return kmeans_results

# Running Kmeans from 1 to k
k = 8
kmeans_results = run_KMeans(k, final_df)

df = final_df.to_numpy()

"""### Elbow Method"""

from yellowbrick.cluster import KElbowVisualizer
km = cluster.KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(1,8))

visualizer.fit(df)
visualizer.show

"""### Silhouette Score"""

avg_dict = dict()
for n_clusters, kmeans in kmeans_results.items():
  kmeans_labels = kmeans.predict(df)
  silhouette_avg = silhouette_score(df, kmeans_labels) # Average Score for all Samples
  avg_dict.update( {silhouette_avg : n_clusters} )

Avg = []
Ks =[]

for avg in sorted(avg_dict.keys(), reverse=True):
  Avg.append(avg.round(4))
  Ks.append(avg_dict[avg])

  silhouette_list = pd.DataFrame(
      {'K' : Ks,
       'Average' : Avg
      })

silhouette_list = silhouette_list.sort_values(by=['K'])
silhouette_list

#plot average
plt.plot(silhouette_list.K, silhouette_list.Average)

"""Silhouette Score merupakan metode untuk mengukur perbedaan setiap klaster antara rata - rata jarak observasi dengan klaster di dalamnya (mean intra cluster distance) dan rata - rata jarak observasi dengan klaster terdekatnya (mean nearest cluster distance). Nilai dari silhoutte score berkisar -1 sampai 1, semakin mendekati 1, artinya klaster tersebut memiliki persebaran yang cukup baik dalam menjelaskan observasi - observasinya. Dan ketika nilai silhoutte score mendekati -1, artinya klaster tersebut belum bisa menjelaskan observasi yang ada. Namun ketika silhoutte score bernilai mendekati 0, artinya klaster kemungkinan mengalami overlapping. Berdasarkan plot di atas, 2 klaster merupakan klaster yang paling optimal untuk menjelaskan observasi yang ada. Karena rata - rata silhoutte score pada 2 klaster sebesar 0.0032 menjadi yang tertinggi daripada rata - rata silhoutte score pada klaster lainnya. Namun, nilai k tersebut kemungkinan mengalami overlapping."""

from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(4, 2, figsize=(15,8))
for i in [2,3,4,5,6,7,8]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = cluster.KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, color='yellowbrick', ax = ax[q-1][mod])
    visualizer.fit(df)

"""Berdasarkan plot di atas, dapat dinyatakan bahwa 2 klaster merupakan nilai klaster yang optimal. Hal ini dikarenakan dari kedua klaster tersebut memiliki nilai postif yang cukup besar dan nilai negatif yang cukup kecil dibandingkan dengan klaster - klaster lainnya, sehingga memungkinkan juga mengalami overlapping. Ketebalan dari silhoutte score dari masing - masing klaster merepresentasikan bahwa klaster tersebut saling berdistribusi uniform. Terlihat pada 3 klaster, meskipun memiliki ketebalan yang cukup baik antar klaster dan memiliki nilai positif yang cukup besar, namun ada nilai negatif silhoutte score yang cukup besar diantara klaster tersebut, sehingga akan mengalami overlapping yang berlebihan, begitu pula dengan 4 klaster  dan 5 klaster. Perhatikan juga pada 6 klaster, 7 klaster, dan 8 klaster, klaster - klaster tersebut memiliki silhoutte score positif yang cukup besar, namun ketebalan antar klaster tersebut kurang merata serta memiliki silhoutte score negatif yang cukup besar. Sehingga, meskipun beberapa klaster dapat menjelaskan sebagian observasi dengan baik, klaster - klaster tersebut juga tidak dapat menjelaskan sebagian observasi juga. Dan klaster - klaster tersebut tidak berdistribusi uniform. Ketika disimulasikan untuk 9 klaster, terlihat nilai silhoutte score sudah tidak ada, yang artinya data ini tidak dapat dijelaskan lagi untuk klaster yang lebih besar dari 8. Jadi, jumlah klaster yang optimal untuk menjelaskan data ini berjumlah 2. Meskipun 2 klaster tidak memiiliki nilai silhoutte score yang paling positif, namun dapat meminimalisir silhoutte score yang negatif, agar tidak terjadi overlapping yang berlebihan.

# Cluster Analysis
"""

def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vectorizer.get_feature_names_out()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def plotWords(dfs, n_feats):
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.show()

best_result = 2
kmeans = kmeans_results.get(best_result)

final_df_array = final_df.to_numpy()
prediction = kmeans.predict(final_df)
n_feats = 20
dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
plotWords(dfs, 13)

"""### Map of Words"""

# Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
def centroidsDict(centroids, index):
    a = centroids.T[index].sort_values(ascending = False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update( {a[i,0] : a[i,1]} )

    return centroid_dict

def generateWordClouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in range(0, len(centroids)):
        centroid_dict = centroidsDict(centroids, i)
        wordcloud.generate_from_frequencies(centroid_dict)

        plt.figure()
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

#Bener
centroids = pd.DataFrame(kmeans.cluster_centers_)
centroids.columns = final_df.columns
generateWordClouds(centroids)

"""### Preparing our final groups for visualization"""

# Assigning the cluster labels to each country
labels = kmeans.labels_
data['label'] = labels
data.head()

"""### Visualization the Clustered Countries in a Map"""

# Map Viz
import json
import geopandas as gpd

# Loading countries polygons
geo_path = 'world-countries.json'
country_geo = json.load(open(geo_path))
gpf = gpd.read_file(geo_path)

# Merging on the alpha-3 country codes
merge = pd.merge(gpf, data, left_on='id', right_on='alpha-3')
data_to_plot = merge[["id", "name", "label", "geometry"]]

data_to_plot.head(3)

import branca.colormap as cm

# Creating a discrete color map
values = data_to_plot[['label']].to_numpy()
color_step = cm.StepColormap(['r', 'y','g','b', 'm'], vmin=values.min(), vmax=values.max(), caption='step')

color_step

"""### Painting the Groups into a Choropleth Map"""

import folium
from branca.element import Figure

def make_geojson_choropleth(display, data, colors):
    '''creates geojson choropleth map using a colormap, with tooltip for country names and groups'''
    group_dict = data.set_index('id')['label'] # Dictionary of Countries IDs and Clusters
    tooltip = folium.features.GeoJsonTooltip(["name", "label"], aliases=display, labels=True)
    return folium.GeoJson(data[["id", "name","label","geometry"]],
                          style_function = lambda feature: {
                               'fillColor': colors(group_dict[feature['properties']['id']]),
                               #'fillColor': test(feature),
                               'color':'black',
                               'weight':0.5
                               },
                          highlight_function = lambda x: {'weight':2, 'color':'black'},
                          smooth_factor=2.0,
                          tooltip = tooltip)

# Makes map appear inline on notebook
def display(m, width, height):
    """Takes a folium instance and embed HTML."""
    fig = Figure(width=width, height=height)
    fig.add_child(m)
    return fig

# Initializing our Folium Map
m = folium.Map(location=[43.5775, -10.106111], zoom_start=2.3, tiles='cartodbpositron')

# Making a choropleth map with geojson
geojson_choropleth = make_geojson_choropleth(["Country:", "Group:"], data_to_plot, color_step)
geojson_choropleth.add_to(m)

width, height = 1300, 675
display(m, width, height)
m
