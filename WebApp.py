import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import re 
from sklearn.metrics.pairwise import cosine_similarity

# Pembersihan teks
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
stopworda = set(['were', 'should', "needn't", 'just', 'not', 's', "it's", 'between', 'are', 'all', 'did', 'until', 'will', 'your', 'against', 'needn', 'that', 'very', 'to', "shan't", 'in', 'doing', "hadn't", 'he', "couldn't", 'what', 'such', 'too', 'an', 'here', 'won', 'they', 've', "weren't", 'both', 'other', 'why', 'above', 'same', 'ma', 'ours', 'm', "shouldn't", 'haven', 'before', "wasn't", 'him', 'shan', 'yourselves', 'by', 'yours', 'his', 'the', 'because', 'most', "haven't", 'was', 'any', 'my', 'but', 'then', 'their', 'our', 'y', 'nor', 'shouldn', 'didn', "you'll", "didn't", 'of', 'isn', 'wouldn', 'theirs', 'them', 'few', 'there', 'd', 'further', 'o', 'myself', 'ain', 'at', 'now', 'be', "wouldn't", 'doesn', 'during', 'hers', "isn't", 'yourself', 'hadn', 'weren', 'some', "aren't", "she's", "that'll", 'over', 'she', 't', 'while', 'couldn', 'these', 'hasn', 'with', 'its', 'this', 'we', 'down', "mightn't", 'ourselves', 'me', 'once', 'how', 'about', 'through', 'and', "you're", 'll', 'as', "don't", 'am', 'having', 'whom', "you'd", 'for', 'have', 'can', 're', 'who', 'had', 'themselves', 'it', 'itself', 'again', 'does', 'i', 'into', 'if', 'her', "mustn't", "you've", "won't", 'those', 'on', 'or', 'under', 'only', 'so', 'mustn', 'a', 'don', 'being', 'is', 'where', 'herself', 'wasn', 'off', 'more', 'up', 'has', 'below', 'no', 'been', 'do', 'own', "should've", 'aren', "doesn't", "hasn't", 'than', 'which', 'each', 'from', 'you', 'when', 'after', 'out', 'himself', 'mightn'])

# Dataset
url = 'https://raw.githubusercontent.com/psyifa/Aplikasi-Sains-Data/main/PreprocessingDatasetSephora.csv'
Data = pd.read_csv(url)

# Page
title = "Beauty Things"
subtitle = "Let's find skincare for you!"

st.set_page_config(page_title = title,
                   page_icon = None,
                   layout = "wide")
      
st.sidebar.header("Select the options :")

with st.sidebar:          
      Category = st.selectbox('Category', Data['secondary_category'].unique())
      Skin = st.selectbox('Skin Tone', Data['skin_tone'].unique())
      Type = st.selectbox('Skin Type', Data['skin_type'].unique())
      
def clean_text(text): 
    text = text.lower() # lowercase text
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopworda) # hapus stopword dari kolom deskripsi
    return text
  

def main():
    st.title("Beauty Things")
    st.write("Let's find skincare for you!")
    st.write("Dataset Visualization :")
    
    # Bubble Chart
    category_counts = Data['secondary_category'].value_counts()
    colors = plt.cm.get_cmap('tab20c', len(category_counts))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(category_counts.index, category_counts.values, s=category_counts.values*10, c=colors(range(len(category_counts))), alpha=0.7)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Number of Products in Each Category (Bubble Chart)')
    ax.set_xticks(category_counts.index)
    ax.set_xticklabels(category_counts.index, rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
#     Filter Data
    filtered_data = Data.loc[Data.secondary_category == Category]
    filtered_data = filtered_data.loc[filtered_data.skin_tone == Skin]
    filtered_data = filtered_data.loc[filtered_data.skin_type == Type]
    
    Product = st.selectbox('Product Name', filtered_data['product_name'].unique())
    st.write(filtered_data[filtered_data.product_name == Product].review_text)
    st.write('Other Recommendations :')
    
    Data['review_clean'] = Data['review_text'].apply(clean_text)

    Data.set_index('product_name', inplace=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(Data['review_clean'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(Data.index)
    
#     Rekomendasi
    def recommendations(name, cos_sim = cos_sim):
        recommendedProducts = []
        recommendedProductsreview = []
        idx =indices[indices == name].index[0]
        score = pd.Series(cos_sim[idx]).sort_values(ascending = False)
        top_10 = list(score.iloc[1:11].index)
        for i in top_10:
            recommendedProducts.append(list(Data.index)[i])
            recommendedProductsreview.append(list(Data.review_text)[i])
        return recommendedProducts, recommendedProductsreview, score
    
    Rproduct, Rreview, score = recommendations(Product)
 
    Rprint = pd.DataFrame(({'Product': Rproduct,'Review': Rreview, 'Similarity Score':list(score.iloc[1:11])}))

    st.table(Rprint)
    
            

if __name__ == "__main__":
    main()
