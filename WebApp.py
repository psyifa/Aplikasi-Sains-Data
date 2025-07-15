import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re 

# Pembersihan teks
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
stopworda = setimport streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re 

# Pembersihan teks
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
stopworda = set([...])  # tetap gunakan stopword set yang sama seperti sebelumnya (dari kode kamu)

# Dataset
url = 'https://raw.githubusercontent.com/psyifa/Aplikasi-Sains-Data/main/PreprocessingDatasetSephora.csv'
Data = pd.read_csv(url)

# Page settings
st.set_page_config(page_title="Beauty Things", page_icon=None, layout="wide")
st.title("Beauty Things")

# Tampilkan akurasi model
try:
    with open("akurasi_model.txt", "r") as f:
        accuracy = float(f.read().strip())
    st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
except:
    st.warning("Model accuracy not available.")

st.write("Let's find skincare for you!")
st.sidebar.header("Select the options :")

# Sidebar filters
with st.sidebar:          
    Category = st.selectbox('Category', Data['secondary_category'].unique())
    Skin = st.selectbox('Skin Tone', Data['skin_tone'].unique())
    Type = st.selectbox('Skin Type', Data['skin_type'].unique())

# Fungsi cleaning text
def clean_text(text): 
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopworda)
    return text

def main():
    st.write("Dataset Visualization :")
    
    # Bubble Chart
    category_counts = Data['secondary_category'].value_counts()
    colors = plt.cm.get_cmap('tab20c', len(category_counts))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(category_counts.index, category_counts.values, s=category_counts.values*10, c=colors(range(len(category_counts))), alpha=0.7)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Number of Products in Each Category (Bubble Chart)')
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(category_counts.index, rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Filter berdasarkan sidebar
    filtered_data = Data[
        (Data.secondary_category == Category) &
        (Data.skin_tone == Skin) &
        (Data.skin_type == Type)
    ]

    if filtered_data.empty:
        st.info("No products match your filter selection.")
        return

    Product = st.selectbox('Product Name', filtered_data['product_name'].unique())
    st.write(filtered_data[filtered_data.product_name == Product].review_text)

    st.write('Other Recommendations :')

    # Preprocessing review
    Data['review_clean'] = Data['review_text'].apply(clean_text)
    Data.set_index('product_name', inplace=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(Data['review_clean'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(Data.index)

    # Fungsi rekomendasi
    def recommendations(name, cos_sim=cos_sim):
        recommendedProducts = []
        recommendedProductsreview = []
        idx_list = indices[indices == name].index
        if idx_list.empty:
            return None, None, None
        idx = idx_list[0]
        score = pd.Series(cos_sim[idx]).sort_values(ascending=False)
        top_10 = list(score.iloc[1:11].index)
        for i in top_10:
            recommendedProducts.append(list(Data.index)[i])
            recommendedProductsreview.append(list(Data.review_text)[i])
        return recommendedProducts, recommendedProductsreview, score

    # Tampilkan hasil rekomendasi atau pesan info
    if Product in Data.index:
        Rproduct, Rreview, score = recommendations(Product)
        if Rproduct:
            Rprint = pd.DataFrame({
                'Product': Rproduct,
                'Review': Rreview,
                'Similarity Score': list(score.iloc[1:11])
            })
            st.table(Rprint)
        else:
            st.info("No recommendations available for this product.")
    else:
        st.info("No recommendations available for this product.")

if __name__ == "__main__":
    main()

# Dataset
url = 'https://raw.githubusercontent.com/psyifa/Aplikasi-Sains-Data/main/PreprocessingDatasetSephora.csv'
Data = pd.read_csv(url)

# Page settings
st.set_page_config(page_title="Beauty Things", page_icon=None, layout="wide")
st.title("Beauty Things")

# Tampilkan akurasi model
try:
    with open("akurasi_model.txt", "r") as f:
        accuracy = float(f.read().strip())
    st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
except:
    st.warning("Model accuracy not available.")

st.write("Let's find skincare for you!")
st.sidebar.header("Select the options :")

# Sidebar filters
with st.sidebar:          
    Category = st.selectbox('Category', Data['secondary_category'].unique())
    Skin = st.selectbox('Skin Tone', Data['skin_tone'].unique())
    Type = st.selectbox('Skin Type', Data['skin_type'].unique())

# Fungsi cleaning text
def clean_text(text): 
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopworda)
    return text

def main():
    st.write("Dataset Visualization :")
    
    # Bubble Chart
    category_counts = Data['secondary_category'].value_counts()
    colors = plt.cm.get_cmap('tab20c', len(category_counts))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(category_counts.index, category_counts.values, s=category_counts.values*10, c=colors(range(len(category_counts))), alpha=0.7)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Number of Products in Each Category (Bubble Chart)')
    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(category_counts.index, rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Filter berdasarkan sidebar
    filtered_data = Data[
        (Data.secondary_category == Category) &
        (Data.skin_tone == Skin) &
        (Data.skin_type == Type)
    ]

    if filtered_data.empty:
        st.info("No products match your filter selection.")
        return

    Product = st.selectbox('Product Name', filtered_data['product_name'].unique())
    st.write(filtered_data[filtered_data.product_name == Product].review_text)

    st.write('Other Recommendations :')

    # Preprocessing review
    Data['review_clean'] = Data['review_text'].apply(clean_text)
    Data.set_index('product_name', inplace=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(Data['review_clean'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(Data.index)

    # Fungsi rekomendasi
    def recommendations(name, cos_sim=cos_sim):
        recommendedProducts = []
        recommendedProductsreview = []
        idx_list = indices[indices == name].index
        if idx_list.empty:
            return None, None, None
        idx = idx_list[0]
        score = pd.Series(cos_sim[idx]).sort_values(ascending=False)
        top_10 = list(score.iloc[1:11].index)
        for i in top_10:
            recommendedProducts.append(list(Data.index)[i])
            recommendedProductsreview.append(list(Data.review_text)[i])
        return recommendedProducts, recommendedProductsreview, score

    # Tampilkan hasil rekomendasi atau pesan info
    if Product in Data.index:
        Rproduct, Rreview, score = recommendations(Product)
        if Rproduct:
            Rprint = pd.DataFrame({
                'Product': Rproduct,
                'Review': Rreview,
                'Similarity Score': list(score.iloc[1:11])
            })
            st.table(Rprint)
        else:
            st.info("No recommendations available for this product.")
    else:
        st.info("No recommendations available for this product.")

if __name__ == "__main__":
    main()
