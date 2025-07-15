import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
url = 'https://raw.githubusercontent.com/psyifa/Aplikasi-Sains-Data/main/PreprocessingDatasetSephora.csv'
data = pd.read_csv(url)

# Split the dataset
X = data['review_text']
y = data['product_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Evaluate
accuracy = model.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)

# Save model & vectorizer
joblib.dump(model, "trained_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Save accuracy to txt
with open("akurasi_model.txt", "w") as f:
    f.write(str(accuracy))
