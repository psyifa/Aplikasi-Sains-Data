import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
url = 'https://raw.githubusercontent.com/psyifa/Aplikasi-Sains-Data/main/PreprocessingDatasetSephora.csv'
data = pd.read_csv(url)

# Split the dataset into training and testing sets
X = data['review_text']
y = data['product_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Evaluate the model on the testing set
X_test_tfidf = vectorizer.transform(X_test)
accuracy = model.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)

# Save the trained model
model.save("trained_model.pkl")
vectorizer.save("vectorizer.pkl")
