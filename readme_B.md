# Sentiment Analysis of Customer Reviews

## **Introduction**
This document presents the implementation of a sentiment analysis model that classifies customer reviews as positive or negative. The project includes data preprocessing, feature extraction using TF-IDF, and training a classification model using machine learning.

## **Implementation Steps**

### **1. Data Collection**
We use the publicly available IMDb movie reviews dataset, which consists of labeled reviews (positive and negative). The dataset is preloaded from the `nltk.corpus.movie_reviews` module.

### **2. Data Preprocessing**
Preprocessing involves text cleaning, tokenization, stopword removal, and lowercasing.

#### **Preprocessing Function**
```python
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)
```

### **3. Feature Extraction**
We use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors. TF-IDF is chosen over Bag of Words (BoW) because it assigns importance to words based on their relevance across the corpus.

#### **TF-IDF Vectorization**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text data to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['label']  # Target labels
```

### **4. Model Selection and Training**
We train a **Logistic Regression** classifier to predict sentiment labels.

#### **Model Training**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### **5. Evaluation**
We evaluate the model using accuracy, precision, recall, and F1-score.

#### **Performance Evaluation**
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
```

## **Challenges and Insights**
### **Challenges:**
- Handling noisy text data (e.g., misspellings, sarcasm, contextual variations).
- TF-IDF vectorization may not fully capture semantic meaning compared to deep learning approaches.

### **Insights:**
- Logistic Regression provides a strong baseline for text classification.
- TF-IDF works well for sentiment classification but can be improved using embeddings (e.g., Word2Vec, BERT).
- Further improvements can be achieved through hyperparameter tuning and ensemble models.

## **Conclusion**
This sentiment analysis pipeline effectively classifies customer reviews with good accuracy. Future improvements could include:
- **Deep learning models** such as LSTMs or transformers.
- **Expanding the dataset** for more robust generalization.
- **Fine-tuning hyperparameters** for better performance.

---
📌 **End of Part B Documentation** ✅

