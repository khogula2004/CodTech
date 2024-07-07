import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the data
file_path = './twitter_training.csv'  # replace with your actual file path
data = pd.read_csv(file_path)

# Rename columns for clarity
data.columns = ['id', 'category', 'sentiment', 'text']

# Extract relevant columns
data = data[['sentiment', 'text']]

# Drop rows with NaN values in the text column
data = data.dropna(subset=['text'])

# Preprocess the text (convert to lowercase and remove punctuation)
data['text'] = data['text'].str.lower().str.replace('[^\w\s]', '', regex=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)  # Increased max_iter for better convergence
model.fit(X_train_vect, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vect)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Function to predict sentiment of new text
def predict_sentiment(text):
    text_processed = re.sub('[^\w\s]', '', text.lower())  # Preprocess the text
    text_vect = vectorizer.transform([text_processed])  # Vectorize the text
    prediction = model.predict(text_vect)  # Predict sentiment
    return prediction[0]

# Interactive loop to predict sentiment of new text
try:
    while True:
        new_text = input("Enter text to analyze sentiment (or type 'exit' to quit): ")
        if new_text.lower() == 'exit':
            print("Exiting...")
            break
        predicted_sentiment = predict_sentiment(new_text)
        print(f"The sentiment of the new text '{new_text}' is: {predicted_sentiment}")
except KeyboardInterrupt:
    print("\nExiting...")

