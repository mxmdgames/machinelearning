
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
# Sample dataset (you should replace this with your real dataset)
emails = [
    "Buy now! Special offer!",
    "Hello, how are you?",
    "Claim your prize now!",
    "Meeting tomorrow at 10 AM",
    "Exclusive deal for you!",
    "Free gift with purchase",
    "Important meeting today",
    "Limited-time offer",
    "Great discounts inside",
    "Check out our new products",
    "Last chance to win",
    "Hello, just checking in",
    "Don't miss out on this opportunity",
    "Your account has been locked",
    "Urgent: action required",
    "Meeting rescheduled to 2 PM",
    "Special promotion for loyal customers",
    "Congratulations, you've won!",
    "Upcoming event details",
    "Get 50% off today only",
]

labels = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]

# Generate additional samples by randomly selecting from the existing ones
import random

additional_emails = []
additional_labels = []

# Number of additional samples to generate
num_additional_samples = 80

for _ in range(num_additional_samples):
    index = random.randint(0, len(emails) - 1)
    additional_emails.append(emails[index])
    additional_labels.append(labels[index])

# Combine the original and additional datasets
emails.extend(additional_emails)
labels.extend(additional_labels)

# Print the total number of samples
print("Total number of samples:", len(emails))

# You can now use the extended dataset for training your model.



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text to numerical features
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transform the testing data (only transform, don't fit again)
X_test_vectorized = vectorizer.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_vectorized, y_train)

# Make predictions on the testing data
predictions = model.predict(X_test_vectorized)

# Print the predictions
print(predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, predictions)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, predictions)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, predictions)
print("F1-score:", f1)

# Assuming you have a trained model named 'model'
with open('emailtest.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
# Save the vectorizer to a file (replace 'vectorizer_file.pkl' with your desired filename)
with open('vectorizeremailtest.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

