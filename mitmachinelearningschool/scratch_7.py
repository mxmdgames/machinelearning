import tkinter as tk
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from main import model

# Load the original vectorizer (replace 'vectorizer_file.pkl' with the actual filename)
with open('vectorizeremailtest.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to classify email
def classify_email():
    email_text = email_entry.get()  # Get email text from the input field
    # Preprocess the email_text using the loaded vectorizer
    email_vectorized = vectorizer.transform([email_text])
    # Use your trained model to predict whether it's spam or not spam
    prediction = model.predict(email_vectorized)
    if prediction[0] == 1:
        result_label.config(text="Spam", fg="red")
    else:
        result_label.config(text="Not Spam", fg="green")

# ... (the rest of your classification code remains the same)


# Function to classify email
def classify_email():
    email_text = email_entry.get()  # Get email text from the input field
    # Preprocess the email_text using the fitted vectorizer
    email_vectorized = vectorizer.transform([email_text])
    # Use your trained model to predict whether it's spam or not spam
    prediction = model.predict(email_vectorized)
    if prediction[0] == 1:
        result_label.config(text="Spam", fg="red")
    else:
        result_label.config(text="Not Spam", fg="green")

# Create the main application window
root = tk.Tk()
root.title("Email Classifier")

# Create GUI components
email_label = tk.Label(root, text="Enter an email:")
email_entry = tk.Entry(root, width=50)
classify_button = tk.Button(root, text="Classify", command=classify_email)
result_label = tk.Label(root, text="", font=("Arial", 16))

# Place GUI components on the window
email_label.pack()
email_entry.pack()
classify_button.pack()
result_label.pack()

# Start the Tkinter main loop
root.mainloop()
