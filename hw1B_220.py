#!/usr/bin/env python
# coding: utf-8

# Kiara LaRocca | klarocca@ucsc.edu
# NLP 220 | Assignment 1
# October 25, 2024
# 
# This is Part B of Assignment 1. In Part B, we are doing binary classification with 5 different classifiers. The goal is to tune the hyper-parameters of each to find the best accuracy and then evaluate the classifiers on a test set. I did not use grid-search in this part, as I wanted to gain more experience with manually tuning parameters.

# In[1]:


# Import packages and libraries
import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[2]:


# Read in data from each folder and save as a CSV file for easy use in the future
'''# Define file paths
neg_path = 'C:/Users/kiara/OneDrive/Documents/NLP/220/aclImdb/train/neg'
pos_path = 'C:/Users/kiara/OneDrive/Documents/NLP/220/aclImdb/train/pos'

data = []

#Read negative files and label with 'neg'
for filename in os.listdir(neg_path):
    if filename.endswith(".txt"):
        with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            data.append([filename, content, 'neg'])  # Add 'neg' label for negative sentiment

# Read positive files and label with 'pos'
for filename in os.listdir(pos_path):
    if filename.endswith(".txt"):
        with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            data.append([filename, content, 'pos'])  # Add 'pos' label for positive sentiment

# Create a DataFrame with filename, content, and sentiment columns
df = pd.DataFrame(data, columns=['filename', 'content', 'sentiment'])

# Save the DataFrame to a CSV file
df.to_csv('sentiment.csv', index=False)'''


# In[3]:


# Repeating the above steps, but with the test set of data
'''
# Define file paths
neg_path = 'C:/Users/kiara/OneDrive/Documents/NLP/220/aclImdb/test/neg'
pos_path = 'C:/Users/kiara/OneDrive/Documents/NLP/220/aclImdb/test/pos'

test_data = []

# Read negative files and label with 'neg'
for filename in os.listdir(neg_path):
    if filename.endswith(".txt"):
        with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            if pd.notna(content):  # Skip NaN values
                test_data.append([filename, content, 'neg'])  # Add 'neg' label for negative sentiment

# Read positive files and label with 'pos'
for filename in os.listdir(pos_path):
    if filename.endswith(".txt"):
        with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            if pd.notna(content):  # Skip NaN values
                test_data.append([filename, content, 'pos'])  # Add 'pos' label for positive sentiment

# Create a DataFrame with filename, content, and sentiment columns
df = pd.DataFrame(test_data, columns=['filename', 'content', 'sentiment'])

# Drop rows with any NaN values in the DataFrame
df.dropna(inplace=True)

# Save the DataFrame to a CSV file
df.to_csv('sentiment_test.csv', index=False)
'''


# In[4]:


# Remove NaN values and view training DataFrame
df = pd.read_csv('sentiment.csv')
df = df.dropna(subset=['content', 'sentiment'])
df.head()


# In[5]:


# Remove NaN values and view testing DataFrame
test_df = pd.read_csv('sentiment_test.csv')
test_df = test_df.dropna(subset=['content', 'sentiment'])
test_df.head()


# In[6]:


# Create features and labels
X = df['content'].str.lower()  # Features
y = df['sentiment'].map({'neg': 0, 'pos': 1})  # Labels


# In[7]:


# Sentiment Class Distribution
df['sentiment'].value_counts()


# In[8]:


# Create the train/test split (90/10)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)


# In[9]:


# Naive Bayes Model
# Feature extraction for naive bayes
vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_train_nb = vectorizer.fit_transform(X_train)
X_val_nb = vectorizer.transform(X_val)

# Train naive bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train)

# Evaluate on validation set
y_pred_nb = nb_model.predict(X_val_nb)
nb_accuracy = accuracy_score(y_val, y_pred_nb)
print(f"Naive Bayes: Validation Accuracy = {nb_accuracy:.4f}")


# In[10]:


# Load test data features
X_test = test_df['content'].str.lower()

# Vectorize the test data using the same vectorizer used during training
X_test_nb = vectorizer.transform(X_test)

# Predict labels for the test data using the trained model
y_test_pred_nb = nb_model.predict(X_test_nb)

# Map predicted numeric labels back to 'neg' and 'pos'
y_test_pred_mapped = ['neg' if label == 0 else 'pos' for label in y_test_pred_nb]

# If true labels are available in the test CSV
if 'sentiment' in test_df.columns:
    y_test_true = test_df['sentiment'].map({'neg': 0, 'pos': 1})  # Convert true labels to numeric

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test_true, y_test_pred_nb)
    print(f"Naive Bayes: Test Accuracy = {test_accuracy:.4f}")
else:
    print("True labels not found in the test dataset. Accuracy cannot be calculated.")

# Save the predicted labels to the test DataFrame and export to CSV
test_df['predicted_label'] = y_test_pred_mapped
test_df.to_csv('sentiment_test_with_predictions.csv', index=False)

print("Predictions have been saved to 'sentiment_test_with_predictions.csv'")


# In[11]:


# Logistic Regression Model
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=10000, min_df=3, stop_words='english')

# Feature extraction for logistic regression
X_train_lr = vectorizer.fit_transform(X_train)
X_val_lr = vectorizer.transform(X_val)

# Train logistic regression model
logistic_model = LogisticRegression(C=1.0)
logistic_model.fit(X_train_lr, y_train)

# Evaluate on validation set
y_pred_lr = logistic_model.predict(X_val_lr)
logistic_accuracy = accuracy_score(y_val, y_pred_lr)
print(f"Logistic Regression: Validation Accuracy = {logistic_accuracy:.4f}")


# In[12]:


# Vectorize the test data using the same vectorizer used during training
X_test_log = vectorizer.transform(X_test)

# Predict labels for the test data using the trained model
y_test_pred_log = logistic_model.predict(X_test_log)

# Map predicted numeric labels (0 and 1) back to 'neg' and 'pos'
y_test_pred_mapped = ['neg' if label == 0 else 'pos' for label in y_test_pred_log]

# If true labels are available in the test CSV
if 'sentiment' in test_df.columns:
    y_test_true = test_df['sentiment'].map({'neg': 0, 'pos': 1})  # Convert true labels to numeric

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test_true, y_test_pred_log)
    print(f"Logistic Regression: Test Accuracy = {test_accuracy:.4f}")
else:
    print("True labels not found in the test dataset. Accuracy cannot be calculated.")

# Save the predicted labels to the test DataFrame and export to CSV
test_df['predicted_label'] = y_test_pred_mapped
test_df.to_csv('sentiment_test_with_predictions.csv', index=False)

print("Predictions have been saved to 'sentiment_test_with_predictions.csv'")


# In[13]:


# Random Forest Model
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=20000)

# Feature extraction for random forest
X_train_rf = vectorizer.fit_transform(X_train)
X_val_rf = vectorizer.transform(X_val)

# Train random forest model
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_rf, y_train)

# Evaluate on validation set
y_pred_rf = rf_model.predict(X_val_rf)
rf_accuracy = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest: Validation Accuracy = {rf_accuracy:.4f}")


# In[14]:


# Vectorize the test data using the same vectorizer used during training
X_test_rf = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)

# Train the model on the training data
rf_model.fit(X_train_rf, y_train)

# Predict labels for the test data using the trained model
y_test_pred_rf = rf_model.predict(X_test_rf)  # Outputs binary predictions (0 or 1)
y_pred = (y_test_pred_rf >= 0.5).astype(int)  # Convert to binary labels

# Map predicted numeric labels (0 and 1) back to 'neg' and 'pos'
y_test_pred_mapped = ['neg' if label == 0 else 'pos' for label in y_pred]

# If true labels are available in the test CSV
if 'sentiment' in test_df.columns:
    y_test_true = test_df['sentiment'].map({'neg': 0, 'pos': 1})  # Convert true labels to numeric

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test_true, y_pred)
    print(f"Random Forest: Test Accuracy = {test_accuracy:.4f}")
else:
    print("True labels not found in the test dataset. Accuracy cannot be calculated.")

# Save the predicted labels to the test DataFrame and export to CSV
test_df['predicted_label'] = y_test_pred_mapped
test_df.to_csv('sentiment_test_with_predictions.csv', index=False)

print("Predictions have been saved to 'sentiment_test_with_predictions.csv'")


# In[ ]:


# Gradient Boosting Model
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=10000)

# Feature extraction for gradient boosting
X_train_gb = vectorizer.fit_transform(X_train)
X_val_gb = vectorizer.transform(X_val)

# Train gradient boosting model
gb_model = GradientBoostingClassifier(n_estimators=300, max_depth=10, random_state=42)
gb_model.fit(X_train_gb, y_train)

# Evaluate on validation set
y_pred_gb = gb_model.predict(X_val_gb)
gb_accuracy = accuracy_score(y_val, y_pred_gb)
print(f"Gradient Boosting: Validation Accuracy = {gb_accuracy:.4f}")


# In[ ]:


# Vectorize the test data using the same vectorizer used during training
vectorizer = TfidfVectorizer(ngram_range=(1,3))

# Feature extraction for linear regression
X_train_lf = vectorizer.fit_transform(X_train)
X_val_lf = vectorizer.transform(X_val)

# Fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_lf, y_train)

# Predict continuous values and convert them to binary labels (0 or 1)
y_pred_continuous = linear_model.predict(X_val_lf)
y_pred = (y_pred_continuous >= 0.5).astype(int)  # Convert to binary labels


# Evaluate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Linear Regression: Validation Accuracy = {accuracy:.4f}")


# In[ ]:


# Vectorize the test data using the same vectorizer used during training
X_test_lin = vectorizer.transform(X_test)

# Predict labels for the test data using the trained model
y_test_pred_lin = linear_model.predict(X_test_lin)  # Outputs binary predictions (0 or 1)
y_pred = (y_test_pred_lin >= 0.5).astype(int)  # Convert to binary labels

# Map predicted numeric labels (0 and 1) back to 'neg' and 'pos'
y_test_pred_mapped = ['neg' if label == 0 else 'pos' for label in y_pred]

# If true labels are available in the test CSV
if 'sentiment' in test_df.columns:
    y_test_true = test_df['sentiment'].map({'neg': 0, 'pos': 1})  # Convert true labels to numeric

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test_true, y_pred)
    print(f"Linear Regression: Test Accuracy = {test_accuracy:.4f}")
else:
    print("True labels not found in the test dataset. Accuracy cannot be calculated.")

# Save the predicted labels to the test DataFrame and export to CSV
test_df['predicted_label'] = y_test_pred_mapped
test_df.to_csv('sentiment_test_with_predictions.csv', index=False)

print("Predictions have been saved to 'sentiment_test_with_predictions.csv'")


# In[ ]:




