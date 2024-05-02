import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc




# Load the dataset
file_path = 'transaction.csv'
data = pd.read_csv(file_path)

# Create new features
data['Address_Mismatch'] = (data['Shipping Address'] != data['Billing Address']).astype(int)
data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
data['Transaction_Hour'] = data['Transaction Date'].dt.hour



# Selecting features
X = data.drop(['Transaction ID', 'Customer ID', 'Transaction Date', 'Is Fraudulent',
               'Shipping Address', 'Billing Address', 'IP Address'], axis=1)
y = data['Is Fraudulent']

# Encoding categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Payment Method', 'Product Category', 'Device Used', 'Customer Location']
for column in categorical_columns:
    X[column] = label_encoder.fit_transform(X[column].astype(str))

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train, y_train)

# Evaluating the model
y_pred = rf_classifier.predict(X_test)
classification_report_rf = classification_report(y_test, y_pred)
confusion_matrix_rf = confusion_matrix(y_test, y_pred)
print(classification_report_rf)

# Saving the model and label encoder
joblib.dump(rf_classifier, 'random_forest_model.pkl')
joblib.dump(label_encoder, 'label_encoder.joblib')


# Plotting distribution of Transaction Amount
plt.figure(figsize=(10, 6))
sns.histplot(data['Transaction Amount'], bins=50, kde=True, color='#0077B6')
plt.title('Distribution of Transaction Amount', fontsize=15)
plt.xlabel('Transaction Amount', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# Plotting distribution of Customer Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Customer Age'], bins=30, kde=True, color='#00B4D8')
plt.title('Distribution of Customer Age', fontsize=15)
plt.xlabel('Customer Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# Plotting distribution of Quantity
plt.figure(figsize=(10, 6))
sns.countplot(x=data['Quantity'], palette='Blues')
plt.title('Distribution of Quantity', fontsize=15)
plt.xlabel('Quantity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# Plotting distribution of Payment Method
plt.figure(figsize=(10, 6))
sns.countplot(x=data['Payment Method'], palette='Blues')
plt.title('Distribution of Payment Method', fontsize=15)
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# Plotting distribution of Product Category
plt.figure(figsize=(10, 6))
sns.countplot(x=data['Product Category'], palette='Blues')
plt.title('Distribution of Product Category', fontsize=15)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# Plotting distribution of Device Used
plt.figure(figsize=(10, 6))
sns.countplot(x=data['Device Used'], palette='Blues')
plt.title('Distribution of Device Used', fontsize=15)
plt.xlabel('Device Used', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# Plotting distribution of Is Fraudulent
plt.figure(figsize=(10, 6))
sns.countplot(x=data['Is Fraudulent'], palette='Blues')
plt.title('Distribution of Is Fraudulent', fontsize=15)
plt.xlabel('Is Fraudulent', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# Plotting correlation heatmap of numeric features
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features', fontsize=15)
plt.show()


# Predict probabilities for ROC curve
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Display AUC score
print("AUC Score:", roc_auc)



# Cross-validation for accuracy
cross_val_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), cross_val_scores, marker='o', linestyle='--', color='b')
plt.title('Cross Validation Scores for Random Forest Classifier')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.show()



# Reformatting and plotting the confusion matrix with better formatting
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_rf, cmap='Blues', annot=True, fmt='.0f')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


import time

start_time = time.time()
# Code for fitting your model
end_time = time.time()
turnaround_time = end_time - start_time
print("Turnaround Time:", turnaround_time, "seconds")

