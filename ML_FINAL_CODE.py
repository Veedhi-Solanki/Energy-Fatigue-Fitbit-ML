#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the file is named 'merged_data_final.csv'
file_path = 'merged_data_final.csv'  # Update with the actual path

# Load the data
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())

# Get the percentage of missing values in each column
missing_percentage = data.isnull().mean() * 100
print(missing_percentage)


# In[50]:


# Drop columns with very high percentage of missing values
columns_to_drop = ['TrackerDistance', 'Fat', 'LogId','WeightPounds', 'IsManualReport'] 
data = data.drop(columns=columns_to_drop)

data = pd.read_csv(file_path)
# Correct datetime conversion and feature extraction
data['ActivityDate'] = pd.to_datetime(data['ActivityDate'])
data['DayOfYear'] = data['ActivityDate'].dt.dayofyear
data.drop(['ActivityDate'], axis=1, inplace=True)

# Impute missing values for 'WeightKg' with the mean of non-missing values
data['WeightKg'] = data['WeightKg'].fillna(data['WeightKg'].mean())

# Impute missing values for other numerical columns with their mean values
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for column in numerical_cols:
    if column != 'WeightKg':  # Skip 'WeightKg' as we've already imputed it
        data[column] = data[column].fillna(data[column].mean())

# Check the remaining missing values
print(data.isnull().mean() * 100)
# Display the cleaned dataframe
print(data.info())
print(data.head())
print(data.dtypes)

# Save the engineered dataset
data.to_csv('cleaned_imputed.csv', index=False)
rows, columns = data.shape
print("Number of rows:", rows)
print("Number of columns:", columns)


# In[51]:


# Summary statistics for the cleaned data
print(data.describe())

# Visualizing distributions of key numerical features
columns_to_plot = ['TotalSteps', 'TotalDistance', 'Calories', 'WeightKg']

for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

# Heatmap of the correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Time series plot for 'TotalSteps'
plt.figure(figsize=(15, 7))
sns.lineplot(x='DayOfYear', y='TotalSteps', data=data, estimator='mean', ci=None)
plt.title('Total Steps Over Time')
plt.xlabel('Day of Year')
plt.ylabel('Total Steps')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

rows, columns = data.shape
print(data.shape)  # Print the shape to track changes
print("Number of rows:", rows)
print("Number of columns:", columns)


# In[53]:


# Feature 1: Sleep Efficiency (TotalMinutesAsleep / TotalTimeInBed)
# Sleep efficiency has been linked to sleep quality, which affects energy levels and fatigue (Bei, B., Wiley, J. F., Trinder, J., & Manber, R. (2016)).
data['SleepEfficiency'] = data['TotalMinutesAsleep'] / data['TotalTimeInBed']
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data['SleepEfficiency'].fillna(data['SleepEfficiency'].mean(), inplace=True)

# Feature 2: Total Active Minutes (VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes)
# Physical activity levels have been associated with better energy levels and reduced fatigue (Puetz, T. W. (2006)).
data['TotalActiveMinutes'] = data['VeryActiveMinutes'] + data['FairlyActiveMinutes'] + data['LightlyActiveMinutes']
data['InactiveMinutes'] = data['SedentaryMinutes'] - data['SedentaryActiveDistance']*60  # if the distance is in miles, and assuming 1 mile = 15 minutes walking

# Feature 3: Activity Diversity - Variance of different types of activities
# Diversity in physical activity has been linked to overall well-being and might influence energy levels (Saanijoki, T. et al. (2015)).
data['ActivityDiversity'] = data[['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes']].var(axis=1) #or std.

# Feature 4: Average Daily Heart Rate
# Average heart rate can reflect physical fitness levels and stress, which are related to fatigue (Thayer, J. F., Yamamoto, S. S., & Brosschot, J. F. (2010)).
# This assumes you have heart rate data available in your dataset.
data['AverageHeartRate'] = (data['Max'] + data['Min']) / 2
data['HRSpread'] = data['Max'] - data['Min']

# Feature 5: Sleep Quality Indicator
# Combining sleep efficiency and total sleep time to derive a sleep quality indicator, based on guidelines suggesting that both quantity and quality of sleep are important for reducing fatigue (Hirshkowitz, M. et al. (2015)).
data['SufficientSleep'] = ((data['TotalMinutesAsleep'] >= 420) & (data['SleepEfficiency'] > 0.85)).astype(int)

# Normalize or scale features
scaler = MinMaxScaler()
features_to_scale = ['TotalActiveMinutes', 'ActivityDiversity', 'SleepEfficiency', 'TotalMinutesAsleep', 'AverageHeartRate']
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
# Handling remaining missing values
data.dropna(inplace=True)
print(data.head())
# Display basic information about the DataFrame
print(data.info())
# Display summary statistics for numerical columns
print(data.describe())

print("Feature engineering completed and data saved.")

# Get the dimensions of the DataFrame
rows, columns = data.shape
print("Number of rows:", rows)
print("Number of columns:", columns)
data.to_csv('your_file.csv', index=False)  # Set index=False to avoid saving row numbers as a column


# In[29]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Selecting the relevant features for clustering
features_for_clustering = ['TotalSteps', 'TotalActiveMinutes', 'SedentaryMinutes', 'Calories',
                           'TotalMinutesAsleep', 'SleepEfficiency', 'AverageHeartRate', 
                           'ActivityDiversity', 'SufficientSleep', 'StepsChange', 'WeightChange']

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features_for_clustering])

# Applying KMeans clustering
sum_of_squared_distances = []
K = range(1, 15)  # Let's test for a range of K to find the optimal

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km = km.fit(scaled_features)
    sum_of_squared_distances.append(km.inertia_)

# Plotting the elbow plot
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[31]:


# Let's assume the elbow graph showed that the optimal number of clusters is 3
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_features)

# Assigning the labels to the dataframe
data['Cluster'] = kmeans.labels_

# Now we can look at the mean values of features within each cluster
for i in range(optimal_k):
    print(f"Cluster {i} averages:")
    print(data[data['Cluster'] == i][features_for_clustering].mean())
    print('\n')


# In[32]:


# Assign energy level labels based on cluster membership
data['EnergyLevel'] = data['Cluster'].map({
    0: 'Low Energy',
    1: 'High Energy',
    2: 'Balanced Energy',
    3: 'Variable Energy'
})

# Assign fatigue level labels based on cluster membership
data['FatigueLevel'] = data['Cluster'].map({
    0: 'High Fatigue',
    1: 'Moderate Fatigue',
    2: 'Low Fatigue',
    3: 'High Fatigue'
})
print(data.columns)

data.to_csv('final.csv', index=False) 


# In[ ]:





# In[34]:


from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Encode labels
label_encoder_energy = LabelEncoder()
label_encoder_fatigue = LabelEncoder()
data['EnergyLevelEncoded'] = label_encoder_energy.fit_transform(data['EnergyLevel'])
data['FatigueLevelEncoded'] = label_encoder_fatigue.fit_transform(data['FatigueLevel'])

# Prepare feature matrix X and target vectors y_energy, y_fatigue
X = data.drop(['EnergyLevel', 'FatigueLevel', 'EnergyLevelEncoded', 'FatigueLevelEncoded'], axis=1)
y_energy = data['EnergyLevelEncoded']
y_fatigue = data['FatigueLevelEncoded']

# Split the data for Energy Level prediction
X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(X, y_energy, test_size=0.3, random_state=42)

# Split the data for Fatigue Level prediction
X_train_fatigue, X_test_fatigue, y_train_fatigue, y_test_fatigue = train_test_split(X, y_fatigue, test_size=0.3, random_state=42)

# Now, you don't need to re-scale the splits or re-encode the labels

# Display the sizes of the splits for energy level and fatigue level prediction
print(f"Training set size (Energy Level): {X_train_energy.shape[0]} rows, {X_train_energy.shape[1]} columns")
print(f"Testing set size (Energy Level): {X_test_energy.shape[0]} rows, {X_test_energy.shape[1]} columns")
print(f"Training set size (Fatigue Level): {X_train_fatigue.shape[0]} rows, {X_train_fatigue.shape[1]} columns")
print(f"Testing set size (Fatigue Level): {X_test_fatigue.shape[0]} rows, {X_test_fatigue.shape[1]} columns")


# In[39]:


# Initialize and train Random Forest classifier for Energy Level prediction
rf_classifier_energy = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_energy.fit(X_train_energy, y_train_energy)

# Predict on the test set for Energy Level
rf_predictions_energy = rf_classifier_energy.predict(X_test_energy)

# Evaluate Random Forest classifier for Energy Level prediction
rf_accuracy_energy = accuracy_score(y_test_energy, rf_predictions_energy)
print("Random Forest Accuracy (Energy Level):", rf_accuracy_energy)

# Initialize and train Gradient Boosting classifier for Energy Level prediction
gb_classifier_energy = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_classifier_energy.fit(X_train_energy, y_train_energy)

# Predict on the test set for Energy Level
gb_predictions_energy = gb_classifier_energy.predict(X_test_energy)

# Evaluate Gradient Boosting classifier for Energy Level prediction
gb_accuracy_energy = accuracy_score(y_test_energy, gb_predictions_energy)
print("Gradient Boosting Accuracy (Energy Level):", gb_accuracy_energy)


# In[41]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming rf_predictions_energy and gb_predictions_energy are predictions made on the test set

# For the Random Forest classifier on Energy Level
cm_rf_energy = confusion_matrix(y_test_energy, rf_predictions_energy)  # Using y_test_energy directly
print("Confusion Matrix - Random Forest on Energy Level:")
print(cm_rf_energy)
sns.heatmap(cm_rf_energy, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest - Energy Level')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# For the Gradient Boosting classifier on Energy Level
cm_gb_energy = confusion_matrix(y_test_energy, gb_predictions_energy)  # Using y_test_energy directly
print("Confusion Matrix - Gradient Boosting on Energy Level:")
print(cm_gb_energy)
sns.heatmap(cm_gb_energy, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Gradient Boosting - Energy Level')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# In[42]:


feature_importances_energy_rf = rf_classifier_energy.feature_importances_
# Use X_train_energy's columns as they were directly used for training the RF model
features = X_train_energy.columns
importance_df_energy_rf = pd.DataFrame({
    'Feature': features, 
    'Importance': feature_importances_energy_rf
}).sort_values(by='Importance', ascending=False)
print("Feature Importances - Random Forest Classifier for Energy Level:")
print(importance_df_energy_rf)

feature_importances_gb_energy = gb_classifier_energy.feature_importances_
# Use the same feature set for the GB model as was used for the RF model
importance_df_gb_energy = pd.DataFrame({
    'Feature': features,  # Reusing features from X_train_energy as it's the same for GB
    'Importance': feature_importances_gb_energy
}).sort_values(by='Importance', ascending=False)
print("Feature Importances - Gradient Boosting Classifier for Energy Level:")
print(importance_df_gb_energy)


# In[43]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Assuming earlier steps have properly prepared and encoded the target, and scaled the features as needed.

# StratifiedKFold setup
strat_k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Random Forest Classifier for Energy Level
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=strat_k_fold, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_energy, y_train_energy)  # Use non-encoded labels directly if your labels were not explicitly encoded before splitting
print("Best parameters for RF (Energy):", grid_search_rf.best_params_)
print("Best score for RF (Energy):", grid_search_rf.best_score_)

# Gradient Boosting Classifier for Energy Level
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4],
    'min_samples_split': [2, 5]
}
gb_classifier = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(estimator=gb_classifier, param_grid=param_grid_gb, cv=strat_k_fold, scoring='accuracy', n_jobs=-1)
grid_search_gb.fit(X_train_energy, y_train_energy)  # Again, using the straightforward dataset and labels
print("Best parameters for GB (Energy):", grid_search_gb.best_params_)
print("Best score for GB (Energy):", grid_search_gb.best_score_)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




