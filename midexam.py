import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np



# Load Data
def load_data():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
    df['Height'] = df['Height'] / 100  # Convert cm to meters
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)  # Calculate BMI
    return df

df = load_data()

# Streamlit UI
# Sidebar
st.sidebar.markdown("## Obesity Data Analysis")
st.sidebar.markdown("### Subject : Probability and Statistics")

st.sidebar.markdown("""
1. **Janet Dewi Evangeline** | 001202400146 
2. **Mohammad Waarits Harahap** | 001202400025
3. **Navisa Ersa Sabina** | 001202400083 
""")
st.title("Obesity Data Analysis")
st.write("### Overview of the Dataset")
st.dataframe(df.head())

# Descriptive Statistics
st.write("### Descriptive Statistics")
st.write(df.describe())
st.write("Explanation: Descriptive statistics provide an overview of data distribution, including mean, standard deviation, and data range. The higher the standard deviation, the greater the variation from the mean.")

# BMI Distribution by Obesity Level
st.write("### BMI Distribution by Obesity Level")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df['NObeyesdad'], y=df['BMI'], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
st.write("Explanation: The boxplot shows BMI distribution across different obesity categories. The higher the obesity level, the higher the median BMI, with varying BMI ranges in each category. This occurs because obesity categories have a higher weight distribution compared to others.")

# Pie Chart for Obesity Category Distribution
st.write("### Obesity Category Distribution")
# Count category distribution
obesity_counts = df['NObeyesdad'].value_counts()
# Create pie chart
fig, ax = plt.subplots()
ax.pie(obesity_counts, labels=obesity_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
ax.axis("equal")  # To make pie chart a perfect circle
st.pyplot(fig)
# Explanation of categories
st.write("### Explanation of Obesity Categories")
st.write("""
The **NObeyesdad** column represents different obesity categories based on BMI:
1. **Underweight** (BMI < 18.5)
2. **Normal Weight** (BMI 18.5 - 24.9)
3. **Overweight Level I** (BMI 25 - 29.9)
4. **Overweight Level II** (BMI 30 - 34.9)
5. **Obesity Type I** (BMI 35 - 39.9)
6. **Obesity Type II or III** (BMI ≥ 40)
The pie chart shows the distribution of these categories in the dataset.
""")

# Scatter Plot: Weight vs BMI
st.write("### Scatter Plot: Weight vs BMI")
fig, ax = plt.subplots()
sns.scatterplot(x=df['Weight'], y=df['BMI'], hue=df['NObeyesdad'], palette='coolwarm', ax=ax)
st.pyplot(fig)
st.write("Explanation: The scatter plot shows a positive linear relationship between weight and BMI. The higher a person's weight, the higher their BMI. This is due to the BMI formula being directly dependent on weight.")




# Regression (Predicting BMI from Weight)
st.write("### Linear Regression: Predicting BMI from Weight")
X = df[['Weight']]
y = df['BMI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.6f}")

fig, ax = plt.subplots()
sns.scatterplot(x=X_test['Weight'], y=y_test, label="Actual")
sns.lineplot(x=X_test['Weight'], y=y_pred, color='red', label="Predicted")
st.pyplot(fig)
st.write("Explanation: Linear regression shows that weight strongly correlates with BMI. The higher a person's weight, the higher their BMI. This model predicts BMI based on weight with reasonable accuracy.")

# Clustering (K-Means) on BMI vs Age
st.write("### Clustering (K-Means) on BMI vs Age")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster_BMI_Age'] = kmeans.fit_predict(df[['BMI', 'Age']])

fig, ax = plt.subplots()
sns.scatterplot(x=df['Age'], y=df['BMI'], hue=df['Cluster_BMI_Age'], palette='viridis', ax=ax)
st.pyplot(fig)
st.write("Explanation: The clustering graph shows three main groups based on BMI and Age. The pattern suggests how BMI tends to change with age. Different clusters may indicate different obesity risk levels based on age groups.")

# Encoding categorical features
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder if needed later

# Selecting features
features = ['Weight', 'Height', 'Age', 'FAF', 'TUE', 'family_history_with_overweight', 'SMOKE', 'SCC', 'CH2O', 'FCVC']
X = df[features]
y = df['NObeyesdad']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.4f}")

# Feature Importance
st.write("### Random Forest (Sorted)")
feature_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
fig, ax = plt.subplots()
feature_importances.plot(kind='bar', ax=ax)
st.pyplot(fig)
st.write("Explanation: This chart shows how important each feature is in predicting obesity levels using the Random Forest model. Features with higher values have a greater impact on obesity predictions.")

# Feature Explanation
st.write("### Feature Explanation")
st.write("""
- **Weight & Height**: Weight in kg and height in meters.
- **FCVC** (Frequency of vegetable consumption (scale from 1 to 3).
- **Age**: A person’s age in years.
- **TUE (Time Using Technology)**: Time spent using technology devices (scale 0-3).
- **CH2O (Daily Water Intake)**: Amount of water consumed daily (scale 1-3).
- **FAF (Physical Activity Frequency)**: How often a person exercises (scale 0-3).
- **Family History with Overweight**: Whether there is a family history of obesity (yes/no).
- **SCC (Caloric Consumption Monitoring)**: Whether a person monitors their calorie intake (yes/no).
- **SMOKE**: Whether a person smokes (yes/no).
""")

#Heatmap Korelasi
st.write("### Feature Correlation Heatmap")
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)
st.write("Explanation: This heatmap shows the correlation between different features. A higher absolute value means a stronger relationship. For example, BMI is highly correlated with weight, while physical activity might have a negative correlation with obesity.")

# Empirical Rule Visualization
st.write("### Empirical Rule Visualization for BMI")

mu = df['BMI'].mean()
sigma = df['BMI'].std()

# Generate values for normal distribution
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

fig, ax = plt.subplots()
ax.plot(x, y, label='Normal Distribution', color='blue')

# Marking the Empirical Rule areas
for i, color in zip(range(1, 4), ['green', 'yellow', 'red']):
    ax.fill_between(x, y, where=((x >= mu - i*sigma) & (x <= mu + i*sigma)), color=color, alpha=0.3)

ax.axvline(mu, color='black', linestyle='dashed', label="Mean (μ)")
ax.axvline(mu - sigma, color='green', linestyle='dashed', label="±1σ")
ax.axvline(mu + sigma, color='green', linestyle='dashed')
ax.axvline(mu - 2*sigma, color='yellow', linestyle='dashed', label="±2σ")
ax.axvline(mu + 2*sigma, color='yellow', linestyle='dashed')
ax.axvline(mu - 3*sigma, color='red', linestyle='dashed', label="±3σ")
ax.axvline(mu + 3*sigma, color='red', linestyle='dashed')

ax.legend()
st.pyplot(fig)
st.write("Explanation: This graph shows the BMI distribution based on the empirical rule. Most data (68%) falls within one standard deviation from the mean, while only about 0.3% of data is beyond three standard deviations.")

# Z-Score Calculation
st.write("### Z-Score Calculation for Weight")

df['Weight_Z'] = (df['Weight'] - df['Weight'].mean()) / df['Weight'].std()
st.write(df[['Weight', 'Weight_Z']].head())

st.write("Explanation: The Z-Score shows how far a person’s weight is from the mean in standard deviation units. If the Z-Score is close to 0, it means the weight value is near the average.")




