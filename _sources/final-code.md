# Final Code


```python
FINAL CODE
```


```python
### Step 1: Load and Explore Dataset
We load the Wine Quality dataset and display the first few rows to understand the structure.
```


```python
import requests
import os

# Create the data folder if it doesn't exist
data_folder = "../data"
os.makedirs(data_folder, exist_ok=True)

# Dataset URL and output file path
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
output_file = os.path.join(data_folder, "winequality-red.csv")

# Download and save the dataset
response = requests.get(url)
with open(output_file, 'wb') as file:
    file.write(response.content)

print(f"Dataset downloaded and saved at {output_file}")
```

    Dataset downloaded and saved at ../data/winequality-red.csv



```python
!pip install pandas seaborn matplotlib numpy
```

    Requirement already satisfied: pandas in /opt/anaconda3/lib/python3.12/site-packages (2.2.2)
    Requirement already satisfied: seaborn in /opt/anaconda3/lib/python3.12/site-packages (0.13.2)
    Requirement already satisfied: matplotlib in /opt/anaconda3/lib/python3.12/site-packages (3.9.2)
    Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.12/site-packages (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2023.3)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (24.1)
    Requirement already satisfied: pillow>=8 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (10.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (3.1.2)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)



```python
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced visualizations
```


```python
import pandas as pd
print(f"Pandas version: {pd.__version__}")
```

    Pandas version: 2.2.2



```python
data = pd.read_csv("../data/winequality-red.csv", sep=";")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Step 2: Exploratory Data Analysis (EDA)
In this step, we analyze the dataset to identify missing values, understand the distribution of features, and explore relationships with the target variable `quality`.
```


```python
# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Summary statistics of numerical features
print("\nSummary Statistics:")
print(data.describe())

# Distribution of the target variable
sns.countplot(x="quality", data=data)
plt.title("Distribution of Wine Quality")
plt.show()

# Pairplot of selected features
selected_features = ["quality", "alcohol", "sulphates", "citric acid"]
sns.pairplot(data[selected_features], hue="quality", diag_kind="kde")
plt.title("Relationships Between Selected Features and Quality")
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

    Missing Values:
    fixed acidity           0
    volatile acidity        0
    citric acid             0
    residual sugar          0
    chlorides               0
    free sulfur dioxide     0
    total sulfur dioxide    0
    density                 0
    pH                      0
    sulphates               0
    alcohol                 0
    quality                 0
    dtype: int64
    
    Summary Statistics:
           fixed acidity  volatile acidity  citric acid  residual sugar  \
    count    1599.000000       1599.000000  1599.000000     1599.000000   
    mean        8.319637          0.527821     0.270976        2.538806   
    std         1.741096          0.179060     0.194801        1.409928   
    min         4.600000          0.120000     0.000000        0.900000   
    25%         7.100000          0.390000     0.090000        1.900000   
    50%         7.900000          0.520000     0.260000        2.200000   
    75%         9.200000          0.640000     0.420000        2.600000   
    max        15.900000          1.580000     1.000000       15.500000   
    
             chlorides  free sulfur dioxide  total sulfur dioxide      density  \
    count  1599.000000          1599.000000           1599.000000  1599.000000   
    mean      0.087467            15.874922             46.467792     0.996747   
    std       0.047065            10.460157             32.895324     0.001887   
    min       0.012000             1.000000              6.000000     0.990070   
    25%       0.070000             7.000000             22.000000     0.995600   
    50%       0.079000            14.000000             38.000000     0.996750   
    75%       0.090000            21.000000             62.000000     0.997835   
    max       0.611000            72.000000            289.000000     1.003690   
    
                    pH    sulphates      alcohol      quality  
    count  1599.000000  1599.000000  1599.000000  1599.000000  
    mean      3.311113     0.658149    10.422983     5.636023  
    std       0.154386     0.169507     1.065668     0.807569  
    min       2.740000     0.330000     8.400000     3.000000  
    25%       3.210000     0.550000     9.500000     5.000000  
    50%       3.310000     0.620000    10.200000     6.000000  
    75%       3.400000     0.730000    11.100000     6.000000  
    max       4.010000     2.000000    14.900000     8.000000  



    
![png](dataexplo_files/dataexplo_8_1.png)
    



    
![png](dataexplo_files/dataexplo_8_2.png)
    



    
![png](dataexplo_files/dataexplo_8_3.png)
    



```python
### Step 3: Simulated Database and SQL Operations
In this step, we simulate SQL operations by splitting the dataset into separate tables and performing joins.
```


```python
# Save a portion of the dataset as separate CSVs (simulating separate database tables)
data[['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol']].to_csv("../data/wine_features.csv", index=False)
data[['quality']].to_csv("../data/wine_labels.csv", index=False)

# Load and join the tables
wine_features = pd.read_csv("../data/wine_features.csv")
wine_labels = pd.read_csv("../data/wine_labels.csv")

# Simulate an SQL join
wine_data_joined = pd.concat([wine_features, wine_labels], axis=1)

# Display the joined data
print(wine_data_joined.head())
```

       fixed acidity  volatile acidity  citric acid  alcohol  quality
    0            7.4              0.70         0.00      9.4        5
    1            7.8              0.88         0.00      9.8        5
    2            7.8              0.76         0.04      9.8        5
    3           11.2              0.28         0.56      9.8        6
    4            7.4              0.70         0.00      9.4        5



```python
### Step 4: Data Preprocessing Using Pipelines
In this step, we preprocess the dataset by scaling the numerical features to standardize them.
```


```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define features and target
X = wine_data_joined.drop("quality", axis=1)
y = wine_data_joined["quality"]

# Define numerical features for scaling
numerical_features = X.columns.tolist()

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ]
)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Display the first few rows of the scaled training data
print("Scaled Training Data:")
print(X_train_scaled[:5])
```

    Scaled Training Data:
    [[ 0.21833164  0.88971201  0.19209222  1.12317723]
     [-1.29016623 -1.78878251  0.65275338  1.40827174]
     [ 1.49475291 -0.78434707  1.01104539 -0.58738978]
     [ 0.27635078  0.86181102 -0.06383064 -0.96751578]
     [ 0.04427419  2.81487994 -0.62686095 -0.49235828]]



```python
### Step 5: Model Training and Evaluation
In this step, we train a Random Forest classifier to predict wine quality and evaluate its performance.
```


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# Test set predictions
y_pred = model.predict(X_test_scaled)

# Classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))
```

    Cross-Validation Accuracy Scores: [0.62890625 0.67578125 0.6015625  0.640625   0.67058824]
    Mean Cross-Validation Accuracy: 0.6434926470588235
    
    Classification Report on Test Set:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         1
               4       0.50      0.10      0.17        10
               5       0.74      0.77      0.75       130
               6       0.66      0.67      0.66       132
               7       0.55      0.57      0.56        42
               8       0.00      0.00      0.00         5
    
        accuracy                           0.67       320
       macro avg       0.41      0.35      0.36       320
    weighted avg       0.66      0.67      0.66       320
    



```python
### Step 6.1: Save the Trained Model
We save the trained Random Forest model to the outputs folder for reuse.
```


```python
import pandas as pd

# Load the split datasets
wine_features = pd.read_csv("../data/wine_features.csv")
wine_labels = pd.read_csv("../data/wine_labels.csv")

# Simulate SQL join by combining the datasets
wine_data_joined = pd.concat([wine_features, wine_labels], axis=1)

# Display the first few rows to verify
print(wine_data_joined.head())
```

       fixed acidity  volatile acidity  citric acid  alcohol  quality
    0            7.4              0.70         0.00      9.4        5
    1            7.8              0.88         0.00      9.8        5
    2            7.8              0.76         0.04      9.8        5
    3           11.2              0.28         0.56      9.8        6
    4            7.4              0.70         0.00      9.4        5



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Split the dataset into features (X) and target (y)
X = wine_data_joined.drop("quality", axis=1)
y = wine_data_joined["quality"]

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)
    ]
)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

print("Model training complete.")
```

    Model training complete.



```python
import joblib

# Save the trained model
model_path = "../outputs/wine_quality_model.joblib"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")
```

    Model saved at ../outputs/wine_quality_model.joblib



```python
pip install mlflow scikit-learn numpy pandas matplotlib
```

    Collecting mlflow
      Downloading mlflow-2.19.0-py3-none-any.whl.metadata (30 kB)
    Requirement already satisfied: scikit-learn in /opt/anaconda3/lib/python3.12/site-packages (1.5.1)
    Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.12/site-packages (1.26.4)
    Requirement already satisfied: pandas in /opt/anaconda3/lib/python3.12/site-packages (2.2.2)
    Requirement already satisfied: matplotlib in /opt/anaconda3/lib/python3.12/site-packages (3.9.2)
    Collecting mlflow-skinny==2.19.0 (from mlflow)
      Downloading mlflow_skinny-2.19.0-py3-none-any.whl.metadata (31 kB)
    Requirement already satisfied: Flask<4 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.0.3)
    Requirement already satisfied: Jinja2<4,>=2.11 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.1.4)
    Requirement already satisfied: alembic!=1.10.0,<2 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (1.13.3)
    Collecting docker<8,>=4.0.0 (from mlflow)
      Downloading docker-7.1.0-py3-none-any.whl.metadata (3.8 kB)
    Collecting graphene<4 (from mlflow)
      Downloading graphene-3.4.3-py2.py3-none-any.whl.metadata (6.9 kB)
    Collecting gunicorn<24 (from mlflow)
      Downloading gunicorn-23.0.0-py3-none-any.whl.metadata (4.4 kB)
    Requirement already satisfied: markdown<4,>=3.3 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.4.1)
    Requirement already satisfied: pyarrow<19,>=4.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (16.1.0)
    Requirement already satisfied: scipy<2 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (1.13.1)
    Requirement already satisfied: sqlalchemy<3,>=1.4.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (2.0.34)
    Requirement already satisfied: cachetools<6,>=5.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (5.3.3)
    Requirement already satisfied: click<9,>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (8.1.7)
    Requirement already satisfied: cloudpickle<4 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (3.0.0)
    Collecting databricks-sdk<1,>=0.20.0 (from mlflow-skinny==2.19.0->mlflow)
      Downloading databricks_sdk-0.40.0-py3-none-any.whl.metadata (38 kB)
    Requirement already satisfied: gitpython<4,>=3.1.9 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (3.1.43)
    Requirement already satisfied: importlib_metadata!=4.7.0,<9,>=3.7.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (7.0.1)
    Collecting opentelemetry-api<3,>=1.9.0 (from mlflow-skinny==2.19.0->mlflow)
      Downloading opentelemetry_api-1.29.0-py3-none-any.whl.metadata (1.4 kB)
    Collecting opentelemetry-sdk<3,>=1.9.0 (from mlflow-skinny==2.19.0->mlflow)
      Downloading opentelemetry_sdk-1.29.0-py3-none-any.whl.metadata (1.5 kB)
    Requirement already satisfied: packaging<25 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (24.1)
    Requirement already satisfied: protobuf<6,>=3.12.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (4.25.3)
    Requirement already satisfied: pyyaml<7,>=5.1 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (6.0.1)
    Requirement already satisfied: requests<3,>=2.17.3 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (2.32.3)
    Collecting sqlparse<1,>=0.4.0 (from mlflow-skinny==2.19.0->mlflow)
      Downloading sqlparse-0.5.3-py3-none-any.whl.metadata (3.9 kB)
    Requirement already satisfied: joblib>=1.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (3.5.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2023.3)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.4.4)
    Requirement already satisfied: pillow>=8 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (10.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (3.1.2)
    Requirement already satisfied: Mako in /opt/anaconda3/lib/python3.12/site-packages (from alembic!=1.10.0,<2->mlflow) (1.2.3)
    Requirement already satisfied: typing-extensions>=4 in /opt/anaconda3/lib/python3.12/site-packages (from alembic!=1.10.0,<2->mlflow) (4.11.0)
    Requirement already satisfied: urllib3>=1.26.0 in /opt/anaconda3/lib/python3.12/site-packages (from docker<8,>=4.0.0->mlflow) (2.2.3)
    Requirement already satisfied: Werkzeug>=3.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from Flask<4->mlflow) (3.0.3)
    Requirement already satisfied: itsdangerous>=2.1.2 in /opt/anaconda3/lib/python3.12/site-packages (from Flask<4->mlflow) (2.2.0)
    Requirement already satisfied: blinker>=1.6.2 in /opt/anaconda3/lib/python3.12/site-packages (from Flask<4->mlflow) (1.6.2)
    Collecting graphql-core<3.3,>=3.1 (from graphene<4->mlflow)
      Downloading graphql_core-3.2.5-py3-none-any.whl.metadata (10 kB)
    Collecting graphql-relay<3.3,>=3.1 (from graphene<4->mlflow)
      Downloading graphql_relay-3.2.0-py3-none-any.whl.metadata (12 kB)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/lib/python3.12/site-packages (from Jinja2<4,>=2.11->mlflow) (2.1.3)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Collecting google-auth~=2.0 (from databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow)
      Downloading google_auth-2.37.0-py2.py3-none-any.whl.metadata (4.8 kB)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitpython<4,>=3.1.9->mlflow-skinny==2.19.0->mlflow) (4.0.7)
    Requirement already satisfied: zipp>=0.5 in /opt/anaconda3/lib/python3.12/site-packages (from importlib_metadata!=4.7.0,<9,>=3.7.0->mlflow-skinny==2.19.0->mlflow) (3.17.0)
    Collecting deprecated>=1.2.6 (from opentelemetry-api<3,>=1.9.0->mlflow-skinny==2.19.0->mlflow)
      Downloading Deprecated-1.2.15-py2.py3-none-any.whl.metadata (5.5 kB)
    Collecting opentelemetry-semantic-conventions==0.50b0 (from opentelemetry-sdk<3,>=1.9.0->mlflow-skinny==2.19.0->mlflow)
      Downloading opentelemetry_semantic_conventions-0.50b0-py3-none-any.whl.metadata (2.3 kB)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.17.3->mlflow-skinny==2.19.0->mlflow) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.17.3->mlflow-skinny==2.19.0->mlflow) (3.7)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.17.3->mlflow-skinny==2.19.0->mlflow) (2024.12.14)
    Requirement already satisfied: wrapt<2,>=1.10 in /opt/anaconda3/lib/python3.12/site-packages (from deprecated>=1.2.6->opentelemetry-api<3,>=1.9.0->mlflow-skinny==2.19.0->mlflow) (1.14.1)
    Requirement already satisfied: smmap<5,>=3.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython<4,>=3.1.9->mlflow-skinny==2.19.0->mlflow) (4.0.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/anaconda3/lib/python3.12/site-packages (from google-auth~=2.0->databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow) (0.2.8)
    Collecting rsa<5,>=3.1.4 (from google-auth~=2.0->databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow)
      Downloading rsa-4.9-py3-none-any.whl.metadata (4.2 kB)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /opt/anaconda3/lib/python3.12/site-packages (from pyasn1-modules>=0.2.1->google-auth~=2.0->databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow) (0.4.8)
    Downloading mlflow-2.19.0-py3-none-any.whl (27.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.4/27.4 MB[0m [31m32.7 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading mlflow_skinny-2.19.0-py3-none-any.whl (5.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.9/5.9 MB[0m [31m29.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading docker-7.1.0-py3-none-any.whl (147 kB)
    Downloading graphene-3.4.3-py2.py3-none-any.whl (114 kB)
    Downloading gunicorn-23.0.0-py3-none-any.whl (85 kB)
    Downloading databricks_sdk-0.40.0-py3-none-any.whl (629 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m629.7/629.7 kB[0m [31m18.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading graphql_core-3.2.5-py3-none-any.whl (203 kB)
    Downloading graphql_relay-3.2.0-py3-none-any.whl (16 kB)
    Downloading opentelemetry_api-1.29.0-py3-none-any.whl (64 kB)
    Downloading opentelemetry_sdk-1.29.0-py3-none-any.whl (118 kB)
    Downloading opentelemetry_semantic_conventions-0.50b0-py3-none-any.whl (166 kB)
    Downloading sqlparse-0.5.3-py3-none-any.whl (44 kB)
    Downloading Deprecated-1.2.15-py2.py3-none-any.whl (9.9 kB)
    Downloading google_auth-2.37.0-py2.py3-none-any.whl (209 kB)
    Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Installing collected packages: sqlparse, rsa, gunicorn, graphql-core, deprecated, opentelemetry-api, graphql-relay, google-auth, docker, opentelemetry-semantic-conventions, graphene, databricks-sdk, opentelemetry-sdk, mlflow-skinny, mlflow
    Successfully installed databricks-sdk-0.40.0 deprecated-1.2.15 docker-7.1.0 google-auth-2.37.0 graphene-3.4.3 graphql-core-3.2.5 graphql-relay-3.2.0 gunicorn-23.0.0 mlflow-2.19.0 mlflow-skinny-2.19.0 opentelemetry-api-1.29.0 opentelemetry-sdk-1.29.0 opentelemetry-semantic-conventions-0.50b0 rsa-4.9 sqlparse-0.5.3
    Note: you may need to restart the kernel to use updated packages.



```python
pip install xgboost scikit-learn mlflow pandas numpy
```

    Collecting xgboost
      Downloading xgboost-2.1.3-py3-none-macosx_12_0_arm64.whl.metadata (2.1 kB)
    Requirement already satisfied: scikit-learn in /opt/anaconda3/lib/python3.12/site-packages (1.5.1)
    Requirement already satisfied: mlflow in /opt/anaconda3/lib/python3.12/site-packages (2.19.0)
    Requirement already satisfied: pandas in /opt/anaconda3/lib/python3.12/site-packages (2.2.2)
    Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.12/site-packages (1.26.4)
    Requirement already satisfied: scipy in /opt/anaconda3/lib/python3.12/site-packages (from xgboost) (1.13.1)
    Requirement already satisfied: joblib>=1.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (3.5.0)
    Requirement already satisfied: mlflow-skinny==2.19.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (2.19.0)
    Requirement already satisfied: Flask<4 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.0.3)
    Requirement already satisfied: Jinja2<4,>=2.11 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.1.4)
    Requirement already satisfied: alembic!=1.10.0,<2 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (1.13.3)
    Requirement already satisfied: docker<8,>=4.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (7.1.0)
    Requirement already satisfied: graphene<4 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.4.3)
    Requirement already satisfied: gunicorn<24 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (23.0.0)
    Requirement already satisfied: markdown<4,>=3.3 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.4.1)
    Requirement already satisfied: matplotlib<4 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (3.9.2)
    Requirement already satisfied: pyarrow<19,>=4.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (16.1.0)
    Requirement already satisfied: sqlalchemy<3,>=1.4.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow) (2.0.34)
    Requirement already satisfied: cachetools<6,>=5.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (5.3.3)
    Requirement already satisfied: click<9,>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (8.1.7)
    Requirement already satisfied: cloudpickle<4 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (3.0.0)
    Requirement already satisfied: databricks-sdk<1,>=0.20.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (0.40.0)
    Requirement already satisfied: gitpython<4,>=3.1.9 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (3.1.43)
    Requirement already satisfied: importlib_metadata!=4.7.0,<9,>=3.7.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (7.0.1)
    Requirement already satisfied: opentelemetry-api<3,>=1.9.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (1.29.0)
    Requirement already satisfied: opentelemetry-sdk<3,>=1.9.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (1.29.0)
    Requirement already satisfied: packaging<25 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (24.1)
    Requirement already satisfied: protobuf<6,>=3.12.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (4.25.3)
    Requirement already satisfied: pyyaml<7,>=5.1 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (6.0.1)
    Requirement already satisfied: requests<3,>=2.17.3 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (2.32.3)
    Requirement already satisfied: sqlparse<1,>=0.4.0 in /opt/anaconda3/lib/python3.12/site-packages (from mlflow-skinny==2.19.0->mlflow) (0.5.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2023.3)
    Requirement already satisfied: Mako in /opt/anaconda3/lib/python3.12/site-packages (from alembic!=1.10.0,<2->mlflow) (1.2.3)
    Requirement already satisfied: typing-extensions>=4 in /opt/anaconda3/lib/python3.12/site-packages (from alembic!=1.10.0,<2->mlflow) (4.11.0)
    Requirement already satisfied: urllib3>=1.26.0 in /opt/anaconda3/lib/python3.12/site-packages (from docker<8,>=4.0.0->mlflow) (2.2.3)
    Requirement already satisfied: Werkzeug>=3.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from Flask<4->mlflow) (3.0.3)
    Requirement already satisfied: itsdangerous>=2.1.2 in /opt/anaconda3/lib/python3.12/site-packages (from Flask<4->mlflow) (2.2.0)
    Requirement already satisfied: blinker>=1.6.2 in /opt/anaconda3/lib/python3.12/site-packages (from Flask<4->mlflow) (1.6.2)
    Requirement already satisfied: graphql-core<3.3,>=3.1 in /opt/anaconda3/lib/python3.12/site-packages (from graphene<4->mlflow) (3.2.5)
    Requirement already satisfied: graphql-relay<3.3,>=3.1 in /opt/anaconda3/lib/python3.12/site-packages (from graphene<4->mlflow) (3.2.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/lib/python3.12/site-packages (from Jinja2<4,>=2.11->mlflow) (2.1.3)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib<4->mlflow) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib<4->mlflow) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib<4->mlflow) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib<4->mlflow) (1.4.4)
    Requirement already satisfied: pillow>=8 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib<4->mlflow) (10.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib<4->mlflow) (3.1.2)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Requirement already satisfied: google-auth~=2.0 in /opt/anaconda3/lib/python3.12/site-packages (from databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow) (2.37.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitpython<4,>=3.1.9->mlflow-skinny==2.19.0->mlflow) (4.0.7)
    Requirement already satisfied: zipp>=0.5 in /opt/anaconda3/lib/python3.12/site-packages (from importlib_metadata!=4.7.0,<9,>=3.7.0->mlflow-skinny==2.19.0->mlflow) (3.17.0)
    Requirement already satisfied: deprecated>=1.2.6 in /opt/anaconda3/lib/python3.12/site-packages (from opentelemetry-api<3,>=1.9.0->mlflow-skinny==2.19.0->mlflow) (1.2.15)
    Requirement already satisfied: opentelemetry-semantic-conventions==0.50b0 in /opt/anaconda3/lib/python3.12/site-packages (from opentelemetry-sdk<3,>=1.9.0->mlflow-skinny==2.19.0->mlflow) (0.50b0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.17.3->mlflow-skinny==2.19.0->mlflow) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.17.3->mlflow-skinny==2.19.0->mlflow) (3.7)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.17.3->mlflow-skinny==2.19.0->mlflow) (2024.12.14)
    Requirement already satisfied: wrapt<2,>=1.10 in /opt/anaconda3/lib/python3.12/site-packages (from deprecated>=1.2.6->opentelemetry-api<3,>=1.9.0->mlflow-skinny==2.19.0->mlflow) (1.14.1)
    Requirement already satisfied: smmap<5,>=3.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython<4,>=3.1.9->mlflow-skinny==2.19.0->mlflow) (4.0.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/anaconda3/lib/python3.12/site-packages (from google-auth~=2.0->databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in /opt/anaconda3/lib/python3.12/site-packages (from google-auth~=2.0->databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow) (4.9)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /opt/anaconda3/lib/python3.12/site-packages (from pyasn1-modules>=0.2.1->google-auth~=2.0->databricks-sdk<1,>=0.20.0->mlflow-skinny==2.19.0->mlflow) (0.4.8)
    Downloading xgboost-2.1.3-py3-none-macosx_12_0_arm64.whl (1.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m17.8 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: xgboost
    Successfully installed xgboost-2.1.3
    Note: you may need to restart the kernel to use updated packages.



```python
import pandas as pd

# Load the dataset using the correct absolute path
dataset_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
data = pd.read_csv(dataset_path)

# Print columns for verification
print("Columns in dataset:", data.columns)
```

    Columns in dataset: Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol'],
          dtype='object')



```python
import pandas as pd

# Paths to the datasets
features_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
labels_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_labels.csv"

# Load features and labels
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Check the structure of the labels file
print("Labels columns:", labels.columns)

# Ensure there is a column to merge on (e.g., 'id')
if 'id' in features.columns and 'id' in labels.columns:
    data = features.merge(labels, on='id')
else:
    # If there's no 'id' column, concatenate the files assuming correct alignment
    data = pd.concat([features, labels], axis=1)

# Verify the final dataset
print("Final dataset columns:", data.columns)
```

    Labels columns: Index(['quality'], dtype='object')
    Final dataset columns: Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')



```python
if 'quality' not in data.columns:
    raise ValueError("The target column 'quality' is missing in the dataset.")
```


```python
import pandas as pd
import mlflow

# Paths to the datasets
features_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
labels_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_labels.csv"

# Load features and labels
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Check for an ID column for merging
if "id" in features.columns and "id" in labels.columns:
    data = features.merge(labels, on="id")
else:
    # Concatenate datasets assuming alignment
    data = pd.concat([features, labels], axis=1)

print("Final dataset columns:", data.columns)

# Ensure the target column exists
if "quality" not in data.columns:
    raise ValueError(f"The target column 'quality' is missing in the dataset. Columns found: {data.columns}")
```

    Final dataset columns: Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')



```python
import pandas as pd

# Paths to the datasets
features_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
labels_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_labels.csv"

# Load features and labels
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Check if a common column exists for merging (e.g., 'id')
if "id" in features.columns and "id" in labels.columns:
    data = features.merge(labels, on="id")
else:
    # If no common column, assume rows are aligned and concatenate
    data = pd.concat([features, labels], axis=1)

# Verify the dataset
print("Final dataset columns:", data.columns)

# Ensure the target column is present
if "quality" not in data.columns:
    raise ValueError("The target column 'quality' is missing in the dataset.")

# Define features and target
X = data.drop("quality", axis=1)
y = data["quality"]
```

    Final dataset columns: Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')



```python
import pandas as pd

# Paths to your datasets
features_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
labels_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_labels.csv"

# Load datasets
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Check the structure of the files
print("Features columns:", features.columns)
print("Labels columns:", labels.columns)

# Merge or concatenate datasets to include the 'quality' column
if "id" in features.columns and "id" in labels.columns:
    data = features.merge(labels, on="id")  # Merge if a common column exists
else:
    # Assuming aligned rows, concatenate features and labels
    data = pd.concat([features, labels], axis=1)

# Check final structure
print("Final dataset columns:", data.columns)

# Ensure the 'quality' column exists
if "quality" not in data.columns:
    raise ValueError("The target column 'quality' is missing in the dataset.")

# Split features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Print confirmation
print("Data preparation complete!")
```

    Features columns: Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol'],
          dtype='object')
    Labels columns: Index(['quality'], dtype='object')
    Final dataset columns: Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')
    Data preparation complete!



```python
pip install pandas sqlite3
```

    Requirement already satisfied: pandas in /opt/anaconda3/lib/python3.12/site-packages (2.2.2)
    [31mERROR: Could not find a version that satisfies the requirement sqlite3 (from versions: none)[0m[31m
    [0m[31mERROR: No matching distribution found for sqlite3[0m[31m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
import sqlite3

# Connect to the SQLite database
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
conn = sqlite3.connect(db_path)

# Check the schema of the 'wine_features' table
cursor = conn.execute("PRAGMA table_info(wine_features)")
print("Schema of 'wine_features':")
for column in cursor.fetchall():
    print(column)

# Drop the table if needed
drop_table = input("Do you want to drop the 'wine_features' table? (yes/no): ").strip().lower()
if drop_table == "yes":
    conn.execute("DROP TABLE IF EXISTS wine_features")
    print("Dropped the 'wine_features' table.")

conn.close()
```

    Schema of 'wine_features':


    Do you want to drop the 'wine_features' table? (yes/no):  yes


    Dropped the 'wine_features' table.



```python
import sqlite3
conn = sqlite3.connect('/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
conn.close()
```

    [('wine_features',), ('wine_quality',)]



```python
import sqlite3

# Path to your SQLite database
db_path = '/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db'

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

conn.close()
```

    Tables in the database: [('wine_features',), ('wine_quality',)]



```python
import sqlite3

# Path to your SQLite database
db_path = '/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db'

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check the schema of the table
cursor.execute("PRAGMA table_info(wine_features);")
schema = cursor.fetchall()
print("Schema of 'wine_features':")
for col in schema:
    print(col)

conn.close()
```

    Schema of 'wine_features':
    (0, 'id', 'INTEGER', 0, None, 1)
    (1, 'fixed_acidity', 'REAL', 0, None, 0)
    (2, 'volatile_acidity', 'REAL', 0, None, 0)
    (3, 'citric_acid', 'REAL', 0, None, 0)
    (4, 'residual_sugar', 'REAL', 0, None, 0)
    (5, 'chlorides', 'REAL', 0, None, 0)
    (6, 'free_sulfur_dioxide', 'REAL', 0, None, 0)
    (7, 'total_sulfur_dioxide', 'REAL', 0, None, 0)
    (8, 'density', 'REAL', 0, None, 0)
    (9, 'pH', 'REAL', 0, None, 0)
    (10, 'sulphates', 'REAL', 0, None, 0)
    (11, 'alcohol', 'REAL', 0, None, 0)



```python
import sqlite3

db_path = '/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db'
conn = sqlite3.connect(db_path)

query = "SELECT COUNT(*) FROM wine_features;"
cursor = conn.execute(query)
row_count = cursor.fetchone()[0]
print(f"Number of rows in 'wine_features': {row_count}")

conn.close()
```

    Number of rows in 'wine_features': 0



```python
import sqlite3

# Database path
db_path = '/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db'
conn = sqlite3.connect(db_path)

# Count rows
query = "SELECT COUNT(*) FROM wine_features;"
cursor = conn.execute(query)
row_count = cursor.fetchone()[0]
print(f"Number of rows in 'wine_features': {row_count}")

conn.close()
```

    Number of rows in 'wine_features': 0



```python
import sqlite3

def list_tables(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return tables
    except Exception as e:
        return f"Error: {e}"

# Check wine_data.db
db_path_data = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
print(f"Tables in {db_path_data}: {list_tables(db_path_data)}")

# Check wine_quality.db
db_path_quality = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db"
print(f"Tables in {db_path_quality}: {list_tables(db_path_quality)}")
```

    Tables in /Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db: [('wine_features',)]
    Tables in /Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_quality.db: [('wine_features',), ('wine_quality',)]



```python
import sqlite3
import pandas as pd

# Path to database
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"

# Connect to the database
conn = sqlite3.connect(db_path)

# Query to count rows in wine_features table
query = "SELECT COUNT(*) FROM wine_features;"
try:
    cursor = conn.cursor()
    cursor.execute(query)
    count = cursor.fetchone()[0]
    print(f"Number of rows in 'wine_features': {count}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
    conn.close()

```

    Number of rows in 'wine_features': 1599



```python
import sqlite3
import pandas as pd
from ydata_profiling import ProfileReport

# Path to SQLite database
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"

# Establish connection
conn = sqlite3.connect(db_path)

# SQL Query to fetch data
query = "SELECT * FROM wine_features;"

# Fetch data
data = pd.read_sql_query(query, conn)

# Debug: Check if the DataFrame is empty
print(data.head())
print(f"Number of rows in the DataFrame: {len(data)}")

# Generate profiling report if the DataFrame is not empty
if len(data) > 0:
    print("Generating data profile report...")
    profile = ProfileReport(data, title="Wine Features Profile Report", explorative=True)
    profile.to_file("wine_features_profile.html")
    print("Data profile report generated: wine_features_profile.html")
else:
    raise ValueError("DataFrame is empty. Please provide a non-empty DataFrame.")
```

       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    0            7.4              0.70         0.00             1.9      0.076   
    1            7.8              0.88         0.00             2.6      0.098   
    2            7.8              0.76         0.04             2.3      0.092   
    3           11.2              0.28         0.56             1.9      0.075   
    4            7.4              0.70         0.00             1.9      0.076   
    
       free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    0                 11.0                  34.0   0.9978  3.51       0.56   
    1                 25.0                  67.0   0.9968  3.20       0.68   
    2                 15.0                  54.0   0.9970  3.26       0.65   
    3                 17.0                  60.0   0.9980  3.16       0.58   
    4                 11.0                  34.0   0.9978  3.51       0.56   
    
       alcohol  
    0      9.4  
    1      9.8  
    2      9.8  
    3      9.8  
    4      9.4  
    Number of rows in the DataFrame: 1599
    Generating data profile report...



    Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]



    Generate report structure:   0%|          | 0/1 [00:00<?, ?it/s]



    Render HTML:   0%|          | 0/1 [00:00<?, ?it/s]



    Export report to file:   0%|          | 0/1 [00:00<?, ?it/s]


    Data profile report generated: wine_features_profile.html



```python

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File /opt/anaconda3/lib/python3.12/site-packages/pandas/core/indexes/base.py:3805, in Index.get_loc(self, key)
       3804 try:
    -> 3805     return self._engine.get_loc(casted_key)
       3806 except KeyError as err:


    File index.pyx:167, in pandas._libs.index.IndexEngine.get_loc()


    File index.pyx:196, in pandas._libs.index.IndexEngine.get_loc()


    File pandas/_libs/hashtable_class_helper.pxi:7081, in pandas._libs.hashtable.PyObjectHashTable.get_item()


    File pandas/_libs/hashtable_class_helper.pxi:7089, in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'quality'

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    Cell In[45], line 17
         14 conn.close()
         16 # Check the distribution of the 'quality' column
    ---> 17 quality_counts = data['quality'].value_counts().sort_index()
         19 # Display the distribution
         20 print("Quality Distribution:")


    File /opt/anaconda3/lib/python3.12/site-packages/pandas/core/frame.py:4102, in DataFrame.__getitem__(self, key)
       4100 if self.columns.nlevels > 1:
       4101     return self._getitem_multilevel(key)
    -> 4102 indexer = self.columns.get_loc(key)
       4103 if is_integer(indexer):
       4104     indexer = [indexer]


    File /opt/anaconda3/lib/python3.12/site-packages/pandas/core/indexes/base.py:3812, in Index.get_loc(self, key)
       3807     if isinstance(casted_key, slice) or (
       3808         isinstance(casted_key, abc.Iterable)
       3809         and any(isinstance(x, slice) for x in casted_key)
       3810     ):
       3811         raise InvalidIndexError(key)
    -> 3812     raise KeyError(key) from err
       3813 except TypeError:
       3814     # If we have a listlike key, _check_indexing_error will raise
       3815     #  InvalidIndexError. Otherwise we fall through and re-raise
       3816     #  the TypeError.
       3817     self._check_indexing_error(key)


    KeyError: 'quality'



```python
import sqlite3

# Connect to the SQLite database
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
conn = sqlite3.connect(db_path)

# Query to check schema
schema_query = "PRAGMA table_info(wine_features)"
schema = conn.execute(schema_query).fetchall()
conn.close()

# Print schema
print("Schema of 'wine_features':")
for col in schema:
    print(col)
```

    Schema of 'wine_features':
    (0, 'fixed acidity', 'REAL', 0, None, 0)
    (1, 'volatile acidity', 'REAL', 0, None, 0)
    (2, 'citric acid', 'REAL', 0, None, 0)
    (3, 'residual sugar', 'REAL', 0, None, 0)
    (4, 'chlorides', 'REAL', 0, None, 0)
    (5, 'free sulfur dioxide', 'REAL', 0, None, 0)
    (6, 'total sulfur dioxide', 'REAL', 0, None, 0)
    (7, 'density', 'REAL', 0, None, 0)
    (8, 'pH', 'REAL', 0, None, 0)
    (9, 'sulphates', 'REAL', 0, None, 0)
    (10, 'alcohol', 'REAL', 0, None, 0)



```python
import sqlite3
import pandas as pd

# Paths to the CSV files and database
features_csv_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
labels_csv_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_labels.csv"
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"

# Read the features and labels
features = pd.read_csv(features_csv_path)
labels = pd.read_csv(labels_csv_path)

# Merge features and labels into a single DataFrame
features["quality"] = labels["quality"]

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

# Drop the 'wine_features' table if it exists
conn.execute("DROP TABLE IF EXISTS wine_features")

# Write the merged DataFrame to the database
features.to_sql("wine_features", conn, if_exists="replace", index=False)

conn.close()
print("Normalization complete. 'wine_features' table updated with 'quality' column.")
```

    Normalization complete. 'wine_features' table updated with 'quality' column.



```python
conn = sqlite3.connect(db_path)

schema_query = "PRAGMA table_info(wine_features)"
schema = conn.execute(schema_query).fetchall()
conn.close()

print("Schema of 'wine_features' after update:")
for col in schema:
    print(col)
```

    Schema of 'wine_features' after update:
    (0, 'fixed acidity', 'REAL', 0, None, 0)
    (1, 'volatile acidity', 'REAL', 0, None, 0)
    (2, 'citric acid', 'REAL', 0, None, 0)
    (3, 'residual sugar', 'REAL', 0, None, 0)
    (4, 'chlorides', 'REAL', 0, None, 0)
    (5, 'free sulfur dioxide', 'REAL', 0, None, 0)
    (6, 'total sulfur dioxide', 'REAL', 0, None, 0)
    (7, 'density', 'REAL', 0, None, 0)
    (8, 'pH', 'REAL', 0, None, 0)
    (9, 'sulphates', 'REAL', 0, None, 0)
    (10, 'alcohol', 'REAL', 0, None, 0)
    (11, 'quality', 'INTEGER', 0, None, 0)



```python
import pandas as pd

conn = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT * FROM wine_features", conn)
conn.close()

# Check the distribution of the 'quality' column
quality_counts = data['quality'].value_counts().sort_index()
print("Quality Distribution:")
print(quality_counts)
```

    Quality Distribution:
    quality
    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: count, dtype: int64



```python
import pandas as pd
import sqlite3

# Connect to the SQLite database
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
conn = sqlite3.connect(db_path)

# Load data
data = pd.read_sql_query("SELECT * FROM wine_features", conn)
conn.close()

# Check the distribution of the 'quality' column
quality_counts = data['quality'].value_counts().sort_index()
print("Quality Distribution:")
print(quality_counts)
```

    Quality Distribution:
    quality
    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: count, dtype: int64



```python
from ydata_profiling import ProfileReport

# Generate the profile report
profile = ProfileReport(data, title="Wine Features with Quality Profile Report", explorative=True)
profile_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/notebooks/wine_features_with_quality_profile.html"
profile.to_file(profile_path)
print(f"Data profile report generated: {profile_path}")
```


    Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]



    Generate report structure:   0%|          | 0/1 [00:00<?, ?it/s]



    Render HTML:   0%|          | 0/1 [00:00<?, ?it/s]



    Export report to file:   0%|          | 0/1 [00:00<?, ?it/s]


    Data profile report generated: /Users/lohithvattikuti/Desktop/Wine_Quality_project/notebooks/wine_features_with_quality_profile.html



```python
from sklearn.model_selection import train_test_split

# Features and labels
X = data.drop(columns=["quality"])
y = data["quality"]

# Perform stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Stratified Train/Test split complete.")
```

    Stratified Train/Test split complete.



```python
quality_counts = data['quality'].value_counts().sort_index()
print(quality_counts)
```

    quality
    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: count, dtype: int64



```python
quality_proportions = quality_counts / quality_counts.sum()
print(quality_proportions)
```

    quality
    3    0.006254
    4    0.033146
    5    0.425891
    6    0.398999
    7    0.124453
    8    0.011257
    Name: count, dtype: float64



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train Distribution:\n", y_train.value_counts(normalize=True))
print("Test Distribution:\n", y_test.value_counts(normalize=True))
```

    Train Distribution:
     quality
    5    0.426114
    6    0.398749
    7    0.124316
    4    0.032838
    8    0.011728
    3    0.006255
    Name: proportion, dtype: float64
    Test Distribution:
     quality
    5    0.425000
    6    0.400000
    7    0.125000
    4    0.034375
    8    0.009375
    3    0.006250
    Name: proportion, dtype: float64



```python
import mlflow
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# List of missing experiment names
experiment_names = ["Experiment #3", "Experiment #4", "PCA - Experiment #5"]

# Check runs and metrics
for exp_name in experiment_names:
    print(f"Checking {exp_name}...")
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment:
        experiment_id = experiment.experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if not runs.empty:
            print(f"Found runs for {exp_name}:")
            for _, run in runs.iterrows():
                print(f"Run ID: {run['run_id']}")
                print("Metrics logged:")
                print(run.filter(like="metrics").dropna())  # Show all metrics
        else:
            print(f"No runs found for {exp_name}.")
    else:
        print(f"Experiment {exp_name} not found.")
```

    Checking Experiment #3...
    Experiment Experiment #3 not found.
    Checking Experiment #4...
    Experiment Experiment #4 not found.
    Checking PCA - Experiment #5...
    Experiment PCA - Experiment #5 not found.



```python

```

This script loads wine features and labels from CSV files, combines them into a single dataset, and stores the resulting normalized table into a SQLite database. It also provides an option to drop the existing table before writing new data.

---
# Data Normlisation and other Experiments

```python
import pandas as pd
import sqlite3

# Paths to the CSV files
features_file = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_features.csv"
labels_file = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_labels.csv"

# Load the CSV files
wine_features = pd.read_csv(features_file)
wine_labels = pd.read_csv(labels_file)

# Combine features and labels into a single DataFrame
if 'id' in wine_features.columns and 'id' in wine_labels.columns:
    wine_data = wine_features.merge(wine_labels, on="id")
else:
    wine_data = pd.concat([wine_features, wine_labels], axis=1)

print(f"Features columns: {wine_features.columns}")
print(f"Labels columns: {wine_labels.columns}")
print(f"Final dataset columns: {wine_data.columns}")

# Connect to the SQLite database
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
conn = sqlite3.connect(db_path)

# Optionally drop the table if it exists
drop_table = input("Do you want to drop the 'wine_features' table? (yes/no): ").strip().lower()
if drop_table == "yes":
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS wine_features")
    conn.commit()
    print("Dropped the 'wine_features' table.")

# Write the DataFrame to the SQLite database, replacing the table if it exists
wine_features.to_sql("wine_features", conn, if_exists="replace", index=False)
print("Data has been written to the 'wine_features' table in the SQLite database.")

# Fetch data back from the database and load it into a Pandas DataFrame
query = "SELECT * FROM wine_features;"
retrieved_data = pd.read_sql(query, conn)

print("Data fetched from the database:")
print(retrieved_data.head())

# Close the connection
conn.close()

# Experiments

## Description
This section contains the code implementation for Experiment #1 in the Wine Quality Prediction project. The experiment uses a Logistic Regression model, with data loaded from an SQLite database, and tracks the results using MLflow.


### Experiment #1: Logistic Regression

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import mlflow
import sqlite3

# Set the tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Experiment #1"
mlflow.set_experiment(experiment_name)

# Load dataset from SQLite database
db_path = "./data/wine_data.db"
conn = sqlite3.connect(db_path)
query = "SELECT * FROM wine_features"
data = pd.read_sql_query(query, conn)
conn.close()

# Split the dataset
X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline for Logistic Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Perform cross-validation
with mlflow.start_run(run_name="Logistic Regression - Experiment #1"):
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    mean_score = scores.mean()
    std_score = scores.std()

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log parameters, metrics, and artifacts
    mlflow.log_param("classifier", "Logistic Regression")
    mlflow.log_metric("CV Mean F1 Score", mean_score)
    mlflow.log_metric("CV Std F1 Score", std_score)
    mlflow.log_metric("Test F1 Score", f1)
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Mean CV F1-Score: {mean_score:.4f}, Std: {std_score:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")

# Experiment #2: Comparing Classifiers for Wine Quality Prediction

This script performs a comparative evaluation of four classifiers (Logistic Regression, Ridge Classifier, Random Forest, and XGBoost) for wine quality prediction. The experiment uses MLFlow for tracking metrics and SQLite for data management.

---

## Table of Contents

1. [Setup and Dataset Loading](#setup-and-dataset-loading)
2. [Data Preprocessing](#data-preprocessing)
3. [Experiment: Logistic Regression](#experiment-logistic-regression)
4. [Experiment: Ridge Classifier](#experiment-ridge-classifier)
5. [Experiment: Random Forest](#experiment-random-forest)
6. [Experiment: XGBoost Classifier](#experiment-xgboost-classifier)

---

## 1. Setup and Dataset Loading

```python
import sqlite3  # Ensure sqlite3 is imported
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

# Set the MLFlow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Load the dataset
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
table_name = "wine_features"

# Load data directly from the database
conn = sqlite3.connect(db_path)  # Ensure sqlite3 is used here
data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
conn.close()

    

# Experiment #3: Feature Engineering and Logistic Regression

This script focuses on feature engineering to create new features and evaluates their impact on Logistic Regression performance for the wine quality prediction task. The experiment tracks metrics using MLFlow integrated with DagsHub.

---

## Table of Contents

1. [Setup and MLFlow Configuration](#setup-and-mlflow-configuration)
2. [Data Loading and Verification](#data-loading-and-verification)
3. [Feature Engineering](#feature-engineering)
4. [Model Training and Evaluation](#model-training-and-evaluation)

---

## 1. Setup and MLFlow Configuration

```python
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import pandas as pd
import sqlite3
import os

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set MLFlow tracking to DagsHub
DAGS_HUB_URL = "https://dagshub.com/vattikutilohith/wine_quality_project.mlflow"
mlflow.set_tracking_uri(DAGS_HUB_URL)
mlflow.sklearn.autolog()

# Database location and table
db_path = "data/wine_data.db"
table_name = "wine_features"

# Load data from the database
conn = sqlite3.connect(db_path)
dataset = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
conn.close()

# Verify the dataset schema
print("Schema of 'wine_features':", dataset.dtypes)
print("Dataset columns:", dataset.columns)

# Feature Engineering: Create new features by combining existing ones
dataset['density_pH_ratio'] = dataset['density'] / dataset['pH']
dataset['total_acidity'] = dataset['fixed acidity'] + dataset['volatile acidity']
dataset['sulfur_to_alcohol'] = dataset['total sulfur dioxide'] / dataset['alcohol']

# Define features and target
X = dataset.drop("quality", axis=1)
y = dataset["quality"]

# Normalize the target variable
y = y - y.min()  # Ensures classes start at 0

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline with Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

try:
    # Start MLFlow run for the experiment
    with mlflow.start_run(run_name="Feature Engineering - Experiment #3"):
        # Cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_macro')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Cross-Validation F1 Scores: {scores}")
        print(f"Mean CV F1-Score: {mean_score:.4f}, Std: {std_score:.4f}")

        # Fit on the full training data
        pipeline.fit(X_train, y_train)

        # Predict on test data
        y_pred = pipeline.predict(X_test)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

        # Log metrics in MLFlow
        mlflow.log_metric("Mean CV F1 Score", mean_score)
        mlflow.log_metric("CV Std F1 Score", std_score)
        mlflow.log_metric("Test F1 Score", f1_score(y_test, y_pred, average="macro"))
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        # Log confusion matrix metrics
        mlflow.log_metric("TP", cm[1][1] if cm.shape[0] > 1 else 0)
        mlflow.log_metric("TN", cm[0][0])
        mlflow.log_metric("FP", cm[0][1] if cm.shape[1] > 1 else 0)
        mlflow.log_metric("FN", cm[1][0] if cm.shape[0] > 1 else 0)

        print("Experiment #3 - Feature Engineering completed successfully.")

except mlflow.exceptions.MlflowException as e:
    print("Error occurred during the MLflow run. Please check your credentials or server status.")
    print("Error details:", str(e))


# Experiment #4: Feature Selection and Random Forest Classifier

This experiment demonstrates feature selection techniques using Variance Threshold and Random Forest importance. The selected features are used to train a Random Forest classifier. Metrics and the final model are tracked using MLFlow.

---

## Table of Contents

1. [Setup and MLFlow Configuration](#setup-and-mlflow-configuration)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Feature Selection](#feature-selection)
4. [Model Training and Evaluation](#model-training-and-evaluation)

---

## 1. Setup and MLFlow Configuration

```python
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import mlflow
import joblib
import sqlite3

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Feature Selection - Experiment #4"

# Create or set the experiment
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Load the dataset from SQLite database
db_path = "data/wine_data.db"
conn = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT * FROM wine_features", conn)
conn.close()

# Verify the final dataset
print("Final dataset columns:", data.columns)

if "quality" not in data.columns:
    raise ValueError("The target column 'quality' is missing in the dataset.")

X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize the target variable
y = y - y.min()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature selection and train the model
with mlflow.start_run(run_name="Feature Selection Experiment"):
    # Variance Threshold
    print("Applying Variance Threshold...")
    var_thresh = VarianceThreshold(threshold=0.01)
    X_train_var = var_thresh.fit_transform(X_train)
    
    # Feature Importance using Random Forest
    print("Selecting features using Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_var, y_train)
    selector = SelectFromModel(rf, prefit=True)
    X_train_selected = selector.transform(X_train_var)

    # Pipeline with selected features
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    
    # Cross-validation
    print("Performing cross-validation...")
    scores = cross_val_score(pipeline, X_train_selected, y_train, cv=5, scoring='f1_macro')
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Train and evaluate on test set
    pipeline.fit(X_train_selected, y_train)
    X_test_var = var_thresh.transform(X_test)
    X_test_selected = selector.transform(X_test_var)
    y_pred = pipeline.predict(X_test_selected)
    f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Log parameters and metrics to MLFlow
    mlflow.log_param("Variance Threshold", 0.01)
    mlflow.log_metric("CV Mean F1 Score", mean_score)
    mlflow.log_metric("CV Std F1 Score", std_score)
    mlflow.log_metric("Test F1 Score", f1)
    
    # Log confusion matrix
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    # Save the model
    output_model_path = "./outputs/feature_selected_model.jolib"
    os.makedirs("./outputs", exist_ok=True)
    joblib.dump(pipeline, output_model_path)
    mlflow.log_artifact(output_model_path, artifact_path="models")

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Cross-Validation Mean F1 Score: {mean_score:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Experiment '{experiment_name}' completed successfully.")

# Experiment #5: PCA Dimensionality Reduction and Random Forest Classifier

This experiment explores dimensionality reduction using PCA (Principal Component Analysis) and evaluates its impact on a Random Forest classifier. It includes visualizing the PCA results with a scree plot and tracks metrics using MLFlow.

---

## Table of Contents

1. [Setup and MLFlow Configuration](#setup-and-mlflow-configuration)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [PCA Analysis](#pca-analysis)
4. [Model Training and Evaluation](#model-training-and-evaluation)

---

## 1. Setup and MLFlow Configuration

```python
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt

# Set MLFlow credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set the tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Experiment name
experiment_name = "Experiment #5 - PCA Dimensionality Reduction"
mlflow.set_experiment(experiment_name)

# Load the dataset
features_path = "./data/wine_features.csv"
labels_path = "./data/wine_labels.csv"
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Combine features and labels
data = pd.concat([features, labels], axis=1)

# Separate features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize the target variable
y = y - y.min()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA analysis
pca = PCA()
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

# Scree plot
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid()
plt.savefig("./outputs/scree_plot.png")
print("Scree plot saved as 'scree_plot.png'.")

# Perform PCA with selected components
n_components = 8  # Adjust based on the scree plot
pca = PCA(n_components=n_components)

# Create a pipeline with PCA and Random Forest
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", pca),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Run the experiment
with mlflow.start_run(run_name="PCA Dimensionality Reduction - Random Forest"):
    # Cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    mean_cv_score = scores.mean()
    std_cv_score = scores.std()

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    test_f1_score = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log parameters, metrics, and artifacts
    mlflow.log_param("n_components", n_components)
    mlflow.log_metric("CV Mean F1 Score", mean_cv_score)
    mlflow.log_metric("CV Std F1 Score", std_cv_score)
    mlflow.log_metric("Test F1 Score", test_f1_score)
    mlflow.log_artifact("./outputs/scree_plot.png", artifact_path="plots")
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Mean CV F1-Score: {mean_cv_score:.4f}, Std: {std_cv_score:.4f}")
    print(f"Test F1-Score: {test_f1_score:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")


 # Experiment #6: Stacked Model with Random Forest, XGBoost, and Logistic Regression

This experiment implements a Stacking Classifier with Random Forest and XGBoost as base models and Logistic Regression as the meta-classifier. The performance is evaluated on the wine quality dataset using cross-validation and test metrics.

---

## Table of Contents

1. [Setup and MLFlow Configuration](#setup-and-mlflow-configuration)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Stacked Model Definition](#stacked-model-definition)
4. [Model Training and Evaluation](#model-training-and-evaluation)

---

## 1. Setup and MLFlow Configuration

```python
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import mlflow
import joblib

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set MLFlow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Custom Experiment - Stacked Model #6"
mlflow.set_experiment(experiment_name)

# Load the dataset
features_path = "./data/wine_features.csv"
labels_path = "./data/wine_labels.csv"
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Combine features and labels
data = pd.concat([features, labels], axis=1)

# Define features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize the target variable
y = y - y.min()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the base models
base_models = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("xgb", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss"))
]

# Define the meta-classifier
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Define the Stacked Classifier
stacked_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("stacked", StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5))
])

# Start the MLFlow run
with mlflow.start_run(run_name="Stacked Model - Experiment #6"):
    # Perform cross-validation
    scores = cross_val_score(stacked_pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Train the model
    stacked_pipeline.fit(X_train, y_train)

    # Test set evaluation
    y_pred = stacked_pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log metrics to MLFlow
    mlflow.log_param("Base Models", "Random Forest, XGBoost")
    mlflow.log_param("Meta Model", "Logistic Regression")
    mlflow.log_metric("CV Mean F1 Score", mean_score)
    mlflow.log_metric("CV Std F1 Score", std_score)
    mlflow.log_metric("Test F1 Score", test_f1)
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    # Save the model
    model_path = "./outputs/stacked_model.joblib"
    joblib.dump(stacked_pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Mean CV F1-Score: {mean_score:.4f}, Std: {std_score:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")

# Experiment #7: Hyperparameter Tuning of XGBoost Classifier

This experiment performs hyperparameter tuning for the XGBoost classifier using GridSearchCV. The best parameters are identified, and the model performance is evaluated using cross-validation and test metrics. Results and the final model are tracked using MLFlow.

---

## Table of Contents

1. [Setup and MLFlow Configuration](#setup-and-mlflow-configuration)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Hyperparameter Tuning with GridSearchCV](#hyperparameter-tuning-with-gridsearchcv)
4. [Model Evaluation and Logging](#model-evaluation-and-logging)

---

## 1. Setup and MLFlow Configuration

```python
import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"  # Token

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Custom Experiment - Hyperparameter Tuning #7"

# Create or set the experiment
mlflow.set_experiment(experiment_name)

# Load the dataset
features_path = "./data/wine_features.csv"
labels_path = "./data/wine_labels.csv"

# Load features and labels
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Combine features and labels
data = pd.concat([features, labels], axis=1)

# Define features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize target variable
y = y - y.min()  # Ensure classes start at 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost Classifier and parameter grid
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring="f1_macro", cv=5, verbose=1, n_jobs=-1)

# Start MLFlow run for the experiment
with mlflow.start_run(run_name="XGBoost Hyperparameter Tuning - Experiment #7"):
    # Fit GridSearchCV
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    mean_cv_score = grid_search.best_score_
    
    # Evaluate on the test set
    y_pred = best_model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("Mean CV F1 Score", mean_cv_score)
    mlflow.log_metric("Test F1 Score", test_f1)
    
    # Log confusion matrix
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")
    
    # Save and log the best model
    model_path = "./outputs/xgboost_best_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    print(f"Best Parameters: {best_params}")
    print(f"Mean CV F1 Score: {mean_cv_score:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")

                 



                 
# Save Best Model from MLFlow Experiments

This script fetches all experiments tracked in MLFlow, identifies the best run based on the `Test F1 Score`, and saves the corresponding model locally for further use.

---

## Table of Contents

1. [Setup and MLFlow Configuration](#setup-and-mlflow-configuration)
2. [Search for the Best Model](#search-for-the-best-model)
3. [Save the Best Model Locally](#save-the-best-model-locally)

---

## 1. Setup and MLFlow Configuration

```python
import mlflow
import joblib

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Initialize variables to track the best model
best_run = None
best_score = -1

# Fetch all experiments
experiments = mlflow.search_experiments()

for experiment in experiments:
    experiment_id = experiment.experiment_id
    print(f"Checking Experiment: {experiment.name}")
    try:
        # Search for runs in the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="metrics.Test F1 Score IS NOT NULL",  # Ensure metric exists
            order_by=["metrics.`Test F1 Score` DESC"]  # Use correct syntax for ordering
        )

        if not runs.empty:
            # Get the top run
            top_run = runs.iloc[0]
            score = top_run.get("metrics.Test F1 Score", None)

            if score and score > best_score:
                best_score = score
                best_run = top_run
    except Exception as e:
        print(f"Error processing experiment {experiment.name}: {e}")
# Check if a best run was found
if best_run is None:
    print("No valid runs found with Test F1 Score.")
else:
    print(f"Best run found: {best_run['run_id']} with Test F1 Score: {best_score}")

    # Fetch the logged model
    model_uri = f"runs:/{best_run['run_id']}/model"
    print(f"Best model URI: {model_uri}")

    # Load and save the model locally
    try:
        model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(model, "best_model.joblib")
        print("Best model saved as 'best_model.joblib'")
    except Exception as e:
        print(f"Error loading or saving the best model: {e}")


             

# F1 Scores Comparison Across Experiments

This script fetches experiment data from MLFlow, extracts key performance metrics (CV Mean F1 Score and Test F1 Score), and visualizes the results in a bar chart for comparison.

---

## Table of Contents

1. [Setup and Experiment Configuration](#setup-and-experiment-configuration)
2. [Fetch Data from MLFlow](#fetch-data-from-mlflow)
3. [Visualize F1 Scores](#visualize-f1-scores)

---

## 1. Setup and Experiment Configuration

```python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# List of experiment names
experiment_names = [
    "Experiment #1",
    "Experiment #2",
    "Experiment #3",
    "Experiment #4",
    "PCA - Experiment #5",
    "Custom Experiment - Stacked Model #6",
    "Custom Experiment - Hyperparameter Tuning #7"
]

# Initialize lists to store results
experiments = []
cv_means = []
test_f1s = []



                
                                                                                                                                                    # Model Deployment: Wine Quality Prediction API

This FastAPI application serves as an API for predicting wine quality using a stacked model. The API provides endpoints for both single and batch predictions based on input wine features.

---

## Table of Contents

1. [Setup and Model Loading](#setup-and-model-loading)
2. [API Endpoints](#api-endpoints)
    - [Health Check Endpoint](#health-check-endpoint)
    - [Single Prediction Endpoint](#single-prediction-endpoint)
    - [Batch Prediction Endpoint](#batch-prediction-endpoint)
3. [Input Schemas](#input-schemas)

---

## 1. Setup and Model Loading

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List

# Load the new stacked model (without xgboost)
model_path = "./outputs/stacked_model_no_xgboost.joblib"  # Updated model path
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="API to predict wine quality based on input features",
    version="1.0.0",
)
@app.get("/health")
def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "API is running!"}

@app.post("/predict")
def predict(features: WineFeatures):
    """
    Predict wine quality based on input features.
    """
    try:
        # Convert features to numpy array for prediction
        feature_values = np.array([
            features.fixed_acidity,
            features.volatile_acidity,
            features.citric_acid,
            features.residual_sugar,
            features.chlorides,
            features.free_sulfur_dioxide,
            features.total_sulfur_dioxide,
            features.density,
            features.pH,
            features.sulphates,
            features.alcohol,
        ]).reshape(1, -1)

        # Use the model to predict
        prediction = model.predict(feature_values)
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

@app.post("/predict_batch")
def predict_batch(batch: BatchWineFeatures):
    """
    Predict wine quality for a batch of inputs.
    """
    try:
        # Convert batch data into numpy array
        batch_features = np.array([
            [
                item.fixed_acidity,
                item.volatile_acidity,
                item.citric_acid,
                item.residual_sugar,
                item.chlorides,
                item.free_sulfur_dioxide,
                item.total_sulfur_dioxide,
                item.density,
                item.pH,
                item.sulphates,
                item.alcohol,
            ] for item in batch.data
        ])

        predictions = model.predict(batch_features)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch predictions: {e}")

class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., example=7.4)
    volatile_acidity: float = Field(..., example=0.7)
    citric_acid: float = Field(..., example=0.0)
    residual_sugar: float = Field(..., example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=11.0)
    total_sulfur_dioxide: float = Field(..., example=34.0)
    density: float = Field(..., example=0.9978)
    pH: float = Field(..., example=3.51)
    sulphates: float = Field(..., example=0.56)
    alcohol: float = Field(..., example=9.4)

    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        }
class BatchWineFeatures(BaseModel):
    data: List[WineFeatures]



# Streamlit App: Wine Quality Prediction

This Streamlit app provides an interactive interface to predict wine quality by communicating with the deployed FastAPI backend.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Input Features](#input-features)
3. [Prediction Workflow](#prediction-workflow)

---

## 1. Introduction

```python
import streamlit as st
import requests

# Streamlit app title
st.title("Wine Quality Prediction App")

# Description
st.write("""
This Streamlit app interacts with the deployed FastAPI backend to predict wine quality based on input features.
""")

# Input fields
st.header("Input Wine Features")
fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# Prediction button
if st.button("Predict Quality"):
    # Prepare the input data for API
    input_data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    # API URL
    api_url = "https://wine-quality-project.onrender.com/predict"

    # Make a POST request to the FastAPI backend
    try:
        response = requests.post(api_url, json=input_data)
        if response.status_code == 200:
            prediction = response.json().get("prediction", "No prediction returned")
            st.success(f"Predicted Wine Quality: {prediction}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")


       
                                                                                                                                                
                                                                                                                                                   
                                                                                                                                                
                                                                                                                                                


