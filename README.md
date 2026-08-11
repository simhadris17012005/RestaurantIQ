# 🍽️ RestaurantIQ

A **production-inspired Restaurant Intelligence and Recommendation System** built using **Machine Learning**, **Natural Language Processing**, **Recommendation Systems**, and **Geographical Data Analysis**.

The project analyzes a dataset of **9,551 restaurants across 141 cities** and provides multiple machine learning capabilities including restaurant rating prediction, personalized restaurant recommendations, cuisine classification, and location-based analysis.

---

# 📌 Project Overview

Restaurants generate large amounts of structured and location-based data such as ratings, votes, cuisines, pricing, services, and geographical coordinates.

RestaurantIQ uses this information to build an end-to-end machine learning system capable of understanding restaurant characteristics, predicting ratings, recommending restaurants based on user preferences, classifying cuisines, and analyzing restaurant distribution across cities.

The project demonstrates how multiple Machine Learning and Data Science techniques can be combined into a single practical application.

---

# 🎯 Project Objectives

* Build an end-to-end Machine Learning project
* Analyze restaurant data from multiple cities
* Predict restaurant aggregate ratings
* Develop a content-based restaurant recommendation system
* Classify restaurants based on their primary cuisine
* Perform geographical restaurant analysis
* Analyze restaurant density and ratings by city
* Apply feature engineering and preprocessing
* Compare multiple machine learning algorithms
* Generate interactive geographical visualizations
* Organize reusable and modular ML pipelines

---

# ✨ Features

✅ Restaurant Rating Prediction

✅ Content-Based Restaurant Recommendation

✅ Cuisine Classification

✅ Geographical Restaurant Analysis

✅ TF-IDF Feature Extraction

✅ Cosine Similarity Recommendation

✅ Random Forest Regression

✅ Decision Tree Regression

✅ Linear Regression

✅ Logistic Regression Classification

✅ Restaurant Density Analysis

✅ City-Level Rating Analysis

✅ Interactive Restaurant Map

✅ Feature Importance Analysis

✅ Data Cleaning & Preprocessing

✅ Modular Python Architecture

---

# 🛠 Technology Stack

## Programming Language

* Python 3.x

## Machine Learning

* Scikit-learn
* Random Forest
* Decision Tree
* Linear Regression
* Logistic Regression

## Natural Language Processing

* TF-IDF Vectorization
* Text Feature Engineering
* Cosine Similarity

## Data Processing

* Pandas
* NumPy

## Visualization

* Leaflet.js
* Interactive Maps
* Geographical Data Visualization

## Development Tools

* VS Code
* Git
* GitHub
* Jupyter Notebook

---

# 🧠 Machine Learning Concepts

This project implements:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Categorical Encoding
* Numerical Feature Processing
* Regression
* Classification
* Model Comparison
* Feature Importance Analysis
* TF-IDF Vectorization
* Cosine Similarity
* Content-Based Filtering
* Class Imbalance Analysis
* Geographical Data Analysis
* Model Evaluation

---

# 📂 Project Structure

```text
RestaurantIQ/
│
├── data/
│   └── restaurants.csv
│
├── src/
│   ├── preprocess.py
│   ├── rating_prediction.py
│   ├── recommender.py
│   ├── cuisine_classification.py
│   ├── location_analysis.py
│   └── run_all.py
│
├── outputs/
│   ├── models/
│   ├── maps/
│   └── csv_summaries/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

# 📊 Dataset

The project uses a restaurant dataset containing:

* **9,551 restaurants**
* **141 cities**
* Restaurant ratings
* Customer votes
* Cuisine information
* Average cost
* Price range
* Table booking availability
* Online delivery availability
* Restaurant locations
* Latitude and longitude
* City information

The dataset provides both numerical, categorical, textual, and geographical features for machine learning analysis.

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <your-repository-url>
```

---

## Navigate to Project

```bash
cd RestaurantIQ
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run Complete Project

Navigate to the source directory:

```bash
cd src
```

Run all four modules:

```bash
python run_all.py
```

This executes:

```text
Rating Prediction
        │
        ▼
Recommendation Engine
        │
        ▼
Cuisine Classification
        │
        ▼
Location Analysis
```

---

# ▶ Run Individual Modules

## Rating Prediction

```bash
python rating_prediction.py
```

## Restaurant Recommendation

```bash
python recommender.py
```

## Cuisine Classification

```bash
python cuisine_classification.py
```

## Location Analysis

```bash
python location_analysis.py
```

---

# 🔄 System Workflow

```text
Restaurant Dataset

        │
        ▼

Data Preprocessing

        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
Rating Prediction       Recommendation Engine
        │                      │
        ▼                      ▼
Regression Models        TF-IDF + Cosine Similarity
        │                      │
        └──────────┬───────────┘
                   │
                   ▼
          Cuisine Classification
                   │
                   ▼
            Classification
                   │
                   ▼
          Geographical Analysis
                   │
                   ▼
        City Statistics + Map
```

---

# 📈 Task 1 — Rating Prediction

The first module predicts a restaurant's **Aggregate Rating** using features such as:

* Votes
* City
* Location
* Cuisine
* Average Cost
* Price Range
* Table Booking
* Online Delivery
* Other restaurant attributes

## Models Compared

| Model             |        R² |       MAE |
| ----------------- | --------: | --------: |
| Random Forest     | **0.933** | **0.222** |
| Decision Tree     |     0.919 |     0.242 |
| Linear Regression |     0.268 |     0.779 |

### 🏆 Best Model

**Random Forest**

The Random Forest model achieved an **R² score of 0.933** and an **MAE of 0.222**, significantly outperforming Linear Regression and slightly outperforming the Decision Tree.

### Key Insight

**Votes** is the dominant predictive feature, contributing approximately **94% of the model's feature importance**.

This indicates that customer engagement is strongly associated with rating predictability. Restaurants with more customer votes provide considerably more stable information for predicting aggregate ratings.

Location and cuisine contribute additional but comparatively smaller signals.

---

# 🍴 Task 2 — Restaurant Recommendation

RestaurantIQ implements a **content-based recommendation engine**.

The recommender creates a combined restaurant profile using:

* Cuisine
* City
* Price tier

These profiles are transformed into numerical representations using **TF-IDF Vectorization**.

Restaurant similarity is then calculated using:

**Cosine Similarity**

---

# 🧠 Recommendation Workflow

```text
User Preferences

      │
      ▼

Cuisine + City + Price Range

      │
      ▼

Restaurant Text Profiles

      │
      ▼

TF-IDF Vectorization

      │
      ▼

Cosine Similarity

      │
      ▼

Rank Restaurants

      │
      ▼

Top-N Recommendations
```

---

# 💡 Example

```python
recommend(
    "Italian",
    city="New Delhi",
    price_range=2,
    top_n=5
)
```

The system returns the most similar restaurants matching the user's cuisine, city, and price preferences.

---

# 🍜 Task 3 — Cuisine Classification

The third module predicts a restaurant's **primary cuisine** from non-cuisine signals.

The model uses features such as:

* Location
* City
* Average Cost
* Price Range
* Table Booking
* Online Delivery
* Rating behavior

The goal is to determine whether the available restaurant characteristics can provide enough information to infer its primary cuisine.

---

# 🤖 Models Compared

The project compares:

* Logistic Regression
* Random Forest

### 🏆 Best Model

**Random Forest**

The best model achieves approximately **45% accuracy across 10 cuisine classes**.

---

# 📊 Classification Analysis

Performance varies significantly between cuisine categories.

The model performs particularly well for the majority **North Indian** class, achieving approximately **0.83 recall**.

However, minority classes such as:

* South Indian
* Continental
* Other less frequent cuisines

are more frequently misclassified.

This behavior is largely explained by **class imbalance** within the dataset.

---

# 🚀 Possible Improvements

Cuisine classification could potentially be improved using:

* Class weighting
* SMOTE
* Oversampling minority classes
* Undersampling majority classes
* Cuisine-specific textual features
* More balanced training data
* Hyperparameter optimization

---

# 📍 Task 4 — Location-Based Analysis

The fourth module performs geographical analysis of restaurants across cities.

The analysis includes:

* Restaurant density
* Average restaurant rating
* Average cost
* Price range
* City-level comparisons
* Restaurant coordinates

The project also generates an interactive **Leaflet map** containing restaurant locations.

Generated map:

```text
outputs/restaurant_map.html
```

---

# 🗺️ Geographical Analysis Workflow

```text
Restaurant Coordinates

        │
        ▼

Latitude + Longitude

        │
        ▼

City-Level Aggregation

        │
        ├── Restaurant Count
        ├── Average Rating
        ├── Average Cost
        └── Price Range
                │
                ▼
        Interactive Leaflet Map
```

---

# 📊 Sample Finding

**New Delhi** contains approximately **4,758 restaurants**, making it by far the most densely represented city in the dataset.

However, **Gurgaon** achieves a higher average rating:

```text
New Delhi  → 2.81
Gurgaon    → 3.00
```

Despite having roughly five times fewer restaurants, Gurgaon has a slightly higher average rating.

This suggests that restaurant quantity and average customer rating do not necessarily increase together and may indicate a smaller, more curated restaurant market.

---

# 📁 Generated Outputs

Running the project generates analytical outputs inside:

```text
outputs/
```

These may include:

* Trained machine learning models
* Recommendation outputs
* Cuisine classification results
* City-level summary CSV files
* Feature importance results
* Interactive restaurant map

---

# 🧪 Model Evaluation

## Rating Prediction

Evaluation metrics:

* R² Score
* Mean Absolute Error (MAE)

Best result:

```text
Random Forest
R²  = 0.933
MAE = 0.222
```

## Cuisine Classification

Evaluation metrics:

* Accuracy
* Precision
* Recall
* Classification performance by class

Best result:

```text
Random Forest
Accuracy ≈ 45%
```

---

# 🔍 Key Insights

### 1. Customer Engagement Matters

Votes are the strongest predictor of restaurant ratings, accounting for approximately **94% of Random Forest feature importance**.

### 2. Restaurant Popularity Is Not the Same as Rating

New Delhi has substantially more restaurants than Gurgaon, but Gurgaon has a higher average rating.

### 3. Cuisine Classification Is Challenging

Without directly using cuisine information, predicting the primary cuisine is difficult, particularly for minority classes.

### 4. Class Imbalance Influences Performance

Majority cuisines receive significantly better classification performance than less frequent cuisines.

### 5. Restaurant Recommendations Can Be Preference-Driven

Combining cuisine, city, and price tier provides a simple but effective content-based recommendation strategy.

---

# 🧪 Testing

Individual modules can be tested by executing:

```bash
python rating_prediction.py
python recommender.py
python cuisine_classification.py
python location_analysis.py
```

For future development, automated unit and integration tests can be added under:

```text
tests/
```

---

# 🐳 Docker

Docker support can be added to package the complete RestaurantIQ environment.

Example workflow:

```bash
docker build -t restaurantiq .
```

Run:

```bash
docker run restaurantiq
```

---

# ☁ Deployment

The project can be extended and deployed using:

* Docker
* AWS
* Azure
* Google Cloud
* Render
* Railway

A future web application could expose the recommendation and prediction models through REST APIs.

---

# 🚀 Future Improvements

* Hyperparameter tuning
* SMOTE and class-weighted cuisine classification
* Deep Learning models
* Advanced recommendation algorithms
* Collaborative filtering
* Hybrid recommendation system
* Restaurant search API
* Interactive dashboard
* FastAPI backend
* React frontend
* User authentication
* Restaurant review sentiment analysis
* Real-time restaurant data
* Automated ML retraining
* CI/CD using GitHub Actions
* Dockerized production deployment
* Cloud deployment
* Advanced geographical clustering

---

# 📚 Learning Outcomes

This project demonstrates practical experience in:

* Machine Learning
* Regression
* Classification
* Recommendation Systems
* Natural Language Processing
* TF-IDF
* Cosine Similarity
* Feature Engineering
* Data Preprocessing
* Model Evaluation
* Class Imbalance Analysis
* Random Forest
* Decision Trees
* Logistic Regression
* Geographical Data Analysis
* Interactive Data Visualization
* Python
* Scikit-learn
* Pandas
* NumPy

---

# 🌐 Project Links

## GitHub

<your-github-repository-url>

## Scikit-learn

[Scikit-learn](https://scikit-learn.org/?utm_source=chatgpt.com)

## Pandas

[Pandas](https://pandas.pydata.org/?utm_source=chatgpt.com)

## Leaflet

[Leaflet.js](https://leafletjs.com/?utm_source=chatgpt.com)

---

# 🏷️ GitHub Topics

```text
restaurant-analysis
machine-learning
data-science
recommendation-system
content-based-filtering
tfidf
cosine-similarity
random-forest
decision-tree
logistic-regression
classification
regression
scikit-learn
pandas
numpy
nlp
data-visualization
geographical-analysis
leaflet
python
portfolio-project
```

---

# 👨‍💻 Author

**Simhadri Bhukya**

B.Tech – Computer Science Engineering

Machine Learning • AI • Data Science • Full Stack Development

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgements

Special thanks to the open-source community and the developers of:

* Scikit-learn
* Pandas
* NumPy
* Leaflet.js
* Python

---

⭐ If you found this project useful, consider giving it a star on GitHub!

