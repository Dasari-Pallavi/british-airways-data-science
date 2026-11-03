# British Airways Data Science Job Simulation

This repository contains the complete portfolio deliverables developed for the British Airways Data Science virtual experience program on Forage. The project covers rule-based lounge capacity optimization, predictive booking behavior modeling, and customer sentiment analytics.

## 📊 Project Overview

| Task | Core Question | Data Science Method | Core Deliverable / Result |
| :--- | :--- | :--- | :--- |
| **1. Lounge Demand (Heathrow T3)** | How many customers will use the lounge? | Eligibility Matrix + Usage-Rate Lookup Table | Reusable capacity matrix saved to `outputs/lounge_lookup.csv` |
| **2. Booking Prediction** | Which customers will complete a holiday package booking? | Ensemble Random Forest Classifier | **Accuracy: 80.92%** <br> **ROC-AUC: 0.7412** |
| **3. Review Analysis** | What are the recurring operational complaints from passengers? | Text Preprocessing + Lexical VADER Sentiment Engine | Automated Text Analytics Dashboard & WordCloud Generation |

## 📈 Key Executive Findings
* **Extreme Target Skew:** Completed conversions account for only **14.96%** of total interactions. A basic model guessing "No" would yield an empty 85% accuracy. To correct this, the algorithm was trained with balanced class weights, optimizing the headline **ROC-AUC score to 0.7412**.
* **Top Purchase Drivers:** Feature importance vectors proved that `purchase_lead` (booking lead time), `flight_hour`, and `length_of_stay` are the dominant predictive metrics guiding conversion intent.
* **Top Complaint Themes:** Natural Language Processing (NLP) text classification isolated **Staff Interaction/Service Friction**, **Cancellations/Delays**, and **Cabin Seat Comfort** as the top recurring operational pain points within the negative feedback slice.

## 🛠️ System Tools & Framework Stack
* **Programming Language:** Python 3.10+
* **Data Processing Engineering:** Pandas, NumPy
* **Machine Learning Architectures:** Scikit-Learn (Ensemble Random Forest Classifiers)
* **Statistical Graphics Engines:** Matplotlib, Seaborn
* **Natural Language Text Processing (NLP):** NLTK Lexicon Sentiment Engine (VADER), WordCloud

## 📁 Repository Directory Structure
* `data/` : Raw CSV transactional matrices (`customer_booking.csv` and `reviews_data1.csv`). *Note: Excluded from version control via .gitignore to prevent data leakage.*
* `notebooks/` : Interactive step-by-step Jupyter Notebook logs:
  * `01_lounge_eligibility.ipynb` (Lounge capacity tracking code)
  * `02_booking_prediction.ipynb` (Machine learning pipeline code)
  * `03_review_analysis.ipynb` (NLP sentiment analysis code)
* `images/` : Exported diagnostic visual charts (`feature_importance.png`, `review_sentiment_distribution.png`, `customer_review_wordcloud.png`).
* `outputs/` : Final processed deliverables (`lounge_eligibility_lookup.csv`).

## 🖼️ Primary Project Visualizations
### Machine Learning Feature Importance Weights
![Feature Importance](../images/feature_importance.png)

### Customer Sentiment Text WordCloud
![Review WordCloud](../images/customer_review_wordcloud.png)

## 🚀 How to Run the Environment Locally
1. Clone this repository to your local directory machine:
   ```bash
   git clone https://github.com/Dasari-Pallavi/british-airways-data-science.git
   
   cd british-airways-data-science
   ```
2. Activate your isolated virtual environment and build requirements components:
   ```bash
   source venv/Scripts/activate  # On Windows Git Bash
   pip install -r requirements.txt
   ```
3. Boot up the analytical development space environment:
   ```bash
   jupyter notebook
   ```

## 📊 Data References & Licensing
* **Flight Schedules & Lounge Profiles:** Simulative mock structures derived from Forage virtual experience internship assets.
* **Customer Review Ingress Logs:** Scraped customer satisfaction corpus downloaded from the [British Airways Customer Reviews Dataset on Kaggle](https://kaggle.com). Licensed via Open Database License (ODbL).

## 🛠️ Operating Limitations & Next Optimization Steps
* **Lookup Calibrations:** Current lounge usage parameters are driven by strategic statistical assumptions. Integrating live physical barcode scanners at Heathrow terminal entry points would calibrate the model.
* **Minority Conversion Class Sensitivity:** Due to strict conversion boundaries, minority target recall rates present future space for model adjustment. Future optimization sprints will test Gradient Boosting methods (XGBoost/LightGBM) along with Hyperparameter Tuning strategies.

