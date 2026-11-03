# British Airways Data Science Job Simulation: Operational Analytics & Predictive Modeling

This repository contains the complete portfolio deliverables developed for the British Airways Data Science virtual experience program on Forage. The project covers rule-based lounge capacity optimization, predictive booking behavior modeling, and customer sentiment analytics.

## 📊 Model Performance Summary
* **Overall Classification Accuracy:** 80.92%
* **Area Under ROC Curve (ROC-AUC):** 0.7412
* **Core Insight:** `purchase_lead` (booking lead time) and `flight_hour` were identified as the strongest predictive indicators influencing final transaction completion metrics.

## 🛠️ System Tools & Framework Stack
* **Programming Language:** Python 3.10+
* **Data Processing Engineering:** Pandas, NumPy
* **Machine Learning Architectures:** Scikit-Learn (Ensemble Random Forest Classifiers)
* **Statistical Graphics Engines:** Matplotlib, Seaborn
* **Natural Language Text Processing (NLP):** NLTK Lexicon Sentiment Engine (VADER)

## 📁 Repository Directory Structure
* `/data` - Raw CSV transaction entries and customer text records.
* `/notebooks` - Step-by-step Jupyter Notebook records tracking tasks 1 through 3.
* `/images` - Exported analytical feature maps and sentiment plots.
* `/outputs` - Completed lounge demand reference lookup tables.

## 📈 Executive Tasks Executed

### Task 1: Lounge Capacity Optimization Logic (Heathrow T3)
* Programmed a modular operational lookup model framework linking client fare profiles and loyalty clusters directly to demand propagation indexes.
* Created a reusable lookup table containing clear rule-based business logic to help the Airport Planning team predict peak space constraints during scheduling intervals.

### Task 2: Holiday Booking Propensity Prediction Model
* Addressed severe minority classification skew parameters using a tuned Random Forest model running balanced class-weight distributions.
* Reached an **80.92% accuracy score** evaluated directly via ROC-AUC metrics over standard classification targets.

### Task 3: Sentiment Mining across 3,500+ Customer Reviews
* Implemented Lexical Valence Scoring architectures to process unstructured sentiment profiles.
* Successfully categorized customer reviews into distinct Positive, Neutral, and Negative sentiments, profiling major operational pain points such as flight delays and luggage mishandling.

