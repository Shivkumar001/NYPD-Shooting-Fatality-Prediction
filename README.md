# NYPD Shooting Incident Fatality Prediction

> Predicting whether an NYPD shooting incident results in fatality using Machine Learning, SMOTE, and SHAP Explainability.

## 📌 Project Overview

Gun violence is a critical public safety challenge in New York City. This project explores whether machine learning can predict whether a reported shooting incident will result in a fatality based on information available about the incident.

The project uses historical NYPD shooting incident data and follows an end-to-end machine learning workflow:

**Data → EDA → Preprocessing → Feature Engineering → SMOTE → Model Training → Model Evaluation → SHAP Explainability**

The final model selected for the project is **LightGBM**, which achieved the strongest overall performance among the five evaluated classifiers.

---

## 🎯 Objectives

* Perform exploratory data analysis on historical NYPD shooting incidents.
* Identify temporal, spatial, and demographic patterns associated with fatality.
* Engineer meaningful features from incident date, time, location, and demographic information.
* Handle class imbalance using **SMOTE**.
* Train and compare multiple classification algorithms.
* Evaluate models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
* Explain model predictions using **SHAP (SHapley Additive exPlanations)**.
* Select a final model based on predictive performance.

---

## 📊 Dataset

The project uses the **NYPD Historic Shooting Incident Dataset**.

### Dataset Summary

| Metric                  |                 Value |
| ----------------------- | --------------------: |
| Total incidents         |                27,312 |
| Original columns        |                    21 |
| Final modeling features |                    17 |
| Time period             |             2006–2022 |
| Fatal incidents         |                 5,266 |
| Non-fatal incidents     |                22,046 |
| Fatality rate           |                 19.3% |
| Task                    | Binary Classification |

The target variable is:

`STATISTICAL_MURDER_FLAG`

* `True` → Fatal
* `False` → Non-Fatal

The dataset contains temporal, spatial, perpetrator, and victim-related information. The accompanying data dictionary documents fields including incident date/time, borough, precinct, jurisdiction, perpetrator demographics, victim demographics, and geographic coordinates.

---

## 🔎 Exploratory Data Analysis

The analysis examined shooting incidents across multiple dimensions.

### Geographic Patterns

Location-related variables were among the strongest predictors of fatality. Latitude, longitude, precinct, and borough captured important geographic differences in incident outcomes.

Staten Island had the lowest number of incidents but the highest observed fatality rate at **20.9%**, while Manhattan had the lowest borough-level fatality rate at **17.6%**.

### Temporal Patterns

The analysis identified strong temporal patterns:

* Late-night incidents between **10 PM and 5 AM** showed higher fatality risk.
* **11 PM** appeared as the peak incident hour.
* Shooting incidents declined substantially from 2013–2019.
* A significant increase occurred in 2020.

### Victim Demographics

Fatality rates increased with victim age:

| Victim Age Group | Fatality Rate |
| ---------------- | ------------: |
| <18              |         13.0% |
| 18–24            |         16.7% |
| 25–44            |         21.8% |
| 45–64            |         25.0% |
| 65+              |         30.9% |

The analysis found that victims aged 65+ had substantially higher observed fatality rates than younger victims.

---

## 🧹 Data Preprocessing

Several preprocessing steps were performed before model training.

### Missing Values

Columns with extremely high missingness were removed:

* `LOC_OF_OCCUR_DESC`
* `LOC_CLASSFCTN_DESC`
* `LOCATION_DESC`

Perpetrator demographic missing values were retained as an explicit:

`UNKNOWN`

category because the absence of identified perpetrator information can itself contain useful information.

Small amounts of missing geographic/jurisdiction information were filled using appropriate imputation.

### Feature Engineering

New features were created from the original incident information:

* `HOUR`
* `YEAR`
* `MONTH`
* `IS_NIGHT`
* `IS_WEEKEND`
* `IS_NYPD_JURISDICTION`

Categorical variables were encoded for machine learning.

`StandardScaler` was used for feature scaling, with the scaler fitted on the training data before transforming the training and test sets.

---

## ⚖️ Class Imbalance & SMOTE

The target variable was imbalanced:

* **80.7% Non-Fatal**
* **19.3% Fatal**

This represents approximately a **4.2:1 imbalance ratio**.

A model predicting every incident as non-fatal could achieve approximately 80.7% accuracy while completely failing to identify fatal incidents.

To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data.

### Before SMOTE

* 21,848 training samples
* Fatality proportion: 19.3%

### After SMOTE

* 35,270 training samples
* 17,635 Non-Fatal
* 17,635 Fatal

Importantly, SMOTE was applied **only to the training data**. The test set retained the real-world class distribution for evaluation.

---

## 🤖 Model Comparison

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Linear SVM
5. LightGBM

### Performance Comparison

| Model               |   Accuracy |  Precision | Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | -----: | ---------: | ---------: |
| **LightGBM**        | **0.8812** | **0.6941** | 0.5940 | **0.6403** | **0.9114** |
| Random Forest       |     0.8756 |     0.6820 | 0.5640 |     0.6173 |     0.8920 |
| Decision Tree       |     0.8501 |     0.6105 | 0.5820 |     0.5959 |     0.8402 |
| SVM                 |     0.8290 |     0.5802 | 0.5950 |     0.5875 |     0.8610 |
| Logistic Regression |     0.7940 |     0.4710 | 0.6120 |     0.5322 |     0.8240 |

### 🏆 Final Model: LightGBM

LightGBM achieved the highest:

* **F1 Score:** 0.6403
* **ROC-AUC:** 0.9114
* **Accuracy:** 88.12%
* **Precision:** 69.41%

It was therefore selected as the final model for the project.

---

## 📈 LightGBM Evaluation

### Final Metrics

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 88.12% |
| Precision | 69.41% |
| Recall    | 59.40% |
| F1 Score  | 64.03% |
| ROC-AUC   | 0.9114 |

### Confusion Matrix

|                  | Predicted Non-Fatal | Predicted Fatal |
| ---------------- | ------------------: | --------------: |
| Actual Non-Fatal |               4,206 |             249 |
| Actual Fatal     |                 427 |             581 |

The ROC-AUC of **0.9114** indicates strong discrimination between fatal and non-fatal incidents.

Because false negatives represent missed fatal incidents, the project also considers the precision-recall trade-off and the possibility of lowering the classification threshold when prioritizing recall.

---

## 🔍 SHAP Explainability

Model explainability was performed using **SHAP** to understand which features influenced the model's predictions.

### Most Important Features

The strongest built-in feature importance signals included:

1. Latitude
2. Longitude
3. Precinct
4. Hour
5. Victim Age Group
6. Year
7. Month
8. Perpetrator Age Group
9. Borough
10. Victim Race

### Key SHAP Insights

**Location** was the strongest predictor. Geographic variables such as latitude, longitude, and precinct captured substantial variation in predicted fatality risk.

**Incident hour** was also important, with late-night incidents contributing strongly to the model's predictions.

**Victim age** showed a meaningful relationship with predicted fatality risk, with older age groups associated with higher observed fatality rates.

SHAP analysis was used to move beyond simply knowing which features were important and investigate how individual features contributed toward or away from a prediction.

---

## 🧠 Model Artifacts

The repository includes the trained model and preprocessing artifacts:

```text
label_encoders.pkl
lgbm_fatality_model.pkl
standard_scaler.pkl
```

These artifacts correspond to the preprocessing and final LightGBM model used in the project.

---

## 📁 Repository Structure

```text
NYPD-Shooting-Fatality-Prediction/
│
├── data/
│   └── NYPD_Shooting_Incident_Data__Historic_.csv
│
├── models/
│   ├── label_encoders.pkl
│   ├── lgbm_fatality_model.pkl
│   └── standard_scaler.pkl
│
├── notebooks/
│   └── Shooting_Incident_Fatality.ipynb
│
├── docs/
│   └── NYPD_Shooting_Fatality_Presentation.pptx
│
├── .gitignore
├── README.md
├── requirements.txt
└── dictionary.txt
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* LightGBM
* SHAP
* Jupyter Notebook
* Joblib

---

## 📌 Key Takeaways

* Shooting fatality is strongly associated with **location, time, and victim characteristics**.
* The dataset contains a significant class imbalance, making accuracy alone an insufficient evaluation metric.
* **SMOTE applied only to the training data** helped the models learn the minority fatal class.
* **LightGBM** provided the strongest overall predictive performance.
* **SHAP** provided model-level and individual-prediction explainability.
* Geographic features were among the strongest signals identified by both model importance and SHAP analysis.

---

## ⚠️ Important Note

This project is an analytical and machine learning exercise using historical data. It should **not be interpreted as a system for making real-world law-enforcement decisions**.

Predictions from historical data can reflect biases, missing information, and historical patterns in the underlying dataset. Any real-world deployment would require extensive validation, fairness assessment, monitoring, domain-expert review, and appropriate safeguards.

---

## 👤 Author

**Shiv Kumar**

Data Analyst 
