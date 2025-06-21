# Parkinson's Disease Detection from Voice Recordings

## Project Overview

This project aims to develop and evaluate machine learning models for the early detection of Parkinson's Disease (PD) based on various biomedical voice measurements. By analyzing vocal features, we seek to differentiate between healthy individuals and those affected by PD. This work leverages a publicly available dataset and employs several supervised learning algorithms to achieve high classification accuracy.

## Table of Contents

1.  [Problem Formulation](https://www.google.com/search?q=%23problem-formulation)
2.  [Dataset](https://www.google.com/search?q=%23dataset)
3.  [Key Features](https://www.google.com/search?q=%23key-features)
4.  [Machine Learning Approach](https://www.google.com/search?q=%23machine-learning-approach)
5.  [Evaluation Metrics](https://www.google.com/search?q=%23evaluation-metrics)
6.  [Project Structure and Code Execution](https://www.google.com/search?q=%23project-structure-and-code-execution)
7.  [Results](https://www.google.com/search?q=%23results)
8.  [Conclusion and Future Work](https://www.google.com/search?q=%23conclusion-and-future-work)
9.  [Requirements](https://www.google.com/search?q=%23requirements)
10. [License](https://www.google.com/search?q=%23license)
11. [Contact](https://www.google.com/search?q=%23contact)

## Problem Formulation

The core problem addressed is a **binary classification** task:

  * **Class 0:** Healthy individuals
  * **Class 1:** Individuals with Parkinson's Disease

The goal is to build a predictive model that can accurately assign an individual to one of these two classes based on their voice measurements.

## Dataset

  * **Name:** Parkinson's Disease Classification
  * **Source:** UCI Machine Learning Repository
  * **Link:** [https://archive.ics.uci.edu/dataset/174/parkinsons](https://archive.ics.uci.edu/dataset/174/parkinsons)
  * **Description:** The dataset comprises a range of 22 biomedical voice measurements collected from 195 individuals, 147 of whom have Parkinson's disease (status=1) and 48 are healthy (status=0). This dataset is known for its high discriminative power regarding PD.

## Key Features

The dataset includes various voice features, providing a comprehensive characterization of vocal quality. Some of the key features include:

  * `MDVP:Fo(Hz)`: Average vocal fundamental frequency
  * `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
  * `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency
  * `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`: Measures of vocal jitter (variations in fundamental frequency)
  * `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`: Measures of vocal shimmer (variations in amplitude)
  * `NHR`: Noise-to-Harmonic Ratio
  * `HNR`: Harmonic-to-Noise Ratio
  * `RPDE`: Recurrence Period Density Entropy (non-linear dynamic complexity)
  * `DFA`: Detrended Fluctuation Analysis (non-linear dynamic complexity)
  * `spread1`, `spread2`, `D2`, `PPE`: Various non-linear dynamic complexity measures

The `status` column serves as the target variable (0 or 1). The `name` column is an identifier and is excluded from the feature set.

## Machine Learning Approach

This project utilizes a supervised learning paradigm, exploring several classification algorithms:

  * **Support Vector Machine (SVM):** A powerful algorithm for classification, particularly effective in high-dimensional spaces.
  * **Random Forest Classifier:** An ensemble learning method that builds multiple decision trees and merges their predictions for improved accuracy and reduced overfitting.
  * **Gradient Boosting Classifier:** Another ensemble method that builds trees sequentially, with each new tree correcting errors of previous ones.
  * **Neural Networks (MLPClassifier):** A type of artificial neural network capable of learning complex non-linear relationships.

### Data Preprocessing Steps:

1.  **Feature and Target Separation:** The dataset is split into features (X) and the target variable (y).
2.  **Feature Scaling:** `StandardScaler` is applied to normalize the numerical features, which is crucial for algorithms like SVM and Neural Networks that are sensitive to feature scales. This is done within a `Pipeline` during hyperparameter tuning to prevent data leakage.
3.  **Train-Test Split:** The data is divided into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` to maintain the original class distribution in both subsets, addressing the dataset's class imbalance.
4.  **Hyperparameter Tuning:** `GridSearchCV` with `StratifiedKFold` cross-validation is used to find the optimal hyperparameters for the best-performing models (Gradient Boosting and Random Forest), ensuring robust model selection.
5.  **Cross-Validation:** Stratified K-Fold Cross-Validation (5-fold) is implemented to provide a more reliable estimate of model performance and generalization capability.

## Evaluation Metrics

Given the class imbalance in the dataset (approximately 75% Parkinson's, 25% Healthy), the following evaluation metrics are prioritized for a comprehensive assessment:

  * **Accuracy:** The proportion of correctly classified instances.
  * **Precision:** The proportion of positive identifications that were actually correct (minimizes False Positives).
  * **Recall (Sensitivity):** The proportion of actual positives that were correctly identified (minimizes False Negatives).
  * **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure.
  * **ROC AUC (Receiver Operating Characteristic Area Under the Curve):** A robust metric that measures the classifier's ability to distinguish between classes across various threshold settings, less sensitive to class imbalance.

## Project Structure and Code Execution

The project code is designed to be executed step-by-step within a Google Colab environment for ease of access to libraries and cloud resources.

The complete code is structured in sequential cells for clarity and progression:

1.  **Setup and Data Loading:** Imports libraries, downloads the dataset, and performs initial data inspection (`.head()`, `.info()`, `.describe()`).
2.  **Exploratory Data Analysis (EDA):** Drops the 'name' column, checks for missing values, analyzes target variable distribution (`.value_counts()`), visualizes feature distributions by status, and computes a correlation matrix and feature correlation with the target.
3.  **Data Preprocessing:** Separates X and y, applies `StandardScaler` (within a pipeline for tuning), and performs `train_test_split` with stratification.
4.  **Model Training and Evaluation:** Initializes, trains, and evaluates SVM, Random Forest, Gradient Boosting, and Neural Network models. Prints detailed metrics and confusion matrices for each. Compares model performance in a summary table and plots ROC curves.
5.  **Further Improvements (Hyperparameter Tuning & Cross-Validation):** Implements `GridSearchCV` with `Pipeline` and `StratifiedKFold` for tuning the best models (Gradient Boosting and Random Forest). Performs final evaluation of tuned models on the test set and calculates feature importances. Includes a brief discussion on addressing imbalance with SMOTE.

**To run this project:**

1.  Open a new Google Colab notebook.
2.  Copy and paste the code from each step into separate cells.
3.  Run the cells sequentially.

The dataset will be automatically downloaded from the UCI repository.

## Results

Initial model evaluation showed promising results across the board. After hyperparameter tuning with GridSearchCV and cross-validation, the **Gradient Boosting Classifier** and **Random Forest Classifier** demonstrated outstanding performance on the test set:

### Final Tuned Model Performance on Test Set

| Model             | Accuracy | Precision | Recall   | F1-Score | ROC AUC  |
| :---------------- | :------- | :-------- | :------- | :------- | :------- |
| **Tuned Gradient Boosting** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Tuned Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

Both models achieved perfect scores on the hold-out test set, including 100% precision, recall, and accuracy, resulting in zero false positives and zero false negatives.

### Confusion Matrices (Tuned Models)

**Tuned Gradient Boosting Classifier:**

```
[[10  0]
 [ 0 29]]
```

**Tuned Random Forest Classifier:**

```
[[10  0]
 [ 0 29]]
```

### Feature Importance (Top 10)

**Tuned Gradient Boosting Classifier:**
| Feature          | Importance |
| :--------------- | :--------- |
| PPE              | 0.389      |
| MDVP:Fo(Hz)      | 0.166      |
| MDVP:Fhi(Hz)     | 0.117      |
| RPDE             | 0.093      |
| spread2          | 0.053      |
| spread1          | 0.051      |
| MDVP:Shimmer(dB) | 0.034      |
| D2               | 0.027      |
| Shimmer:APQ5     | 0.018      |
| DFA              | 0.016      |

**Tuned Random Forest Classifier:**
| Feature        | Importance |
| :------------- | :--------- |
| PPE            | 0.151      |
| spread1        | 0.109      |
| MDVP:Fo(Hz)    | 0.079      |
| spread2        | 0.066      |
| MDVP:Flo(Hz)   | 0.064      |
| MDVP:Fhi(Hz)   | 0.050      |
| Jitter:DDP     | 0.044      |
| NHR            | 0.042      |
| MDVP:Jitter(Abs) | 0.040      |
| Shimmer:APQ5   | 0.039      |

**Key Insights from Feature Importance:**

  * **PPE (Pitch Period Entropy)** consistently emerged as the most critical feature across both top-performing models, highlighting the significance of vocal irregularity in PD detection.
  * Other important features include fundamental frequency measures (`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`) and various non-linear dynamic complexity measures (`spread1`, `spread2`, `RPDE`). These align with clinical observations of vocal changes in Parkinson's patients.

## Conclusion and Future Work

This project successfully demonstrates the high potential of machine learning, specifically Gradient Boosting and Random Forest classifiers, for accurately detecting Parkinson's Disease from biomedical voice measurements. The perfect scores on the test set are highly encouraging and underscore the discriminative power of the features in this dataset.

**Future work could include:**

  * **External Validation:** Testing the models on new, independent datasets from different sources to confirm their generalization capabilities beyond the UCI dataset.
  * **Clinical Integration:** Collaborating with medical professionals to explore how these models could be integrated into clinical practice as a screening or monitoring tool.
  * **Real-time Application:** Developing a system for real-time voice analysis and PD risk assessment.
  * **Dataset Expansion:** Investigating the impact of larger and more diverse datasets on model performance and robustness.
  * **Advanced Techniques for Imbalance:** While stratification was effective here, for more severe imbalance, exploring techniques like SMOTE within a cross-validation pipeline could be beneficial.
  * **Deep Learning Architectures:** For larger datasets, more complex deep learning models might offer further improvements.

## Requirements

The project uses standard Python libraries for data science and machine learning. All necessary libraries can be installed via pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn # imbalanced-learn for SMOTE if used
```

The code is designed for a Google Colab environment, which comes with most of these libraries pre-installed.

## License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

## Contact

For any questions or further discussion, please feel free to reach out.
