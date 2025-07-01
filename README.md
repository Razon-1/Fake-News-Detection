# Fake News Detection

This project implements a comprehensive machine learning pipeline to detect fake news articles by leveraging natural language processing (NLP) techniques and multiple classifiers. The aim is to classify news articles as **fake** or **true** with high accuracy, using ensemble learning and thorough model evaluation.

## Dataset

- **Source:** [Kaggle - Fake News Detection Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)  
- **Files:**
  - `Fake.csv`: Contains labeled fake news articles
  - `True.csv`: Contains labeled true news articles

## Project Overview

The project performs the following steps:

1. **Data Loading & Merging:** Combine fake and true news datasets with proper labeling.  
2. **Data Cleaning & Preprocessing:**  
   - Remove null or duplicate records  
   - Text normalization: lowercasing, removing punctuation, stopwords, and stemming/lemmatization  
3. **Feature Engineering:**  
   - Convert text data into numerical features using TF-IDF Vectorizer  
   - Perform feature selection using the chi-squared test to retain the most relevant features  
4. **Model Training:**  
   - Train multiple classifiers including:  
     - Random Forest Classifier  
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)  
     - Decision Tree Classifier  
   - Use an ensemble Voting Classifier to combine predictions for improved performance  
5. **Model Evaluation:**  
   - Calculate accuracy, precision, recall, and F1-score for each model  
   - Generate confusion matrices and classification reports  
6. **Visualization:**  
   - Visualize model performance metrics  
   - Display word clouds and other relevant graphs to understand text distribution and model outcomes  

## File Structure

- `main.ipynb`  
  Complete Jupyter notebook with data preprocessing, feature engineering, model training, evaluation, and visualizations.

- `train_model.py`  
  Python script to train the model pipeline and save the best performing model (`model.pkl`).

- `app.py` *(optional)*  
  Flask-based web application script to deploy the saved model and perform real-time fake news prediction.

- `model.pkl`  
  Serialized saved model file for inference.

- `sample_news.csv`  
  Sample news data for testing or demonstration purposes.

## Installation & Setup

1. **Python Version:**  
   Ensure Python 3.7 or above is installed.

2. **Install Required Packages:**

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk flask
````

3. **Download Dataset:**
   Download and extract the `Fake.csv` and `True.csv` files from the Kaggle link above into the project directory.

## Usage

### Run Notebook

* Open `main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
* Execute cells sequentially to preprocess data, train models, and visualize results.

### Train Model Script

* To train the model and save it as `model.pkl`, run:

  ```bash
  python train_model.py
  ```

### Run Web App (Optional)

* Launch the Flask app for prediction interface:

  ```bash
  python app.py
  ```

* Access the web app via [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## Results Summary

* The ensemble Voting Classifier achieves the highest performance by combining strengths of individual models.
* Detailed metrics and confusion matrices for each model help identify strengths and weaknesses.
* Visualizations such as word clouds provide insight into frequent terms in fake vs. true news.

## License

This project is intended for educational and research purposes only. Please review the Kaggle dataset license before commercial or redistribution use.
