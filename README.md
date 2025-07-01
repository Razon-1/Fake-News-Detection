# Fake News Detection

This project implements a machine learning pipeline to detect fake news articles using a labeled dataset. The workflow includes data preprocessing, feature engineering, model training, evaluation, and visualization.

## Dataset
- **Source:** [Kaggle - Fake News Detection](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)
- **Files:**
  - `Fake.csv`: Contains fake news articles
  - `True.csv`: Contains true news articles

## Features
- Data cleaning and preprocessing
- Feature extraction using TF-IDF
- Feature selection with chi-squared test
- Model training with multiple classifiers:
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
- Ensemble learning with Voting Classifier
- Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)
- Visualization of results

## Files
- `main.ipynb`: Main Jupyter notebook with the full workflow
- `app.py`: (Optional) Script for deploying or running the model
- `train_model.py`: Script for training and saving the model
- `model.pkl`: Saved trained model
- `sample_news.csv`: Sample news for testing

## Usage
1. **Install dependencies:**
   - Python 3.7+
   - Install required packages:
     ```bash
     pip install pandas numpy scikit-learn matplotlib seaborn
     ```
2. **Run the notebook:**
   - Open `main.ipynb` in Jupyter or VS Code and run all cells.
3. **Train model via script (optional):**
   - Run `python train_model.py` to train and save the model.

## Results
- The notebook provides detailed evaluation metrics and visualizations for each model.
- Ensemble methods are used to improve performance.

## License
This project is for educational purposes. Please check the dataset's license for usage restrictions.
