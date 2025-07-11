# ANN-Classification-Churn-Prediction
# Customer Churn Prediction with Artificial Neural Networks

This project predicts customer churn for a bank using an Artificial Neural Network (ANN). It includes data preprocessing, model training, hyperparameter tuning, and a Streamlit web app for interactive predictions.

## Project Structure

- [`app.py`](app.py): Streamlit web app for customer churn prediction.
- [`experiments.ipynb`](experiments.ipynb): Data preprocessing and ANN model training.
- [`hyperparametertuningann.ipynb`](hyperparametertuningann.ipynb): Hyperparameter tuning using grid search.
- [`prediction.ipynb`](prediction.ipynb): Example of making predictions with the trained model.
- [`Resources/Churn_Modelling.csv`](Resources/Churn_Modelling.csv): Dataset used for training and evaluation.
- `model.h5`, `regression_model.h5`: Saved Keras models.
- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl`: Saved encoders and scaler for consistent preprocessing.
- `logs/`, `regressionlogs/`: TensorBoard logs for model training visualization.

## Setup

1. **Install dependencies**
pip install -r requirements.txt

2. **Run the Streamlit app**


3. **Explore Notebooks**

- Use Jupyter or VS Code to open and run the notebooks for data exploration, training, and prediction.

## Usage

- The Streamlit app allows you to input customer details and predicts the probability of churn.
- Notebooks demonstrate data preprocessing, model training, hyperparameter tuning, and batch predictions.

## Model Training Workflow

1. **Data Preprocessing**  
- Drop irrelevant columns.
- Encode categorical variables (`Gender` with LabelEncoder, `Geography` with OneHotEncoder).
- Scale features using StandardScaler.

2. **Model Training**  
- Build and train an ANN using Keras.
- Use callbacks like EarlyStopping and TensorBoard for monitoring.

3. **Hyperparameter Tuning**  
- Use grid search to find the optimal number of layers and neurons.

4. **Saving Artifacts**  
- Save the trained model and preprocessing objects for consistent inference.

## Making Predictions

- The app and [`prediction.ipynb`](prediction.ipynb) show how to preprocess new data and make predictions using the saved model and encoders.

## References

- Dataset: [Churn_Modelling.csv](Resources/Churn_Modelling.csv)
- Main code: [`app.py`](app.py), [`experiments.ipynb`](experiments.ipynb), [`hyperparametertuningann.ipynb`](hyperparametertuningann.ipynb)

---

