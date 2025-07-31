# Heart-Disease-Prediction

****Overview****

This project is a web-based application built with Streamlit to predict the likelihood of heart disease based on patient attributes. It uses a Random Forest Classifier trained on a heart disease dataset (heart.csv) with 303 records and 13 features. The app allows users to input patient details (e.g., age, sex, cholesterol) and receive a prediction of heart disease risk along with the probability.

The project includes data exploration (via a Jupyter Notebook) and a deployable Streamlit app, demonstrating machine learning in healthcare.

****Features****





Interactive Interface: Users can input patient data through a form and get real-time predictions.



Machine Learning Model: Utilizes a Random Forest Classifier for accurate predictions.



Dataset: Based on heart.csv, containing features like Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, and HeartDisease (target).



Deployment: Designed for local execution or deployment on Streamlit Cloud.

****Repository Structure****





app.py: Main Streamlit application script.



heart.csv: Dataset used for training and predictions.



requirements.txt: List of Python dependencies.



Heart_disease_prediction (1).ipynb: Jupyter Notebook with data exploration and analysis.



rf_model.pkl (optional): Pre-trained Random Forest model.



.gitignore: Excludes unnecessary files (e.g., __pycache__).

****Installation****





Clone the Repository:

git clone https://github.com/yaswitha122/Heart Disease Prediction.git
cd <repo-name>



Set Up a Virtual Environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:

pip install -r requirements.txt



Run the App Locally:

streamlit run app.py





Open the provided URL (e.g., http://localhost:8501) in a browser.

****Usage****





Open the app in a browser (local or deployed).



Enter patient details in the form (e.g., Age, Sex, ChestPainType).



Click Predict to see the heart disease risk and probability.



Optionally, view the dataset preview in the app.

****Dataset****

The heart.csv dataset contains 303 records with the following features:





Age: Patient age (numeric).



Sex: Gender (M = male, F = female).



ChestPainType: Chest pain type (ATA, NAP, ASY, TA).



RestingBP: Resting blood pressure (mm Hg).



Cholesterol: Serum cholesterol (mm/dl).



FastingBS: Fasting blood sugar (>120 mg/dl: 1 = true, 0 = false).



RestingECG: Resting ECG results (Normal, ST, LVH).



MaxHR: Maximum heart rate achieved.



ExerciseAngina: Exercise-induced angina (Y = yes, N = no).



Oldpeak: ST depression induced by exercise.



ST_Slope: Slope of the peak exercise ST segment (Up, Flat, Down).



HeartDisease: Target (0 = no heart disease, 1 = heart disease).

****Model****





Algorithm: Random Forest Classifier (n_estimators=100, random_state=42).



Preprocessing: Categorical variables are one-hot encoded.



Training: The model is trained on 80% of the dataset, with 20% reserved for testing.



Output: Predicts binary outcome (0 or 1) and probability of heart disease.



****Future Improvements****





Add feature importance visualization.



Include model evaluation metrics (e.g., accuracy, precision, recall).



Support additional machine learning models for comparison.



Enhance UI with custom styling and visualizations.
