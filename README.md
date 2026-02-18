🏠 Hyderabad House Price Prediction
This project is a Machine Learning application designed to predict house rental prices in Hyderabad based on various features such as area, number of bedrooms, and locality. It includes a complete pipeline from data preprocessing and model training to a web-based user interface.

🚀 Live Demo
The application is built using Streamlit, providing an interactive web interface for users to input house details and receive instant price predictions.

📂 Project Structure
app.py: The Streamlit web application script.

data.ipynb: Jupyter notebook containing Data Cleaning, Exploratory Data Analysis (EDA), and Model Training.

Hyderabad_House_Data.csv: The raw dataset used for training.

house_price_model.pkl: The trained Linear Regression model.

scaler.pkl: The StandardScaler object used for feature scaling.

model_columns.pkl: A saved list of the feature columns used during training to ensure consistent input data.

📊 Workflow
Data Preprocessing:

Handled missing values in features like Washrooms, Tennants, and Area.

Converted categorical data into numerical formats using techniques like pd.to_numeric and One-Hot Encoding (get_dummies).

Scaled numerical features using StandardScaler to improve model performance.

Machine Learning Model:

The project utilizes a Linear Regression algorithm.

Evaluation Metrics (Sample performance):

MAE: ~5380

R² Score: ~0.558 (Standard baseline for house price regression).

Deployment:

The model and preprocessing objects were serialized using joblib.

Streamlit is used to create a user-friendly frontend where users can select Bedroom types (e.g., 1 BHK Apartment, 2 BHK, etc.), Area, and Washrooms.

🛠️ Installation & Usage
To run this project locally, follow these steps:

Clone the repository:

Bash
git clone <your-repository-link>
cd <repository-folder>
Install dependencies:

Bash
pip install streamlit pandas joblib scikit-learn
Run the Streamlit app:

Bash
streamlit run app.py
🛠️ Technologies Used
Python (Pandas, NumPy)

Scikit-Learn (Model Training & Preprocessing)

Streamlit (Web UI)

Joblib (Model Serialization)
