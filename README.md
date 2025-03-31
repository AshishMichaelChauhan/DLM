# 📌 Heart Disease Prediction Using Artificial Neural Networks (ANN)
🔗 **Streamlit Dashboard Link:** https://yewtgdfgvt8eycptwuutkx.streamlit.app/
** Dataset Link: https://drive.google.com/file/d/1s1YBmkDnjankZZU3CxbEtn5TwddvRA0w/view?usp=sharing

---

## 🏆 Project Details 
This project aims to predict the presence of heart disease using a deep learning model built with **Artificial Neural Networks (ANN)**.  
The model is trained on the **Heart Disease Dataset** containing patient health records with various clinical and cardiovascular attributes.  
The primary focus is on **hyperparameter tuning** to enhance model performance.  
The dataset (`heart_disease.csv`) is included in this repository.

---

## 👥 Contributors
- **Ashish Michael Chauhan** – 055007

---

## 🔑 Key Activities
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical attributes.
- **Model Development:** Constructing an ANN with customizable hyperparameters via Streamlit UI.
- **Hyperparameter Tuning:** Investigating the impact of different hyperparameter values on accuracy.
- **Visualization & Insights:** Analyzing model performance using loss and accuracy plots.
- **Managerial Interpretation:** Extracting actionable insights for healthcare decision-making.

---

## 💻 Technologies Used
- **Python**
- **TensorFlow & Keras** (for ANN modeling)
- **Pandas & NumPy** (for data manipulation)
- **Matplotlib & Seaborn** (for visualization)
- **Scikit-learn** (for preprocessing and evaluation)
- **Streamlit** (for interactive dashboard development)

---

## 📊 Nature of Data 
The dataset consists of structured clinical data with both categorical and numerical variables representing patients' medical history and test results.

---

## 📌 Variable Information
| **Feature**    | **Type**       | **Description**                              |
|----------------|----------------|----------------------------------------------|
| Age            | Continuous     | Age of the patient (years)                   |
| Sex            | Categorical    | Gender (1 = Male, 0 = Female)                |
| CP             | Categorical    | Chest pain type (0-3)                        |
| Trestbps       | Continuous     | Resting blood pressure (mm Hg)               |
| Chol           | Continuous     | Serum cholesterol level (mg/dl)              |
| Fbs            | Binary         | Fasting blood sugar > 120 mg/dl (1 = yes)    |
| Restecg        | Categorical    | Resting electrocardiographic results         |
| Thalach        | Continuous     | Maximum heart rate achieved                  |
| Exang          | Binary         | Exercise-induced angina (1 = yes)            |
| Oldpeak        | Continuous     | ST depression induced by exercise            |
| Slope          | Categorical    | Slope of peak exercise ST segment            |
| Ca             | Continuous     | Number of major vessels colored              |
| Thal           | Categorical    | Thalassemia type                             |
| Target         | Binary         | 1 = Heart disease present, 0 = No disease    |

---

## 🎯 Problem Statements
- Can ANN models effectively classify patients at risk of heart disease?
- What impact does hyperparameter tuning have on model accuracy?
- Can the model achieve high accuracy without overfitting?

---

## 🏗️ Model Information
- **Input Layer:** Accepts all numerical and encoded categorical features.
- **Hidden Layers:** Number of layers and neurons customizable via Streamlit UI.
- **Activation Functions:** ReLU, Tanh, Sigmoid (chosen dynamically by the user).
- **Dropout Rate:** Adjustable to prevent overfitting.
- **Optimizer Options:** Adam, SGD, RMSprop.
- **Loss Function Options:** Binary Cross-Entropy, Mean Squared Error, Hinge Loss.
- **Output Layer:** Single neuron with Sigmoid activation for binary classification.

---

## 📉 Observations from Hyperparameter Tuning 
### 1️⃣ Number of Hidden Layers
- **1-2 layers:** Moderate accuracy (~85%).
- **3-4 layers:** Optimal accuracy (~91%) without overfitting.
- **5+ layers:** Slight improvement but increased computational cost.

### 2️⃣ Neurons per Layer
- **10-50 neurons:** Stable and consistent training.
- **50+ neurons:** Marginal improvement, but risk of overfitting increases.

### 3️⃣ Activation Functions
- **ReLU:** Best performance in hidden layers.
- **Sigmoid:** Used in the output layer for binary classification.
- **Tanh:** Works well but slightly inferior to ReLU.

### 4️⃣ Optimizer Comparison
- **Adam:** Best performance, balances speed and accuracy.
- **SGD:** Slower, requires more epochs for convergence.
- **RMSprop:** Works well but sometimes unstable.

### 5️⃣ Dropout Rate
- **0-0.2:** Best accuracy (~91%).
- **0.3-0.5:** Reduces overfitting but may hurt performance.

### 6️⃣ Epochs
- **50 epochs:** Sufficient for convergence.
- **100+ epochs:** No significant improvement, risk of overfitting.

---

## 📈 Managerial Insights 
### 🔹 **Healthcare Applications**
- This ANN model can assist doctors and healthcare professionals in **early diagnosis of heart disease**.
- Automating heart disease detection reduces **human error** and speeds up treatment decisions.

### 🔹 **Business Value**
- Hospitals and insurance companies can use **AI-driven risk assessment** for better patient outcomes.
- Implementing ANN-based diagnostic tools can **reduce costs and workloads** for cardiologists.

### 🔹 **Future Improvements**
- **Feature Engineering:** Adding more relevant clinical features for better accuracy.
- **Hybrid Models:** Combining ANN with other machine learning techniques (e.g., XGBoost, Random Forest).
- **Explainability:** Using techniques like **SHAP values** to understand feature importance in predictions.

---

## 🚀 **Conclusion**
This project successfully demonstrates how deep learning can be leveraged for **heart disease prediction**.  
The ANN model achieves **~99.95% accuracy** after hyperparameter tuning, making it a powerful tool for **healthcare analytics**.
