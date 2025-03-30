import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Streamlit UI with Dark Theme
st.set_page_config(page_title="ANN Dashboard", layout="wide")

# Apply dark theme styling
st.markdown("""
    <style>
        body { background-color: #0e1117; color: #c0c0c0; }
        .sidebar .sidebar-content { background-color: #1c1e26; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff; }
        .stButton>button { background-color: #4CAF50; color: white; }
        .stFileUploader>div>div { background-color: #1c1e26; color: #c0c0c0; }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title('🌑 ANN-Based Prediction Dashboard (Dark Theme)')

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload CSV File", type=['csv'])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Dataset Preview:", df.head())
    
    # Feature Selection
    target_column = st.selectbox("🎯 Select Target Column", df.columns)
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Data Preprocessing
    X = df[feature_columns]
    y = df[target_column]
    
    # Encoding categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    
    # Scaling numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2224)
    
    # Sidebar for Hyperparameter Selection
    st.sidebar.header("⚙️ Model Hyperparameters")
    
    num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
    neurons_per_layer = []
    activation_functions = []
    
    for i in range(num_layers):
        neurons = st.sidebar.slider(f"Neurons in Layer {i+1}", 5, 100, 10)
        activation = st.sidebar.selectbox(f"Activation for Layer {i+1}", ['relu', 'tanh', 'sigmoid'], index=0)
        neurons_per_layer.append(neurons)
        activation_functions.append(activation)
    
    optimizer = st.sidebar.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop'])
    loss_function = st.sidebar.selectbox("Loss Function", ['binary_crossentropy', 'mean_squared_error', 'hinge'])
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
    epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50)

    # Model Training Button
    if st.button("🚀 Train Model"):
        # Build ANN Model
        model = Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1],)))
        
        for i in range(num_layers):
            model.add(Dense(units=neurons_per_layer[i], activation=activation_functions[i]))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Compile Model
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        
        st.write(f"🔧 Using Optimizer: {optimizer}")
        st.write(f"🔧 Neurons per layer: {neurons_per_layer}")
        st.write(f"🔧 Dropout Rate: {dropout_rate}")
        st.write(f"🔧 Epochs: {epochs}")
        
        # Train Model
        with st.spinner('Training ANN Model... Please wait!'):
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=1)
        
        st.success("🎉 Model Training Completed!")
        
        # Plot Loss
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss', color='cyan')
        ax.plot(history.history['val_loss'], label='Validation Loss', color='magenta')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)
        
        # Plot Accuracy
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Training Accuracy', color='lime')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)
        
        # Final Evaluation
        loss, acc = model.evaluate(X_test, y_test)
        st.write(f"### 📈 Final Validation Accuracy: {acc * 100:.2f}%")
