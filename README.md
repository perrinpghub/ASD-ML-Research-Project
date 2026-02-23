ASD Screening Support Tool (Prototype)

A Machine Learning–based screening support system for Autism Spectrum Disorder (ASD).
This project demonstrates a complete end-to-end ML pipeline — from dataset processing to deployment as a local web application using Streamlit.

Developed as part of a Bachelor of Science research project at Southeast University, Bangladesh.

Project Overview

The system is designed to:
Process screening-style ASD data
Apply feature selection and balancing techniques
Train ensemble ML models
Provide real-time predictions via a simple web interface

⚠️ This tool is a research prototype and not intended for clinical diagnosis.

Folder Structure
ASD_Tool/
│
├── venv/
├── app.py
├── asd_detection_pipeline_v2.py
├── train_and_save.py
├── asd_screening_model.joblib
├── asd_synthetic_dataset_v2.csv
│
├── confusion_matrix.png
├── roc_curve.png
├── Figure_1_ASD_Pipeline.png
│
├── generate_figures.py
├── make_figure1.py
├── requirements.txt
│
├── X_test.pkl
├── y_test.pkl
└── appendix_B_outputs.txt
Machine Learning Pipeline
Data loading and cleaning
Categorical encoding
Feature scaling (Min–Max)
Feature selection using RFE
Class imbalance handling using SMOTE
Training soft voting ensemble models
Saving trained model
Deployment using Streamlit

System Requirements

Windows 10 (tested)
Python 3.8 or higher
Required Libraries
pandas
numpy
scikit-learn
imbalanced-learn
joblib
streamlit
matplotlib

Installation
Activate virtual environment
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Train Model
python train_and_save.py
or
python asd_detection_pipeline_v2.py
Run the Screening Tool
streamlit run app.py

Open in browser:
http://localhost:8501

📋 Input Features

Age,Gender,Ethnicity,Parent education,Jaundice history,Family ASD history,Screening tool score,Eye contact score,Speech delay score,
Repetitive behavior, scoreSensory sensitivity score, Social interaction score.

Output
ASD Risk Prediction (ASD / Not ASD)

Confidence Probability
Evaluation Files
confusion_matrix.png
roc_curve.png
Figure_1_ASD_Pipeline.png

Ethical Disclaimer
Uses synthetic dataset
For academic demonstration only
Not a replacement for medical diagnosis
Requires clinical validation before real use

Authors

Arman Kayes
Md. Jahid Gazi
Md. Abdur Rahim

Department of Computer Science and Engineering
Southeast University, Bangladesh
