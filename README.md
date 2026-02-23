# ASD Screening Support Tool (Prototype)

A Machine Learning--based screening support system for Autism Spectrum
Disorder (ASD).\
This project demonstrates a complete end-to-end ML pipeline --- from
dataset processing to deployment as a local web application using
Streamlit.

Developed as part of a Bachelor of Science research project at Southeast
University, Bangladesh.

## Project Overview

-   Feature selection using RFE\
-   Class balancing using SMOTE\
-   Ensemble ML models\
-   Streamlit-based web tool

Research prototype only (not for clinical diagnosis).

## Folder Structure

ASD_Tool/ - app.py - train_and_save.py - asd_detection_pipeline_v2.py -
asd_screening_model.joblib - asd_synthetic_dataset_v2.csv -
requirements.txt

## How to Run

Activate environment in windows cmd: venv\Scripts\activate

Install packages: pip install -r requirements.txt

Run tool: streamlit run app.py

## Inputs

Age, gender, education, screening scores, medical history.

## Output

ASD risk prediction with confidence.

## Authors

Arman Kayes\
Md. Jahid Gazi\
Md. Abdur Rahim

Southeast University, Bangladesh

Academic and research use only.
