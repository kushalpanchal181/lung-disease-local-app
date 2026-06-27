# AI-Based Lung Disease Detection Local Prototype

This project is a local prototype for AI-based lung disease detection from chest X-ray images.  
The system allows a user to upload a chest X-ray image, scan it using a trained AI model, and return a prediction result with a confidence score.

## Project Structure

```text
lung-disease-local-app/
│
├── dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   │
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   │
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── model/
│   └── lung_model.h5
│
├── notebooks/
│   └── model experimentation notebooks
│
├── results/
│   └── evaluation results, graphs and confusion matrix
│
├── screenshots/
│   └── app and implementation screenshots
│
├── testing/
│   └── testing scripts and test notes
│
├── aws/
│   └── future AWS deployment notes
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore