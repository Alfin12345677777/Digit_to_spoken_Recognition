Perfect 👌 — here’s a clean, professional, and Markdown-formatted version of your README.md with your full project description and the added instruction about downloading the model using download_model.py.

You can copy and paste this directly into your repo root.

🧠 Conversion of Handwritten Recognition to Voice Recognition

Authors:
Carlos Alonso Gradillas¹, Park John², Alfin William Adi Putra³
¹ Computer Science, CEU San Pablo, Madrid, Spain
² ICT Convergence, Handong Global University, Pohang, Korea
³ ICT Convergence, Handong Global University, Pohang, Indonesia

Roles:

Project Manager: Park John

Data Analyst: Carlos Alonso

Project Coordinator: Alfin William Adi Putra

📅 Project Plan & Schedule
Week	Task
4	Data Preparation
5	Model Preparation
6	Training Model
7	Training Model
8	Midterm Report
9	Data Update
10	Model Update
11	Training & Testing
12	Final Presentation & Demonstration
13	Final Report (Conference Paper Form)
🎯 Objective

The primary aim of this project is to develop a robust system capable of converting handwritten text into spoken words using the K-Nearest Neighbors (KNN) algorithm.
This innovative approach bridges the gap between traditional handwritten communication and modern voice recognition technology.

Through the use of KNN, our system will analyze handwritten characters, recognize patterns, and accurately transcribe them into audible speech.
This technology has potential applications for visually impaired users and for those who prefer verbal interaction over reading text.

📚 Background Research

We analyzed five key research papers to guide model design and evaluation:

1. Real Time Handwritten Digits Recognition Using Convolutional Neural Network

Kaveti Upender & Venkata Siva Kumar Pasupuleti

Focuses on real-time handwritten digit recognition using MNIST and OpenCV.

Achieves near-human accuracy using CNNs.

2. Handwritten Digit Recognition using Machine and Deep Learning Algorithms

Ritik Dixit, Rishika Kushwah, and Samay Pashine

Compares SVM, MLP, and CNN algorithms for handwritten digits.

Discusses model efficiency in accuracy, error rates, and training time.

3. Image Recognition and Voice Translation for the Visually Impaired

Sandeep Pasupuleti, Lahari Dadi, Manikumar Gadi, R. Krishnaveni

Uses VGG16 and LSTM to translate recognized text into speech for accessibility.

Combines CNN and RNN for improved translation accuracy.

4. An IoT System for Converting Handwritten Text to Editable Format via Gesture Recognition

Nidhi Patel

Proposes an IoT system using Raspberry Pi and edge detection for gesture recognition.

Converts handwritten content into editable text using OpenCV and ML algorithms.

5. NeuroWrite: Predictive Handwritten Digit Classification using Deep Neural Networks

Kottakota Asish, P. Sarath Teja, R. Kishan Chander, D. Deva Hema

Employs deep neural networks for large handwritten datasets (MNIST).

Highlights CNN and few-shot learning approaches for digit classification.

⚙️ Method

We use the MNIST dataset as our primary data source:

TensorFlow MNIST Dataset

For extended experiments, we may use EMNIST (letters and symbols):

Kaggle EMNIST Dataset

The system workflow:

Collect and preprocess handwritten digit/letter data.

Train CNN and KNN models to classify input.

Convert recognized text into speech using text-to-speech synthesis.

Evaluation metrics include:

Mean Average Precision (mAP)

Intersection over Union (IoU)

Precision & Recall

These ensure accurate detection, bounding box overlap, and robustness across test data.

🧩 Models Used

KNN – Simple baseline for handwritten character classification.

CNN – Captures spatial features and invariance in handwriting.

RNN – Adds sequential learning for improving speech conversion.

We aim to achieve:

>95% accuracy with CNN

~94% accuracy with RNN

and extend functionality to recognize both numeric and alphabetic characters with speech output.

🧱 Expected Results

A model that:

Accurately recognizes handwritten digits and letters

Converts recognized text to voice output

Provides strong accessibility support for visually impaired users

📥 How to Get the Model

The trained model file (new_model.h5) is too large for GitHub.
You can download it automatically from Google Drive using the provided script:

# 1. Install dependency
pip install gdown

# 2. Run the script
python download_model.py


This script will download the model from:
Google Drive Link

The model will be saved automatically into:

AlexLeNet-master/new_model.h5

🧾 Citation

If you use or reference this project, please cite as:

Gradillas, C. A., Park, J., & Alfin W. A. P. (2025).
Conversion of Handwritten Recognition to Voice Recognition.
Handong Global University – CEU San Pablo Joint Project.
