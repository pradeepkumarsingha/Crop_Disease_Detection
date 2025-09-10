# 🌱 Crop Disease Detection using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to detect plant diseases from leaf images. Built with **TensorFlow**, **Keras**, **OpenCV**, and **Streamlit**, this project allows users to upload a leaf image and get the predicted crop type, disease, and confidence score through an interactive web app.

This model is trained on **30,000 leaf images of major crops from Odisha**, achieving a **training accuracy of 95%** and **validation accuracy of 88%**.

---

## Features

- Detects multiple crop diseases from leaf images  
- Shows **predicted crop type** and **disease**  
- Displays **confidence score** of the prediction  
- Interactive **Streamlit GUI**  
- Trained on **major crops of Odisha**  
- Uses **CNN** for accurate disease detection  

---

## Technologies Used

- Python 3.x  
- TensorFlow & Keras (Deep Learning)  
- OpenCV (Image Processing)  
- Streamlit (Web App GUI)  
- NumPy & Pandas  
- Pickle (Saving Label Encoder)  

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/pradeepkumarsingha/Crop_disease_Detection.git
cd Crop_disease_Detection
```
## Dataset
-30,000 leaf images of major crops from Odisha
-Dataset preprocessing includes:
-Resizing to 100x100 pixels
-Grayscale conversion (if CNN trained on grayscale)
-Normalization
-Labels are encoded using LabelEncoder and saved as labels_encoder.pkl

## Model Performance
-Training Accuracy: 95%
-Validation Accuracy: 88%
-Loss Function: Categorical Cross-Entropy
-Optimizer: Adam
-Epochs: 20

## Project Structure
```bash
Crop_disease_Detection/
│
├─ app.py                              # Streamlit web app
├─ plant_disease_cnn_model_final.h5    # Trained CNN model
├─ labels_encoder.pkl                   # Label encoder
├─ clean_data/                          # Images and labels (excluded from GitHub)
├─ requirements.txt                     # Required Python packages
└─ README.md                            # Project documentation
```
## Contributing
Feel free to open issues or submit pull requests for improvements.

## Contact

**Pradeep Kumar Singha**  
- **GitHub:** [github.com/pradeepkumarsingha](https://github.com/pradeepkumarsingha)  
- **Email:** [pradeep@example.com](mailto:mr.pradeepkumarsingha@gmail.com)
