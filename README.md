# 🧠 A Proactive Approach in Detection of Pneumonia Disease using VGG16

## 📄 Project Overview
This project focuses on the **automatic detection of pneumonia** from **chest X-ray images** using **deep learning techniques**, specifically **Convolutional Neural Networks (CNNs)** and transfer learning models such as **VGG16** and **ResNet50**.  
The objective is to design a reliable and efficient diagnostic system that assists medical professionals in early and accurate identification of pneumonia cases.

---

## 🎯 Objectives
- Develop a **deep learning model (CNN)** for detecting pneumonia from X-ray images.  
- Compare performance metrics between **CNN**, **VGG16**, and **ResNet50** models.  
- Improve **accuracy**, **recall**, and **F1-score** for robust medical image classification.  
- Provide a solution that reduces diagnostic time and enhances treatment outcomes.

---

## 🧰 Tools & Technologies Used
- **Programming Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn  
- **Environment:** Google Colab (GPU Runtime)  
- **Model Architectures:** CNN, VGG16, ResNet50  

---

## 🧩 Methodology

### 1. **Data Collection**
- Dataset: Publicly available **Chest X-ray dataset** from **Kaggle**  
- Total Images: **5880 samples**  
  - **Training:** 4707 images (80%)  
  - **Testing:** 1173 images (20%)

### 2. **Preprocessing**
- Images resized to **224×224×3**  
- Normalized pixel values between **0 and 1**  
- Noise reduction using filtering techniques  
- Data augmentation for improving model generalization  

### 3. **Model Architecture**
#### **CNN Model**
- Multiple **Conv2D** and **MaxPooling** layers  
- Activation function: **ReLU**  
- Regularization: **Dropout layers** to prevent overfitting  
- Final layer: **Dense layer with Sigmoid activation** for binary classification  

#### **VGG16 and ResNet50**
- Pre-trained on ImageNet  
- Fine-tuned on chest X-ray dataset for pneumonia detection  
- Comparative performance analysis conducted  

### 4. **Evaluation Metrics**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## 📊 Results

| Model | Precision | Recall | F1-Score | Accuracy |
|--------|------------|---------|-----------|-----------|
| CNN | 98% | 98% | 97% | **99.82%** |
| ResNet50 | 95% | 95% | 95% | 95.37% |
| VGG16 | 73% | 73% | 71% | 73.40% |

✅ The **CNN model** achieved the highest accuracy and outperformed the other models in all evaluation metrics.

---

## 💡 Key Insights
- Increasing the number of layers does **not always improve accuracy** — optimization and parameter tuning are crucial.  
- The **CNN model** provides a balance between accuracy and computational efficiency.  
- The proposed model can be extended to detect other lung diseases such as **COVID-19**.  

---

## 🧠 Conclusion
The developed **CNN-based pneumonia detection system** demonstrates exceptional accuracy (99.82%) and reliability.  
It effectively distinguishes between **normal** and **pneumonic** chest X-ray images, offering a potential decision-support tool for radiologists.  
This model can significantly reduce diagnostic errors and assist in **early-stage pneumonia detection**.

---

## 👩‍💻 Authors
- **Nithin Neelisetti**  
  Dept. of Artificial Intelligence & Data Science, KLEF University  
  📧 [nithin79379@gmail.com](mailto:nithin79379@gmail.com)

- **K. Teja Swaroop**  
  Dept. of Artificial Intelligence & Data Science, KLEF University  
  📧 [karanamthejaswaroop@gmail.com](mailto:karanamthejaswaroop@gmail.com)

---

## 🧾 References
The project is based on the analysis of several studies involving CNN, ResNet50, and VGG16 models for pneumonia detection.  
Key research references include:
1. Luka Račić et al., *Pneumonia Detection Using Deep Learning Based on CNN* (2022)  
2. Yan Han et al., *Pneumonia Detection on Chest X-ray using Radiomic Features* (2022)  
3. Dalya S. Al-Dulaimi et al., *Development of Pneumonia Disease Detection Model Based on Deep Learning Algorithm* (2022)  
4. T. Rajasenbagam et al., *Detection of Pneumonia Infection in Lungs from Chest X-ray Images using Deep CNN* (2021)  
...and others cited in the main paper.

---

## 📌 Future Work
- Incorporate **larger and more diverse datasets** for improved generalization.  
- Explore **ensemble learning** and **vision transformers** for enhanced accuracy.  
- Deploy as a **web or mobile application** for real-time diagnosis.  

---

## 🏁 Conclusion Summary
> The proposed CNN model achieves **state-of-the-art performance** for pneumonia detection, with **99.82% accuracy**, making it a promising AI-based diagnostic support system for healthcare applications.
