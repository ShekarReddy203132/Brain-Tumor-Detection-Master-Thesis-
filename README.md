# An Automatic Deep Learning Approach for Brain Tumor Detection using Multimodal Datasets

This repository contains Barin Tumor Detection research titled **"An Automatic Deep Learning approach for Brain Tumor Detection using Multimodal Datasets"** by Hanuman Chandra Shekar Reddy Tangirala.

## Project Overview

Brain tumors are critical medical conditions requiring **timely and accurate diagnosis** to improve treatment outcomes. Traditional diagnostic techniques, such as manual analysis of Magnetic Resonance Imaging (MRI) scans, are often time-consuming, subjective, and prone to human error. This research harnesses **deep learning techniques**, particularly Convolutional Neural Networks (CNNs), to automatically detect and classify brain tumors using a multimodal MRI dataset. The motivation stems from the increasing demand for intelligent, accurate, and fast diagnostic tools in healthcare, aiming to analyze subtle features often overlooked by the human eye.

## Objectives

The primary objectives of this thesis were:
*   To **implement and evaluate multiple CNN models** (VGG16, VGG19, ResNet50, EfficientNetB0, MobileNetV2, GoogleNet/InceptionV3, and DarkNet53) for brain tumor classification using the Br35H dataset.
*   To **compare the performance** of each CNN model based on key metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, AUC-ROC, and Confusion Matrix.
*   To **select the top three CNN models** based on performance and integrate them using ensemble learning techniques for improved classification.
*   To **extract and concatenate features** from the selected CNN models and classify tumor types using a Support Vector Machine (SVM).
*   To **perform a comparative analysis** of all three approaches to highlight improvements and limitations.

## Dataset

The research utilizes the **Br35H dataset**, a publicly available multimodal brain MRI image dataset.
*   **Composition:** The dataset includes four tumor classes: **glioma, meningioma, pituitary, and notumor**. Each image is in JPEG format with a resolution of 512x512 pixels.
*   **Distribution:** The dataset comprises 6000 images, equally distributed with 1500 images per class.
*   **Split:** It was divided using an **80:20 ratio** for training and testing, ensuring balanced evaluation.
*   **Preprocessing:** Images underwent resizing (to 224x224 pixels) and normalization (pixel values 0-1).
*   **Data Augmentation:** Techniques like rotation, flipping, zooming, brightness adjustment, shear, and shift were applied to the training data to increase variability and reduce overfitting.

## Methodology

The proposed framework is divided into four key phases:

1.  **Individual CNN Model Implementation:**
    *   Seven widely-used CNN architectures were implemented and evaluated: **VGG16, VGG19, ResNet50, EfficientNetB0, MobileNetV2, InceptionV3 (GoogleNet), and DarkNet53**.
    *   Models were initialized with ImageNet weights and fine-tuned on the Br35H dataset.
    *   Hyperparameters were optimized (e.g., Learning Rate: 1e-4 to 1e-6, Batch Size: 32, Epochs: 10, Optimizer: Adam/RMSprop, Loss Function: Categorical Cross-Entropy).

2.  **Ensemble Learning Approach:**
    *   The **top three CNNs** based on performance (InceptionV3, VGG19, and MobileNetV2) were selected.
    *   These models were combined using a **weighted soft voting ensemble approach**. This method predicts class probabilities from each model, assigns weights based on model accuracy, and makes a final prediction by summing weighted probabilities.

3.  **Feature Extraction and SVM Classification:**
    *   Deep features were extracted from the penultimate (second-last) dense layer of the three best-performing models (VGG19, MobileNetV2, InceptionV3).
    *   These features were flattened and **concatenated into a single feature vector**.
    *   The combined feature vector was then used for classification by a **Support Vector Machine (SVM) classifier** with an RBF kernel.

4.  **Result Comparison:**
    *   A comprehensive comparison of all three approaches was performed to determine the most effective method.

## Key Findings & Results

The research demonstrates the effectiveness of ensemble and hybrid learning approaches for robust, high-accuracy brain tumor classification.

*   **Individual Models:**
    *   Among individual models, **InceptionV3 achieved the highest accuracy of 99.00%**, with an AUC-ROC of 0.9997.
    *   It was closely followed by **VGG19 (98.83% accuracy, 0.9991 AUC-ROC)** and MobileNetV2 (94.50% accuracy, 0.9968 AUC-ROC).
    *   EfficientNetB0 showed the lowest performance among the tested models.

*   **Ensemble Model:**
    *   The ensemble model (InceptionV3, VGG19, MobileNetV2 with weighted soft voting) achieved an **improved accuracy of 99.17%**, an F1-score of 99.17%, and an **AUC-ROC of 0.9998**.
    *   This result surpassed individual models, demonstrating enhanced balance across all tumor classes and more stable predictions.

*   **CNN+SVM Hybrid Model:**
    *   The CNN+SVM hybrid model performed competitively with **98.67% accuracy** and an AUC-ROC of 0.9992.
    *   This validated the power of feature-level fusion, showing excellent classification accuracy.

## Contributions & Significance

This research offers a significant contribution to the domain of medical imaging and diagnostic automation by:
*   Providing a **deep learning-based solution for accurate brain tumor detection**.
*   **Reducing dependency on manual diagnosis** and improving consistency.
*   Demonstrating the **effectiveness of ensemble and hybrid methods** in enhancing performance.
*   Laying the groundwork for **real-time, AI-assisted diagnostic tools** in healthcare.
*   A detailed implementation and evaluation of multiple CNN architectures for multiclass brain tumor classification.
*   A thorough comparative analysis of individual models, ensemble model, and CNN+SVM strategy.

## Tools and Technologies

The following tools, libraries, and platforms were used:
*   **Programming Language:** Python 3.9
*   **Deep Learning Libraries:** TensorFlow 2.10, Keras, Scikit-learn
*   **Image Processing Tools:** OpenCV, NumPy
*   **Data Manipulation and Analysis:** NumPy, Pandas
*   **Visualization Libraries:** Matplotlib, Seaborn
*   **Development Environment:** Jupyter Notebook
*   **Cloud Platform:** Google Colab / Ubuntu (Cloud-based) with NVIDIA Tesla T4 GPU

## Limitations

While the results were promising, the study identified several limitations:
*   **Dataset Constraint:** Only the Br35H dataset was used, limiting generalizability without external validation.
*   **Limited Modalities:** Only 2D slices were used; 3D volumetric analysis was not considered.
*   **Computational Cost:** Training multiple deep models and ensemble strategies increased training time and resource requirements.
*   **Model Interpretability:** Models acted as black boxes, and techniques for explaining predictions (like Grad-CAM) were not used.

## Future Work

To build upon this work, the following future research directions are suggested:
*   **Use of Multi-institutional or Cross-Dataset Validation:** To evaluate robustness and real-world applicability.
*   **3D CNN Implementation:** Exploring 3D CNNs on volumetric MRI data for more detailed spatial understanding.
*   **Explainable AI (XAI):** Incorporating visualization tools like Grad-CAM or SHAP for clinical trust.
*   **Real-Time Model Optimization:** Using techniques like pruning or quantization for deployment on edge devices.
*   **Clinical Validation:** Collaborating with medical professionals for real-world feedback in hospital environments.

---
