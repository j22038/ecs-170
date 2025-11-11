# Project Check-In  
**Date:** November 2025  
**Project Title:** Plant Disease Classification using CNN  

## 1. Project Overview  
Our project aims to implement a Convolutional Neural Network (CNN) capable of classifying plant leaves as **healthy** or **diseased**. The goal is to create a model that could be practically used in agriculture or home gardening to help users quickly identify plant diseases from images.  

Since the proposal, our main focus has been on **dataset selection**, **image preprocessing**, and **initial model setup**.

---

## 2. Dataset and Resources  
We selected the [Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle. This dataset provides labeled images of various plant species with corresponding health conditions, making it suitable for our classification objective.  

We are developing our model and preprocessing pipeline in **Jupyter Notebooks**, as this environment supports:
- Iterative experimentation and visualization during preprocessing and model development.  
- Inline display of results (e.g., plots and intermediate outputs).  
- Ease of documentation and explanation alongside code for collaborative work.  

Libraries in use so far include:
- `cv2` for image processing (resizing, grayscale conversion, padding).

---

## 3. Progress and Milestones  

### Completed / In Progress
- **Dataset Selection:** Finalized choice of the [Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).  
- **Preprocessing Implementation:**  
  We implemented an image preprocessing pipeline in Python using **OpenCV** and **Pathlib** to ensure consistent input dimensions for our CNN.  
  Key preprocessing steps:
  - **Grayscale Conversion:** Converts color images to grayscale to reduce computational complexity and focus on texture/shape features.  
  - **Aspect-Ratio Preserving Resizing with Padding:** Resizes each image to a fixed size of `128×128` pixels while maintaining the aspect ratio by adding constant padding.  
  - **Batch Directory Processing:** Recursively traverses through all image directories, processes valid image files (`.jpg`, `.jpeg`, `.png`), and stores the results in a structured `data/processed` folder. 


### Next Steps
- Complete model training on a small subset of the dataset to verify pipeline functionality.  
- Tune hyperparameters and compare results using grayscale versus color images.  
- Integrate metrics visualization for training/validation accuracy and loss.  

---

## 4. Challenges Encountered  
One challenge we are addressing is **balancing image quality and training efficiency**.  
High-resolution images improve feature extraction but significantly increase computational load. By converting images to grayscale and reducing resolution, we aim to maintain classification performance while shortening training time.  

At this stage, we are still **uncertain about where and how we will train the full model**. We have begun initial testing on our personal computers, but we are not yet sure whether they will be sufficient for complete training. We also do not yet know how long full model training might take or what level of computational resources will ultimately be required. Determining the most practical training setup will be an important next step.

---

## 5. Reflections and Learning  
Working on preprocessing has provided insights into how critical **data preparation** is in deep learning projects. Small preprocessing decisions can drastically influence both accuracy and runtime.  

We've also gained practical experience using Jupyter Notebooks for collaborative AI development, balancing readability, experimentation, and reproducibility.  

---

## 6. Feedback Requests  
At this stage, we would appreciate feedback on:
- Advice regarding where we might train the full model (e.g., local machines vs. cloud services).

---

## 7. Figures (Placeholder Examples)
*(Perhaps some images here of raw vs processed images)*  

---

## 8. Summary  
*(Some sort of summary)*

---

