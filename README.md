# Machine Learning Projects
## Table of Contents
0. [Introduction, PyTorch basics 📖👓](#Introduction,-PyTorch-basics-📖👓)
1. [Regression ↗️🏫](#Regression-↗️🏫)
2. [Classification 🌸🔍](#Classification-🌸🔍)
3. [Clustering Techniques 🧩☯️](#Clustering-Techniques-🧩☯️)
4. [Neural Networks Use 💸💰](#Neural-Networks-Use-💸💰)
5. [CNN Emotion Classification Based on Facial Images 😃📸](#CNN-Emotion-Classification-Based-on-Facial-Images-😃📸)
6. [NLP Tasks with RNN Models 📝🧠](#NLP-Tasks-with-RNN-Models-📝🧠)
7. [LSTM Models for Time Series Prediction 📈⏰](#LSTM-Models-for-Time-Series-Prediction-📈⏰)
8. [VAE Models for Image Generation 🖼️✨](#VAE-Models-for-Image-Generation-🖼️✨)
9. [Reinforcement Learning with DQN CartPole Environment 🕹️🤖](#Reinforcement-Learning-with-DQN-CartPole-Environment-🕹️🤖)
10. [Conclusion and final toughts 😎🕸️](#Reinforcement-Learning-with-DQN-CartPole-Environment-🕹️🤖)

## [Introduction, PyTorch basics 📖👓](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/0_Introduction.ipynb)
### **Brief Description of Work Done:**
- Tensor Creation: Demonstrates various methods of creating tensors including filling with zeros, ones, sequences, and converting from NumPy arrays.
- Tensor Operations: Shows basic tensor operations such as addition, multiplication, mean calculation, and sum calculation.
- Reshaping Tensors: Illustrates reshaping tensors into different shapes and flattening them into one-dimensional tensors.
- Indexing and Slicing Tensors: Explains how to access specific elements, rows, columns, and diagonals of tensors.
- Tensor and Numpy Operations: Covers conversion between PyTorch tensors and NumPy arrays and changing data types.
- Matrix Operations: Includes matrix multiplication and tensor transposition.
- Advanced Indexing: Demonstrates advanced indexing techniques using logical masks.
- In-Place Operations: Shows how to perform in-place operations on tensors and observe the changes.
  
### This project serves as a friendly guide to understanding various functionalities of PyTorch tensors. By going through the provided code snippets and exercises I gained hands-on experience with tensor creation, manipulation, indexing, and basic operations in PyTorch. Additionally, it provided a foundation for further exploration into deep learning using PyTorch.

## [Regression ↗️🏫](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/1_Regression.ipynb)
### **Brief Description of Work Done:**
This project implements and compares linear regression models using both PyTorch and scikit-learn. It includes experiments to analyze the impact of noise levels and training set sizes on the performance of the models.

### **Implementation**
#### PyTorch Model
- The PyTorch model is implemented using a simple linear regression class (`LinearRegressionSimple`) inheriting from `nn.Module`.
- A custom dataset class (`RegressionDataset`) is created to handle the data.
- The training loop (`train`) and validation function (`validate`) are defined to train the model.
- The model is trained using stochastic gradient descent (SGD) with mean squared error (MSE) loss.

#### Ccikit-learn Model
- The scikit-learn linear regression model is implemented using `LinearRegression` from `sklearn.linear_model`.
- The model is trained using the `fit` method and evaluated using mean squared error (MSE) and coefficient of determination (R^2).

#### Different Levels of Noise
- Investigated the impact of different noise levels on model performance.
- Generated datasets with varying noise levels and trained both models.
- Created plots showing how noise affects MSE and R^2 for both models.
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/9ec2d76f-b2b2-48db-9e28-a1136cf4d8a8" width="500"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/9ec2d76f-b2b2-48db-9e28-a1136cf4d8a8)
#### Analysis of the Impact of Training Set Size
- Analyzed the impact of training set size on model performance.
- Created different splits of the data into training and test sets.
- Trained and evaluated both models for each split and plotted the results.

### **Conclusion**
- Both PyTorch and scikit-learn models were successfully implemented and compared.
- Analysis of noise levels and training set sizes provided insights into model performance under different conditions.
### Overall, the project demonstrates the versatility and effectiveness of linear regression models implemented using PyTorch and scikit-learn.

## [Classification 🌸🔍](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/2_Classification.ipynb)
### **Brief Description of Work Done:**
This project involves classification tasks using both PyTorch and scikit-learn libraries. 

### ***Datasets**
- MNIST dataset: A widely used dataset for handwritten digit recognition. It consists of 28x28 grayscale images of handwritten digits (0-9) and their corresponding labels.
- 'customers.csv': A dataset containing information about customers, including their gender, age, annual income, and spending score.

### **Implementation**
The code is divided into two main parts: 
1. PyTorch Implementation:
   - Data transformation using `torchvision.transforms.Compose`.
   - Creation of MNIST training and test datasets.
   - Exploration of the MNIST dataset, including visualizing images and calculating probabilities.
   - Implementation of a Naive Bayes classifier using PyTorch for digit classification.
   - Prediction of digit classes and evaluation of classification accuracy.

2. scikit-learn Implementation:
   - Loading and preprocessing of the 'customers.csv' dataset.
   - Exploration of the dataset through pairplot visualization.
   - Splitting the data into training and testing sets.
   - Building classification models using Gaussian Naive Bayes, Logistic Regression, and Decision Tree Classifier.
   - Evaluation of the models using accuracy, precision, recall, and F1 score metrics.

### **Conclusion**
Through this project, I gained experience in:
- Preprocessing and transforming data using PyTorch and scikit-learn.
- Implementing classification models for different types of data (images and tabular data).
- Evaluating model performance using various metrics.

## [Clustering Techniques 🧩☯️](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/3_Clustering.ipynb)
## [Neural Networks Use 💸💰](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/4_Neural_Network.ipynb)
## [CNN Emotion Classification Based on Facial Images 😃📸](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/5_CNN.ipynb)
## [NLP Tasks with RNN Models 📝🧠](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/6_RNN.ipynb)
## [LSTM Models for Time Series Prediction 📈⏰](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/7_LSTM.ipynb)
## [VAE Models for Image Generation 🖼️✨](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/8_VAE.ipynb)
## [Reinforcement Learning with DQN CartPole Environment 🕹️🤖](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/9_Reinforcement_learning.ipynb)


Lorem ipsum



# Assets
<details>
![2_NoiseImpact](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/4d26c491-b391-4a29-99e8-512ed2e074c0)
![2_TrainingSetSizeImpact](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/59b385e1-90ac-494e-b269-31d97be6ce52)
![3_Classification_MNIST_Sum_of_images](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/995bebb5-a05c-484b-bc7e-c8b1f91907c6)
![3_Classification_PairPlot](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/e30218de-8e6a-42b7-b5a3-a3cc71d2b40f)
![3_Clssification_Number_prediction](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/4ff9499e-0ed3-4bd5-a112-9db5a100b130)
![4_Clustering_Cluster_Scatter](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/028d82b5-83bd-41a5-a509-aca8e22068f7)
![4_Clustering_DBSCAN_vs_K-Means](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/5011508a-71f3-494e-9ffb-9b574301f67c)
![4_Clustering_Dendogram](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/849997e6-9067-4658-b2cd-2e45f3957cb3)
![5_Neural_Network_Confusion_matrix](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/74be21c3-f665-4b54-b916-d0ab0a6e0ea7)
![5_Neural_Network_Iris_PairPlot](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/04017833-322c-48ad-ade1-08bc542a1d9d)
![6_CNN_Feature](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/3f42772b-bf54-430d-9882-00485d7f4f8a)
![6_CNN_Feature_maps](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/d0e99ccb-d661-4795-aedc-c844739cb7ab)
![7_RNN_Training_and_val_accuracy](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/37805164-658b-4d99-8de4-6e3dfaebdd03)
![8_LSTM_Telecomunications_Network_Metrics](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/e723f66a-7046-4e64-9ff1-280c5b0974cc)
![8_LSTM_yfinance](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/09eb9346-985d-48a4-aa7a-5fa0f77dd3e7)
![9_VAE_FashionMNIST_Reconstruction](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/bfa42427-d978-4842-8d53-925fd050fa92)
![9_VAE_MNIST_GeneratedNumbersImages](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/2ed5db68-b732-4fb7-a464-11597be7f99d)
![10_Reinforcement_Learning](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/5157c59a-4842-4b88-91b8-ea1f20ae2c1a)
</details>








[Back to Top](#Machine-Learning-Projects)

