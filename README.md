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
9. [Reinforcement Learning with DQN CartPole Environment 🕹️🤖](##Reinforcement-Learning-with-DQN-CartPole-Environment)
10. [Conclusion and final toughts 😎🕸️](##Reinforcement-Learning-with-DQN-CartPole-Environment)

# Knowledge Gained

**Through these projects, I've acquired a diverse set of skills and knowledge in machine learning and deep learning, including:**

- Understanding and implementing various machine learning algorithms such as linear regression, classification, clustering, and neural networks.
- Hands-on experience with deep learning frameworks such as PyTorch and TensorFlow/Keras.
- Preprocessing and transforming data for different machine learning tasks, including image classification, natural language processing, and time series prediction.
- Evaluating model performance using appropriate metrics and techniques.
- Exploring advanced topics in deep learning, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), long short-term memory (LSTM) networks, variational autoencoders (VAEs), and reinforcement learning.
- Applying machine learning and deep learning techniques to real-world datasets and problems across various domains.

### **These projects have not only enhanced my technical skills but also fostered a deeper appreciation for the complexities and possibilities of machine learning and artificial intelligence by applying machine learning and deep learning techniques to real-world datasets and problems across various domains.**</br>
**_Enjoy!_**
<br></br>
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
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
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
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/4d26c491-b391-4a29-99e8-512ed2e074c0" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/4d26c491-b391-4a29-99e8-512ed2e074c0)
#### Analysis of the Impact of Training Set Size
- Analyzed the impact of training set size on model performance.
- Created different splits of the data into training and test sets.
- Trained and evaluated both models for each split and plotted the results.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/59b385e1-90ac-494e-b269-31d97be6ce52" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/59b385e1-90ac-494e-b269-31d97be6ce52)
<br></br>

### **Conclusion**
- Both PyTorch and scikit-learn models were successfully implemented and compared.
- Analysis of noise levels and training set sizes provided insights into model performance under different conditions.
### Overall, the project demonstrates the versatility and effectiveness of linear regression models implemented using PyTorch and scikit-learn.
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [Classification 🔢🔍](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/2_Classification.ipynb)
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
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/995bebb5-a05c-484b-bc7e-c8b1f91907c6" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/995bebb5-a05c-484b-bc7e-c8b1f91907c6)
<br></br>
   - Implementation of a Naive Bayes classifier using PyTorch for digit classification.
   - Prediction of digit classes and evaluation of classification accuracy.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/4ff9499e-0ed3-4bd5-a112-9db5a100b130" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/4ff9499e-0ed3-4bd5-a112-9db5a100b130)
<br></br>
2. scikit-learn Implementation:
   - Loading and preprocessing of the 'customers.csv' dataset.
   - Exploration of the dataset through pairplot visualization.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/e30218de-8e6a-42b7-b5a3-a3cc71d2b40f" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/e30218de-8e6a-42b7-b5a3-a3cc71d2b40f)
<br></br>
   - Splitting the data into training and testing sets.
   - Building classification models using Gaussian Naive Bayes, Logistic Regression, and Decision Tree Classifier.
   - Evaluation of the models using accuracy, precision, recall, and F1 score metrics.

### **Conclusion**
Through this project, I gained experience in:
- Preprocessing and transforming data using PyTorch and scikit-learn.
- Implementing classification models for different types of data (images and tabular data).
- Evaluating model performance using various metrics.
<br></br>
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [Clustering Techniques 🌸☯️](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/3_Clustering.ipynb)

### **Brief Description of Work Done:**
This project involves implementing and comparing different clustering algorithms using both PyTorch and scikit-learn libraries.

### **Datasets**
- Synthetic datasets: Generated using functions like `make_blobs` and `make_moons` from scikit-learn to simulate various clustering scenarios.

### **Implementation**
1. PyTorch Implementation:
   - Utilized the `kmeans_pytorch` library to perform K-Means clustering on synthetic data.
   - Demonstrated K-Means clustering and prediction using GPU.
   - Visualized the clustering results and centroids using matplotlib.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/028d82b5-83bd-41a5-a509-aca8e22068f7" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/028d82b5-83bd-41a5-a509-aca8e22068f7)
<br></br>

2. scikit-learn Implementation:
   - Applied K-Means clustering to a synthetic dataset and utilized the elbow method to determine the optimal number of clusters.
   - Implemented hierarchical agglomerative clustering and visualized the dendrogram.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/849997e6-9067-4658-b2cd-2e45f3957cb3" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/849997e6-9067-4658-b2cd-2e45f3957cb3)
<br></br>
   - Employed DBSCAN clustering on a dataset with dense regions and compared the results with K-Means clustering.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/5011508a-71f3-494e-9ffb-9b574301f67c" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/5011508a-71f3-494e-9ffb-9b574301f67c)
<br></br>

### **Conclusion**
Through this project, I gained experience in:
- Implementing clustering algorithms using both PyTorch and scikit-learn.
- Visualizing clustering results and centroids.
- Understanding the differences between K-Means, hierarchical, and density-based clustering algorithms.
<br></br>
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [Neural Networks Use 💸💰](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/4_Neural_Network.ipynb)

### **Brief Description of Work Done:**
In this project, I implemented regression and classification models using neural networks on two different datasets.

### ***Datasets**
1. Diabetes Dataset:
   - Contains information about diabetes patients, including various health metrics and a binary target variable indicating diabetes presence.
2. Iris Dataset:
   - Contains samples of iris flowers, with features describing the flower properties and a categorical target variable indicating the species.

### **Implementation**
1. Diabetes Dataset:
   - Loaded the dataset and performed initial analysis, including checking for missing values and calculating descriptive statistics.
   - Built a binary classification model to predict diabetes presence based on health metrics using PyTorch.
   - Evaluated the model's performance using metrics such as accuracy, confusion matrix, and classification report.
   - Conducted cross-validation to verify the model's stability and made necessary optimizations.
2. Iris Dataset:
   - Loaded the dataset and performed initial analysis, including data exploration through visualizations.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/04017833-322c-48ad-ade1-08bc542a1d9d" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/04017833-322c-48ad-ade1-08bc542a1d9d)
<br></br>
   - Prepared the data for modeling by scaling features and splitting into training and testing sets.
   - Built a multi-class classification model to predict iris species based on flower properties using PyTorch.
   - Evaluated the model's performance using metrics such as accuracy and cross-validated RMSE.
   - Visualized confusion matrix to understand the classification results.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/74be21c3-f665-4b54-b916-d0ab0a6e0ea7" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/74be21c3-f665-4b54-b916-d0ab0a6e0ea7)
<br></br>

### **Conclusion**
Through these implementations:
- I gained experience in building neural network models for both regression and classification tasks.
- Explored different evaluation metrics for model performance assessment.
- Utilized PyTorch for both model implementation and evaluation, showcasing its flexibility and effectiveness in neural network development.
<br></br>
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [CNN Emotion Classification Based on Facial Images 😃📸](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/5_CNN.ipynb)

### **Brief Description of Work Done:**
The provided code consists of two main sections. The first section involves building and training a Convolutional Neural Network (CNN) model for classifying images from the CIFAR-10 dataset. The second section involves a similar process for emotion classification based on facial images using the FER-2013 dataset. Here's a breakdown of the tasks performed:

### **Datasets**
- **CIFAR-10 Dataset:**
  - Downloaded from the CIFAR-10 dataset repository.
  - Consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- **FER-2013 Dataset:**
  - Contains facial images categorized into seven different emotions: angry, disgust, fear, happy, sad, surprise, and neutral.
  - Split into training and testing sets.

### **Implementation**
The code is implemented using PyTorch, a popular deep learning framework. It utilizes various modules from the `torch` and `torchvision` libraries for building and training CNN models. Key implementation details include:
- Definition of CNN model architectures for both datasets.
- Configuration of loss functions (Cross-Entropy Loss) and optimization algorithms (Stochastic Gradient Descent (SGD) and Adam).
- Training loops for iterating over epochs, batches, and data loading.
- Model evaluation on test datasets to compute accuracy metrics.

1. **CIFAR-10 Dataset Processing and CNN Model Training:**
   - Loading the CIFAR-10 dataset and preparing it for training.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/3f42772b-bf54-430d-9882-00485d7f4f8a" width="400"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/3f42772b-bf54-430d-9882-00485d7f4f8a)
<br></br>
   - Building a CNN model architecture using PyTorch.
   - Training the CNN model on the CIFAR-10 dataset.
   - Evaluating the trained model's accuracy on the test dataset.
   - Saving the trained model's parameters to a file.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/d0e99ccb-d661-4795-aedc-c844739cb7ab" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/d0e99ccb-d661-4795-aedc-c844739cb7ab)
<br></br>
2. **FER-2013 Dataset Processing and CNN Model Training:**
   - Loading the FER-2013 dataset and preparing it for training.
   - Building a CNN model architecture for emotion classification.
   - Training the CNN model on the FER-2013 dataset.
   - Evaluating the trained model's accuracy on the test dataset.
   - Saving the trained model's parameters to a file.

### **Conclusion**
Through this project, the following knowledge and tasks were gained:
- Dataset loading and preprocessing using torchvision.
- Building CNN architectures using PyTorch's `nn.Module` API.
- Training and evaluating CNN models for image classification tasks.
- Utilizing GPU acceleration for faster model training (if available).
- Saving and loading trained model parameters for future use.

### Overall, this project provided hands-on experience in developing deep learning models for image classification tasks, demonstrating the practical application of CNNs in real-world scenarios.
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [NLP Tasks with RNN Models 📝🧠](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/6_RNN.ipynb)

### **Brief Description of Work Done:**
The provided code demonstrates the development and training of Recurrent Neural Network (RNN) models for natural language processing (NLP) tasks using TensorFlow and Keras. Three main tasks are performed:
1. Sentiment Analysis on IMDb Movie Reviews
2. Text Classification on the Reuters Dataset
3. Sentiment Analysis on IMDb Movie Reviews with LSTM (Long Short-Term Memory) layers

### **Datasets:**
- **IMDb Dataset:**
  - Contains movie reviews along with their sentiment labels (positive or negative).
  - Loaded using the `imdb.load_data()` function from Keras datasets.
- **Reuters Dataset:**
  - Consists of short newswires categorized into 46 mutually exclusive topics.
  - Loaded using the `reuters.load_data()` function from Keras datasets.

### **Implementation:**
The code implementation involves several steps for each task:
1. **Data Preparation:**
   - Loading and preprocessing datasets, including padding sequences for uniform length.
2. **Model Building:**
   - Building RNN models using Sequential API with embedding layers and RNN/LSTM layers.
   - Configuring appropriate input shapes and layer configurations for each model.
3. **Model Compilation:**
   - Compiling models with suitable optimizers, loss functions, and metrics.
4. **Training:**
   - Training models on training data with a specified number of epochs and batch size.
   - Utilizing a portion of the training data as a validation set for monitoring model performance.
5. **Evaluation and Testing:**
   - Evaluating trained models on the test dataset to assess their performance.
   - Analyzing results using accuracy, loss, classification reports, and confusion matrices.
 <br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/37805164-658b-4d99-8de4-6e3dfaebdd03" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/37805164-658b-4d99-8de4-6e3dfaebdd03)
<br></br> 

### **Conclusion:**
Through this project, various aspects of NLP model development using RNNs and LSTMs were explored, including data preprocessing, model architecture design, training, and evaluation. Key takeaways include:
- Understanding the importance of padding sequences for fixed-length input.
- Implementing different RNN architectures for sentiment analysis and text classification tasks.
- Evaluating model performance using accuracy metrics and analyzing results using classification reports and confusion matrices.

### Overall, the project provides valuable insights into leveraging RNNs and LSTMs for NLP tasks and demonstrates the effectiveness of deep learning techniques in handling sequential data.
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [LSTM Models for Time Series Prediction 📈⏰](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/7_LSTM.ipynb)

### **Brief Description of Work Done:**
The provided code involves the following tasks:
1. Preprocessing and analysis of time series data from a telecommunications dataset.
2. Implementation of an LSTM (Long Short-Term Memory) model for time series prediction.
3. Prediction of stock prices using an LSTM model with data from the Apple stock.

### **Datasets:**
1. **Telecommunications Time Series Data:**
   - The dataset contains time series data related to telecommunications metrics such as user numbers, interference, and success rates.
   - Loaded from a CSV file and preprocessed to handle missing values and select relevant columns.
   - Specific subsets of data are used for analysis and modeling.

2. **Stock Price Data (Apple - AAPL):**
   - Stock price data for the Apple company is downloaded using the Yahoo Finance API.
   - The data is normalized and split into training and testing sets for LSTM model training.

### **Implementation:**
1. **Telecommunications Data Analysis:**
   - Preprocessing of the dataset, including handling missing values and selecting relevant columns.
   - Visualization of time series data to understand patterns and trends.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/e723f66a-7046-4e64-9ff1-280c5b0974cc" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/e723f66a-7046-4e64-9ff1-280c5b0974cc)
<br></br>

2. **LSTM Model for Time Series Prediction:**
   - Implementation of an LSTM model using PyTorch for time series prediction.
   - Data preparation involves creating sequences of input-output pairs.
   - The model architecture consists of LSTM and linear layers.
   - Training of the model using an Adam optimizer and Mean Squared Error loss function.

3. **Stock Price Prediction with LSTM:**
   - Downloading and preprocessing of stock price data for Apple.
   - Creation of input-output sequences for LSTM modeling.
   - Building an LSTM model using Keras with TensorFlow backend.
   - Training the model, monitoring loss, and visualizing training/validation loss.
   - Making predictions on the test set and comparing them with actual stock prices.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/09eb9346-985d-48a4-aa7a-5fa0f77dd3e7" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/09eb9346-985d-48a4-aa7a-5fa0f77dd3e7)
<br></br>
### **Conclusion:**
Through these implementations, various aspects of time series analysis and prediction using LSTM models were explored. Key takeaways include:
- Preprocessing steps such as handling missing data and selecting relevant features are crucial for effective model training.
- LSTM models offer powerful capabilities for capturing temporal dependencies in sequential data.
- Training and evaluation of LSTM models require careful consideration of hyperparameters and monitoring of loss metrics.
- LSTM models can be applied to diverse domains such as telecommunications metrics and financial time series data with promising results.
<br></br>
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [VAE Models for Image Generation 🖼️✨](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/8_VAE.ipynb)

### **Brief Description of Work Done:**
The code involves the implementation of a Variational Autoencoder (VAE) using PyTorch for two different datasets: MNIST and Fashion-MNIST. The key tasks performed include:

### **Datasets:**
1. **MNIST Dataset:**
   - Handwritten digit images dataset consisting of grayscale images of size 28x28 pixels.
   - Used for training the VAE model to reconstruct digit images.

2. **Fashion-MNIST Dataset:**
   - Fashion product images dataset containing grayscale images of size 28x28 pixels.
   - Employed for training the VAE model to reconstruct fashion product images.

### **Implementation:**
The implementation includes the following steps for both MNIST and Fashion-MNIST datasets:

**MNIST Numbers reconstruction:**
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/2ed5db68-b732-4fb7-a464-11597be7f99d" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/2ed5db68-b732-4fb7-a464-11597be7f99d)
<br></br>
**Fashion-MNIST Dataset:**
1. Preparing the data: 
   - Loading and preprocessing the MNIST and Fashion-MNIST datasets.
   - Creating data loaders for efficient batching during training.

2. Implementing the VAE model:
   - Defining the architecture for the VAE model, including encoder and decoder networks.
   - Incorporating a reparameterization trick for sampling from the latent space.

3. Training the VAE model:
   - Optimizing the VAE model parameters using the Adam optimizer.
   - Monitoring the loss function during training to assess model performance.

4. Results Analysis:
   - Evaluating the reconstruction quality of the VAE model by comparing original images with their reconstructed counterparts.
   - Generating new images by sampling from the latent space and passing through the decoder.

**VAE Model Definition:**
   - Architecture setup for the VAE model including encoder and decoder networks.
   - Incorporation of the reparameterization trick for sampling from the latent space.

**Model Training:**
   - Training the VAE model using the Adam optimizer and minimizing the loss function.
   - Evaluating the model's performance by monitoring the loss function during training.

**Results Analysis:**
   - Visual comparison of original images with their reconstructions to assess the quality of reconstruction.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/bfa42427-d978-4842-8d53-925fd050fa92" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/bfa42427-d978-4842-8d53-925fd050fa92)
<br></br>

### **Conclusion:**
Through the implementation of the VAE model on MNIST and Fashion-MNIST datasets, the code demonstrates the capability of VAEs to learn meaningful representations of high-dimensional data and generate new samples. The analysis of reconstruction quality provides insights into the effectiveness of the VAE in capturing and reconstructing the underlying structure of the input data.
<br></br>
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
## [Reinforcement Learning with DQN CartPole Environment](https://github.com/tobi303x/Machine-Learning-Projects/blob/main/9_Reinforcement_learning.ipynb)🕹️🤖

### **Brief Description of Work Done:**
The code implements a Deep Q-Network (DQN) agent using PyTorch to play the "CartPole-v1" game environment from the OpenAI Gym.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/9fea097c-8c8e-45bb-af17-3f8b6c7915a5" width="400"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/9fea097c-8c8e-45bb-af17-3f8b6c7915a5)
<br></br>
_Credits: https://www.gymlibrary.dev/environments/classic_control/cart_pole/_
<br></br>

### **Implementation:**
The implementation includes the following key components:

1. **Setting up the game environment:**
   - Importing necessary libraries including Gym.
   - Creating the "CartPole-v1" environment.

2. **Implementing the DQN agent:**
   - Defining a neural network model representing the DQN.
   - Initializing the replay memory to store experiences.
   - Implementing the ε-greedy exploration strategy to select actions.
3. **DQN Model Definition:**
   - Architecture setup for the neural network model representing the DQN.
   - Implementation of the forward pass method to compute state-action values.

4. **Training the DQN Agent:**
   - Optimization of the DQN model by updating weights based on transitions sampled from the replay memory.
   - Updating the target network using a soft update strategy to stabilize training.

5. **ε-Greedy Exploration:**
   - Implementation of the ε-greedy exploration strategy to balance exploration and exploitation during action selection.

6. **Visualizing Training Progress:**
   - Plotting the duration of each episode during training to monitor progress.
   - Displaying average episode durations over time to assess training stability.
<br></br>
[<img src="https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/5157c59a-4842-4b88-91b8-ea1f20ae2c1a" width="600"/>](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/5157c59a-4842-4b88-91b8-ea1f20ae2c1a)
<br></br>

### **Conclusion:**
Through the implementation of the DQN agent, the code demonstrates the reinforcement learning approach to train an agent to play the "CartPole-v1" game environment. By optimizing the DQN model and updating target networks, the agent learns to balance the pole on the cart, achieving stable training progress over multiple episodes.
<br></br>
**[Back to Top](#Machine-Learning-Projects)**
<br></br>
#### Assets
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
![cart_pole](https://github.com/tobi303x/Machine-Learning-Projects/assets/114963170/9fea097c-8c8e-45bb-af17-3f8b6c7915a5)
</details>
