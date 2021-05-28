# Credit-Card-Fraud-Detection

## Introduction

One of the big legal problems in the credit card business is fraud. The key goals of the research are, firstly to recognize recognize the different forms of fraudulent credit cards, secondly, to explore
alternative methods utilized in fraud detection. The sub-aim is to evaluate, present, and examine recent results in the identification of credit card fraud. The article sets out terms common in fraud involving credit cards and highlighting figures and key statistics in this field. Various measures such as Logistic Regression, Random Forest, Autoencoder and SMOTE can be taken and enforced based on the type of fraud faced by the credit card industry or financial institutions. In terms of cost savings and efficiency, the proposals made in this report are likely to have beneficial attributes. The importance of applying these techniques examined here in minimizing credit card fraud. Yet when legitimate credit card users are misclassified as fraudulent there are still ethical issues.

The Credit Card Fraud Detection project is used to identify whether a new transaction is fraudulent or not by modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. We will use various predictive models to see how accurate they are in detecting whether a transaction is a normal payment or a fraud. 

## Dataset

We are using the datasets provided by Kaggle. This data set includes all transactions recorded over the course of two days. As described in the dataset, the features are scaled and the names of the features are not shown due to privacy reasons.

The dataset consists of numerical values from the 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28. Furthermore, there is no metadata about the original features provided, so pre-analysis or feature study could not be done.

## Aims

The main aim of this report is to gain the ability to research various machine learning and deep  learning algorithms along with its wrong mechanisms based on fraud credit cards and gain knowledge about the techniques which make complete algorithms.

## Objectives

* Getting information by proper research
* Understanding algorithms and its working mechanism
* Able to understand different algorithm based on CCFD
* Detect precision, recall, f1-score based on algorithm
* Understand the technique to handle imbalanced dataset
* Able to visualize the graph of dataset
* Create the report based on the project

## Academic Questions
There are certain questions arose during the planning of the proposed algorithms which are listed below:

* What sort of problem is this project going to solve?
* How actual is fraud detected?
* What are the challenges involved in developing an algorithm to detect the fraud card?
* Are there any similar projects?
* Is the proposed research feasible to handle an imbalanced dataset?

## Report Structure 

* Introduction: This section includes general information about the topic along with aims and objectives.
* Literature review: This section includes all the background research regarding the similar systems.
* Project plan: This section includes the detail plan to complete the project
* System Design: This section includes working flow to detect fraud card.
* Applied algorithms: This section includes the machine and deep algorithms used during development.
* Requirement specifications: This section includes functional and non-functional requirements of the project.
* Final application: This section includes the final execution of all machine learning and deep learning algorithms with confusion matrix and classification report.
* Answering academic questions: This section includes the answer to the academic questions.
* Conclusion: This section includes the conclusion to the entire project and future escalations.

## Major Steps Involved in Credit Card Fraud Transaction Models:

* The unavailability of full credit card details because it is private property and neither customer nor banks can disclose their information resulting in insufficient and under-trained models.
* A single powerful algorithm that can work reliably is impossible in any environment and can outperform any other algorithm.
* There is a lack of an efficient and competitive evaluation criterion that not only describes the efficiency of the system but also offers better comparative outcomes across several approaches.
* The Inability of the program to successfully respond to growing changes because modern misleading (fraudulent) technique and genuine improvements are made in user purchasing habits.

## Answer of Academic Questions

* What sort of problem is this project going to solve?<br>
Answer: Today’s era is the era of technology. So, as the increase in technology, the number of fraudulent transactions is rapidly increasing. The proposed research is based on detecting fraud cards and genuine card using different machine learning and deep learning techniques. During training and pre-processing, the machine learning and neural network is used to find user behavior and predict fraud card from data points.

* How actually fraud is detected?<br>
Answer: AI removes the time-consuming tasks and allows full data preprocessing within milliseconds and identifies complicated patterns in the most effective way to detect fraud credit card.

* What are the challenges involved in developing an algorithm to detect the fraud card?<br>
Answer: While doing research, the problem occurs where legal transactions appear just like illegitimate transactions, owing to certain circumstances. Illegitimate transactions could appear as legal transactions in another way. Most of the features have categorical data when analyzing the credit card data. So, handling categorical data is the most difficult part which means most of machine learning is not suitable to handle categorical data. The most difficult problem is to feature selection and choice of an algorithm that can detect fraud credit cards where an algorithm cannot detect the new type of data as fraud or normal transaction.

* Are there any similar projects?<br>
Answer: Yes, there are many similar researches regarding detecting fraud credit card. According to (Jain, et al., 2019), algorithms like SVM, ANN, KNN, Bayesian Network, Decision tree, and Logistic Regression are used to detect fraud credit cards. According to (Kumar, et al., 2019), algorithms like KNN, Random Forest, and Proposed Algorithm are used to identify the complex fraudulent pattern of credit cards. Similarly, NetVerify is a web application to prevent and detect fraud credit card which use machine learning, computer vision, and biometric facial recognition to see recognized patterns of human reviews.

* Is the proposed research feasible to handle an imbalanced dataset?<br>
Answer: Yes, the proposed research is feasible to handle imbalanced datasets because a technique algorithm known as SMOTE helps to produce minority synthetic samples and used to train the classifier.

## Following are the conclusion what I have got from the research

* How do you handle an imbalanced dataset?<br>
Answer: The obvious difficulty of addressing the class imbalanced is that one of the classes lacks records. Most machine learning algorithms with imbalanced datasets do not work very well. So, we prefer you to use SMOTE oversampling technique to handle an imbalanced dataset which helps to align the dataset by increasing the unusual sample size.

* How do you classify the feature and target class of the dataset?<br>
Answer: The dataset for your research is known as supervised data where the target class variable is dependent to feature class variables and feature class variable is independent variables. So, class 1 which is regarded as fraud class, and class 0 is regarded as normal transactions is the target variable and, the rest of the other are your feature class variables.

* On What basis you remove the null values from the dataset?<br>
Answer: One important step in data wrangling is the removal of null values from the dataset because it adversely affects any machine learning algorithm’s performance and accuracy. So, before implementing any machine learning algorithms to the dataset, it is really important to delete null values form the dataset.

* On what basis is the visualization allocated?<br>
Answer: Data visualization involved graph or map to facilitate the identification of trends, patterns, and outliers of broad datasets and it is important to promote the interpretation and analysis of human brain data. In your research visualization is allocated to find the distribution of fraud class and normal class, the relation of time with fraud and normal class, data after handling imbalanced dataset, the graph of features from v1 to v28.

* What is the curse of dimensionality and what are some ways to deal with it?<br>
Answer: The curse of dimensionality applies to the anomalies that arise while classifying, storing, and evaluating high-dimensional data not found in low dimensional spaces, especially the problem of data “closeness” and data “sparsity”. Dimensionality reduction is used to solve the curse of dimensionality by reducing the feature space.

* The dataset obtained from Kaggle contains only numerical input variables which are the result of PCA transformation. So, why is PCA needed in Machine Learning?<br>
Answer: PCA is an unsupervised and non-parametric mathematical method used in machine learning for predictive models and used mainly to reduce the dimensionality of a dataset with several variables compared with each other through maximizing accumulation with variations present in the dataset. Some of the applications of PCA in machine learning are listed below <br>
  - In lower-dimensional space, we can visualize the large complex data.
  - We can use it as a technique for selecting features.
  - For supervised learning issues, we will use the key components as data.

* What are the types of data mining techniques that can detect the actual card and fraudulent card? <br>
Answer: Logistic Regression, Bayesian Network, Hidden Markov Model, Decision Tree, Random Forest classifier are the types of data mining techniques that can detect the actual card and fraudulent card.

* How does one choose which algorithm is best suitable for the dataset at hand?<br>
Answer: To choose algorithm we are looking for precision and recall, specificity and sensitivity which are the accuracy measurement metrics of algorithm. Also, the ROC curve, TPR, and FPR can be used for algorithm selection.

* How to apply machine learning in fraud detection?<br>
Answer: To identify the fraud transactions, data collection in the machine learning model is the initial step and analyzes all the collected data, segments it, and extracts the features it requires. And the model finds the complex pattern of the training dataset which is consider as a fraud card.

* What are the factors I must consider before comparing the performance of two-meta algorithms applied to a problem?<br>
Answer: The speed of the convergence and convergence rate with the detection rate is one of the factors.

* Why do we need a validation set and a test set?<br>
Answer: In fact, the validation set is used to build a model and neural network which is considered as an integral part of the training set. Similarly, the training set is used for evaluating the performance of the model and neural network.

* On what basis is k-Fold cross-validation allocated?<br>
Answer: Cross-validation is a resampling process that is used on a small dataset to validate machine learning models. In another word, to use a limited sample to predict how the model is going to act normally and to draw conclusions regarding data not used during model testing.

* What are some factors that explain the success and recent rise of machine learning and deep learning?<br>
Answer: A large number of data accessible and strong processing capacity allows big business tospend massive capital in this technology. Rather than seeing this as the advent of emerging technologies, it is the product of significant corporate participation. Ultimately generate further jobs because of broad business participation and massive expenditure in science means that more individuals are drawn into it.

