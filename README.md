# A Machine Learning Approach for the Diagnosis of Parkinson's Disease via Speech Analysis

## Introduction
- Parkinson’s Disease is the second most prevalent neurodegenerative disorder after Alzheimer’s, affecting more than 10 million people worldwide. Parkinson’s is characterized primarily by the deterioration of motor and cognitive ability.
- There is no single test which can be administered for diagnosis. Instead, doctors must perform a careful clinical analysis of the patient’s medical history. 
- Unfortunately, this method of diagnosis is highly inaccurate. A study from the National Institute of Neurological Disorders finds that early diagnosis (having symptoms for 5 years or less) is only 53% accurate. This is not much better than random guessing, but an early diagnosis is critical to effective treatment.
- Because of these difficulties, I investigate a machine learning approach to accurately diagnose Parkinson’s, using a dataset of various speech features (a non-invasive yet characteristic tool) from the University of Oxford.
- WHY SPEECH FEATURES? Speech is very predictive and characteristic of Parkinson’s disease; almost every Parkinson’s patient experiences severe vocal degradation (inability to produce sustained phonations, tremor, hoarseness), so it makes sense to use voice to diagnose the disease. Voice analysis gives the added benefit of being non-invasive, inexpensive, and very easy to extract clinically.

## Background
### Parkinson's Disease
- Parkinson’s is a progressive neurodegenerative condition resulting from the death of the dopamine containing cells of the substantia nigra (which plays an important role in movement). 
- Symptoms include: “frozen” facial features, bradykinesia (slowness of movement), akinesia (impairment of voluntary movement), tremor, and voice impairment.
- Typically, by the time the disease is diagnosed, 60% of nigrostriatal neurons have degenerated, and 80% of striatal dopamine have been depleted. 

### Performance Metrics
- * TP = true positive, FP = false positive, TN = true negative, FN = false negative
- Accuracy: (TP+TN)/(P+N)
- Matthews Correlation Coefficient: 1=perfect, 0=random, -1=completely inaccurate 

### Algorithms Employed
- Logistic Regression (LR): Uses the sigmoid logistic equation 11+exwith weights (coefficient values) and biases (constants) to model the probability of a certain class for binary classification. An output of 1 represents one class, and an output of 0 represents the other. Training the model will learn the optimal weights and biases.
- Linear Discriminant Analysis (LDA): Assumes that the data is Gaussian and each feature has the same variance. LDA estimates the mean and variance for each class from the training data, and then uses properties of statistics (Bayes theorem , Gaussian distribution, etc) to compute the probability of a particular instance belonging to a given class. The class with the largest probability is the prediction.
- k Nearest Neighbors (KNN): Makes predictions about the validation set using the entire training set. KNN makes a prediction about a new instance by searching through the entire set to find the k “closest” instances. “Closeness” is determined using a proximity measurement (Euclidean) across all features. The class that the majority of the k closest instances belong to is the class that the model predicts the new instance to be.
- Decision Tree (DT): Represented by a binary tree, where each root node represents an input variable and a split point, and each leaf node contains an output used to make a prediction.
- Neural Network (NN): Models the way the human brain makes decisions. Each neuron takes in 1+ inputs, and then uses an activation function to process the input with weights and biases to produce an output. Neurons can be arranged into layers, and multiple layers can form a network to model complex decisions. Training the network involves using the training instances to optimize the weights and biases. 
- Naive Bayes (NB): Simplifies the calculation of probabilities by assuming that all features are independent of one another (a strong but effective assumption). Employs Bayes Theorem: to calculate the probabilities that the instance to be predicted is in each class, then finds the class with the highest probability.
- Gradient Boost (GB): Generally used when seeking a model with very high predictive performance. Used to reduce bias and variance (“error”) by combining multiple “weak learners” (not very good models) to create a “strong learner” (high performance model). Involves 3 elements: a loss function (error function) to be optimized, a weak learner (decision tree) to make predictions, and an additive model to add trees to minimize the loss function. Gradient descent is used to minimize error after adding each tree (one by one). 

