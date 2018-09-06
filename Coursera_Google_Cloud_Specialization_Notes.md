Machine Learning with TensorFlow on Google Cloud Platform
https://www.coursera.org/specializations/machine-learning-tensorflow-gcp

Course 1. How Google does Machine Learning
Module 2: What it means to be AI first
-pretrained model is a good option or a starting point for some ML problems
-ML is not just data statistics, it is also a logic. Anytime when rules are complicated to handcraft, you can create one using ML = scaling beyond handwritten rules
-think of ML as building API's  (i.e. NPL API, Vision API)
-ML can be used to customize services for each customer (i.e. personalize, recommendation systems, user interfaces)
-manual data analysis helps building successful ML. gives you good insight to your data and to the model you will have to build.
 
Module 3: How Google does AI
Steps to building ML systems for your business
defining KPI's
collecting data (hardest1)
building infrastructure (hardesst2): making sure that model can be trained many times automatically and smoothly, making sure model can be served at scale for end users
optimizing ML algorithm
integration
 
-use off the shelf algorithms especially for NLP and Perception. It's too costly to try to design your own.
 
Path to ML
Individual contributor : build prototype and try ideas
Delegation: ramp up to include more people (formalize the process, diversity in human responses). Great ML systems will need humans in the loop (reviewing algorithms, response, etc)
Digitization: automate mundane parts of the process, build infrastructure (i.e. method to  extract data), separate software/IT to ML project
Big Data and Analytics: measure and achieve data-driven success, reiterate define success (what users want, , clean or organize data for ML models, tune ML algorithm
Machine Learning: automate feedback loop that can outpace human scale, performance indicator (i.e. one brain that can learn from many interactions (one box picking cognition that can learn from performing at different sites)
-We can produce up to 3 without partner or customers.
-You need a structured approach to building ML product for your business 
 
Building Use Cases
If the use case was an ML problem....
1) What is being predicted?
2) What data is needed?

Now imagine the ML problem is a question of software:
3) What is the API for the problem during prediction?
4) Who will use this service? How are they doing it today?

Lastly, cast it in the framework of a data problem. What are some key actions to collect, analyze, predict, and react to the data/predictions (different input features might require different actions)
5) What data are we analyzing?
6) What data are we predicting?
7) What data are we reacting to?


Module 4: Inclusive ML
-bias in ML
Interaction bias
Latent bias
Selection bias
-build more inclusive ML
Evaluate your model over subgroups to identify areas of improvement
Confusion Matrix (model predictions (+/-) vs Labels (+/-))
https://en.wikipedia.org/wiki/Sensitivity_and_specificity
-Facets : tool for accessing biases in ML data , quick understanding of the distributions of values across features of their dataset
 
Module 5: Python notebook in the cloud
-Datalab = jupyter notebook on cloud running on vm. Can work with git to store in dm or cloud
https://github.com/GoogleCloudPlatform/datalab-samples/blob/master/basemap/earthquakes.ipynb
-In ML, outliers are learned not removed. So you need to work with all data, not just sampled data to create a good model.l.
-API has pre-trained models (Vision, Vide, Speech, Cloud Translation, Natural Language,


Course 2. Launching into Machine Learning
Module 1: Practical ML
-which model to use? Do you have data? If so, does it have finite number of class?
-machine learning is all about experimentation!
-data
Unstructured data
Structured data
-supervised : has labels
Regression (continuous label)
Loss function: Mean square error ( L =||y - Xw||^2 ) , root mean squared error
Linear Regression ( y = Xw, w = weights)
Classification (discrete number of values/labels)/
Loss function: Cross entropy
Perceptron (binary classifier)
-unsupervised: don't have labels or ground truth. It's all about discovery
Clustering
 
ML History
-Linear Regression (1800s)
Gradient descent & gradient descent optimizer
Learning Rate (hyperparameter for optimizing gradient descent method)
Bias and weights (parameters to be learned and optimized)
-Perceptron (1940)s : but cannot learn XOR function
Step-wise (all or nothing) activation function
-Neural Network (1960s)
Combine layers of perceptron
Activation function
Linear
Sigmoid (good for predicting probabilities) 0 to 1
Tanh (might have saturation point) -1 to 1
ReLU (negative input space = zero space)  0 to +inf
ELU (more computationally expensive) -inf to +inf
More neuron -> higher dimensional feature space
-Decision Trees (1980s)
Classification
Not good for generalization
-Kernel methods (1990s)
For classes of new nonlinear models 
Nonlinear Support Vector Machines (SVMs)
Nonlinear activation + sigmoid output
-Random Forests (2000s)
Boosted Trees
Classification
Regression
Random sampling to improve generalization
Out-of-Bag data
Random Patches
Stacking / meta-learners
Bias similar to individual tree
-Modern Neural Networks (2010s)
-improved via more data and higher computational power
-Deep Neural Network
-dropout layers
-good validation testing data
-most important : generalization
-main takeaway:
ML research reuses bits and pieces of techniques from other algorithms from the past to combine together to make ever powerful models and most importantly experiments
Experimentation is a key to getting the best performance using your data to solve your challenge
 
Module 2: Introduction to Optimization
-ML models = parameters (updates/learns) + hyper-parameters (set before training)
-Learning model
Goal: search for best parameter with min loss
Linear model(2 parameters, 2p) : bias and weight
RMSE
Cross entropy = sum of  positive term + negative term
-Gradient descent (optimizer)
Which direction?
Step size?
Search for a minima by descending the gradient
A correct and constant step size can be proven difficult
Loss function slope provides direction and step size in search
Scaling hyper parameter = learning rate. It's fixed during training
-To improve training time
mini batches (10-1000 examples: batch size (hyperparameter)
Frequency to check the loss. Not ideal method
TensorFlow playground link: https://goo.gl/EEuEGp
Model(network architecture) itself does feature engineering
-Performance Metrics
Identify Inappropriate Minima
Don't reflect relationship btw features and label
Won't generalize well
Confusion matrix (Reference (neg/pos) vs. Model Prediction (pos/neg)
Precision (high True  positive/ all (truth and false) positive)
Recall ( True positive / (True Positive + False Negative))

Module 3: Generalization and Sampling
When is the most accurate model not to pick?
Best model is a model that perform best in the unseen data
-Generalization
Split data into training set, validation test, and test set
In order to avoid getting entirely new data,
Split each set into multiple set -> bootstrapping / cross validation (good for little data)
Overfit or memorizing data set, BAD
-Tuning hyperparameter
Chose hyper parameter or a model that does better with the validation data not training data (lower loss)
If it fail at testing, you need to get new data for training and validation and teat
 
-Google BigQuery to split datasets  -> creating repeatable samples
Use Hash Function
Rand() is dangerous, bc each sample needs to contain distinct element
Need noisy and well distributed dataset to ensure each hash is unique to a element
Carefully chose which field will split your data
It needs to be independent of a feature you need to make a prediction
Develop tensorflow code on a small subset of data, then scale it out to the cloud
Sample rather than take first N instances bc data might be sorted
Data preprocessing; removing data that doesn't make sense or re-calculating for desirable label/prediction
Save those data samples into files(ex. .CSV)
-Well thought out benchmark is important in evaluating ML performance
Exploring .> Creating those data sets -> Benchmarking (via very simple model calculate a RMSE to beat. New (DNN) model will have to beat this)


Course 3. Intro to TensorFlow
Module 1: Intro to TensorFlow
You can train on a cloud, and use trained model on portable device with less compute power. You can update the model remotely or personalize the model based on user's usage aka fine tune.
-Debugging
Eager
Shape problems [rows, columns]
Tf.reshape()
Tf.expand_dims()
Tf.slice()
Tf.squeeze()
Data type problems
Tf.cast()
Debugging full programs
Tf.logging.set_verbosity()
Tf.Print()  #to log specific tensor values
Tfdbg  # in terminal window. (ex) python xyz.py --debug
TensorBoard
-Visualization
TensorBoard

Module 2: Estimator API
-Estimator API in TensorFlow
Pick features to feed into model
Pick estimators
-Dataset API
Organize datasets into epochs and batches
-distributed training
Tf.estimator.train_and_estimate()
Tf.estimator.evalspec()
-Tensorboard
Tf.summary
Watch training, eval loss curve, activation histograms,
Inserting serving input function to convert raw input data to a format that can be fed into a model for making prediction


Course 4. Feature Engineering

Course 5. Art and Science of Machine Learning
