# DL-and-RL
Self-Study Guide for Deep Learning and Reinforcement Learning
(I am building this guide as I study)

## Deep Learning 

### MOOC/Websites
- Courseara Andrew Ng Deep Learning Specialization [[link](https://www.coursera.org/specializations/deep-learning)]
- Convolutional Neural Networks for Visual Recognition [[CS2321n](http://cs231n.stanford.edu/)]
- Understanding LSTM Neworks [[RNN](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)]
- Tensorflow Tutorials [[Hvass](https://github.com/Hvass-Labs/TensorFlow-Tutorials)]
- A Tutorial on 3D Deep Learning [[link](http://3ddl.stanford.edu/)]
- 3D Deep Learning Workshop [[link](http://3ddl.cs.princeton.edu/2016/)]
- Machine Learning with Google Cloud Platform [[link]](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp)]


### Book
- Deep Learning, Ian Goodfellow, Yshua Bengio and Aaron Courville [[book](http://www.deeplearningbook.org/)]


### Papers

- Understanding deep learning requires rethinking generalization [[Zhang et al](https://arxiv.org/pdf/1611.03530.pdf)]
- Attention is all you need [[Vaswani et al.](https://arxiv.org/abs/1706.03762)]
- Faster R-CNN [[RPN](https://arxiv.org/abs/1506.01497)]

Convolutional Neural Network Architecture

- [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [VGG](https://arxiv.org/pdf/1409.1556.pdf)
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- Network in Network [[paper](https://arxiv.org/abs/1312.4400)]
- [Inception Network](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)

Distributed Network for Deep Learning

- [DistBelief](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
- Revisiting distributed synchronous SGD [[paper](https://arxiv.org/pdf/1604.00981.pdf)]

Computer Vision: Object Detection
- Retinanet
- YOLO
- SSD
- Faster RCNN
- Mask RCNN

3D Deep Learning
- Frustum PointNets for 3D Object Detection from RGB-D Data [[link](https://arxiv.org/pdf/1711.08488.pdf)]


## Reinforcement Learning

### MOOC/Websites
- Udacity Reinforcement Learning [[link](https://www.udacity.com/course/reinforcement-learning--ud600)]
- Temporal difference learning [lecture](http://videolectures.net/deeplearning2017_sutton_td_learning/)
- [BURLAP Tutorial](http://burlap.cs.brown.edu/tutorials/index.html)


### Book
-  Richard Sutton and Andrew Barto, Reinforcement Learning: An Introduction (2nd Edition Draft, 2017) [[Book]](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf)


### Papers
- Temporal difference learning [Sutton 1988](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.1503&rep=rep1&type=pdf)
- [Q Learning](https://link.springer.com/article/10.1007/BF00992698)
- Adaptive Heuristic Critic Method (AHC) 

Robotics/Motor Skills
- RL of motor skills with policy gradient [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.142.8735&rep=rep1&type=pdf)
- Efficient distributed RL through agreement [Varshavskaya et al](http://people.csail.mit.edu/lpk/papers/dars08.pdf)

Multi-Agent Reinforcement Learning
- Survey on Multi-agent RL [[Busoniu et al](http://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/07_019.pdf)]

Distributed & Scalable Systems
- Ray RLlib: A Composable and Scalable Reinforcement Learning Library [[Liang et al](https://arxiv.org/pdf/1712.09381.pdf)]

## Deep Reinforcement Learning

### MOOC/Website				
- Deep Reinforcement Learning, UC Berkeley [[CS294](http://rll.berkeley.edu/deeprlcourse/)]
- Deep RL Bootcamp, UCBerkeley & OpenAI [[link](https://sites.google.com/view/deep-rl-bootcamp/lectures)]
- Deep learning and reinforcement learning summer school [[lectures](http://videolectures.net/deeplearning2017_montreal/)]
- Deep Reinforcement Learning (John Schulman, OpenAI) [[Video](https://www.youtube.com/watch?v=PtAIh9KSnjo)]
- Tensorflow Tutorial #16 Reinforcement Learning [[link](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb)]


### Papers

Video Games
- Neural Fitted Q Iteration [[NFQ](https://pdfs.semanticscholar.org/2820/01869bd502c7917db8b32b75593addfbbc68.pdf)]
- Deep Q Network [DQN][[Nature](https://www.nature.com/articles/nature14236)]
- Deep Q Learning [[link](https://arxiv.org/abs/1312.5602)]
- Deterministic Deep Policy Gradient [[DDPG](https://arxiv.org/abs/1509.02971)] 
- Universal Value function Approximators [[UVFA](http://proceedings.mlr.press/v37/schaul15.pdf)]

Robotics
- End-to-end training of deep visuomotor policies [[Levin et al.](https://arxiv.org/abs/1504.00702)]
- Hindsight Experience Replay [[HER](https://arxiv.org/abs/1707.01495)]
- Sim-to-real transfer for robotic control with dynamics randomization [[paper](https://arxiv.org/pdf/1710.06537.pdf)]
- Domain randomization for sim-to-real transfer [[Tobin et al](https://arxiv.org/pdf/1703.06907.pdf)]
- Vision-based Multi-task Manipulation with Inexpensive Robot [[Rahmatizadeh et al](https://arxiv.org/abs/1707.02920)]
- DART: Noise Injection for Robust Imitation Learning [[Laskey et al](http://goldberg.berkeley.edu/pubs/DART-CoRL17-cam-ready.pdf)]
-

Surgical Robotics
- Multilateral Surgical Pattern Cutting with DRL [[link](http://goldberg.berkeley.edu/pubs/2017-icra-cutting-final.pdf)]
- (Unsupervised Learning) Transition State Clustering [[link](http://goldberg.berkeley.edu/pubs/krishnan-ijrr-submission-final.pdf)]
- Learning by Obsercation for Surgical Subtask [[Murali et al](http://goldberg.berkeley.edu/pubs/DVRK-Learning-icra2015.pdf)]

Scalable & Distributed Systems
- Parrellel Methods for DRL [[DeepMind Paper](https://arxiv.org/pdf/1507.04296.pdf)]
- DDRL through Agreement [[paper](http://people.csail.mit.edu/lpk/papers/dars08.pdf)]
- [Distributed Deep Q-Learning](https://stanford.edu/~rezab/classes/cme323/S15/projects/deep_Qlearning_report.pdf)
- Deep multi-user RL [[paper](https://arxiv.org/pdf/1704.02613.pdf)]
- Massively parallel methods for DRL [[paper](https://arxiv.org/pdf/1507.04296.pdf)]
- HORDE: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction [[]()]

Model-based deep reinforcement learning
- Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning

Inverse Reinforcement Learning/Inverse Optimal Control [Nagabandi et al.]
- Guided Cost Learning: Deep inverse optimal control via policy opimization [Finn et al]
- Generative Adversarial Imitation Learning [Ho and Ermon](https://arxiv.org/abs/1606.03476) [code](https://github.com/openai/imitation)


### Code
- openAI [baselines](https://github.com/openai/baselines)
- Neural Network Dynamics for Model-based deep reinforcement learning with model-free fine-tuning [code](https://github.com/nagaban2/nn_dynamics)
- End-to-end learning with deep visuomotor policies [code](http://rll.berkeley.edu/gps/)
- Vision-based multi task manipluation with inexpensive robot [code](https://github.com/rrahmati/roboinstruct-2)
