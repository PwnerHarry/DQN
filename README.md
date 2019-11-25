# DQN

A PyTorch Implementation of DQN from paper "Human-level Control through Deep Reinforcement Learning"
https://www.nature.com/articles/nature14236
This implementation contains the classical replay buffer as well as the prioritized version (default), while being extremely loyal to the original hyperparameter settings.
Please note that if you want to change the replay buffer back to the classical one, do remember to change the hyperparameter "time_learning_starts" to 50000.