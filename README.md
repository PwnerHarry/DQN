# DQN

A PyTorch Implementation of DQN from paper "Human-level Control through Deep Reinforcement Learning"
https://www.nature.com/articles/nature14236


This implementation uses the cpprb package for replay buffers, while being extremely loyal to the original hyperparameter settings.

Please note this implementation uses the suggested optimizer setting in the Rainbow paper, please take care of it if you want to reproduce the results exactly the same to the Nature paper.

Please note that if you want to change the replay buffer back to the classical one, do remember to change the hyperparameter "time_learning_starts" to 50000.

The implementation is validated on MsPacman, on which the it gives very similar results compared to those reported.
