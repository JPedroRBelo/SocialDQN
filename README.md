# SocialDQN

SocialDQN is a deep reinforcement learning system based on Deep Q-Learning (DNQ) for robots that directly interact with humans. Currently, the main objective of SocialDQN is to allow the robot to learn to identify human interactive behaviors and, from this, to exercise socially acceptable actions.

This project is based on the work of Qureshi *et al* (2016), which involves modeling the Multimodal Deep Q-Network for Social Human-Robot Interaction (MDQN). Unlike MDQN, SocialDQN was developed in Python 3.8 and has support for social signals (emotions, focus of attention, visible human face), additional rewards, and support for training and validation in the SimDRLSR (Deep Reinforcement Learning Simulator for Social Robotics) simulator.


## Actions
- Wait: 
- Look:
- Handwave:
- Handshake:

## States: 
- 8 x grayscale images
- One-hot-vector: social signals information

## Reward:
- Succesfull Handshake
- Failed Handshake
- Succesfull Handwave (person look at robot)
- Failed Handwave
- Failed Episode

# Instalation




[1] Ahmed Hussain Qureshi, Yutaka Nakamura, Yuichiro Yoshikawa and Hiroshi Ishiguro, "Robot gains social intelligence through Multimodal Deep Reinforcement Learning", Proceedings of IEEE-RAS International Conference on Humanoid Robots (Humanoids), pp. 745-751, Cancun, Mexico 2016.
