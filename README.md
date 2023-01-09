# NIPS2022_EP_BNP
Official Implementation of NIPS 2022 paper Pre-activation Distributions Expose Backdoor Neurons.


## Abstract
Convolutional neural networks (CNN) can be manipulated to perform specific behaviors when encountering a particular trigger pattern without affecting the performance on normal samples, which is referred to as backdoor attack. The backdoor attack is usually achieved by injecting a small proportion of poisoned samples into the training set, through which the victim trains a model embedded with the designated backdoor. In this work, we demonstrate that backdoor neurons are exposed by their pre-activation distributions, where populations from benign data and poisoned data show significantly different moments. This property is shown to be attack-invariant and allows us to efficiently locate backdoor neurons. On this basis, we make several proper assumptions on the neuron activation distributions, and propose two backdoor neuron detection strategies based on (1) the differential entropy of the neurons, and (2) the Kullback-Leibler divergence between the benign sample distribution and a poisoned statistics based hypothetical distribution. Experimental results show that our proposed defense strategies are both efficient and effective against various backdoor attacks.

## Training

Run the following code to train a poisoned model:
```
python train.py --attack-type badnets --poisoning-rate 0.1 --trigger-size 3 --manual-seed 0
```
Feel free to change the hyperparameters to adapt to your desired settings.

## Testing
This repository also includes our proposed Channel Lipschitzness based Pruning (CLP).

Recover the model with EP, BNP and CLP:
```
python test.py --attack-type badnets --poisoning-rate 0.1 --trigger-size 3 --manual-seed 0
```
Note that the hypereparameters should be consistent with the previous trained model.

## Notification

If you are testing CLP on other network architectures, e.g. Preact ResNet or VGG, please accordingly change the code of CLP to better fit the architecture.

## Citation

In this repository, we have collected the code for our neuron pruning-based defenses against backdoor attacks. Please cite with the below bibTex if you find them helpful to your research.

```
@inproceedings{
zheng2022preactivation,
title={Pre-activation Distributions Expose Backdoor Neurons},
author={Runkai Zheng and Rongjun Tang and Jianze Li and Li Liu},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=wwW-1k1ljIg}
}

@inproceedings{zheng2022data,
  title={Data-free backdoor removal based on channel lipschitzness},
  author={Zheng, Runkai and Tang, Rongjun and Li, Jianze and Liu, Li},
  booktitle={European Conference on Computer Vision},
  pages={175--191},
  year={2022},
  organization={Springer}
}

```