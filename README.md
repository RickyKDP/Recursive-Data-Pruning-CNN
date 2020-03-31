# Recursive Data Pruning Convolutional Neural Network (ReDP-CNN)
This project is the source code of the paper, Improving Convolutional Neural Network for Text Classification by Recursive Data Pruning.

Requierment: Tensorflow >= 1.12.0

The proposed recursive data-pruning strategy works as follows: after the standard training, all the convolutional filters are evaluated in terms of the discriminative power of the features generated in the pooling layer. The filters with low evaluation scores are then identified, and words captured by the low-scored filters are pruned from the training data. This procedure is iterated and task-irrelevant words are eliminated recursively. In the end, the CNN trained by the cleaned data will fit to task-relevant words only, and this in turn leads to better generation. The overall procedure is below:

![image](http://github.com/RickyKDP/Recursive-Data-Pruning-CNN/overall.eps)

