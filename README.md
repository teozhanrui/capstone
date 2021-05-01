# Capstone: Classifying Multilabel Research Articles
completed on 29th April 2021

#### Author: Teo Zhan Rui

This classification task is a part of a hackathon hosted by Analytics Vidhya. 

https://datahack.analyticsvidhya.com/contest/janatahack-independence-day-2020-ml-hackathon/#ProblemStatement


With the explosion of data, research articles are also readily available on the internet. In order to aid in search, articles need to be labeled according to their relevant topics. Multilabel classification poses additional challenges on top of multiclass problems as the targets are no longer mutually exclusive. An article can have more than 1 label akin to how a movie can be of action and comedy genre at the same time. 

In notebooks 1a) to 1c), I begin with Sklearn models to get a baseline micro f1-score. This is the metric chosen by the hackathon organizer and shall also be for evaluation of my models. Sklearn models support only binary or multiclass classification. In order to predict each of the 6 target labels, 6 models have to be built using the OneVsRest Classifier wrapper. Fundamentally, they are still not multilabel models. 

In notebooks 2a) to 2d), I explore 6 different neural networks with increasing complexities and then evaluate the best performing model. 

Word2Vec models, neural network models and pretrained models are stored on google drive and not on this respository due to size limit. Please access them at 

https://drive.google.com/drive/folders/1lh1orPb1A5k4gGoLMBgR4uIMJyaahwUR?usp=sharing

My final micro-F1 score on Analytic Vidhya is 0.839, compared to the overall leading score of 0.861. This is a verified score by Analytic Vidhya but not an official submission as the hackathon ended in August 2020. Please see notebook 2d) for a submission screenshot.

## Problem Statement

1) What are the challenges of multilabel classification?

2) What are the suitable metrics for measuring performance?

3) Which model produced the best scores and are suitable for deployment?

## Executive Summary

**Results**

| library | model                                      |     type    | micro F1 | hamming loss | classes with nil prediction | lowest recall |
|---------|--------------------------------------------|:-----------:|:--------:|:------------:|:---------------------------:|:-------------:|
| sklearn | logistic regression                        | one vs rest |   0.799  |    0.0709    |              0              |     0.041     |
| sklearn | support vector machine                     | one vs rest |   0.811  |    0.0756    |              0              |     0.016     |
| keras   | simple neural network without word vectors |  multilabel |   0.798  |    0.0825    |              2              |     0.000     |
| keras   | simple RNN with LexVec                     |  multilabel |   0.677  |    0.1205    |              3              |     0.000     |
| keras   | LSTM and GRU with LexVec                   |  multilabel |   0.807  |    0.0776    |              0              |     0.066     |
| ktrain  | biGRU                                      |  multilabel |   0.82   |    0.0733    |              0              |     0.356     |
| ktrain  | BERT                                       |  multilabel |   0.825  |    0.0729    |              0              |     0.352     |
| ktrain  | distilBERT                                 |  multilabel |   0.833  |    0.0702    |              0              |     0.557     |

**Final Model Selection: distilBERT** <br>
The best performing model produced the highest micro F1 and lowest hamming loss. Also the lowest recall among 6 classes is **0.557** for the most poorly classified Quantitative Biology class, which is a marked improvement over all other models.

precision    recall  f1-score   support

    Computer Science      0.819     0.884     0.850      1692
             Physics      0.911     0.874     0.892      1226
         Mathematics      0.852     0.774     0.811      1150
          Statistics      0.776     0.809     0.792      1069
Quantitative Biology      0.602     0.557     0.579       122
Quantitative Finance      0.889     0.711     0.790        45

           micro avg      0.833     0.834     0.833      5304
           macro avg      0.808     0.768     0.786      5304
        weighted avg      0.835     0.834     0.833      5304
         samples avg      0.860     0.869     0.845      5304

hamming loss : 0.07024235200635677 

**Conclusions:**

1) Multilabel classification is an extension from binary or multiclass classification. However this project showed that good results can no longer be obtain easily from basic models. This is further exacerbated by presence of 2 minority classes of less than 3%.

2) Accuracy is not a suitable metric as a prediction is considered wrong for 1 single misclassified label out of any 6. Hence it is very punitive. 

Micro F1, which is the competition metric chosen by the hackathon organizers, measures all round performance. On the flipside, it can mask poor performance on minority. The recall score should be observed for every class. 

Hamming loss is easily understood and can be easily apply on individual prediction or across all predictions. 

3) Both distilBERT and BERT produced the best performances but BERT's space requirement is prohibitive. In addition, the output model of BERT is relatively large at 1.3 GB and that of distillBERT is about 250 MB.

**Suggested Further Study**
1) Tune other hyperparameters such as optimizer and learning rates. 

2) Summarize using BART or T5 text to half the length and repeat training using BERT. 

3) Relabel or remove the data with obviously wrong ground truths. 

#### Sources: 

https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/ 
https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
https://towardsdatascience.com/nlp-101-word2vec-skip-gram-and-cbow-93512ee24314
https://github.com/adam0ling/twitter_sentiment
https://github.com/amaiya/ktrain/blob/master/tutorials/tutorial-04-text-classification.ipynb


