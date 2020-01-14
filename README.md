# Project in Computer Science: Identifying and Categorizing Offensive Language in Social Media (OffensEval)

This project is about the development of a system for identifying and categorizing offensive language in social media, specifically Twitter. The project is based on SemEval 2020 Task 6, which is the second iteration of the OffensEval task organized at SemEval 2019. For this task, the Offensive Language Identification dataset (OLID) is used. The dataset contains English tweets annotated using a hierarchical three-level annotation model. For the development of the project, some different models were trained. This project is based on the Sub-task A, which consists of classifying tweets in Offensive and Not Offensive.

The baseline of the project is implementing a Logistic Regression classifier, which corresponds to the file *logistic_regression.py*. The rest of the project consists in implementing different deep learning models, which corresponds to the jupyter notebook *deep_learning.ipynb*.

There are also two example of how to run different machine learning models.

## Getting Started

Logistic Regression:

To run the program it is necessary to install the sklearn python module.

Deep Learning models:

To run the programm, it is necessary to run the *First steps* section in the jupyter notebook. It imports all the needed modules to run the system.

This code was run in Google CoLab. It is recommended to run the code in CoLab due to the high computational cost. 

### Prerequisites

Logistic Regression:
To run the program it is necessary to install:
- Sklearn python module.

Deep Learning models:
To run the program it is necessary to install:
- Keras python module.
- Sklearn python module.

You also need to download the dataset and the embedding files. The dataset is stored in the *OLID_dataset* directory. The embedding files cannot be upload because of their size. To download them follow the steps in the README file in the *embeddings* directory.

## Running the program

For the Deep Learning models, the python notebook is organized in different sections:
- First steps
- Data preprocessing
- Pre-trained word embeddings
- Training Deep Learning models
- Evaluating Deep Learning models

The program has to be run like the following. First, run the *First steps* code. Then, the *Data preprocessing* and the *Pre-trained word embeddings*. The next section is divided into different cells, which each cell corresponds to the initialization of a model. From all the cells, you only need to run the one related to the model you want to try. After running one of the several models, you need to run directly the code in the *Evaluating Deep Learning models* section. This final section will plot the results of the trained model using the validation and test dataset.

If you want to try an Ensemble model, you need to run first the models you want to ensemble. After running each model you have to save it in a list. To make predictions with the Ensemble model, follow the code in the subsection *Prediction for ensembling models*.

## License

This repository is covered by the license BSD 2-clause, see file LICENSE.

