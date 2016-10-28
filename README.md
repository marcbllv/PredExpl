# Prediction explanations

### Why explaining individual predictions

Machine learning models and algorithms often come as black boxes for the end
user. They are good at finding mappings between a set of features and a target,
but they do not really describe the underlying relationship between a given set
of feature and the prediction made by the model.

In many cases, what we would be interested in would be to know not only the
prediction, but also *why* such a prediction has been made by the model, on which
feature the model based it prediction, and what to turn the model output into
something else.

Typical examples could be:

- In fraud detection: why has this person been predicted as a fraudster ?
- In predictive maintenance: why is it supposed to break, and what can we do about it ?
- In pricing prediction: which features tends to lower or increase the price ?

This is going to become a field of major importance soon as the European Union will impose a right to explanation for citizens: when a decision that concerns directly an individual will be taken by an algorithm, this person could ask the decision maker why such a decision has been made. In Machine Learning, this means for companies to be able to extract explanations automatically even from black-boxes algorithms.

### How we can do it

What we want is to measure the influence of each feature and extract the most
important ones regarding a specific prediction.

Several methods has been published and are being developped. They can mainly be classified into model-dependant and model-independant explanation system.

#### Model dependant methods

The first solution is to look at how a given model actually works, and try to
extract each feature contribution.

#####Linear models

In a linear model, the final prediction is the sum of the feature value and the regression coefficient, plus some overall bias. So on a prediction level, multiplying each regression coefficient by the feature value give the direct influence of the feature value on the prediction.

Moreover, the coefficient directly gives an idea on how the prediction is going to evolve when perturbing a feature.

***Image***

##### Tree-based algorithms

For a decision tree, extracting features contributions means check the path in the tree to reach a given point.

***Image***

Each split on the path moves the prediction up or down, and at each split a single feature is involved. So computing the output difference at each split will indicate the feature influence for that split. For a given feature, we can sum up the contributions for all the splits this is feature is involved.

$$P(y = red | x1, x2) = Global_Bias + \sum_f contribution_f(x)$$











This notebook will describe a method developped by Marco Tulio Ribeiro et Al.
in this article: ["Why Should I Trust You?": Explaining the Predictions of Any
Classifier](https://arxiv.org/abs/1602.04938). Their explainer, called LIME,
will be applied to the [German Credit
dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)
from UCI. This dataset features several banking features about individuals
willing to get a loan, and a target variable indicating whether the bank should
lend the money or not. The target is **1 for risky borrower**, and **0 for non
risky borrowers**.

#### Imports & data retrieval

Let's import some packages & the data

