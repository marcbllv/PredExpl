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

### How we can do it

What we want is to measure the influence of each feature and extract the most
important ones regarding a specific prediction.

Several methods has been published and are being developped. They can mainly be classified into model-dependant and model-independant explanation system.

#### Model dependant methods

The first solution is to look at how a given model actually works, and try to
extract each feature contribution.

#####Linear models

In a linear model, the final prediction is the sum of the feature value and the regression coefficient, plus some overall bias. So on a prediction level, multiplying each regression coefficient by the feature value give the direct influence of the feature value on the prediction.

***Image***








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

