from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

class LimeWrapper:

    def __init__(self, train, model, target_column, categorical=[],
                 kernel_width=None, discretizer='quartile', verbose=False):
        if not isinstance(train, pd.DataFrame):
            raise Exception('Use a pandas DataFrame to store the train set')

        train.fillna(0.0)

        # Handling target
        if target_column not in train.columns:
            raise Exception('Target column not in train set.')
        target = train[target_column]
        train_x = train.drop(target_column, axis=1)

        # Check discretizer values
        if discretizer == None:
            discretize = False
        else:
            discretize = True
            if discretizer not in ['quartile', 'decile', 'entropy']:
                discretizer = 'quartile'

            if discretizer == 'entropy':
                training_labels = target
            else:
                training_labels = None


        # Features - detecting categoricals and label encoding
        features = train_x.columns.tolist()
        cat = train_x.select_dtypes(include=[object]).columns.tolist() 
        cat = cat + categorical
        cat_idx = [features.index(c) for c in cat]

        categorical_names = {}
        for c in cat_idx:
            feat = features[c]
            le = LabelEncoder()
            train_x[feat] = le.fit_transform(train_x[feat])
            categorical_names[c] = le.classes_

        # Classif or Reg ? Classif if less than 20 different values among first 1000 rows
        if target.values.dtype.type == np.string_:
            pred_type = 'classification'
        else:
            pred_type = 'classification' if np.unique(target[:1000]).shape[0] < 20 else 'regression'

        if pred_type == 'classification':
            class_names = np.unique(target)
            self.labels = [1]
            self.predict_fn = lambda x: model.predict_proba(x)
        else:
            class_names = ['lower', 'higher']
            self.labels = [0]
            self.predict_fn = lambda x: model.predict(x).reshape(-1, 1)

        # setting up the explainer
        self.explainer = LimeTabularExplainer(train_x.values,
                feature_names=features, 
                class_names=class_names,
                categorical_features=cat_idx,
                categorical_names=categorical_names,
                kernel_width=kernel_width,
                discretize_continuous=discretize,
                discretizer=discretizer, 
                training_labels=training_labels,
                verbose=verbose)

    def explain(self, instance, num_features=10, num_samples=1000, labels=None, show=True):     
        if not isinstance(instance, pd.Series):
            raise Exception('Use a pandas Serie to store the train set')

        exp = self.explainer.explain_instance(instance.values, self.predict_fn,
                num_features=num_features, num_samples=num_samples,
                labels=self.labels)
        if show:
            exp.show_in_notebook()
                    
        return exp
