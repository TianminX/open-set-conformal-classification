import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn import calibration
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.stats import weibull_min
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample

import copy


class Oracle:
    def __init__(self, model):
        self.model = model
    
    def fit(self,X,y):
        return self

    def predict(self, X):
        return self.model.sample(X)        

    def predict_proba(self, X):
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        prob = self.model.compute_prob(X)
        prob = np.clip(prob, 1e-6, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob


class SVC:
    def __init__(self, calibrate=False,
                 kernel = 'linear',
                 C = 1,
                 clip_proba_factor = 0.1,
                 random_state = 2020):
        self.model = svm.SVC(kernel = kernel,
                             C = C,
                             probability = True,
                             random_state = random_state)
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y)) 
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        else:
            self.calibrated = None
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):        
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor/self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob

class RFC:
    def __init__(self, calibrate=False,
                 n_estimators = 1000,
                 criterion="gini", 
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=1,
                 clip_proba_factor=0.1,
                 random_state = 2020):
        
        self.model = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                     criterion=criterion,
                                                     max_depth=max_depth,
                                                     max_features=max_features,
                                                     min_samples_leaf=min_samples_leaf,
                                                     random_state = random_state)
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y)) 
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        else:
            self.calibrated = None
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):        
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor/self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob

class NNet:
    def __init__(self, calibrate=False,
                 hidden_layer_sizes = 64,
                 batch_size = 128,
                 learning_rate_init = 0.01,
                 max_iter = 20,
                 clip_proba_factor = 0.1,
                 random_state = 2020):
        
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                   batch_size=batch_size,
                                   learning_rate_init=learning_rate_init,
                                   max_iter=max_iter,
                                   random_state=random_state)
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y)) 
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        else:
            self.calibrated = None
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):        
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor/self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob


# class RFC_many:
#     def __init__(self, calibrate=False,
#                  n_estimators=200,            # changed
#                  criterion="gini",
#                  max_depth=15,                # changed
#                  max_features="sqrt",         # changed
#                  min_samples_leaf=2,          # changed
#                  clip_proba_factor=0.1,
#                  class_weight='balanced_subsample',
#                  random_state=2020):
#
#         self.model = ensemble.RandomForestClassifier(
#             n_estimators=n_estimators,
#             criterion=criterion,
#             max_depth=max_depth,
#             max_features=max_features,
#             min_samples_leaf=min_samples_leaf,
#             class_weight=class_weight,
#             random_state=random_state,
#             n_jobs=-1               # important for parallel speed
#         )
#         self.calibrate = calibrate
#         self.num_classes = 0
#         self.factor = clip_proba_factor
#
#     def fit(self, X, y):
#         self.num_classes = len(np.unique(y))
#         self.model_fit = self.model.fit(X, y)
#         if self.calibrate:
#             self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
#                                                                  method='sigmoid',
#                                                                  cv=10)
#         else:
#             self.calibrated = None
#         return copy.deepcopy(self)
#
#     def predict(self, X):
#         return self.model_fit.predict(X)
#
#     def predict_proba(self, X):
#         if (len(X.shape) == 1):
#             X = X.reshape((1, X.shape[0]))
#         if self.calibrated is None:
#             prob = self.model_fit.predict_proba(X)
#         else:
#             prob = self.calibrated.predict_proba(X)
#         prob = np.clip(prob, self.factor / self.num_classes, 1.0)
#         prob = prob / prob.sum(axis=1)[:, None]
#         return prob


class RFC_many:
    def __init__(self, calibrate=False,
                 n_estimators=200,
                 criterion="gini",
                 max_depth=15,
                 max_features="sqrt",
                 min_samples_leaf=2,
                 clip_proba_factor=0.1,
                 class_weight='balanced_subsample',
                 random_state=2020):
        """
        Initializes the RandomForest classifier with the given parameters.

        Parameters:
          - calibrate (bool): Whether to apply probability calibration.
          - n_estimators, criterion, max_depth, max_features, min_samples_leaf, class_weight:
            parameters passed to RandomForestClassifier.
          - clip_proba_factor (float): Factor to clip the predicted probabilities.
          - random_state (int): Random seed.
        """
        self.model = ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        self.calibrated = None  # To store the calibrated model if calibration is used

    def fit(self, X, y):
        """
        Fits the RandomForest model to the data. Optionally applies calibration.

        Parameters:
          - X (array-like): Feature matrix.
          - y (array-like): Target vector.

        Returns:
          - A deep copy of the fitted classifier.
        """
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='sigmoid',
                cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts class labels for the input samples.

        Parameters:
          - X (array-like): Input samples.

        Returns:
          - array-like: Predicted class labels.
        """
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """
        Predicts class probabilities for the input samples.

        When y_calib is provided (e.g. during calibration), the method computes the union
        of training labels (self.classes_) and the unique labels in y_calib. For every extra
        label (i.e. a label in y_calib not seen during training), a noise probability is generated.
        The noise for each extra label for each sample is drawn from Uniform(low, high), where
        low = (p_min/4) and high = (p_min/2) and p_min is the smallest predicted probability
        (over the seen training classes) for that sample. Finally, the augmented probability
        vector is renormalized to sum to 1.

        When y_calib is not provided (e.g. during test-time prediction), if the union of labels
        was computed previously (stored in self.full_classes), it is used to add extra columns.

        Parameters:
          - X (array-like): Input samples.
          - y_calib (array-like, optional): True labels for the samples (used to determine extra unseen labels).

        Returns:
          - new_prob (array-like): An array of shape (n_samples, #unique(Y_train) + #extra unseen labels)
            containing the probabilities for each label in the union of Y_train and y_calib.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Obtain predicted probabilities for training classes.
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        # Clip probabilities to avoid extremely low values and renormalize.
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1, keepdims=True)

        n, K = p_seen.shape  # Here K should equal self.num_classes

        # Determine the extra unseen labels.
        if y_calib is not None:
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            self.full_classes = np.concatenate([self.classes_, extra])
        else:
            if hasattr(self, 'full_classes'):
                extra = self.full_classes[self.num_classes:]
            else:
                extra = np.array([])

        extra_count = len(extra)
        if extra_count > 0:
            new_prob = np.empty((n, K + extra_count))
            for i in range(n):
                p_row = p_seen[i]
                p_min = p_row.min()  # smallest seen probability in the row
                # Generate one noise value per extra unseen label.
                noise = np.random.uniform(low=p_min / 4, high=p_min / 2, size=extra_count)
                augmented = np.concatenate([p_row, noise])
                new_prob[i] = augmented / augmented.sum()
        else:
            new_prob = p_seen

        return new_prob


class KNN:
    def __init__(self, calibrate=False,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1):
        """
        Initializes the KNN classifier with the given parameters.

        Parameters:
        - calibrate (bool): Whether to apply probability calibration.
        - n_neighbors (int): Number of neighbors to use.
        - weights (str or callable): Weight function used in prediction.
        - algorithm (str): Algorithm used to compute the nearest neighbors.
        - leaf_size (int): Leaf size passed to the underlying tree algorithm.
        - p (int): Power parameter for the Minkowski metric.
        - metric (str or callable): Metric to use for distance computation.
        - metric_params (dict): Additional keyword arguments for the metric function.
        - n_jobs (int): Number of parallel jobs to run.
        - clip_proba_factor (float): Factor to clip the predicted probabilities.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs
        )
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        self.calibrated = None  # To store the calibrated model if calibration is used

    def fit(self, X, y):
        """
        Fits the KNN model to the data. Optionally applies calibration.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target vector.

        Returns:
        - self: Fitted KNN instance.
        """
        self.num_classes = len(np.unique(y))
        self.model_fit = self.model.fit(X, y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='sigmoid',
                cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts the class labels for the input samples.

        Parameters:
        - X (array-like): Input samples.

        Returns:
        - array-like: Predicted class labels.
        """
        return self.model_fit.predict(X)

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input samples.

        Parameters:
        - X (array-like): Input samples.

        Returns:
        - array-like: Predicted class probabilities.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)

        # Clip probabilities to avoid extremely low values
        prob = np.clip(prob, self.factor / self.num_classes, 1.0)
        # Normalize to ensure probabilities sum to 1
        prob = prob / prob.sum(axis=1)[:, None]

        return prob

class KnnUnseenCalib:
    def __init__(self, calibrate=False,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1):
        """
        Initializes the KNN classifier with the given parameters.

        Parameters:
          - calibrate (bool): Whether to apply probability calibration.
          - n_neighbors, weights, algorithm, etc.: parameters passed to KNeighborsClassifier.
          - clip_proba_factor (float): Factor to clip the predicted probabilities.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs
        )
        self.calibrate = calibrate
        self.factor = clip_proba_factor
        self.calibrated = None  # To store the calibrated model if calibration is used

    def fit(self, X, y):
        """
        Fits the KNN model to the data. Optionally applies calibration.

        Parameters:
          - X (array-like): Feature matrix.
          - y (array-like): Target vector.

        Returns:
          - A deep copy of the fitted KNN instance.
        """
        # Store the training labels (sorted) and their count.
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='sigmoid',
                cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts the class labels for the input samples.

        Parameters:
          - X (array-like): Input samples.

        Returns:
          - array-like: Predicted class labels.
        """
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """
        Predicts class probabilities for the input samples.

        When y_calib is provided (e.g. during calibration), the method computes
        the union of training labels (self.classes_) and the unique labels in y_calib.
        For every extra label (i.e. a label in y_calib not seen during training),
        a noise probability is generated. The noise for each extra label for each sample
        is drawn from Uniform(0, p_min), where p_min is the smallest predicted probability
        (over the seen training classes) for that sample. Finally, the augmented probability
        vector is renormalized so that it sums to 1.

        When y_calib is not provided (e.g. during test-time prediction), if the union of labels
        was computed previously (stored in self.full_classes), it is used to add extra columns.

        Parameters:
          - X (array-like): Input samples.
          - y_calib (array-like, optional): True labels for the samples (used to determine
            extra unseen labels).

        Returns:
          - new_prob (array-like): An array of shape
              (n_samples, #unique(Y_train) + #extra unseen labels)
            containing the probabilities for each label in the union of Y_train and Y_calib.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Obtain predicted probabilities for training classes.
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        # # Clip probabilities to avoid extremely low values and renormalize.
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]


        n, K = p_seen.shape   # Here K should equal self.num_classes

        # Determine the extra unseen labels.
        if y_calib is not None:
            # Compute extra labels as those in y_calib that are not in training.
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            # full_classes is the union of training classes and extra labels.
            self.full_classes = np.concatenate([self.classes_, extra])
        else:
            # If no y_calib is provided, but we already computed the union during calibration,
            # use the extra columns saved in self.full_classes.
            if hasattr(self, 'full_classes'):
                extra = self.full_classes[self.num_classes:]
            else:
                extra = np.array([])

        extra_count = len(extra)
        if extra_count > 0:
            new_prob = np.empty((n, K + extra_count))
            for i in range(n):
                p_row = p_seen[i]
                p_min = p_row.min()  # use the smallest seen probability as scale
                # Generate one noise value per extra unseen label.
                noise = np.random.uniform(low=p_min/(4*(K + extra_count)), high=p_min/(2*(K + extra_count)), size=extra_count)
                # Concatenate the seen probabilities and the noise values.
                augmented = np.concatenate([p_row, noise])
                # Re-normalize the augmented vector.
                new_prob[i] = augmented / augmented.sum()
        else:
            new_prob = p_seen

        return new_prob


class KnnUnseenCalibOrder:
    def __init__(self, calibrate=False,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1):
        """
        Initializes the KNN classifier with the given parameters.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs
        )
        self.calibrate = calibrate
        self.factor = clip_proba_factor
        self.calibrated = None

    def fit(self, X, y):
        """
        Fits the KNN model to the data. Optionally applies calibration.
        """
        # Store the training labels (sorted) and their count.
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='sigmoid',
                cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts the class labels for the input samples.
        """
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """
        Predicts class probabilities for the input samples, maintaining lexicographic order.

        When y_calib is provided, unseen classes are inserted in their proper sorted position,
        maintaining overall lexicographic order of all classes.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Obtain predicted probabilities for training classes
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        # Clip probabilities to avoid extremely low values and renormalize
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape  # K should equal self.num_classes

        # Determine the full set of classes (training + unseen) in sorted order
        if y_calib is not None:
            # Get all unique classes from both training and calibration, sorted
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))

            # Create mapping from training classes to their indices in full_classes
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx

        else:
            # Use previously computed full_classes if available
            if not hasattr(self, 'full_classes'):
                # No calibration was done, return original probabilities
                return p_seen

        # Build the new probability matrix with proper ordering
        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Fill in the probabilities for training classes at their correct positions
        for train_idx, train_class in enumerate(self.classes_):
            full_idx = np.where(self.full_classes == train_class)[0][0]
            new_prob[:, full_idx] = p_seen[:, train_idx]

        # Add noise for unseen classes
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False

        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            for i in range(n):
                p_min = p_seen[i].min()
                # Generate noise for unseen classes
                noise = np.random.uniform(
                    low=p_min / (4 * (K + num_unseen)),
                    high=p_min / (2 * (K + num_unseen)),
                    size=num_unseen
                )
                # Place noise at unseen class positions
                new_prob[i, unseen_indices] = noise

            # Renormalize each row to sum to 1
            new_prob = new_prob / new_prob.sum(axis=1)[:, None]

        return new_prob


class OpenSetKNN:
    def __init__(self, calibrate=False,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1,
                 noise_scale=1e-6):  # Added parameter for noise scale
        """
        Initializes the KNN classifier with the given parameters.
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs
        )
        self.calibrate = calibrate
        self.factor = clip_proba_factor
        self.calibrated = None
        self.noise_scale = noise_scale  # Scale for random noise to avoid ties

    def fit(self, X, y):
        """
        Fits the KNN model to the data. Optionally applies calibration.
        """
        # Store the training labels (sorted) and their count.
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)

        # Store training labels to compute singleton count
        self.y_train = y
        self.n_train = len(y)

        # Count labels that appear only once (singletons) using numpy
        unique_labels, counts = np.unique(y, return_counts=True)
        self.n_singletons = np.sum(counts == 1)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='sigmoid',
                cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts the class labels for the input samples.
        """
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """
        Predicts class probabilities for the input samples, maintaining lexicographic order.

        When y_calib is provided, unseen classes are inserted in their proper sorted position,
        maintaining overall lexicographic order of all classes.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Obtain predicted probabilities for training classes
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        # Clip probabilities to avoid extremely low values and renormalize
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape  # K should equal self.num_classes

        # Determine the full set of classes (training + unseen) in sorted order
        if y_calib is not None:
            # Get all unique classes from both training and calibration, sorted
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))

            # Create mapping from training classes to their indices in full_classes
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx

        else:
            # Use previously computed full_classes if available
            if not hasattr(self, 'full_classes'):
                # No calibration was done, return original probabilities
                return p_seen

        # Build the new probability matrix with proper ordering
        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Identify unseen classes
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False

        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            # Calculate base probability for unseen classes
            # Formula: (1 + n_singletons) / ((1 + n_train) * num_unseen)
            base_unseen_prob = (1 + self.n_singletons) / ((1 + self.n_train) * num_unseen)

            # Calculate total probability mass for unseen classes
            total_unseen_prob = base_unseen_prob * num_unseen

            # Calculate scaling factor for seen class probabilities
            # They need to sum to (1 - total_unseen_prob)
            remaining_prob = 1.0 - total_unseen_prob

            # Ensure remaining_prob is positive and reasonable
            if remaining_prob <= 0.5:
                # If the unseen probability is too high, cap it
                remaining_prob = 0.5  # Reserve at least 50% for seen classes
                total_unseen_prob = 0.5
                base_unseen_prob = total_unseen_prob / num_unseen

            # For each sample
            for i in range(n):
                # Assign probabilities to unseen classes with small random noise
                for unseen_idx in unseen_indices:
                    # Add small random noise to avoid ties
                    noise = np.random.uniform(-self.noise_scale, self.noise_scale)
                    new_prob[i, unseen_idx] = base_unseen_prob * (1 + noise)

                # Scale seen class probabilities to fit in remaining probability mass
                for train_idx, train_class in enumerate(self.classes_):
                    full_idx = np.where(self.full_classes == train_class)[0][0]
                    new_prob[i, full_idx] = p_seen[i, train_idx] * remaining_prob

            # Final renormalization to ensure exact sum to 1 (accounting for noise)
            new_prob = new_prob / new_prob.sum(axis=1)[:, None]
        else:
            # No unseen classes, just copy the seen probabilities
            for train_idx, train_class in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == train_class)[0][0]
                new_prob[:, full_idx] = p_seen[:, train_idx]

        return new_prob


###############################################################################
# OpenMax helpers (shared by both neural-net and KNN variants)
###############################################################################

def _fit_weibull_tail(distances, tail_size):
    """Fit a Weibull distribution on the *tail_size* largest distances.

    Returns (shape, scale) or None if fitting fails.
    """
    if len(distances) < 2:
        return None
    tail = np.sort(distances)[-tail_size:]
    tail = tail[tail > 0]
    if len(tail) < 3:
        return None
    try:
        c, _, scale = weibull_min.fit(tail, floc=0)
        return (c, scale)
    except Exception:
        return None


def _openmax_revision(p_seen, w):
    """Apply the OpenMax probability revision.

    Parameters
    ----------
    p_seen : ndarray, shape (n, K)
        Base classifier probabilities for K seen classes.
    w : ndarray, shape (n, K)
        Weibull-CDF revision weights in [0, 1].

    Returns
    -------
    prob : ndarray, shape (n, K+1)
        Columns 0..K-1  = p_k * (1 - w_k)     (revised seen-class probs)
        Column  K       = sum_k p_k * w_k      (unknown-class prob)
        Rows sum to 1 (no renormalization needed).
    """
    p_revised = p_seen * (1.0 - w)
    p_unknown = np.sum(p_seen * w, axis=1, keepdims=True)
    return np.hstack([p_revised, p_unknown])


###############################################################################
# Original OpenMax  (MLP + penultimate-layer activations)
###############################################################################

class OpenMaxMLP:
    """
    Original OpenMax classifier (Bendale & Boult, 2016).

    Uses an MLP neural network.  After training, for each class k:
      - Compute the Mean Activation Vector (MAV) mu_k from correctly
        classified training points in the penultimate layer.
      - Fit a Weibull distribution on the tail of ||act_i - mu_k||
        for training points i of class k.

    At test time:
      1. Forward pass through the MLP to get penultimate-layer activation v(x).
      2. Compute softmax probabilities p_k(x) from the network output.
      3. For each class k, compute d_k = ||v(x) - mu_k|| and
         w_k = WeibullCDF_k(d_k).
      4. Only revise the top alpha_rank classes; set w_k = 0 for the rest.
      5. Revised probs:
           pi_k(x)      = p_k(x) * (1 - w_k)
           pi_unknown(x) = sum_k p_k(x) * w_k
      6. Return (n, K+1) matrix.
    """

    def __init__(self,
                 hidden_layer_sizes=(128,),
                 activation='relu',
                 max_iter=500,
                 random_state=None,
                 tail_size=20,
                 alpha_rank=None):
        """
        Parameters
        ----------
        hidden_layer_sizes : tuple
            Architecture of the MLP (last element is the penultimate layer).
        tail_size : int
            Number of largest MAV distances used for Weibull fitting per class.
        alpha_rank : int or None
            Revise only the top alpha_rank classes.  None = revise all.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank

    # ----- helpers -----------------------------------------------------------

    def _penultimate_activations(self, X):
        """Forward pass up to (but not including) the output layer."""
        a = X
        # All layers except the last one
        for i in range(len(self.model_.coefs_) - 1):
            a = a @ self.model_.coefs_[i] + self.model_.intercepts_[i]
            if self.activation == 'relu':
                a = np.maximum(a, 0)
            elif self.activation == 'tanh':
                a = np.tanh(a)
            elif self.activation == 'logistic':
                a = 1.0 / (1.0 + np.exp(-a))
        return a

    # ----- main API ----------------------------------------------------------

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)

        # Train MLP
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model_.fit(X, y)

        # Penultimate-layer activations for training data
        acts = self._penultimate_activations(X)
        preds = self.model_.predict(X)

        # Per-class MAV and Weibull fitting
        self.mav_ = {}
        self.weibull_params_ = {}
        all_tail_dists = []

        for k in self.classes_:
            # Use only correctly classified training points
            mask = (y == k) & (preds == k)
            if mask.sum() == 0:
                mask = (y == k)       # fallback: use all points of class k
            acts_k = acts[mask]
            self.mav_[k] = acts_k.mean(axis=0)

            dists = np.linalg.norm(acts_k - self.mav_[k], axis=1)
            all_tail_dists.extend(dists[dists > 0].tolist())
            self.weibull_params_[k] = _fit_weibull_tail(dists, self.tail_size)

        # Pooled Weibull fallback
        self.pooled_weibull_ = None
        if all_tail_dists:
            pooled = np.array(all_tail_dists)
            n_tail = min(self.tail_size * 10, len(pooled))
            self.pooled_weibull_ = _fit_weibull_tail(pooled, n_tail)

        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X, y_calib=None):
        """Return (n, K+1) probability matrix.  y_calib is unused."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]
        K = self.num_classes

        # Base softmax probabilities (n, K)
        p_seen = self.model_.predict_proba(X)

        # Penultimate-layer activations
        acts = self._penultimate_activations(X)

        # Weibull revision weights
        w = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            params = self.weibull_params_.get(k) or self.pooled_weibull_
            if params is not None:
                c, scale = params
                d = np.linalg.norm(acts - self.mav_[k], axis=1)
                w[:, j] = weibull_min.cdf(d, c, loc=0, scale=scale)

        # Only revise top alpha_rank classes per sample
        if self.alpha_rank is not None and self.alpha_rank < K:
            for i in range(n):
                top = np.argsort(-p_seen[i])[:self.alpha_rank]
                mask = np.ones(K, dtype=bool)
                mask[top] = False
                w[i, mask] = 0.0

        return _openmax_revision(p_seen, w)


###############################################################################
# OpenMax-KNN  (KNN + centroid distances for Weibull fitting)
###############################################################################

class OpenMaxKNN:
    """
    OpenMax algorithm adapted for KNN.

    Instead of neural-network activations, uses distances from test points
    to per-class centroids in the original feature space.

    Training:
      1. Fit KNN.
      2. For each class k, compute centroid mu_k and within-class distances.
      3. Fit Weibull on tail distances.  Classes with < tail_size samples
         use a pooled Weibull.

    Prediction for x:
      1. Base KNN probabilities p_k(x), k = 1..K.
      2. d_k(x) = ||x - mu_k||.
      3. w_k(x) = WeibullCDF_k(d_k(x)) for top alpha_rank classes.
      4. pi_k(x)      = p_k(x) * (1 - w_k(x))
         pi_unknown(x) = sum_k p_k(x) * w_k(x)
      5. Return (n, K+1).
    """

    def __init__(self,
                 n_neighbors=5,
                 weights='distance',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=1e-20,
                 tail_size=20,
                 alpha_rank=None):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
            leaf_size=leaf_size, p=p, metric=metric,
            metric_params=metric_params, n_jobs=n_jobs
        )
        self.factor = clip_proba_factor
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        self.model_fit = self.model.fit(X, y)

        # Per-class centroids and Weibull
        self.centroids_ = {}
        self.weibull_params_ = {}
        all_tail_dists = []

        for k in self.classes_:
            X_k = X[y == k]
            self.centroids_[k] = X_k.mean(axis=0)

            if len(X_k) >= 2:
                dists = np.linalg.norm(X_k - self.centroids_[k], axis=1)
                all_tail_dists.extend(dists[dists > 0].tolist())
                self.weibull_params_[k] = _fit_weibull_tail(dists, self.tail_size)
            else:
                self.weibull_params_[k] = None

        # Pooled Weibull fallback
        self.pooled_weibull_ = None
        if all_tail_dists:
            pooled = np.array(all_tail_dists)
            n_tail = min(self.tail_size * 10, len(pooled))
            self.pooled_weibull_ = _fit_weibull_tail(pooled, n_tail)

        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """Return (n, K+1) probability matrix.  y_calib is unused."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]
        K = self.num_classes

        # Base KNN probabilities
        p_seen = self.model_fit.predict_proba(X)
        p_seen = np.clip(p_seen, self.factor / K, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        # Distances to centroids
        distances = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            distances[:, j] = np.linalg.norm(X - self.centroids_[k], axis=1)

        # Weibull revision weights
        w = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            params = self.weibull_params_.get(k) or self.pooled_weibull_
            if params is not None:
                c, scale = params
                w[:, j] = weibull_min.cdf(distances[:, j], c, loc=0, scale=scale)

        # Only revise top alpha_rank classes per sample
        if self.alpha_rank is not None and self.alpha_rank < K:
            for i in range(n):
                top = np.argsort(-p_seen[i])[:self.alpha_rank]
                mask = np.ones(K, dtype=bool)
                mask[top] = False
                w[i, mask] = 0.0

        return _openmax_revision(p_seen, w)


class LogisticRegressionUnseenCalib:
    def __init__(self, calibrate=False, clip_proba_factor=0.1, multi_class = 'multinomial', solver='lbfgs'):
        """
        Initializes the logistic regression classifier with the given parameters.

        Parameters:
          - calibrate (bool): Whether to apply probability calibration.
          - clip_proba_factor (float): Factor to clip the predicted probabilities.
        """
        self.model = LogisticRegression(solver=solver)
        self.calibrate = calibrate
        self.factor = clip_proba_factor
        self.calibrated = None  # To store the calibrated model if calibration is used

    def fit(self, X, y):
        """
        Fits the logistic regression model to the data. Optionally applies calibration.

        Parameters:
          - X (array-like): Feature matrix.
          - y (array-like): Target vector.

        Returns:
          - A deep copy of the fitted logistic regression instance.
        """
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='sigmoid',
                cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts the class labels for the input samples.

        Parameters:
          - X (array-like): Input samples.

        Returns:
          - array-like: Predicted class labels.
        """
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """
        Predicts class probabilities for the input samples.

        When y_calib is provided (e.g., during calibration), the method computes
        the union of training labels (self.classes_) and the unique labels in y_calib.
        For every extra label (i.e. a label in y_calib not seen during training),
        a noise probability is generated. The noise for each extra label for each sample
        is drawn from Uniform(low, high) where low is based on the minimum predicted probability
        from the seen classes. Finally, the augmented probability vector is renormalized to sum to 1.

        Parameters:
          - X (array-like): Input samples.
          - y_calib (array-like, optional): True labels for the samples (used to determine
            extra unseen labels).

        Returns:
          - new_prob (array-like): An array of shape
              (n_samples, #unique(Y_train) + #extra unseen labels)
            containing the probabilities for each label in the union of Y_train and y_calib.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Obtain predicted probabilities for training classes.
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        # Clip probabilities to avoid extremely low values and renormalize.
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape  # Here K should equal self.num_classes

        # Determine the extra unseen labels.
        if y_calib is not None:
            # Compute extra labels as those in y_calib that are not in training.
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            # full_classes is the union of training classes and extra labels.
            self.full_classes = np.concatenate([self.classes_, extra])
        else:
            # If no y_calib is provided, but we already computed the union during calibration,
            # use the extra columns saved in self.full_classes.
            if hasattr(self, 'full_classes'):
                extra = self.full_classes[self.num_classes:]
            else:
                extra = np.array([])

        extra_count = len(extra)
        if extra_count > 0:
            new_prob = np.empty((n, K + extra_count))
            for i in range(n):
                p_row = p_seen[i]
                p_min = p_row.min()  # use the smallest seen probability as scale
                # Generate one noise value per extra unseen label.
                noise = np.random.uniform(low=p_min / 4, high=p_min / 2, size=extra_count)
                # Concatenate the seen probabilities and the noise values.
                augmented = np.concatenate([p_row, noise])
                # Re-normalize the augmented vector.
                new_prob[i] = augmented / augmented.sum()
        else:
            new_prob = p_seen

        return new_prob


class GaussianNaiveBayesUnseenCalib:
    def __init__(self, calibrate=False, clip_proba_factor=0.1):
        """
        Initializes the Gaussian Naive Bayes classifier with the given parameters.

        Parameters:
          - calibrate (bool): Whether to apply probability calibration.
          - clip_proba_factor (float): Factor to clip the predicted probabilities.
        """
        self.model = GaussianNB()
        self.calibrate = calibrate
        self.factor = clip_proba_factor
        self.calibrated = None  # To store the calibrated model if calibration is used

    def fit(self, X, y):
        """
        Fits the Gaussian Naive Bayes model to the data. Optionally applies calibration.

        Parameters:
          - X (array-like): Feature matrix.
          - y (array-like): Target vector.

        Returns:
          - A deep copy of the fitted Gaussian Naive Bayes instance.
        """
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='sigmoid',
                cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts the class labels for the input samples.

        Parameters:
          - X (array-like): Input samples.

        Returns:
          - array-like: Predicted class labels.
        """
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """
        Predicts class probabilities for the input samples.

        When y_calib is provided (e.g., during calibration), the method computes
        the union of training labels (self.classes_) and the unique labels in y_calib.
        For every extra label (i.e., a label in y_calib not seen during training),
        a noise probability is generated. The noise for each extra label for each sample
        is drawn from a uniform distribution scaled by the minimum predicted probability
        from the seen classes. Finally, the augmented probability vector is renormalized to sum to 1.

        Parameters:
          - X (array-like): Input samples.
          - y_calib (array-like, optional): True labels for the samples (used to determine extra unseen labels).

        Returns:
          - new_prob (array-like): An array of shape (n_samples, #unique(Y_train) + #extra unseen labels)
            containing the probabilities for each label in the union of Y_train and y_calib.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Obtain predicted probabilities for training classes.
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        # Clip probabilities to avoid extremely low values and renormalize.
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape  # Here K should equal self.num_classes

        # Determine extra unseen labels.
        if y_calib is not None:
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            self.full_classes = np.concatenate([self.classes_, extra])
        else:
            if hasattr(self, 'full_classes'):
                extra = self.full_classes[self.num_classes:]
            else:
                extra = np.array([])

        extra_count = len(extra)
        if extra_count > 0:
            new_prob = np.empty((n, K + extra_count))
            for i in range(n):
                p_row = p_seen[i]
                p_min = p_row.min()  # Use the smallest seen probability as scale.
                # Generate one noise value per extra unseen label.
                noise = np.random.uniform(low=p_min / 4, high=p_min / 2, size=extra_count)
                # Concatenate the seen probabilities and the noise values.
                augmented = np.concatenate([p_row, noise])
                # Re-normalize the augmented vector.
                new_prob[i] = augmented / augmented.sum()
        else:
            new_prob = p_seen

        return new_prob


class NNetUnseenCalib:
    def __init__(self, calibrate=False,
                 hidden_layer_sizes=(256, 128, 64),  # Deeper architecture
                 batch_size=256,
                 learning_rate_init=0.001,
                 learning_rate_schedule='adaptive',  # Adaptive learning rate
                 max_iter=200,  # More epochs
                 early_stopping=True,  # Prevent overfitting
                 validation_fraction=0.1,
                 n_iter_no_change=20,
                 activation='relu',
                 solver='adam',
                 alpha=0.001,  # L2 regularization
                 dropout_rate=0.2,  # Dropout for regularization
                 clip_proba_factor=1e-6,
                 random_state=2020):
        """
        Enhanced Neural Network with better regularization and architecture
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            learning_rate=learning_rate_schedule,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            activation=activation,
            solver=solver,
            alpha=alpha,
            random_state=random_state,
            verbose=False
        )
        self.calibrate = calibrate
        self.factor = clip_proba_factor
        self.dropout_rate = dropout_rate
        self.calibrated = None

    def fit(self, X, y):
        """
        Fits the NN model with proper handling of many classes
        """
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)

        # Scale features for better NN training
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit the model
        self.model_fit = self.model.fit(X_scaled, y)

        if self.calibrate:
            # Use isotonic regression for better calibration with many classes
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit,
                method='isotonic',  # Better for many classes
                cv=3  # Fewer folds to have more data per fold
            )
            self.calibrated.fit(X_scaled, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        """
        Predicts class labels
        """
        X_scaled = self.scaler.transform(X)
        return self.model_fit.predict(X_scaled)

    def predict_proba(self, X, y_calib=None):
        """
        Predicts probabilities with handling for unseen classes
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        X_scaled = self.scaler.transform(X)

        # Get base predictions
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X_scaled)
        else:
            p_seen = self.calibrated.predict_proba(X_scaled)

        # Temperature scaling for sharper predictions
        temperature = 0.5  # < 1 makes predictions sharper
        p_seen = np.exp(np.log(p_seen + 1e-10) / temperature)
        p_seen = p_seen / p_seen.sum(axis=1, keepdims=True)

        # Clip and normalize
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1, keepdims=True)

        n, K = p_seen.shape

        # Handle unseen classes (same logic as KNN)
        if y_calib is not None:
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            self.full_classes = np.concatenate([self.classes_, extra])
        else:
            if hasattr(self, 'full_classes'):
                extra = self.full_classes[self.num_classes:]
            else:
                extra = np.array([])

        extra_count = len(extra)
        if extra_count > 0:
            new_prob = np.empty((n, K + extra_count))
            for i in range(n):
                p_row = p_seen[i]
                p_min = p_row.min()
                noise = np.random.uniform(low=p_min / 10, high=p_min / 5, size=extra_count)
                augmented = np.concatenate([p_row, noise])
                new_prob[i] = augmented / augmented.sum()
        else:
            new_prob = p_seen

        return new_prob


# Add this improved NN class to black_boxes.py
class NNetRobust:
    """
    Robust Neural Network for highly imbalanced multi-class problems
    """

    def __init__(self,
                 hidden_layer_sizes=(512, 256, 128),
                 batch_size=256,
                 learning_rate_init=0.001,
                 max_iter=300,
                 early_stopping=True,
                 validation_fraction=0.15,
                 n_iter_no_change=30,
                 activation='relu',
                 solver='adam',
                 alpha=0.01,  # Stronger L2 regularization
                 clip_proba_factor=1e-6,
                 use_class_weight=True,
                 random_state=2020):

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            learning_rate='adaptive',
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            activation=activation,
            solver=solver,
            alpha=alpha,
            random_state=random_state,
            verbose=False
        )
        self.factor = clip_proba_factor
        self.use_class_weight = use_class_weight

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Compute sample weights if requested
        if self.use_class_weight:
            sample_weights = compute_sample_weight('balanced', y)
            # Can't directly pass sample_weight to MLPClassifier
            # So we'll use oversampling for rare classes instead

            # Oversample rare classes
            X_resampled = []
            y_resampled = []

            for class_label in self.classes_:
                class_mask = y == class_label
                X_class = X_scaled[class_mask]
                y_class = y[class_mask]

                # Determine how many samples we need
                n_samples = len(y_class)
                if n_samples < 10:  # Oversample rare classes
                    n_target = min(10, len(y) // self.num_classes)
                    X_class_resampled, y_class_resampled = resample(
                        X_class, y_class,
                        n_samples=n_target,
                        replace=True,
                        random_state=42
                    )
                    X_resampled.append(X_class_resampled)
                    y_resampled.append(y_class_resampled)
                else:
                    X_resampled.append(X_class)
                    y_resampled.append(y_class)

            X_scaled = np.vstack(X_resampled)
            y = np.hstack(y_resampled)

        # Fit the model
        self.model_fit = self.model.fit(X_scaled, y)

        return copy.deepcopy(self)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model_fit.predict(X_scaled)

    def predict_proba(self, X, y_calib=None):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        X_scaled = self.scaler.transform(X)
        p_seen = self.model_fit.predict_proba(X_scaled)

        # Apply temperature scaling
        temperature = 0.5  # Make predictions sharper
        eps = 1e-10
        p_seen = np.clip(p_seen, eps, 1.0)
        p_seen_log = np.log(p_seen)
        p_seen = np.exp(p_seen_log / temperature)
        p_seen = p_seen / p_seen.sum(axis=1, keepdims=True)

        # Final clipping and normalization
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1, keepdims=True)

        # Handle unseen classes
        n, K = p_seen.shape

        if y_calib is not None:
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            self.full_classes = np.concatenate([self.classes_, extra])
        else:
            if hasattr(self, 'full_classes'):
                extra = self.full_classes[self.num_classes:]
            else:
                extra = np.array([])

        extra_count = len(extra)
        if extra_count > 0:
            new_prob = np.empty((n, K + extra_count))
            for i in range(n):
                p_row = p_seen[i]
                p_min = p_row.min()
                noise = np.random.uniform(low=p_min / 20, high=p_min / 10, size=extra_count)
                augmented = np.concatenate([p_row, noise])
                new_prob[i] = augmented / augmented.sum()
        else:
            new_prob = p_seen

        return new_prob


class CosineSimClassifier:
    """
    Optimized cosine similarity classifier for FaceNet embeddings
    Best suited for face recognition with many classes
    """

    def __init__(self,
                 threshold_percentile=10,  # Use percentile of distances for thresholding
                 use_weighted_centroid=True,
                 temperature=0.05,  # Lower = sharper probabilities
                 clip_proba_factor=1e-7,
                 min_samples_for_cov=3):

        self.threshold_percentile = threshold_percentile
        self.use_weighted_centroid = use_weighted_centroid
        self.temperature = temperature
        self.factor = clip_proba_factor
        self.min_samples_for_cov = min_samples_for_cov

        self.centroids = {}
        self.thresholds = {}
        self.sample_counts = {}

    def fit(self, X_full, y):
        """
        Compute class centroids and thresholds
        """
        # Extract and normalize embeddings
        X = X_full[:, :128]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)

        all_within_class_distances = []

        for class_id in self.classes_:
            mask = y == class_id
            class_samples = X[mask]
            n_samples = len(class_samples)
            self.sample_counts[class_id] = n_samples

            if n_samples == 1:
                # Single sample - just store it
                self.centroids[class_id] = class_samples[0]
                self.thresholds[class_id] = 0.5  # Default threshold
            else:
                if self.use_weighted_centroid and n_samples >= self.min_samples_for_cov:
                    # Compute weighted centroid (give more weight to consistent samples)
                    # First compute simple mean
                    mean_centroid = class_samples.mean(axis=0)
                    mean_centroid = mean_centroid / np.linalg.norm(mean_centroid)

                    # Compute similarities to mean
                    sims_to_mean = class_samples @ mean_centroid

                    # Weight by similarity (samples closer to mean get higher weight)
                    weights = np.exp(5 * (sims_to_mean - sims_to_mean.min()))
                    weights = weights / weights.sum()

                    # Weighted centroid
                    centroid = (class_samples.T @ weights).T
                    centroid = centroid / np.linalg.norm(centroid)
                else:
                    # Simple mean centroid
                    centroid = class_samples.mean(axis=0)
                    centroid = centroid / np.linalg.norm(centroid)

                self.centroids[class_id] = centroid

                # Compute within-class distances for threshold
                distances = 1 - (class_samples @ centroid)  # Cosine distance

                # Use percentile of distances as threshold
                self.thresholds[class_id] = np.percentile(distances,
                                                          100 - self.threshold_percentile)

                all_within_class_distances.extend(distances)

        # Compute global statistics for probability calibration
        if all_within_class_distances:
            self.global_distance_mean = np.mean(all_within_class_distances)
            self.global_distance_std = np.std(all_within_class_distances) + 1e-6
        else:
            self.global_distance_mean = 0.2
            self.global_distance_std = 0.1

        return copy.deepcopy(self)

    def predict_proba(self, X_full, y_calib=None):
        """
        Compute probabilities based on cosine similarity
        """
        if len(X_full.shape) == 1:
            X_full = X_full.reshape((1, -1))

        # Extract and normalize embeddings
        X = X_full[:, :128]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        n = len(X)

        # Compute similarities to all centroids
        similarities = np.zeros((n, self.num_classes))

        for i, class_id in enumerate(self.classes_):
            centroid = self.centroids[class_id]
            similarities[:, i] = X @ centroid

        # Convert similarities to probabilities using softmax with temperature
        # Higher similarity = higher probability
        logits = similarities / self.temperature

        # Subtract max for numerical stability
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Apply confidence adjustment based on sample counts
        # Classes with more training samples get slight probability boost
        if self.use_weighted_centroid:
            count_weights = np.array([np.log(self.sample_counts[c] + 1)
                                      for c in self.classes_])
            count_weights = count_weights / count_weights.sum()
            # Mild adjustment only
            probs = 0.9 * probs + 0.1 * count_weights

        # Clip and normalize
        probs = np.clip(probs, self.factor / self.num_classes, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Handle unseen classes
        if y_calib is not None:
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            if len(extra) > 0:
                new_prob = np.empty((n, self.num_classes + len(extra)))
                for i in range(n):
                    p_row = probs[i]
                    # Very small probability for unseen classes
                    p_min = p_row.min()
                    noise = np.random.uniform(low=p_min / 200, high=p_min / 100, size=len(extra))
                    augmented = np.concatenate([p_row, noise])
                    new_prob[i] = augmented / augmented.sum()
                return new_prob

        return probs

    def predict(self, X_full):
        """
        Predict class with highest similarity
        """
        probs = self.predict_proba(X_full)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_similarities(self, X_full):
        """
        Get raw cosine similarities (useful for debugging)
        """
        X = X_full[:, :128]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        similarities = np.zeros((len(X), self.num_classes))
        for i, class_id in enumerate(self.classes_):
            centroid = self.centroids[class_id]
            similarities[:, i] = X @ centroid

        return similarities


class CosineClassifier:
    """
    Cosine similarity-based classifier for face embeddings
    """

    def __init__(self, threshold=0.5, clip_proba_factor=1e-7):
        self.threshold = threshold
        self.factor = clip_proba_factor
        self.class_centroids = {}
        self.class_stds = {}

    def fit(self, X_full, y):
        # Use only embeddings
        X = X_full[:, :128]

        # L2 normalize
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)

        # Compute class centroids and spreads
        for class_id in self.classes_:
            mask = y == class_id
            class_samples = X[mask]

            # Compute centroid
            centroid = class_samples.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            self.class_centroids[class_id] = centroid

            # Compute spread (std of cosine similarities)
            if len(class_samples) > 1:
                sims = class_samples @ centroid
                self.class_stds[class_id] = np.std(sims) + 0.01
            else:
                self.class_stds[class_id] = 0.1

        return copy.deepcopy(self)

    def predict_proba(self, X_full, y_calib=None):
        if len(X_full.shape) == 1:
            X_full = X_full.reshape((1, -1))

        X = X_full[:, :128]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        n = len(X)
        probs = np.zeros((n, self.num_classes))

        # Compute similarities to all class centroids
        for i, class_id in enumerate(self.classes_):
            centroid = self.class_centroids[class_id]
            similarities = X @ centroid

            # Convert to probabilities using Gaussian assumption
            std = self.class_stds[class_id]
            probs[:, i] = np.exp(-0.5 * ((1 - similarities) / std) ** 2)

        # Normalize to probabilities
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)

        # Clip and renormalize
        probs = np.clip(probs, self.factor / self.num_classes, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Handle unseen classes (same as before)
        if y_calib is not None:
            extra = np.setdiff1d(np.unique(y_calib), self.classes_)
            if len(extra) > 0:
                new_prob = np.empty((n, self.num_classes + len(extra)))
                for i in range(n):
                    p_row = probs[i]
                    p_min = p_row.min()
                    noise = np.random.uniform(low=p_min / 100, high=p_min / 50, size=len(extra))
                    augmented = np.concatenate([p_row, noise])
                    new_prob[i] = augmented / augmented.sum()
                return new_prob

        return probs

    def predict(self, X_full):
        probs = self.predict_proba(X_full)
        return self.classes_[np.argmax(probs, axis=1)]