import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn import calibration
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.stats import weibull_min
from scipy.spatial.distance import cdist, euclidean, cosine
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


###############################################################################
# GT + KNN  (KNN classifier + Good-Turing estimator for unknown class)
###############################################################################

class GTOpenSetKNN:
    """KNN classifier that appends a Good-Turing unknown-class column.

    predict_proba returns an (n, K+1) matrix where columns 0..K-1 are the
    KNN seen-class probabilities scaled by (1 - p_gt) and column K is the
    constant GT probability.

    With smoothing=True (default), p_gt = (1 + n_singletons) / (1 + n_train),
    capped at 0.5.  With smoothing=False, p_gt is the plain empirical
    Good-Turing estimate n_singletons / n_train (same estimator as the CGTC
    missing-mass plug-in, computed on the training split only).
    """

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
                 smoothing=True):
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
        self.smoothing = smoothing
        self.calibrated = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        self.model_fit = self.model.fit(X, y)

        unique_labels, counts = np.unique(y, return_counts=True)
        self.n_singletons = np.sum(counts == 1)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit, method='sigmoid', cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n = p_seen.shape[0]

        if self.smoothing:
            p_gt = (1 + self.n_singletons) / (1 + self.n_train)
            p_gt = min(p_gt, 0.5)
        else:
            p_gt = self.n_singletons / self.n_train

        p_seen_scaled = p_seen * (1.0 - p_gt)
        p_unknown = np.full((n, 1), p_gt)

        return np.hstack([p_seen_scaled, p_unknown])


###############################################################################
# Distance/confidence-score open-set bases (few-shot friendly, no EVT fitting)
###############################################################################

class KNNDistOpenSet:
    """KNN closed-set classifier + k-th nearest-neighbor distance unknown score.

    The unknown column is the distance from x to its k_score-th nearest
    training point (Sun et al., ICML 2022, "Out-of-Distribution Detection
    with Deep Nearest Neighbors"), divided by dist_scale and clipped to
    [0, 1]. No per-class fitting, so singleton classes cost nothing --
    suited to few-shot many-class regimes where EVT-based methods (OpenMax,
    EVM) cannot fit per-class tails.

    The seen-class columns are the plain KNN probabilities scaled by
    (1 - s); their relative magnitudes are untouched. Intended to be wrapped
    in GTRecalOpenSet, which anchors the average unknown mass at the
    Good-Turing level (the dist_scale normalization is then absorbed into
    the recalibration constant).
    """

    def __init__(self,
                 n_neighbors=5,
                 k_score=None,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1,
                 dist_scale=2.0):
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
        self.k_score = k_score if k_score is not None else n_neighbors
        self.factor = clip_proba_factor
        self.dist_scale = dist_scale

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        self.model_fit = self.model.fit(X, y)
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """Return (n, K+1) probability matrix.  y_calib is unused."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        p_seen = self.model_fit.predict_proba(X)
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        k = min(self.k_score, self.n_train)
        dists, _ = self.model_fit.kneighbors(X, n_neighbors=k)
        s = np.clip(dists[:, -1] / self.dist_scale, 0.0, 1.0)

        return np.hstack([p_seen * (1.0 - s)[:, None], s[:, None]])


class MSPOpenSet:
    """KNN closed-set classifier + maximum-probability unknown score.

    The unknown column is s(x) = 1 - max_k p_k(x) from the same KNN
    (Vaze et al., ICLR 2022, "Open-Set Recognition: A Good Closed-Set
    Classifier Is All You Need"). The seen-class columns are the plain KNN
    probabilities scaled by (1 - s); their relative magnitudes are
    untouched. Intended to be wrapped in GTRecalOpenSet, which anchors the
    average unknown mass at the Good-Turing level.
    """

    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1):
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
        self.factor = clip_proba_factor

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        self.model_fit = self.model.fit(X, y)
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """Return (n, K+1) probability matrix.  y_calib is unused."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        p_seen = self.model_fit.predict_proba(X)
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        s = 1.0 - p_seen.max(axis=1)

        return np.hstack([p_seen * (1.0 - s)[:, None], s[:, None]])


###############################################################################
# PROSER (classifier + data placeholders), Zhou, Ye & Zhan, CVPR 2021
###############################################################################

class PROSERMLP:
    """PROSER open-set classifier on fixed embeddings.

    "Learning Placeholders for Open-Set Recognition" (Zhou, Ye & Zhan,
    CVPR 2021) adapted to precomputed feature vectors: an MLP backbone with
    K class logits plus n_dummy classifier-placeholder logits. The dummy
    (max over the n_dummy logits) acts as the (K+1)-th "unknown" class, so
    predict_proba natively returns a (K+1)-way softmax -- no external
    unknown score or recalibration involved.

    Training:
      1. Closed-set pretraining: cross-entropy on the K class logits.
      2. Placeholder finetuning with three terms:
         - CE over the (K+1)-way combined logits with the true label
           (keeps known classes classified correctly);
         - CE with the true-class logit masked out and target = dummy
           (pushes the dummy to rank second for known samples);
         - data placeholders: manifold mixup at the middle layer between
           same-batch pairs with different labels, target = dummy
           (pseudo-novel samples).

    Note the placeholder losses calibrate the dummy against *synthetic*
    mixup novelty, not against the deployment novelty rate: the level of
    the unknown probability is set by lambda_mask / lambda_mix and the
    training dynamics, not by any estimate of P(new label).
    """

    def __init__(self,
                 hidden_layer_sizes=(256, 128),
                 n_dummy=5,
                 pretrain_epochs=30,
                 finetune_epochs=30,
                 batch_size=128,
                 lr=1e-3,
                 mixup_alpha=2.0,
                 lambda_mask=1.0,
                 lambda_mix=1.0,
                 random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dummy = n_dummy
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mixup_alpha = mixup_alpha
        self.lambda_mask = lambda_mask
        self.lambda_mix = lambda_mix
        self.random_state = random_state

    # ----- helpers -----------------------------------------------------------

    def _combined_logits(self, logits):
        """(B, K + n_dummy) -> (B, K + 1): dummy column = max over dummies."""
        import torch
        K = self.num_classes
        dummy = logits[:, K:].max(dim=1, keepdim=True).values
        return torch.cat([logits[:, :K], dummy], dim=1)

    def _forward(self, x, mix_with=None, lam=None):
        """Forward pass; optionally manifold-mixup at the middle layer."""
        h = self.layer1_(x)
        if mix_with is not None:
            h = lam * h + (1.0 - lam) * self.layer1_(mix_with)
        feat = self.layer2_(h)
        return self.clf_(feat)

    # ----- main API ----------------------------------------------------------

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        K = self.num_classes

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        h1, h2 = self.hidden_layer_sizes
        dim_in = X.shape[1]
        self.layer1_ = nn.Sequential(nn.Linear(dim_in, h1), nn.ReLU())
        self.layer2_ = nn.Sequential(nn.Linear(h1, h2), nn.ReLU())
        self.clf_ = nn.Linear(h2, K + self.n_dummy)

        params = (list(self.layer1_.parameters())
                  + list(self.layer2_.parameters())
                  + list(self.clf_.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr)
        ce = nn.CrossEntropyLoss()

        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        # y is assumed already encoded 0..K-1 (LabelEncoder order); re-encode
        # defensively in case original labels are passed.
        y_enc = np.searchsorted(self.classes_, y)
        y_t = torch.as_tensor(y_enc, dtype=torch.long)

        n = len(y)
        bs = min(self.batch_size, n)

        # ---- Phase 1: closed-set pretraining (K logits only) ----
        for _ in range(self.pretrain_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb, yb = X_t[idx], y_t[idx]
                logits = self._forward(xb)[:, :K]
                loss = ce(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # ---- Phase 2: placeholder finetuning ----
        for _ in range(self.finetune_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb, yb = X_t[idx], y_t[idx]
                B = len(idx)

                logits = self._forward(xb)
                comb = self._combined_logits(logits)          # (B, K+1)

                # (a) classify knowns correctly over K+1
                loss = ce(comb, yb)

                # (b) dummy should rank second: mask the true-class logit
                masked = comb.clone()
                masked[torch.arange(B), yb] = -1e9
                dummy_target = torch.full((B,), K, dtype=torch.long)
                loss = loss + self.lambda_mask * ce(masked, dummy_target)

                # (c) data placeholders: manifold mixup between pairs with
                # different labels, target = dummy
                pair = torch.as_tensor(rng.permutation(B), dtype=torch.long)
                diff = yb != yb[pair]
                if diff.sum() > 0:
                    lam = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
                    logits_mix = self._forward(
                        xb[diff], mix_with=xb[pair][diff], lam=lam)
                    comb_mix = self._combined_logits(logits_mix)
                    mix_target = torch.full((int(diff.sum()),), K,
                                            dtype=torch.long)
                    loss = loss + self.lambda_mix * ce(comb_mix, mix_target)

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.layer1_.eval()
        self.layer2_.eval()
        self.clf_.eval()

        return copy.deepcopy(self)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba[:, :-1], axis=1)]

    def predict_proba(self, X, y_calib=None):
        """Return (n, K+1) probability matrix.  y_calib is unused."""
        import torch

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_t = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = []
            for start in range(0, len(X_t), 4096):
                logits = self._forward(X_t[start:start + 4096])
                comb = self._combined_logits(logits)
                out.append(torch.softmax(comb, dim=1).numpy())
        return np.vstack(out)


###############################################################################
# GT-recalibrated open-set wrapper (base ranking + Good-Turing level)
###############################################################################

class GTRecalOpenSet:
    """Recalibrated open-set classifier: base unknown ranking, GT level.

    Wraps an OpenMax-style base classifier (predict_proba -> (n, K+1) with
    the unknown column last) and rescales its unknown column by a constant
    factor c = p_gt / s_bar so that the average unknown probability on fresh
    data matches the Good-Turing missing-mass estimate p_gt = M1 / n of the
    training split, while keeping the base classifier's x-dependent ranking
    of unknown-ness.

    s_bar (the out-of-sample mean of the raw unknown score) is estimated via
    an internal split of the training data: a scoring copy of the base is
    fit on a (1 - recal_frac) fraction and its unknown score is averaged on
    the held-out recal_frac fraction. In-sample scores would be biased low
    (training points sit near their class centroids/MAVs), which would
    inflate c. The model used for prediction is fit on the FULL training
    split, so the (K+1)-column layout is complete. Everything uses training
    data only, preserving split-conformal validity.

    predict_proba:
        s(x)        = base unknown score (column K)
        p_unk(x)    = min(c * s(x), cap)
        p_seen_k(x) = base seen probs renormalized to sum to 1 - p_unk(x)

    If the held-out mean s_bar is degenerate (~0) or there are no training
    singletons, falls back to the constant p_gt column (the GT baseline).
    """

    def __init__(self, base, recal_frac=0.2, cap=0.9, random_state=None):
        self.base = base
        self.recal_frac = recal_frac
        self.cap = cap
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        n = len(y)
        unique_labels, counts = np.unique(y, return_counts=True)
        self.p_gt_ = np.sum(counts == 1) / n

        # Internal split: scoring copy fit on idx1, held-out mean on idx2
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n2 = max(1, int(self.recal_frac * n))
        idx2, idx1 = perm[:n2], perm[n2:]

        scoring_copy = self.base.fit(X[idx1], y[idx1])
        s_holdout = scoring_copy.predict_proba(X[idx2])[:, -1]
        self.s_bar_ = float(np.mean(s_holdout))

        if self.s_bar_ > 1e-12 and self.p_gt_ > 0:
            self.scale_ = self.p_gt_ / self.s_bar_
        else:
            self.scale_ = None  # fall back to the constant GT column

        # Final model on the full training split (complete class layout)
        self.model_fit = self.base.fit(X, y)
        self.num_classes = self.model_fit.num_classes

        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X, y_calib=None):
        """Return (n, K+1) probability matrix.  y_calib is unused."""
        P = self.model_fit.predict_proba(X)
        s = P[:, -1]
        p_seen = P[:, :-1]

        if self.scale_ is None:
            p_unk = np.full(len(s), self.p_gt_)
        else:
            p_unk = np.minimum(self.scale_ * s, self.cap)

        seen_mass = np.clip(p_seen.sum(axis=1), 1e-12, None)
        p_seen_scaled = p_seen * ((1.0 - p_unk) / seen_mass)[:, None]

        return np.hstack([p_seen_scaled, p_unk[:, None]])


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


###############################################################################
# OpenSet classifiers using OpenMax p_open instead of Good-Turing
###############################################################################

class OpenSetKNNOpenMax:
    """
    KNN classifier that uses an OpenMax-style Weibull revision to estimate the
    probability of the "unknown" class (p_open), then distributes it uniformly
    across unseen calibration labels.

    This replaces the Good-Turing estimator used in OpenSetKNN:
        GT:      p_unseen_per_label = (1 + n_singletons) / ((1 + n_train) * |C_unseen|)
        OpenMax: p_unseen_per_label = p_open(x) / |C_unseen|

    where p_open(x) = sum_k  p_k(x) * w_k(x)  is the OpenMax unknown-class
    probability (feature-dependent, varies per sample).

    Training:
      1. Fit KNN.
      2. For each class k, compute centroid mu_k and within-class distances.
      3. Fit Weibull on tail distances (same as OpenMaxKNN).

    Prediction for x (with unseen calibration labels):
      1. Base KNN probabilities p_k(x), k = 1..K.
      2. d_k(x) = ||x - mu_k||.
      3. w_k(x) = WeibullCDF_k(d_k(x)).
      4. p_open(x) = sum_k p_k(x) * w_k(x).
      5. For each unseen label: p_unseen = p_open(x) / |C_unseen| + noise.
      6. Scale seen probabilities to sum to (1 - p_open(x)).
      7. Return (n, K + |C_unseen|).
    """

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
                 noise_scale=1e-6,
                 tail_size=20,
                 alpha_rank=None):
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
        self.noise_scale = noise_scale
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)
        self.n_train = len(y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit, method='sigmoid', cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        # Per-class centroids and Weibull fitting (same as OpenMaxKNN)
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

    def _compute_p_open(self, X, p_seen):
        """Compute per-sample OpenMax unknown-class probability.

        Returns
        -------
        p_open : ndarray, shape (n,)
            p_open[i] = sum_k p_k[i] * w_k[i]
        """
        n = X.shape[0]
        K = self.num_classes

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

        p_open = np.sum(p_seen * w, axis=1)
        return p_open

    def predict_proba(self, X, y_calib=None):
        """
        Predict class probabilities using OpenMax p_open for unseen labels.

        When y_calib is provided, unseen labels are identified and each receives
        p_open(x) / |C_unseen| (plus small noise to break ties).
        Seen-class probabilities are scaled to sum to (1 - p_open(x)).
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Base KNN probabilities for seen classes
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape

        # Determine unseen labels
        if y_calib is not None:
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx
        else:
            if not hasattr(self, 'full_classes'):
                return p_seen

        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Identify unseen class indices
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False
        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            # Compute per-sample p_open via OpenMax Weibull revision
            p_open = self._compute_p_open(X, p_seen)

            # Cap p_open to avoid degenerate seen-class probabilities
            p_open = np.clip(p_open, 0.0, 0.5)

            for i in range(n):
                # Distribute p_open uniformly across unseen labels + noise
                base_unseen = p_open[i] / num_unseen
                for unseen_idx in unseen_indices:
                    noise = np.random.uniform(-self.noise_scale, self.noise_scale)
                    new_prob[i, unseen_idx] = base_unseen * (1 + noise)

                # Scale seen-class probabilities to fill remaining mass
                remaining = 1.0 - p_open[i]
                for train_idx, train_class in enumerate(self.classes_):
                    full_idx = np.where(self.full_classes == train_class)[0][0]
                    new_prob[i, full_idx] = p_seen[i, train_idx] * remaining

            # Final renormalization
            new_prob = new_prob / new_prob.sum(axis=1)[:, None]
        else:
            for train_idx, train_class in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == train_class)[0][0]
                new_prob[:, full_idx] = p_seen[:, train_idx]

        return new_prob


###############################################################################
# EVM-based open-set KNN
###############################################################################

def _fit_weibull_margin(distances, tail_size):
    """Fit a Weibull distribution on the *tail_size* smallest distances.

    These are margin distances (nearest other-class neighbours), so the
    interesting tail is the *left* (small) tail — unlike OpenMax which uses
    the right (large) tail of within-class distances.

    Returns (shape, scale) or None if fitting fails.
    """
    if len(distances) < 2:
        return None
    tail = np.sort(distances)[:tail_size]
    tail = tail[tail > 0]
    if len(tail) < 3:
        return None
    try:
        c, _, scale = weibull_min.fit(tail, floc=0)
        return (c, scale)
    except Exception:
        return None


class EVMKNN:
    """
    EVM algorithm adapted for KNN — returns (n, K+1) probabilities.

    Analogous to OpenMaxKNN but uses between-class margin distances instead
    of within-class centroid distances for Weibull fitting.

    Training:
      1. Fit KNN.
      2. For each class k, compute margin distances: distance from each
         class-k sample to its nearest other-class sample.
      3. Fit Weibull on the smallest margin distances per class.

    Prediction for x:
      1. Base KNN probabilities p_k(x), k = 1..K.
      2. For each class k, d_k(x) = distance to nearest training sample in k.
      3. inclusion_k(x) = 1 - WeibullCDF_k(d_k(x)).
      4. p_open(x) = 1 - max_k inclusion_k(x).
      5. Return (n, K+1) via revision: seen probs scaled by (1-p_open),
         last column = p_open.
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
                 tail_size=20):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
            leaf_size=leaf_size, p=p, metric=metric,
            metric_params=metric_params, n_jobs=n_jobs
        )
        self.factor = clip_proba_factor
        self.tail_size = tail_size

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        self.model_fit = self.model.fit(X, y)

        # Store training data for nearest-neighbour lookup at prediction time
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Per-class: store sample indices and fit Weibull on margin distances
        self.class_indices_ = {}
        self.weibull_margin_ = {}
        all_margin_dists = []

        for k in self.classes_:
            mask_k = (y == k)
            self.class_indices_[k] = np.where(mask_k)[0]
            X_k = X[mask_k]
            X_other = X[~mask_k]

            if len(X_k) < 1 or len(X_other) < 1:
                self.weibull_margin_[k] = None
                continue

            dists_to_other = cdist(X_k, X_other)
            margin_dists = dists_to_other.min(axis=1)
            all_margin_dists.extend(margin_dists[margin_dists > 0].tolist())
            self.weibull_margin_[k] = _fit_weibull_margin(
                margin_dists, self.tail_size
            )

        # Pooled fallback
        self.pooled_margin_weibull_ = None
        if all_margin_dists:
            pooled = np.array(all_margin_dists)
            n_tail = min(self.tail_size * 10, len(pooled))
            self.pooled_margin_weibull_ = _fit_weibull_margin(pooled, n_tail)

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

        # EVM inclusion probabilities per class
        inclusion = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            idx_k = self.class_indices_[k]
            X_k = self.X_train_[idx_k]

            dists = cdist(X, X_k)
            d_min = dists.min(axis=1)

            params = self.weibull_margin_.get(k) or self.pooled_margin_weibull_
            if params is not None:
                c, scale = params
                cdf_val = weibull_min.cdf(d_min, c, loc=0, scale=scale)
                inclusion[:, j] = 1.0 - cdf_val
            else:
                inclusion[:, j] = 0.5

        p_open = 1.0 - inclusion.max(axis=1)
        p_open = np.clip(p_open, 0.0, 1.0)

        # Build (n, K+1): scale seen probs by (1-p_open), last col = p_open
        p_revised = p_seen * (1.0 - p_open)[:, None]
        p_unknown = p_open[:, None]
        return np.hstack([p_revised, p_unknown])


class OpenSetKNNEVM:
    """
    KNN classifier that uses an EVM (Extreme Value Machine) approach to
    estimate p_open, then distributes it uniformly across unseen calibration
    labels — same interface as OpenSetKNNOpenMax.

    Key difference from OpenMax:
        OpenMax fits Weibull on *within-class* distances to centroid
        (large-tail), measuring how spread out each class is.

        EVM fits Weibull on *between-class* margin distances (small-tail),
        measuring how close other-class samples get to each class.

    Training:
      1. Fit KNN.
      2. For each class k, compute margin distances: for every sample in k,
         find distance to nearest sample NOT in k.
      3. Fit Weibull on the *tail_size* smallest margin distances per class.

    Prediction for x:
      1. Base KNN probabilities p_k(x), k = 1..K.
      2. For each class k, d_k(x) = distance from x to nearest training
         sample in class k.
      3. Inclusion probability:
             P(x in k) = 1 - WeibullCDF_k(d_k(x))
         If d_k(x) is much smaller than typical margin distances, CDF is low
         and inclusion is high; if d_k(x) is comparable to or larger than
         margin distances, inclusion is low.
      4. p_open(x) = 1 - max_k P(x in k).
      5. Distribute p_open across unseen labels (same as OpenSetKNNOpenMax).
    """

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
                 noise_scale=1e-6,
                 tail_size=20):
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
        self.noise_scale = noise_scale
        self.tail_size = tail_size

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.model_fit = self.model.fit(X, y)
        self.n_train = len(y)

        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit, method='sigmoid', cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        # Store training data for nearest-neighbour lookup at prediction time
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Per-class: store sample indices and fit Weibull on margin distances
        self.class_indices_ = {}
        self.weibull_margin_ = {}
        all_margin_dists = []

        for k in self.classes_:
            mask_k = (y == k)
            self.class_indices_[k] = np.where(mask_k)[0]
            X_k = X[mask_k]
            X_other = X[~mask_k]

            if len(X_k) < 1 or len(X_other) < 1:
                self.weibull_margin_[k] = None
                continue

            # For each sample in class k, distance to nearest other-class sample
            dists_to_other = cdist(X_k, X_other)          # (n_k, n_other)
            margin_dists = dists_to_other.min(axis=1)      # (n_k,)
            all_margin_dists.extend(margin_dists[margin_dists > 0].tolist())
            self.weibull_margin_[k] = _fit_weibull_margin(
                margin_dists, self.tail_size
            )

        # Pooled fallback for singleton / tiny classes
        self.pooled_margin_weibull_ = None
        if all_margin_dists:
            pooled = np.array(all_margin_dists)
            n_tail = min(self.tail_size * 10, len(pooled))
            self.pooled_margin_weibull_ = _fit_weibull_margin(pooled, n_tail)

        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def _compute_p_open(self, X, p_seen):
        """Compute per-sample EVM-based unknown-class probability.

        For each class k we compute d_k(x) = distance from x to nearest
        training sample in class k, then:
            inclusion_k(x) = 1 - WeibullCDF_k(d_k(x))
        and
            p_open(x) = 1 - max_k inclusion_k(x)

        Returns
        -------
        p_open : ndarray, shape (n,)
        """
        n = X.shape[0]
        K = self.num_classes
        inclusion = np.zeros((n, K))

        for j, k in enumerate(self.classes_):
            idx_k = self.class_indices_[k]
            X_k = self.X_train_[idx_k]

            # Distance from each test point to nearest training sample in k
            dists = cdist(X, X_k)                # (n, n_k)
            d_min = dists.min(axis=1)             # (n,)

            params = self.weibull_margin_.get(k) or self.pooled_margin_weibull_
            if params is not None:
                c, scale = params
                cdf_val = weibull_min.cdf(d_min, c, loc=0, scale=scale)
                inclusion[:, j] = 1.0 - cdf_val
            else:
                # No Weibull available — fall back to 0.5
                inclusion[:, j] = 0.5

        p_open = 1.0 - inclusion.max(axis=1)
        return p_open

    def predict_proba(self, X, y_calib=None):
        """
        Predict class probabilities using EVM p_open for unseen labels.

        Same interface as OpenSetKNNOpenMax.predict_proba.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Base KNN probabilities for seen classes
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape

        # Determine unseen labels
        if y_calib is not None:
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx
        else:
            if not hasattr(self, 'full_classes'):
                return p_seen

        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Identify unseen class indices
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False
        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            # Compute per-sample p_open via EVM
            p_open = self._compute_p_open(X, p_seen)

            # Cap p_open to avoid degenerate seen-class probabilities
            p_open = np.clip(p_open, 0.0, 0.5)

            for i in range(n):
                # Distribute p_open uniformly across unseen labels + noise
                base_unseen = p_open[i] / num_unseen
                for unseen_idx in unseen_indices:
                    noise = np.random.uniform(-self.noise_scale, self.noise_scale)
                    new_prob[i, unseen_idx] = base_unseen * (1 + noise)

                # Scale seen-class probabilities to fill remaining mass
                remaining = 1.0 - p_open[i]
                for train_idx, train_class in enumerate(self.classes_):
                    full_idx = np.where(self.full_classes == train_class)[0][0]
                    new_prob[i, full_idx] = p_seen[i, train_idx] * remaining

            # Final renormalization
            new_prob = new_prob / new_prob.sum(axis=1)[:, None]
        else:
            for train_idx, train_class in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == train_class)[0][0]
                new_prob[:, full_idx] = p_seen[:, train_idx]

        return new_prob


class OpenSetMLPOpenMax:
    """
    MLP classifier that uses OpenMax-style Weibull revision on penultimate-layer
    activations to estimate p_open, then distributes it uniformly across unseen
    calibration labels.

    Same idea as OpenSetKNNOpenMax but uses:
      - MLP for base classification
      - Penultimate-layer activations for MAV centroids and Weibull fitting
    """

    def __init__(self,
                 hidden_layer_sizes=(128,),
                 activation='relu',
                 max_iter=500,
                 random_state=None,
                 clip_proba_factor=0.1,
                 noise_scale=1e-6,
                 tail_size=20,
                 alpha_rank=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.factor = clip_proba_factor
        self.noise_scale = noise_scale
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank

    def _penultimate_activations(self, X):
        """Forward pass up to (but not including) the output layer."""
        a = X
        for i in range(len(self.model_.coefs_) - 1):
            a = a @ self.model_.coefs_[i] + self.model_.intercepts_[i]
            if self.activation == 'relu':
                a = np.maximum(a, 0)
            elif self.activation == 'tanh':
                a = np.tanh(a)
            elif self.activation == 'logistic':
                a = 1.0 / (1.0 + np.exp(-a))
        return a

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
            mask = (y == k) & (preds == k)
            if mask.sum() == 0:
                mask = (y == k)
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

    def _compute_p_open(self, X, p_seen):
        """Compute per-sample OpenMax unknown-class probability using MAV distances."""
        n = X.shape[0]
        K = self.num_classes

        acts = self._penultimate_activations(X)

        w = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            params = self.weibull_params_.get(k) or self.pooled_weibull_
            if params is not None:
                c, scale = params
                d = np.linalg.norm(acts - self.mav_[k], axis=1)
                w[:, j] = weibull_min.cdf(d, c, loc=0, scale=scale)

        if self.alpha_rank is not None and self.alpha_rank < K:
            for i in range(n):
                top = np.argsort(-p_seen[i])[:self.alpha_rank]
                mask = np.ones(K, dtype=bool)
                mask[top] = False
                w[i, mask] = 0.0

        return np.sum(p_seen * w, axis=1)

    def predict_proba(self, X, y_calib=None):
        """
        Predict class probabilities using OpenMax p_open for unseen labels.

        When y_calib is provided, unseen labels are identified and each receives
        p_open(x) / |C_unseen| (plus small noise to break ties).
        Seen-class probabilities are scaled to sum to (1 - p_open(x)).
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Base MLP probabilities for seen classes
        p_seen = self.model_.predict_proba(X)
        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape

        # Determine unseen labels
        if y_calib is not None:
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx
        else:
            if not hasattr(self, 'full_classes'):
                return p_seen

        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Identify unseen class indices
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False
        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            p_open = self._compute_p_open(X, p_seen)
            p_open = np.clip(p_open, 0.0, 0.5)

            for i in range(n):
                base_unseen = p_open[i] / num_unseen
                for unseen_idx in unseen_indices:
                    noise = np.random.uniform(-self.noise_scale, self.noise_scale)
                    new_prob[i, unseen_idx] = base_unseen * (1 + noise)

                remaining = 1.0 - p_open[i]
                for train_idx, train_class in enumerate(self.classes_):
                    full_idx = np.where(self.full_classes == train_class)[0][0]
                    new_prob[i, full_idx] = p_seen[i, train_idx] * remaining

            new_prob = new_prob / new_prob.sum(axis=1)[:, None]
        else:
            for train_idx, train_class in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == train_class)[0][0]
                new_prob[:, full_idx] = p_seen[:, train_idx]

        return new_prob


###############################################################################
# Hybrid: KNN base classifier + Original OpenMax (MLP) for unknown probability
###############################################################################

def _eucos_distance(vec_a, vec_b):
    """Compute eucos distance (euclidean/200 + cosine) as in the original
    OSDN code (Bendale & Boult, 2016).  Returns a scalar."""
    return euclidean(vec_a, vec_b) / 200.0 + cosine(vec_a, vec_b)


def _fit_weibull_tail_eucos(distances, tail_size):
    """Fit a Weibull on the *tail_size* largest eucos distances.

    Same logic as ``_fit_weibull_tail`` but kept separate so the original
    helper is not affected.  Returns (shape, scale) or None.
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


class OpenSetKNNwithMLPOpenMax:
    """
    KNN classifier for K seen-class probabilities, combined with an MLP-based
    OpenMax estimator for the unknown-class probability.

    This allows direct comparison with the Good-Turing baseline (OpenSetKNN):
    both use the same KNN for p_seen; the only difference is how p_unknown is
    estimated:
        GT:      p_unknown = (1 + n_singletons) / (1 + n_train)
        OpenMax: p_unknown computed via the original OpenMax algorithm

    The OpenMax part faithfully follows the OSDN reference implementation
    (https://github.com/abhijitbendale/OSDN):
      1. eucos distance  = euclidean/200 + cosine  (for MAV distances)
      2. Graded alpha weights: w_rank = (alpha_rank+1 - rank) / alpha_rank
         for the top alpha_rank classes; 0 for the rest
      3. Revision on raw logits (pre-softmax), then re-softmax to obtain
         K+1 probabilities (K seen + 1 unknown)

    p_open is then distributed uniformly across unseen calibration labels.
    """

    def __init__(self, calibrate=False,
                 # KNN params
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1,
                 noise_scale=1e-6,
                 # MLP OpenMax params
                 hidden_layer_sizes=(128,),
                 activation='relu',
                 max_iter=500,
                 mlp_random_state=None,
                 tail_size=20,
                 alpha_rank=10):
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
        self.noise_scale = noise_scale
        # MLP OpenMax settings
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.mlp_random_state = mlp_random_state
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)

        # --- Fit KNN (for p_seen) ---
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit, method='sigmoid', cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        # --- Fit MLP (for OpenMax p_open) ---
        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            max_iter=self.max_iter,
            random_state=self.mlp_random_state
        )
        self.mlp_.fit(X, y)

        # Penultimate-layer activations for training data
        acts = self._penultimate_activations(X)
        preds = self.mlp_.predict(X)

        # Per-class MAV and Weibull fitting using eucos distance
        self.mav_ = {}
        self.weibull_params_ = {}

        for k in self.classes_:
            mask = (y == k) & (preds == k)
            if mask.sum() == 0:
                mask = (y == k)
            acts_k = acts[mask]
            self.mav_[k] = acts_k.mean(axis=0)

            # Compute eucos distances from each training point to its class MAV
            eucos_dists = np.array([
                _eucos_distance(acts_k[i], self.mav_[k])
                for i in range(len(acts_k))
            ])
            self.weibull_params_[k] = _fit_weibull_tail_eucos(
                eucos_dists, self.tail_size
            )

        return copy.deepcopy(self)

    def _penultimate_activations(self, X):
        """Forward pass through MLP up to (but not including) the output layer."""
        a = X
        for i in range(len(self.mlp_.coefs_) - 1):
            a = a @ self.mlp_.coefs_[i] + self.mlp_.intercepts_[i]
            if self.activation == 'relu':
                a = np.maximum(a, 0)
            elif self.activation == 'tanh':
                a = np.tanh(a)
            elif self.activation == 'logistic':
                a = 1.0 / (1.0 + np.exp(-a))
        return a

    def _raw_logits(self, X):
        """Compute the raw output-layer logits (before softmax)."""
        acts = self._penultimate_activations(X)
        return acts @ self.mlp_.coefs_[-1] + self.mlp_.intercepts_[-1]

    def predict(self, X):
        return self.model_fit.predict(X)

    def _compute_p_open(self, X):
        """Compute per-sample OpenMax unknown-class probability.

        Faithfully follows the OSDN reference implementation:
          1. Get raw logits from MLP output layer.
          2. Rank classes by logit value (descending).
          3. Compute graded alpha weights for top alpha_rank classes:
             alpha_weight[rank] = (alpha_rank + 1 - rank) / alpha_rank
          4. For each class, compute eucos distance to MAV and Weibull w_score.
          5. modified_logit_k = logit_k * (1 - w_score_k * alpha_weight_k)
             unknown_logit   += logit_k - modified_logit_k
          6. Softmax over [modified_logits, unknown_logit] to get K+1 probs.
          7. Return p_open = prob[K] (the unknown-class probability).
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        n = X.shape[0]
        K = self.num_classes
        alpha_rank = min(self.alpha_rank, K) if self.alpha_rank is not None else K

        logits = self._raw_logits(X)              # (n, K)
        acts = self._penultimate_activations(X)    # (n, d)

        # Pre-compute eucos distances to all class MAVs: (n, K)
        eucos_dists = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            for i in range(n):
                eucos_dists[i, j] = _eucos_distance(acts[i], self.mav_[k])

        # Pre-compute Weibull w_scores: (n, K)
        w_scores = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            params = self.weibull_params_.get(k)
            if params is not None:
                c, scale = params
                w_scores[:, j] = weibull_min.cdf(
                    eucos_dists[:, j], c, loc=0, scale=scale
                )

        # Per-sample OpenMax revision on logits
        p_open = np.zeros(n)
        for i in range(n):
            # Rank classes by logit (descending) — matches OSDN's argsort
            ranked_list = np.argsort(-logits[i])

            # Graded alpha weights — matches OSDN line 136:
            #   alpha_weights = [((alpharank+1) - i)/float(alpharank)
            #                    for i in range(1, alpharank+1)]
            ranked_alpha = np.zeros(K)
            for r in range(alpha_rank):
                ranked_alpha[ranked_list[r]] = (
                    (alpha_rank + 1 - (r + 1)) / float(alpha_rank)
                )

            # Revise logits — matches OSDN line 159:
            #   modified_fc8_score = score * (1 - wscore * ranked_alpha)
            #   unknown_score     += score - modified_fc8_score
            modified_logits = logits[i].copy()
            unknown_logit = 0.0
            for j in range(K):
                modified_logits[j] = (
                    logits[i, j] * (1.0 - w_scores[i, j] * ranked_alpha[j])
                )
                unknown_logit += logits[i, j] - modified_logits[j]

            # Softmax over K+1 values — matches OSDN computeOpenMaxProbability()
            all_logits = np.append(modified_logits, unknown_logit)
            all_logits_shifted = all_logits - np.max(all_logits)
            exp_logits = np.exp(all_logits_shifted)
            openmax_probs = exp_logits / exp_logits.sum()

            p_open[i] = openmax_probs[K]  # the unknown-class probability

        return p_open

    def predict_proba(self, X, y_calib=None):
        """
        Predict class probabilities.

        Seen-class probabilities come from KNN.
        Unknown-class probability comes from MLP-based OpenMax (p_open).
        p_open is distributed uniformly across unseen calibration labels.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Base KNN probabilities for seen classes
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape

        # Determine unseen labels
        if y_calib is not None:
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx
        else:
            if not hasattr(self, 'full_classes'):
                return p_seen

        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Identify unseen class indices
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False
        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            # Compute p_open via original OpenMax algorithm
            p_open = self._compute_p_open(X)

            # Clip to valid probability range
            p_open = np.clip(p_open, 0.0, 0.5)

            for i in range(n):
                # Distribute p_open uniformly across unseen labels + noise
                base_unseen = p_open[i] / num_unseen
                for unseen_idx in unseen_indices:
                    noise = np.random.uniform(-self.noise_scale, self.noise_scale)
                    new_prob[i, unseen_idx] = base_unseen * (1 + noise)

                # Scale seen-class probabilities to fill remaining mass
                remaining = 1.0 - p_open[i]
                for train_idx, train_class in enumerate(self.classes_):
                    full_idx = np.where(self.full_classes == train_class)[0][0]
                    new_prob[i, full_idx] = p_seen[i, train_idx] * remaining

            # Final renormalization
            new_prob = new_prob / new_prob.sum(axis=1)[:, None]
        else:
            for train_idx, train_class in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == train_class)[0][0]
                new_prob[:, full_idx] = p_seen[:, train_idx]

        return new_prob


###############################################################################
# Hybrid: Good-Turing anchor + OpenMax feature-dependent perturbation
###############################################################################

class OpenSetKNNwithGTOpenMaxHybrid:
    """
    KNN classifier for seen-class probabilities, with unseen-class probability
    computed as a Good-Turing estimate *modulated* by OpenMax's feature-dependent
    unknown score.

    The idea:
        p_gt          = (1 + n_singletons) / (1 + n_train)   [global anchor]
        p_open(x)     = OpenMax unknown probability            [feature-dependent]
        mean_p_open   = mean of p_open over calibration data   [normaliser]

        p_unseen(x)   = clip(p_gt * p_open(x) / mean_p_open, eps, cap)

    This preserves the well-calibrated Good-Turing average while allowing
    OpenMax to redistribute unseen mass across test points based on features.
    """

    def __init__(self, calibrate=False,
                 # KNN params
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 clip_proba_factor=0.1,
                 noise_scale=1e-6,
                 # MLP OpenMax params
                 hidden_layer_sizes=(128,),
                 activation='relu',
                 max_iter=500,
                 mlp_random_state=None,
                 tail_size=20,
                 alpha_rank=10,
                 # Hybrid modulation params
                 p_unseen_cap=0.5):
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
        self.noise_scale = noise_scale
        # MLP OpenMax settings
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.mlp_random_state = mlp_random_state
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank
        # Hybrid params
        self.p_unseen_cap = p_unseen_cap

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        self.y_train = y

        # Singleton count for Good-Turing
        unique_labels, counts = np.unique(y, return_counts=True)
        self.n_singletons = np.sum(counts == 1)

        # --- Fit KNN (for p_seen) ---
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit, method='sigmoid', cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        # --- Fit MLP (for OpenMax p_open) ---
        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            max_iter=self.max_iter,
            random_state=self.mlp_random_state
        )
        self.mlp_.fit(X, y)

        # Penultimate-layer activations for training data
        acts = self._penultimate_activations(X)
        preds = self.mlp_.predict(X)

        # Per-class MAV and Weibull fitting using eucos distance
        self.mav_ = {}
        self.weibull_params_ = {}

        for k in self.classes_:
            mask = (y == k) & (preds == k)
            if mask.sum() == 0:
                mask = (y == k)
            acts_k = acts[mask]
            self.mav_[k] = acts_k.mean(axis=0)

            eucos_dists = np.array([
                _eucos_distance(acts_k[i], self.mav_[k])
                for i in range(len(acts_k))
            ])
            self.weibull_params_[k] = _fit_weibull_tail_eucos(
                eucos_dists, self.tail_size
            )

        return copy.deepcopy(self)

    def _penultimate_activations(self, X):
        a = X
        for i in range(len(self.mlp_.coefs_) - 1):
            a = a @ self.mlp_.coefs_[i] + self.mlp_.intercepts_[i]
            if self.activation == 'relu':
                a = np.maximum(a, 0)
            elif self.activation == 'tanh':
                a = np.tanh(a)
            elif self.activation == 'logistic':
                a = 1.0 / (1.0 + np.exp(-a))
        return a

    def _raw_logits(self, X):
        acts = self._penultimate_activations(X)
        return acts @ self.mlp_.coefs_[-1] + self.mlp_.intercepts_[-1]

    def predict(self, X):
        return self.model_fit.predict(X)

    def _compute_p_open(self, X):
        """Compute per-sample OpenMax unknown-class probability.

        Same algorithm as OpenSetKNNwithMLPOpenMax._compute_p_open.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        n = X.shape[0]
        K = self.num_classes
        alpha_rank = min(self.alpha_rank, K) if self.alpha_rank is not None else K

        logits = self._raw_logits(X)
        acts = self._penultimate_activations(X)

        eucos_dists = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            for i in range(n):
                eucos_dists[i, j] = _eucos_distance(acts[i], self.mav_[k])

        w_scores = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            params = self.weibull_params_.get(k)
            if params is not None:
                c, scale = params
                w_scores[:, j] = weibull_min.cdf(
                    eucos_dists[:, j], c, loc=0, scale=scale
                )

        p_open = np.zeros(n)
        for i in range(n):
            ranked_list = np.argsort(-logits[i])
            ranked_alpha = np.zeros(K)
            for r in range(alpha_rank):
                ranked_alpha[ranked_list[r]] = (
                    (alpha_rank + 1 - (r + 1)) / float(alpha_rank)
                )

            modified_logits = logits[i].copy()
            unknown_logit = 0.0
            for j in range(K):
                modified_logits[j] = (
                    logits[i, j] * (1.0 - w_scores[i, j] * ranked_alpha[j])
                )
                unknown_logit += logits[i, j] - modified_logits[j]

            all_logits = np.append(modified_logits, unknown_logit)
            all_logits_shifted = all_logits - np.max(all_logits)
            exp_logits = np.exp(all_logits_shifted)
            openmax_probs = exp_logits / exp_logits.sum()

            p_open[i] = openmax_probs[K]

        return p_open

    def calibrate_p_open(self, X_calib):
        """Compute mean p_open on calibration data for normalisation.

        Must be called after fit() and before predict_proba() on test data.
        """
        p_open_calib = self._compute_p_open(X_calib)
        self.mean_p_open_calib_ = p_open_calib.mean()
        if self.mean_p_open_calib_ < 1e-12:
            self.mean_p_open_calib_ = 1e-12
        return p_open_calib

    def predict_proba(self, X, y_calib=None):
        """
        Predict class probabilities.

        Seen-class probabilities come from KNN.
        Unseen-class probability = Good-Turing anchor modulated by OpenMax ratio.

        If calibrate_p_open() has not been called, falls back to computing
        mean_p_open from the current batch X (less ideal but still works).
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Base KNN probabilities for seen classes
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape

        # Determine unseen labels
        if y_calib is not None:
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx
        else:
            if not hasattr(self, 'full_classes'):
                return p_seen

        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Identify unseen class indices
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False
        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            # Good-Turing anchor
            p_gt = (1 + self.n_singletons) / (1 + self.n_train)

            # OpenMax feature-dependent scores
            p_open = self._compute_p_open(X)

            # Auto-calibrate on the first predict_proba call (calibration data)
            if not hasattr(self, 'mean_p_open_calib_') and y_calib is not None:
                self.calibrate_p_open(X)

            # Normaliser: use calibration mean if available, else batch mean
            if hasattr(self, 'mean_p_open_calib_'):
                mean_p_open = self.mean_p_open_calib_
            else:
                mean_p_open = p_open.mean()
                if mean_p_open < 1e-12:
                    mean_p_open = 1e-12

            for i in range(n):
                # Modulated unseen probability
                ratio = p_open[i] / mean_p_open
                p_unseen_total = np.clip(
                    p_gt * ratio, 1e-20, self.p_unseen_cap
                )

                base_unseen = p_unseen_total / num_unseen
                for unseen_idx in unseen_indices:
                    noise = np.random.uniform(-self.noise_scale, self.noise_scale)
                    new_prob[i, unseen_idx] = base_unseen * (1 + noise)

                # Scale seen-class probabilities to fill remaining mass
                remaining = 1.0 - p_unseen_total
                for train_idx, train_class in enumerate(self.classes_):
                    full_idx = np.where(self.full_classes == train_class)[0][0]
                    new_prob[i, full_idx] = p_seen[i, train_idx] * remaining

            # Final renormalization
            new_prob = new_prob / new_prob.sum(axis=1)[:, None]
        else:
            for train_idx, train_class in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == train_class)[0][0]
                new_prob[:, full_idx] = p_seen[:, train_idx]

        return new_prob


###############################################################################
# Hybrid: Good-Turing anchor + KNN-based OpenMax feature-dependent perturbation
###############################################################################

class OpenSetKNNwithGTKNNOpenMaxHybrid:
    """
    KNN classifier for seen-class probabilities, with unseen-class probability
    computed as a Good-Turing estimate *modulated* by a KNN-based OpenMax
    feature-dependent unknown score.

    Same idea as OpenSetKNNwithGTOpenMaxHybrid but using KNN-based OpenMax
    (centroid distances + Weibull in feature space) instead of MLP-based OpenMax
    (penultimate-layer activations + eucos distance).

    The formula:
        p_gt          = (1 + n_singletons) / (1 + n_train)   [global anchor]
        p_open(x)     = sum_k p_k(x) * w_k(x)                [KNN OpenMax]
        mean_p_open   = mean of p_open over calibration data   [normaliser]

        p_unseen(x)   = clip(p_gt * p_open(x) / mean_p_open, eps, cap)
    """

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
                 noise_scale=1e-6,
                 tail_size=20,
                 alpha_rank=None,
                 # Hybrid modulation params
                 p_unseen_cap=0.5):
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
        self.noise_scale = noise_scale
        self.tail_size = tail_size
        self.alpha_rank = alpha_rank
        self.p_unseen_cap = p_unseen_cap

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.n_train = len(y)
        self.y_train = y

        # Singleton count for Good-Turing
        unique_labels, counts = np.unique(y, return_counts=True)
        self.n_singletons = np.sum(counts == 1)

        # --- Fit KNN ---
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(
                self.model_fit, method='sigmoid', cv=10
            )
            self.calibrated.fit(X, y)
        else:
            self.calibrated = None

        # Per-class centroids and Weibull fitting
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

    def _compute_p_open(self, X, p_seen):
        """Compute per-sample KNN-based OpenMax unknown-class probability.

        Same algorithm as OpenSetKNNOpenMax._compute_p_open.
        """
        n = X.shape[0]
        K = self.num_classes

        distances = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            distances[:, j] = np.linalg.norm(X - self.centroids_[k], axis=1)

        w = np.zeros((n, K))
        for j, k in enumerate(self.classes_):
            params = self.weibull_params_.get(k) or self.pooled_weibull_
            if params is not None:
                c, scale = params
                w[:, j] = weibull_min.cdf(distances[:, j], c, loc=0, scale=scale)

        if self.alpha_rank is not None and self.alpha_rank < K:
            for i in range(n):
                top = np.argsort(-p_seen[i])[:self.alpha_rank]
                mask = np.ones(K, dtype=bool)
                mask[top] = False
                w[i, mask] = 0.0

        p_open = np.sum(p_seen * w, axis=1)
        return p_open

    def calibrate_p_open(self, X_calib, p_seen_calib):
        """Compute mean p_open on calibration data for normalisation.

        Must be called after fit() and before predict_proba() on test data.
        """
        p_open_calib = self._compute_p_open(X_calib, p_seen_calib)
        self.mean_p_open_calib_ = p_open_calib.mean()
        if self.mean_p_open_calib_ < 1e-12:
            self.mean_p_open_calib_ = 1e-12
        return p_open_calib

    def predict_proba(self, X, y_calib=None):
        """
        Predict class probabilities.

        Seen-class probabilities come from KNN.
        Unseen-class probability = Good-Turing anchor modulated by KNN OpenMax ratio.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))

        # Base KNN probabilities for seen classes
        if self.calibrated is None:
            p_seen = self.model_fit.predict_proba(X)
        else:
            p_seen = self.calibrated.predict_proba(X)

        p_seen = np.clip(p_seen, self.factor / self.num_classes, 1.0)
        p_seen = p_seen / p_seen.sum(axis=1)[:, None]

        n, K = p_seen.shape

        # Determine unseen labels
        if y_calib is not None:
            all_classes_set = set(self.classes_).union(set(np.unique(y_calib)))
            self.full_classes = np.array(sorted(all_classes_set))
            self.train_to_full_idx = {}
            for i, cls in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == cls)[0][0]
                self.train_to_full_idx[i] = full_idx
        else:
            if not hasattr(self, 'full_classes'):
                return p_seen

        num_full_classes = len(self.full_classes)
        new_prob = np.zeros((n, num_full_classes))

        # Identify unseen class indices
        unseen_mask = np.ones(num_full_classes, dtype=bool)
        for train_class in self.classes_:
            idx = np.where(self.full_classes == train_class)[0][0]
            unseen_mask[idx] = False
        unseen_indices = np.where(unseen_mask)[0]
        num_unseen = len(unseen_indices)

        if num_unseen > 0:
            # Good-Turing anchor
            p_gt = (1 + self.n_singletons) / (1 + self.n_train)

            # KNN-based OpenMax feature-dependent scores
            p_open = self._compute_p_open(X, p_seen)

            # Auto-calibrate on the first predict_proba call (calibration data)
            if not hasattr(self, 'mean_p_open_calib_') and y_calib is not None:
                self.calibrate_p_open(X, p_seen)

            # Normaliser: use calibration mean if available, else batch mean
            if hasattr(self, 'mean_p_open_calib_'):
                mean_p_open = self.mean_p_open_calib_
            else:
                mean_p_open = p_open.mean()
                if mean_p_open < 1e-12:
                    mean_p_open = 1e-12

            for i in range(n):
                # Modulated unseen probability
                ratio = p_open[i] / mean_p_open
                p_unseen_total = np.clip(
                    p_gt * ratio, 1e-20, self.p_unseen_cap
                )

                base_unseen = p_unseen_total / num_unseen
                for unseen_idx in unseen_indices:
                    noise = np.random.uniform(-self.noise_scale, self.noise_scale)
                    new_prob[i, unseen_idx] = base_unseen * (1 + noise)

                # Scale seen-class probabilities to fill remaining mass
                remaining = 1.0 - p_unseen_total
                for train_idx, train_class in enumerate(self.classes_):
                    full_idx = np.where(self.full_classes == train_class)[0][0]
                    new_prob[i, full_idx] = p_seen[i, train_idx] * remaining

            # Final renormalization
            new_prob = new_prob / new_prob.sum(axis=1)[:, None]
        else:
            for train_idx, train_class in enumerate(self.classes_):
                full_idx = np.where(self.full_classes == train_class)[0][0]
                new_prob[:, full_idx] = p_seen[:, train_idx]

        return new_prob