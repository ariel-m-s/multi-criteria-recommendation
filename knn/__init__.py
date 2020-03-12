import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from knn.similarity import similarity_matrix, sim_avg, sim_min, sim_max


WRONG_SHAPE_MSG = ("Missing users or items. "
                   "Please split the datasets differently and train me again.")
FORBIDDEN_SIM_TYPE_MSG = ("Similarity type not allowed. "
                          "Please choose 'min', 'avg' or 'max'")


class MultiCriteriaKnnModel:
    """
    A multi-criteria rating system (rating vector)
    Nearest Neighbors (KNN) implementation (non-optimized algorithm).
    """
    ALLOWED_SIM_TYPES = ("min", "avg", "max")

    def __init__(self):
        """
        Constructs the KNN model.
        """
        self.trained = False
        self._sim_type = MultiCriteriaKnnModel.ALLOWED_SIM_TYPES[0]

    def fit(self, ratings_mats):
        """
        Trains the KNN model.

        Parameters:
        rating_mats (array-like) -- contains the training rating matrices
        of shape users x [times] items.

        Does not return.
        """
        assert len(ratings_mats) > 1

        ratings_sim_mats = tuple(map(similarity_matrix, ratings_mats[1:]))
        self._avg_sim_mat = sim_avg(ratings_sim_mats)
        self._min_sim_mat = sim_min(ratings_sim_mats)
        self._max_sim_mat = sim_max(ratings_sim_mats)

        self.overall = ratings_mats[0]
        self.overall[self.overall == 0] = np.nan

        self.trained = True
        self.update()

    @property
    def sim_type(self):
        """
        GETS similarity type (can be any of self.ALLOWED_SIM_TYPES).

        Returns:
        (string) -- the similarity type.
        """
        assert self.trained
        return self._sim_type

    @sim_type.setter
    def sim_type(self, value):
        """
        SETS similarity type (can be any of self.ALLOWED_SIM_TYPES).

        Parameters:
        value (string) -- the similarity type.

        Does not return.
        """
        assert self.trained
        if value not in MultiCriteriaKnnModel.ALLOWED_SIM_TYPES:
            raise ValueError(FORBIDDEN_SIM_TYPE_MSG)
        self._sim_type = value
        self.update()

    def update(self):
        """
        Updates the similarity matrix
        corresponding to the set similarity type.

        Does not return.
        """
        assert self.trained
        if self._sim_type == "min":
            self.sim_mat = self._min_sim_mat
        elif self._sim_type == "avg":
            self.sim_mat = self._avg_sim_mat
        elif self._sim_type == "max":
            self.sim_mat = self._max_sim_mat

    def predict(self, usr_idx, itm_idx, k=10):
        """
        Estimates an overall rating value for a specific user-item pair.

        Parameters:
        usr_idx (integer; >= 0) -- the user's index in the rating matrix.
        itm_idx (integer; >= 0) -- the item's index in the rating matrix.
        k (integer; >= 0; default 10) -- the number of 'nearest neighbors'.

        Returns:
        (float; in [1, 5]) -- the estimated overall rating value.
        """
        assert self.trained
        mask = np.logical_not(np.isnan(self.overall[:, itm_idx]))

        usr_sims = self.sim_mat[usr_idx]
        usr_sims_filtered = usr_sims[mask]
        usr_ind = np.argpartition(usr_sims_filtered, -k)[-k:]
        overall_filtered = self.overall[mask]

        norm_sum = np.average(
            overall_filtered[usr_ind, itm_idx] - np.nanmean(
                overall_filtered[usr_ind], axis=1),
            weights=usr_sims_filtered[usr_ind])
        usr_bias = np.nanmean(self.overall[usr_idx])
        return usr_bias + norm_sum

    def test(self, rating_mat, k=10):
        """
        Reports the performance of the testing set.

        Parameters:
        rating_mats (matrix-like) -- the testing rating matrix
        of shape users x [times] items.
        k (integer; >= 0; default 10) -- the number of 'nearest neighbors'.

        Returns (2-tuple):
        (float; in [0, 4]) -- the RMSE (root mean squared error; MSE**0.5).
        (float; in [0, 4]) -- the MAE (mean absolute error).
        """
        assert self.trained
        if rating_mat.shape != self.overall.shape:
            raise ValueError(WRONG_SHAPE_MSG)

        rating_mat[rating_mat == 0] = np.nan
        source = coo_matrix(rating_mat)
        real_ratings = []
        predictions = []

        for usr_idx, itm_idx, value in zip(
                source.row, source.col, source.data):
            if not np.isnan(value):
                real_ratings.append(value)
                predictions.append(self.predict(usr_idx, itm_idx, k))
        return (mean_squared_error(real_ratings, predictions) ** 0.5,
                mean_absolute_error(real_ratings, predictions))
