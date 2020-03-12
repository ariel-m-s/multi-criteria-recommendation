import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def similarity_matrix(feat_mat):
    """
    Computes the row-wise cosine similarity matrix of a rating matrix.

    Parameters:
    feat_mat (matrix-like) -- the rating matrix.

    Does not return.
    """
    sim_mat = cosine_similarity(feat_mat)
    np.fill_diagonal(sim_mat, 0)
    return sim_mat


def sim_avg(sim_mats):
    """
    Computes the element-wise average (mean) of some matrices.

    Parameters:
    sim_mats (array-like) -- the matrices.

    Returns:
    (matrix-like) -- the matrix with the average (mean) values.
    """
    return np.array(sim_mats).mean(axis=0)


def sim_min(sim_mats):
    """
    Computes the element-wise minimum of some matrices.

    Parameters:
    sim_mats (array-like) -- the matrices.

    Returns:
    (matrix-like) -- the matrix with the minimum values.
    """
    return np.array(sim_mats).min(axis=0)


def sim_max(sim_mats):
    """
    Computes the element-wise maximum of some matrices.

    Parameters:
    sim_mats (array-like) -- the matrices.

    Returns:
    (matrix-like) -- the matrix with the maximum values.
    """
    return np.array(sim_mats).max(axis=0)
