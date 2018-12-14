import numpy as np
import os
# Plotoimg Functions
import matplotlib
import time
import scipy
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.metrics.pairwise import euclidean_distances

data_dir = './'
# Load the digtis from the given file
digits_path = os.path.join(data_dir, 'digits-labels.npz')
load_dic = np.load(digits_path)
digits_labels = load_dic['l']
digits_data = load_dic['d']

n_total_digits = digits_data.shape[1]
# just check one input datapoint and its corresponding label:
indx = np.random.randint(n_total_digits)
print(n_total_digits)
print("Showing a digit: {}".format(digits_labels[indx]))

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state

from mds_fast import (
    distance_matrix,
    update_distance_matrix,
    c_pertub_error as best_pertubation,
    mse as mse1d,
    mse2 as mse2d,
)


def _log_iteration(turn, radius, prev_error, error):
    print("Turn {0}: Radius {1}: (prev, error decrease, error): "
          "({2}, {3}, {4})"
          .format(turn, radius, prev_error, prev_error - error, error))


def _radius_update(radius, error, prev_error, tolerance=1e-4):
    if error >= prev_error or prev_error - error <= error * tolerance:
        return radius * 0.5
    return radius


def _point_sampling(points, keep_percent=1.0, turn=-1, recalculate_each=-1):
    if keep_percent > 1.0 or 1.0 - keep_percent < 1e-5:
        return points
    if turn > 0 and recalculate_each > 0 and turn % recalculate_each == 0:
        return points
    keep = int(points.shape[0] * keep_percent)
    return np.random.choice(points, size=keep, replace=False)


def initialize_prob_matrix(n_samples,
                           n_components,
                           mode,
                           initial_prob):
    bs_prob = np.ones((n_samples, 2 * n_components),
                      dtype=np.float64)
    if mode == 'randomized' or mode == 'bootstrapped':
        bs_prob.fill(initial_prob)

    return bs_prob


def pattern_search_mds(d_goal, init=None, n_components=2, starting_radius=1.0,
                       radius_update_tolerance=1e-4, sample_points=1.0,
                       explore_dim_percent=1.0, max_iter=1000,
                       radius_barrier=1e-3, n_jobs=1, verbose=0,
                       random_state=None,
                       mode='bootstraped',
                       initial_prob=0.5,
                       a_bs=0.05,
                       prob_thresh=0.2):
    n_samples = d_goal.shape[0]
    random_state = check_random_state(random_state)
    xs = (init if init is not None
          else random_state.rand(n_samples, n_components))
    d_current = distance_matrix(xs)
    points = np.arange(xs.shape[0])

    bs_prob = initialize_prob_matrix(n_samples,
                                     n_components,
                                     mode,
                                     initial_prob)

    radius = starting_radius
    turn = 0
    error = mse2d(d_goal, d_current)
    prev_error = np.Inf
    time_logger = []

    if verbose >= 2:
        time_logger.append([time.time(), 0, error/2.])

    while turn <= max_iter and radius > radius_barrier:
        turn += 1
        radius = _radius_update(
            radius, error, prev_error, tolerance=radius_update_tolerance)
        prev_error = error
        for point in points:
            point_error = mse1d(d_goal[point], d_current[point])
            (optimum_error,
             optimum_k,
             optimum_step,
             optimum_coord_ind) = best_pertubation(
                xs, radius, d_current, d_goal,
                bs_prob,
                point,
                a_bs,
                prob_thresh,
                percent=explore_dim_percent, n_jobs=n_jobs)
            if point_error > optimum_error:
                error -= (point_error - optimum_error)
                d_current = update_distance_matrix(
                    xs, d_current, point, optimum_step, optimum_k)
                xs[point, optimum_k] += optimum_step

        if verbose >= 2:
            time_logger.append([time.time(),
                                turn,
                                error/2.])
    if verbose >= 1:
        print("Ending Error: {}".format(error))

    return xs, error/2., turn, time_logger


class MDS(BaseEstimator):
    def __init__(self,
                 n_components=2,
                 starting_radius=1.0,
                 max_iter=2,
                 radius_barrier=1e-3,
                 explore_dim_percent=1.0,
                 sample_points=1.0,
                 radius_update_tolerance=1e-4,
                 verbose=0,
                 random_state=None,
                 a_bs=0.05,
                 prob_thresh=0.2,
                 mode='bootstrapped',
                 initial_prob=0.1,
                 n_jobs=1,
                 dissimilarity='precomputed'):
        self.radius_update_tolerance = radius_update_tolerance
        self.sample_points = sample_points
        self.n_components = n_components
        self.starting_radius = starting_radius
        self.max_iter = max_iter
        self.radius_barrier = radius_barrier
        self.explore_dim_percent = explore_dim_percent
        self.num_epochs = 0
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.dissimilarity = dissimilarity

        available_modes = ['bootstrapped', 'randomized', 'full_search']
        if mode in available_modes:
            self.mode = mode
        else:
            raise NotImplementedError("Mode: {} is not yet available."
                                      " Try one of: {} instead"
                                      "".format(mode, available_modes))

        if mode in ['randomized', 'full_search'] and not a_bs == 0.:
            print("a_bs for randomized and full search modes of "
                  "coordinate search MDS should be 0. because there "
                  "should be no updates of the probability for each "
                  "dimension. Assigning a_bs = 0 now...")
            self.a_bs = 0.
        else:
            self.a_bs = a_bs
        # in the case of bootstrapped mode prob_thresh plays the role
        # of the probability threshold that all dimensions have
        self.prob_thresh = prob_thresh

        # In the case of bootstrapped search this is
        # the probability that all dimensions have initially
        if mode == 'full_search':
            print("initial_prob for full search mode of "
                  "coordinate search MDS should be 1."
                  "And there should be no updates of the probability "
                  "for each dimension. Assigning initial_prob = 1 "
                  "now...")
            self.initial_prob = 1.
        else:
            self.initial_prob = initial_prob
        self.initial_prob = initial_prob

    def fit_transform(self, X, init=None):
        X = X.astype(np.float64)
        X = check_array(X)
        d_goal = (X if self.dissimilarity == 'precomputed'
                  else distance_matrix(X))
        self.embedding_, self.error_, self.n_iter_, time_logger\
            = pattern_search_mds(
            d_goal, init=init, n_components=self.n_components,
            starting_radius=self.starting_radius, max_iter=self.max_iter,
            sample_points=self.sample_points,
            explore_dim_percent=self.explore_dim_percent,
            radius_update_tolerance=self.radius_update_tolerance,
            radius_barrier=self.radius_barrier,
            n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state,
            a_bs=self.a_bs,
            initial_prob=self.initial_prob,
            prob_thresh=self.prob_thresh,
            mode=self.mode
        )
        return self.embedding_, time_logger

    def fit(self, X, init=None):
        self.fit_transform(X, init=init)

        return self


# Create the distance graph for all the input
from itertools import product


def get_nearest_neighbors_edges(x, k=6, distance_metric='euclidean'):
    n_dim, n_samples = x.shape
    edges = []
    pairwise_distances = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(x.T, metric=distance_metric))
    for s_id in np.arange(n_samples):
        neighbors = np.argsort(pairwise_distances[s_id, :])
        closest_neighbors = neighbors[:k + 1]
        for neigh_id in closest_neighbors:
            edges.append(
                [s_id, neigh_id, pairwise_distances[s_id, neigh_id]])
    return edges


def floyd_warshall(x, k=6, distance_metric='euclidean'):
    edges = get_nearest_neighbors_edges(x, k=6)
    n_dim, n_samples = x.shape
    D = np.full((n_samples, n_samples), np.finfo(np.float16).max,
                dtype=np.float32)
    for (st_ind, end_id, distance) in edges:
        D[st_ind][end_id] = distance

    #     D_true = D.copy()
    #     for k in np.arange(n_samples):
    #         for i in np.arange(n_samples):
    #             for j in np.arange(n_samples):
    #                 if D_true[i][j] > D_true[i][k] + D_true[k][j]:
    #                     D_true[i][j] = D_true[i][k] + D_true[k][j]

    for k in range(n_samples):
        D = np.minimum(D,
                       D[:, int(k), np.newaxis] + D[np.newaxis, int(k),
                                                  :])

    #     assert np.array_equal(D, D_true)

    return D


def pairwise_euclidean(X):
    return euclidean_distances(X, X)


def thymios_ISOMAP(x, n_components=3, k_neighbors=6):
    print("Computing the distances using Floyd Warshall algorithm...")
    Geodesic_D = floyd_warshall(x, k=k_neighbors,
                                distance_metric='euclidean')
    print("Performing Pattern Search MDS on the Geodesic distances...")
    pat_search_MDS_creator = MDS(n_components=n_components,
                                 dissimilarity='precomputed')
    x_low_rank = pat_search_MDS_creator.fit_transform(Geodesic_D)
    print("ISOMAP solution Ready")
    return x_low_rank.T

if __name__ == "__main__":
    # do some evaluation using mnist data and selected digits
    selected_digits = [4, 7, 8]
    groups_counts = np.bincount(digits_labels)
    sorted_numberes_indxes = digits_labels.argsort()
    sums_of_counts = np.cumsum(groups_counts)
    list_of_sel_digits_data = [sorted_numberes_indxes[sums_of_counts[dig-1]:sums_of_counts[dig]]
                               for dig in selected_digits]
    selected_digits_data = digits_data[:, np.concatenate(list_of_sel_digits_data, axis=0)]
    selected_digits_labels = digits_labels[np.concatenate(list_of_sel_digits_data, axis=0)]

    n_digits = 1000
    selected_indexes = np.random.choice(selected_digits_labels.shape[0],
                                        n_digits, replace=False)
    sel_digits_labels = selected_digits_labels[selected_indexes]
    sel_digits_data = selected_digits_data[:, selected_indexes]

    D_target = pairwise_euclidean(sel_digits_data.T)
    np.set_printoptions(threshold=np.nan)
    pat_search_MDS_creator = MDS(n_components=2,
                                 starting_radius=10.,
                                 explore_dim_percent=1.,
                                 max_iter=200,
                                 mode='bootstrapped',
                                 prob_thresh=0.3,
                                 initial_prob=.6,
                                 a_bs=0.05,
                                 verbose=2,
                                 dissimilarity='precomputed')
    (x_low_rank,
     time_logger) = pat_search_MDS_creator.fit_transform(D_target[:1000,
                                                                  :1000])
    print (time_logger)
