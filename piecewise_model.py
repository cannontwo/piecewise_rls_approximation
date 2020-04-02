import numpy as np

from scipy.spatial import cKDTree as KDTree
from local_rls_filter import RLSFilter

class PWAModel():
    """
    Class representing a piecewise-affine regression model on a Voronoi
    partition.
    """
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.ref_points = [np.array([0.5, 0.5])]
        self.kdt = KDTree(np.array(self.ref_points), leafsize=5)

        self.local_models = {}
        self.local_models[tuple(self.ref_points[0])] = RLSFilter(2, self.output_dim, self.ref_points[0].reshape((2, 1)))

        self.t = 1.0

    def get_nearest_idx(self, query):
        return self.kdt.query(query, k=1)[1]

    def get_nearest_reference(self, query):
        kdt_query = query.reshape((2,))
        return self.ref_points[self.kdt.query(kdt_query, k=1)[1]]

    def _remove_ref(self, ref):
        i = 0
        size = len(self.ref_points)
        while i != size and not np.array_equal(self.ref_points[i], ref):
            i += 1

        if i != size:
            self.ref_points.pop(i)
        else:
            raise ValueError("Ref not found")

    def _valid_ref(self, ref):
        assert(ref.shape == (2, 1))
        x_part = ref[0][0] >= 0.0 and ref[0][0] <= 1.0
        y_part = ref[1][0] >= 0.0 and ref[1][0] <= 1.0

        return x_part and y_part

    def _line_search(self, ref, vec):
        assert(np.linalg.norm(vec) >= 1e-10)
        vec = vec.reshape((2, 1))
        ref = ref.reshape((2, 1))
        coef = 1.0
        line_ref = self.get_nearest_reference(ref + coef * vec)
        while not np.array_equal(line_ref.reshape((2, 1)), ref) or not self._valid_ref(ref + coef * vec):
            coef *= 0.5
            line_ref = self.get_nearest_reference(ref + coef * vec)

            if np.array_equal(ref, ref + coef * vec):
                return ref + 0.001 * np.random.randn(2, 1)

        assert((ref + coef * vec).shape == (2, 1))
        return ref + coef * vec

    def _split_ref(self, ref):
        model = self.local_models[tuple(ref)]

        U, s, _ = np.linalg.svd(model.pred_error_covar)
        if model.t > 2 and s[0] < 0.001/float(model.t):
            return 

        model = self.local_models.pop(tuple(ref))
        max_sing_vec = U[:, 0].reshape((2, 1))

        new_ref_1 = self._line_search(ref, max_sing_vec).reshape((2,))
        new_ref_2 = self._line_search(ref, -1.0 * max_sing_vec).reshape((2,))

        self._remove_ref(ref)
        self.ref_points.append(new_ref_1)
        self.ref_points.append(new_ref_2)

        self.local_models[tuple(new_ref_1)] = RLSFilter(2, self.output_dim, new_ref_1.reshape((2, 1)))
        self.local_models[tuple(new_ref_2)] = RLSFilter(2, self.output_dim, new_ref_2.reshape((2, 1)))

        self.kdt = KDTree(np.array(self.ref_points), leafsize=5)

        self.t -= len(model.data)
        for state, output in model.data:
            self.process_datum(state, output)

    def remove_random_ref(self):
        ref_idx = np.random.randint(0, len(self.ref_points))
        ref = self.ref_points[ref_idx]

        model = self.local_models.pop(tuple(ref))
        self._remove_ref(ref)

        self.kdt = KDTree(np.array(self.ref_points), leafsize=5)
        self.t -= len(model.data)
        for state, output in model.data:
            self.process_datum(state, output)

    def process_datum(self, state, output):
        assert(state.shape == (2,1))
        assert(output.shape == (self.output_dim, 1))

        self.t += 1.0

        ref = self.get_nearest_reference(state)
        model = self.local_models[tuple(ref)]
        model.process_datum(state, output)

        if model.t > 10:
            self._split_ref(ref)

    def predict(self, x):
        assert(x.shape == (2,1))
        ref = self.get_nearest_reference(x)

        return self.local_models[tuple(ref)].predict(x)
