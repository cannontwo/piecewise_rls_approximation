import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial import cKDTree as KDTree
from local_rls_filter import RLSFilter

class PWAModel():
    """
    Class representing a piecewise-affine regression model on a Voronoi
    partition.
    """
    def __init__(self):
        self.ref_points = [np.array([0.5, 0.5])]
        self.kdt = KDTree(np.array(self.ref_points), leafsize=5)

        self.local_models = {}
        self.local_models[tuple(self.ref_points[0])] = RLSFilter(2, 1, self.ref_points[0].reshape((2, 1)))

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

    def _line_search(self, ref, vec):
        assert(np.linalg.norm(vec) >= 1e-10)
        vec = vec.reshape((2, 1))
        print("Line search vec is {}".format(vec))
        ref = ref.reshape((2, 1))
        coef = 1.0
        line_ref = self.get_nearest_reference(ref + coef * vec)
        print("Line ref is {}".format(line_ref.reshape((2, 1))))
        print(np.array_equal(line_ref.reshape((2, 1)), ref))
        while not np.array_equal(line_ref.reshape((2, 1)), ref) or np.linalg.norm(line_ref - np.array([0.5, 0.5])) > 1.0:
            coef *= 0.5
            line_ref = self.get_nearest_reference(ref + coef * vec)
            print("Line search for ref {} checking {}".format(ref, ref + coef * vec))

            if np.array_equal(ref, ref + coef * vec):
                return ref + 0.001 * np.random.randn(2, 1)

        assert((ref + coef * vec).shape == (2, 1))
        return ref + coef * vec

    def _split_ref(self, ref):
        model = self.local_models[tuple(ref)]

        U, s, _ = np.linalg.svd(model.pred_error_covar)
        print("Singular values are {}".format(s))
        if model.t > 2 and s[0] < 0.001/float(model.t):
            return 

        model = self.local_models.pop(tuple(ref))
        max_sing_vec = U[:, 0].reshape((2, 1))

        new_ref_1 = self._line_search(ref, max_sing_vec).reshape((2,))
        new_ref_2 = self._line_search(ref, -1.0 * max_sing_vec).reshape((2,))

        print("Splitting ref {} into refs {} and {}".format(ref, new_ref_1, new_ref_2))

        self._remove_ref(ref)
        self.ref_points.append(new_ref_1)
        self.ref_points.append(new_ref_2)

        print(tuple(new_ref_1))
        self.local_models[tuple(new_ref_1)] = RLSFilter(2, 1, new_ref_1.reshape((2, 1)))
        self.local_models[tuple(new_ref_2)] = RLSFilter(2, 1, new_ref_2.reshape((2, 1)))

        self.kdt = KDTree(np.array(self.ref_points), leafsize=5)

        for state, output in model.data:
            self.t -= len(model.data)
            self.process_datum(state, output)

    def process_datum(self, state, output):
        assert(state.shape == (2,1))

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

    def plot_model(self, ax, title):
        x = np.linspace(0.0, 1.0, 768)
        y = np.linspace(0.0, 1.0, 768)

        nx = len(x)
        ny = len(y)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(nx):
            for j in range(ny):
                input_vec = np.array([X[i,j], Y[i,j]]).reshape((2,1))
                Z[i, j] = self.predict(input_vec)

        plt.sca(ax)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.contourf(x, y, Z, levels=255, cmap='gray')
        plt.clim(0., 1.)
        plt.colorbar()
        ax.set_title(title)

        #ref_x = []
        #ref_y = []
        #for ref_point in self.ref_points:
        #    ref_x.append(ref_point[0])
        #    ref_y.append(ref_point[1])

        #ax.scatter(ref_x, ref_y, c='k')
