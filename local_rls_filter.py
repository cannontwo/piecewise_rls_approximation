import numpy as np

class RLSFilter():
    """
    Class representing the state of a recursive least squares estimator for a
    MIMO affine-linear system.
    """

    def __init__(self, state_dim, output_dim, ref_state):
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.ref_state = ref_state.reshape((self.state_dim, 1))

        self.param_dim = state_dim
        self.intercept = np.zeros((self.output_dim, 1))

        self.t = 1.0

        self.theta = np.zeros((self.param_dim, self.output_dim))
        self.covar = np.eye(self.param_dim)
        self.pred_error_covar = np.zeros((self.state_dim, self.state_dim))

        self.data = []

    def _make_feature_vec(self, state):
        assert(state.shape == (self.state_dim, 1))

        rel_state = state - self.ref_state
        assert(rel_state.shape == (self.state_dim, 1))

        return rel_state.transpose()

    def _update_covar(self, feat):
        assert(feat.shape == (1, self.param_dim))

        denom = 1.0 + feat.dot(self.covar).dot(feat.transpose())
        assert(denom.shape == (1, 1))

        num = self.covar.dot(feat.transpose()).dot(feat).dot(self.covar)
        assert(num.shape == (self.param_dim, self.param_dim))

        self.covar = self.covar - (num / denom)

    def _update_theta(self, feat, output):
        assert(feat.shape == (1, self.param_dim))
        assert(output.shape == (self.output_dim, 1))

        pred_error = output.transpose() - feat.dot(self.theta)
        correction = self.covar.dot(feat.transpose()).dot(pred_error)
        self.theta = self.theta + correction

        diff = output - feat.dot(self.theta).transpose()
        assert(diff.shape == (self.output_dim, 1))
        self.intercept = self.t * self.intercept + diff

        self.t += 1.0
        self.intercept /= float(self.t)

    def _update_pred_error_covar(self, state, output):
        pred = self.predict(state)
        feat = self._make_feature_vec(state)
        pred_error_coef = np.linalg.norm(output - pred)

        # self.t already updated by _update_theta
        self.pred_error_covar = (self.t - 1.0) * self.pred_error_covar + (pred_error_coef * feat.transpose().dot(feat))
        self.pred_error_covar /= float(self.t)
        assert(self.pred_error_covar.shape == (self.state_dim, self.state_dim))

    def process_datum(self, state, output):
        self.data.append(tuple((state, output)))

        feat = self._make_feature_vec(state)

        self._update_covar(feat)
        self._update_theta(feat, output)
        self._update_pred_error_covar(state, output)

    def predict(self, state):
        feat = self._make_feature_vec(state)
        prediction = feat.dot(self.theta).transpose() + self.intercept
        assert(prediction.shape == (self.output_dim, 1))

        return prediction

    def get_identified_mats(self):
        """
        Returns the current estimated A, B, and c matrices.
        """
        return self.theta.transpose(), self.intercept

class RLSFilterAnalyticIntercept():
    """
    Class representing the state of a recursive least squares estimator for a
    MIMO affine-linear system.
    """

    def __init__(self, state_dim, output_dim, ref_state):
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.param_dim = state_dim

        self.t = 0.0

        self.intercept = np.zeros((self.output_dim, 1))
        self.ref_state = ref_state

        self.theta = np.zeros((self.param_dim, self.output_dim))
        self.corrected_theta = np.zeros_like(self.theta)
        self.feat_mean = np.zeros((1, self.param_dim))
        self.output_mean = np.zeros((1, self.output_dim))
        self.covar = np.eye(self.param_dim)
        self.pred_error_covar = np.zeros((self.state_dim, self.state_dim))

        self.data = []

    def _make_feature_vec(self, state):
        assert(state.shape == (self.state_dim, 1))

        rel_state = state - self.ref_state
        assert(rel_state.shape == (self.state_dim, 1))

        return rel_state.transpose()

    def _update_covar(self, U, C, V):
        assert(U.shape == (self.param_dim, 2))
        assert(C.shape == (2, 2))
        assert(V.shape == (2, self.param_dim))

        inv_part = np.linalg.inv(C) + V.dot(self.covar).dot(U)
        print("inv_part is {}".format(inv_part))
        print("inv(inv_part) is {}".format(np.linalg.inv(inv_part)))
        update = self.covar.dot(U).dot(np.linalg.inv(inv_part)).dot(V).dot(self.covar)

        self.covar = self.covar - update

    def _update_theta(self, C_t, feat, output):
        assert(feat.shape == (1, self.param_dim))
        assert(output.shape == (self.output_dim, 1))
        assert(C_t.shape == (self.param_dim, self.param_dim))

        inner_term = feat.transpose().dot(output.transpose()) - C_t.dot(self.theta)
        update = self.covar.dot(inner_term)
        self.theta = self.theta + update

    def _update_pred_error_covar(self, state, output):
        pred = self.predict(state)
        feat = self._make_feature_vec(state)
        pred_error_coef = np.linalg.norm(output - pred)

        # self.t already updated by _update_theta
        self.pred_error_covar = (self.t - 1.0) * self.pred_error_covar + (pred_error_coef * feat.transpose().dot(feat))
        self.pred_error_covar /= float(self.t)
        assert(self.pred_error_covar.shape == (self.state_dim, self.state_dim))

    def _update_output_mean(self, output):
        assert(output.shape == (self.output_dim, 1))
        self.output_mean = self.output_mean + (1.0 / self.t) * (output.transpose() - self.output_mean)

    def _update_feat_mean(self, feat):
        assert(feat.shape == (1, self.param_dim))
        self.feat_mean = self.feat_mean + (1.0 / self.t) * (feat - self.feat_mean)

    def _make_U(self, feat):
        assert(feat.shape == (1, self.param_dim))
        return np.block([self.feat_mean.transpose(), feat.transpose()])

    def _make_V(self, feat):
        assert(feat.shape == (1, self.param_dim))
        return np.block([[self.feat_mean],[feat]])

    def _make_C(self):
        return (1 / self.t ** 2) * np.array([[((2.0 * self.t - 1.0) ** 2) - 2.0 * (self.t ** 2), -(2.0 * self.t - 1.0) * (self.t - 1.0)],
                         [-(2.0 * self.t - 1.0) * (self.t - 1.0), (self.t - 1.0) ** 2]])

    def process_datum(self, state, output):
        self.data.append(tuple((state, output)))
        feat = self._make_feature_vec(state)
        self.t += 1.0

        if self.t == 1.0:
            self._update_feat_mean(feat)
            self._update_output_mean(output)
            return

        U = self._make_U(feat)
        V = self._make_V(feat)
        C = self._make_C()
        C_t = U.dot(C).dot(V)

        self._update_covar(U, C, V)
        self._update_output_mean(output)
        self._update_feat_mean(feat)
        self._update_theta(C_t, feat, output)
        self.corrected_theta = self.theta - ((2 * self.t - 1) * self.covar.dot(self.feat_mean.transpose()).dot(self.output_mean))
        self.intercept = (self.output_mean - self.feat_mean.dot(self.corrected_theta)).transpose()
        self._update_pred_error_covar(state, output)

    def get_identified_mats(self):
        """
        Returns the current estimated A, B, and c matrices.
        """
        return self.corrected_theta.transpose(), self.intercept

    def predict(self, state):
        feat = self._make_feature_vec(state)
        prediction = feat.dot(self.corrected_theta).transpose() + self.intercept
        assert(prediction.shape == (self.output_dim, 1))

        return prediction
