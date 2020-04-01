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
        assert(prediction.shape == (1, 1))

        return prediction

    def get_identified_mats(self):
        """
        Returns the current estimated A, B, and c matrices.
        """
        return self.theta.transpose(), self.intercept
