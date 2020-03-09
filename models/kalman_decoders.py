import numpy as np
import numpy as cp  # You could also try to use cupy when there's more taps
from tqdm import tqdm
import scipy
# Just for debugging, later delete
import pdb


# Since writing UKF too, refer to the paper for comparison between KF and UKF in BMI context:
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006243
class KalmanFilter():
    """Kalman Filter class.

    Ridge Regression optimized implementation of Kalman Filter.
    Using cupy (GPU computing ver of numpy) to speed up the computation.

    Arguments:
        arg1: something.

    Properties:
    """

    def __init__(self, args):
        """
        Defining a constructor.

        Arguments:
            config: config file specifying configurations.
        """

        self.args = args
        assert self.args.kalman_order > 0, "Kalman order should be from 1 ~"
        if args.kalman_gpu:
            import cupy as cp

    def train(self, trainX, trainY):
        """Training Kalman Filter.

        Put summary here

        Arguments:
            trainX: input data in numpy array. dimension: [num_samples, num_eeg_chans]
            trainY: input target in numpy array. dimension: [num_samples, num_kin_joints]
        """

        # Kalman order n
        n = self.args.kalman_order

        # First get X and X_lag
        # It's confusing but the original paper use X for state
        # Y for measurement so need to switch
        X_shifted = cp.asarray(trainY.T)
        Y_shifted = cp.asarray(trainX.T)

        # ===== F: Transition matrix =====
        self.train_F(n, X_shifted)

        # ===== Q: Covariance matrix =====
        self.train_Q(n)

        # From here, KF implementation is different from UKF

        # ===== H: Measurement matrix (It's similar to B in UKF) =====
        # First get Y
        # Remove the first (kalman_order - future_steps) columns and last future_steps columns
        Y = Y_shifted[:, n-1:-self.args.future_step]
        # Then calculate H which is equivalent to B (equation 21)
        inside_inv = self.X_lag @ self.X_lag.T + \
            self.args.kalman_lambda_B * cp.eye(self.state_dim * n)
        self.H = Y @ self.X_lag.T @ cp.linalg.inv(inside_inv)

        # ===== R: Noise covariance matrix =====
        # First calculate E_B
        E_B = Y - self.H @ self.X_lag
        # Then calculate R (equation 23)
        self.R = E_B @ E_B.T / (self.num_samp - n + 1 - self.dim)

        # Other things
        self.x_t = cp.zeros((self.dim, 1))
        self.P_t = self.Q

    def train_F(self, n, X_shifted):
        """Train Transition matrix F

        Separated for inheritance purpose for UKF.

        Arguments:
            n: kalman order
        """

        self.X, self.X_lag = self._state_model(X_shifted)
        self.state_dim = self.X.shape[0]
        self.dim = self.state_dim * n if n is not 0 else 1
        # Calculate F_part: partial F (equation 18)
        inside_inv = self.X_lag @ self.X_lag.T + \
            self.args.kalman_lambda_F * cp.eye(self.dim)
        self.F_part = self.X @ self.X_lag.T @ cp.linalg.inv(inside_inv)
        # Finally create F (equation 19)
        if (n == 0) or (n == 1):
            self.F = self.F_part
        else:
            self.F = cp.vstack((
                self.F_part,
                cp.hstack((
                    cp.eye(self.state_dim * (n - 1)),
                    cp.zeros((self.state_dim * (n - 1), self.state_dim))
                ))
            ))

    def train_Q(self, n):
        """Train Covariance matrix Q

        """

        # First calculate E_F
        E_F = self.X - self.F_part @ self.X_lag
        # Then calculate Q_part (equation 20)
        self.num_samp = self.X.shape[1]
        Q_part = E_F @ E_F.T / (self.num_samp - n - self.dim)
        # Finally create Q (equation 21)
        if (n == 0) or (n == 1):
            self.Q = Q_part
        else:
            self.Q = cp.vstack((
                cp.hstack((Q_part,
                               cp.zeros(
                                   (self.state_dim, self.dim-self.state_dim))
                               )),
                cp.zeros((self.dim-self.state_dim, self.dim))
            ))

    def process(self, testX, first_state):
        """Testing Kalman Filter.

        This processes whole dataset using "predict" and "update" per sample.
        If you want to fix the KF at some point or want to use custom predict and update
        algorithms, you could just use the "predict" and "update" methods from the class.

        Arguments:
            testX: Obsrvation data (e.g. EEG) dimension: [num_samples, num_eeg_chans]

        Returns:
            predicted: Predicted state data (e.g. Kinematics) dimension: [num_samples, num_features]
        """

        # Going to predict sample by sample, making matrix of given data
        num_samp = testX.shape[0]
        num_states = self.args.num_chan_kin
        Z = cp.asarray(testX.T)

        # Initializations
        states = []
        states.append(first_state.tolist())

        # Make prediction sample by sample
        for i in tqdm(range(num_samp - 1)):
            # predict step
            self.predict()

            # update step
            measure_samp = Z[:, i].reshape(Z.shape[0], 1)
            self.update(measure_samp)

            # Log the results
            pred_extracted = self.x_t
            states.append(pred_extracted.T.tolist()[0])

        return np.array(states)

    def predict(self):
        """Predict step

        Reference: Table 1. from Li et al. 2009 (Left side)
        """

        # x_pt: x_t' = predicted state
        # P_pt: P_t' = predicted state covariance
        self.x_pt = cp.real(self.F @ self.x_t)
        self.P_pt = cp.real(self.F @ self.P_t @ self.F.T + self.Q)

    def update(self, measure_samp):
        """Update step

        Reference: Table 1. from Li et al. 2009 (Left side)
        """
        self.z_t = self.H @ self.x_pt
        self.S_t = self.H @ self.P_pt @ self.H.T + self.R

        # Kalman Gain
        self.K_t = self.P_pt @ self.H.T @ cp.linalg.inv(self.S_t)

        # Make predictions
        self.x_t = self.x_pt + self.K_t @ (measure_samp - self.z_t)
        self.P_t = (cp.eye(self.dim) - self.K_t @ self.H) @ self.P_pt

    def _state_model(self, X_orig):
        """A function to construct state model

        Put summary here.
        Reference: Check the paragraphs 1 and 2 in page 16 of
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0006243&type=printable

        Arguments:
            X_orig: Original training data position and velocities (e.g. kinematics)
                    dimension: [num_features, num_samples]

        Returns:
            X:      State matrix (e.g. kinematics)
                    dimension: [num_features, num_samples-kalman_order]
                    in the origitnal paper, dim = [4, T]
            X_lag:  State matrix with lagged information.
                    dimension: [num_features*kalman_order, num_samples-kalman_order]
                    in the original paper, dim = [4n, T]
        """

        # Get number of samples
        num_samp = X_orig.shape[1]
        # Get the kalman order "n"
        n = self.args.kalman_order
        # Based on the kalman order, create X_lag
        # X_lag: State matrix with lagged information.
        #        dimension: [num_features*kalman_order, num_samples-kalman_order]
        # Exceptional case: kalman_order = 0. If this is the case, X and X_lag is the same.
        #                   However, kalman_order shouldn't be 0.
        if n == 0:
            X = X_orig
            X_lag = X_orig
            return X, X_lag
        else:
            # For creating a stacked matrix for lagged matrix
            for i in range(n):
                # Stack the matrix in order
                if i == 0:
                    X_orig_stacked = X_orig
                else:
                    # Pre-defining lagged matrix with zeros
                    X_orig_lag = cp.zeros(X_orig.shape)
                    # Replace part of the above defined zero matrix with lagged matrix
                    X_orig_lag[:, :-(i+1)] = X_orig[:, i:-1]
                    # Vertical concatenation of columns i-1, i-2,... ,i-n of matrix X_orig
                    X_orig_stacked = cp.vstack([X_orig_lag, X_orig_stacked])
            # Now you just need to extract the first num_samp - kalman_order columns
            X_lag = X_orig_stacked[:, :num_samp - n]
            # Make sure you omit the first kalman_order columns for X as stated in the paper
            X = X_orig[:, n:]

            return X, X_lag


# TODO: Later inherit from KF
class UnscentedKalmanFilter():
    def __init__(self, args):
        self.args = args
        self.positive_definite_count = 0
        if args.ukf_gpu:
            import cupy as cp

    def train(self, trainX, trainY):
        """Training Kalman Filter.

        Put summary here

        Arguments:
            trainX: input data in numpy array. dimension: [num_samples, num_eeg_chans]
            trainY: input target in numpy array. dimension: [num_samples, num_kin_joints]
        """

        # Kalman order n
        n = self.args.kalman_order

        # First get X and X_lag
        # It's confusing but the original paper use X for state
        # Y for measurement so need to switch
        X_shifted = cp.asarray(trainY.T)
        Y_shifted = cp.asarray(trainX.T)

        # ===== F: Transition matrix =====
        self.train_F(n, X_shifted)

        # ===== Q: Covariance matrix =====
        self.train_Q(n)

        # ===== B: Measurement matrix =====
        # First get Y
        # Remove the first (kalman_order - future_steps) columns and last future_steps columns
        Y = Y_shifted[:, n-1:-self.args.future_step]
        X_aug = self._augment(self.X_lag)
        # Then calculate H which is equivalent to B (equation 21)
        inside_inv = X_aug @ X_aug.T + self.args.kalman_lambda_B * \
            cp.eye((self.state_dim+1) * n)
        self.B = Y @ X_aug.T @ cp.linalg.inv(inside_inv)

        # ===== R: Noise covariance matrix =====
        # First calculate E_B
        E_B = Y - self.B @ X_aug
        # Then calculate R (equation 23)
        self.R = E_B @ E_B.T / (self.num_samp - n + 1 - self.dim)

        # Other things
        self.x_t = cp.zeros((self.dim, 1))
        self.P_t = self.Q
        self.dim = self.dim
        # initialize weights
        self._calc_weights(self.args)

    def train_F(self, n, X_shifted):
        """Train Transition matrix F

        Separated for inheritance purpose for UKF.

        Arguments:
            n: kalman order
        """

        self.X, self.X_lag = self._state_model(X_shifted)
        self.state_dim = self.X.shape[0]
        self.dim = self.state_dim * n if n is not 0 else 1
        # Calculate F_part: partial F (equation 18)
        inside_inv = self.X_lag @ self.X_lag.T + \
            self.args.kalman_lambda_F * cp.eye(self.dim)
        self.F_part = self.X @ self.X_lag.T @ cp.linalg.inv(inside_inv)
        # Finally create F (equation 19)
        if (n == 0) or (n == 1):
            self.F = self.F_part
        else:
            self.F = cp.vstack((
                self.F_part,
                cp.hstack((
                    cp.eye(self.state_dim * (n - 1)),
                    cp.zeros((self.state_dim * (n - 1), self.state_dim))
                ))
            ))

    def train_Q(self, n):
        """Train Covariance matrix Q

        """

        # First calculate E_F
        E_F = self.X - self.F_part @ self.X_lag
        # Then calculate Q_part (equation 20)
        self.num_samp = self.X.shape[1]
        Q_part = E_F @ E_F.T / (self.num_samp - n - self.dim)
        # Finally create Q (equation 21)
        if (n == 0) or (n == 1):
            self.Q = Q_part
        else:
            self.Q = cp.vstack((
                cp.hstack((Q_part,
                               cp.zeros(
                                   (self.state_dim, self.dim-self.state_dim))
                               )),
                cp.zeros((self.dim-self.state_dim, self.dim))
            ))

    def process(self, testX, first_state):
        """Testing Kalman Filter.

        Put summary here.
        :param
        - testX: Obsrvation data (e.g. EEG) dimension: [num_samples, num_eeg_chans]
        :return predicted: Predicted state data (e.g. Kinematics) dimension: [num_samples, num_features]
        """

        # Going to predict sample by sample, making matrix of given data
        num_samp = testX.shape[0]
        num_states = self.args.num_chan_kin
        Y = cp.asarray(testX.T)

        # Initializations
        states = []
        states.append(first_state.tolist())

        # Make prediction sample by sample
        for i in tqdm(range(num_samp - 1)):
            # ===== Predict step =====
            self.predict()

            # ===== Update step =====
            measure_samp = Y[:, i].reshape(Y.shape[0], 1)
            self.update(measure_samp)

            # Log the results
            pred_extracted = self.x_t
            states.append(pred_extracted.T.tolist()[0])

        return np.array(states)

    def predict(self):
        """Predict step

        Reference: Table 1. from Li et al. 2009 (Left side)
        """

        # x_pt: x_t' = predicted state
        # P_pt: P_t' = predicted state covariance
        self.x_pt = cp.real(self.F @ self.x_t)
        self.P_pt = cp.real(self.F @ self.P_t @ self.F.T + self.Q)

    def _calc_weights(self, args):
        """Calculate weights

        Refer to the paper for more details. Equations 13a and 13b
        """
        # Initialize
        self.weights = cp.zeros((2*self.dim+1, 1))
        # For weight w_0 (equation 13a)
        self.weights[0] = self.args.ukf_kappa / \
            (self.dim + self.args.ukf_kappa)
        # For weight w_i (equation 13b)
        for i in range(1, 2*self.dim+1):
            self.weights[i] = 0.5 / (self.dim + self.args.ukf_kappa)

    def _augment(self, X_orig):
        """Augmentation

        Put summary here.
        Reference: Check the paragraphs "To fit B," in page 16

        Arguments:
            X_orig: Original training data position and velocities (e.g. kinematics)
                    dimension: [num_features, num_samples]

        Returns:
            X_aug:  Augmented training data
                    dimension: [(num_features+augment)*kalman_order, num_samples-kalman_order]
                    in the original paper, dim = [6n, T]
        """
        # Get number of samples
        num_samp = X_orig.shape[1]
        # Get the kalman order "n"
        n = self.args.kalman_order
        # Based on the kalman order, create X_lag
        # X_aug: State matrix with lagged information.
        # Exceptional case: kalman_order = 1. If this is the case, X and X_aug is the same.
        if n == 1:
            # Just add augmented
            aug_row = cp.linalg.norm(X_orig, axis=0)
            X_stacked = cp.vstack((X_orig, aug_row))
            X_aug = X_stacked
            return X_aug
        else:
            # For creating a stacked matrix for lagged matrix
            for i in range(n):
                # Stack the matrix in order
                if i == 0:
                    # Add norm row
                    aug_row = cp.linalg.norm(X_orig, axis=0)
                    X_orig_stacked = cp.vstack((X_orig, aug_row))
                else:
                    # Pre-defining lagged matrix with zeros
                    X_orig_lag = cp.zeros(X_orig.shape)
                    # Replace part of the above defined zero matrix with lagged matrix
                    X_orig_lag[:, :-(i+1)] = X_orig[:, i+1:]
                    # Calculate the norm and create a row to stack
                    aug_row = cp.linalg.norm(X_orig_lag, axis=0)
                    X_orig_lag_aug = cp.vstack((X_orig_lag, aug_row))
                    # Vertical concatenation of columns i-1, i-2,... ,i-n of matrix X_orig
                    X_orig_stacked = cp.vstack(
                        [X_orig_lag_aug, X_orig_stacked])

            # Now you just need to extract the first num_samp - kalman_order columns
            X_aug = X_orig_stacked[:, :num_samp - n]

            return X_aug

    def _get_sigmapoints(self):
        """A function to generate sigma points.

        Reference: Li et al. 2009 (equations 8a, 8b, 8c)

        """

        n = self.x_pt.shape[0]
        # There's no best way. But gives singularity problems need to check for the best choice.
        if self.args.ukf_decomp_type == "cholesky":
            # Deal with singularity matrix by utilizing previously saved P_pt
            try:
                root_P_pt = cp.linalg.cholesky(self.P_pt)
                self.prev_P_pt = self.P_pt
            except:
                # Reinitialize
                self.P_pt = self.prev_P_pt
                root_P_pt = cp.linalg.cholesky(self.P_pt)
        elif self.args.ukf_decomp_type == "sqrtm":
            try:
                root_P_pt = cp.real(scipy.linalg.sqrtm(self.P_pt))
                self.prev_P_pt = self.P_pt
            except:
                self.P_pt = self.prev_P_pt
                root_P_pt = cp.real(scipy.linalg.sqrtm(self.P_pt))
        # Initializations
        sigmas = cp.zeros((n, 2*n+1))
        # First sigma is just the mean itself (equation 8a)
        sigmas[:, 0] = self.x_pt[:, 0]
        mult_term = cp.sqrt(self.dim + self.args.ukf_kappa)
        # equation 8b
        sigmas[:, 1:n+1] = self.x_pt + mult_term * root_P_pt
        # equation 8c
        sigmas[:, n+1:2*n+1] = self.x_pt - mult_term * root_P_pt

        self.sigmas = sigmas

    def update(self, measure_samp):
        """Update step for UKF.

        """

        # 1) Compute Sigma Points
        self._get_sigmapoints()
        self.X_aug = self._augment(self.sigmas)
        # 2) Perform Unscented Transform
        # TODO: Below two will be included in UT
        self.Z = self.B @ self.X_aug
        self.z_pt = self.Z @ self.weights

        # Covariance calculations
        temp = self.Z[:, 0] - self.z_pt[:, 0]
        ttemp = temp.reshape(temp.shape[0], 1) @ temp.reshape(
            temp.shape[0], 1).T
        self.P_zz = self.weights[0] * ttemp + self.R
        for k in range(1, 2*self.dim+1):
            temp = self.Z[:, k] - self.z_pt[:, 0]
            ttemp = temp.reshape(temp.shape[0], 1) @ temp.reshape(
                temp.shape[0], 1).T
            self.P_zz += self.weights[k] * ttemp

        # State-observation Covariance
        temp1 = self.sigmas[:, 0] - self.x_pt[:, 0]
        temp2 = self.Z[:, 0] - self.z_pt[:, 0]
        ttemp = temp1.reshape(temp1.shape[0], 1) @ temp2.reshape(
            temp2.shape[0], 1).T
        self.P_xz = self.weights[0] * ttemp
        for k in range(1, 2*self.dim+1):
            temp1 = self.sigmas[:, k] - self.x_pt[:, 0]
            temp2 = self.Z[:, k] - self.z_pt[:, 0]
            ttemp = temp1.reshape(temp1.shape[0], 1) @ temp2.reshape(
                temp2.shape[0], 1).T
            self.P_xz += self.weights[k] * ttemp

        # Kalman Gain
        self.K_t = self.P_xz @ cp.linalg.inv(self.P_zz)

        # Make predictions
        self.x_t = self.x_pt + self.K_t @ (measure_samp - self.z_pt)
        self.P_t = self.P_pt - self.K_t @ self.P_xz.T

        # When x_t is exploding, reinitialize
        if cp.sum(self.x_t) > 1e+100:
            self.x_t = cp.zeros((self.dim, 1))

        # Check self.P_pt for positive definiteness.
        # If not, there's going to be a singularity problem.
        if cp.all(cp.linalg.eigvalsh(self.P_t) > 0):
            self.positive_definite_count += 1
        else:
            # Reinitialize
            self.P_t = self.Q

    def _state_model(self, X_orig):
        """A function to construct state model

        Put summary here.
        Reference: Check the paragraphs 1 and 2 in page 16 of
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0006243&type=printable

        Arguments:
            X_orig: Original training data position and velocities (e.g. kinematics)
                    dimension: [num_features, num_samples]

        Returns:
            X:      State matrix (e.g. kinematics)
                    dimension: [num_features, num_samples-kalman_order]
                    in the origitnal paper, dim = [4, T]
            X_lag:  State matrix with lagged information.
                    dimension: [num_features*kalman_order, num_samples-kalman_order]
                    in the original paper, dim = [4n, T]
        """

        # Get number of samples
        num_samp = X_orig.shape[1]
        # Get the kalman order "n"
        n = self.args.kalman_order
        # Based on the kalman order, create X_lag
        # X_lag: State matrix with lagged information.
        #        dimension: [num_features*kalman_order, num_samples-kalman_order]
        # Exceptional case: kalman_order = 0. If this is the case, X and X_lag is the same.
        #                   However, kalman_order shouldn't be 0.
        if n == 0:
            X = X_orig
            X_lag = X_orig
            return X, X_lag
        else:
            # For creating a stacked matrix for lagged matrix
            for i in range(n):
                # Stack the matrix in order
                if i == 0:
                    X_orig_stacked = X_orig
                else:
                    # Pre-defining lagged matrix with zeros
                    X_orig_lag = cp.zeros(X_orig.shape)
                    # Replace part of the above defined zero matrix with lagged matrix
                    X_orig_lag[:, :-(i+1)] = X_orig[:, i:-1]
                    # Vertical concatenation of columns i-1, i-2,... ,i-n of matrix X_orig
                    X_orig_stacked = cp.vstack([X_orig_lag, X_orig_stacked])
            # Now you just need to extract the first num_samp - kalman_order columns
            X_lag = X_orig_stacked[:, :num_samp - n]
            # Make sure you omit the first kalman_order columns for X as stated in the paper
            X = X_orig[:, n:]

            return X, X_lag
