import numpy as np
from playground.common.contact_aided_invariant_ekf.observation import Observation, check_measurement_is_empty
from typing import Optional
from playground.common.contact_aided_invariant_ekf.robot_state import RobotState
from playground.common.contact_aided_invariant_ekf.lie_group import exp_so3, exp_sek3, adjoint_sek3, skew
from playground.common.contact_aided_invariant_ekf.noise_params import NoiseParams
from playground.common.contact_aided_invariant_ekf.utils import remove_row_and_column, GRAVITY_ACCELERATION
from playground.common.contact_aided_invariant_ekf.robot_kinematics import Kinematics


class Landmark:
    def __init__(self, id_in: int, position_in: np.ndarray):
        """
        Represents a landmark with an ID and a 3D position.

        Args:
            id_in (int): The landmark ID.
            position_in (np.ndarray): A 3D numpy array representing the landmark position.
        """
        self.id = id_in
        self.position = np.array(position_in, dtype=float)  # Ensure it's a NumPy array

    def __repr__(self):
        return f"Landmark(id={self.id}, position={self.position})"


class RightInEKF:
    def __init__(self, state: Optional[RobotState] = None, noise_params: Optional[NoiseParams] = None):
        self.g_ = np.array([0, 0, GRAVITY_ACCELERATION])

        if state is None:
            state = RobotState()
        self.state_ = state

        if noise_params is None:
            noise_params = NoiseParams()
        self.noise_params_ = noise_params

        self.prior_landmarks_ = {}
        self.estimated_landmarks_ = {}
        self.estimated_contact_positions_ = {}
        self.contacts_ = {}

    def get_state(self):
        return self.state_

    def set_state(self, state):
        self.state_ = state

    def get_noise_params(self):
        return self.noise_params_

    def set_noise_params(self, params):
        self.noise_params_ = params

    def get_prior_landmarks(self):
        return self.prior_landmarks_

    def set_prior_landmarks(self, prior_landmarks):
        self.prior_landmarks_ = prior_landmarks

    def get_estimated_landmarks(self):
        return self.estimated_landmarks_

    def get_estimated_contact_positions(self):
        return self.estimated_contact_positions_

    def set_contacts(self, contacts: tuple[int, bool]):
        for contact_tuple in contacts:
            contact_id, is_contact = contact_tuple
            self.contacts_[contact_id] = is_contact

    def get_contacts(self):
        return self.contacts_


    def propagate(self, m: np.ndarray, dt: float):
        """
        A priori Invariant EKF update. Note: bias does not evolve so it is not captured in the state, however, is considered in the covariance matrix.
        """
        w = m[:3] - self.state_.get_gyroscope_bias().squeeze()  # Angular Velocity
        a = m[3:] - self.state_.get_accelerometer_bias().squeeze()  # Linear Acceleration

        X = self.state_.get_x()
        P = self.state_.get_p()

        # Extract State
        R = self.state_.get_rotation()
        v = self.state_.get_velocity()
        p = self.state_.get_position()

        # Strapdown IMU motion model
        phi = w * dt
        R_pred = R @ exp_so3(phi)
        v_pred = v + (R @ a + self.g_) * dt
        p_pred = p + v * dt + 0.5 * (R @ a + self.g_) * dt**2

        # Set new state (bias has constant dynamics)
        self.state_.set_rotation(R_pred)
        self.state_.set_velocity(v_pred)
        self.state_.set_position(p_pred)

        # ---- Linearized invariant error dynamics -----
        dimX = self.state_.dim_x()
        dimP = self.state_.dim_p()
        dimTheta = self.state_.dim_theta()

        A = np.zeros((dimP, dimP))

        # TODO: we don't need to compute inertia terms for A everytime.
        A[3:6, 0:3] = skew(self.g_)
        A[6:9, 3:6] = np.eye(3)

        # Bias terms
        A[0:3, dimP-dimTheta:dimP-dimTheta+3] = -R
        A[3:6, dimP-dimTheta+3:dimP-dimTheta+6] = -R

        # TODO: vectorize?
        for i in range(3, dimX):
            A[3*i-6:3*i-3, dimP-dimTheta:dimP-dimTheta+3] = -skew(X[0:3, i]) @ R
        # Vectorized computation for the loop
        # indices = np.arange(3, dimX)
        # A[3*indices-6, dimP-dimTheta:dimP-dimTheta+3] = -skew(X[0:3, indices]) @ R

        # Noise terms. TODO: vectorize
        Qk = np.zeros((dimP, dimP))
        Qk[0:3, 0:3] = self.noise_params_.get_gyroscope_cov()
        Qk[3:6,  3:6] = self.noise_params_.get_accelerometer_cov()

        for _, idx in self.estimated_contact_positions_.items():
            Qk[3+3*(idx-3):3+3*(idx-3)+3, 3+3*(idx-3):3+3*(idx-3)+3] = self.noise_params_.get_contact_cov()

        Qk[dimP-dimTheta:dimP-dimTheta+3, dimP-dimTheta:dimP-dimTheta+3] = self.noise_params_.get_gyroscope_bias_cov()
        Qk[dimP-dimTheta+3:dimP-dimTheta+6, dimP-dimTheta+3:dimP-dimTheta+6] = self.noise_params_.get_accelerometer_bias_cov()

        # Discretization
        identity = np.eye(dimP)
        Phi = identity + A * dt  # Fast approximation of exp(A*dt)
        Adj = identity.copy()
        Adj[:dimP-dimTheta, :dimP-dimTheta] = adjoint_sek3(X)
        PhiAdj = Phi @ Adj
        Qk_hat = PhiAdj @ np.diag(np.diag(Qk)) @ PhiAdj.T * dt

        # Propagate Covariance
        P_pred = Phi @ P @ Phi.T + Qk_hat

        # Set new covariance
        self.state_.set_p(P_pred)

    def correct(self, obs: Observation):
        if obs.is_empty():
            return
        # print("Obs: ",obs)
        P = self.state_.get_p()
        PHT = P @ obs.H.T
        S = obs.H @ PHT + obs.N
        K = PHT @ np.linalg.inv(S)

        BigX = self.state_.copy_diag_x(obs.Y.shape[0] // self.state_.dim_x())
        Z = BigX @ obs.Y - obs.b
        delta = K @ obs.PI @ Z

        dX = exp_sek3(delta[:-self.state_.dim_theta()])
        dTheta = delta[-self.state_.dim_theta():]

        X_new = dX @ self.state_.get_x()
        Theta_new = self.state_.get_theta().squeeze() + dTheta

        self.state_.set_x(X_new)
        self.state_.set_theta(Theta_new)

        IKH = np.eye(self.state_.dim_p()) - K @ obs.H
        P_new = IKH @ P @ IKH.T + K @ obs.N @ K.T
        self.state_.set_p(P_new)

    def correct_landmarks(self, measured_landmarks: list[Landmark], current_rotation: np.ndarray):
        Y = np.array([])
        b = np.array([])
        H = np.zeros((0, self.state_.dim_p()))
        N = np.zeros((0, 0))
        PI = np.zeros((0, 0))

        new_landmarks = []
        used_landmark_ids = set()

        for landmark in measured_landmarks:
            if landmark.id in used_landmark_ids:
                # print("Duplicate landmark ID detected! Skipping measurement.")
                continue
            used_landmark_ids.add(landmark.id)
            
            if landmark.id in self.prior_landmarks_:
                p_wl = self.prior_landmarks_[landmark.id]
                dim_x = self.state_.dim_x()
                dim_p = self.state_.dim_p()
                
                Y = np.append(Y, np.zeros(dim_x))
                Y[-dim_x+3:-dim_x+6] = landmark.position
                Y[-dim_x+4] = 1
                
                b = np.append(b, np.zeros(dim_x))
                b[-dim_x+3:-dim_x+6] = p_wl
                b[-dim_x+4] = 1
                
                new_H = np.zeros((3, dim_p))
                new_H[:, :3] = self.skew(p_wl)
                new_H[:, 6:9] = -np.eye(3)
                H = np.vstack((H, new_H))
                
                new_N = np.zeros((N.shape[0] + 3, N.shape[1] + 3))
                new_N[:N.shape[0], :N.shape[1]] = N
                new_N[-3:, -3:] = current_rotation @ self.noise_params_.get_landmark_cov() @ current_rotation.T
                N = new_N
                
                new_PI = np.zeros((PI.shape[0] + 3, PI.shape[1] + dim_x))
                new_PI[:PI.shape[0], :PI.shape[1]] = PI
                new_PI[-3:, -dim_x:] = np.zeros((3, dim_x))
                new_PI[-3:, -3:] = np.eye(3)
                PI = new_PI
                
            elif landmark.id in self.estimated_landmarks:
                dim_x = self.state_.dim_x()
                dim_p = self.state_.dim_p()
                idx = self.estimated_landmarks[landmark.id]
                
                Y = np.append(Y, np.zeros(dim_x))
                Y[-dim_x+3:-dim_x+6] = landmark.position
                Y[-dim_x+4] = 1
                Y[-dim_x+idx] = -1
                
                b = np.append(b, np.zeros(dim_x))
                b[-dim_x+4] = 1
                b[-dim_x+idx] = -1
                
                new_H = np.zeros((3, dim_p))
                new_H[:, 6:9] = -np.eye(3)
                new_H[:, 3*idx-6:3*idx-3] = np.eye(3)
                H = np.vstack((H, new_H))
                
                new_N = np.zeros((N.shape[0] + 3, N.shape[1] + 3))
                new_N[:N.shape[0], :N.shape[1]] = N
                new_N[-3:, -3:] = current_rotation @ self.noise_params_.get_landmark_cov() @ current_rotation.T
                N = new_N
                
                new_PI = np.zeros((PI.shape[0] + 3, PI.shape[1] + dim_x))
                new_PI[:PI.shape[0], :PI.shape[1]] = PI
                new_PI[-3:, -dim_x:] = np.zeros((3, dim_x))
                new_PI[-3:, -3:] = np.eye(3)
                PI = new_PI

            else:
                new_landmarks.append(landmark)
                
        obs = Observation(Y, b, H, N, PI)
        if not obs.is_empty():
            self.correct(obs)
        
        if new_landmarks:
            X_aug = self.state_.get_x()
            P_aug = self.state_.get_p()
            p = self.state_.get_position()
            
            for landmark in new_landmarks:
                start_idx = X_aug.shape[0]
                X_aug = np.pad(X_aug, ((0, 1), (0, 1)), mode='constant')
                X_aug[start_idx, start_idx] = 1
                X_aug[:3, start_idx] = p + current_rotation @ landmark.position
                
                F = np.zeros((self.state_.dim_p() + 3, self.state_.dim_p()))
                F[:self.state_.dim_p() - self.state_.dim_theta(), :self.state_.dim_p() - self.state_.dim_theta()] = np.eye(self.state_.dim_p() - self.state_.dim_theta())
                F[self.state_.dim_p() - self.state_.dim_theta():self.state_.dim_p() - self.state_.dim_theta() + 3, 6:9] = np.eye(3)
                F[self.state_.dim_p() - self.state_.dim_theta() + 3:, self.state_.dim_p() - self.state_.dim_theta():] = np.eye(self.state_.dim_theta())
                
                G = np.zeros((F.shape[0], 3))
                G[-self.state_.dim_theta()-3:, :] = current_rotation
                P_aug = (F @ P_aug @ F.T + G @ self.noise_params_.get_landmark_cov() @ G.T)
                
                self.state_.set_x(X_aug)
                self.state_.set_p(P_aug)
                self.estimated_landmarks_[landmark.id] = start_idx
    
    def correct_kinematics(self, measured_kinematics: list[Kinematics]):
        remove_contacts = []
        new_contacts: list[Kinematics] = []
        used_contact_ids = set()
        # TODO: find out if we want R to actually be held constant and not utilized for other updates.
        R = self.state_.get_rotation()

        for kinematic in measured_kinematics:
            contact_id = kinematic.id

            # Detect and skip if an ID is not unique
            if contact_id in used_contact_ids:
                # print("Duplicate contact ID detected! Skipping measurement.")
                continue
            else:
                used_contact_ids.add(contact_id)

            # Find contact indicator for the kinematics measurement
            if contact_id not in self.contacts_:
                continue  # Skip if contact state is unknown
            contact_indicated = self.contacts_[contact_id]

            # Find the estimated contact position
            found = contact_id in self.estimated_contact_positions_
            # print(f"Contact indicated: {contact_indicated}, found: {found}")

            # If contact is not indicated and id is found in estimated_contacts_, then remove state
            if not contact_indicated and found:
                remove_contacts.append((contact_id, self.estimated_contact_positions_[contact_id]))
            # If contact is indicated and id is not found in estimated_contacts_, then augment state
            elif contact_indicated and not found:
                new_contacts.append(kinematic)
            # If contact is indicated and id is found in estimated_contacts_, then correct using kinematics
            elif contact_indicated and found:
                obs = self._create_observation_from_kinematics(kinematics=kinematic, current_rotation=R)
                if not obs.is_empty():
                    self.correct(obs)

        # Remove contacts from state
        self._remove_contacts_from_state(remove_contacts)

        # Add contacts to state
        self._add_contacts_to_state(new_contacts, current_rotation=R)

    def _create_observation_from_kinematics(
            self,
            kinematics: Kinematics,
            current_rotation: np.ndarray) -> Observation:
        dim_x = self.state_.dim_x()
        dim_p = self.state_.dim_p()

        Y = np.zeros(dim_x)
        b = np.zeros(dim_x)
        H = np.zeros((3, dim_p))
        N = np.zeros((3, 3))
        PI = np.zeros((3, dim_x))

        # If empty, we will just return an empty observation.
        if check_measurement_is_empty(Y=Y):
            return Observation(Y, b, H, N, PI)

        # Modify Y
        Y[0:3] = kinematics.pose[0:3, 3]  # p_bc
        Y[4] = 1
        Y[self.estimated_contact_positions_[kinematics.id]] = -1

        # Modify b
        b[0:3] = np.zeros(3)
        b[4] = 1
        b[self.estimated_contact_positions_[kinematics.id]] = -1

        # Modify H
        H[0:3, 6:9] = -np.eye(3)
        H[0:3, 3 * self.estimated_contact_positions_[kinematics.id] - 6:3 * self.estimated_contact_positions_[kinematics.id] - 3] = np.eye(3)

        # Modify N
        N[0:3, 0:3] = current_rotation @ kinematics.covariance[3:6, 3:6] @ current_rotation.T

        # Modify PI
        PI[0:3, 0:3] = np.eye(3)

        # Create Observation object (if defined)
        obs = Observation(Y, b, H, N, PI)
        return obs

    def _remove_contacts_from_state(self, remove_contacts: list[tuple]):
        if remove_contacts:
            X_rem = self.state_.get_x()
            P_rem = self.state_.get_p()
            for contact in sorted(remove_contacts, key=lambda x: x[1], reverse=True):  
                # Sort in descending order to avoid index shift issues
                self.estimated_contact_positions_.pop(contact[0], None)

                # Remove row and column from X
                X_rem = remove_row_and_column(X_rem, contact[1])

                # Remove 3 rows and columns from P
                start_index = 3 + 3 * (contact[1] - 3)
                for _ in range(3):
                    P_rem = remove_row_and_column(P_rem, start_index)

                # Update indices in estimated landmarks and contact positions
                self.estimated_landmarks_ = {
                    k: (v - 1 if v > contact[1] else v)
                    for k, v in self.estimated_landmarks_.items()
                }
                self.estimated_contact_positions_ = {
                    k: (v - 1 if v > contact[1] else v)
                    for k, v in self.estimated_contact_positions_.items()
                }

            # Update state
            self.state_.set_x(X_rem)
            self.state_.set_p(P_rem)
    
    def _add_contacts_to_state(self, new_contacts: list[Kinematics], current_rotation: np.ndarray):
        # Augment state with newly detected contacts
        if new_contacts:
            X_aug = self.state_.get_x()
            P_aug = self.state_.get_p()
            p = self.state_.get_position()
            for contact in new_contacts:
                # Initialize new landmark mean
                start_index = X_aug.shape[0]
                
                # Resize the matrix by adding an additional row and column
                X_aug = np.pad(X_aug, ((0, 1), (0, 1)), mode='constant', constant_values=0)
                
                # Set the new diagonal entry to 1
                X_aug[start_index, start_index] = 1
                
                # Update the new column with the transformed position
                X_aug[:3, start_index] = p + current_rotation @ contact.pose[:3, 3]
                # print("X_aug: ", X_aug)

                # Initialize new landmark covariance
                F = np.zeros((self.state_.dim_p() + 3, self.state_.dim_p()))

                # Block assignment
                F[:self.state_.dim_p() - self.state_.dim_theta(), :self.state_.dim_p()  - self.state_.dim_theta()] = np.eye(self.state_.dim_p()  - self.state_.dim_theta())  # for old X
                F[self.state_.dim_p() - self.state_.dim_theta():self.state_.dim_p() - self.state_.dim_theta() + 3, 6:9] = np.eye(3)  # for new landmark
                F[self.state_.dim_p() - self.state_.dim_theta() + 3:, self.state_.dim_p() - self.state_.dim_theta():] = np.eye(self.state_.dim_theta())  # for theta
                # print("F: ", F)

                G = np.zeros((F.shape[0], 3))
                G[-self.state_.dim_theta() - 3:-self.state_.dim_theta(), :] = current_rotation
                # print("G: ", G)
                P_aug = F @ P_aug @ F.T + G @ contact.covariance[3:6, 3:6] @ G.T

                # print("P_aug: ", P_aug)
                # Update state and covariance
                self.state_.set_x(X_aug)
                self.state_.set_p(P_aug)

                # Add to list of estimated contact positions
                self.estimated_contact_positions_[contact.id] = start_index
