import numpy as np


class NoiseParams:
    def __init__(
        self,
        gyroscope_nosie: float = 0.01,
        accelerometer_noise: float = 0.1,
        gyroscope_bias_noise: float = 0.00001,
        accelerometer_bias_noise: float = 0.0001,
        landmark_noise: float = 0.1,
        contact_noise: float = 0.1,
    ):
        # Default Constructor
        self.set_gyroscope_noise(gyroscope_nosie)
        self.set_accelerometer_noise(accelerometer_noise)
        self.set_gyroscope_bias_noise(gyroscope_bias_noise)
        self.set_accelerometer_bias_noise(accelerometer_bias_noise)
        self.set_landmark_noise(landmark_noise)
        self.set_contact_noise(contact_noise)

    def set_gyroscope_noise(self, std):
        self.Qg = std * std * np.identity(3)

    def set_gyroscope_noise_vector(self, std):
        self.Qg = np.array([[std[0] * std[0], 0, 0],
                            [0, std[1] * std[1], 0],
                            [0, 0, std[2] * std[2]]])

    def set_gyroscope_noise_matrix(self, cov):
        self.Qg = cov

    def set_accelerometer_noise(self, std):
        self.Qa = std * std * np.identity(3)

    def set_accelerometer_noise_vector(self, std):
        self.Qa = np.array([[std[0] * std[0], 0, 0],
                            [0, std[1] * std[1], 0],
                            [0, 0, std[2] * std[2]]])

    def set_accelerometer_noise_matrix(self, cov):
        self.Qa = cov

    def set_gyroscope_bias_noise(self, std):
        self.Qbg = std * std * np.identity(3)

    def set_gyroscope_bias_noise_vector(self, std):
        self.Qbg = np.array([[std[0] * std[0], 0, 0],
                             [0, std[1] * std[1], 0],
                             [0, 0, std[2] * std[2]]])

    def set_gyroscope_bias_noise_matrix(self, cov):
        self.Qbg = cov

    def set_accelerometer_bias_noise(self, std):
        self.Qba = std * std * np.identity(3)

    def set_accelerometer_bias_noise_vector(self, std):
        self.Qba = np.array([[std[0] * std[0], 0, 0],
                             [0, std[1] * std[1], 0],
                             [0, 0, std[2] * std[2]]])

    def set_accelerometer_bias_noise_matrix(self, cov):
        self.Qba = cov

    def set_landmark_noise(self, std):
        self.Ql = std * std * np.identity(3)

    def set_landmark_noise_vector(self, std):
        self.Ql = np.array([[std[0] * std[0], 0, 0],
                            [0, std[1] * std[1], 0],
                            [0, 0, std[2] * std[2]]])

    def set_landmark_noise_matrix(self, cov):
        self.Ql = cov

    def set_contact_noise(self, std):
        self.Qc = std * std * np.identity(3)

    def set_contact_noise_vector(self, std):
        self.Qc = np.array([[std[0] * std[0], 0, 0],
                            [0, std[1] * std[1], 0],
                            [0, 0, std[2] * std[2]]])

    def set_contact_noise_matrix(self, cov):
        self.Qc = cov

    def get_gyroscope_cov(self):
        return self.Qg

    def get_accelerometer_cov(self):
        return self.Qa

    def get_gyroscope_bias_cov(self):
        return self.Qbg

    def get_accelerometer_bias_cov(self):
        return self.Qba

    def get_landmark_cov(self):
        return self.Ql

    def get_contact_cov(self):
        return self.Qc

    def __str__(self):
        return (f"--------- Noise Params -------------\n"
                f"Gyroscope Covariance:\n{self.Qg}\n"
                f"Accelerometer Covariance:\n{self.Qa}\n"
                f"Gyroscope Bias Covariance:\n{self.Qbg}\n"
                f"Accelerometer Bias Covariance:\n{self.Qba}\n"
                f"Landmark Covariance:\n{self.Ql}\n"
                f"Contact Covariance:\n{self.Qc}\n"
                f"-----------------------------------")
