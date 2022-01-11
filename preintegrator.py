import numpy as np


class ImuPreintegrator:
    def __init__(self, max_history=1000, gyro_bias=0, accel_bias=0):
        pass

    def add_measurement(self, imu_meas):
        # Add to history
        # update RMI
        # update jacobian
        pass

    def reset(self):
        """
        Clears the rolling RMIs to zero/identity.
        """
        pass

    def get_rmis(self):
        """
        Returns the RMIs and corresponding jacobian corresponding to the time
        period between this function call, and the previous function call.

        This will be really fast because we are computing the RMIs as we go.
        """
        pass

    def get_rmis_between(self, t_i, t_j, compute_jacobians=True):
        """
        Returns the RMIs and jacobians corresponding to time points t_i < t_j.

        This will be slower in general since we need to recompute from scratch
        """
        pass

    def _update_rmis(self, rmis, imu_meas):
        pass

    def update_biases(self, gyro_bias, accel_bias):
        pass
