"""
Single Gaussian Model
"""

from sunkit_dem import GenericModel
import astropy.units as u
import numpy as np
from astropy.modeling.models import Gaussian1D
from scipy.optimize import minimize
import warnings

__all__ = ["SingleGaussian"]


class SingleGaussian(GenericModel):
    def _model(self, dem0=1e23 / (u.cm ** 5 * u.K), logT0=None, delta_logT=None):
        # Predefine some quantities
        delta_T_kernel = (
            np.diff(self.temperature_bin_edges) * self.kernel_matrix
        ).value
        logT_centers = np.log10(self.temperature_bin_centers.value)
        uncertainty = np.array(
            [
                c.uncertainty.array.squeeze() if c.uncertainty is not None else 1
                for c in self.data
            ]
        )
        initial_conditions = (
            dem0.value,
            logT_centers.mean() if logT0 is None else logT0,
            np.diff(logT_centers)[0] if delta_logT is None else delta_logT,
        )

        # Define chi-squared between model and data
        N_free = self.wavelength.shape[0] - 3

        def func(x):
            g = Gaussian1D(*x)
            data_model = (delta_T_kernel * g(logT_centers)).sum(axis=1)
            return np.sqrt(
                (((self.data_matrix.value - data_model) / uncertainty) ** 2).sum()
                / N_free
            )

        # Run minimization
        res = minimize(func, initial_conditions, method="Nelder-Mead",)
        if not res.success:
            warnings.warn(res.message)

        unit = (
            self.data_matrix.unit
            / self.temperature_bin_edges.unit
            / self.kernel_matrix.unit
        )

        dem = Gaussian1D(*res.x)(logT_centers) * unit

        return {
            "dem": Gaussian1D(*res.x)(logT_centers) * unit,
            "uncertainty": None,
        }

    @classmethod
    def defines_model_for(cls, *args, **kwargs):
        return kwargs.get("model") == "single_gaussian"

