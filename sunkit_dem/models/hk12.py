"""
Hannah-Kontar (2012) DEM model definition
"""
import sys

import numpy as np
import astropy.units as u
from scipy.interpolate import splrep, splev
from sunkit_dem import GenericModel

# from synthesizAR.instruments.sdo import _TEMPERATURE_RESPONSE

# Clone this repo and replace path below: https://github.com/ianan/demreg
sys.path.append("/Users/pwright/Documents/personal/sunkit-dem/demreg/python/")
from dn2dem_pos import dn2dem_pos


class HK12Model(GenericModel):
    def _model(
        self,
        alpha=1.0,
        increase_alpha=1.5,
        max_iterations=10,
        guess=None,
        use_em_loci=False,
    ):
        errors = np.array([c.uncertainty.array.squeeze() for c in self.data]).T
        dem, edem, elogt, chisq, dn_reg = dn2dem_pos(
            self.data_matrix.value.T,
            errors,
            self.kernel_matrix.value.T,
            np.log10(self.temperature_bin_centers.to(u.K).value),
            self.temperature_bin_edges.to(u.K).value,
            max_iter=max_iterations,
            reg_tweak=alpha,
            rgt_fact=increase_alpha,
            dem_norm0=guess,
            gloci=use_em_loci,
        )
        dem_unit = (
            self.data_matrix.unit
            / self.kernel_matrix.unit
            / self.temperature_bin_edges.unit
        )
        dem = dem.T * dem_unit
        uncertainty = edem.T * dem_unit
        em = dem * np.diff(self.temperature_bin_edges)
        T_error_upper = self.temperature_bin_centers * (10 ** elogt - 1)
        T_error_lower = self.temperature_bin_centers * (1 - 1 / 10 ** elogt)
        return {
            "dem": dem,
            "uncertainty": uncertainty,
            "em": em,
            "temperature_errors_upper": T_error_upper,
            "temperature_errors_lower": T_error_lower,
            "chi_squared": np.atleast_1d(chisq),
        }

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get("model") == "hk12"
