from __future__ import division
import numpy as np
import logging
import math
import os
import sys
from scipy.stats import norm
from scipy.special import erf

from shrike.compliant_logging import DataCategory

if os.path.exists('dpne.zip'):
    sys.path.insert(0, 'dpne.zip')

from dpne.dpne_utils import log

class GaussianProcess:
    def __init__(self, n, epsilon, delta, eta, tokens_per_user, max_ngram_size, budget_distribute="uniform", num_valid_cur_step=0, num_extracted_prev_step=0):
        Delta_0 = tokens_per_user
        self.Delta_0 = Delta_0 # tokens_per_user
        # calculate threshold
        g_param = self.calibrate_analytic_gaussian_mechanism(epsilon=epsilon, delta=delta / 2, global_sensitivity=1, tol=1.e-12)
        # distribute g_param uniformly with g_param * sqrt(n)

        if budget_distribute == 1.0:
            # uniformly distribute budget (including unigram)
            g_param = g_param * math.sqrt(float(max_ngram_size))
        else:
            const_multiplier = math.sqrt(sum([(1.0/math.pow(budget_distribute, 2*i)) for i in range(1, max_ngram_size+1)]))
            g_param = g_param * math.pow(budget_distribute, n) * const_multiplier

        if n == 1:
            # extracting unigram
            f_g_rho = lambda t: 1.0 / math.sqrt(t) + g_param * norm.ppf((1.0 - delta / 2.0) ** (1.0 / t))
            g_rho = max([f_g_rho(t) for t in range(1, Delta_0 + 1)])
        else:
            # extracting longer n-gram (n>1)
            # calculate based on eta value and # of n-grams extracted from previous step.
            g_rho = g_param * norm.ppf(1.0 - eta * min(num_extracted_prev_step / num_valid_cur_step, 1.0))

        self.g_param = g_param
        self.g_rho = g_rho
        
        log(logging.INFO, DataCategory.PUBLIC, "Params Delta_0={0}, delta={1:.2e}, g_param={2}, g_rho={3}".format(Delta_0, delta, g_param, g_rho))


    def calibrate_analytic_gaussian_mechanism(self, epsilon, delta, global_sensitivity, tol=1.e-12):
        """Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

        Args:
            epsilon (float) : target epsilon (epsilon > 0)
            delta (float) : target delta (0 < delta < 1)
            global_sensitivity (float) : upper bound on L2 global sensitivity (GS >= 0)
            tol (float) : error tolerance for binary search (tol > 0)

        Returns:
            float: sigma (standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity global_sensitivity value)
        """
        def calculate_phi(t_val):
            return 0.5*(1.0 + erf(float(t_val)/math.sqrt(2.0)))

        def case_a(epsilon, s_val):
            return calculate_phi(math.sqrt(epsilon * s_val)) - math.exp(epsilon) * calculate_phi(-math.sqrt(epsilon * (s_val+2.0)))

        def case_b(epsilon, s_val):
            return calculate_phi(-math.sqrt(epsilon * s_val)) - math.exp(epsilon) * calculate_phi(-math.sqrt(epsilon * (s_val+2.0)))

        def doubling_trick(predicate_stop, s_inf, s_sup):
            while not predicate_stop(s_sup):
                s_inf = s_sup
                s_sup = 2.0*s_inf
            return s_inf, s_sup

        def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
            s_mid = s_inf + (s_sup - s_inf) / 2.0
            while not predicate_stop(s_mid):
                if predicate_left(s_mid):
                    s_sup = s_mid
                else:
                    s_inf = s_mid
                s_mid = s_inf + (s_sup-s_inf)/2.0
            return s_mid

        delta_thr = case_a(epsilon, 0.0)

        if delta == delta_thr:
            alpha = 1.0

        else:
            if delta > delta_thr:
                predicate_stop_dt = lambda s: case_a(epsilon, s) >= delta
                function_s_to_delta = lambda s: case_a(epsilon, s)
                predicate_left_bs = lambda s: function_s_to_delta(s) > delta
                function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)

            else:
                predicate_stop_dt = lambda s: case_b(epsilon, s) <= delta
                function_s_to_delta = lambda s: case_b(epsilon, s)
                predicate_left_bs = lambda s: function_s_to_delta(s) < delta
                function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)

            predicate_stop_bs = lambda s: abs(function_s_to_delta(s) - delta) <= tol

            s_inf, s_sup = doubling_trick(predicate_stop_dt, 0.0, 1.0)
            s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
            alpha = function_s_to_alpha(s_final)

        sigma = alpha * global_sensitivity / math.sqrt(2.0 * epsilon)
        log(logging.INFO, DataCategory.PUBLIC, "selected sigma value for Gaussian: {0}".format(sigma))
        return sigma

    def exceeds_threshold(self, val):
        nval = val + np.random.normal(0, self.g_param)
        if nval > self.g_rho:
            return True
        else:
            return False
    