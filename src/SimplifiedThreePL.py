"""
SimplifiedThreePL.py

This file contains the implementation of the SimplifiedThreePL class.
- The Experiment.py and SignalDetection.py files are provided by the professor.
- This SimplifiedThreePL implementation was generated with assistance from ChatGPT o3-mini.

Author: Aiden Hai
Date: 03/04/2025
"""


import numpy as np
from scipy.optimize import minimize
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection


class SimplifiedThreePL:

    def __init__(self, experiment: Experiment):
        if not isinstance(experiment, Experiment):
            raise TypeError("Must provide an Experiment object.")
        
        # Save reference
        self._experiment = experiment
        
        # The assignment says we have 5 conditions with known difficulties:
        self._difficulties = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        
        # Extract correct/incorrect counts from each SignalDetection, expect exactly 5 conditions.
        if len(self._experiment.conditions) != 5:
            raise ValueError("Experiment must have exactly 5 conditions for this assignment.")
        
        correct_counts = []
        incorrect_counts = []
        
        for sdt in self._experiment.conditions:
            if not isinstance(sdt, SignalDetection):
                raise TypeError("Each condition must be a SignalDetection object.")
            correct_counts.append(sdt.n_correct_responses())
            incorrect_counts.append(sdt.n_incorrect_responses())
        
        self._correct_array = np.array(correct_counts, dtype=float)
        self._incorrect_array = np.array(incorrect_counts, dtype=float)
        
        # Private attributes
        self._base_rate = None         # c in [0,1]
        self._logit_base_rate = None   # log(c/(1-c)) in (-∞, +∞)
        self._discrimination = None    # a
        self._is_fitted = False        # not fitted yet

    def summary(self):
        """
        Returns a dictionary:
          - n_total: total number of trials
          - n_correct: total number of correct trials
          - n_incorrect: total number of incorrect trials
          - n_conditions: number of conditions (should be 5)
        """
        total_correct = np.sum(self._correct_array)
        total_incorrect = np.sum(self._incorrect_array)
        total_trials = total_correct + total_incorrect
        
        return {
            "n_total": int(total_trials),
            "n_correct": int(total_correct),
            "n_incorrect": int(total_incorrect),
            "n_conditions": len(self._correct_array)
        }
    
    def _logit_to_base_rate(self, q):
        """
        Convert logit q -> c in [0,1]. c = 1 / (1 + exp(-q))
        """
        return 1.0 / (1.0 + np.exp(-q))
    
    def _base_rate_to_logit(self, c):
        """
        Convert c in (0,1) -> logit q in (-∞, +∞). q = log(c / (1-c))
        """
        # Here we might want to clamp c to avoid log(0)
        eps = 1e-12
        c = np.clip(c, eps, 1 - eps)
        return np.log(c / (1 - c))

    def predict(self, parameters):
        """
        Given parameters (a, q), where
          a = discrimination
          q = logit_base_rate
        Return an array of predicted probabilities for each of the 5 conditions.
        
        The formula: p_i = c + (1-c) / [1 + exp(a * b_i)]
        but we store c = 1/(1+exp(-q)).
        """
        a, q = parameters
        c = self._logit_to_base_rate(q)
        
        # difficulties: b_i = [2,1,0,-1,-2]
        # p_i = c + (1-c) / (1 + exp(a*b_i))
        exponent = np.exp(a * self._difficulties)
        p = c + (1.0 - c) / (1.0 + exponent)
        
        return p

    def negative_log_likelihood(self, parameters):
        """
        NLL = - sum( correct_i * log(p_i) + incorrect_i * log(1 - p_i) ), using p_i from predict().
        """
        p = self.predict(parameters)
        # Avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        
        correct_term = self._correct_array * np.log(p)
        incorrect_term = self._incorrect_array * np.log(1.0 - p)
        nll = -np.sum(correct_term + incorrect_term)
        return nll
    
    def fit(self):
        # Initial guesses
        init_a = 1.0
        init_c = 0.2  # c must be in (0,1)
        init_q = self._base_rate_to_logit(init_c)  # logit transform

        # Define the objective in terms of (a, q).
        def objective(params):
            return self.negative_log_likelihood(params)
        
        # No direct bounding needed, since a in (-∞, +∞), q in (-∞, +∞)
        # We can use method='BFGS' or 'L-BFGS-B' (with no bounds).
        init_params = [init_a, init_q]
        result = minimize(objective, init_params, method='L-BFGS-B')
        
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        
        best_a, best_q = result.x
        self._discrimination = best_a
        self._logit_base_rate = best_q
        # Convert back to c
        self._base_rate = self._logit_to_base_rate(best_q)
        
        self._is_fitted = True

    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._base_rate
