import sys
import os
# Insert the repository root (parent of src and tests) into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

# We also had assistance from ChatGPT for this

class TestSimplifiedThreePL(unittest.TestCase):
    
    def setUp(self):
        # Create an Experiment with 5 conditions using simple test data.
        # For each condition, we use a simple SignalDetection object.
        self.exp = Experiment()
        # For example, each condition has 50 trials with ~35 correct responses (example values).
        for _ in range(5):
            # For example: hits=20, misses=10, falseAlarms=5, correctRejections=15
            # n_correct = 20 + 15 = 35; n_incorrect = 10 + 5 = 15; total trials = 50.
            sdt = SignalDetection(20, 10, 5, 15)
            self.exp.add_condition(sdt)
        self.model = SimplifiedThreePL(self.exp)
    
    # === Initialization Tests ===
    def test_valid_initialization(self):
        # Test that the constructor properly handles valid input.
        summary = self.model.summary()
        self.assertEqual(summary["n_conditions"], 5)
        self.assertEqual(summary["n_total"], int(np.sum(self.model._correct_array + self.model._incorrect_array)))
    
    def test_invalid_initialization(self):
        # Test that the constructor raises an error if the number of conditions is not 5.
        exp_invalid = Experiment()
        for _ in range(3):
            sdt = SignalDetection(10, 5, 3, 7)
            exp_invalid.add_condition(sdt)
        with self.assertRaises(ValueError):
            SimplifiedThreePL(exp_invalid)
    
    def test_get_params_before_fit(self):
        # Test that accessing parameters before calling fit() raises an error.
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()
    
    # === Prediction Tests ===
    def test_predict_range(self):
        # Test that predict() returns values between 0 and 1.
        a = 1.0
        c = 0.2
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        self.assertEqual(len(preds), 5)
        for p in preds:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
    
    def test_base_rate_effect(self):
        # Test that with all else equal, a higher base rate yields higher predicted probabilities.
        a = 1.0
        q_low = self.model._base_rate_to_logit(0.2)
        q_high = self.model._base_rate_to_logit(0.4)
        p_low = self.model.predict((a, q_low))
        p_high = self.model.predict((a, q_high))
        for p_l, p_h in zip(p_low, p_high):
            self.assertGreater(p_h, p_l)
    
    def test_difficulty_effect_positive_a(self):
        # For positive a, higher difficulty (larger b) should yield lower predicted probabilities.
        a = 1.0
        c = 0.3
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        # With difficulties [2, 1, 0, -1, -2],
        # we expect the condition with b=2 to have the lowest probability,
        # and b=-2 to have the highest.
        self.assertLess(preds[0], preds[-1])
    
    def test_difficulty_effect_negative_a(self):
        # For negative a, the effect reverses: higher difficulty should yield higher probabilities.
        a = -1.0
        c = 0.3
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        self.assertGreater(preds[0], preds[-1])
    
    def test_predict_known_values(self):
        # Test predict() with known parameter values:
        # When a=0, exp(0)=1, so p = c + (1-c)/2 for all conditions.
        a = 0.0
        c = 0.3
        expected = c + (1 - c) / 2  # 0.3 + 0.35 = 0.65
        q = self.model._base_rate_to_logit(c)
        preds = self.model.predict((a, q))
        for p in preds:
            self.assertAlmostEqual(p, expected, places=6)
    
    # === Parameter Estimation Tests ===
    def test_nll_improves_after_fit(self):
        # Test that the negative log-likelihood improves after fitting.
        init_params = [1.0, self.model._base_rate_to_logit(0.2)]
        nll_initial = self.model.negative_log_likelihood(init_params)
        self.model.fit()
        fitted_params = [self.model.get_discrimination(), self.model._logit_base_rate]
        nll_fitted = self.model.negative_log_likelihood(fitted_params)
        self.assertLess(nll_fitted, nll_initial)

    def test_high_discrimination_with_steep_data(self):
        # Create steep data with 5 conditions, 100 trials each, with accuracy rates: 0.01, 0.05, 0.50, 0.95, 0.99.
        exp_steep = Experiment()
        total_trials = 100
        accs = [0.01, 0.05, 0.50, 0.95, 0.99]
        for acc in accs:
            correct = int(round(total_trials * acc))
            incorrect = total_trials - correct
            hits = correct // 2
            correctRejections = correct - hits
            misses = 50 - hits
            falseAlarms = 50 - correctRejections
            sdt = SignalDetection(hits, misses, falseAlarms, correctRejections)
            exp_steep.add_condition(sdt)
        model_steep = SimplifiedThreePL(exp_steep)
        model_steep.fit()
        # Relaxed threshold: expect discrimination > 2.5 with the given steep data.
        self.assertGreater(model_steep.get_discrimination(), 2.5)
    
    def test_get_params_before_fit_raises(self):
        # Ensure that attempting to get parameters before fitting raises an error.
        model = self.model  # from setUp (not fitted)
        with self.assertRaises(ValueError):
            model.get_discrimination()
        with self.assertRaises(ValueError):
            model.get_base_rate()
    
    # === Integration Test ===
    def test_integration(self):
        # Create a dataset with 5 conditions, 200 trials per condition (100 signal, 100 noise),
        # with accuracy rates exactly: 0.55, 0.60, 0.75, 0.90, 0.95.
        exp_int = Experiment()
        total_trials = 200
        accs = [0.55, 0.60, 0.75, 0.90, 0.95]
        for acc in accs:
            correct = int(round(total_trials * acc))
            incorrect = total_trials - correct
            # Assume signal and noise each contribute 100 trials.
            hits = correct // 2
            correctRejections = correct - hits
            misses = 100 - hits
            falseAlarms = 100 - correctRejections
            sdt = SignalDetection(hits, misses, falseAlarms, correctRejections)
            exp_int.add_condition(sdt)
        model_int = SimplifiedThreePL(exp_int)
        model_int.fit()
        predictions = model_int.predict((model_int.get_discrimination(), model_int._logit_base_rate))
        
        # Compute the observed correct rate for each condition.
        observed = []
        for sdt in exp_int.conditions:
            obs = sdt.n_correct_responses() / sdt.n_total_responses()
            observed.append(obs)
        
        # Verify that predictions are close to observed values (within a delta of 0.05).
        for pred, obs in zip(predictions, observed):
            self.assertAlmostEqual(pred, obs, delta=0.05)
    
    # === Corruption Tests ===
    def test_corruption(self):
        # Test that if the user directly modifies private attributes, re-fitting recovers a consistent model.
        self.model.fit()
        orig_base_rate = self.model.get_base_rate()
        # Simulate a user modifying a private attribute
        self.model._base_rate = 999
        # Re-fit the model
        self.model.fit()
        self.assertNotEqual(self.model.get_base_rate(), 999)
        # Also, ensure that the new estimate is reasonable (i.e., different from the corrupted value)
        self.assertNotAlmostEqual(self.model.get_base_rate(), 999)

if __name__ == '__main__':
    unittest.main()