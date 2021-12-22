import unittest
import atoml_cmp.evaluation
import pandas as pd
import numpy as np


class ConfusionMatrixPositiveTest(unittest.TestCase):
    def test_behavior_on_identical(self):
        input1 = pd.Series([1, 0, 1, 1, 0, 0, 1, 1])
        input2 = pd.Series([1, 0, 1, 1, 0, 0, 1, 1])
        equal_flag, confusion_matrix = atoml_cmp.evaluation.create_confusion_matrix(input1, input2, False)
        self.assertEqual(equal_flag, True)
        np.testing.assert_equal(confusion_matrix, [[3, 0], [0, 5]])

    def test_behavior_on_different(self):
        input1 = pd.Series([1, 0, 1, 1, 0, 0, 1, 1])
        input2 = pd.Series([1, 0, 1, 1, 1, 0, 1, 1])
        equal_flag, confusion_matrix = atoml_cmp.evaluation.create_confusion_matrix(input1, input2, False)
        self.assertEqual(equal_flag, False)
        tp, tn, fp, fn = 2, 5, 0, 1
        np.testing.assert_equal(confusion_matrix, [[tp, fn], [fp, tn]])

    def test_all_zero(self):
        input1 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        input2 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        equal_flag, confusion_matrix = atoml_cmp.evaluation.create_confusion_matrix(input1, input2, False)
        self.assertEqual(equal_flag, True)
        tp, tn, fp, fn = 8, 0, 0, 0
        np.testing.assert_equal(confusion_matrix, [[tp, fn], [fp, tn]])

    def test_all_ones(self):
        input1 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        input2 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        equal_flag, confusion_matrix = atoml_cmp.evaluation.create_confusion_matrix(input1, input2, False)
        self.assertEqual(equal_flag, True)
        tp, tn, fp, fn = 0, 8, 0, 0
        np.testing.assert_equal(confusion_matrix, [[tp, fn], [fp, tn]])

    def test_all_different_mixed(self):
        input1 = pd.Series([1, 0, 1, 1, 0, 1, 0, 1])
        input2 = pd.Series([0, 1, 0, 0, 1, 0, 1, 0])
        equal_flag, confusion_matrix = atoml_cmp.evaluation.create_confusion_matrix(input1, input2, False)
        self.assertEqual(equal_flag, False)
        tp, tn, fp, fn = 0, 0, 5, 3
        np.testing.assert_equal(confusion_matrix, [[tp, fn], [fp, tn]])

    def test_all_different_same(self):
        input1 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        input2 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        equal_flag, confusion_matrix = atoml_cmp.evaluation.create_confusion_matrix(input1, input2, False)
        self.assertEqual(equal_flag, False)
        tp, tn, fp, fn = 0, 0, 8, 0
        np.testing.assert_equal(confusion_matrix, [[tp, fn], [fp, tn]])

    def test_print(self):
        input1 = pd.Series([1, 0, 1, 1, 0, 0, 1, 1])
        input2 = pd.Series([1, 0, 1, 1, 0, 0, 1, 1])
        print("\n### Test print output for 'create_confusion_matrix':")
        equal_flag, confusion_matrix = atoml_cmp.evaluation.create_confusion_matrix(input1, input2, True)
        self.assertEqual(equal_flag, True)
        np.testing.assert_equal(confusion_matrix, [[3, 0], [0, 5]])


class Chi2Test(unittest.TestCase):
    def test_equal_inputs(self):
        input1 = pd.Series([1, 0, 1, 1, 0, 0, 1, 1])
        input2 = input1.copy(deep=True)
        p = atoml_cmp.evaluation.chi2_statistic(input1, input2, print_all=False)
        self.assertEqual(p, 1.0)

    def test_all_zero(self):
        input1 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        input2 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        p = atoml_cmp.evaluation.chi2_statistic(input1, input2, print_all=False)
        self.assertEqual(p, 1.0)

    def test_all_zero_different_types(self):
        input1 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        input2 = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p = atoml_cmp.evaluation.chi2_statistic(input1, input2, print_all=False)
        self.assertEqual(p, 1.0)

    def test_all_one(self):
        input1 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        input2 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        p = atoml_cmp.evaluation.chi2_statistic(input1, input2, print_all=False)
        self.assertEqual(p, 1.0)

    def test_all_one_different_types(self):
        input1 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        input2 = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        p = atoml_cmp.evaluation.chi2_statistic(input1, input2, print_all=False)
        self.assertEqual(p, 1.0)

    def test_different_and_constant(self):
        input1 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
        input2 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
        p = atoml_cmp.evaluation.chi2_statistic(input1, input2, print_all=False)
        self.assertAlmostEqual(p, 0.0, 2)

    def mediocre_fit(self):
        input1 = pd.Series([1, 0, 0, 0, 0, 0, 1, 1])
        input2 = pd.Series([0, 1, 0, 0, 1, 1, 0, 0])
        p = atoml_cmp.evaluation.chi2_statistic(input1, input2, print_all=False)
        self.assertNotEqual(p, 0.0)
        self.assertNotEqual(p, 1.0)
