import unittest
import xmlrunner
import pandas as pd
import numpy as np
import threading
import functools
import inspect
import math
import traceback
import warnings

from parameterized import parameterized
from scipy.io.arff import loadarff
from scipy.stats import chisquare, ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


class TestTimeoutException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# thanks to https://gist.github.com/vadimg/2902788
def timeout(duration, default=None):
    def decorator(func):
        class InterruptableThread(threading.Thread):
            def __init__(self, args, kwargs):
                threading.Thread.__init__(self)
                self.args = args
                self.kwargs = kwargs
                self.result = default
                self.daemon = True
                self.exception = None

            def run(self):
                try:
                    self.result = func(*self.args, **self.kwargs)
                except Exception as e:
                    self.exception = e

        @functools.wraps(func)
        def wrap(*args, **kwargs):
            it = InterruptableThread(args, kwargs)
            it.start()
            it.join(duration)
            if it.is_alive():
                raise TestTimeoutException('timeout after %i seconds for test %s' % (duration, func))
            if it.exception:
                raise it.exception
            return it.result
        return wrap
    return decorator

class test_SKLEARN_GaussianNB(unittest.TestCase):

    params = [("{'var_smoothing':0.000000001,}", {'var_smoothing':0.000000001,}),
              ("{'var_smoothing':0.0000001,}", {'var_smoothing':0.0000001,}),
             ]

    def assert_morphtest(self, evaluation_type, testcase_name, iteration, deviations_class, deviations_score, pval_chisquare, pval_kstest):
        if evaluation_type=='score_exact':
            self.assertEqual(deviations_score, 0)
        elif evaluation_type=='class_exact':
            self.assertEqual(deviations_class, 0)
        elif evaluation_type=='score_stat':
            self.assertTrue(pval_kstest>0.05)
        elif evaluation_type=='class_stat':
            self.assertTrue(pval_chisquare>0.05)
        else:
            raise ValueError('invalid evaluation_type: %s (allowed: score_exact, class_exact, score_stat, class_stat' % evaluation_type)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Const_RANDNUM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/RANDNUM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/RANDNUM_%i_Const.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Const_UNIFORM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNIFORM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNIFORM_%i_Const.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Const_IONOSPHERE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/IONOSPHERE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/IONOSPHERE_%i_Const.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Const_UNBALANCE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNBALANCE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNBALANCE_%i_Const.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Opposite_RANDNUM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/RANDNUM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/RANDNUM_%i_Opposite.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = abs(1-prediction_original)
            scores_expected = scores_original[:,1]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Opposite_UNIFORM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNIFORM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNIFORM_%i_Opposite.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = abs(1-prediction_original)
            scores_expected = scores_original[:,1]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Opposite_IONOSPHERE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/IONOSPHERE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/IONOSPHERE_%i_Opposite.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = abs(1-prediction_original)
            scores_expected = scores_original[:,1]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Opposite_UNBALANCE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNBALANCE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNBALANCE_%i_Opposite.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = abs(1-prediction_original)
            scores_expected = scores_original[:,1]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Scramble_RANDNUM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/RANDNUM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/RANDNUM_%i_Scramble.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Scramble_UNIFORM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNIFORM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNIFORM_%i_Scramble.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Scramble_IONOSPHERE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/IONOSPHERE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/IONOSPHERE_%i_Scramble.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Scramble_UNBALANCE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNBALANCE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNBALANCE_%i_Scramble.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Reorder_RANDNUM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/RANDNUM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/RANDNUM_%i_Reorder.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Reorder_UNIFORM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNIFORM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNIFORM_%i_Reorder.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Reorder_IONOSPHERE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/IONOSPHERE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/IONOSPHERE_%i_Reorder.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Reorder_UNBALANCE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNBALANCE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNBALANCE_%i_Reorder.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_morph_df.values, class_index_morph, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_morph_df.values, class_index_morph, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Same_RANDNUM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/RANDNUM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/RANDNUM_%i_Same.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Same_UNIFORM(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNIFORM_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNIFORM_%i_Same.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Same_IONOSPHERE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/IONOSPHERE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/IONOSPHERE_%i_Same.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Same_UNBALANCE(self, name, kwargs):
        for iter in range(1,1+1):
            data_original, meta_original = loadarff('/morphdata/UNBALANCE_%i.arff' % iter)
            data_morphed, meta_morphed = loadarff('/morphdata/UNBALANCE_%i_Same.arff' % iter)
            lb_make = LabelEncoder()
            
            data_original_df = pd.DataFrame(data_original)
            data_original_df["classAtt"] = lb_make.fit_transform(data_original_df["classAtt"])
            data_original_df = pd.get_dummies(data_original_df)
            
            data_morph_df = pd.DataFrame(data_morphed)
            data_morph_df["classAtt"] = lb_make.fit_transform(data_morph_df["classAtt"])
            data_morph_df = pd.get_dummies(data_morph_df)
            
            class_index_original = -1
            for i, s in enumerate(data_original_df.columns):
                if 'classAtt' in s:
                    class_index_original = i
            class_index_morph = -1
            for i, s in enumerate(data_morph_df.columns):
                if 'classAtt' in s:
                    class_index_morph = i
            
            classifier_original = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_original.fit(np.delete(data_original_df.values, class_index_original, axis=1),data_original_df.values[:,class_index_original])
            classifier_morph = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier_morph.fit(np.delete(data_morph_df.values, class_index_morph, axis=1),data_morph_df.values[:,class_index_morph])
            prediction_original = classifier_original.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            prediction_morph = classifier_morph.predict(np.delete(data_original_df.values, class_index_original, axis=1))
            
            if hasattr(classifier_original, 'predict_proba'):
	            scores_original = classifier_original.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
	            scores_morph = classifier_morph.predict_proba(np.delete(data_original_df.values, class_index_original, axis=1))
            else:
	            scores_original = np.array([prediction_original, abs(1-prediction_original)])
	            scores_morph = np.array([prediction_morph, abs(1-prediction_morph)])
            
            prediction_expected = prediction_original;
            scores_expected = scores_original[:,0]
            
            deviations_class = sum(prediction_morph!=prediction_expected)
            deviations_score = sum(scores_morph[:,0]!=scores_expected)
            count_expected_1 = prediction_expected.sum()
            count_expected_0 = len(prediction_expected)-count_expected_1
            count_morph_1 = prediction_morph.sum()
            count_morph_0 = len(prediction_morph)-count_morph_1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval_chisquare = chisquare([count_morph_0,count_morph_1], [count_expected_0, count_expected_1]).pvalue
                pval_kstest = ks_2samp(scores_morph[:,0], scores_expected).pvalue
            
            # handles situation if all data is in one class
            if math.isnan(pval_chisquare) and deviations_class==0:
                pval_chisquare = 1.0
            testcase_name = inspect.stack()[0][3]
            
            self.assert_morphtest('score_exact', testcase_name, iter, deviations_class, deviations_score, pval_chisquare, deviations_class)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Uniform(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/Uniform_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/Uniform_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MinFloat(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/MinFloat_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/MinFloat_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_VerySmall(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/VerySmall_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/VerySmall_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MinDouble(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/MinDouble_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/MinDouble_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MaxFloat(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/MaxFloat_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/MaxFloat_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_VeryLarge(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/VeryLarge_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/VeryLarge_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MaxDouble(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/MaxDouble_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/MaxDouble_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Split(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/Split_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/Split_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_LeftSkew(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/LeftSkew_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/LeftSkew_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_RightSkew(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/RightSkew_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/RightSkew_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_OneClass(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/OneClass_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/OneClass_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Bias(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/Bias_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/Bias_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Outlier(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/Outlier_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/Outlier_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Zeroes(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/Zeroes_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/Zeroes_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_RandomNumeric(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/RandomNumeric_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/RandomNumeric_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_DisjointNumeric(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('/smokedata/DisjointNumeric_%i_training.arff' % iter)
            testdata, testmeta = loadarff('/smokedata/DisjointNumeric_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = GaussianNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            



if __name__ == '__main__':
    unittest.main()
#    with open('results.xml', 'wb') as output:
#        unittest.main(
#            testRunner=xmlrunner.XMLTestRunner(output=output),
#            failfast=False, buffer=False, catchbreak=False)