import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def sklearn_GaussianNB(X_train, y_train, X_test):
	# default parameter:
	# priors=None, var_smoothing=1e-09
	clf = GaussianNB(priors=None, var_smoothing=1e-9)
	#clf = GaussianNB(priors=None, var_smoothing=1e-9)
	y_pred = clf.fit(X_train, y_train).predict(X_test)
	return y_pred

def sklearn_MultinomialNB(X_train, y_train, X_test):
	# default parameter:
	# alpha=1.0, fit_prior=True, class_prior=None
	clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
	y_pred = clf.fit(X_train, y_train).predict(X_test)
	return y_pred

def sklearn_LogisticRegression(X_train, y_train, X_test):
	# default parameter:
	# (penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
	# intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
	# max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
	# l1_ratio=None)

	# parameter fit to spark:
	# (penalty='none', dual=False, tol=1e-6, C=10.0, fit_intercept=True, 
	# intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
	# max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
	# l1_ratio=None)

	clf = LogisticRegression(penalty='none', 
							dual=False, 
							tol=1e-6, 
							C=10.0, 
							fit_intercept=True, 
							intercept_scaling=1, 
							class_weight=None, 
							random_state=None, # not relevant 
							solver='lbfgs', 
							max_iter=100, 
							multi_class='auto', 
							verbose=0, 
							warm_start=False, 
							n_jobs=None, 
							l1_ratio=None
							)

	y_pred = clf.fit(X_train, y_train).predict(X_test)
	return y_pred

def sklearn_LinearSVM(X_train, y_train, X_test):
	# default parameter:
	# penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, 
	# multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, 
	# verbose=0, random_state=None, max_iter=1000

	clf = LinearSVC(loss='hinge', max_iter=100, C=1.0, tol=1e-6)
	y_pred = clf.fit(X_train, y_train).predict(X_test)
	return y_pred