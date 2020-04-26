import sys, os, time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler

DATA_LOCATION = './data/'
MIN_NON_NAN_RATIO = .1


class data_loader():

	def __init__(self):


		self.x_train = pd.read_csv(DATA_LOCATION + 'train_features.csv')
		self.x_test = pd.read_csv(DATA_LOCATION + 'test_features.csv')
		self.y_train = pd.read_csv(DATA_LOCATION + 'train_labels.csv')

		# self.x_train = self.x_train.iloc[:1200,:]
		# self.x_test = self.x_test.iloc[:1200,:]
		# self.y_train = self.y_train.iloc[:100,:]

		self.data_keys = self.x_train.keys()

		self.remove_cols_without_data()

		self.useable_feature_keys = self.analyse_features()

		self.transform_features()

		self.normalise_data()

		print('\nloaded arrays:\nx_train: [', self.x_train.shape, ']\n', self.x_train)
		print('\nx_test: [', self.x_test.shape, ']\n', self.x_test)
		print('\nx_train: [', self.y_train.shape, ']\n', self.y_train)
		print('\n***\n')

		
	def remove_cols_without_data(self):
				
		no_value_cols = np.asarray(np.where(self.x_train.apply(pd.Series.count) == 0))
		no_value_col_names = list(self.data_keys[no_value_cols])
		if no_value_col_names == []:
			print('\nno cols removed... all contain data\n')
		else:
			print('\nremoving cols without values: ', no_value_col_names, '\n')
			self.x_train = self.x_train.drop(columns = no_value_col_names)
			self.x_test = self.x_test.drop(columns = no_value_col_names)

	def analyse_features(self):

		nonNaN_ratios = self.x_train.apply(pd.Series.count) / (len(self.x_train))
		useable_features = nonNaN_ratios[nonNaN_ratios >= MIN_NON_NAN_RATIO]
		useable_feature_keys = useable_features.keys()

		self.x_train = self.x_train[useable_feature_keys]
		self.x_test = self.x_test[useable_feature_keys]

		print('\nnumber of non-NaNs for each col [percentage of total]:')
		print(np.round(100. * nonNaN_ratios, 1))
		print('\nuseable_feature_keys:')
		print(useable_feature_keys, '\n\n')

		return useable_feature_keys

	def transform_features(self):

		def transfrom_to_scalar(input_data, mode):

			if mode == 'mean':
				out_scalar = np.nanmean(input_data)
			if mode == 'var':
				out_scalar = np.var(input_data)
			if mode == 'min':
				out_scalar = np.amin(input_data)
			if mode == 'max':
				out_scalar = np.amax(input_data)

			if out_scalar > 0.:
				return out_scalar
			else:
				return 0.

		def transform_features_dummy_fn(df):

			unique_ids = df['pid'].unique()

			dummy_df = []

			for i in range(len(unique_ids)):

				patient_data = df.loc[df['pid'] == unique_ids[i]]

				# print('\npatient data:', patient_data, '\n')

				dummy_df.append(
					{
					'pid': unique_ids[i],
					self.useable_feature_keys[2] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[2]], 'mean'),

					self.useable_feature_keys[3] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[3]], 'mean'),
					self.useable_feature_keys[3] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[3]], 'var'),
					self.useable_feature_keys[3] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[3]], 'min'),
					self.useable_feature_keys[3] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[3]], 'max'),

					self.useable_feature_keys[4] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[4]], 'mean'),
					self.useable_feature_keys[4] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[4]], 'var'),
					self.useable_feature_keys[4] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[4]], 'min'),
					self.useable_feature_keys[4] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[4]], 'max'),

					self.useable_feature_keys[5] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[5]], 'mean'),
					self.useable_feature_keys[5] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[5]], 'var'),
					self.useable_feature_keys[5] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[5]], 'min'),
					self.useable_feature_keys[5] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[5]], 'max'),

					self.useable_feature_keys[6] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[6]], 'mean'),
					self.useable_feature_keys[6] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[6]], 'var'),
					self.useable_feature_keys[6] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[6]], 'min'),
					self.useable_feature_keys[6] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[6]], 'max'),

					self.useable_feature_keys[7] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[7]], 'mean'),
					self.useable_feature_keys[7] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[7]], 'var'),
					self.useable_feature_keys[7] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[7]], 'min'),
					self.useable_feature_keys[7] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[7]], 'max'),

					self.useable_feature_keys[8] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[8]], 'mean'),
					self.useable_feature_keys[8] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[8]], 'var'),
					self.useable_feature_keys[8] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[8]], 'min'),
					self.useable_feature_keys[8] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[8]], 'max'),

					self.useable_feature_keys[9] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[9]], 'mean'),
					self.useable_feature_keys[9] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[9]], 'var'),
					self.useable_feature_keys[9] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[9]], 'min'),
					self.useable_feature_keys[9] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[9]], 'max'),

					self.useable_feature_keys[10] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[10]], 'mean'),
					self.useable_feature_keys[10] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[10]], 'var'),
					self.useable_feature_keys[10] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[10]], 'min'),
					self.useable_feature_keys[10] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[10]], 'max'),

					self.useable_feature_keys[11] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[11]], 'mean'),
					self.useable_feature_keys[11] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[11]], 'var'),
					self.useable_feature_keys[11] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[11]], 'min'),
					self.useable_feature_keys[11] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[11]], 'max'),

					self.useable_feature_keys[12] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[12]], 'mean'),
					self.useable_feature_keys[12] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[12]], 'var'),
					self.useable_feature_keys[12] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[12]], 'min'),
					self.useable_feature_keys[12] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[12]], 'max'),

					self.useable_feature_keys[13] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[13]], 'mean'),
					self.useable_feature_keys[13] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[13]], 'var'),
					self.useable_feature_keys[13] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[13]], 'min'),
					self.useable_feature_keys[13] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[13]], 'max'),

					self.useable_feature_keys[14] + '_mean': transfrom_to_scalar(patient_data[self.useable_feature_keys[14]], 'mean'),
					self.useable_feature_keys[14] + '_var': transfrom_to_scalar(patient_data[self.useable_feature_keys[14]], 'var'),
					self.useable_feature_keys[14] + '_min': transfrom_to_scalar(patient_data[self.useable_feature_keys[14]], 'min'),
					self.useable_feature_keys[14] + '_max': transfrom_to_scalar(patient_data[self.useable_feature_keys[14]], 'max'),
					})

			dummy_df = pd.DataFrame(dummy_df)

			return dummy_df

		self.x_train = transform_features_dummy_fn(self.x_train)
		self.x_test = transform_features_dummy_fn(self.x_test)

	def normalise_data(self):

		combined_set = pd.concat(
			[
			self.x_train,
			self.x_test
			], axis = 0)

		scaling_fn = StandardScaler()
		scaling_fn.fit(combined_set.iloc[:,1:])

		self.x_train.iloc[:,1:] = scaling_fn.transform(self.x_train.iloc[:,1:])
		self.x_test.iloc[:,1:] = scaling_fn.transform(self.x_test.iloc[:,1:])





class classifier():

	def __init__(self, dataset, plots = True):

		self.dataset = dataset

		self.training_fraction = .7
		self.svm_max_iter = 1e6

		self.binary_targets = [
			'LABEL_BaseExcess',
			'LABEL_Fibrinogen',
			'LABEL_AST',
			'LABEL_Alkalinephos',
			'LABEL_Bilirubin_total',
			'LABEL_Lactate',
			'LABEL_TroponinI',
			'LABEL_SaO2',
			'LABEL_Bilirubin_direct',
			'LABEL_EtCO2',
			'LABEL_Sepsis',
			]

		features_for_trees = .6

		#linear model based classifier with SGD training
		self.random_forest_gini = RandomForestClassifier(
			n_estimators=1024, 
			criterion='gini', 
			max_features=features_for_trees, 
			n_jobs=-1)
		self.logistic_regression = LogisticRegression(max_iter=10000)
		self.support_vectors_linear = SVC(
			kernel='linear', 
			probability=True,
			max_iter=self.svm_max_iter, 
			verbose=0)
		self.support_vectors_rbf = SVC(
			kernel='rbf', 
			probability=True, 
			max_iter=self.svm_max_iter, 
			verbose=0)
		self.support_vectors_poly2 = SVC(
			kernel='poly', 
			degree=2,
			probability=True, 
			max_iter=self.svm_max_iter, 
			verbose=0)
		self.support_vectors_poly3 = SVC(
			kernel='poly', 
			degree=3,
			probability=True, 
			max_iter=self.svm_max_iter, 
			verbose=0)
		self.ada_boost = AdaBoostClassifier(n_estimators=1024)
		self.gradient_boost = GradientBoostingClassifier(
			n_estimators=1024, 
			tol=1e-5)

		self.models = [
			self.random_forest_gini, 
			self.logistic_regression,
			self.support_vectors_linear,
			self.support_vectors_rbf,
			self.support_vectors_poly2,
			self.support_vectors_poly3,
			self.ada_boost,
			]

		self.model_names = [
			'Random Forest Gini',
			'Logistic Regression',
			'Support Vectors Linear',
			'Support Vectors RBF',
			'Support Vectors Poly 2',
			'Support Vectors Poly 3',
			'AdaBoost ',
			]


		for i in range(len(self.binary_targets)):

			self.target_specific_fitting_and_anaylsis(self.binary_targets[i])


	def target_specific_fitting_and_anaylsis(self, target_feature):

		print('\n\n*****************************************\nstart fitting and evaluation for ' + target_feature + '\n')

		x_training, x_validation, y_training, y_validation = self.stratified_split(target_feature)

		self.fit_models(x_training, y_training)

		self.roc_analysis(x_validation, y_validation, target_feature)


	def stratified_split(self, target_feature):

		stratified_splitter = StratifiedShuffleSplit(n_splits=1, train_size=self.training_fraction, random_state=1993)

		for train_index, test_index in stratified_splitter.split(np.zeros(self.dataset.y_train.shape[0]), self.dataset.y_train[target_feature]):
		# for train_index, test_index in stratified_splitter.split(np.zeros(len(dummyset)), dummyset['churn']):

			# print("\n\nsplitting dataset\nTRAIN:", train_index, "TEST:", test_index)
			x_training, x_validation = self.dataset.x_train.iloc[train_index], self.dataset.x_train.iloc[test_index]
			y_training, y_validation = self.dataset.y_train.iloc[train_index], self.dataset.y_train.iloc[test_index]

		return x_training, x_validation, y_training[target_feature], y_validation[target_feature]


	def fit_models(self, x_training, y_training):

		print('\nfitting models to data ...')
		for i in range(len(self.models)):

			start_time = time.time()

			self.models[i].fit(x_training.iloc[:,1:], y_training)

			end_time = time.time()
			print('fitted', self.model_names[i], 'in '.ljust(30 - len(self.model_names[i]), '.'), str(np.round(abs(end_time - start_time), 2)), 'sec')

		print('\n\n\n')

	def roc_analysis(self, x_validation, y_validation, target_feature, set_recall = .9):

		plt.figure()
		plt.title('Classifier ROC - ' + target_feature)

		roc_df = []

		y_true = np.reshape(np.asarray(y_validation), (-1,1))

		for i in range(len(self.models)):


			predictions = self.models[i].predict_proba(x_validation.iloc[:,1:])
			predictions = np.reshape(np.asarray(predictions[:,1]), (-1,1))


			fp_rate, tp_rate, thresholds = roc_curve(y_true, predictions)

			roc_auc = auc(fp_rate, tp_rate)

			plt.plot(fp_rate, tp_rate, label = self.model_names[i] + '   auroc: ' + str(np.round(roc_auc, 2)))


			roc_dummy = pd.DataFrame({
				'fp_rate': fp_rate,
				'tp_rate': tp_rate,
				'threshold': thresholds,
				'tp_dummy': tp_rate,
				})

			roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] > set_recall] = 0.
			roc_dummy['tp_dummy'][roc_dummy['tp_dummy'] < roc_dummy['tp_dummy'].max()] = 0.
			roc_dummy['tp_dummy'] /= roc_dummy['tp_dummy'].max()
			roc_dummy = roc_dummy.loc[roc_dummy['tp_dummy'] > .5]
			if roc_dummy.shape[0] > 1: roc_dummy = roc_dummy.iloc[0,:]

			recall_threshold = np.float(roc_dummy['threshold'])


			predictions_binary = predictions
			predictions_binary[predictions_binary >= recall_threshold] = 1.
			predictions_binary[predictions_binary < recall_threshold] = 0.

			num_true_positives = np.sum(np.abs(predictions_binary) * np.abs(y_true))
			num_false_positives = np.sum(np.abs(predictions_binary) * np.abs(1. - y_true))
			num_true_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(1. - y_true))
			num_false_negatives = np.sum(np.abs(1. - predictions_binary) * np.abs(y_true))

			num_total_positives = np.sum(np.abs(y_true))
			num_total_negatives = np.sum(np.abs(1. - y_true))

			num_total_positives_predicted = np.sum(np.abs(predictions_binary))

			recall = num_true_positives / num_total_positives
			selectivity = num_true_negatives / num_total_negatives
			precision = num_true_positives / (num_true_positives + num_false_positives)
			accuracy = (num_true_positives + num_true_negatives) / len(y_true)
			f1score = (2 * num_true_positives) / (2 * num_true_positives + num_false_positives + num_false_negatives)
			informedness = recall + selectivity - 1.

			roc_df.append({
				'model': self.model_names[i],
				'auroc': roc_auc,
				'recall': recall,
				'selectivity': selectivity,
				'precision': precision,
				'accuracy': accuracy,
				'f1score': f1score,
				'informedness': informedness,
				'#TP': num_true_positives,
				'#FP': num_false_positives,
				'#TN': num_true_negatives,
				'#FN': num_false_negatives,
				})


		roc_df = pd.DataFrame(roc_df)
		roc_df.set_index('model')
		print('\n\n\nclassification results on validation set for recall approx.', \
			set_recall, ':\n', roc_df.round(2).sort_values('auroc', ascending=False), '\n\n\n')

		
		plt.plot([0,1],[0,1],'r--')
		plt.legend(loc='lower right')
		plt.xlim([0., 1.])
		plt.ylim([0., 1.])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.grid()
		plt.savefig('results/classification_roc_analysis_' +  target_feature + '.pdf')
		plt.close()


	



data = data_loader()

classifier(data)




