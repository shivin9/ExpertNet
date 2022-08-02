import os
import pandas as pd
# from fancyimpute import KNN
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

def clean_data(direc):
	os.chdir(direc)
	cnt = 0
	data = {}
	patient_id = 0
	imputer = KNN(7) # use 7 nearest rows which have a feature to fill in each row’s missing features 
	for file in tqdm(os.listdir("./")):
		temp = pd.read_csv(file, sep='|')
		columns = temp.columns
		# >24 hr stay in ICU
		if len(temp) >= 24:
			# < 15% NA vital values according to paper but we include all records
			data[cnt] = temp
			if sum(data[cnt].isna().sum()[:7]/(7*len(data[cnt]))) < 1.0:
			
				# imputer = KNN(7) # use 7 nearest rows which have a feature to fill in each row’s missing features 
				data[cnt] = imputer.fit_transform(data[cnt])
				data[cnt] = pd.DataFrame(data[cnt], columns=columns)
				# data[cnt]['ID'] = patient_id
				cnt += 1
				patient_id += 1

	for i in tqdm(range(len(data))):
		data[i]['Sofa_O2'] = data[i].apply(lambda x: Sofa_Oxygen(x.SaO2, x.FiO2), axis=1)
		data[i]['Sofa_MAP'] = data[i]['MAP'].apply(Sofa_MAP)
		data[i]['Sofa_Bilirubin'] = data[i]['Bilirubin_total'].apply(Sofa_Bilirubin)
		data[i]['Sofa_Creatinin'] = data[i]['Creatinine'].apply(Sofa_Creatinine)
		data[i]['Sofa_Platelets'] = data[i]['Platelets'].apply(Sofa_Platelets)
		# data[cnt] = imputer.fit_transform(data[cnt])
	os.chdir('../')
	return data


def save(data, direc):
	os.chdir(direc)
	for i in range(len(data)):
		data[i].to_csv("patient_"+str(i)+".csv",header=True,index=False)
	os.chdir("../")


# Tell directory where records are
def get_processed_data(direc, hours):
	cnt = 0
	os.chdir(direc)
	data = {}
	y = []
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file, sep=',')
		# To predict if patients will get sepsis in next 24 hours
		# last_hour = min(hours+23, len(data[cnt])-1)
		if len(data[cnt][data[cnt]['SepsisLabel'] == 1]) == 0:
			last_hour = -1
		else:
			last_hour =  data[cnt][data[cnt]['SepsisLabel'] == 1].index[0]

		y.append(data[cnt].iloc[last_hour].SepsisLabel)

		data[cnt] = data[cnt].iloc[:last_hour]
		# columns = data[cnt].columns
		# y.append(data[cnt].iloc[-1].SepsisLabel)
		cnt += 1
		if cnt % 100 == 0:
			print("Read record #" + str(cnt))

	for i in tqdm(range(len(data))):
		data[i]['Sofa_O2'] = data[i].apply(lambda x: Sofa_Oxygen(x.SaO2, x.FiO2), axis=1)
		data[i]['Sofa_MAP'] = data[i]['MAP'].apply(Sofa_MAP)
		data[i]['Sofa_Bilirubin'] = data[i]['Bilirubin_total'].apply(Sofa_Bilirubin)
		data[i]['Sofa_Creatinin'] = data[i]['Creatinine'].apply(Sofa_Creatinine)
		data[i]['Sofa_Platelets'] = data[i]['Platelets'].apply(Sofa_Platelets)
	os.chdir('../../')
	return data, y


# Tell directory where records are
def get_aki(direc, ori_direc, t=24):
	os.chdir(direc)
	cnt = 0
	data = {}
	labels = pd.read_csv("listfile.csv")
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file, sep=',')
		# get the last record of the 1st day... to be used for prediction
		if len(data[cnt]) >= t and file in labels.stay.values:
			data[cnt] = data[cnt].iloc[t-1]
			data[cnt]['y'] = int(labels[labels.stay == file].y_true)
			cnt += 1

	os.chdir(ori_direc)
	return data


# Tell directory where records are
def get_aki_exp(direc, ori_direc, t_end=24):
	os.chdir(direc)
	cnt = 0
	data = {}
	labels = pd.read_csv("listfile.csv")
	final = []
	ys = []
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file)
		data[cnt] = data[cnt].ffill(axis=0)
		data[cnt][:t_end] = data[cnt][:t_end].bfill(axis=0)

		# get the last record of the 1st day... to be used for prediction
		if len(data[cnt]) >= t_end and file in list(labels.stay) and file != "listfile.csv":
			data[cnt]['Sofa_O2'] = data[cnt].apply(lambda x: Sofa_Oxygen(x.SO2, x.FiO2), axis=1)
			data[cnt]['Sofa_MAP'] = data[cnt]['Mean Airway Pressure'].apply(Sofa_MAP)
			data[cnt]['Sofa_Bilirubin'] = data[cnt]['Bilirubin Total'].apply(Sofa_Bilirubin)
			data[cnt]['Sofa_Creatinin'] = data[cnt]['Creatinine'].apply(Sofa_Creatinine)
			data[cnt]['Sofa_Platelets'] = data[cnt]['Platelets'].apply(Sofa_Platelets)

			# feats = ['Mean blood pressure', 'GCS Total', 'Mean Airway Pressure', 'PIP', 'Heart Rate',
			# 'Bilirubin Total', 'PaO2', 'Respiratory rate',
			# 'Ventilator', 'SO2', 'Temperature']
			columns = data[cnt].columns
			feats = columns[:79]
			
			ori_cols = list(data[cnt].columns)[:90]
			base = list(data[cnt][ori_cols].iloc[t_end-1])

			for feat in feats:
				df_tmp = data[cnt][feat].copy()
				cols = []
				feat_agg = []
				for agg in ['first', 'last', 'lowest', 'highest', 'median']:
				    if agg == 'first':
				        X_add = df_tmp[df_tmp.index[0]]
				    elif agg == 'last':
				        X_add = df_tmp[df_tmp.index[-1]]
				    elif agg == 'lowest':
				        X_add = df_tmp.min()
				    elif agg == 'highest':
				        X_add = df_tmp.max()
				    elif agg == 'median':
				        X_add = df_tmp.median()
				    else:
				        print('Unrecognized aggregation {}. Skipping.'.format(agg))
				        
				    # X_add = X_add.reset_index()
				    cols.append(feat + '_' + agg)
				    feat_agg.append(X_add)

				base += feat_agg
				ori_cols += cols

			final.append(base)
			# data[cnt] = pd.DataFrame(base, columns=ori_cols)
			ys.append(int(labels[labels.stay == file].y_true))
			cnt += 1
	os.chdir(ori_direc)
	final = pd.DataFrame(final, columns=ori_cols)
	final['y'] = ys
	return final



def save(data, direc, start_idx=0):
	os.chdir(direc)
	for i in range(len(data)):
		data[i].to_csv("patient_"+str(i+start_idx)+".csv",header=True,index=False)
	os.chdir("../")


# Tell directory where records are
def get_processed_data(direc, hours):
	cnt = 0
	os.chdir(direc)
	data = {}
	y = []
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file, sep=',')
		# To predict if patients will get sepsis in next 24 hours
		# last_hour = min(hours+23, len(data[cnt])-1)
		if len(data[cnt][data[cnt]['SepsisLabel'] == 1]) == 0:
			last_hour = -1
		else:
			last_hour =  data[cnt][data[cnt]['SepsisLabel'] == 1].index[0]

		if last_hour < 24 or last_hour != -1:
			continue

		y.append(data[cnt].iloc[last_hour].SepsisLabel)

		data[cnt] = data[cnt].iloc[:last_hour]
		# columns = data[cnt].columns
		# y.append(data[cnt].iloc[-1].SepsisLabel)
		cnt += 1
		if cnt % 100 == 0:
			print("Read record #" + str(cnt))

	for i in tqdm(range(len(data))):
		data[i]['Sofa_O2'] = data[i].apply(lambda x: Sofa_Oxygen(x.SaO2, x.FiO2), axis=1)
		data[i]['Sofa_MAP'] = data[i]['MAP'].apply(Sofa_MAP)
		data[i]['Sofa_Bilirubin'] = data[i]['Bilirubin_total'].apply(Sofa_Bilirubin)
		data[i]['Sofa_Creatinin'] = data[i]['Creatinine'].apply(Sofa_Creatinine)
		data[i]['Sofa_Platelets'] = data[i]['Platelets'].apply(Sofa_Platelets)
	os.chdir('../../')
	return data, y


# Tell directory where records are
def get_aki(direc):
	os.chdir(direc)
	cnt = 0
	data = {}
	labels = pd.read_csv("listfile.csv")
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file, sep=',')
		# get the last record of the 1st day... to be used for prediction
		if len(data[cnt]) >= 24 and file in labels.stay.values:
			data[cnt] = data[cnt].iloc[23]
			data[cnt]['y'] = int(labels[labels.stay == file].y_true)
			cnt += 1

	os.chdir('../../../')
	return data


def get_aki_TS(direc, ori_direc, t_end=24):
	os.chdir(direc)
	cnt = 0
	data = {}
	labels = pd.read_csv("listfile.csv")
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file, sep=',')
		# get the last record of the 1st day... to be used for prediction
		if len(data[cnt]) >= t_end and file in labels.stay.values:
			data[cnt]['y'] = int(labels[labels.stay == file].y_true)
			cnt += 1

	os.chdir(ori_direc)
	return data


# Tell directory where records are
def get_dataset_expanded(direc, ori_direc, t_end=24):
	os.chdir(direc)
	cnt = 0
	data = {}
	labels = pd.read_csv("listfile.csv")
	final = []
	ys = []
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file, sep='|')
		# get the last record of the 1st day... to be used for prediction
		if len(data[cnt]) >= t_end and file in list(labels.filename.values):
			# Forward filling imputation
			data[cnt] = data[cnt].ffill(axis=0)
			data[cnt][:t_end] = data[cnt][:t_end].bfill(axis=0)
			data[cnt]['Sofa_O2'] = data[cnt].apply(lambda x: Sofa_Oxygen(x.SaO2, x.FiO2), axis=1)
			data[cnt]['Sofa_MAP'] = data[cnt]['MAP'].apply(Sofa_MAP)
			data[cnt]['Sofa_Bilirubin'] = data[cnt]['Bilirubin_total'].apply(Sofa_Bilirubin)
			data[cnt]['Sofa_Creatinin'] = data[cnt]['Creatinine'].apply(Sofa_Creatinine)
			data[cnt]['Sofa_Platelets'] = data[cnt]['Platelets'].apply(Sofa_Platelets)

			if len(data[cnt][data[cnt]['SepsisLabel'] == 1]) == 0:
				last_hour = 100000
			else:
				last_hour =  data[cnt][data[cnt]['SepsisLabel'] == 1].index[0]

			if last_hour < t_end:
				continue

			# feats = ['Mean blood pressure', 'GCS Total', 'Mean Airway Pressure', 'PIP', 'Heart Rate',
			# 'Bilirubin Total', 'PaO2', 'Respiratory rate',
			# 'Ventilator', 'SO2', 'Temperature']
			full_feats = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
			partial_feats = ['HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos',\
			'Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium',\
			'Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
			
			ori_cols = list(data[cnt].columns)
			base = list(data[cnt].iloc[t_end-1])

			for feat in full_feats+partial_feats:
				df_tmp = data[cnt][feat].copy()
				cols = []
				feat_agg = []
				for agg in ['first', 'last', 'lowest', 'highest', 'median']:
				    if agg == 'first':
				        X_add = df_tmp[df_tmp.index[0]]
				    elif agg == 'last':
				        X_add = df_tmp[df_tmp.index[-1]]
				    elif agg == 'lowest':
				        X_add = df_tmp.min()
				    elif agg == 'highest':
				        X_add = df_tmp.max()
				    elif agg == 'median':
				        X_add = df_tmp.median()
				    else:
				        print('Unrecognized aggregation {}. Skipping.'.format(agg))
				        
				    # X_add = X_add.reset_index()
				    cols.append(feat + '_' + agg)
				    feat_agg.append(X_add)
				base += feat_agg
				ori_cols += cols

			final.append(base)
			# data[cnt] = pd.DataFrame(base, columns=ori_cols)
			ys.append(int(labels[labels.filename == file].sepsis_label))
			cnt += 1

	os.chdir(ori_direc)
	final = pd.DataFrame(final, columns=ori_cols)
	final['y'] = ys
	final = final.drop(['SepsisLabel'], axis=1)
	return final


# Tell directory where records are
def get_sepsis_TS(direc, ori_direc, t_end=24):
	os.chdir(direc)
	cnt = 0
	data = {}
	labels = pd.read_csv("listfile.csv")
	final = []
	ys = []
	for file in tqdm(os.listdir("./")):
		data[cnt] = pd.read_csv(file, sep='|')
		# get the last record of the 1st day... to be used for prediction
		if len(data[cnt]) >= t_end and file in list(labels.filename.values):
			# Forward filling imputation
			data[cnt] = data[cnt].ffill(axis=0)
			data[cnt][:t_end] = data[cnt][:t_end].bfill(axis=0)
			data[cnt]['Sofa_O2'] = data[cnt].apply(lambda x: Sofa_Oxygen(x.SaO2, x.FiO2), axis=1)
			data[cnt]['Sofa_MAP'] = data[cnt]['MAP'].apply(Sofa_MAP)
			data[cnt]['Sofa_Bilirubin'] = data[cnt]['Bilirubin_total'].apply(Sofa_Bilirubin)
			data[cnt]['Sofa_Creatinin'] = data[cnt]['Creatinine'].apply(Sofa_Creatinine)
			data[cnt]['Sofa_Platelets'] = data[cnt]['Platelets'].apply(Sofa_Platelets)

			if len(data[cnt][data[cnt]['SepsisLabel'] == 1]) == 0:
				last_hour = 100000
			else:
				last_hour =  data[cnt][data[cnt]['SepsisLabel'] == 1].index[0]

			if last_hour < 24:
				continue

			else:
				base = list(data[cnt])
				final.append(base)
				# data[cnt] = pd.DataFrame(base, columns=ori_cols)
				ys.append(int(labels[labels.filename == file].sepsis_label))
				cnt += 1

	os.chdir(ori_direc)
	final = pd.DataFrame(final, columns=ori_cols)
	final['y'] = ys
	final = final.drop(['SepsisLabel'], axis=1)
	return final


if __name__ == '__main__':
	final_train = get_aki_exp('./train', ori_direc=os.curdir, t_end=24)
	final_test = get_aki_exp('./test', ori_direc=os.curdir, t_end=24)

	y_train = final_train.y
	y_test = final_test.y

	final_train = final_train.drop(['y'], axis=1)
	final_test = final_test.drop(['y'], axis=1)
	final = pd.concat([final_train, final_test])
	final_y = pd.concat([y_train, y_test])
	final = final.fillna(0)
	final.to_csv('X.csv', index=False)
	final_y.to_csv('y.csv', index=False)


def get_sofa_features(data):
	sofa_features = []
	sepsis_patients = []
	non_sepsis_patients = []
	for i in tqdm(range(len(data))):
		# choose only sepsis patients
		if data[i].iloc[-1]['SepsisLabel'] == 1.0:
			# Average SOFA score for the 1st two day
			rec = pd.DataFrame.mean(data[i].iloc[:48][['Sofa_O2', 'Sofa_MAP', 'Sofa_Bilirubin', 'Sofa_Creatinin', 'Sofa_Platelets']])
			# select patients with SOFA score > 2
			# if sum(rec) > 2:
			sepsis_patients.append(i)
			sofa_features.append(rec)

		# All sepsis patients have sepsis till the end of their ICU stay if they ever catch it
		else:
			non_sepsis_patients.append(i)

	sofa_features = np.array(sofa_features)
	return sofa_features, sepsis_patients, non_sepsis_patients


# for visualization purposes
# X_embedded = TSNE(n_components=2).fit_transform(sofa_features)                                                                                    
# plt.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.2)                                                                                             
# plt.show()


# approximating SpO2 with SaO2 as that is only given in the data
# SaO2 ~ SpO2; https://www.resmedjournal.com/article/S0954-6111(13)00053-X/pdf
def Sofa_Oxygen(SaO2, FiO2):
	if FiO2 == 0:
		return 0
	val = SaO2/FiO2
	if val >= 302:
		return 0
	elif 221 <= val < 302:
		return 1
	elif 142 <= val < 221:
		return 2
	elif 67 <= val < 142:
		return 3
	else:
		return 4


def Sofa_MAP(val):
	if val >= 70:
		return 0
	elif val < 70:
		return 1


def Sofa_Bilirubin(val):
	if val < 1.2:
		return 0
	elif 1.2 <= val < 1.9:
		return 1
	elif 1.9 <= val < 6.0:
		return 2
	elif 6.0 <= val < 12.0:
		return 3
	else:
		return 4


def Sofa_Creatinine(val):
	if val < 1.2:
		return 0
	elif 1.2 <= val < 1.9:
		return 1
	elif 1.9 <= val < 3.4:
		return 2
	elif 3.4 <= val < 4.9:
		return 3
	else:
		return 4


def Sofa_Platelets(val):
	if val >= 150:
		return 0
	elif 100 <= val < 150:
		return 1
	elif 50 <= val < 100:
		return 2
	elif 20 <= val < 50:
		return 3
	else:
		return 4
