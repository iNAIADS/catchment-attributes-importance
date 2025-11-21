import pandas as pd
import os.path

DIR = 'datasets/streamflow_indices'

def load_hydrological_indices(filename='station_catalogue.csv', do_area_normalization=False, database_filter=None):

	df = pd.read_csv(os.path.join(DIR, filename), sep=",", encoding = "utf-8", encoding_errors='replace', dtype={"no_ori":"str"})
	
	no_ori_USGS = df[df["database"]=="USGS"]["no_ori"]
	no_ori_USGS = ['0'+x if len(x)<8 else x for x in no_ori_USGS]
	df.loc[df["database"]=="USGS", "no_ori"] = no_ori_USGS

	df.rename(columns={"no_ori":"STAID"}, inplace=True)

	if database_filter is not None:
		if type(database_filter) == str:
			database_filter = [database_filter]
		df = df[df["database"].isin(database_filter)]

	if do_area_normalization:
		df["Qmean"] = 86.4*df["Qmean"]/df["area"]
		df.dropna(subset=['Qmean'], inplace=True)


	return df