import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd

from gages_utils import load_gages_dataset
from hydroIndices_utils import load_hydrological_indices

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, normalize

from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_score, StratifiedKFold, ParameterGrid, RandomizedSearchCV

import xgboost as xgb
import shap
from scipy.stats import spearmanr

from joblib import Parallel, delayed
import multiprocessing as mp
from sklearn.base import clone
import ast
import wandb
from scipy.stats import randint, uniform

from econml.dml import CausalForestDML
from econml.inference import BootstrapInference
from copy import copy

def make_causal_traits(X_forCausal, traits_inCommunites, y, params, nan_feature_names = [], model_str="XGB"):
    #traits_forCausal = X_forCausal.columns.values.tolist()
    #num_traits_forCausal = len(traits_forCausal)
    #traits_forCausal = []
    num_drop=0
    #traits_forCausal_withRedundancy = {}
    #for i in range(num_traits_forCausal):
    dict_results = {"feature":[], 'ATE':[], 'ATE_lb':[], 'ATE_ub':[], 'abs_CATE':[]}
    for i in range(len(traits_inCommunites)):
        ate_inComm = []
        #print(traits_forCausal)
        for treatment in traits_inCommunites[i]:

            try:
                if treatment in nan_feature_names:
                    ate_inComm.append(0.)
                    continue
                 
                print("T", i, treatment)
                W_feats = [c for c in traits_inCommunites[i] if c != treatment]
                #controls = [c for j,c in enumerate(traits_forCausal) if j != i-num_drop]
                #controls = [c for j,c in enumerate(X_forCausal) if j != i]
                controls = [c[0] for j,c in enumerate(traits_inCommunites) if j != i]
                print("W", W_feats)
                print("X", controls)
                
                # Prepare arrays
                T = X_forCausal[treatment]  # treatment (continuous)
                X_controls = X_forCausal[controls]  # controls (confounders)
                if len(W_feats)>0:
                    W = X_forCausal[W_feats]
                else:
                    W = None
                Y = copy(y)
        
                ### OVERRIDE X and W
                ###X_controls = X_df[W_feats+controls].values; W = None
            
                #rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
                
                # instantiate model
                est = CausalForestDML(
                    model_t=xgb.XGBRegressor(**params, random_state=42, n_jobs=6),
                    model_y=xgb.XGBRegressor(**params, random_state=42, n_jobs=6),
                    #discrete_treatment=False,
                    cv=5,
                    n_estimators=1000,
                    random_state=42,
                    criterion='mse',
                    n_jobs=6
                )

                
                est.fit(Y, T, X=X_controls, W=W)#, inference=BootstrapInference(n_bootstrap_samples=10))
                
                # Estimate average treatment effect (ATE) and confidence interval
                ate = est.ate(X=X_controls)         # global ATE
                ate_lb, ate_ub = est.ate_interval(X=X_controls)  # CI (if available)
                print('ATE', ate, 'CI', ate_lb, ate_ub)
                
                # Heterogeneous effects
                pred = est.effect(X=X_controls)  # CATE for each sample
                lb, ub = est.effect_interval(X=X_controls)  # CI (if available)
                abs_cate = np.mean(np.abs(pred))
                print('abs average effect', abs_cate)
                #corr, p_val = spearmanr(T, pred)
                #print("correlation", corr, p_val)
                dict_results["feature"].append(treatment)
                dict_results["ATE"].append(ate)
                dict_results["ATE_lb"].append(ate_lb)
                dict_results["ATE_ub"].append(ate_ub)
                dict_results["abs_CATE"].append(abs_cate)
                """
                X_unif = np.zeros((100, X_controls.shape[1]))
                X_unif[:, 0] = np.linspace(-2, 2, 100)
                pred_unif = est.effect(X=X_unif)
                lb_unif, ub_unif = est.effect_interval(X=X_unif)  # CI (if available)
                """
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                #plt.plot(X_unif[:, 0], pred_unif, label='forestdml (causal forest)')
                #plt.fill_between(X_unif[:, 0], lb_unif, ub_unif, alpha=.4, label='CI')
                plt.scatter(T, pred)
                plt.show()
            except:
                dict_results["feature"].append(treatment)
                dict_results["ATE"].append(0.)
                dict_results["ATE_lb"].append(0.)
                dict_results["ATE_ub"].append(0.)
                dict_results["abs_CATE"].append(0.)

    
        #    ate_inComm.append(abs_cate)
        #for j in range(len(ate_inComm)):
        #    if ate_inComm[j]>=0.05:
        #        traits_forCausal_withRedundancy[traits_inCommunites[i][j]]=ate_inComm[j]
                
        #maxabs_ate_idx = np.argmax(ate_inComm)
        #if ate_inComm[maxabs_ate_idx]>=0.05:
        #    traits_forCausal.append(traits_inCommunites[i][maxabs_ate_idx])
        #    print("using", traits_inCommunites[i][maxabs_ate_idx], "in position", i)
        """
            traits_forCausal[i-num_drop] = traits_inCommunites[i][maxabs_ate_idx]
        else:
            print("eliminated", traits_forCausal[i-num_drop])
            print(traits_forCausal.pop(i-num_drop))
            num_drop+=1
        """
        #print(traits_forCausal_withRedundancy)
    return pd.DataFrame(dict_results)



def load_dataset(target_name, hydroIndices_filename, filtering_dict=None, keep_STAID=False, normalize_target=True):
    
    df_gages, traits = load_gages_dataset(do_process_raw_columns=True, processing_mapping="C0", class_=None, add_traits_to_be_removed=["runoffs", "ASPECT_DEGREES", "BFI_AVE"], clusters_filename="datasets/catchments_classes.csv")

    if filtering_dict is not None:
        for i,j in filtering_dict.items():
            df_gages = df_gages[df_gages[i]==j]

    #df_hydroIndices = load_hydrological_indices(database_filter='USGS', do_area_normalization=False)
    df_hydroIndices = pd.read_csv(hydroIndices_filename, sep=",", encoding = "utf-8", encoding_errors='replace', dtype={"STAID":"str"})

    df_gages_hydroIndices = pd.merge(df_gages, df_hydroIndices, on="STAID", how='inner')

    if target_name in ["Qmean"]:
        df_gages_hydroIndices[target_name] = 86.4*df_gages_hydroIndices[target_name]/df_gages_hydroIndices["DRAIN_SQKM"]

    df_gages_hydroIndices_traits = df_gages_hydroIndices[traits]
    X = StandardScaler().fit_transform(df_gages_hydroIndices_traits)
    df_gages_hydroIndices_traits_standardized = pd.DataFrame(X, columns=df_gages_hydroIndices_traits.columns)
    y = df_gages_hydroIndices[target_name].values
    if normalize_target:
        y = (y - np.mean(y))/np.std(y)

    if keep_STAID:
        df_gages_hydroIndices_traits.insert(loc=0, column='STAID', value=df_gages_hydroIndices["STAID"])
        df_gages_hydroIndices_traits_standardized.insert(loc=0, column='STAID', value=df_gages_hydroIndices["STAID"])

    return df_gages_hydroIndices_traits, traits, df_gages_hydroIndices_traits_standardized, X, y

class RepeatedStratifiedKFoldContinuous:
    """
    Repeated Stratified K-Fold cross-validator for continuous targets.

    This cross-validator provides train/test indices to split data in train/test
    sets. It is a variant of Stratified K-Fold that returns stratified folds,
    repeated `n_repeats` times. The stratification is done by first binning
    the continuous target variable `y`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    n_bins : int, default=5
        The number of bins to use for discretizing the continuous target.
        Uses quantile-based binning.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the shuffling for each repetition.
        Pass an int for reproducible output across multiple function calls.
    """
    def __init__(self, n_splits=5, n_repeats=10, n_bins=5, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")
        
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_bins = n_bins
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the total number of splitting iterations in the cross-validator."""
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems. The continuous
            target is binned for stratification.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        # Bin the continuous target variable once
        try:
            y_binned = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')
            num_bins_created = len(np.unique(y_binned))
            if num_bins_created < self.n_bins:
                 print(f"Warning: Dropped bins due to duplicate edges. Created {num_bins_created} bins instead of {self.n_bins}.")
        except ValueError as e:
            # Fallback to standard binning if quantile binning fails
            print(f"Warning: Quantile binning failed: {e}. Falling back to standard binning.")
            y_binned = pd.cut(y, bins=self.n_bins, labels=False, duplicates='drop')

        # Create a random number generator for reproducibility
        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_repeats):
            # Create a new StratifiedKFold instance for each repeat with a new random seed
            # This ensures that each repetition is a different shuffle of the data
            skf_seed = rng.randint(np.iinfo(np.int32).max)
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=skf_seed)
            
            for train_idx, test_idx in skf.split(X, y_binned):
                yield train_idx, test_idx


def make_parallel_MI(A, B, n_neighbors):
    return mutual_info_regression(A, B, n_neighbors=n_neighbors, discrete_features=False, random_state=42)

# Mutual Inofrmation
def make_MI(X, n_neighbors_list = [2,3,4], do_plot=True):

    Adj_MI = np.zeros((len(n_neighbors_list), X.shape[1], X.shape[1]))
    
    for j,n_neighbors in enumerate(n_neighbors_list):
        print(n_neighbors)
        #MI = Parallel(n_jobs=6)(delayed(make_parallel_MI)(X[:,i+1:], X[:,i], n_neighbors) for i in range(X.shape[1]-1))
        #MI = Parallel(n_jobs=6)(delayed(make_parallel_MI)(X[:,i:], X[:,i], n_neighbors) for i in range(X.shape[1]))
        with mp.Pool(processes=6) as pool_MI:
            MI = pool_MI.starmap(make_parallel_MI, [(X[:,i:], X[:,i], n_neighbors) for i in range(X.shape[1])])
        for i in range(len(MI)):
            #Adj_MI[j,i,i+1:] = MI[i]
            #Adj_MI[j,i+1:,i] = MI[i]
            Adj_MI[j,i,i:] = MI[i]
            Adj_MI[j,i:,i] = MI[i]
        
        for u in range(Adj_MI.shape[1]):
            for v in range(u+1, Adj_MI.shape[2]):
                Adj_MI[j,u,v] = Adj_MI[j,u,v]/((Adj_MI[j,u,u] + Adj_MI[j,v,v])/2.)
                Adj_MI[j,v,u] = Adj_MI[j,v,u]/((Adj_MI[j,u,u] + Adj_MI[j,v,v])/2.)
                #Adj_MI[j,u,v] = Adj_MI[j,u,v]/(np.max([Adj_MI[j,u,u], Adj_MI[j,v,v]]))
                #Adj_MI[j,v,u] = Adj_MI[j,v,u]/(np.max([Adj_MI[j,u,u], Adj_MI[j,v,v]]))
        #np.fill_diagonal(Adj_MI[j,:,:],np.nan)
        #Adj_MI[j,:,:] /= np.nanmax(Adj_MI[j,:,:])
        np.fill_diagonal(Adj_MI[j,:,:],0.)
        
    Adj_MI = np.mean(Adj_MI, axis=0)

    if do_plot:
        plt.hist(Adj_MI[Adj_MI>0].flatten(), bins=np.logspace(np.log10(np.min(Adj_MI[Adj_MI>0])),1,100))
        plt.xscale("log")

    return (Adj_MI)

def make_target_corr(X, y, p_val_thresh = 0.05):
    spCorr_toTarget = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        trait_values = X[:,i].reshape(-1, 1)
        corr, p_val = spearmanr(trait_values, y)
        if p_val<=p_val_thresh:
            spCorr_toTarget[i] = corr
    return spCorr_toTarget

def make_target_MI(X, y, n_neighbors_list = [2,3,4], do_plot=True):

    MI_toTarget = np.zeros((len(n_neighbors_list), X.shape[1]))
    
    for j,n_neighbors in enumerate(n_neighbors_list):
        print(n_neighbors)
        MI = Parallel(n_jobs=6)(delayed(make_parallel_MI)(X[:,i].reshape(-1, 1), y, n_neighbors) for i in range(X.shape[1]))
        for i in range(len(MI)):
            MI_toTarget[j,i] = MI[i]
        MI_self = make_parallel_MI(y.reshape(-1,1), y, n_neighbors)
        MI_toTarget[j,:] /= MI_self

    MI_toTarget = np.mean(MI_toTarget, axis=0)

    if do_plot:
        plt.hist(MI_toTarget.flatten(), bins=np.logspace(-1,1,100))
        plt.xscale("log")

    return MI_toTarget

def plot_barchart(df, fname_suffix, target_name='', r2_test=None):


    df_sorted = df.sort_values(by="Importance", ascending=False)
    #df_sorted = df_sorted[:np.min([50, len(df_sorted)])]
    top_features = ["%s \n(%s)" % (df_sorted["Feature"].values[i], df_sorted["Trait Categories"].values[i]) for i in np.arange(len(df_sorted))]

    norm_corr = (df_sorted["Spearman Correlation"] + 1) / 2  # [-1, 1] → [0, 1]
    colors = plt.cm.bwr(norm_corr)  # Use blue-white-red colormap

    category_df = df.groupby("Trait Categories")[["Importance", "Importance Std", "Spearman Correlation"]].agg(np.nanmean).reset_index()
    #category_df["Spearman Correlation"] = np.nan_to_num(df.groupby("Trait Categories")["Spearman Correlation"].agg('mean').values)
    category_df_sorted = category_df.sort_values(by="Importance", ascending=False)
    #print(category_df_sorted)
    top_categories = category_df_sorted["Trait Categories"].values
    norm_corr_category = (category_df_sorted["Spearman Correlation"] + 1) / 2  # [-1, 1] → [0, 1]
    colors_category = plt.cm.bwr(norm_corr_category)  # Use blue-white-red colormap

    fig, ax = plt.subplots(1,2,figsize=(20, 10))
    bar_width = 0.65
    top_n = len(df_sorted)
    y_pos = np.arange(top_n)
    top_n_category = len(category_df_sorted)
    y_pos_category = np.arange(top_n_category)

    if len(y_pos)>=30:
        ax[0].barh(y_pos[:30],
                pd.concat([df_sorted.iloc[:15,:], df_sorted.iloc[-15:,:]])["Importance"],
                xerr=pd.concat([df_sorted.iloc[:15], df_sorted.iloc[-15:]])["Importance Std"],
                   height=bar_width, color=np.vstack([colors[:15,:], colors[-15:,:]]), edgecolor='k', lw=1)
        ax[0].set_yticks(y_pos[:30])
        ax[0].set_yticklabels(np.concatenate([top_features[:15], top_features[-15:]]))
    else:
        ax[0].barh(y_pos,
                df_sorted["Importance"],
                xerr=df_sorted["Importance Std"],
                height=bar_width, color=colors, edgecolor='k', lw=1)
        ax[0].set_yticks(y_pos)
        ax[0].set_yticklabels(top_features)

    ax[0].invert_yaxis()  # Invert the y-axis

    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-1, vmax=1))
    cbar = plt.colorbar(sm, label="Spearman Correlation", ax=ax[0])

    ax[1].barh(y_pos_category, category_df_sorted.iloc[:top_n_category,:]["Importance"],
               #xerr=category_df_sorted.iloc[:top_n_category,:]["Importance Std"],
               height=bar_width, color=colors_category, edgecolor='k', lw=1)
    ax[1].invert_yaxis()  # Invert the y-axis

    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-1, vmax=1))
    cbar_categories = plt.colorbar(sm, label="Spearman Correlation", ax=ax[1])

    #plt.yticks(y_pos, top_features)
    ax[0].set_xlabel("Mean |SHAP Value|")
    ax[1].set_xlabel("Mean |SHAP Value|")

    ax[1].set_yticks(y_pos_category)
    ax[1].set_yticklabels(top_categories)


    if r2_test is not None:
        ax[0].set_title("%s (R2=%.3f)\nTraits" % (target_name, r2_test))
        ax[1].set_title("%s (R2=%.3f)\nCategories" % (target_name, r2_test))
    else:
        ax[0].set_title("%s\nTraits" % (target_name))
        ax[1].set_title("%s\nCategories" % (target_name))
    #plt.legend()
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig("barchart_shap_importance_%s.png" % (fname_suffix), dpi=600)
    plt.show()

def make_parallel_shap(X, y, cv_results, i):
    explainers = shap.Explainer(cv_results['estimator'][i], seed=42)(X[cv_results['indices']['test'][i]], check_additivity=False)
    shap_values = np.abs(explainers.values).mean(axis=0)
    
    spearman_corr = []
    for j in range(len(shap_values)):
        val_data = explainers[:,j].data
        val_values = explainers[:,j].values
        if (len(np.unique(val_data))==1) | (len(np.unique(val_values))==1):
            spearman_corr.append(np.nan)
        else:
            corr, p_val = spearmanr(val_data, val_values)
            if p_val>0.05: corr=0
            spearman_corr.append(corr)
            
    return [shap_values, spearman_corr]

def make_CVshap(X, y, best_params, n_repeats=1):
    
    #rkf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    rkf = RepeatedStratifiedKFoldContinuous(n_splits=5, n_repeats=n_repeats, random_state=42)
    
    model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=1)
                
    cv_results = cross_validate(model, X, y, cv=rkf, scoring=['neg_root_mean_squared_error', 'r2'], return_estimator=True, return_indices=True, n_jobs=6)
    shap_values_corr = np.array(Parallel(n_jobs=6)(delayed(make_parallel_shap)(X, y, cv_results, i) for i in range(len(cv_results['estimator']))))

    df_final_shap = pd.DataFrame({k: cv_results[k].tolist() for k in ['test_neg_root_mean_squared_error', 'test_r2'] if k in cv_results})
    df_final_shap["shap_importance"] = [shap_values_corr[i,0,:].tolist() for i in np.arange(shap_values_corr.shape[0])]
    df_final_shap["spearman_correlation"] = [shap_values_corr[i,1,:].tolist() for i in np.arange(shap_values_corr.shape[0])]

    return df_final_shap

def wandb_wrapper(X, y, count=10, early_stopping_rounds=100, n_repeats=3):
    def train_xgb_reg_cv(config=None):
        #with wandb.init(config=config) as run:
        #    config = run.config
        wandb.init()
        config = wandb.config
        
        # Define model
        model = xgb.XGBRegressor(
            n_estimators=5000,#config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            #subsample=config.subsample,
            #colsample_bytree=config.colsample_bytree,
            gamma=config.gamma,
            reg_lambda=config.reg_lambda,
            reg_alpha=config.reg_alpha,
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
            eval_metric="rmse"
        )
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print("rmse", rmse)
        wandb.log({"rmse": rmse})
        """
        kf = RepeatedStratifiedKFoldContinuous(n_splits=5, n_repeats=n_repeats, random_state=42)
        #kf = KFold(n_splits=5, random_state=42, shuffle=True)
        #kf = RepeatedKFold(n_splits=5, random_state=42, n_repeats=n_repeats)
        rmses = []
        r2s = []
        best_iters = []
        #feature_importances = []
    
        for train_idx, val_idx in kf.split(X,y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            m = clone(model)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_iter = m.get_booster().best_iteration + 1  # actual rounds used
            preds = m.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            best_iters.append(best_iter)
            rmses.append(rmse)
            r2s.append(r2)
            
            # Track feature importances
            #booster = m.get_booster()
            #importance = booster.get_score(importance_type="gain")
            #feature_importances.append(importance)
        
        mean_best_iter = int(np.round(np.mean(best_iters)))
        mean_rmse = np.mean(rmses)
        mean_r2 = np.mean(r2s)
        print("RMSE", rmses, mean_rmse)
        print("R^2", r2s, mean_r2)
        wandb.log({"rmse": mean_rmse, "best_iter":mean_best_iter})
        
    
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'rmse', 'goal': 'minimize'},
        'parameters': {
            #'n_estimators': {'values': [100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]},
            'learning_rate': {'min': 0.001, 'max': 1.0},
            'max_depth': {'values': [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]},
            #'subsample': {'min': 0.5, 'max': 1.0},
            #'colsample_bytree': {'min': 0.5, 'max': 1.0},
            'gamma': {'min': 0.0, 'max': 50.0},
            'reg_lambda': {'min': 0.0, 'max': 50.0},
            'reg_alpha': {'min': 0.0, 'max': 50.0},
        }
    }
    
    project = "xgboost-nested-hpo"  # 👈 must match your actual W&B project
    entity = None  # Or your W&B username if using a team/org
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    wandb.agent(sweep_id, function=train_xgb_reg_cv, count=count)
    
    # Automatically retrieve best parameters from sweep
    api = wandb.Api()
    sweep = api.sweep(f"{entity + '/' if entity else ''}{project}/{sweep_id}")
    
    df_wandb = pd.DataFrame()
    list_rmse = []
    list_params = []
    list_best_iter = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        list_params.append({k.replace("config/", ""): v["value"] for k, v in ast.literal_eval(run.config).items() if not k.startswith("_")})
        list_rmse.append(ast.literal_eval(run.summary_metrics)["rmse"])
        list_best_iter.append(ast.literal_eval(run.summary_metrics)["best_iter"])
    df_wandb["rmse"] = list_rmse
    df_wandb["params"] = list_params
    df_wandb["best_iter"] = list_best_iter

    return df_wandb

def do_RFECV_shap(X_df, y, best_params={}, n_repeats=1):

    X_RFE = X_df.values
    traits_RFE = X_df.columns.values

    shap_importance_ave = []
    num_feats = []
    dict_results = {'rmse_test':{},'r2_test':{},'shap_importance':{},'traits':{}}
    thresholds_steps = [(50, 1), (100, 5), (1000, 10)]

    while X_RFE.shape[1]>0:

        num_feats_toKeep = 50
        for j in range(len(thresholds_steps))[::-1]:
            if X_RFE.shape[1] <= thresholds_steps[j][0]:
                num_feats_toKeep = thresholds_steps[j][1]
                

        print(X_RFE.shape[1], num_feats_toKeep)

        df_shap = make_CVshap(X_RFE, y, best_params, n_repeats=n_repeats)
        num_feats.append(X_RFE.shape[1])
        
        dict_results['rmse_test'].setdefault(X_RFE.shape[1], (-df_shap["test_neg_root_mean_squared_error"].values).tolist())
        dict_results['r2_test'].setdefault(X_RFE.shape[1], df_shap["test_r2"].values.tolist())
        print(np.mean(dict_results['rmse_test'][X_RFE.shape[1]]), np.mean(dict_results['r2_test'][X_RFE.shape[1]]))
        shap_importance_ave = np.array([np.array(x) for x in df_shap["shap_importance"].values]).mean(axis=0)
        dict_results['shap_importance'].setdefault(X_RFE.shape[1], shap_importance_ave.tolist())
        dict_results['traits'].setdefault(X_RFE.shape[1], traits_RFE.tolist())
        lowest_imp_feat_idxs = np.argsort(shap_importance_ave)[:num_feats_toKeep]
        feats_toKeep_idxs = [j for j in range(X_RFE.shape[1]) if j not in lowest_imp_feat_idxs]
        lowest_imp_feats = traits_RFE[lowest_imp_feat_idxs]
        X_RFE = X_RFE[:, feats_toKeep_idxs]
        traits_RFE = traits_RFE[feats_toKeep_idxs]
        print(lowest_imp_feat_idxs, lowest_imp_feats, X_RFE.shape, len(traits_RFE))
        final_df = pd.DataFrame(dict_results, index=num_feats)

    return final_df


def xgb_randomsearch_cv(
    X, y,
    #param_grid,
    n_iter=10,
    n_splits=5,
    n_repeats=1,
    random_state=42,
    #early_stopping_rounds=100,
    #complexity_penalty=1e-4,
    #flatness_tolerance=0.005,
    ):

    rkf = RepeatedStratifiedKFoldContinuous(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)    

    param_grid = {
        "n_estimators": [100,200,300,500,750,1000,2000,3000],
        "max_depth": randint(2, 20),
        "learning_rate": uniform(0.01, 0.3),
    }

    model = xgb.XGBRegressor(
        #n_estimators=5000,  # large upper limit
        random_state=random_state,
        n_jobs=6,
        eval_metric="rmse",
        #**params,
        #early_stopping_rounds=early_stopping_rounds
        )


    # 4. Randomized Search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,             # Number of random combinations to try
        scoring=["neg_mean_squared_error", 'r2'],
        refit='neg_mean_squared_error',
        cv=rkf,
        verbose=1,
        random_state=42,
        n_jobs=6
    )

    # 5. Fit
    random_search.fit(X, y)

    # 6. Best model and score
    best_params = random_search.best_params_
    print("Best parameters:", random_search.best_params_)
    print("Best CV score:", random_search.best_score_)

    results_df = pd.DataFrame(random_search.cv_results_)

    return results_df

def xgb_gridsearch_cv(
    X, y,
    param_grid,
    n_splits=5,
    n_repeats=1,
    random_state=42,
    early_stopping_rounds=50,
    #complexity_penalty=1e-4,
    flatness_tolerance=0.005,
    verbose=True
    ):
    """
    Perform grid search for XGBoost with repeated cross-validation.
    Logs RMSE and R² for each run into a pandas DataFrame.
    
    Parameters
    ----------
    X, y : array-like
        Training features and targets.
    param_grid : dict
        Dictionary of parameters to grid search, e.g.
        {'max_depth': [3, 6], 'learning_rate': [0.01, 0.1], 'n_estimators': [200]}
    n_splits : int
        Number of folds for K-fold cross-validation.
    n_repeats : int
        Number of repetitions for repeated CV.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with one row per parameter set containing mean ± std of RMSE and R².
    best_params : dict
        Parameter set with the lowest mean RMSE.
    """
    
    rkf = RepeatedStratifiedKFoldContinuous(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)    
    #rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    all_results = []

    for i,params in enumerate(ParameterGrid(param_grid)):
        rmse_scores, r2_scores, best_rounds = [], [], []
        
        #if verbose:
        #    print(f"Evaluating params: {params}")
        
        for train_idx, test_idx in rkf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            #dtrain = xgb.DMatrix(X_train, label=y_train)
            #dtest = xgb.DMatrix(X_test, label=y_test)

            
            model = xgb.XGBRegressor(
                n_estimators=5000,  # large upper limit
                random_state=random_state,
                n_jobs=6,
                eval_metric="rmse",
                **params,
                early_stopping_rounds=early_stopping_rounds,
            )
            
            """
            # Universal callback for early stopping
            early_stop = xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True,
                data_name="validation_0",
                metric_name="rmse"
            )
            """
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                #early_stopping_rounds=early_stopping_rounds,
                verbose=False
                )
            """
            bst = xgb.train(
                n_estimators=5000,
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                callbacks=[CustomLogger(), early_stop_callback]
            """

            best_iter = model.get_booster().best_iteration + 1  # actual rounds used
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            best_rounds.append(best_iter)

        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_rounds = np.mean(best_rounds)

        #penalized_rmse = mean_rmse + complexity_penalty * mean_rounds

        result = {
            **params,
            'mean_RMSE': np.mean(rmse_scores),
            'std_RMSE': np.std(rmse_scores),
            'mean_R2': np.mean(r2_scores),
            'std_R2': np.std(r2_scores),
            'mean_rounds': np.mean(best_rounds),
        #    'score_with_penalty': penalized_rmse


        }

        print(f"{i+1}, Evaluated params: {params}, RMSE: {result['mean_RMSE']}, r2: {result['mean_R2']}")
        #print(rmse_scores)
        #print(r2_scores)

        all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by='mean_RMSE', inplace=True)
    #best_params = results_df.iloc[0].to_dict()
 
    best_rmse = results_df["mean_RMSE"].min()
    candidates = results_df[results_df["mean_RMSE"] <= best_rmse * (1 + flatness_tolerance)]
    print(len(candidates))
    preferred = candidates.sort_values("mean_rounds").iloc[0].to_dict()

    if verbose:
        print("\nBest trade-off model:")
        print(preferred)

    return results_df, preferred
