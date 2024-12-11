import sys, os, yaml
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb

from scipy.stats import norm
from scipy import interpolate
from pathlib import Path
from econml.dr import LinearDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from synthetic_data_XGBoost.synthetic_data_generation import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.parent.absolute())


def load_yaml(path_relative):
    return yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))

def load_all_yaml_from_directory(path_relative):
    path = get_project_path() + path_relative
    files = os.listdir(path)
    configs = []
    for file in files:
        if file.endswith(".yaml"):
            configs.append(load_yaml(path_relative + file[:-5]))
    return configs

def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)





def rbf_linear_kernel(X1, X2, length_scales=np.array([0.1,0.1]), alpha=np.array([0.1,0.1]), var=5):  # works with 2D covariates only
    distances = np.linalg.norm((X1[:, None, :] - X2[None, :, :]) / length_scales, axis=2)
    rbf_term = var * np.exp(-0.5 * distances**2)
    linear_term = np.dot(np.dot(X1, np.diag(alpha)), X2.T)
    return rbf_term + linear_term

def data_generation(gp_params, all_covs, big_n_rct, big_n_obs, n_MC, X_range, U_range, pasx, seed):
    gp_funcs = {}

    gp_funcs["om_A0"] = sample_outcome_model_gp(X_range, U_range, gp_params["om_A0_par"], seed + 0)  # GP - outcome model under treatment A=0
    gp_funcs["om_A1"] = sample_outcome_model_gp(X_range, U_range, gp_params["om_A1_par"], seed + 1)  # GP - outcome model under treatment A=1
    gp_funcs["w_sel"] = sample_outcome_model_gp(X_range, U_range, gp_params["w_sel_par"], seed + 2)  # GP - selection score model P(S=1|X)
    gp_funcs["w_trt"] = sample_outcome_model_gp(X_range, U_range, gp_params["w_trt_par"], seed + 3)  # GP - propensity score in OBS study P(A=1|X, S=2)

    SyntheticData = SyntheticDataModule(big_n_rct, big_n_obs, n_MC, gp_funcs, all_covs, X_range, U_range, pasx, seed + 4)
    mean_trail, _ = SyntheticData.get_true_mean()
    df_comp_big, df_obs = SyntheticData.get_obs_df() 

    return mean_trail, df_comp_big, df_obs

def sample_outcome_model_gp(X, U, param, seed):  # works with 2D covariates only

    np.random.seed(seed)
    XX, UU = np.meshgrid(X, U)
    XU_flat = np.c_[XX.ravel(), UU.ravel()]

    mean = np.zeros(len(XU_flat))

    if param["kernel"] == "rbf":
        K = rbf_linear_kernel(XU_flat, XU_flat, np.array(param["ls"]), np.array(param["alpha"]))

    f_sample = np.random.multivariate_normal(mean, K)
    Y = f_sample.reshape(XX.shape)

    # gp_func = interpolate.interp2d(X, U, Y, kind="linear")
    gp_func = interpolate.RectBivariateSpline(X, U, Y.T)

    return gp_func

def estimate_e(X, A, model_e):
    '''
        Estimate propensity score using a model_e
    '''
    e = model_e.fit(X, A).predict_proba(X)[:, 1]
    return e.reshape(-1, 1)

def estimate_mu(X, A, y, model_y):
    '''
        Estimate response function using a model_y
    '''
    train_data = np.concatenate((X, A), axis=1)
    mu = model_y.fit(train_data, y.reshape(-1, 1))
    test_0 = np.concatenate((X, np.zeros_like(A)), axis=1)
    test_1 = np.concatenate((X, np.ones_like(A)), axis=1)
    mu0 = mu.predict(test_0)
    mu1 = mu.predict(test_1)

    return mu0, mu1


def get_estimates(dataset_train, dataset_val, delta, significance_level = 0.05):

    '''Param settinig'''
    alpha = significance_level
    z_alpha = norm.ppf(1 - alpha/2)
    Y_train = np.stack(np.array(dataset_train['y']))
    A_train = np.array(dataset_train['A']).reshape(-1, 1)
    X_train = np.array(dataset_train['X']).reshape(-1, 1)
    n = Y_train.shape[0]
    d = X_train.shape[1]

    '''IPW Implementation test'''
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', booster='gblinear', n_estimators=100, learning_rate=0.1)
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1)

    e = estimate_e(X_train, A_train, xgb_clf)
    mu0, mu1 = estimate_mu(X_train, A_train, Y_train, xgb_reg)
    aipw = (A_train * Y_train / e - (1 - A_train) * Y_train / (1 - e)) - \
            ((A_train - e) / e * (1 - e)) * ((1-e) * mu1 + e * mu0)
    ate_est_aipw = np.mean(aipw)
    ate_ci_aipw = (ate_est_aipw - z_alpha * np.sqrt(np.var(aipw)/n), ate_est_aipw + z_alpha * np.sqrt(np.var(aipw)/n))
    print("var_aipw", np.var(aipw))
    print("ate_est_aipw", ate_est_aipw)
    print("ate_ci_aipw", ate_ci_aipw)

    '''Normal/Asymptotic setting: IPW/DR-learner'''
    est = LinearDRLearner(model_regression=GradientBoostingRegressor(),
                        model_propensity=GradientBoostingClassifier())
    y_train = Y_train.reshape(n)
    est.fit(y_train, A_train, X=X_train)
    cate_n = est.effect(X_train)
    ate_est_norm_trial = est.ate(X_train)
    ate_ci_norm_trial = est.ate_interval(X_train, alpha=alpha)
    print("ate_est_trial", ate_est_norm_trial)
    print("ate_ci_trial", ate_ci_norm_trial)

    '''Normal/Asymptotic setting + PPI '''
    N = np.stack(np.array(dataset_val['y'])).shape[0]
    N_train = int(N/2)
    N_eval = N - N_train
    Y_N_train = np.stack(np.array(dataset_val['y']))[:N_train, :]
    A_N_train = np.array(dataset_val['A']).reshape(-1, 1)[:N_train, :]
    X_N_train = np.array(dataset_val['X']).reshape(-1, 1)[:N_train, :]
    X_N_eval = np.array(dataset_val['X']).reshape(-1, 1)[N_train:, :]
    A_N_eval = np.array(dataset_val['A']).reshape(-1, 1)[N_train:, :]
    Y_N_eval = np.stack(np.array(dataset_val['y']))[N_train:, :]

    # est_2 = LinearDRLearner(model_regression=GradientBoostingRegressor(),
    #                 model_propensity=GradientBoostingClassifier())
    # y_N_train = Y_N_train.reshape(N_train)
    # est_2.fit(y_N_train, T_N_train, X=X_N_train)

    est_2 = LinearDRLearner(model_regression=GradientBoostingRegressor(), model_propensity=GradientBoostingClassifier())
    y_N_train = Y_N_train.reshape(N_train)
    est_2.fit(y_N_train, A_N_train, X=X_N_train)
    cate_N = est_2.effect(X_N_eval)
    ate_N = np.mean(cate_N)
    var_N = np.var(cate_N)

    cate_N = est_2.effect(X_N_eval)
    ate_N = np.mean(cate_N)
    var_N = np.var(cate_N)

    pred_n = est_2.effect(X_train).reshape(-1, 1)  
    mean_rectifier = np.mean(aipw - pred_n)
    var_rectifier = np.var(aipw - pred_n)

    ate_est_ppi = ate_N + mean_rectifier
    ate_ci_norm_ppi = (ate_est_ppi - z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval), \
                       ate_est_ppi + z_alpha * np.sqrt(var_rectifier/n + var_N/N_eval))
    print("var_ppi", var_rectifier)
    print("ate_est_ppi", ate_est_ppi)
    print("ate_ci_ppi", ate_ci_norm_ppi)

    '''Normal/Asymptotic setting: Observational data only'''
    ate_est_obs = ate_N
    ate_ci_obs = (ate_est_obs - z_alpha * np.sqrt(var_N/N_eval), \
                  ate_est_obs + z_alpha * np.sqrt(var_N/N_eval))
    print("ate_ci_obs", ate_ci_obs)

    return [ate_est_aipw, ate_est_norm_trial, ate_est_ppi, ate_est_obs], \
           [ate_ci_aipw, ate_ci_norm_trial, ate_ci_norm_ppi, ate_ci_obs]

def sim_cases(seed, df_rct, df_obs, significance_level, delta):

        ate_estimates, ate_ci = get_estimates(df_rct, df_obs, delta, significance_level)

        # plot_case_rmse(save_dir, case_idx, estimates, mu_a_gt)

        # consider how to set the streamline of the code
        # stat_bias_sq_est = np.mean(ate_estimates[-1] - mean_trail, axis=0) ** 2
        # stat_var_est = np.std(ate_estimates, axis=0) ** 2
        # mse = np.mean((ate_estimates[-1] - mean_trail) ** 2, axis=0)
        # rmse = np.sqrt(mse)        

        # return stat_bias_sq_est, stat_var_est, rmse, ate_est, ate_cf
        return ate_estimates, ate_ci