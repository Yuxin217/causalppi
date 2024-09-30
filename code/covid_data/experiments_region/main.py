import numpy as np
import pandas as pd
import os, json, itertools, sys

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from covid_data.run_utils import *

def run_experiment(config_run, scenario_name, sample_name, seed_list):

    alpha = config_run['data']['alpha']
    delta = config_run['data']['delta']

    data_seed = config_run['data']['data_seed']
    n_rct_param = config_run['sample_settings'][sample_name]['n_rct']
    n_rct_list = [n_rct_param[0]+i*n_rct_param[1] for i in range(n_rct_param[2])]
    big_n_rct = sum(n_rct_list)
    n_obs_param = config_run['sample_settings'][sample_name]['n_obs']
    n_obs_list = [n_obs_param[0]+i*n_obs_param[1] for i in range(n_obs_param[2])]
    big_n_obs = sum(n_obs_list)
    # n_MC = config_run['data']['n_MC']

    pasx = config_run['pasx']
    gp_params = config_run['data_gen'][scenario_name]

    name_list = ["normal_aipw", "normal_trial", "normal_ppi", "normal_obs"]

    for seed in seed_list:
        df = {}
        df['n_rct'] = []
        df['n_obs'] = []

        mean_trail, big_df_rct, big_df_obs = data_generation(gp_params, big_n_rct, big_n_obs, data_seed + seed)

        for i in range(len(name_list)):
            df[name_list[i] + "_est"] = []
            df[name_list[i] + "_width"] = []

        for iter in range(len(n_rct_list)):
            n_rct = n_rct_list[iter]
            n_obs = n_obs_list[iter]
            df_rct = big_df_rct.iloc[sum(n_rct_list[:iter]) : sum(n_rct_list[:iter+1])]
            df_obs = big_df_obs.iloc[sum(n_obs_list[:iter]) : sum(n_obs_list[:iter+1])]
            print(f"Start: {iter+1}-th iteration, small data size: {n_rct}, large data size: {n_obs}.")  

            ate_est, ate_ci = sim_cases(seed, df_rct, df_obs, alpha, delta)
            
            for j in range(len(name_list)):
                df[name_list[j] + "_est"].append(ate_est[j])
                df[name_list[j] + "_width"].append(0.5 * (ate_ci[j][1] - ate_ci[j][0]))

            # print(f"done with {iter+1}-th iteration")
            df['n_rct'].append(n_rct)
            df['n_obs'].append(n_obs)

        df["true_ate"] = [mean_trail for i in range(iter+1)]

        relative_path = config_run["relative_path"]
        save_dir = f"{relative_path}/exp_results/{scenario_name}/{sample_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        estimates = pd.DataFrame(df)
        estimates.to_csv(f"{save_dir}/estimates_{seed}.csv")
   

if __name__ == "__main__":
    # seed_list = [0, 1, 2, 6, 7]
    seed_list = [1, 2, 3, 4, 5]
    scenario_list = ["scenario_1"]
    sample_list = ["small_n", "ratio", "large_N"]
    # sample_list = ["small_n"]

    relative_path = f"/covid_data/experiments_region"
    config = load_yaml(f"{relative_path}/config")

    for scenario_name, sample_name in itertools.product(scenario_list, sample_list):
        run_experiment(config, scenario_name, sample_name, seed_list)
        print(f"Done with {scenario_name} and {sample_name}.")
