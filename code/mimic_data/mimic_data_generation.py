import sys
import numpy as np
import pandas as pd
from scipy.special import expit
# from importlib import reload
from sklearn.linear_model import LogisticRegression, LinearRegression


class MIMICDataModule:
    def __init__(self,
                gp_params,
                n_rct=200,
                n_obs=10000,
                read_in_dir = '/mimic_data/data_source/',
                seed=42,):
        
        self.n_rct = n_rct
        self.n_obs = n_obs
        self.seed = seed
        self.X_columns = ['gender',	'age', 'heart rate', 'sodium', 'blood cells', 'glucose', 'hematocrit', 'respiratory rate']

        self.gp_params = gp_params
        self.w_trt_params = self.gp_params['w_trt_par']   # GP for the propensity score in OBS study P(A=1 | X, S=2)
    
        self.read_in_source_data = pd.read_csv(sys.path[0] + read_in_dir + 'mimic_data_train.csv')

        std_info = pd.read_csv(sys.path[0] + read_in_dir + 'mimic_std_information.csv')
        self.df_denormalize = pd.DataFrame()
        self.df_denormalize['gender'] = self.read_in_source_data['gender']
        self.df_denormalize['age'] = self.read_in_source_data['age']
        for col in ['heart rate', 'sodium', 'blood cells', 'glucose', 'hematocrit', 'respiratory rate']:
            self.df_denormalize[col] = self.read_in_source_data[col] * std_info[col][1] ** 2 + std_info[col][0] 
        self.df_denormalize['y'] = self.read_in_source_data['Y']* std_info['Y'][1] ** 2 + std_info['Y'][0] 
        self.df_denormalize['A'] = self.read_in_source_data['A']
        self.original_n = len(self.read_in_source_data)

    def _generate_data(self):
        np.random.seed(self.seed + 1)
        frac = self.n_rct/self.original_n

        df_rct = self.df_denormalize.sample(frac=frac, random_state=self.seed+1).reset_index(drop=True)

        return df_rct
    
    def _generate_data_obs(self):

        '''Initiate dataframe'''
        df_obs = pd.DataFrame()
        np.random.seed(self.seed + 2)

        df_obs[self.X_columns + ['A', 'y']] = self.df_denormalize[self.X_columns + ['A', 'y']].sample(self.n_obs, replace=True)

        return df_obs

    def get_df(self):
        np.random.seed(self.seed)
        self.df  = self._generate_data()
        self.df_obs = self._generate_data_obs()

        return self.df.copy(), self.df_obs.copy()
    
    def rbf_linear_kernel(self, X1, X2, length_scales=np.array([0.1,0.1]), alpha=np.array([0.1,0.1]), var=1):  # works with 2D covariates only
        X1 = X1 / length_scales
        X2 = X2 / length_scales
        
        distances =  -2.0 * np.dot(X1, X2.T) + np.dot(X1, X1.T) + np.dot(X2, X2.T)
        rbf_term = var * np.exp(-0.5 * distances**2)
        linear_term = alpha[-1] * np.dot(X2, X2.T)
        return rbf_term + linear_term

    
    def kernel_sample(self, X, U):
        np.random.seed(self.seed)
        XU = np.concatenate((X, U), axis=1)

        mean = np.zeros(len(XU))

        if self.w_trt_params["kernel"] == "rbf":
            K = self.rbf_linear_kernel(XU, XU, np.array(self.w_trt_params["ls"]), np.array(self.w_trt_params["alpha"]))

        f_sample = np.random.multivariate_normal(mean, K)
        prob_a = f_sample.reshape(U.shape)

        return prob_a

    def get_true_mean(self):
        true_ate = 0
        std_ate = 0
        return true_ate, std_ate

    def get_true_mean(self, n_MC=1000):

        df_rct = self.df_denormalize[self.X_columns + ['A', 'y']].sample(n_MC, replace=True)
        y0 = df_rct[df_rct['A'] == 0]['y']
        y1 = df_rct[df_rct['A'] == 1]['y']

        true_ate = np.mean(y1) - np.mean(y0)
        return true_ate
    
if __name__ == "__main__":
    gp_params = {
        "om_A0_par": {"kernel": "rbf", "ls": [1,1,1,1,1], "alpha": [1,1]},
        "om_A1_par": {"kernel": "rbf", "ls": [0.5, 0.5, 0.5, 0.5, 0.5], "alpha": [5, 5]},
        "w_trt_par": {"kernel": "rbf", "ls":  [1000000, 1000000, 1000000, 1000000, 1000000], "alpha": [0.5, 0]}
    }

    mimic_data = MIMICDataModule(gp_params, 200, 10000, '/data_source/', 42)
    print(mimic_data.get_true_mean(n_MC=10000))