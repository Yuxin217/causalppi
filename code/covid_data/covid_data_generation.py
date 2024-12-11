import sys
import numpy as np
import pandas as pd
from scipy.special import expit
# from importlib import reload
from sklearn.linear_model import LogisticRegression, LinearRegression


class CovidDataModule:
    def __init__(self,
                gp_params,
                n_rct=200,
                n_obs=10000,
                read_in_dir = '',
                target_col = 'Region',
                seed=42):
        
        self.n_rct = n_rct
        self.n_obs = 2*n_obs
        self.seed = seed
        self.target_col = target_col
        if target_col == 'Region':
            self.x_col = 'Ethnicity'
        else:
            self.target_col = 'Ethnicity'
            self.x_col = 'Region'
        self.X_columns = ['Age', 'Sex'] + [self.x_col]
        self.comorbidity_columns = ['Cardiovascular', 'Asthma', 'Diabetis', 'Pulmonary', 'Immunosuppresion', 'Obesity', 'Liver', 'Neurologic', 'Renal']

        self.gp_params = gp_params
        self.om_A0_params = self.gp_params['om_A0_par']   # GP - outcome model under treatment A=0
        self.om_A1_params = self.gp_params['om_A1_par']   # GP - outcome model under treatment A=1
        self.w_trt_params = self.gp_params['w_trt_par']   # GP for the propensity score in OBS study P(A=1 | X, S=2)
        
        self.read_in_dir = read_in_dir


    def save_processed_data(self):
        self.data = pd.read_csv(sys.path[-1] + self.read_in_dir + 'covid_normalised_numericalised.csv')
        self.original_n = len(self.data)

        df = pd.DataFrame()
        df['comorbidity'] = self.data[self.comorbidity_columns].max(1).values
        df[self.X_columns] = self.data[self.X_columns]
        if self.target_col == 'Region':
            df[self.target_col] = np.where(self.data[self.target_col].isin([1, 2]), 1, 0)
        else:
            df[self.target_col] = np.where(self.data[self.target_col].isin([1]), 1, 0)
        
        df['U'] = 2 * np.random.rand(len(df), 1) - 1  # Uniform[-1,1]
        df['y0'] = self.kernel_sample(np.array(df[self.X_columns + [self.x_col]]), np.array(df['U']).reshape(-1, 1), self.om_A0_params)
        print("done y0")
        df['y1'] = self.kernel_sample(np.array(df[self.X_columns + [self.x_col]]), np.array(df['U']).reshape(-1, 1), self.om_A1_params)
        print("done y1")
        
        
        prob_A = self.kernel_sample(np.array(df[self.X_columns + [self.x_col]]), np.array(df['U']).reshape(-1, 1), self.w_trt_params)
        print("done a")

        df["P(A=1|X)"] = np.clip(expit(prob_A), 0.1, 0.9)
        df["A"] = np.array(df["P(A=1|X)"] > np.random.uniform(size=self.original_n), dtype=int)
        
        df['y'] = df['y1'] * df['A'] + df['y0'] * (1 - df['A'])
        self.df = df.drop(columns=["P(A=1|X)", "U", "y1", "y0"])

        self.df.to_csv(sys.path[-1] + self.read_in_dir + f'covid_pre_processed_data_{self.target_col}.csv')

        return "Data processed and saved"


    def _generate_data(self, data_get):
        np.random.seed(self.seed + 1)
        if self.target_col == 'Region':
            self.target_col = 'region'

        df_rct_origin = data_get[data_get[self.target_col] == 1]
        df_rct = df_rct_origin[self.X_columns + ['comorbidity', 'A', 'y']].sample(self.n_rct, replace=True)

        return df_rct
    
    def _generate_data_obs(self, data_get):

        '''Initiate dataframe'''
        np.random.seed(self.seed + 2)
        
        df_obs_origin = data_get[data_get[self.target_col] == 0]
        df_obs = df_obs_origin[self.X_columns + ['comorbidity', 'A', 'y']].sample(self.n_obs, replace=True)

        return df_obs

    def get_df(self):
        np.random.seed(self.seed)
        data_get = pd.read_csv(sys.path[-1] + self.read_in_dir + f'covid_pre_processed_data_{self.target_col}.csv')
        self.df  = self._generate_data(data_get)
        self.df_obs = self._generate_data_obs(data_get)

        return self.df.copy(), self.df_obs.copy()
    
    def rbf_linear_kernel(self, X1, X2, length_scales=np.array([0.1,0.1]), alpha=np.array([0.1,0.1]), var=1):  # works with 2D covariates only
        X1 = X1 / length_scales
        X2 = X2 / length_scales

        distances =  -2.0 * np.dot(X1, X2.T) + np.dot(X1, X1.T) + np.dot(X2, X2.T)

        rbf_term = var * np.exp(-0.5 * distances**2)
        linear_term = alpha[-1] * np.dot(X2, X2.T)
        return alpha[0] * rbf_term + alpha[-1] * linear_term

    
    def kernel_sample(self, X, U, gp_params):

        np.random.seed(self.seed)
        XU = np.concatenate((X, U), axis=1)
        mean = np.zeros(len(XU))

        if gp_params["kernel"] == "rbf":
            K = self.rbf_linear_kernel(XU, XU, np.array(gp_params["ls"]), np.array(gp_params["alpha"]))

        f_sample = np.random.multivariate_normal(mean, K)
        prob_a = f_sample.reshape(U.shape)

        return prob_a

    def get_true_mean(self, n_MC=1000):
        data_get = pd.read_csv(sys.path[0] + f"/{self.read_in_dir}/covid_pre_processed_data_{self.target_col}.csv")
        if self.target_col == 'Region':
            self.target_col = 'region'

        df_rct_origin = data_get[data_get[self.target_col] == 1]
        df_rct = df_rct_origin[self.X_columns + ['comorbidity', 'A', 'y']].sample(n_MC, replace=True)
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
    covid_data = CovidDataModule(gp_params, 200, 10000, 'data_source', 'Ethnics', 42)
    print(covid_data.get_true_mean(n_MC=10000))
