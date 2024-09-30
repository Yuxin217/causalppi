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
        # self.n_MC = n_MC 
        self.seed = seed
        self.X_columns = ['gender',	'age', 'heart rate', 'sodium', 'blood cells', 'glucose', 'hematocrit', 'respiratory rate']

        self.gp_params = gp_params
        self.w_trt_params = self.gp_params['w_trt_par']   # GP for the propensity score in OBS study P(A=1 | X, S=2)
        # self.prop_clip_lb = pasx["lb"]  #  exclude patients whose probability of treatment is < 0.1 
        # self.prop_clip_ub = pasx["ub"]  #  exclude patients whose probability of treatment is > 0.9
    
        self.read_in_source_data = pd.read_csv(sys.path[0] + read_in_dir + 'mimic_data_train.csv')
        # self.read_in_source_data = pd.read_csv(sys.path[-1] + read_in_dir + 'mimic_data_train.csv')

        std_info = pd.read_csv(sys.path[0] + read_in_dir + 'mimic_std_information.csv')
        # std_info = pd.read_csv(sys.path[-1] + read_in_dir + 'mimic_std_information.csv')
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
        # df = pd.DataFrame()
        # df[self.X_columns] = self.df_denormalize[self.X_columns]
        # df["A"] = self.df_denormalize['A']
        # df["y"] = self.df_denormalize['y']

        df_rct = self.df_denormalize.sample(frac=frac, random_state=self.seed+1).reset_index(drop=True)

        return df_rct
    
    def _generate_data_obs(self):

        '''Initiate dataframe'''
        df_obs = pd.DataFrame()
        np.random.seed(self.seed + 2)
        # df_obs[self.X_columns] = self.df[self.X_columns].sample(self.n_obs, replace=True)

        # frac = self.n_obs/self.original_n
        # df_obs = pd.DataFrame()
        # df_obs[self.X_columns] = self.df_denormalize[self.X_columns]
        # df["A"] = self.df_denormalize['A']
        # df["y"] = self.df_denormalize['y']

        # df_obs = self.df_denormalize.sample(frac=frac, random_state=self.seed+1).reset_index(drop=True)

        df_obs[self.X_columns + ['A', 'y']] = self.df_denormalize[self.X_columns + ['A', 'y']].sample(self.n_obs, replace=True)

        return df_obs
    
        # # '''fit and smaple propensity score'''
        # # df_obs[self.X_columns] = self.df[self.X_columns].sample(self.n_obs, replace=True)
        # # df_obs['U'] = 2 * np.random.rand(self.n_obs, 1) - 1  # Uniform[-1,1]
        # # prob_A = self.kernel_sample(np.array(df_obs[self.X_columns]), np.array(df_obs['U']).reshape(-1, 1))
        # # df_obs["P(A=1|X)"] = np.clip(expit(prob_A), self.prop_clip_lb, self.prop_clip_ub)
        # # df_obs["A"] = np.array(df_obs["P(A=1|X)"] > np.random.uniform(size=self.n_obs), dtype=int)


        # model_e = LogisticRegression().fit(np.array(self.read_in_source_data[self.X_columns]), np.array(self.read_in_source_data["assign"]))
        # prob_e = model_e.predict_proba(np.array(df_obs[self.X_columns]))[:, 1]
        # # prob_y = np.clip(expit(prob_y), self.prop_clip_lb, self.prop_clip_ub)
        # df_obs["A"] = np.array(prob_e > np.random.uniform(size=self.n_obs), dtype=int)
        
        # # '''fit and smaple y'''
        # train_data = np.concatenate((np.array(self.read_in_source_data[self.X_columns]), np.array(self.read_in_source_data["assign"]).reshape(-1, 1)), axis=1)
        # model_y = LogisticRegression().fit(train_data, np.array(self.read_in_source_data["outcome"]))
        # test_data = np.concatenate((np.array(df_obs[self.X_columns]), np.array(df_obs["A"]).reshape(-1, 1)), axis=1)
        # prob_y = model_y.predict_proba(test_data)[:, 1]
        # # prob_y = np.clip(expit(prob_y), self.prop_clip_lb, self.prop_clip_ub)
        # df_obs["y"] = np.array(prob_y> np.random.uniform(size=self.n_obs), dtype=int)

        # # df_obs = df_obs.drop(columns=["P(A=1|X)", "U"])
        
        # return df_obs

    def get_df(self):
        np.random.seed(self.seed)
        self.df  = self._generate_data()
        self.df_obs = self._generate_data_obs()

        return self.df.copy(), self.df_obs.copy()
    
    def rbf_linear_kernel(self, X1, X2, length_scales=np.array([0.1,0.1]), alpha=np.array([0.1,0.1]), var=1):  # works with 2D covariates only
        X1 = X1 / length_scales
        X2 = X2 / length_scales
        
        # X1_norm2 = np.sum(np.dot(X1, X1.T), axis = 1).reshape(-1, 1)
        # X2_norm2 = np.sum(np.dot(X2, X2.T), axis = 1).reshape(-1, 1)

        # X1_norm2 = np.linalg.norm(X1, ord = 2, axis = 1).reshape(-1, 1)
        # X2_norm2 = np.linalg.norm(X2, ord = 2, axis = 1).reshape(-1, 1)

        # distances =  -2.0 * np.dot(X1, X2.T) + np.tile(X1_norm2, (1, X2.shape[0])) + np.tile(X2_norm2.T, (X1.shape[0], 1))
        distances =  -2.0 * np.dot(X1, X2.T) + np.dot(X1, X1.T) + np.dot(X2, X2.T)

        # distances = np.linalg.norm((X1[:, None, :] - X2[None, :, :]) / length_scales, axis=2)
        rbf_term = var * np.exp(-0.5 * distances**2)
        # linear_term = np.dot(np.dot(X1, np.diag(alpha)), X2.T)
        linear_term = alpha[-1] * np.dot(X2, X2.T)
        return rbf_term + linear_term

    
    def kernel_sample(self, X, U):
        np.random.seed(self.seed)
        XU = np.concatenate((X, U), axis=1)
        # XX, UU = np.meshgrid(X, U)
        # XU_flat = np.c_[XX.ravel(), UU.ravel()]

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
    # print(covid_data.save_processed_data())
    print(mimic_data.get_true_mean(n_MC=10000))