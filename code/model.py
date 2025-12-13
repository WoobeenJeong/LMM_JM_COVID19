import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import patsy

# -----------------------------------------------------------
# 3. Manual Linear Mixed Model Class (수정됨)
# -----------------------------------------------------------

class ManualLinearMixedModel:
    def __init__(self, formula, data, group_col):
        self.formula = formula
        self.data = data
        self.group_col = group_col
        
        # Design Matrix 생성
        self.y_df, self.X_df = patsy.dmatrices(formula, data, return_type='dataframe')
        # 1차원 배열로 변환 (중요)
        self.y = self.y_df.iloc[:, 0].values.flatten()
        self.X = self.X_df.values
        self.groups = data[group_col].values
        self.unique_groups = np.unique(self.groups)
        
        self.p = self.X.shape[1] # 파라미터 개수
        
        # 결과 저장용 변수 초기화
        self.beta_hat = None
        self.sigma_b_hat = None
        self.sigma_e_hat = None
        self.residuals = None
        self.logLik = None
        self.n_params = None

    def neg_log_likelihood(self, params):
        beta = params[:self.p]
        sigma_b = np.exp(params[self.p])
        sigma_e = np.exp(params[self.p + 1])
        
        nll = 0
        for grp in self.unique_groups:
            mask = (self.groups == grp)
            y_i = self.y[mask]
            X_i = self.X[mask]
            n_i = len(y_i)
            
            resid = y_i - X_i @ beta
            
            # V_i = sigma_b^2 * J + sigma_e^2 * I
            V_i = (sigma_b**2) * np.ones((n_i, n_i)) + (sigma_e**2) * np.eye(n_i)
            
            try:
                sign, logdet = np.linalg.slogdet(V_i)
                inv_V_resid = np.linalg.solve(V_i, resid)
                quad_form = resid.T @ inv_V_resid
                nll += 0.5 * (logdet + quad_form + n_i * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                return np.inf
        return nll

    def fit(self):
        # 초기값
        beta_init = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        initial_params = np.concatenate([beta_init, [0.0, 0.0]])
        
        # 최적화
        result = minimize(self.neg_log_likelihood, initial_params, method='L-BFGS-B') # BFGS 계열 사용
        
        # [수정된 부분] 최적화된 파라미터 저장
        self.opt_params = result.x
        self.beta_hat = self.opt_params[:self.p]
        self.sigma_b_hat = np.exp(self.opt_params[self.p])
        self.sigma_e_hat = np.exp(self.opt_params[self.p + 1])
        
        # [에러 해결의 핵심] 잔차(Residuals)와 LogLikelihood 계산 및 저장
        self.logLik = -result.fun
        self.n_params = len(result.x)
        
        # Marginal Fitted Values & Residuals
        self.fitted_values = (self.X @ self.beta_hat).flatten()
        self.residuals = (self.y - self.fitted_values).flatten() 
        
        return result

    def summary(self):
        print(f"Log-Likelihood: {-self.logLik:.2f}")
        print(f"Random Effect SD (Sigma_b): {self.sigma_b_hat:.4f}")
        print(f"Residual SD (Sigma_e): {self.sigma_e_hat:.4f}")

    def predict(self, newdata):
        _, X_new = patsy.dmatrices(self.formula, newdata, return_type='dataframe')
        return (X_new @ self.beta_hat).flatten()

# -----------------------------------------------------------
# 4. Manual Joint Model Class
# -----------------------------------------------------------

class ManualJointModel:
    def __init__(self, models_dict, data, group_col):
        self.models_dict = models_dict
        self.data = data
        self.group_col = group_col
        self.fitted_models = {}
        # 빈 DataFrame 생성
        self.residual_df = pd.DataFrame(index=data.index)
        
    def fit(self):
        # print("\nFitting individual LMMs...")
        for name, formula in self.models_dict.items():
            # print(f"  - Fitting {name}...")
            # 위에서 정의한 ManualLinearMixedModel 호출
            lmm = ManualLinearMixedModel(formula, self.data, self.group_col)
            lmm.fit()
            
            self.fitted_models[name] = lmm
            # 계산된 잔차 저장 (이제 에러가 발생하지 않음)
            self.residual_df[name] = lmm.residuals
            
        # print("Calculating Joint Statistics...")
        self.calc_residual_covariance()
        self.calc_metrics()
        
    def calc_residual_covariance(self):
        self.res_cov_matrix = self.residual_df.cov()
        self.res_corr_matrix = self.residual_df.corr()
        
    def calc_metrics(self):
        self.iccs = {}
        total_loglik = 0
        total_params = 0
        
        for name, model in self.fitted_models.items():
            rand_var = model.sigma_b_hat**2
            resid_var = model.sigma_e_hat**2
            
            icc = rand_var / (rand_var + resid_var)
            self.iccs[name] = {
                'Random_Var': rand_var,
                'Residual_Var': resid_var,
                'ICC': icc
            }
            total_loglik += model.logLik
            total_params += model.n_params

        # AIC/BIC Approximation
        k = len(self.fitted_models)
        n_corr_params = k * (k - 1) / 2
        self.AIC = 2 * (total_params + n_corr_params) - 2 * total_loglik
        self.BIC = (total_params + n_corr_params) * np.log(len(self.data)) - 2 * total_loglik
        
    def summary(self):
        print("\n" + "="*50)
        print("   MANUAL JOINT MODEL RESULTS")
        print("="*50)
        
        print("\n1. Variance Components & ICC")
        print(f"{'Outcome':<10} | {'Random Var':<12} | {'Residual Var':<12} | {'ICC':<8}")
        print("-" * 50)
        for name, res in self.iccs.items():
            print(f"{name:<10} | {res['Random_Var']:.4f}       | {res['Residual_Var']:.4f}       | {res['ICC']:.4f}")
            
        print("\n2. Residual Correlation Matrix")
        print(self.res_corr_matrix)
        
        print(f"\n3. Model Fit Statistics")
        print(f"   AIC: {self.AIC:.2f}")
        print(f"   BIC: {self.BIC:.2f}")