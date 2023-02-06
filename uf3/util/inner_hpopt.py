import numpy as np
import pandas as pd
from uf3.representation import bspline
from uf3.regression import least_squares


class inner_hp_opt:
         
    def __init__(self, df_data):
        self.df_data = df_data
        self.get_training_keys()
        self.get_std()

    def get_training_keys(self, test_prefix = "test"):
        print(self.df_data)
        training_keys = [k for k in self.df_data.index if test_prefix not in k]
        if len(training_keys)==0:
            raise ValueError("The list is empty \n input correct prefix for test") 
        self.training_keys = training_keys    
    
    def get_std(self):
        dk = self.df_data.loc[self.training_keys]
        f1 = np.concatenate(dk["fx"].to_numpy().flatten())
        f2 = np.concatenate(dk["fy"].to_numpy().flatten())
        f3 = np.concatenate(dk["fz"].to_numpy().flatten())
        force = np.concatenate((f1, f2, f3))
        force_std = np.std(force)
        energy = dk["energy"] #.to_numpy().flatten()
        energy_std = np.std(energy)
        self.force_std = force_std
        self.energy_std = energy_std
        self.energy = energy
        self.force  = force

    def split_data_cv(self, kfold):
        training_indices = np.array([np.where(self.df_data.index == key)[0][0] for key in self.training_keys])
        np.random.shuffle(training_indices)
        cv_split = np.array_split(training_indices, kfold)
        self.cv_split = cv_split
 
    def cost_fn(self, **kwargs):
        cost_weight = kwargs.get("cost_weight", 0.5)
        rmse_e = kwargs.get("rmse_e")
        rmse_f = kwargs.get("rmse_f")
        curve_2b = kwargs.get("curve_2b", 1e-6)
        curve_3b = kwargs.get("curve_3b", 1e-6)
        ridge_3b = kwargs.get("ridge_3b", 1e-8)
        lamda = kwargs.get("lamda", 0.00)
        cost_fn = (
                ((cost_weight * rmse_e) / ((self.energy_std)))
                + (((1 - cost_weight) * rmse_f) / ((self.force_std)))
                - (lamda
                * (np.log10(curve_2b) + np.log10(curve_3b) + np.log10(ridge_3b)))
            )
        return cost_fn
# lambda

    def make_grid(self, **kwargs):
        ef_weight = kwargs.get("ef_weight", [0.5])
        lamda = kwargs.get("lamda", [1e-6])
        cost_weight = kwargs.get("cost_weight", [0.5])
        ridge_1b = kwargs.get("ridge_1b", [1e-8])
        ridge_2b = kwargs.get("ridge_2b", [1e-8])
        curve_2b = kwargs.get("curve_2b", [1e-6])
        curve_3b = kwargs.get("curve_3b", [1e-6])
        ridge_3b = kwargs.get("ridge_3b", [1e-8])

        ef_weight, lamda, cost_weight, ridge_1b, ridge_2b, ridge_3b, curve_2b, curve_3b  = np.meshgrid(
            ef_weight, lamda, cost_weight, ridge_1b, ridge_2b, ridge_3b, curve_2b, curve_3b
        )

        ef_weight = ef_weight.flatten()
        cost_weight = cost_weight.flatten()
        lamda = lamda.flatten()
        ridge_1b = ridge_1b.flatten()
        ridge_2b = ridge_2b.flatten()
        ridge_3b = ridge_3b.flatten()
        curve_2b = curve_2b.flatten()
        curve_3b = curve_3b.flatten()
        
        self.ef_weight = ef_weight
        self.cost_weight = cost_weight
        self.lamda = lamda
        self.ridge_1b = ridge_1b
        self.ridge_2b = ridge_2b
        self.ridge_3b = ridge_3b
        self.curve_2b = curve_2b
        self.curve_3b = curve_3b


    def inner_hp(self, filename, bspline_config):
        
        all_hp = {}

        for i in range(len(self.curve_3b)):
            params = dict(
                ridge_1b=self.ridge_1b[i],
                ridge_2b=self.ridge_2b[i],
                ridge_3b=self.ridge_3b[i],
                curvature_2b=self.curve_2b[i],
                curvature_3b=self.curve_3b[i],
            )

            regularizer = bspline_config.get_regularization_matrix(**params)

            model = least_squares.WeightedLinearModel(
                bspline_config, regularizer=regularizer
            )

            cost_dict = {}
            for j in range(len(self.cv_split)):
                # do the cross validation here
                val_set = self.cv_split[j]
                train_set = np.delete(self.cv_split, j)
                # join the train set
                train_set = np.concatenate(train_set)
                model.fit_from_file(
                    filename,
                    self.df_data.index[train_set],
                    weight=self.ef_weight[i],
                    batch_size=150,
                    energy_key="energy",
                    progress="none",
                )

                y_e, p_e, y_f, p_f, rmse_e, rmse_f = model.batched_predict(
                    filename, keys=self.df_data.index[val_set]
                )

                cost_para = dict(
                    cost_weight=self.cost_weight[i],
                    lamda=self.lamda[i],
                    rmse_e=rmse_e,
                    rmse_f=rmse_f,
                    ridge_3b=self.ridge_3b[i],
                    curve_2b=self.curve_2b[i],
                    curve_3b=self.curve_3b[i],
                )
                
                cost = self.cost_fn(**cost_para)
                cost_dict.setdefault(i, []).append(cost)

            cost_cv = np.mean(cost_dict[i])
            all_hp[i] = {
                'cost': cost_cv,
                'cost_weight': self.cost_weight[i],
                'ef_weight': self.ef_weight[i],
                'lamda': self.lamda[i],
                'regularizers': params,
                }

        all_hp = pd.DataFrame.from_dict(all_hp, orient="index")
        best_hyperparameter = all_hp.loc[all_hp['cost'].idxmin()]
        self.best_hyperparameter = best_hyperparameter
        self.all_hp = all_hp
        return best_hyperparameter, all_hp