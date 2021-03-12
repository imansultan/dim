

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit,ParameterGrid,KFold
from joblib import Parallel, delayed
import multiprocessing
import sys
sys.path.append('..')
from spam import model
#import os
#import shutil

def build_job_func(iteration,param_grid_values_list,kf,verbose_cv,
                   X_data_training,y_training,max_backfit_iter,
                   prediction_f_init_value,verbose_training):
    param_grid_value=param_grid_values_list[iteration]
    alpha_cv_value=param_grid_value['alpha']
    smooth_factor_cv_value=param_grid_value['spline_smooth_factor']
    list_mse_val=list()
    list_mse_train=list()
    if verbose_cv:
        print((param_grid_values_list[iteration]))
    for train_index, val_index in kf.split(X_data_training):
        X_train, X_val = X_data_training[list(train_index),:], X_data_training[list(val_index),:]
        y_train, y_val = y_training[list(train_index)], y_training[list(val_index)]

        spam_instance=model.SpamModel(alpha=alpha_cv_value,max_iter=max_backfit_iter,
                                      prediction_f_init=prediction_f_init_value,
                                      spline_smooth=smooth_factor_cv_value,
                                      verbose=verbose_training)
        prediction_f_train,y_train_pred=spam_instance.fit(X_train,y_train)
        prediction_f_val,y_val_pred=spam_instance.predict(X_val)

        mse_train=mean_squared_error(y_train,y_train_pred)
        mse_val=mean_squared_error(y_val,y_val_pred)
        list_mse_train.append(mse_train)
        list_mse_val.append(mse_val)
    avg_mse_train_cv=np.average(list_mse_train)
    avg_mse_val_cv=np.average(list_mse_val)

    return [alpha_cv_value,smooth_factor_cv_value,avg_mse_train_cv,avg_mse_val_cv]

class ModelSelector:

    def __init__(self,smooth_factor_grid,alpha_grid,
                 X_data_training,y_training,
                 max_backfit_iter,num_jobs_factor,
                 cv_folds):

        self.smooth_factor_grid=smooth_factor_grid
        self.alpha_grid=alpha_grid
        self.X_data_training=X_data_training
        self.y_training=y_training
        self.max_backfit_iter=max_backfit_iter
        self.num_jobs_factor=num_jobs_factor
        self.cv_folds_num=cv_folds

    def get_optimal_cv_parameters(self,prediction_f_init_value,
                                  verbose_training=False,
                                  verbose_cv=True):
        self.prediction_f_init_value=prediction_f_init_value
        self.verbose_training=verbose_training
        self.verbose_cv=verbose_cv

        param_grid_dict=dict(spline_smooth_factor=list(self.smooth_factor_grid),
                             alpha=list(self.alpha_grid))
        param_grid_values_list=list(ParameterGrid(param_grid_dict))
        kf = KFold(n_splits=self.cv_folds_num,shuffle=True,random_state=256)

        num_cpus = multiprocessing.cpu_count()
        number_of_jobs=num_cpus*self.num_jobs_factor
        pool = Parallel(n_jobs=number_of_jobs,verbose=0,pre_dispatch='all')
        print(("Seeding %i training jobs in parallel" %(number_of_jobs)))
        cv_results_list=pool((delayed(build_job_func)(iteration_counter,param_grid_values_list,
              kf,self.verbose_cv,self.X_data_training,self.y_training,self.max_backfit_iter,
              self.prediction_f_init_value,self.verbose_training) for iteration_counter in range(0,len(param_grid_values_list))))
        cv_results=np.array(cv_results_list)
        mse_cv_train_per_parameter=cv_results[:,2]
        mse_cv_val_per_parameter=cv_results[:,3]
        min_val_mse_idx=np.argmin(mse_cv_val_per_parameter)
        best_parameters={'alpha':cv_results[min_val_mse_idx,0],'spline_smooth_factor':cv_results[min_val_mse_idx,1]}
        diagnostics={'param_grid_values_list':param_grid_values_list,
                     'mse_cv_train_per_parameter':mse_cv_train_per_parameter,
                     'mse_cv_val_per_parameter':mse_cv_val_per_parameter,
                     'best_mse_train':mse_cv_train_per_parameter[min_val_mse_idx],
                     'best_mse_val':mse_cv_val_per_parameter[min_val_mse_idx]}

        return best_parameters,diagnostics

    def run_stability_selection(self,test_size_ratio,num_shuffles,stab_gap):

        num_true_features=np.size(self.X_data_training,axis=1)
        rs = ShuffleSplit(n_splits=num_shuffles, test_size=test_size_ratio, random_state=256)
        stability_counts_array=np.zeros((num_shuffles,num_true_features),dtype=float)
        shuffles_counter=0

        pred_f_train_3d=np.empty((np.size(self.X_data_training,axis=0),
                                  np.size(self.X_data_training,axis=1),
                                  num_shuffles),dtype=None)
        pred_f_train_3d[:,:,:,]=np.nan
        y_2d_training=np.empty((np.size(self.X_data_training,axis=0),num_shuffles),dtype=None)
        y_2d_training[:,:]=np.nan
        y_2d_test=np.empty((np.size(self.X_data_training,axis=0),num_shuffles),dtype=None)
        y_2d_test[:,:]=np.nan
        diagnostics_list=list()
        diagnostics_cv_list=list()
        best_parameters_cv_list=list()
        for train_index, test_index in rs.split(self.X_data_training):
            print(("Shuffle %i out of %i" %(shuffles_counter+1,num_shuffles)))

            train_index_list=list(train_index)
            test_index_list=list(test_index)
            X_train = self.X_data_training[train_index_list,:]
            y_train = self.y_training[train_index_list]
            X_test = self.X_data_training[test_index_list,:]
            y_test = self.y_training[test_index_list]

            best_parameters_cv,diagnostics_cv=self.get_optimal_cv_parameters(prediction_f_init_value=None,
                                                                verbose_training=False,verbose_cv=False)
            diagnostics_cv_list.append(diagnostics_cv)
            best_parameters_cv_list.append([best_parameters_cv['alpha'],best_parameters_cv['spline_smooth_factor']])

            spam_instance=model.SpamModel(alpha=best_parameters_cv['alpha'],
                                          max_iter=self.max_backfit_iter,
                                          prediction_f_init=None,
                                          spline_smooth=best_parameters_cv['spline_smooth_factor'],
                                          verbose=False)
            prediction_f_train,y_train_pred=spam_instance.fit(X_train,y_train)
            prediction_f_test,y_test_pred=spam_instance.predict(X_test)

            rmse_test=np.round(np.sqrt(mean_squared_error(y_test,y_test_pred)),2)
            rmse_train=np.round(np.sqrt(mean_squared_error(y_train,y_train_pred)),2)
            num_feat_sel_train=np.count_nonzero(np.sum(np.abs(prediction_f_train),0))
            num_feat_sel_test=np.count_nonzero(np.sum(np.abs(prediction_f_test),0))
            diagnostics=[rmse_train,rmse_test,num_feat_sel_train,num_feat_sel_test]
            diagnostics_list.append(diagnostics)

            pred_f_train_3d[train_index_list,:,shuffles_counter]=prediction_f_train
            y_2d_training[train_index_list,shuffles_counter]=y_train_pred
            y_2d_test[test_index_list,shuffles_counter]=y_test_pred
            pseudo_feature_weights=np.transpose(np.sum(np.abs(prediction_f_train),axis=0))
            nonzero_feat_counts=np.copy(pseudo_feature_weights)
            nonzero_feat_counts[nonzero_feat_counts!=0]=1
            stability_counts_array[shuffles_counter,:]=np.asarray(np.transpose(nonzero_feat_counts),dtype=float)
            shuffles_counter+=1

        self.diagnostics_ss_array=np.array(diagnostics_list)
        self.best_parameters_cv_array=np.array(best_parameters_cv_list)

        stability_values=np.sum(stability_counts_array,axis=0)/num_shuffles
#        directory_save='dataset/output_data/data_frames/'
#        shutil.rmtree(directory_save)
#        os.makedirs(directory_save)
#        np.savetxt(directory_save+'stability_values.txt',stability_values,fmt='%.2f',
#                   delimiter='\t')
        max_stab=np.max(stability_values)
        stab_threshold=max_stab-stab_gap
        stable_feat_indices=list(np.where(stability_values>=stab_threshold)[0])
        self.stability_values=stability_values

        return stable_feat_indices,pred_f_train_3d,y_2d_training,y_2d_test