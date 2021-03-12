import matplotlib
matplotlib.use('Agg')

import numpy as np
import pylab as pl
import os
import sys
import pandas as pd
sys.path.append('..')
from spam import model
import shutil

class BuildPlots:

    def __init__(self,num_shuffles,test_size_ratio,feature_names,
                 stable_feat_indices,y_train,X_train,n_bins):
        self.num_shuffles=num_shuffles
        self.test_size_ratio=test_size_ratio
        self.feature_names=feature_names
        self.stable_feature_indices=stable_feat_indices
        self.y_train=y_train
        self.X_train=X_train
        self.directory_save_ss='dataset/output_data/plots/ss/'
        self.directory_save_splines='dataset/output_data/plots/splines/'
        self.n_bins=n_bins

    def plot_stability_selection_results(self,features_stability,y_2d_training,y_2d_test,diagnostics_ss_array):
        if os.path.exists(self.directory_save_ss):
            shutil.rmtree(self.directory_save_ss)
        os.makedirs(self.directory_save_ss)

        selected_features_stability=features_stability[self.stable_feature_indices]
        selected_features_names=self.feature_names[self.stable_feature_indices]
        pl.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        pl.title("Stability of selected features after %d runs and %.2f ratio of training data per split"\
        %(self.num_shuffles,self.test_size_ratio))
        pl.bar(list(range(len(selected_features_stability))),selected_features_stability,color="b",align="center")
        pl.xticks(list(range(len(selected_features_stability))),selected_features_names,rotation=90)
        pl.xlim([-1, len(selected_features_stability)])
        pl.ylabel('Probability of being selected')
        pl.xlabel('Selected feature name')
        pl.grid(True)
        pl.savefig(self.directory_save_ss+'stability_plot.png')

        y_2d_training_avg=np.nanmean(y_2d_training,axis=1)
        y_2d_training_sd=np.nanstd(y_2d_training,axis=1)
        pl.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        pl.plot(np.linspace(0,len(self.y_train),len(self.y_train)),y_2d_training,"r.")
        y_average_pred_training,=pl.plot(np.linspace(0,len(self.y_train),len(self.y_train)),y_2d_training_avg,"b--")
        pl.errorbar(np.linspace(0,len(self.y_train),len(self.y_train)),
                    y_2d_training_avg,
                    yerr=y_2d_training_sd,
                    fmt='o')
        y_true,=pl.plot(np.linspace(0,len(self.y_train),len(self.y_train)),self.y_train,"k-")
        pl.xlabel("Example")
        pl.ylabel("Y value")
        pl.title("Predictions overall")
        pl.grid()
        pl.legend([y_average_pred_training,y_true],
                  ["Y average training predicted","Y true"])
        pl.savefig(self.directory_save_ss+'spam_y_training.png', dpi=100)

        y_2d_test_avg=np.nanmean(y_2d_test,axis=1)
        y_2d_test_sd=np.nanstd(y_2d_test,axis=1)
        pl.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        pl.plot(np.linspace(0,len(self.y_train),len(self.y_train)),y_2d_test,"r.")
        y_average_pred_test,=pl.plot(np.linspace(0,len(self.y_train),len(self.y_train)),y_2d_test_avg,"b--")
        pl.errorbar(np.linspace(0,len(self.y_train),len(self.y_train)),
                    y_2d_test_avg,
                    yerr=y_2d_test_sd,
                    fmt='o')
        y_true,=pl.plot(np.linspace(0,len(self.y_train),len(self.y_train)),self.y_train,"k-")
        pl.xlabel("Example")
        pl.ylabel("Y value")
        pl.title("Predictions overall")
        pl.grid()
        pl.legend([y_average_pred_test,y_true],
                  ["Y average test predicted","Y true"])
        pl.savefig(self.directory_save_ss+'spam_y_test.png', dpi=100)

        scores_train,scores_test=diagnostics_ss_array[:,0],diagnostics_ss_array[:,1]
        pl.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        n,bins,patches=pl.hist(scores_test,bins=self.n_bins,normed=1,facecolor='g',alpha=0.75)
        pl.xlabel('Score')
        pl.ylabel('Frequency')
        pl.title('Histogram of Score on Test data')
        pl.grid(True)
        pl.savefig(self.directory_save_ss+'score_hist_test_plot.png')

        pl.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        n,bins,patches=pl.hist(scores_train,bins=self.n_bins,normed=1,facecolor='g',alpha=0.75)
        pl.xlabel('Score')
        pl.ylabel('Frequency')
        pl.title('Histogram of Score on Training data')
        pl.grid(True)
        pl.savefig(self.directory_save_ss+'score_hist_train_plot.png')

    def plot_splines(self,pred_f_3d,best_parameters_cv_array,verbose_plot=False):
        if os.path.exists(self.directory_save_splines):
            shutil.rmtree(self.directory_save_splines)
        os.makedirs(self.directory_save_splines)

        directory_save_dfs='dataset/output_data/data_frames/'
        if os.path.exists(directory_save_dfs):
            shutil.rmtree(directory_save_dfs)
        os.makedirs(directory_save_dfs)

        pred_f_3d_avg=np.nanmean(pred_f_3d,axis=2)
        pred_f_3d_sd=np.nanstd(pred_f_3d,axis=2)
        pred_f_3d_max=np.nanmax(pred_f_3d,axis=2)
        pred_f_3d_min=np.nanmin(pred_f_3d,axis=2)

        spline_smooth=np.round(np.mean(best_parameters_cv_array[:,1]),2)
        for i in range(len(self.stable_feature_indices)):
            if verbose_plot:
                print(("Generating figure %i out of %i" %(i+1,len(self.stable_feature_indices))))
            sel_idx=self.stable_feature_indices[i]
            pl.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
            ii = np.argsort(self.X_train[:, sel_idx])
            y_avg_splined,dummy_variable=model.train_spline(self.X_train[:, sel_idx][ii],
                                                            pred_f_3d_avg[:, sel_idx][ii],
                                                            self.X_train[:, sel_idx][ii],
                                                            smooth_factor=spline_smooth)
            y_sd_lower,dummy_variable=model.train_spline(self.X_train[:, sel_idx][ii],
                                                         y_avg_splined-pred_f_3d_sd[:, sel_idx][ii],
                                                         self.X_train[:, sel_idx][ii],
                                                         smooth_factor=spline_smooth)
            y_sd_upper,dummy_variable=model.train_spline(self.X_train[:, sel_idx][ii],
                                                         y_avg_splined+pred_f_3d_sd[:, sel_idx][ii],
                                                         self.X_train[:, sel_idx][ii],
                                                         smooth_factor=spline_smooth)
            pl.plot(self.X_train[:, sel_idx][ii],y_avg_splined, "r-")
            pl.plot(self.X_train[:, sel_idx][ii],pred_f_3d_avg[:, sel_idx][ii],"k.")
#            pl.errorbar(self.X_train[:, sel_idx][ii],pred_f_3d_avg[:, sel_idx][ii],
#                    yerr=pred_f_3d_sd[:, sel_idx][ii],fmt='go')
            pl.fill_between(self.X_train[:, sel_idx][ii],
                            y_sd_lower,y_sd_upper,alpha=1,
                            edgecolor='gainsboro',facecolor='gainsboro')
            pl.xlabel("Feature value")
            pl.ylabel("Prediction f(x) value")
            pl.title("Predictions for feature "+self.feature_names[sel_idx])
            pl.grid()
            pl.savefig(self.directory_save_splines+'spam_feature_'+str(sel_idx)+'.png')
            pl.close()
            dataframe_data=np.transpose(np.array([self.X_train[:, sel_idx][ii],
                                                  pred_f_3d_avg[:, sel_idx][ii],
                                                  pred_f_3d_sd[:, sel_idx][ii],
                                                  pred_f_3d_max[:, sel_idx][ii],
                                                  pred_f_3d_min[:, sel_idx][ii]]))
            dataframe_columns=['feat_values_sorted',
                               'pred_f_3d_avg_sorted',
                               'pred_f_3d_sd_sorted',
                               'pred_f_3d_max_sorted',
                               'pred_f_3d_min_sorted']
            df_save=pd.DataFrame(data=dataframe_data,
                                 columns=dataframe_columns)
            df_save.to_csv(directory_save_dfs+'spam_feature_'+str(sel_idx)+'.txt',
                           header=True, index=False, sep='\t', float_format='%.5f')

        return True