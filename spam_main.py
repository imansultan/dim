#import matplotlib
#matplotlib.use('Agg')

import numpy as np
from model_selection import cross_validate
from visualization import graphics
from service_package import utils
import argparse as ap

import warnings
warnings.filterwarnings("ignore")

def main(test_size_ratio,num_shuffles,stab_gap,
         X_train_pathway,y_train_pathway,
         feat_names_pathway,example_names_pathway,
         cv_folds,alpha_start,alpha_end,num_alphas,
         smoothness_power_start,smoothness_power_end,
         smoothness_base,num_feat_filtered,data_contamination,
         max_backfit_iter=20,num_jobs_factor=2,
         X_test_pathway=None,y_test_pathway=None):

    print("Data loading")
    [X_train,y_train],[feat_names_full,example_names],[X_test,y_test]=\
    utils.load_data(X_train_pathway,y_train_pathway,feat_names_pathway,
                    example_names_pathway,data_contamination,
                    X_test_pathway,y_test_pathway,num_feat_filtered)

    print("Parameter grid seeding")
    smooth_factors_list=utils.get_smoothness_grid(smoothness_power_start,
                                                  smoothness_power_end,
                                                  smooth_base=smoothness_base)
    alphas_list=utils.get_alphas_grid(alpha_start,alpha_end,num_alphas)

    print("Grid search of parameters via cross-validation and stability selection")
    gridsearch=cross_validate.ModelSelector(smooth_factor_grid=smooth_factors_list,
                                            alpha_grid=alphas_list,
                                            X_data_training=X_train,
                                            y_training=y_train,
                                            max_backfit_iter=max_backfit_iter,
                                            num_jobs_factor=num_jobs_factor,
                                            cv_folds=cv_folds)
    stable_feat_indices,pred_f_3d,y_2d_training,y_2d_test=\
    gridsearch.run_stability_selection(test_size_ratio,num_shuffles,stab_gap)

    num_bins_hist=15
    print("Visualization of results")
    plots_instance=graphics.BuildPlots(num_shuffles,
                                       test_size_ratio,
                                       feat_names_full,
                                       stable_feat_indices,
                                       y_train,X_train,num_bins_hist)
    plots_instance.plot_stability_selection_results(gridsearch.stability_values,
                                                    y_2d_training,y_2d_test,
                                                    gridsearch.diagnostics_ss_array)
    plots_instance.plot_splines(pred_f_3d,gridsearch.best_parameters_cv_array)

    print("Done!")

if __name__ == "__main__":
    __spec__ = None

    parser = ap.ArgumentParser(description='Run SpAM model')
    parser.add_argument('train_data_name',
                        action="store",
                        type=str,
                        help="Name of the training data tab-delimited file")
    parser.add_argument('train_label_name',
                        action="store",
                        type=str,
                        help="Name of the training data label tab-delimited file")
    parser.add_argument('features_file_name',
                        action="store",
                        type=str,
                        help="Name of the features tab-delimited file")
    parser.add_argument('examples_file_name',
                        action="store",
                        type=str,
                        help="Name of the examples tab-delimited file")
    parser.add_argument('-test_size_ratio',
                        action="store",
                        type=float,
                        default=0.2,
                        help="Ratio of a test dataset, must be in (0,1] interval, default is 0.2")
    parser.add_argument('-stab_gap',
                        action="store",
                        type=float,
                        default=0.2,
                        help="Stability selection gap threshold, must be in (0,1] interval, default is 0.2")
    parser.add_argument('-num_shuffles',
                        action="store",
                        type=int,
                        default=10,
                        help="Number of random stratified shuffles, default is 10")
    parser.add_argument('-cv_folds',
                        action="store",
                        type=int,
                        default=5,
                        help="Number of random cross-validation shuffles, default is 5")
    parser.add_argument('-alpha_bounds',
                        action="store",
                        nargs='+',
                        type=float,
                        default=[0.1,0.9],
                        help="Range boundary values of alpha hyperparameter, default are [0.1,0.9]")
    parser.add_argument('-smoothess_power_bounds',
                        action="store",
                        nargs='+',
                        type=int,
                        default=[0,8],
                        help="Range boundary values of spline smoothnes power values, default are [0,8]")
    parser.add_argument('-num_alphas',
                        action="store",
                        type=int,
                        default=5,
                        help="Number of alphas in alpha list, default is 5")
    parser.add_argument('-smoothness_base',
                        action="store",
                        type=int,
                        default=2,
                        help="Power base for spline smoothness values, default is 2")
    parser.add_argument('-num_feat_filtered',
                        action="store",
                        type=int,
                        default=100,
                        help="Number of preselected features, default is 100")
    parser.add_argument('-data_contamination',
                        action="store",
                        type=float,
                        default=0.1,
                        help="Ratio of outliers in data, default is 0.1")

    args = parser.parse_args()
    X_train_file_name=args.train_data_name
    y_train_file_name=args.train_label_name
    features_file_name=args.features_file_name
    examples_file_name=args.examples_file_name
    test_size_ratio=args.test_size_ratio
    stab_gap=args.stab_gap
    num_shuffles=args.num_shuffles
    cv_folds=args.cv_folds
    alpha_bounds=args.alpha_bounds
    num_alphas=args.num_alphas
    smoothess_power_bounds=args.smoothess_power_bounds
    smoothness_base=args.smoothness_base
    num_feat_filtered=args.num_feat_filtered
    data_contamination=args.data_contamination

    input_data_folder_path='dataset/input_data/'
    X_train_pathway=input_data_folder_path+X_train_file_name
    y_train_pathway=input_data_folder_path+y_train_file_name
    feat_names_pathway=input_data_folder_path+features_file_name
    example_names_pathway=input_data_folder_path+examples_file_name

    [alpha_start,alpha_end]=alpha_bounds
    [smoothness_power_start,smoothness_power_end]=smoothess_power_bounds

    main(test_size_ratio,num_shuffles,stab_gap,
         X_train_pathway,y_train_pathway,
         feat_names_pathway,example_names_pathway,
         cv_folds,
         alpha_start,alpha_end,num_alphas,
         smoothness_power_start,smoothness_power_end,
         smoothness_base,num_feat_filtered,
         data_contamination)