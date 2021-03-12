
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from sklearn.ensemble import IsolationForest

def raise_error(error_msg):
    exit('ERROR! '+error_msg)

def make_dataset(n_features,n_informative,n_examples,X_pathway,
                 y_pathway,feat_names_pathway,example_id_pathway):

    X,y=make_regression(n_samples=n_examples,
                       n_features=n_features,
                       n_informative=n_informative,
                       random_state=256)
    feat_names=['Feature '+str(i) for i in range(0,n_features)]
    example_ids=['Example '+str(i) for i in range(0,n_examples)]
    np.savetxt(X_pathway,X,fmt='%0.2f',delimiter='\t')
    np.savetxt(y_pathway,y,fmt='%0.2f',delimiter='\t')
    np.savetxt(feat_names_pathway,feat_names,fmt='%s',delimiter='\t')
    np.savetxt(example_id_pathway,example_ids,fmt='%s',delimiter='\t')

def load_data(X_train_pathway,y_train_pathway,
              feat_names_pathway,example_names_pathway,
              data_contamination,
              X_test_pathway=None,y_test_pathway=None,
              num_feat_filtered=None):

    feat_names_original=np.loadtxt(feat_names_pathway,dtype=str,delimiter ='\t')
    example_names=np.loadtxt(example_names_pathway,dtype=str,delimiter ='\t')
    feat_names_full=np.array([x[1:].replace("'", "") for x in feat_names_original])

    X_train_original=np.loadtxt(X_train_pathway,dtype=float,delimiter ='\t')
    y_train_original=np.loadtxt(y_train_pathway,dtype=float,delimiter ='\t')
    scaler_x=StandardScaler(with_mean=True,with_std=True)
    X_train=scaler_x.fit_transform(X_train_original)
    scaler_y=StandardScaler(with_mean=True,with_std=True)
    y_train=np.ndarray.flatten(scaler_y.fit_transform(y_train_original.reshape(-1, 1)))

    if X_test_pathway is not None and y_test_pathway is not None:
        X_test_original=np.loadtxt(X_test_pathway,dtype=float,delimiter ='\t')
        y_test_original=np.loadtxt(y_test_pathway,dtype=float,delimiter ='\t')
        X_test=scaler_x.transform(X_test_original)
        y_test=np.ndarray.flatten(scaler_y.transform(y_test_original.reshape(-1, 1)))
    else:
        X_test,y_test=None,None

    if num_feat_filtered is None or num_feat_filtered>=np.size(X_train,1):
        [X_train_return,y_train_return]=[X_train,y_train]
        [feat_names_full_return,example_names_return]=[feat_names_full,example_names]
        [X_test_return,y_test_return]=[X_test,y_test]
    else:
        linear_selector=SelectKBest(score_func=f_regression,k=num_feat_filtered)
        nonlinear_selector=SelectKBest(score_func=mutual_info_regression,k=num_feat_filtered)
        linear_selector.fit(X_train,y_train)
        nonlinear_selector.fit(X_train,y_train)
        linear_indices=linear_selector.get_support(indices=True)
        nonlinear_indices=nonlinear_selector.get_support(indices=True)
        selected_feat_indices=list(set(list(linear_indices)+list(nonlinear_indices)))
        outlier_detector=IsolationForest(n_estimators=100,contamination=data_contamination,
                                         n_jobs=1)
        outlier_detector.fit(X_train)
        is_inlier_array=outlier_detector.predict(X_train)
        inlier_indices=list(list(np.where(is_inlier_array==1))[0])
        [X_train_return,y_train_return]=[X_train[:,selected_feat_indices],y_train]
        [X_train_return,y_train_return]=[X_train_return[inlier_indices,:],y_train_return[inlier_indices]]
        [feat_names_full_return,example_names_return]=[feat_names_full[selected_feat_indices],example_names[inlier_indices]]
        if X_test_pathway is not None and y_test_pathway is not None:
            [X_test_return,y_test_return]=[X_test[:,selected_feat_indices],y_test]
        else:
            [X_test_return,y_test_return]=[X_test,y_test]

    return [X_train_return,y_train_return],[feat_names_full_return,example_names_return],[X_test_return,y_test_return]

def get_alphas_grid(alpha_start,alpha_end,num_alphas=None):

    if num_alphas is not None:
        alphas_list=np.linspace(start=alpha_start,stop=alpha_end,num=num_alphas)
    else:
        num_alphas=int((alpha_end-alpha_start)/0.1)+1
        alphas_list=np.linspace(start=alpha_start,stop=alpha_end,num=num_alphas)

    return alphas_list

def get_smoothness_grid(smooth_power_start,smooth_power_end,smooth_base=10):

    num_s=smooth_power_end-smooth_power_start+1
    smoothness_list=np.logspace(start=smooth_power_start,stop=smooth_power_end,
                                num=num_s,base=smooth_base,endpoint=True)
    return smoothness_list