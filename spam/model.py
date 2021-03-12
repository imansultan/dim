
import numpy as np
from scipy.interpolate import BSpline,splrep

def train_spline(x_train,y_train,x_predict,smooth_factor=None,degree_fit=3):
    x_train_sorted_idx=np.argsort(x_train)
    x_train_sorted=x_train[x_train_sorted_idx]
    y_train_sorted=y_train[x_train_sorted_idx]
    x_predict_sorted_idx=np.argsort(x_predict)
    x_predict_original_idx=np.argsort(x_predict_sorted_idx)
    x_predict_sorted=x_predict[x_predict_sorted_idx]
    t_bspl,c_bspl,k_bspl=splrep(x_train_sorted,y_train_sorted,
                                s=smooth_factor,k=degree_fit)
    spl=BSpline(t_bspl,c_bspl,k_bspl)
    y_predict_sorted=spl(x_predict_sorted)
    y_predict_original=y_predict_sorted[x_predict_original_idx]

    return y_predict_original,spl

class SpamModel:

    def __init__(self,alpha=1.,max_iter=3,prediction_f_init=None,
                 spline_smooth=3,verbose=True):

        self.alpha=alpha
        self.max_iter=max_iter
        self.prediction_f_init=prediction_f_init
        self.spline_smooth=spline_smooth
        self.verbose=verbose

    def fit(self, X_train, y_train):
        number_s_train, number_f_train = X_train.shape
        self.inv_sqrt_number_s_train = 1. / np.sqrt(number_s_train)
        total_count = 0
        if self.prediction_f_init is None:
            prediction_f = np.zeros([number_s_train, number_f_train])
        else:
            prediction_f = self.prediction_f_init.copy()
        Residue = y_train - prediction_f.sum(1)

        def soft_thresholding():
            soft_thresh = 1. - self.alpha / s_j
            if soft_thresh > 0:
                f_j = soft_thresh * P_j
                f_j = f_j - f_j.mean()
                prediction_f[:, j] = f_j
                Residue[:] = Residue[:] - f_j
            else:
                prediction_f[:, j] = 0.
        self.model_parameters=[None] * number_f_train
        while total_count < self.max_iter:
            total_count += 1
            if self.verbose:
                print("Intelligible training %d" % total_count)
            for j in range(number_f_train):
    			# BASED ON Section 3: A Backfitting Algorithm for SpAM  page 3
                Residue[:] = Residue[:] + prediction_f[:, j]
                P_j,trained_spline = train_spline(X_train[:, j], Residue, X_train[:, j],
                                                  smooth_factor=self.spline_smooth)
                self.model_parameters[j]=trained_spline
                s_j = self.inv_sqrt_number_s_train * np.sqrt((P_j ** 2).sum())
                soft_thresholding()

        y_predicted=np.sum(prediction_f,axis=1)

        return prediction_f,y_predicted

    def predict_spline_pretrained(self,x_predict,spline_pretrained):
        x_predict_sorted_idx=np.argsort(x_predict)
        x_predict_original_idx=np.argsort(x_predict_sorted_idx)
        x_predict_sorted=x_predict[x_predict_sorted_idx]
        y_predict_sorted=spline_pretrained(x_predict_sorted)
        y_predict_original=y_predict_sorted[x_predict_original_idx]

        return y_predict_original

    def predict(self,X_test):
        number_s_test, number_f_test = X_test.shape
        prediction_f_test=np.zeros([np.size(X_test,0),np.size(X_test,1)])
        for i in range(0,np.size(X_test,1)):
            X_feat=X_test[:,i]
            y_test_spline=self.predict_spline_pretrained(X_feat,self.model_parameters[i])
            s_j_predict = self.inv_sqrt_number_s_train * np.sqrt((y_test_spline ** 2).sum())
            soft_thresh = 1. - self.alpha / s_j_predict
            if soft_thresh > 0:
                f_i = soft_thresh * y_test_spline
                f_i = f_i - f_i.mean()
                prediction_f_test[:, i] = f_i
            else:
                prediction_f_test[:, i] = 0.

        y_test_predicted=np.sum(prediction_f_test,axis=1)

        return prediction_f_test,y_test_predicted