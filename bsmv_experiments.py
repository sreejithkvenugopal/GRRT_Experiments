import numpy as np
#import matplotlib.pyplot as plt
import os,sys,json
import scipy as sci
from rrt_codes.block_single_measurement_vector import block_single_measurement_vector
from scipy.linalg import hadamard
from scipy.stats import norm
from sklearn import linear_model
from sklearn.linear_model import LassoCV
import argparse


def generate_full_support_from_block_support(block_support,block_size=4):
    ind = []
    for i in block_support:
        ind=ind+[j for j in range(i * block_size, (i + 1) * block_size)]
    return ind

# BOMP with a priori known block sparsity
def BOMP_prior_sparsity(X,y,k_block,l_block):
    n,p=X.shape;n_blocks=np.int(p/l_block)
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    res=y
    block_support=[]
    full_support=[]
    for k in np.arange(k_block):
        corr=np.matmul(X.T,res).flatten()
        corr_norm_per_block=np.array([np.linalg.norm(corr[indices_per_block[k]]) for k in np.arange(n_blocks)])
        block_ind=np.argmax(corr_norm_per_block)
        block_support.append(block_ind)
        full_support=full_support+indices_per_block[block_ind]
        X_new=X[:,full_support];
        try:
            beta_est=np.matmul(np.linalg.pinv(X_new),y).flatten()
        except:
            break
        Beta_est=np.zeros(p)
        Beta_est[full_support]=beta_est
        Beta_est=Beta_est.reshape((p,1))
        res=y-np.matmul(X,Beta_est)
    return Beta_est,full_support,block_support

# BOMP which stops once residual drops below a user provide threshold. threshold could be noise l2_norm or a noise variance based upper bound   
def BOMP_prior_variance(X,y,threshold,l_block):
    n,p=X.shape;n_blocks=np.int(p/l_block)
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    res=y
    block_support=[]
    full_support=[]
    for k in np.arange(n_blocks):
        corr=np.matmul(X.T,res).flatten()
        corr_norm_per_block=np.array([np.linalg.norm(corr[indices_per_block[k]]) for k in np.arange(n_blocks)])
        block_ind=np.argmax(corr_norm_per_block)
        block_support.append(block_ind)
        full_support=full_support+indices_per_block[block_ind]
        X_new=X[:,full_support];
        try:
            beta_est=np.matmul(np.linalg.pinv(X_new),y).flatten()
        except:
            break
        Beta_est=np.zeros(p)
        Beta_est[full_support]=beta_est
        Beta_est=Beta_est.reshape((p,1))
        res=y-np.matmul(X,Beta_est)
        if np.linalg.norm(res)<threshold:
            break
    return Beta_est,full_support,block_support


# BOMP using the validation scheme proposed in "On the theoretical
# analysis of cross validation in compressive sensing" in Proc. ICASSP
## 2014. IEEE, 2014, pp. 3370–3374.
def BOMP_CV(X,y,l_block,cv_fraction):
    #cv_fraction: fraction of measurments for validation.
    n,p=X.shape;n_blocks=np.int(p/l_block)
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    n_cv=np.int(n*cv_fraction)
    indices=np.random.choice(n,n,False).tolist()
    ind_cv=indices[:n_cv]
    ind_train=indices[n_cv:]
    n_train=len(ind_train)
    
    y_train=y[ind_train].reshape((n_train,1)); y_cv=y[ind_cv].reshape((n_cv,1));
    X_train=X[ind_train,:]; X_cv=X[ind_cv,:]
    train_col_norms=np.linalg.norm(X_train,axis=0)+1e-8
    X_train=X_train/train_col_norms
    train_col_norms=train_col_norms.reshape((p,1))
    max_iter=np.int(np.floor(n_train/(l_block)))
    cv_error_list=[np.linalg.norm(y_cv)**2/(n_cv)]
    train_error_list=[np.linalg.norm(y_train)**2/(n_train)]
    min_cv_error=cv_error_list[0]   
    res=y_train
    block_support=[]
    full_support=[]
    
    best_full_support=full_support
    best_block_support=block_support
    best_est=np.zeros((p,1))
    for k in np.arange(max_iter):
        corr=np.matmul(X_train.T,res).flatten()
        corr_norm_per_block=np.array([np.linalg.norm(corr[indices_per_block[k]]) for k in np.arange(n_blocks)])
        block_ind=np.argmax(corr_norm_per_block)
        block_support.append(block_ind)
        full_support=full_support+indices_per_block[block_ind]
        X_new=X_train[:,full_support];
        try:
            beta_est=np.matmul(np.linalg.pinv(X_new),y_train).flatten()
        except:
            break
        Beta_est=np.zeros(p)
        Beta_est[full_support]=beta_est
        Beta_est=Beta_est.reshape((p,1))
        res=y_train-np.matmul(X_train,Beta_est)
        train_error_list.append(np.linalg.norm(res)**2/n_train)
        Beta_scaled=Beta_est*train_col_norms #rescaling to accomodate scaling in X_train
        res_cv=y_cv-np.matmul(X_cv,Beta_scaled)
        cv_error=np.linalg.norm(res_cv)**2/n_cv
        #print(cv_error_list)
        cv_error_list.append(cv_error)
        if cv_error<min_cv_error:
            best_full_support=full_support
            best_block_support=block_support
            best_est=Beta_scaled
            min_cv_error=cv_error
    CV_dict={}
    CV_dict['train_error_list']=train_error_list
    CV_dict['cv_error_list']=cv_error_list
    return best_est,best_full_support,best_block_support,CV_dict
        

# Block sparse version of SPICE from T. Kronvall, S. I. Adalbj¨ornsson, S. Nadig, and A. Jakobsson, “Groupsparse
#regression using the covariance fitting criterion,” Signal Processing,
# vol. 139, pp. 116–130, 2017.
def group_spice(X,y,l_block):
    #implemented with r=infty. homoscedastic version. 
    n,p=X.shape; n_blocks=np.int(p/l_block)
    r=1e8; s=1;
    max_iter=1000;tol=1e-4
    current_iter=0;
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    col_norm=np.linalg.norm(X,axis=0)
    v=np.zeros(n_blocks)
    for k in np.arange(n_blocks):
        ind=indices_per_block[k]
        v[k]=np.linalg.norm(col_norm[ind]**2,s)
        
    A=np.hstack([X,np.eye(n)])
    
    sigma_current=np.sqrt(np.matmul(y.T,y)/n)
    p_current=np.zeros(p)
    for k in np.arange(p):
        p_current[k]=np.matmul(X[:,k].T,y)**2/np.linalg.norm(X[:,k])**4
    
    while current_iter<max_iter:
        current_iter+=1
        R=np.matmul(np.matmul(X,np.diag(p_current)),X.T)+sigma_current*np.eye(n)
        z=np.matmul(np.linalg.inv(R),y)
        sigma_next=sigma_current*np.sqrt(np.linalg.norm(z)**2/n)
        p_next=np.zeros(p)
        for k in np.arange(n_blocks):
            ind=indices_per_block[k]
            block_r_p=np.zeros(len(ind))
            r_z=np.abs(np.matmul(X[:,ind].T,z)).flatten()
    
            p_z=p_current[ind].flatten()
            p_next_block=np.linalg.norm([r_z[j]*p_z[j] for j in np.arange(l_block)])/np.sqrt(v[k])
            p_next[ind]=p_next_block
        if np.linalg.norm(p_next-p_current)<tol:
            break;
        p_current=p_next
        sigma_current=sigma_next
    
    Beta_est=np.zeros(p)
    for k in np.arange(p):
        Beta_est[k]=p_current[k]*np.matmul(X[:,k].T,z)
        
    return Beta_est.reshape((p,1)),p_current,sigma_current

def compute_error(support_true,support_estimate,Beta_true,Beta_estimate):
    Beta_true=np.squeeze(Beta_true); Beta_estimate=np.squeeze(Beta_estimate);
    l2_error=np.linalg.norm(Beta_true-Beta_estimate)**2/np.linalg.norm(Beta_true)**2

    if len(support_estimate)==0:
        support_error=1;
        recall=0
        precision=1;
        pmd=1
        pfd=0
    else:
        support_true=set(support_true); support_estimate=set(support_estimate)

        if support_true==support_estimate:
            support_error=0;
        else:
            support_error=1;
        recall=len(support_true.intersection(support_estimate))/len(support_true)
        precision=len(support_estimate.intersection(support_true))/len(support_estimate)
        
        if len(support_true.difference(support_estimate))>0:
            pmd=1;
        else:
            pmd=0;
        if len(support_estimate.difference(support_true))>0:
            pfd=1
        else:
            pfd=0;
            
            
    return support_error,l2_error,recall,precision,pmd,pfd


def run_experiment(num_iter=100,matrix_type='two_ortho'):
    n,p,k_block,l_block=64,128,3,4
    n_blocks = np.int(p /l_block)
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    SNR=np.linspace(-10,20,10)
    snr=10**(SNR/10)# SNR in real scale
    num_iter=num_iter

    MSE_cv=np.zeros(10);MSE_sparsity=np.zeros(10);MSE_alpha1=np.zeros(10);MSE_alpha2=np.zeros(10);MSE_variance=np.zeros(10);
    MSE_spice=np.zeros(10);MSE_spice_ls=np.zeros(10)
    MSE_variance2=np.zeros(10);

    PE_cv=np.zeros(10);PE_sparsity=np.zeros(10);PE_alpha1=np.zeros(10);PE_alpha2=np.zeros(10);PE_variance=np.zeros(10);
    PE_spice=np.zeros(10);PE_spice_ls=np.zeros(10);
    PE_variance2=np.zeros(10);
    
    PFD_cv=np.zeros(10);PFD_sparsity=np.zeros(10);PFD_alpha1=np.zeros(10);PFD_alpha2=np.zeros(10);PFD_variance=np.zeros(10);
    PFD_spice=np.zeros(10);PFD_spice_ls=np.zeros(10);
    PFD_variance2=np.zeros(10);
    
    PMD_cv=np.zeros(10);PMD_sparsity=np.zeros(10);PMD_alpha1=np.zeros(10);PMD_alpha2=np.zeros(10);PMD_variance=np.zeros(10);
    PMD_spice=np.zeros(10);PMD_spice_ls=np.zeros(10);
    PMD_variance2=np.zeros(10);



    Recall_cv=np.zeros(10);Recall_sparsity=np.zeros(10);Recall_alpha1=np.zeros(10);Recall_alpha2=np.zeros(10);
    Recall_variance=np.zeros(10);Recall_spice=np.zeros(10);Recall_spice_ls=np.zeros(10);
    Recall_variance2=np.zeros(10);

    Precision_cv=np.zeros(10);Precision_sparsity=np.zeros(10);Precision_alpha1=np.zeros(10);
    Precision_alpha2=np.zeros(10);Precision_variance=np.zeros(10);Precision_spice=np.zeros(10);
    Precision_variance2=np.zeros(10);Precision_spice_ls=np.zeros(10);

    bsmv=block_single_measurement_vector();
    
    for snr_iter in np.arange(10):
        print(SNR[snr_iter])
        mse_cv=0;mse_sparsity=0;mse_alpha1=0;mse_alpha2=0;mse_variance=0;mse_variance2=0;mse_spice=0;mse_spice_ls=0;
        pe_cv=0;pe_sparsity=0;pe_alpha1=0;pe_alpha2=0;pe_variance=0;pe_variance2=0;pe_spice=0;pe_spice_ls=0;
        pfd_cv=0;pfd_sparsity=0;pfd_alpha1=0;pfd_alpha2=0;pfd_variance=0;pfd_variance2=0;pfd_spice=0;pfd_spice_ls=0;
        pmd_cv=0;pmd_sparsity=0;pmd_alpha1=0;pmd_alpha2=0;pmd_variance=0;pmd_variance2=0;pmd_spice=0;pmd_spice_ls=0;
        recall_cv=0;recall_sparsity=0;recall_alpha1=0;recall_alpha2=0;recall_variance=0;recall_variance2=0;
        recall_spice=0;recall_spice_ls=0;
        precision_cv=0;precision_sparsity=0;precision_alpha1=0;precision_alpha2=0;precision_variance=0;
        precision_spice=0;precision_spice_ls=0;precision_variance2=0;


        for num in np.arange(num_iter):
            #print(num)
            
            #signal model

            if matrix_type=='two_ortho':
                X=np.hstack([np.eye(n),hadamard(n)/np.sqrt(n)])
            elif matrix_type=='normal':
                X=np.random.randn(n,p)/np.sqrt(n)
            else:
                raise Exception('Invalid matrix type. Give one of normal or two_ortho')
           

            Beta_true=np.zeros((p,1))

            if p % l_block != 0:
                raise Exception(' nfeatures should be a multiple of block_size')


            block_support= np.random.choice(np.arange(n_blocks), size=k_block, replace=False).tolist()
            support_true=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)
            Beta_true[support_true] = np.sign(np.random.randn(len(support_true), 1))
            signal_power=len(support_true)
            # noise_power=nsamples*noisevar. snr=signal_power/noise_power
            noise_var = signal_power/ (n * snr[snr_iter])
            noise = np.random.randn(n, 1) * np.sqrt(noise_var)
            y= np.matmul(X, Beta_true) + noise
            
            
            #GRRT 
            rrt_bomp_dict=bsmv.compute_signal_and_support(X,Y=y,block_size=l_block,alpha_list=[0.1,0.01])
            block_support,Beta_est_alpha1=rrt_bomp_dict[0.1]['support_estimate'],rrt_bomp_dict[0.1]['signal_estimate']
            support_est=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)

            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est_alpha1)
            mse_alpha1+=l2_error;pe_alpha1+=support_error;recall_alpha1+=recall;precision_alpha1+=precision;
            pmd_alpha1+=pmd;pfd_alpha1+=pfd

            block_support,Beta_est_alpha2=rrt_bomp_dict[0.01]['support_estimate'],rrt_bomp_dict[0.01]['signal_estimate']
            support_est=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                Beta_true,Beta_est_alpha2)
            mse_alpha2+=l2_error;pe_alpha2+=support_error;recall_alpha2+=recall;precision_alpha2+=precision;
            pmd_alpha2+=pmd;pfd_alpha2+=pfd

            
            # Groupd SPICE
            Beta_est,p_current,sigma_est=group_spice(X,y,l_block)
            power_per_block=np.zeros(n_blocks)
            for k in np.arange(n_blocks):
                ind=indices_per_block[k]
                power_per_block[k]=(np.linalg.norm(Beta_est.flatten()[ind],2)**2)/len(ind)
            block_support=np.where(power_per_block>1e-2)[0]
            
            support_est=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)

            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_spice+=l2_error;pe_spice+=support_error;recall_spice+=recall;precision_spice+=precision;
            pmd_spice+=pmd;pfd_spice+=pfd

            
            max_power=np.max(power_per_block)
            block_support=np.where(power_per_block>0.2*max_power)[0]
            support_est=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)
            beta_est=np.matmul(np.linalg.pinv(X[:,support_est]),y)
            Beta_est_ls=np.zeros(p);Beta_est_ls[support_est]=beta_est.flatten()
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est_ls)
            mse_spice_ls+=l2_error;pe_spice_ls+=support_error;recall_spice_ls+=recall;precision_spice_ls+=precision;
            pmd_spice_ls+=pmd;pfd_spice_ls+=pfd
            
            
            
            Beta_est,support_est,block_support=BOMP_prior_sparsity(X,y,k_block,l_block)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_sparsity+=l2_error;pe_sparsity+=support_error;recall_sparsity+=recall;precision_sparsity+=precision;
            pmd_sparsity+=pmd;pfd_sparsity+=pfd;


            threshold=np.sqrt(noise_var)*np.sqrt(n+2*np.sqrt(n*np.log(n)))
            Beta_est,support_est,block_support=BOMP_prior_variance(X,y,threshold,l_block)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_variance+=l2_error;pe_variance+=support_error;recall_variance+=recall;precision_variance+=precision;
            pmd_variance+=pmd;pfd_variance+=pfd;


            threshold=np.linalg.norm(noise)
            Beta_est,support_est,block_support=BOMP_prior_variance(X,y,threshold,l_block)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_variance2+=l2_error;pe_variance2+=support_error;recall_variance2+=recall;precision_variance2+=precision;
            pmd_variance2+=pmd;pfd_variance2+=pfd;

            Beta_est,support_est,best_block_support,CV_dict=BOMP_CV(X,y,l_block,cv_fraction=0.1)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_cv+=l2_error;pe_cv+=support_error;recall_cv+=recall;precision_cv+=precision; 
            pmd_cv+=pmd;pfd_cv+=pfd;


        MSE_cv[snr_iter]=mse_cv/num_iter;MSE_sparsity[snr_iter]=mse_sparsity/num_iter;
        MSE_alpha1[snr_iter]=mse_alpha1/num_iter;MSE_alpha2[snr_iter]=mse_alpha2/num_iter;
        MSE_variance[snr_iter]=mse_variance/num_iter;MSE_spice[snr_iter]=mse_spice/num_iter;MSE_spice_ls[snr_iter]=mse_spice_ls/num_iter;
        MSE_variance2[snr_iter]=mse_variance2/num_iter;

        PE_cv[snr_iter]=pe_cv/num_iter;PE_sparsity[snr_iter]=pe_sparsity/num_iter;
        PE_alpha1[snr_iter]=pe_alpha1/num_iter;PE_alpha2[snr_iter]=pe_alpha2/num_iter;
        PE_variance[snr_iter]=pe_variance/num_iter; PE_variance2[snr_iter]=pe_variance2/num_iter; 
        PE_spice[snr_iter]=pe_spice/num_iter;PE_spice_ls[snr_iter]=pe_spice_ls/num_iter;
        
        PFD_cv[snr_iter]=pfd_cv/num_iter;PFD_sparsity[snr_iter]=pfd_sparsity/num_iter;
        PFD_alpha1[snr_iter]=pfd_alpha1/num_iter;PFD_alpha2[snr_iter]=pfd_alpha2/num_iter;
        PFD_variance[snr_iter]=pfd_variance/num_iter; PFD_variance2[snr_iter]=pfd_variance2/num_iter; 
        PFD_spice[snr_iter]=pfd_spice/num_iter;PFD_spice_ls[snr_iter]=pfd_spice_ls/num_iter;
        
        PMD_cv[snr_iter]=pmd_cv/num_iter;PMD_sparsity[snr_iter]=pmd_sparsity/num_iter;
        PMD_alpha1[snr_iter]=pmd_alpha1/num_iter;PMD_alpha2[snr_iter]=pmd_alpha2/num_iter;
        PMD_variance[snr_iter]=pmd_variance/num_iter; PMD_variance2[snr_iter]=pmd_variance2/num_iter; 
        PMD_spice[snr_iter]=pmd_spice/num_iter;PMD_spice_ls[snr_iter]=pmd_spice_ls/num_iter;

        Recall_cv[snr_iter]=recall_cv/num_iter;Recall_sparsity[snr_iter]=recall_sparsity/num_iter;
        Recall_alpha1[snr_iter]=recall_alpha1/num_iter;Recall_alpha2[snr_iter]=recall_alpha2/num_iter;
        Recall_variance[snr_iter]=recall_variance/num_iter;Recall_spice[snr_iter]=recall_spice/num_iter;
        Recall_variance2[snr_iter]=recall_variance2/num_iter;Recall_spice_ls[snr_iter]=recall_spice_ls/num_iter;


        Precision_cv[snr_iter]=precision_cv/num_iter;Precision_sparsity[snr_iter]=precision_sparsity/num_iter;
        Precision_alpha1[snr_iter]=precision_alpha1/num_iter;Precision_alpha2[snr_iter]=precision_alpha2/num_iter;
        Precision_variance[snr_iter]=precision_variance/num_iter;Precision_variance2[snr_iter]=precision_variance2/num_iter;
        Precision_spice[snr_iter]=precision_spice/num_iter;Precision_spice_ls[snr_iter]=precision_spice_ls/num_iter;
        
    print('over')
    print(' experiment over')   
    print('saving results')
    results={}
    results['algo']='SOMP'
    results['experiment_type']='SNR_sweep'
    results['num_iter']=num_iter
    results['n']=n
    results['p']=p
    results['l_block']=l_block
    results['k_block']=k_block
    results['SNR']=SNR.tolist()
    results['matrix_type']=matrix_type
    results['sigmal_type']='pm1' #plus or minus 1.
    results['MSE_cv']=MSE_cv.tolist();results['MSE_variance']=MSE_variance.tolist();results['MSE_variance2']=MSE_variance2.tolist();
    results['MSE_spice']=MSE_spice.tolist();results['MSE_spice_ls']=MSE_spice_ls.tolist();
    results['MSE_sparsity']=MSE_sparsity.tolist();results['MSE_alpha1']=MSE_alpha1.tolist(); results['MSE_alpha2']=MSE_alpha2.tolist(); 
    results['PE_cv']=PE_cv.tolist();results['PE_variance']=PE_variance.tolist();
    results['PE_variance2']=PE_variance2.tolist();results['PE_spice']=PE_spice.tolist();results['PE_spice_ls']=PE_spice_ls.tolist();
    results['PE_sparsity']=PE_sparsity.tolist();results['PE_alpha1']=PE_alpha1.tolist(); results['PE_alpha2']=PE_alpha2.tolist(); 
    results['Recall_cv']=Recall_cv.tolist();results['Recall_variance']=Recall_variance.tolist();
    results['Recall_variance2']=Recall_variance2.tolist();results['Recall_spice']=Recall_spice.tolist();
    results['Recall_sparsity']=Recall_sparsity.tolist();results['Recall_alpha1']=Recall_alpha1.tolist(); 
    results['Recall_alpha2']=Recall_alpha2.tolist(); results['Recall_spice_ls']=Recall_spice_ls.tolist();
    
    results['Precision_cv']=Precision_cv.tolist();results['Precision_variance']=Precision_variance.tolist();
    results['Precision_variance2']=Precision_variance2.tolist();results['Precision_spice']=Precision_spice.tolist();
    results['Precision_sparsity']=Precision_sparsity.tolist();results['Precision_alpha1']=Precision_alpha1.tolist();
    results['Precision_alpha2']=Precision_alpha2.tolist(); results['Precision_spice_ls']=Precision_spice_ls.tolist();
    
    results['PMD_cv']=PMD_cv.tolist();results['PMD_variance']=PMD_variance.tolist();
    results['PMD_variance2']=PMD_variance2.tolist();results['PMD_spice']=PMD_spice.tolist();results['PMD_spice_ls']=PMD_spice_ls.tolist();
    results['PMD_sparsity']=PMD_sparsity.tolist();results['PMD_alpha1']=PMD_alpha1.tolist(); results['PMD_alpha2']=PMD_alpha2.tolist(); 
    
    results['PFD_cv']=PFD_cv.tolist();results['PFD_variance']=PFD_variance.tolist();
    results['PFD_variance2']=PFD_variance2.tolist();results['PFD_spice']=PFD_spice.tolist();results['PFD_spice_ls']=PFD_spice_ls.tolist();
    results['PFD_sparsity']=PFD_sparsity.tolist();results['PFD_alpha1']=PFD_alpha1.tolist(); results['PFD_alpha2']=PFD_alpha2.tolist(); 
    
    
    file_name='BOMP_SNR_sweep_'+matrix_type+'.json'
    print(file_name)
    with open(file_name,'w') as f:
        json.dump(results,f)
    print('results dumped to {}'.format(file_name))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, default=100,
                        help='how many monte carlo iterations?')
    parser.add_argument('--matrix_type', type=str, default='two_ortho',
                        help='matrix type: normal or two_ortho?')
    
    args = parser.parse_args()
    print(args)
    run_experiment(num_iter=args.num_iter,matrix_type=args.matrix_type)

        
        
    