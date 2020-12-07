import numpy as np
#import matplotlib.pyplot as plt
import os,sys,json
import scipy as sci
from rrt_codes.multiple_measurement_vector import multiple_measurement_vector
from scipy.linalg import hadamard
from scipy.stats import norm
from sklearn import linear_model
from sklearn.linear_model import LassoCV
import argparse

def SOMP_CV(X,Y,cv_fraction):
    ## Based on J. Zhang, L. Chen, P. T. Boufounos, and Y. Gu, “On the theoretical
    ####analysis of cross validation in compressive sensing,” in Proc. ICASSP
    ### 2014. IEEE, 2014, pp. 3370–3374.
    #cv_fraction= fraction of samples reserved as CV
    n,p=X.shape
    L=Y.shape[1]
    n_cv=np.int(n*cv_fraction)
    indices=np.random.choice(n,n,False).tolist()
    ind_cv=indices[:n_cv] #cv measurements
    ind_train=indices[n_cv:] # train measurements
    n_train=len(ind_train)
    
    Y_train=Y[ind_train].reshape((n_train,L)); Y_cv=Y[ind_cv].reshape((n_cv,L));
    X_train=X[ind_train,:]; X_cv=X[ind_cv,:]
    train_col_norms=np.linalg.norm(X_train,axis=0)+1e-8
    X_train=X_train/train_col_norms # normalize the columns to ensure unit l2 norm for train design matrix
    train_col_norms=train_col_norms.reshape((p,1))
    #print(train_col_norms.flatten())
    
    max_iter=np.int(n_train) # maximum iterations in train data
    cv_error_list=[np.linalg.norm(Y_cv)**2/(n_cv*L)]
    min_cv_error=cv_error_list[0]
    best_support=[]
    best_beta=np.zeros((p,L))
    
    
    # starting OMP_CV
    res=Y_train
    train_error_list=[np.linalg.norm(res)**2/(n_train*L)]
    support_estimate=[]
    while len(support_estimate)<max_iter:
        correlation=np.matmul(X_train.T,res) # unlike OMP, correlation is a matrix of size nfeatures times nchannels
        corrnorm = np.linalg.norm(correlation,axis=1) # take the frobenius norm of correlations corresponding to each features. you can try different norms here.
        ind = np.argmax(corrnorm)
        support_estimate.append(ind)
        #print(support_estimate)
        Xk=X_train[:,support_estimate].reshape((n_train,len(support_estimate)))

        Xk_pinv = np.linalg.pinv(Xk)
        Beta_est = np.zeros((p,L))
        if len(support_estimate)==1:
            Beta_est[support_estimate,:] = np.matmul(Xk_pinv, Y_train).reshape((1,L))
        else:
            Beta_est[support_estimate,:] = np.matmul(Xk_pinv, Y_train)

        res = Y_train - np.matmul(X_train, Beta_est)
        Beta_scaled=Beta_est/train_col_norms # rescaling to account for scaling of X in train data
        cv_error=np.linalg.norm(Y_cv-np.matmul(X_cv,Beta_scaled))**2/(n_cv*L)
        cv_error_list.append(cv_error)
        
        train_error_list.append(np.linalg.norm(res)**2/(n_train*L))
        
        if cv_error<min_cv_error:
            min_cv_error=cv_error
            best_support=support_estimate.copy()
            best_beta=Beta_scaled.copy()
            
       
    CV_dict={}
    CV_dict['train_error_list']=train_error_list;CV_dict['cv_error_list']=cv_error_list;
    CV_dict['feature_index']=support_estimate
    return CV_dict,best_support,best_beta

# SOMP with prior sparsity as implemented in "H. Li, L. Wang, X. Zhan, and D. K. Jain, “On the fundamental limit
# of orthogonal matching pursuit for multiple measurement vector,” IEEE
# Access, vol. 7, pp. 48 860–48 866, 2019."

def SOMP_prior_sparsity(X,Y,sparsity):
        n,p=X.shape; L=Y.shape[1]
        Y=Y.reshape((n,L))
        res=Y # initialize the residual with observation
        
        support_estimate=[];
        flag=0;
        for k in range(sparsity):
            correlation=np.matmul(X.T,res) # unlike OMP, correlation is a matrix of size nfeatures times nchannels
            corrnorm = np.linalg.norm(correlation,axis=1) # take the frobenius norm of correlations corresponding to each features. you can try different norms here. l2_norm of row of residual correlation is used here.
            ind = np.argmax(corrnorm)
            support_estimate.append(ind)
            #print(support_estimate)
            Xk=X[:,support_estimate].reshape((n,k+1))
            
            Xk_pinv = np.linalg.pinv(Xk)
            Beta_est = np.zeros((p,L))
            if len(support_estimate)==1:
                Beta_est[support_estimate,:] = np.matmul(Xk_pinv, Y).reshape((1,L))
            else:
                Beta_est[support_estimate,:] = np.matmul(Xk_pinv, Y)

            res = Y - np.matmul(X, Beta_est)

            #print('Ill conditioned matrix. OMP stop at iteration:' + str(k))
            flag = 1;
            #break;
        return support_estimate,Beta_est

    
# SOMP with prior threshold on residual norm as implemented in "H. Li, L. Wang, X. Zhan, and D. K. Jain, “On the fundamental limit
# of orthogonal matching pursuit for multiple measurement vector,” IEEE
# Access, vol. 7, pp. 48 860–48 866, 2019.". Thereshold could be based on noise norm or noise variance

    
def SOMP_prior_variance(X,Y,threshold):
        n,p=X.shape; L=Y.shape[1]
        Y=Y.reshape((n,L))
        res=Y # initialize the residual with observation
        
        support_estimate=[];
        flag=0;
        for k in np.arange(n):
            correlation=np.matmul(X.T,res) # unlike OMP, correlation is a matrix of size nfeatures times nchannels
            corrnorm = np.linalg.norm(correlation,axis=1) # take the frobenius norm of correlations corresponding to each features. you can try different norms here.
            ind = np.argmax(corrnorm)
            support_estimate.append(ind)
            #print(support_estimate)
            Xk=X[:,support_estimate].reshape((n,k+1))
            try:
            
                Xk_pinv = np.linalg.pinv(Xk)
                Beta_est = np.zeros((p,L))
                if len(support_estimate)==1:
                    Beta_est[support_estimate,:] = np.matmul(Xk_pinv, Y).reshape((1,L))
                else:
                    Beta_est[support_estimate,:] = np.matmul(Xk_pinv, Y)
            except:
                break

            res = Y - np.matmul(X, Beta_est)

            if np.linalg.norm(res)<threshold:
                break;
        return support_estimate,Beta_est

def spice(X,Y):
    n=X.shape[0];p=X.shape[1]; L=Y.shape[1]
    max_iter=1000; tol=1e-4
    current_iter=0
    
    ### precomputation
    A=np.hstack([X,np.eye(n)])
    A_k_norm=np.linalg.norm(A,axis=0).tolist()
    R_hat=np.matmul(Y,Y.T)/L
    
    W=np.zeros(p+n)
    for k in np.arange(n+p):
        W[k]=np.linalg.norm(A[:,k])**2/np.trace(R_hat)
        
    gamma=np.sum(W[:-n])
    #rint(gamma)
    
    #initialization
    p_current=np.zeros(p+n)
    for k in np.arange(p+n):
        p_current[k]=np.matmul(np.matmul(A[:,k].T,R_hat),A[:,k])/A_k_norm[k]**4
    smallest_n=np.argsort(p_current)[:n].tolist()
    sigma_current=0
    for k in np.arange(n):
        ind=smallest_n[k]
        sigma_current+=p_current[ind]*A_k_norm[ind]**2
    sigma_current=sigma_current/n
    
    #rint(p_current)
    p_current=p_current[:p]
    
    while current_iter<max_iter:
        current_iter+=1
        R_est=np.matmul(np.matmul(X,np.diag(p_current[:p])),X.T)+sigma_current*np.eye(n)
        R_est_inv=np.linalg.inv(R_est)
        R_est_inv_R_hat_sqrt=np.matmul(R_est_inv,R_hat)
        
        rho=0
        for k in np.arange(p):
            rho+=np.sqrt(W[k])*p_current[k]*np.linalg.norm(np.matmul(A[:,k].T,R_est_inv_R_hat_sqrt))
        rho+=np.sqrt(gamma)*sigma_current*np.linalg.norm(R_est_inv_R_hat_sqrt)
        #rint(rho)
        p_next=np.zeros(p)
        for k in np.arange(p):
            p_next[k]=p_current[k]*np.linalg.norm(np.matmul(X[:,k].T,R_est_inv_R_hat_sqrt))/(np.sqrt(W[k])*rho)
        sigma_next=sigma_current*np.linalg.norm(R_est_inv_R_hat_sqrt)/np.sqrt(gamma)*rho
        if np.linalg.norm(p_next-p_current)<tol:
            #print('iter:{}'.format(current_iter))
            break
        
        p_current=p_next;sigma_next=sigma_current;
        
    p_current_list=p_current.tolist()
    for k in np.arange(n):
        p_current_list.append(sigma_current)
    B_est=np.matmul(np.matmul(np.diag(p_current_list),A.T),np.matmul(R_est_inv,Y))
    
    return B_est[:p,:],p_current,sigma_next

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
    n,p,L,k_0=64,128,10,6
    SNR=np.linspace(-10,20,10)
    snr=10**(SNR/10)
    num_iter=num_iter
    MSE_cv=np.zeros(10);MSE_sparsity=np.zeros(10);MSE_alpha1=np.zeros(10);MSE_alpha2=np.zeros(10);MSE_variance=np.zeros(10);
    MSE_spice_ls=np.zeros(10);MSE_spice=np.zeros(10);MSE_unknownA=np.zeros(10);MSE_variance2=np.zeros(10);

    PE_cv=np.zeros(10);PE_sparsity=np.zeros(10);PE_alpha1=np.zeros(10);PE_alpha2=np.zeros(10);PE_variance=np.zeros(10);
    PE_spice_ls=np.zeros(10);PE_spice=np.zeros(10);PE_unknownA=np.zeros(10);PE_variance2=np.zeros(10);

    Recall_cv=np.zeros(10);Recall_sparsity=np.zeros(10);Recall_alpha1=np.zeros(10);Recall_alpha2=np.zeros(10);
    Recall_variance=np.zeros(10);Recall_variance2=np.zeros(10);
    Recall_spice_ls=np.zeros(10);Recall_spice=np.zeros(10);Recall_unknownA=np.zeros(10)

    Precision_cv=np.zeros(10);Precision_sparsity=np.zeros(10);Precision_alpha1=np.zeros(10);
    Precision_alpha2=np.zeros(10);Precision_variance=np.zeros(10);Precision_variance2=np.zeros(10);
    Precision_spice_ls=np.zeros(10);Precision_spice=np.zeros(10);Precision_unknownA=np.zeros(10);
    
    PFD_cv=np.zeros(10);PFD_sparsity=np.zeros(10);PFD_alpha1=np.zeros(10);
    PFD_alpha2=np.zeros(10);PFD_variance=np.zeros(10);PFD_variance2=np.zeros(10);
    PFD_spice_ls=np.zeros(10);PFD_spice=np.zeros(10);PFD_unknownA=np.zeros(10);
    
    PMD_cv=np.zeros(10);PMD_sparsity=np.zeros(10);PMD_alpha1=np.zeros(10);
    PMD_alpha2=np.zeros(10);PMD_variance=np.zeros(10);PMD_variance2=np.zeros(10);
    PMD_spice_ls=np.zeros(10);PMD_spice=np.zeros(10);PMD_unknownA=np.zeros(10);
    
    mmv=multiple_measurement_vector();

    for snr_iter in np.arange(10):
        print(SNR[snr_iter])
        mse_cv=0;mse_sparsity=0;mse_alpha1=0;mse_alpha2=0;mse_variance=0;mse_variance2=0;mse_spice=0;mse_spice_ls=0
        pe_cv=0;pe_sparsity=0;pe_alpha1=0;pe_alpha2=0;pe_variance=0;pe_variance2=0;pe_spice=0;pe_spice_ls=0;
        recall_cv=0;recall_sparsity=0;recall_alpha1=0;recall_alpha2=0;recall_variance=0;recall_variance2=0;
        recall_spice=0;recall_spice_ls=0;
        precision_cv=0;precision_sparsity=0;precision_alpha1=0;precision_alpha2=0;precision_variance=0;
        precision_spice=0;precision_variance2=0;precision_spice_ls=0;
        pmd_cv=0;pmd_sparsity=0;pmd_alpha1=0;pmd_alpha2=0;pmd_variance=0;pmd_variance2=0;pmd_spice=0;pmd_spice_ls=0;
        pfd_cv=0;pfd_sparsity=0;pfd_alpha1=0;pfd_alpha2=0;pfd_variance=0;pfd_variance2=0;pfd_spice=0;pfd_spice_ls=0;

        for num in np.arange(num_iter):
            if matrix_type=='normal':
                X=np.random.randn(n,p)/np.sqrt(n)
            elif matrix_type=='two_ortho':
                X=np.hstack([np.eye(n),hadamard(n)/np.sqrt(n)])
            else:
                print('invalid matrix type')
                raise Exception('invalid matrix type. matrix type must be one of normal or two_ortho')
            support_true=np.random.choice(p,k_0,False).tolist()
            Beta_true=np.zeros((p,L));Beta_true[support_true,:]=np.sign(np.random.randn(k_0,L))
            noise_var=k_0/(n*snr[snr_iter])
            noise=np.random.randn(n,L)*np.sqrt(noise_var)
            Y=np.matmul(X,Beta_true)+noise
            
            
            # GRRT
            rrt_somp_dict=mmv.compute_signal_and_support(X=X,Y=Y,alpha_list=[0.1,0.01])
            support_estimate_alpha1,Beta_est_alpha1=rrt_somp_dict[0.1]['support_estimate'],rrt_somp_dict[0.1]['signal_estimate']
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_alpha1,
                                                                                      Beta_true,Beta_est_alpha1)
            mse_alpha1+=l2_error;pe_alpha1+=support_error;recall_alpha1+=recall;precision_alpha1+=precision;
            pmd_alpha1+=pmd;pfd_alpha1+=pfd

            support_estimate_alpha2,Beta_est_alpha2=rrt_somp_dict[0.01]['support_estimate'],rrt_somp_dict[0.01]['signal_estimate']
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_alpha2,
                                                                                Beta_true,Beta_est_alpha2)
            mse_alpha2+=l2_error;pe_alpha2+=support_error;recall_alpha2+=recall;precision_alpha2+=precision;
            pmd_alpha2+=pmd;pfd_alpha2+=pfd

            
            ## SPICE algorithm
            # SPICE solution is not sparse. We first seelct rows whose avg_power>0.01. 
            # We then select rows whose power is atleast 20% higher than max power per row.
            Beta_est,p_current,sigma_est=spice(X,Y)
            power_per_row=np.linalg.norm(Beta_est,axis=1)**2/L
            
            support_est=np.where(power_per_row>1e-2)[0]

            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_spice+=l2_error;pe_spice+=support_error;recall_spice+=recall;precision_spice+=precision;
            pmd_spice+=pmd;pfd_spice+=pfd
            
            
            max_power=np.max(power_per_row)
            support_est=np.where(power_per_row.flatten()>0.2*max_power)[0]
            beta_est=np.matmul(np.linalg.pinv(X[:,support_est]),Y)
            Beta_est_ls=np.zeros((p,L))
            Beta_est_ls[support_est,:]=beta_est
            
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est_ls)
            mse_spice_ls+=l2_error;pe_spice_ls+=support_error;recall_spice_ls+=recall;precision_spice_ls+=precision;
            pmd_spice_ls+=pmd;pfd_spice_ls+=pfd
            

            support_est,Beta_est=SOMP_prior_sparsity(X,Y,k_0)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_sparsity+=l2_error;pe_sparsity+=support_error;recall_sparsity+=recall;precision_sparsity+=precision;
            pmd_sparsity+=pmd;pfd_sparsity+=pfd
            


            threshold=np.sqrt(noise_var)*np.sqrt(n*L+2*np.sqrt(n*L*np.log(n*L)))
            support_est,Beta_est=SOMP_prior_variance(X,Y,threshold)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_variance+=l2_error;pe_variance+=support_error;recall_variance+=recall;precision_variance+=precision;
            pmd_variance+=pmd;pfd_variance+=pfd


            threshold=np.linalg.norm(noise)
            support_est,Beta_est=SOMP_prior_variance(X,Y,threshold)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_variance2+=l2_error;pe_variance2+=support_error;recall_variance2+=recall;precision_variance2+=precision;
            pmd_variance2+=pmd;pfd_variance2+=pfd

            CV_dict,support_est,Beta_est=SOMP_CV(X,Y,cv_fraction=0.1)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_cv+=l2_error;pe_cv+=support_error;recall_cv+=recall;precision_cv+=precision;  
            pmd_cv+=pmd;pfd_cv+=pfd


        MSE_cv[snr_iter]=mse_cv/num_iter;MSE_sparsity[snr_iter]=mse_sparsity/num_iter;
        MSE_alpha1[snr_iter]=mse_alpha1/num_iter;MSE_alpha2[snr_iter]=mse_alpha2/num_iter;
        MSE_variance[snr_iter]=mse_variance/num_iter;MSE_spice[snr_iter]=mse_spice/num_iter;
        MSE_variance2[snr_iter]=mse_variance2/num_iter;MSE_spice_ls[snr_iter]=mse_spice_ls/num_iter;

        PE_cv[snr_iter]=pe_cv/num_iter;PE_sparsity[snr_iter]=pe_sparsity/num_iter;
        PE_alpha1[snr_iter]=pe_alpha1/num_iter;PE_alpha2[snr_iter]=pe_alpha2/num_iter;
        PE_variance[snr_iter]=pe_variance/num_iter; PE_variance2[snr_iter]=pe_variance2/num_iter; 
        PE_spice[snr_iter]=pe_spice/num_iter;PE_spice_ls[snr_iter]=pe_spice_ls/num_iter;
        
        PMD_cv[snr_iter]=pmd_cv/num_iter;PMD_sparsity[snr_iter]=pmd_sparsity/num_iter;
        PMD_alpha1[snr_iter]=pmd_alpha1/num_iter;PMD_alpha2[snr_iter]=pmd_alpha2/num_iter;
        PMD_variance[snr_iter]=pmd_variance/num_iter; PMD_variance2[snr_iter]=pmd_variance2/num_iter; 
        PMD_spice[snr_iter]=pmd_spice/num_iter;PMD_spice_ls[snr_iter]=pmd_spice_ls/num_iter;
        
        PFD_cv[snr_iter]=pfd_cv/num_iter;PFD_sparsity[snr_iter]=pfd_sparsity/num_iter;
        PFD_alpha1[snr_iter]=pfd_alpha1/num_iter;PFD_alpha2[snr_iter]=pfd_alpha2/num_iter;
        PFD_variance[snr_iter]=pfd_variance/num_iter; PFD_variance2[snr_iter]=pfd_variance2/num_iter; 
        PFD_spice[snr_iter]=pfd_spice/num_iter;PFD_spice_ls[snr_iter]=pfd_spice_ls/num_iter;

        Recall_cv[snr_iter]=recall_cv/num_iter;Recall_sparsity[snr_iter]=recall_sparsity/num_iter;
        Recall_alpha1[snr_iter]=recall_alpha1/num_iter;Recall_alpha2[snr_iter]=recall_alpha2/num_iter;
        Recall_variance[snr_iter]=recall_variance/num_iter;Recall_spice[snr_iter]=recall_spice/num_iter;
        Recall_variance2[snr_iter]=recall_variance2/num_iter;Recall_spice_ls[snr_iter]=recall_spice_ls/num_iter;


        Precision_cv[snr_iter]=precision_cv/num_iter;Precision_sparsity[snr_iter]=precision_sparsity/num_iter;
        Precision_alpha1[snr_iter]=precision_alpha1/num_iter;Precision_alpha2[snr_iter]=precision_alpha2/num_iter;
        Precision_variance[snr_iter]=precision_variance/num_iter;Precision_variance2[snr_iter]=precision_variance2/num_iter;
        Precision_spice[snr_iter]=precision_spice/num_iter;Precision_spice_ls[snr_iter]=precision_spice_ls/num_iter;



    print(' experiment over')   
    print('saving results')
    results={}
    results['algo']='SOMP'
    results['experiment_type']='SNR_sweep'
    results['num_iter']=num_iter
    results['n']=n
    results['p']=p
    results['L']=L
    results['k_0']=k_0
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
    results['Recall_spice_ls']=Recall_spice_ls.tolist();
    results['Recall_sparsity']=Recall_sparsity.tolist();results['Recall_alpha1']=Recall_alpha1.tolist(); 
    results['Recall_alpha2']=Recall_alpha2.tolist(); 
    results['Precision_cv']=Precision_cv.tolist();results['Precision_variance']=Precision_variance.tolist();
    results['Precision_variance2']=Precision_variance2.tolist();results['Precision_spice']=Precision_spice.tolist();
    results['Precision_spice_ls']=Precision_spice_ls.tolist();
    results['Precision_sparsity']=Precision_sparsity.tolist();results['Precision_alpha1']=Precision_alpha1.tolist();
    results['Precision_alpha2']=Precision_alpha2.tolist(); 
    
    results['PMD_cv']=PMD_cv.tolist();results['PMD_variance']=PMD_variance.tolist();
    results['PMD_variance2']=PMD_variance2.tolist();results['PMD_spice']=PMD_spice.tolist();
    results['PMD_spice_ls']=PMD_spice_ls.tolist();
    results['PMD_sparsity']=PMD_sparsity.tolist();results['PMD_alpha1']=PMD_alpha1.tolist();
    results['PMD_alpha2']=PMD_alpha2.tolist(); 
    
    results['PFD_cv']=PFD_cv.tolist();results['PFD_variance']=PFD_variance.tolist();
    results['PFD_variance2']=PFD_variance2.tolist();results['PFD_spice']=PFD_spice.tolist();
    results['PFD_spice_ls']=PFD_spice_ls.tolist();
    results['PFD_sparsity']=PFD_sparsity.tolist();results['PFD_alpha1']=PFD_alpha1.tolist();
    results['PFD_alpha2']=PFD_alpha2.tolist(); 
    
    
    file_name='SOMP_SNR_sweep_'+matrix_type+'_.json'
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
    

    
    
    
    
    
    

    