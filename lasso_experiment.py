import argparse
import numpy as np
import matplotlib.pyplot as plt
import os,sys,json
import scipy as sci
from rrt_codes.single_measurement_vector import single_measurement_vector ## GRRT class
from scipy.linalg import hadamard
from scipy.stats import norm
from sklearn import linear_model
from sklearn.linear_model import LassoCV,lasso_path


def lasso_variance(X,y,noise_var):
    #
    n,p=X.shape
    alpha=2*np.sqrt(noise_var)*np.sqrt(2*np.log(p))/n # same as lambda in GRRT paper. The difference in scaling is to accomodate for the difference between Sklearn, GRRT and Candes et al  definition of LASSO.  
    reg= linear_model.Lasso(alpha=alpha,fit_intercept=False,normalize=False)
    reg=reg.fit(X,y)
    Beta_est=reg.coef_
    return Beta_est

def lasso_cv(X,y):
    n,p=X.shape
    reg = LassoCV(cv=5, random_state=0,fit_intercept=False,normalize=False).fit(X, y) # run 5 fold cv
    reg1= linear_model.Lasso(alpha=reg.alpha_,fit_intercept=False,normalize=False).fit(X,y) # run lasso using the 
    #regularization parameter selected by 5 fold cv. 
    return reg1.coef_

def lasso_sparsity(X,y,k_0):
    # returns the first k_0 unique terms entering  the lasso regularization path
    n,p=X.shape
    _, _, coefs = linear_model.lars_path(X, y.reshape((n,)), method='lasso', verbose=False)
    nfeatures,nodes=coefs.shape
    support=set()


    for k in np.arange(nodes):
        support_k=np.where(np.abs(coefs[:,k])>1e-4)[0]
        support=support.union(set(support_k.tolist()))
        if len(support)==k_0:
            break;
    support=list(support)    
    Xnew=X[:,support].reshape((n,len(support)))
    beta_ls=np.matmul(np.linalg.pinv(Xnew),y).reshape((len(support),1))
    beta_est=np.zeros(p)
    beta_est[support]=beta_ls.flatten()
    beta_est=beta_est.reshape((p,1))
    return beta_est,support

def EFIC(X,y,gamma_list=[2],column_type='unit_l2'):
    # compute EFIC for a list of gamma values.
    # larger gamma means better precision at lower recall. 
    
    n,p=X.shape
    #EFIC params. $a$ deals with how fast column_norm_squared in X increases with sample size n
    if column_type=='unit_l2':
        a=0; # column norm_squared=1
    elif column_type=='n_l2': # column norn = number of measurements
        a=1; # column norm squared=n
    else:
        avg_norm=np.linalg.norm(X)/np.sqrt(p) #norm of all columns
        a=np.log(avg_norm**2)/np.log(n)
    
    d=np.log(p)/np.log(n) # how fast the number of features increases with sample size. 
    
    
    #compute LASSO regularization path
    _, _, coefs = linear_model.lars_path(X, y.reshape((n,)), method='lasso', verbose=False)
    nfeatures,nodes=coefs.shape
    support_estimate_sequence=[]
    cost_efic=[]
    support_list=[]
    beta_list=[]


    for k in np.arange(nodes):
        support_k=np.where(np.abs(coefs[:,k])>1e-4)[0]
        support_estimate_sequence.append(support_k.tolist())
        Xnew=X[:,support_k].reshape((n,len(support_k)))
        beta_ls=np.matmul(np.linalg.pinv(Xnew),y).reshape((len(support_k),1))
        beta_est=np.zeros(p)
        beta_est[support_k]=beta_ls.flatten()
        beta_list.append(beta_est)
        t1=(n-len(support_k)-2)*np.log(np.linalg.norm(y-np.matmul(Xnew,beta_ls))**2)
        t2=np.log(np.linalg.det(np.matmul(Xnew.T,Xnew)))
        cost_efic.append(t1+t2)
        support_list.append(support_k)
    result_dict={}
    # currently efic contains cost associated with terms independent of gamma. Hence, common to all values of gamma. 
    # now add gamma dependednt terms. 
    for gamma_ind in np.arange(len(gamma_list)):
        gamma=gamma_list[gamma_ind]
        c=1-a/(2*d)+gamma/d
        cost_gamma=[]
        for k in np.arange(len(cost_efic)):
            t3=(1+2*c*d)*len(support_list[k])*np.log(n)
            cost_gamma.append(t3+cost_efic[k])
        min_ind=np.argmin(cost_gamma)
        beta_est=beta_list[min_ind].reshape((p,1))
        support_estimate=support_list[min_ind]
        result_dict[gamma_ind]={'signal_estimate':beta_est,'support_estimate':support_estimate}
    return result_dict
        
def scaled_lasso(X,y):
    #assumes a design matrix with unit l2_norm columns. Internally, we rescale it to obtain sqrt{n} normalized columns
    n,p=X.shape
    X_scaled=np.sqrt(n)*X #rescaling to n_l2 norm
    lambda_0=np.sqrt(2*np.log(p))/np.sqrt(n) ## larger lambda gives better precision at lower recall
    max_iter=100; tol=1e-4
    Beta_est=np.matmul(X.T,y)
    cost=[]
    for k in np.arange(max_iter):
        sigma_est=np.linalg.norm(y-np.matmul(X_scaled,Beta_est))/np.sqrt(n) # noise std estimate
        lambda_scaled=sigma_est*lambda_0
        reg=linear_model.Lasso(alpha=lambda_scaled,fit_intercept=False,normalize=False,warm_start=True).fit(X_scaled,y)
        Beta_est_new=reg.coef_.reshape((p,1)) # LASSO estimate
        
        ## components of cost function
        a=np.linalg.norm(y-np.matmul(X_scaled,Beta_est_new))**2/(2*n*sigma_est)
        b=sigma_est/2
        c=lambda_0*np.linalg.norm(Beta_est,1)
        
        if np.linalg.norm(Beta_est_new-Beta_est)<tol:
            break;
        cost.append(a+b+c)
        Beta_est=Beta_est_new
    return Beta_est*np.sqrt(n),sigma_est,cost # rescaling is necessary since we scaled X.

def gen_spice(X,y,q=2):
    #larger value of q gives better precision at lower recall. q=2 as suggested in the paper. 
    n=X.shape[0];p=X.shape[1];
    max_iter=1000; tol=1e-4
    current_iter=0
    
    ### precomputation
    A=np.hstack([X,np.eye(n)])
    A_k_norm=np.linalg.norm(A,axis=0).tolist()

    w_vector=np.zeros(p+n)
    y_norm=np.linalg.norm(y)
    for k in np.arange(n+p):
        w_vector[k]=np.linalg.norm(A[:,k])**2/y_norm**2
    p_current=np.zeros(n+p)
    sigma_current=np.linalg.norm(y-np.mean(y))/np.sqrt(n-1)
    for k in np.arange(p):
        p_current[k]=np.matmul(X[:,k].T,y)**2/np.linalg.norm(X[:,k])**4
    p_current[p:]=sigma_current
    
    while current_iter<max_iter:
        current_iter+=1
        P=np.diag(p_current)
        R=np.matmul(np.matmul(A,P),A.T)
        R_inv=np.linalg.inv(R)
        Beta_est=np.matmul(np.matmul(P,A.T),np.matmul(R_inv,y)).flatten()
        lambda_term1=np.linalg.norm(np.matmul(np.diag(w_vector[:p]),Beta_est[:p].reshape((p,1))),1)
        lambda_term2=np.linalg.norm(n**(1/(2*q))*Beta_est[p:])
        lambda_sqrt=lambda_term1+lambda_term2
        p_next=np.zeros(n+p)
        for k in np.arange(p):
            p_next[k]=np.abs(Beta_est[k])/(lambda_sqrt*np.sqrt(w_vector[k]))
        sigma_next=np.linalg.norm(Beta_est[p:])/(n**(1/(2*q))*lambda_sqrt)
        p_next[p:]=sigma_next
        if np.linalg.norm(p_current-p_next)<tol:
            break
        p_current=p_next
        
    return Beta_est[:p].reshape((p,1)),p_current,sigma_current


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


def run_experiment(num_iter=100,experiment_type='SNR_sweep',matrix_type='two_ortho'):
    if experiment_type=='SNR_sweep':
        N,P,K_0=64,128,4
        SNR=[x for x in np.linspace(0,30,10)]
        snr_array=[10**(x/10) for x in SNR]# snr varies. evrything else constant
    elif experiment_type=='sample_sweep':
        SNR=[10]; # SNR in db
        snr=10**(SNR[0]/10) # SNR in real scale
        P=400;K_0=5;
        N=[np.int(x) for x in np.linspace(25,200,10)] # sample size varies
    elif experiment_type=='sparsity_sweep':
        SNR=[10]
        snr=10**(SNR[0]/10)
        N=200;P=400;
        K_0=[np.int(x) for x in np.linspace(5,50,10)] #sparsity varies
    else:
        print(invalid_experiment_type)
        

        
    num_iter=num_iter
    
    MSE_cv=np.zeros(10);MSE_sparsity=np.zeros(10);MSE_alpha1=np.zeros(10);MSE_alpha2=np.zeros(10);MSE_variance=np.zeros(10);
    MSE_efic=np.zeros(10);MSE_sq=np.zeros(10);MSE_spice=np.zeros(10);MSE_unknownA=np.zeros(10)

    PE_cv=np.zeros(10);PE_sparsity=np.zeros(10);PE_alpha1=np.zeros(10);PE_alpha2=np.zeros(10);PE_variance=np.zeros(10);
    PE_efic=np.zeros(10);PE_sq=np.zeros(10);PE_spice=np.zeros(10);PE_unknownA=np.zeros(10)

    Recall_cv=np.zeros(10);Recall_sparsity=np.zeros(10);Recall_alpha1=np.zeros(10);Recall_alpha2=np.zeros(10);
    Recall_variance=np.zeros(10);Recall_efic=np.zeros(10);
    Recall_sq=np.zeros(10);Recall_spice=np.zeros(10);Recall_unknownA=np.zeros(10)

    Precision_cv=np.zeros(10);Precision_sparsity=np.zeros(10);Precision_alpha1=np.zeros(10);
    Precision_alpha2=np.zeros(10);Precision_variance=np.zeros(10);Precision_efic=np.zeros(10);
    Precision_sq=np.zeros(10);Precision_spice=np.zeros(10);Precision_unknownA=np.zeros(10);
    
    PFD_cv=np.zeros(10);PFD_sparsity=np.zeros(10);PFD_alpha1=np.zeros(10);
    PFD_alpha2=np.zeros(10);PFD_variance=np.zeros(10);PFD_efic=np.zeros(10);
    PFD_sq=np.zeros(10);PFD_spice=np.zeros(10);PFD_unknownA=np.zeros(10);
    
    PMD_cv=np.zeros(10);PMD_sparsity=np.zeros(10);PMD_alpha1=np.zeros(10);
    PMD_alpha2=np.zeros(10);PMD_variance=np.zeros(10);PMD_efic=np.zeros(10);
    PMD_sq=np.zeros(10);PMD_spice=np.zeros(10);PMD_unknownA=np.zeros(10);
    
    
    smv=single_measurement_vector(); # initializing GRRT

    for snr_iter in np.arange(10):
        
        if experiment_type=='sample_sweep':
            n=N[snr_iter]
            p=P
            snr=snr
            k_0=K_0
            print('sample size {}'.format(n))
        elif experiment_type=='SNR_sweep':
            snr=snr_array[snr_iter]
            n=N;p=P;k_0=K_0
            print('SNR:{}'.format(SNR[snr_iter]))
        elif experiment_type=='sparsity_sweep':
            n=N;p=P; snr=snr;
            k_0=K_0[snr_iter]
            print('sparsity:{}'.format(k_0))
        else:
            print('invalid experiment')
            
        mse_cv=0;mse_sparsity=0;mse_alpha1=0;mse_alpha2=0;mse_variance=0;mse_efic=0;mse_sq=0;mse_spice=0;mse_unknownA=0;
        pe_cv=0;pe_sparsity=0;pe_alpha1=0;pe_alpha2=0;pe_variance=0;pe_efic=0;pe_sq=0;pe_spice=0;pe_unknownA=0;
        recall_cv=0;recall_sparsity=0;recall_alpha1=0;recall_alpha2=0;recall_variance=0;
        recall_efic=0;recall_sq=0;recall_spice=0;recall_unknownA=0;
        precision_cv=0;precision_sparsity=0;precision_alpha1=0;precision_alpha2=0;precision_variance=0;
        precision_efic=0;precision_sq=0;precision_spice=0;precision_unknownA=0;
        pfd_cv=0;pfd_sparsity=0;pfd_alpha1=0;pfd_alpha2=0;pfd_variance=0;pfd_efic=0;pfd_sq=0;pfd_spice=0;pfd_unknownA=0;
        pmd_cv=0;pmd_sparsity=0;pmd_alpha1=0;pmd_alpha2=0;pmd_variance=0;pmd_efic=0;pmd_sq=0;pmd_spice=0;pmd_unknownA=0;
        for num in np.arange(num_iter):
            
                
            if matrix_type=='two_ortho':
                X=np.hstack([np.eye(n),hadamard(n)/np.sqrt(n)])
            elif matrix_type=='normal':
                X=np.random.randn(n,p)/np.sqrt(n)
            else:
                raise Exception('invalid matrix type')
                
            support_true=np.random.choice(p,k_0,False).tolist()
            Beta_true=np.zeros(p);Beta_true[support_true]=np.squeeze(np.sign(np.random.randn(k_0)))
            Beta_true=Beta_true.reshape((p,1))
            
            noise_var=k_0/(n*snr)
            noise=np.random.randn(n,1)*np.sqrt(noise_var)
            y=np.matmul(X,Beta_true)+noise

    #         ## LASSO with 5 fold CV
            Beta_est_cv=lasso_cv(X,y)
            Beta_est_cv=np.squeeze(Beta_est_cv)
            support_estimate_cv=np.where(np.abs(Beta_est_cv)>1e-4)[0] # support
            Xnew=X[:,support_estimate_cv] 
            beta_t=np.matmul(np.linalg.pinv(Xnew),y).flatten()
            Beta_est_cv_ls=np.zeros(p);Beta_est_cv_ls[support_estimate_cv]=beta_t # LS estimate of only support

            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_cv,
                                                                  Beta_true,Beta_est_cv)
            mse_cv+=l2_error;pe_cv+=support_error;recall_cv+=recall;precision_cv+=precision;
            pmd_cv+=pmd;pfd_cv+=pfd

            ### LASSO using noise variance as in Candes et al [Near Ideal Model selection .... ]
            ### E. J. Cand`es, Y. Plan et al., “Near-ideal model selection by l1 minimization,”
            ### Ann. Stat., vol. 37, no. 5A, pp. 2145–2177, 2009.
            ### LASSO(\sigma^2)
            Beta_est_pv=lasso_variance(X,y,noise_var=noise_var)
            Beta_est_pv=np.squeeze(Beta_est_pv)
            support_estimate_pv=np.where(np.abs(Beta_est_pv)>1e-4)[0]
            Xnew=X[:,support_estimate_pv]
            beta_t=np.matmul(np.linalg.pinv(Xnew),y).flatten()
            Beta_est_pv_ls=np.zeros(p);Beta_est_pv_ls[support_estimate_pv]=beta_t
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_pv,
                                                                                Beta_true,Beta_est_pv_ls)

            mse_variance+=l2_error;pe_variance+=support_error;recall_variance+=recall;precision_variance+=precision;
            pmd_variance+=pmd;pfd_variance+=pfd
            
            
            
            ## LASSO with prior knowledge of sparsity. Based on LASSO regulariation path
            # LASSO(K_{row})
            Beta_est_ps,support_estimate_ps=lasso_sparsity(X,y,k_0)


            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_ps,
                                                                                Beta_true,Beta_est_ps)
            mse_sparsity+=l2_error;pe_sparsity+=support_error;recall_sparsity+=recall;precision_sparsity+=precision;
            pmd_sparsity+=pmd;pfd_sparsity+=pfd

           # LASSO using GRRT
            rrt_lasso_dict=smv.compute_signal_and_support(X=X,Y=y,algorithm='LASSO',alpha_list=[0.1,0.01])
            
            #GRRT with $\alpha=0.1$
            support_estimate_alpha1,Beta_est_alpha1=rrt_lasso_dict[0.1]['support_estimate'],rrt_lasso_dict[0.1]['signal_estimate']
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_alpha1,
                                                                                      Beta_true,Beta_est_alpha1)
            mse_alpha1+=l2_error;pe_alpha1+=support_error;recall_alpha1+=recall;precision_alpha1+=precision;
            pmd_alpha1+=pmd;pfd_alpha1+=pfd

            ## LASSO using GRRT 
            support_estimate_alpha2,Beta_est_alpha2=rrt_lasso_dict[0.01]['support_estimate'],rrt_lasso_dict[0.01]['signal_estimate']
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_alpha2,
                                                                                Beta_true,Beta_est_alpha2)
            mse_alpha2+=l2_error;pe_alpha2+=support_error;recall_alpha2+=recall;precision_alpha2+=precision;
            pmd_alpha2+=pmd;pfd_alpha2+=pfd

            
            ### scaled lasso. 
            ### T. Sun and C.-H. Zhang, “Scaled sparse linear regression,” Biometrika,
            ### vol. 99, no. 4, pp. 879–898, 2012
            Beta_est_sq,sigma_est,cost=scaled_lasso(X,y)  
            Beta_est_sq=np.squeeze(Beta_est_sq)
            support_estimate_sq=np.where(np.abs(Beta_est_sq)>1e-4)[0]
            Xnew=X[:,support_estimate_sq]
            beta_t=np.matmul(np.linalg.pinv(Xnew),y).flatten()
            Beta_est_sq_ls=np.zeros(p);Beta_est_sq_ls[support_estimate_sq]=beta_t
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_sq,
                                                                                Beta_true,Beta_est_sq_ls)
            mse_sq+=l2_error;pe_sq+=support_error;recall_sq+=recall;precision_sq+=precision;
            pmd_sq+=pmd;pfd_sq+=pfd
            
            
            # Generalized SPICE using q=2
            ## J. Sw¨ard, S. I. Adalbj¨ornsson, and A. Jakobsson, “Generalized sparse
            ## covariance-based estimation,” Signal Processing, vol. 143, pp. 311–319,
            ## 2018.
            Beta_est_spice,p_current,sigma_current=gen_spice(X,y,q=2)
            beta_est=Beta_est_spice.flatten()
            bmax=np.max(np.abs(beta_est))
            support_estimate=np.where(np.abs(beta_est)>0.2*bmax)[0]
            beta_est=np.matmul(np.linalg.pinv(X[:,support_estimate]),y).flatten()
            Beta_est_ls=np.zeros(p)
            Beta_est_ls[support_estimate]=beta_est
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate,
                                                                                Beta_true,Beta_est_ls.reshape((p,1)))
            mse_spice+=l2_error;pe_spice+=support_error;recall_spice+=recall;precision_spice+=precision;
            pmd_spice+=pmd;pfd_spice+=pfd
            
            
            # Extended Fisher Information criteria (EFIC)
            # A. Owrang and M. Jansson, “A model selection criterion for highdimensional
            #  linear regression,” IEEE Trans. Signal Process., vol. 66,
            # no. 13, pp. 3436–3446, July 2018.
            efic_gamma=[2]
            efic_result_dict=EFIC(X,y,efic_gamma)
            Beta_est_efic,support_estimate_efic=efic_result_dict[0]['signal_estimate'],efic_result_dict[0]['support_estimate']

            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_estimate_efic,
                                                                                 Beta_true,Beta_est_efic)
            mse_efic+=l2_error;pe_efic+=support_error;recall_efic+=recall;precision_efic+=precision;
            pmd_efic+=pmd;pfd_efic+=pfd



        MSE_cv[snr_iter]=mse_cv/num_iter;MSE_sparsity[snr_iter]=mse_sparsity/num_iter;
        MSE_alpha1[snr_iter]=mse_alpha1/num_iter;MSE_alpha2[snr_iter]=mse_alpha2/num_iter;
        MSE_variance[snr_iter]=mse_variance/num_iter;MSE_sq[snr_iter]=mse_sq/num_iter;MSE_spice[snr_iter]=mse_spice/num_iter
        MSE_efic[snr_iter]=mse_efic/num_iter;MSE_unknownA[snr_iter]=mse_unknownA/num_iter;
        
        PE_cv[snr_iter]=pe_cv/num_iter;PE_sparsity[snr_iter]=pe_sparsity/num_iter;
        PE_alpha1[snr_iter]=pe_alpha1/num_iter;PE_alpha2[snr_iter]=pe_alpha2/num_iter;
        PE_variance[snr_iter]=pe_variance/num_iter;PE_sq[snr_iter]=pe_sq/num_iter;PE_spice[snr_iter]=pe_spice/num_iter;
        PE_efic[snr_iter]=pe_efic/num_iter;PE_unknownA[snr_iter]=pe_unknownA/num_iter;
        
        Recall_cv[snr_iter]=recall_cv/num_iter;Recall_sparsity[snr_iter]=recall_sparsity/num_iter;
        Recall_alpha1[snr_iter]=recall_alpha1/num_iter;Recall_alpha2[snr_iter]=recall_alpha2/num_iter;
        Recall_variance[snr_iter]=recall_variance/num_iter;Recall_sq[snr_iter]=recall_sq/num_iter;
        Recall_spice[snr_iter]=recall_spice/num_iter;
        Recall_efic[snr_iter]=recall_efic/num_iter;Recall_unknownA[snr_iter]=recall_unknownA/num_iter;
        

        Precision_cv[snr_iter]=precision_cv/num_iter;Precision_sparsity[snr_iter]=precision_sparsity/num_iter;
        Precision_alpha1[snr_iter]=precision_alpha1/num_iter;Precision_alpha2[snr_iter]=precision_alpha2/num_iter;
        Precision_variance[snr_iter]=precision_variance/num_iter;
        Precision_sq[snr_iter]=precision_sq/num_iter;Precision_spice[snr_iter]=precision_spice/num_iter;
        Precision_efic[snr_iter]=precision_efic/num_iter;Precision_unknownA[snr_iter]=precision_unknownA/num_iter;
        
        PMD_cv[snr_iter]=pmd_cv/num_iter;PMD_sparsity[snr_iter]=pmd_sparsity/num_iter;
        PMD_alpha1[snr_iter]=pmd_alpha1/num_iter;PMD_alpha2[snr_iter]=pmd_alpha2/num_iter;
        PMD_variance[snr_iter]=pmd_variance/num_iter;PMD_sq[snr_iter]=pmd_sq/num_iter;PMD_spice[snr_iter]=pmd_spice/num_iter;
        PMD_efic[snr_iter]=pmd_efic/num_iter;PMD_unknownA[snr_iter]=pmd_unknownA/num_iter;
        
        PFD_cv[snr_iter]=pfd_cv/num_iter;PFD_sparsity[snr_iter]=pfd_sparsity/num_iter;
        PFD_alpha1[snr_iter]=pfd_alpha1/num_iter;PFD_alpha2[snr_iter]=pfd_alpha2/num_iter;
        PFD_variance[snr_iter]=pfd_variance/num_iter;PFD_sq[snr_iter]=pfd_sq/num_iter;PFD_spice[snr_iter]=pfd_spice/num_iter;
        PFD_efic[snr_iter]=pfd_efic/num_iter;PFD_unknownA[snr_iter]=pfd_unknownA/num_iter;
        

 
    print('experiment over')   
    print('saving results')
    results={}
    results['algo']='LASSO'
    results['experiment_type']=experiment_type
    results['num_iter']=num_iter
    results['n']=N
    results['p']=P
    results['k_0']=K_0
    results['SNR']=SNR
    results['matrix_type']=matrix_type
    results['sigmal_type']='pm1' #plus or minus 1.
    
    results['MSE_cv']=MSE_cv.tolist();results['MSE_variance']=MSE_variance.tolist();
    results['MSE_spice']=MSE_spice.tolist();results['MSE_sq']=MSE_sq.tolist();
    results['MSE_sparsity']=MSE_sparsity.tolist();results['MSE_alpha1']=MSE_alpha1.tolist(); results['MSE_alpha2']=MSE_alpha2.tolist();
    results['MSE_efic']=MSE_efic.tolist();results['MSE_unknownA']=MSE_unknownA.tolist();
    
    results['PE_cv']=PE_cv.tolist();results['PE_variance']=PE_variance.tolist();
    results['PE_spice']=PE_spice.tolist();results['PE_efic']=PE_efic.tolist();results['PE_sq']=PE_sq.tolist();
    results['PE_sparsity']=PE_sparsity.tolist();results['PE_alpha1']=PE_alpha1.tolist(); results['PE_alpha2']=PE_alpha2.tolist();
    results['PE_unknownA']=PE_unknownA.tolist();
    
    results['PFD_cv']=PFD_cv.tolist();results['PFD_variance']=PFD_variance.tolist();
    results['PFD_spice']=PFD_spice.tolist();results['PFD_efic']=PFD_efic.tolist();results['PFD_sq']=PFD_sq.tolist();
    results['PFD_sparsity']=PFD_sparsity.tolist();results['PFD_alpha1']=PFD_alpha1.tolist(); results['PFD_alpha2']=PFD_alpha2.tolist();
    results['PFD_unknownA']=PFD_unknownA.tolist();
    
    results['PMD_cv']=PMD_cv.tolist();results['PMD_variance']=PMD_variance.tolist();
    results['PMD_spice']=PMD_spice.tolist();results['PMD_efic']=PMD_efic.tolist();results['PMD_sq']=PMD_sq.tolist();
    results['PMD_sparsity']=PMD_sparsity.tolist();results['PMD_alpha1']=PMD_alpha1.tolist(); results['PMD_alpha2']=PMD_alpha2.tolist();
    results['PMD_unknownA']=PMD_unknownA.tolist();
    
    
    results['Recall_cv']=Recall_cv.tolist();results['Recall_variance']=Recall_variance.tolist();
    results['Recall_spice']=Recall_spice.tolist();results['Recall_efic']=Recall_efic.tolist();
    results['Recall_sparsity']=Recall_sparsity.tolist();results['Recall_alpha1']=Recall_alpha1.tolist(); 
    results['Recall_alpha2']=Recall_alpha2.tolist(); 
    results['Recall_sq']=Recall_sq.tolist();results['Recall_unknownA']=Recall_unknownA.tolist();
    
    results['Precision_cv']=Precision_cv.tolist();results['Precision_variance']=Precision_variance.tolist();
    results['Precision_spice']=Precision_spice.tolist();results['Precision_efic']=Precision_efic.tolist();
    results['Precision_sparsity']=Precision_sparsity.tolist();results['Precision_alpha1']=Precision_alpha1.tolist();
    results['Precision_alpha2']=Precision_alpha2.tolist();
    results['Precision_sq']=Precision_sq.tolist(); results['Precision_unknownA']=Precision_unknownA.tolist();
    
    file_name='LASSO_'+experiment_type+'_'+matrix_type+'.json'
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
    parser.add_argument('--experiment_type', type=str, default='SNR_sweep',
                        help='matrix type: SNR_sweep,sample_sweep or sparsity_sweep?')
    
    
    args = parser.parse_args()
    print(args)
    run_experiment(num_iter=args.num_iter,matrix_type=args.matrix_type,experiment_type=args.experiment_type)

    


        
    
    
