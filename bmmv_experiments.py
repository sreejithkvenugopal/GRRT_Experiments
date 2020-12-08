import numpy as np
import matplotlib.pyplot as plt
import os,sys,json
import scipy as sci
from rrt_codes.block_multiple_measurement_vector import block_multiple_measurement_vector
from scipy.linalg import hadamard
import argparse

def generate_full_support_from_block_support(block_support,block_size=4):
    ind = []
    for i in block_support:
        ind=ind+[j for j in range(i * block_size, (i + 1) * block_size)]
    return ind

# BMMV with a priori knowledge of block sparsity
def BMMV_OMP_prior_sparsity(X,Y,k_block,l_block):
    n,p=X.shape;n_blocks=np.int(p/l_block); L=Y.shape[1]
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    res=Y
    block_support=[]
    full_support=[]
    for k in np.arange(k_block):
        corr=np.matmul(X.T,res)
        
        corr_norm_per_block=np.array([np.linalg.norm(corr[indices_per_block[k],:]) for k in np.arange(n_blocks)])
        block_ind=np.argmax(corr_norm_per_block)
        block_support.append(block_ind)
        full_support=full_support+indices_per_block[block_ind]
        X_new=X[:,full_support];
        try:
            beta_est=np.matmul(np.linalg.pinv(X_new),Y)
        except:
            break
        Beta_est=np.zeros((p,L))
        Beta_est[full_support,:]=beta_est
        
        res=Y-np.matmul(X,Beta_est)
    return Beta_est,full_support,block_support

## BMMV_OMP stopping iterations once residual norms drop below a threshold
def BMMV_OMP_prior_variance(X,Y,l_block,threshold):
    n,p=X.shape;n_blocks=np.int(p/l_block); L=Y.shape[1]
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    res=Y
    block_support=[]
    full_support=[]
    for k in np.arange(n_blocks):
        corr=np.matmul(X.T,res)
        
        corr_norm_per_block=np.array([np.linalg.norm(corr[indices_per_block[k],:]) for k in np.arange(n_blocks)])
        block_ind=np.argmax(corr_norm_per_block)
        block_support.append(block_ind)
        full_support=full_support+indices_per_block[block_ind]
        X_new=X[:,full_support];
        try:
            beta_est=np.matmul(np.linalg.pinv(X_new),Y)
        except:
            break
        Beta_est=np.zeros((p,L))
        Beta_est[full_support,:]=beta_est
        
        res=Y-np.matmul(X,Beta_est)
        if np.linalg.norm(res)<threshold:
            break;
    return Beta_est,full_support,block_support

def BMMV_OMP_CV(X,Y,l_block,cv_fraction):
    
    n,p=X.shape;n_blocks=np.int(p/l_block); L=Y.shape[1]
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
        
    n_cv=np.int(n*cv_fraction)
    indices=np.random.choice(n,n,False).tolist()
    ind_cv=indices[:n_cv]
    ind_train=indices[n_cv:]
    n_train=len(ind_train)
    
    Y_train=Y[ind_train].reshape((n_train,L)); Y_cv=Y[ind_cv].reshape((n_cv,L));
    X_train=X[ind_train,:]; X_cv=X[ind_cv,:]
    train_col_norms=np.linalg.norm(X_train,axis=0)+1e-8
    X_train=X_train/train_col_norms
    train_col_norms=train_col_norms.reshape((p,1))
    max_iter=np.int(np.floor(n_train/(l_block)))
    cv_error_list=[(np.linalg.norm(Y_cv)**2)/(n_cv*L)]
    train_error_list=[np.linalg.norm(Y_train)**2/(n_train*L)]
    min_cv_error=cv_error_list[0]   
    res=Y_train
    block_support=[]
    full_support=[]
    
    best_full_support=full_support
    best_block_support=block_support
    best_est=np.zeros((p,L))
  
    
    block_support=[]
    full_support=[]
    for k in np.arange(max_iter):
        corr=np.matmul(X_train.T,res)
        
        corr_norm_per_block=np.array([np.linalg.norm(corr[indices_per_block[k],:]) for k in np.arange(n_blocks)])
        block_ind=np.argmax(corr_norm_per_block)
        block_support.append(block_ind)
        full_support=full_support+indices_per_block[block_ind]
        X_new=X_train[:,full_support];
        try:
            beta_est=np.matmul(np.linalg.pinv(X_new),Y_train)
        except:
            break
        Beta_est=np.zeros((p,L))
        Beta_est[full_support,:]=beta_est
        
        res=Y_train-np.matmul(X_train,Beta_est)
    
        train_error_list.append(np.linalg.norm(res)**2/(n_train*L))
        Beta_scaled=Beta_est*train_col_norms
        res_cv=Y_cv-np.matmul(X_cv,Beta_scaled)
        cv_error=np.linalg.norm(res_cv)**2/(n_cv*L)
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

def run_experiment(num_iter,matrix_type):
    n,p,k_block,l_block,L=64,128,3,4,10
    n_blocks = np.int(p /l_block)
    indices_per_block={}
    for k in np.arange(n_blocks):
        indices_per_block[k]=[j for j in range(k * l_block, (k + 1) *l_block)]
    SNR=np.linspace(-15,15,10)
    snr=10**(SNR/10)# SNR in real scale
    num_iter=num_iter

    MSE_cv=np.zeros(10);MSE_sparsity=np.zeros(10);MSE_alpha1=np.zeros(10);MSE_alpha2=np.zeros(10);MSE_variance=np.zeros(10);
    MSE_spice=np.zeros(10)
    MSE_variance2=np.zeros(10);

    PE_cv=np.zeros(10);PE_sparsity=np.zeros(10);PE_alpha1=np.zeros(10);PE_alpha2=np.zeros(10);PE_variance=np.zeros(10);
    PE_spice=np.zeros(10);
    PE_variance2=np.zeros(10);
    
    PFD_cv=np.zeros(10);PFD_sparsity=np.zeros(10);PFD_alpha1=np.zeros(10);PFD_alpha2=np.zeros(10);PFD_variance=np.zeros(10);
    PFD_spice=np.zeros(10);
    PFD_variance2=np.zeros(10);

    PMD_cv=np.zeros(10);PMD_sparsity=np.zeros(10);PMD_alpha1=np.zeros(10);PMD_alpha2=np.zeros(10);PMD_variance=np.zeros(10);
    PMD_spice=np.zeros(10);
    PMD_variance2=np.zeros(10);


    Recall_cv=np.zeros(10);Recall_sparsity=np.zeros(10);Recall_alpha1=np.zeros(10);Recall_alpha2=np.zeros(10);
    Recall_variance=np.zeros(10);Recall_spice=np.zeros(10);
    Recall_variance2=np.zeros(10);

    Precision_cv=np.zeros(10);Precision_sparsity=np.zeros(10);Precision_alpha1=np.zeros(10);
    Precision_alpha2=np.zeros(10);Precision_variance=np.zeros(10);Precision_spice=np.zeros(10);
    Precision_variance2=np.zeros(10);

    bmmv=block_multiple_measurement_vector();





    for snr_iter in np.arange(10):
        print(SNR[snr_iter])
        mse_cv=0;mse_sparsity=0;mse_alpha1=0;mse_alpha2=0;mse_variance=0;mse_variance2=0;mse_spice=0
        pe_cv=0;pe_sparsity=0;pe_alpha1=0;pe_alpha2=0;pe_variance=0;pe_variance2=0;pe_spice=0;
        pfd_cv=0;pfd_sparsity=0;pfd_alpha1=0;pfd_alpha2=0;pfd_variance=0;pfd_variance2=0;pfd_spice=0;
        pmd_cv=0;pmd_sparsity=0;pmd_alpha1=0;pmd_alpha2=0;pmd_variance=0;pmd_variance2=0;pmd_spice=0;
        recall_cv=0;recall_sparsity=0;recall_alpha1=0;recall_alpha2=0;recall_variance=0;recall_variance2=0;
        recall_spice=0;
        precision_cv=0;precision_sparsity=0;precision_alpha1=0;precision_alpha2=0;precision_variance=0;
        precision_spice=0;precision_variance2=0;


        for num in np.arange(num_iter):
            #print(num)
            
            ### signal model
            if matrix_type=='two_ortho':
                X=np.hstack([np.eye(n),hadamard(n)/np.sqrt(n)])
            elif matrix_type=='normal':
                X=np.random.randn(n,p)/np.sqrt(n)
            else:
                raise Exception('Invalid matrix type')
            block_support= np.random.choice(np.arange(n_blocks), size=k_block, replace=False).tolist()
            support_true=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)
            Beta_true=np.zeros((p,L))
            Beta_true[support_true,:] = np.sign(np.random.randn(len(support_true), L))
            signal_power=len(support_true)
            # noise_power=nsamples*noisevar. snr=signal_power/noise_power
            noise_var = signal_power/ (n * snr[snr_iter])
            noise = np.random.randn(n, L) * np.sqrt(noise_var)
            Y= np.matmul(X, Beta_true) + noise

            
            # GRRT
            rrt_bmmv_dict=bmmv.compute_signal_and_support(X,Y=Y,block_size=l_block,alpha_list=[0.1,0.01])
            block_support,Beta_est_alpha1=rrt_bmmv_dict[0.1]['support_estimate'],rrt_bmmv_dict[0.1]['signal_estimate']
            support_est=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)

            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est_alpha1)
            mse_alpha1+=l2_error;pe_alpha1+=support_error;recall_alpha1+=recall;precision_alpha1+=precision;
            pmd_alpha1+=pmd;pfd_alpha1+=pfd;

            block_support,Beta_est_alpha2=rrt_bmmv_dict[0.01]['support_estimate'],rrt_bmmv_dict[0.01]['signal_estimate']
            support_est=generate_full_support_from_block_support(block_support=block_support,block_size=l_block)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                Beta_true,Beta_est_alpha2)
            mse_alpha2+=l2_error;pe_alpha2+=support_error;recall_alpha2+=recall;precision_alpha2+=precision;
            pmd_alpha2+=pmd;pfd_alpha2+=pfd;

            ##3 BMMV_OMP(K_block)
            Beta_est,support_est,block_support=BMMV_OMP_prior_sparsity(X,Y,k_block,l_block)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_sparsity+=l2_error;pe_sparsity+=support_error;recall_sparsity+=recall;precision_sparsity+=precision;
            pmd_sparsity+=pmd;pfd_sparsity+=pfd;
            
            
            ### BMMV_OMP(\sigma^2)

            threshold=np.sqrt(noise_var)*np.sqrt(n*L+2*np.sqrt(n*L*np.log(n*L)))
            Beta_est,support_est,block_support=BMMV_OMP_prior_variance(X,Y,l_block,threshold)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_variance+=l2_error;pe_variance+=support_error;recall_variance+=recall;precision_variance+=precision;
            pmd_variance+=pmd;pfd_variance+=pfd;
            
            ### BMMV_OMP(\|noise\|_F)

            threshold=np.linalg.norm(noise)
            Beta_est,support_est,block_support=BMMV_OMP_prior_variance(X,Y,l_block,threshold)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_variance2+=l2_error;pe_variance2+=support_error;recall_variance2+=recall;precision_variance2+=precision;
            pmd_variance2+=pmd;pfd_variance2+=pfd;

            Beta_est,support_est,best_block_support,CV_dict=BMMV_OMP_CV(X,Y,l_block,cv_fraction=0.1)
            support_error,l2_error,recall,precision,pmd,pfd=compute_error(support_true,support_est,
                                                                                      Beta_true,Beta_est)
            mse_cv+=l2_error;pe_cv+=support_error;recall_cv+=recall;precision_cv+=precision; 
            pmd_cv+=pmd;pfd_cv+=pfd;


        MSE_cv[snr_iter]=mse_cv/num_iter;MSE_sparsity[snr_iter]=mse_sparsity/num_iter;
        MSE_alpha1[snr_iter]=mse_alpha1/num_iter;MSE_alpha2[snr_iter]=mse_alpha2/num_iter;
        MSE_variance[snr_iter]=mse_variance/num_iter;MSE_spice[snr_iter]=mse_spice/num_iter;
        MSE_variance2[snr_iter]=mse_variance2/num_iter;

        PE_cv[snr_iter]=pe_cv/num_iter;PE_sparsity[snr_iter]=pe_sparsity/num_iter;
        PE_alpha1[snr_iter]=pe_alpha1/num_iter;PE_alpha2[snr_iter]=pe_alpha2/num_iter;
        PE_variance[snr_iter]=pe_variance/num_iter; PE_variance2[snr_iter]=pe_variance2/num_iter; 
        PE_spice[snr_iter]=pe_spice/num_iter;
        
        PMD_cv[snr_iter]=pmd_cv/num_iter;PMD_sparsity[snr_iter]=pmd_sparsity/num_iter;
        PMD_alpha1[snr_iter]=pmd_alpha1/num_iter;PMD_alpha2[snr_iter]=pmd_alpha2/num_iter;
        PMD_variance[snr_iter]=pmd_variance/num_iter; PMD_variance2[snr_iter]=pmd_variance2/num_iter; 
        PMD_spice[snr_iter]=pmd_spice/num_iter;
        
        PFD_cv[snr_iter]=pfd_cv/num_iter;PFD_sparsity[snr_iter]=pfd_sparsity/num_iter;
        PFD_alpha1[snr_iter]=pfd_alpha1/num_iter;PFD_alpha2[snr_iter]=pfd_alpha2/num_iter;
        PFD_variance[snr_iter]=pfd_variance/num_iter; PFD_variance2[snr_iter]=pfd_variance2/num_iter; 
        PFD_spice[snr_iter]=pfd_spice/num_iter;



        Recall_cv[snr_iter]=recall_cv/num_iter;Recall_sparsity[snr_iter]=recall_sparsity/num_iter;
        Recall_alpha1[snr_iter]=recall_alpha1/num_iter;Recall_alpha2[snr_iter]=recall_alpha2/num_iter;
        Recall_variance[snr_iter]=recall_variance/num_iter;Recall_spice[snr_iter]=recall_spice/num_iter;
        Recall_variance2[snr_iter]=recall_variance2/num_iter


        Precision_cv[snr_iter]=precision_cv/num_iter;Precision_sparsity[snr_iter]=precision_sparsity/num_iter;
        Precision_alpha1[snr_iter]=precision_alpha1/num_iter;Precision_alpha2[snr_iter]=precision_alpha2/num_iter;
        Precision_variance[snr_iter]=precision_variance/num_iter;Precision_variance2[snr_iter]=precision_variance2/num_iter;
        Precision_spice[snr_iter]=precision_spice/num_iter;
    print('over')
    print('over')
    print(' experiment over')   
    print('saving results')
    results={}
    results['algo']='BMMV_OMP'
    results['experiment_type']='SNR_sweep'
    results['num_iter']=num_iter
    results['n']=n
    results['p']=p
    results['l_block']=l_block
    results['k_block']=k_block
    results['L']=L
    results['SNR']=SNR.tolist()
    results['matrix_type']=matrix_type
    results['sigmal_type']='pm1' #plus or minus 1.
    results['MSE_cv']=MSE_cv.tolist();results['MSE_variance']=MSE_variance.tolist();results['MSE_variance2']=MSE_variance2.tolist();
    results['MSE_spice']=MSE_spice.tolist();
    results['MSE_sparsity']=MSE_sparsity.tolist();results['MSE_alpha1']=MSE_alpha1.tolist(); results['MSE_alpha2']=MSE_alpha2.tolist(); 
    results['PE_cv']=PE_cv.tolist();results['PE_variance']=PE_variance.tolist();
    results['PE_variance2']=PE_variance2.tolist();results['PE_spice']=PE_spice.tolist();
    results['PE_sparsity']=PE_sparsity.tolist();results['PE_alpha1']=PE_alpha1.tolist(); results['PE_alpha2']=PE_alpha2.tolist();
    
    results['PMD_cv']=PMD_cv.tolist();results['PMD_variance']=PMD_variance.tolist();
    results['PMD_variance2']=PMD_variance2.tolist();results['PMD_spice']=PMD_spice.tolist();
    results['PMD_sparsity']=PMD_sparsity.tolist();results['PMD_alpha1']=PMD_alpha1.tolist(); results['PMD_alpha2']=PMD_alpha2.tolist();
    
    results['PFD_cv']=PFD_cv.tolist();results['PFD_variance']=PFD_variance.tolist();
    results['PFD_variance2']=PFD_variance2.tolist();results['PFD_spice']=PFD_spice.tolist();
    results['PFD_sparsity']=PFD_sparsity.tolist();results['PFD_alpha1']=PFD_alpha1.tolist(); results['PFD_alpha2']=PFD_alpha2.tolist();
    
    results['Recall_cv']=Recall_cv.tolist();results['Recall_variance']=Recall_variance.tolist();
    results['Recall_variance2']=Recall_variance2.tolist();results['Recall_spice']=Recall_spice.tolist();
    results['Recall_sparsity']=Recall_sparsity.tolist();results['Recall_alpha1']=Recall_alpha1.tolist(); 
    results['Recall_alpha2']=Recall_alpha2.tolist(); 
    results['Precision_cv']=Precision_cv.tolist();results['Precision_variance']=Precision_variance.tolist();
    results['Precision_variance2']=Precision_variance2.tolist();results['Precision_spice']=Precision_spice.tolist();
    results['Precision_sparsity']=Precision_sparsity.tolist();results['Precision_alpha1']=Precision_alpha1.tolist();
    results['Precision_alpha2']=Precision_alpha2.tolist(); 
    
    file_name='BMMV_OMP_SNR_sweep_'+matrix_type+'.json'.format(n,p,k_block,l_block,L)
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

    
    
