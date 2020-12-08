import matplotlib.pyplot as plt
import json
import argparse
import numpy as np

def plot_lasso_snr_sweep_results(results,plot_format='2_2'):
   
    linewidth=3.5
    markersize=22
    fontsize=22
    label_fontsize=22
    xticksize=24; yticksize=24
    
    num_iter=results['num_iter']
    n=results['n']
    p=results['p']
    k_0=results['k_0']
    SNR=results['SNR']
    matrix_type=results['matrix_type']
    signal_type=results['sigmal_type'] 
    MSE_cv=results['MSE_cv'];MSE_variance=results['MSE_variance'];MSE_efic=results['MSE_efic'];
    MSE_spice=results['MSE_spice'];MSE_sq=results['MSE_sq'];
    MSE_sparsity=results['MSE_sparsity'];MSE_alpha1=results['MSE_alpha1']; MSE_alpha2=results['MSE_alpha2']; 
    MSE_unknownA=results['MSE_unknownA']; 
    PE_cv=np.array(results['PE_cv']);PE_variance=np.array(results['PE_variance']);PE_sq=np.array(results['PE_sq']);
    PE_efic=np.array(results['PE_efic']);PE_spice=np.array(results['PE_spice']);
    PE_sparsity=np.array(results['PE_sparsity']);PE_alpha1=np.array(results['PE_alpha1']); PE_alpha2=np.array(results['PE_alpha2']); 
    PE_unknownA=np.array(results['PE_unknownA']); 
    
    Recall_cv=results['Recall_cv'];Recall_variance=results['Recall_variance'];
    Recall_efic=results['Recall_efic'];Recall_spice=results['Recall_spice'];
    Recall_sparsity=results['Recall_sparsity'];Recall_alpha1=results['Recall_alpha1']; 
    Recall_alpha2=results['Recall_alpha2']; Recall_sq=results['Recall_sq'];
    Recall_unknownA=results['Recall_unknownA'];
    
    
    Precision_cv=results['Precision_cv'];Precision_variance=results['Precision_variance'];
    Precision_efic=results['Precision_efic'];Precision_spice=results['Precision_spice'];
    Precision_sparsity=results['Precision_sparsity'];Precision_alpha1=results['Precision_alpha1'];
    Precision_alpha2=results['Precision_alpha2']; Precision_sq=results['Precision_sq']; 
    Precision_unknownA=results['Precision_unknownA']; 
    
    PFD_cv=np.array(results['PFD_cv']);PFD_variance=np.array(results['PFD_variance']);PFD_sq=np.array(results['PFD_sq']);
    PFD_efic=np.array(results['PFD_efic']);PFD_spice=np.array(results['PFD_spice']);
    PFD_sparsity=np.array(results['PFD_sparsity']);PFD_alpha1=np.array(results['PFD_alpha1']); PFD_alpha2=np.array(results['PFD_alpha2']); 
    PFD_unknownA=np.array(results['PFD_unknownA']); 
    
    
    PMD_cv=np.array(results['PMD_cv']);PMD_variance=np.array(results['PMD_variance']);PMD_sq=np.array(results['PMD_sq']);
    PMD_efic=np.array(results['PMD_efic']);PMD_spice=np.array(results['PMD_spice']);
    PMD_sparsity=np.array(results['PMD_sparsity']);PMD_alpha1=np.array(results['PMD_alpha1']); PMD_alpha2=np.array(results['PMD_alpha2']); 
    PMD_unknownA=np.array(results['PMD_unknownA']); 
    
    
    if plot_format=='2_2':
        fig=plt.figure(figsize=(30,30))
        label_fontsize=24
        plt.subplot(3,2,1)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.semilogy(SNR,MSE_cv,'r+-', label='CV',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha1,'bs-',label=r'GRRT($\alpha=0.1$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.semilogy(SNR,MSE_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        #plt.savefig('NMSE_LASSO_n200_p400_snr10.eps',bbox_inches='tight')

        plt.subplot(3,2,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha2,'k*-',label=r'GRRT($\alpha=0.01$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)

        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('PCSR',fontsize=label_fontsize)

        plt.subplot(3,2,3)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Recall_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Recall_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('Recall',fontsize=label_fontsize)

        plt.subplot(3,2,4)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Precision_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('Precision',fontsize=label_fontsize)

        
        
        plt.subplot(3,2,5)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,PMD_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('PMD',fontsize=label_fontsize)

        
        plt.subplot(3,2,6)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,PFD_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_spice,'r>-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('PFD',fontsize=label_fontsize)

        
        
    elif plot_format=='1_4':
        fig=plt.figure(figsize=(32,10))
        label_fontsize=24
        plt.subplot(1,3,1)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.semilogy(SNR,MSE_cv,'r+-', label='CV',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha1,'bs-',label=r'GRRT($\alpha=0.1$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha2,'k*-',label=r'GRRT($\alpha=0.01$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.semilogy(SNR,MSE_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        #plt.savefig('NMSE_LASSO_n200_p400_snr10.eps',bbox_inches='tight')

#         plt.subplot(1,4,2)
#         plt.xticks(fontsize=xticksize)
#         plt.yticks(fontsize=yticksize)
#         plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_spice,'r>-',label='SPICE',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_sq,'g<-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)

#         plt.legend(fontsize=fontsize)
#         plt.grid()
#         plt.xlabel('SNR(DB)',fontsize=label_fontsize)
#         plt.ylabel('PCSR',fontsize=label_fontsize)

        plt.subplot(1,3,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Recall_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Recall_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('Recall',fontsize=label_fontsize)

#         plt.subplot(1,3,3)
#         plt.xticks(fontsize=xticksize)
#         plt.yticks(fontsize=yticksize)
#         plt.plot(SNR,Precision_cv,'r+-', linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,Precision_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,Precision_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,Precision_variance,'md-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,Precision_efic,'go-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,Precision_spice,'r>-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,Precision_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,Precision_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
#         #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
#         plt.legend(fontsize=fontsize)
#         plt.grid()
#         plt.xlabel('SNR(DB)',fontsize=label_fontsize)
#         plt.ylabel('Precision',fontsize=label_fontsize)
        plt.subplot(1,3,3)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('PCSR',fontsize=label_fontsize)
    else:
        raise Exception('illegal plot format')
    
    file_name='LASSO_SNR_sweep_'+matrix_type+'_'+plot_format+'.eps'
    plt.savefig(file_name,bbox_inches='tight')
    print('figure saved to {}'.format(file_name))
    file_name='LASSO_SNR_sweep_'+matrix_type+'_'+plot_format+'.png'
    plt.savefig(file_name)
    print('figure saved to {}'.format(file_name))

    
def plot_lasso_sample_sweep_results(results,plot_format='2_2'):
    
    linewidth=3.5
    markersize=22
    fontsize=22
    label_fontsize=22
    xticksize=24; yticksize=24
    
    num_iter=results['num_iter']
    n=results['n']
    p=results['p']
    k_0=results['k_0']
    SNR=results['SNR']
    matrix_type=results['matrix_type']
    signal_type=results['sigmal_type'] 
    MSE_cv=results['MSE_cv'];MSE_variance=results['MSE_variance'];MSE_efic=results['MSE_efic'];
    MSE_spice=results['MSE_spice'];MSE_sq=results['MSE_sq'];
    MSE_sparsity=results['MSE_sparsity'];MSE_alpha1=results['MSE_alpha1']; MSE_alpha2=results['MSE_alpha2']; 
    MSE_unknownA=results['MSE_unknownA']; 
    PE_cv=np.array(results['PE_cv']);PE_variance=np.array(results['PE_variance']);PE_sq=np.array(results['PE_sq']);
    PE_efic=np.array(results['PE_efic']);PE_spice=np.array(results['PE_spice']);
    PE_sparsity=np.array(results['PE_sparsity']);PE_alpha1=np.array(results['PE_alpha1']); PE_alpha2=np.array(results['PE_alpha2']); 
    PE_unknownA=np.array(results['PE_unknownA']); 
    
    Recall_cv=results['Recall_cv'];Recall_variance=results['Recall_variance'];
    Recall_efic=results['Recall_efic'];Recall_spice=results['Recall_spice'];
    Recall_sparsity=results['Recall_sparsity'];Recall_alpha1=results['Recall_alpha1']; 
    Recall_alpha2=results['Recall_alpha2']; Recall_sq=results['Recall_sq'];
    Recall_unknownA=results['Recall_unknownA'];
    
    
    Precision_cv=results['Precision_cv'];Precision_variance=results['Precision_variance'];
    Precision_efic=results['Precision_efic'];Precision_spice=results['Precision_spice'];
    Precision_sparsity=results['Precision_sparsity'];Precision_alpha1=results['Precision_alpha1'];
    Precision_alpha2=results['Precision_alpha2']; Precision_sq=results['Precision_sq']; 
    Precision_unknownA=results['Precision_unknownA']; 
    
    PFD_cv=np.array(results['PFD_cv']);PFD_variance=np.array(results['PFD_variance']);PFD_sq=np.array(results['PFD_sq']);
    PFD_efic=np.array(results['PFD_efic']);PFD_spice=np.array(results['PFD_spice']);
    PFD_sparsity=np.array(results['PFD_sparsity']);PFD_alpha1=np.array(results['PFD_alpha1']); PFD_alpha2=np.array(results['PFD_alpha2']); 
    PFD_unknownA=np.array(results['PFD_unknownA']); 
    
    
    PMD_cv=np.array(results['PMD_cv']);PMD_variance=np.array(results['PMD_variance']);PMD_sq=np.array(results['PMD_sq']);
    PMD_efic=np.array(results['PMD_efic']);PMD_spice=np.array(results['PMD_spice']);
    PMD_sparsity=np.array(results['PMD_sparsity']);PMD_alpha1=np.array(results['PMD_alpha1']); PMD_alpha2=np.array(results['PMD_alpha2']); 
    PMD_unknownA=np.array(results['PMD_unknownA']); 
    
    SNR=n
    if plot_format=='2_2':
        fig=plt.figure(figsize=(30,30))
        label_fontsize=24
        plt.subplot(3,2,1)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.semilogy(SNR,MSE_cv,'r+-', label='CV',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha1,'bs-',label=r'GRRT($\alpha=0.1$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.semilogy(SNR,MSE_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        #plt.savefig('NMSE_LASSO_n200_p400_snr10.eps',bbox_inches='tight')

        plt.subplot(3,2,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha2,'k*-',label=r'GRRT($\alpha=0.01$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)

        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('PCSR',fontsize=label_fontsize)

        plt.subplot(3,2,3)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Recall_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Recall_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('Recall',fontsize=label_fontsize)

        plt.subplot(3,2,4)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Precision_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('Precision',fontsize=label_fontsize)

        
        
        plt.subplot(3,2,5)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,PMD_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('PMD',fontsize=label_fontsize)

        
        plt.subplot(3,2,6)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,PFD_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_spice,'r>-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('PFD',fontsize=label_fontsize)

        
        
    elif plot_format=='1_4':
        fig=plt.figure(figsize=(32,10))
        label_fontsize=24
        plt.subplot(1,3,1)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.semilogy(SNR,MSE_cv,'r+-', label='CV',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha1,'bs-',label=r'GRRT($\alpha=0.1$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha2,'k*-',label=r'GRRT($\alpha=0.01$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.semilogy(SNR,MSE_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        #plt.savefig('NMSE_LASSO_n200_p400_snr10.eps',bbox_inches='tight')

#         plt.subplot(1,4,2)
#         plt.xticks(fontsize=xticksize)
#         plt.yticks(fontsize=yticksize)
#         plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_spice,'r>-',label='SPICE',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_sq,'g<-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)

#         plt.legend(fontsize=fontsize)
#         plt.grid()
#         plt.xlabel('SNR(DB)',fontsize=label_fontsize)
#         plt.ylabel('PCSR',fontsize=label_fontsize)

        plt.subplot(1,3,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Recall_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Recall_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('Recall',fontsize=label_fontsize)

# #         plt.subplot(1,3,3)
# #         plt.xticks(fontsize=xticksize)
# #         plt.yticks(fontsize=yticksize)
# #         plt.plot(SNR,Precision_cv,'r+-', linewidth=linewidth,markersize=markersize)
# #         plt.plot(SNR,Precision_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
# #         plt.plot(SNR,Precision_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
# #         plt.plot(SNR,Precision_variance,'md-',linewidth=linewidth,markersize=markersize)
# #         plt.plot(SNR,Precision_efic,'go-',linewidth=linewidth,markersize=markersize)
# #         plt.plot(SNR,Precision_spice,'r>-',linewidth=linewidth,markersize=markersize)
# #         plt.plot(SNR,Precision_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
# #         plt.plot(SNR,Precision_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
# #         #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
#         plt.legend(fontsize=fontsize)
#         plt.grid()
#         plt.xlabel('Sample size N',fontsize=label_fontsize)
#         plt.ylabel('Precision',fontsize=label_fontsize)
        plt.subplot(1,3,3)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('Sample size N',fontsize=label_fontsize)
        plt.ylabel('PCSR',fontsize=label_fontsize)

    else:
        raise Exception('illegal plot format')
    
    file_name='LASSO_sample_sweep_'+matrix_type+'_'+plot_format+'.eps'
    plt.savefig(file_name,bbox_inches='tight')
    print('figure saved to {}'.format(file_name))
    file_name='LASSO_SNR_sweep_'+matrix_type+'_'+plot_format+'.png'
    plt.savefig(file_name)
    print('figure saved to {}'.format(file_name))

    
def plot_lasso_sparsity_sweep_results(results,plot_format='2_2'):
   
    linewidth=3.5
    markersize=22
    fontsize=22
    label_fontsize=22
    xticksize=24; yticksize=24
    
    num_iter=results['num_iter']
    n=results['n']
    p=results['p']
    k_0=results['k_0']
    SNR=results['SNR']
    matrix_type=results['matrix_type']
    signal_type=results['sigmal_type'] 
    MSE_cv=results['MSE_cv'];MSE_variance=results['MSE_variance'];MSE_efic=results['MSE_efic'];
    MSE_spice=results['MSE_spice'];MSE_sq=results['MSE_sq'];
    MSE_sparsity=results['MSE_sparsity'];MSE_alpha1=results['MSE_alpha1']; MSE_alpha2=results['MSE_alpha2']; 
    MSE_unknownA=results['MSE_unknownA']; 
    PE_cv=np.array(results['PE_cv']);PE_variance=np.array(results['PE_variance']);PE_sq=np.array(results['PE_sq']);
    PE_efic=np.array(results['PE_efic']);PE_spice=np.array(results['PE_spice']);
    PE_sparsity=np.array(results['PE_sparsity']);PE_alpha1=np.array(results['PE_alpha1']); PE_alpha2=np.array(results['PE_alpha2']); 
    PE_unknownA=np.array(results['PE_unknownA']); 
    
    Recall_cv=results['Recall_cv'];Recall_variance=results['Recall_variance'];
    Recall_efic=results['Recall_efic'];Recall_spice=results['Recall_spice'];
    Recall_sparsity=results['Recall_sparsity'];Recall_alpha1=results['Recall_alpha1']; 
    Recall_alpha2=results['Recall_alpha2']; Recall_sq=results['Recall_sq'];
    Recall_unknownA=results['Recall_unknownA'];
    
    
    Precision_cv=results['Precision_cv'];Precision_variance=results['Precision_variance'];
    Precision_efic=results['Precision_efic'];Precision_spice=results['Precision_spice'];
    Precision_sparsity=results['Precision_sparsity'];Precision_alpha1=results['Precision_alpha1'];
    Precision_alpha2=results['Precision_alpha2']; Precision_sq=results['Precision_sq']; 
    Precision_unknownA=results['Precision_unknownA']; 
    
    PFD_cv=np.array(results['PFD_cv']);PFD_variance=np.array(results['PFD_variance']);PFD_sq=np.array(results['PFD_sq']);
    PFD_efic=np.array(results['PFD_efic']);PFD_spice=np.array(results['PFD_spice']);
    PFD_sparsity=np.array(results['PFD_sparsity']);PFD_alpha1=np.array(results['PFD_alpha1']); PFD_alpha2=np.array(results['PFD_alpha2']); 
    PFD_unknownA=np.array(results['PFD_unknownA']); 
    
    
    PMD_cv=np.array(results['PMD_cv']);PMD_variance=np.array(results['PMD_variance']);PMD_sq=np.array(results['PMD_sq']);
    PMD_efic=np.array(results['PMD_efic']);PMD_spice=np.array(results['PMD_spice']);
    PMD_sparsity=np.array(results['PMD_sparsity']);PMD_alpha1=np.array(results['PMD_alpha1']); PMD_alpha2=np.array(results['PMD_alpha2']); 
    PMD_unknownA=np.array(results['PMD_unknownA']); 
    
    SNR=k_0
    if plot_format=='2_2':
        fig=plt.figure(figsize=(30,30))
        label_fontsize=24
        plt.subplot(3,2,1)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.semilogy(SNR,MSE_cv,'r+-', label='CV',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha1,'bs-',label=r'GRRT($\alpha=0.1$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.semilogy(SNR,MSE_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        #plt.savefig('NMSE_LASSO_n200_p400_snr10.eps',bbox_inches='tight')

        plt.subplot(3,2,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha2,'k*-',label=r'GRRT($\alpha=0.01$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)

        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('PCSR',fontsize=label_fontsize)

        plt.subplot(3,2,3)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Recall_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Recall_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('Recall',fontsize=label_fontsize)

        plt.subplot(3,2,4)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Precision_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('Precision',fontsize=label_fontsize)

        
        
        plt.subplot(3,2,5)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,PMD_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('PMD',fontsize=label_fontsize)

        
        plt.subplot(3,2,6)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,PFD_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_spice,'r>-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('PFD',fontsize=label_fontsize)

        
        
    elif plot_format=='1_4':
        fig=plt.figure(figsize=(32,10))
        label_fontsize=24
        plt.subplot(1,3,1)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.semilogy(SNR,MSE_cv,'r+-', label='CV',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha1,'bs-',label=r'GRRT($\alpha=0.1$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha2,'k*-',label=r'GRRT($\alpha=0.01$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.semilogy(SNR,MSE_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        #plt.savefig('NMSE_LASSO_n200_p400_snr10.eps',bbox_inches='tight')

#         plt.subplot(1,4,2)
#         plt.xticks(fontsize=xticksize)
#         plt.yticks(fontsize=yticksize)
#         plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_variance,'md-',label=r'LASSO($\sigma^2$)',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_spice,'r>-',label='SPICE',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_sq,'g<-',linewidth=linewidth,markersize=markersize)
#         plt.plot(SNR,1-PE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)

#         plt.legend(fontsize=fontsize)
#         plt.grid()
#         plt.xlabel('SNR(DB)',fontsize=label_fontsize)
#         plt.ylabel('PCSR',fontsize=label_fontsize)

        plt.subplot(1,3,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Recall_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_efic,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sq,'g<-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        #plt.plot(SNR,Recall_unknownA,'kh-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('Recall',fontsize=label_fontsize)

        plt.subplot(1,3,3)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,1-PE_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_efic,'go-',label='EFIC',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_spice,'r>-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sq,'g<-',label='Scaled LASSO',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-PE_sparsity,'m^-',label=r'LASSO($K_{row}$)',linewidth=linewidth,markersize=markersize)
        
        #plt.plot(SNR,Precision_unknownA,'kh-',label='unkA',linewidth=linewidth,markersize=markersize)
        
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel(r'Sparsity $K_{row}$',fontsize=label_fontsize)
        plt.ylabel('PCSR',fontsize=label_fontsize)
    else:
        raise Exception('illegal plot format')
    
    file_name='LASSO_sparsity_sweep_'+matrix_type+'_'+plot_format+'.eps'
    plt.savefig(file_name,bbox_inches='tight')
    print('figure saved to {}'.format(file_name))
    file_name='LASSO_sparsity_sweep_'+matrix_type+'_'+plot_format+'.png'
    plt.savefig(file_name)
    print('figure saved to {}'.format(file_name))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file_name', type=str, default='abc.json',
                        help='full_path')
    parser.add_argument('--plot_format', type=str, default='2_2',
                        help='2_2 or 1_4')
    parser.add_argument('--experiment_type', type=str, default='SNR_sweep',
                        help='SNR_sweep, sample_sweep or sparsity_sweep')
    
    
    args = parser.parse_args()
    print(args)
    with open(args.result_file_name) as f:
        results=json.load(f)
    if args.experiment_type=='SNR_sweep':
        plot_lasso_snr_sweep_results(results,'1_4')
        plot_lasso_snr_sweep_results(results,'2_2')
    elif args.experiment_type=='sample_sweep':
        plot_lasso_sample_sweep_results(results,'1_4')
        plot_lasso_sample_sweep_results(results,'2_2')
    elif args.experiment_type=='sparsity_sweep':
        plot_lasso_sparsity_sweep_results(results,'1_4')
        plot_lasso_sparsity_sweep_results(results,'2_2')
    else:
        pass
        
        

    