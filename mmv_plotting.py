import matplotlib.pyplot as plt
import json
import argparse
import numpy as np

def plot_somp_snr_sweep_results(results,plot_format='2_2'):
    fig=plt.figure(figsize=(36,28))
    linewidth=3.5
    markersize=22
    fontsize=24
    label_fontsize=24
    xticksize=22; yticksize=22
    
    
    num_iter=results['num_iter']
    n=results['n']
    p=results['p']
    L=results['L']
    k_0=results['k_0']
    SNR=results['SNR']
    matrix_type=results['matrix_type']
    signal_type=results['sigmal_type'] 
    MSE_cv=results['MSE_cv'];MSE_variance=results['MSE_variance'];MSE_variance2=results['MSE_variance2'];
    MSE_spice=results['MSE_spice'];MSE_spice_ls=results['MSE_spice_ls'];
    MSE_sparsity=results['MSE_sparsity'];MSE_alpha1=results['MSE_alpha1']; MSE_alpha2=results['MSE_alpha2']; 
    PE_cv=results['PE_cv'];PE_variance=results['PE_variance'];
    PE_variance2=results['PE_variance2'];PE_spice=results['PE_spice'];PE_spice_ls=results['PE_spice_ls'];
    PE_sparsity=results['PE_sparsity'];PE_alpha1=results['PE_alpha1']; PE_alpha2=results['PE_alpha2']; 
    
    Recall_cv=results['Recall_cv'];Recall_variance=results['Recall_variance'];
    Recall_variance2=results['Recall_variance2'];Recall_spice=results['Recall_spice'];Recall_spice_ls=results['Recall_spice_ls'];
    Recall_sparsity=results['Recall_sparsity'];Recall_alpha1=results['Recall_alpha1']; 
    Recall_alpha2=results['Recall_alpha2']; 
    
    Precision_cv=results['Precision_cv'];Precision_variance=results['Precision_variance'];
    Precision_variance2=results['Precision_variance2'];Precision_spice=results['Precision_spice'];
    Precision_spice_ls=results['Precision_spice_ls'];
    Precision_sparsity=results['Precision_sparsity'];Precision_alpha1=results['Precision_alpha1'];
    Precision_alpha2=results['Precision_alpha2']; 
    
    PMD_cv=results['PMD_cv'];PMD_variance=results['PMD_variance'];
    PMD_variance2=results['PMD_variance2'];PMD_spice=results['PMD_spice'];
    PMD_spice_ls=results['PMD_spice_ls'];
    PMD_sparsity=results['PMD_sparsity'];PMD_alpha1=results['PMD_alpha1'];
    PMD_alpha2=results['PMD_alpha2']; 
    
    PFD_cv=results['PFD_cv'];PFD_variance=results['PFD_variance'];
    PFD_variance2=results['PFD_variance2'];PFD_spice=results['PFD_spice'];
    PFD_spice_ls=results['PFD_spice_ls'];
    PFD_sparsity=results['PFD_sparsity'];PFD_alpha1=results['PFD_alpha1'];
    PFD_alpha2=results['PFD_alpha2']; 
    
    
    
    if plot_format=='2_2':

        plt.subplot(3,2,1)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.semilogy(SNR,MSE_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha1,'bs-',label=r'GRRT($\alpha=0.1$)',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_variance2,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice_ls,'g<-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        #plt.savefig('NMSE_LASSO_n200_p400_snr10.eps',bbox_inches='tight')

        plt.subplot(3,2,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,1-np.array(PE_cv),'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-np.array(PE_alpha1),'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-np.array(PE_alpha2),'k*-',label=r'GRRT($\alpha=0.01$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-np.array(PE_variance),'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-np.array(PE_variance2),'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-np.array(PE_spice),'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-np.array(PE_sparsity),'m^-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,1-np.array(PE_spice_ls),'g<-',linewidth=linewidth,markersize=markersize)
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
        plt.plot(SNR,Recall_variance2,'go-',label=r'SOMP($||{\bf W}||_F$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',label='SPICE',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice_ls,'g<-',linewidth=linewidth,markersize=markersize)
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
        plt.plot(SNR,Precision_variance2,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sparsity,'m^-',label=r'SOMP($k_{row}$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_spice_ls,'g<-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('Precision',fontsize=label_fontsize)
        
        
        plt.subplot(3,2,5)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,PMD_cv,'r+-',label='CV', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_variance2,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PMD_spice_ls,'g<-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
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
        plt.plot(SNR,PFD_variance,'md-',label=r'SOMP($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_variance2,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,PFD_spice_ls,'g<-',linewidth=linewidth,markersize=markersize)
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
        plt.semilogy(SNR,MSE_variance2,'go-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        plt.semilogy(SNR,MSE_spice_ls,'g<-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('NMSE',fontsize=label_fontsize)
        

        plt.subplot(1,3,2)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Recall_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance,'md-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_variance2,'go-',label=r'SOMP($||{\bf W}||_F$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice,'r>-',label='SPICE',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_sparsity,'m^-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice_ls,'g<-',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('Recall',fontsize=label_fontsize)

        plt.subplot(1,3,3)
        plt.xticks(fontsize=xticksize)
        plt.yticks(fontsize=yticksize)
        plt.plot(SNR,Precision_cv,'r+-', linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha1,'bs-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_alpha2,'k*-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_variance,'md-',label=r'SOMP($\sigma^2$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_variance2,'go-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_spice,'r>-',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Precision_sparsity,'m^-',label=r'SOMP($K_{row}$)',linewidth=linewidth,markersize=markersize)
        plt.plot(SNR,Recall_spice_ls,'g<-',label='SPICE+LS',linewidth=linewidth,markersize=markersize)
        plt.legend(fontsize=fontsize)
        plt.grid()
        plt.xlabel('SNR(DB)',fontsize=label_fontsize)
        plt.ylabel('Precision',fontsize=label_fontsize)
    else:
        raise Exception('illegal plot format')
    
    file_name='SOMP_SNR_sweep_'+matrix_type+'_'+plot_format+'.eps'
    plt.savefig(file_name,bbox_inches='tight')
    print('figure saved to {}'.format(file_name))
    file_name='SOMP_SNR_sweep_'+matrix_type+'_'+plot_format+'.png'
    plt.savefig(file_name)
    print('figure saved to {}'.format(file_name))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file_name', type=str, default='abc.json',
                        help='full_path')
    parser.add_argument('--plot_format', type=str, default='2_2',
                        help='2_2 or 1_4')
    
    
    args = parser.parse_args()
    print(args)
    with open(args.result_file_name) as f:
        results=json.load(f)
    plot_somp_snr_sweep_results(results,'2_2')
    plot_somp_snr_sweep_results(results,'1_4')
    
    