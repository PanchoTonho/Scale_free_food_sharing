"""
Copyright (c) 2022, Francisco Plana
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import matplotlib.pyplot as plt
import pandas as pd
from characterize_clusters import preprocess_datafr_1_opt, draw_netw, draw_the_most_representative
import numpy as np
from multiprocessing import Pool, cpu_count 
from functools import partial
import evol_methods as ev
import pickle
import dataframe_methods as df_m
import random
import math
from scipy.stats import norm, spearmanr
from itertools import product
from random import sample
from graph_tool.all import *
import tail_estimation 

plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.family': 'serif'})

def compute_number_arcs_intra_inter_h(all_data,n_bits,N,direct,receive_int,i):
    
    nw_ID = all_data.loc[i,'ID']
    hunters_indexes= all_data.loc[i,'nh']
    
    if receive_int:
        
        #convert to binary, first least binary is at the right extreme
        bin_i = bin(nw_ID)[2:]
        # get strings of uniform length
        bin_i = bin_i.rjust(int(n_bits),"0")
        
        # get adjacency matrix
        if direct==False:
            A = ev.fill_matrix(bin_i,N)
            A = np.tril(A, -1)
        else:
            A = ev.fill_D_matrix(bin_i,N)
            
    else: # input is a list from evolution
        
        A = ev.fill_matrix_from_list(nw_ID,N,direct)
        
    counter_intra = 0
    counter_inter = 0 
    counter_intra_non_h = 0
        
    for i in range(N):
        for j in range(N):
            
            if A[i,j]==1:
                if i in hunters_indexes and j in hunters_indexes: 
                    counter_intra +=1
                elif i in hunters_indexes or j in hunters_indexes:
                    counter_inter += 1
                else:
                    counter_intra_non_h +=1
                    
    return counter_intra, counter_inter, counter_intra_non_h

def get_upper_lower_bootstr_interval(alpha,sample_means,bt_n_iter):

    # percentile confidence intervals
    p = (alpha/2.0) * 100
    lower = np.percentile(sample_means, p)
    p = (1 - alpha) * 100 + p
    upper = np.percentile(sample_means, p)
    print('%.1f confidence interval %.1f%% and %.1f%%' % ((1-alpha)*100, lower, upper))
    
    """
    # normal confidence intervals
    mean_estimate = np.mean(sample_means)
    sd_estimate = np.sqrt(np.sum([(x - mean_estimate)**2 for x in sample_means])/(bt_n_iter-1)) 
    z_score_alpha = norm.interval(alpha)[1]
    lower = mean_estimate - sd_estimate * z_score_alpha
    upper = mean_estimate + sd_estimate * z_score_alpha """
    
    return lower, upper, np.mean(sample_means)

# method to compute the statistics on sample_means in the case this variable
# stores values associated to many conditions. This occurs when this calculation
# is invoked by graph_costcorr_p_II()
def get_upper_lower_bootstr_interval_II(alpha,sample_means,bt_n_iter,perc_conf_interv=True):

    if perc_conf_interv:
        
        # percentile confidence intervals
        p = (alpha/2.0) * 100
        lower_int = np.percentile(sample_means, p, axis=1)
        p = (1 - alpha) * 100 + p
        upper_int = np.percentile(sample_means, p, axis=1)
        #print('%.1f confidence interval %.1f%% and %.1f%%' % ((1-alpha)*100, lower, upper))
        m = np.mean(sample_means,axis=1)
        lower = m - lower_int
        upper = upper_int - m
    else:
        
        s = np.std(sample_means, axis=1)
        SE = s/np.sqrt(bt_n_iter)
        lower = SE * 1.96
        upper = SE * 1.96
        m = np.mean(sample_means,axis=1)
    """
    # normal confidence intervals
    mean_estimate = np.mean(sample_means)
    sd_estimate = np.sqrt(np.sum([(x - mean_estimate)**2 for x in sample_means])/(bt_n_iter-1)) 
    z_score_alpha = norm.interval(alpha)[1]
    lower = mean_estimate - sd_estimate * z_score_alpha
    upper = mean_estimate + sd_estimate * z_score_alpha """
    
    return lower, upper, m

def number_arcs_vs_ph_F_nh(dir_addr,F_nh_greater_half=None,p_NA=True,seed=10,bt_n_iter=1000,alpha=0.05,type_opt=1):
    
    phs = [0.02,0.08,0.15,0.3,0.6]
    
    outp_df, type_min, nw_IDs, nw_mv = preprocess_datafr_1_opt(opt=type_opt,s=400,
                                                                   N=12,FS=None,
                                                                   root=dir_addr,
                              direct=True,n_sample=None,phs=None,proc_NA=p_NA)
    
    pool = Pool()
    """
    mean_intra = []
    sd_intra = []
    mean_inter = []
    sd_inter = []
    mean_intra_non_h = []
    sd_intra_non_h = []"""
    
    all_data = pd.concat([nw_IDs,nw_mv],axis=1)
    all_data.columns = ['ID','F','nh','ph']
    
    all_data.insert(all_data.shape[1],'F*nh',np.multiply(all_data['F'],all_data['nh']))
    
    if not F_nh_greater_half is None:
        
        if F_nh_greater_half:
            indexes=[i for i in all_data.index if len(all_data['F*nh'].loc[i])>12]
            all_data = all_data.loc[indexes]
            nw_mv =nw_mv.loc[indexes]
        else:
            indexes = [i for i in all_data.index if len(all_data['F*nh'].loc[i])<=12]
            all_data = all_data.loc[indexes]
            nw_mv=nw_mv.loc[indexes]
    
    func = partial(compute_number_arcs_intra_inter_h,all_data,11*12,12,True,True)
    
    mean_intra = []
    lower_intra = []
    upper_intra = []
    
    mean_inter = []
    lower_inter = []
    upper_inter = []
    
    mean_intra_non_h = []
    lower_intra_non_h = []
    upper_intra_non_h = []
    
    for p in phs:
        """
        # to use the sd as error
        output = pool.map(func,all_data[nw_mv['ph']==p].index)
        mean_intra.append(np.mean([x[0] for x in output]))
        sd_intra.append(np.std([x[0] for x in output]))
        mean_inter.append(np.mean([x[1] for x in output]))
        sd_inter.append(np.std([x[1] for x in output]))
        mean_intra_non_h.append(np.mean([x[2] for x in output]))
        sd_intra_non_h.append(np.std([x[2] for x in output]))
        """
        
        # to use a bootstrap percentile interval as error
        sample_means_intra = list()
        sample_means_inter = list()
        sample_means_intra_non_h = list()
        
        for i in range(bt_n_iter):
            
            data_sample=all_data[nw_mv['ph']==p].sample(frac=0.7,replace=True,random_state=seed+i)
            
            output = pool.map(func,data_sample.index)
    	
            sample_means_intra.append(np.mean([x[0] for x in output]))
            sample_means_inter.append(np.mean([x[1] for x in output]))
            sample_means_intra_non_h.append(np.mean([x[2] for x in output]))
            
        l, u, m = get_upper_lower_bootstr_interval(alpha,sample_means_intra,bt_n_iter)
        mean_intra.append(m)
        lower_intra.append(m-l)
        upper_intra.append(u-m)
        
        l, u, m = get_upper_lower_bootstr_interval(alpha,sample_means_inter,bt_n_iter)
        mean_inter.append(m)
        lower_inter.append(m-l)
        upper_inter.append(u-m)
        
        l, u, m = get_upper_lower_bootstr_interval(alpha,sample_means_intra_non_h,bt_n_iter)
        mean_intra_non_h.append(m)
        lower_intra_non_h.append(m-l)
        upper_intra_non_h.append(u-m)
        
        
    fig = plt.figure(figsize=(8, 6))
    #fig = plt.figure(figsize=(4, 4))
    plt.errorbar(phs, mean_intra, 
                 yerr = [lower_intra,upper_intra], #uplims=True, lolims=True,
                 label='intra_h')    
    plt.errorbar(phs, mean_inter, 
                 yerr = [lower_inter,upper_inter], #uplims=True, lolims=True,
                 label='inter_h_nonh') 
    plt.errorbar(phs, mean_intra_non_h, 
                 yerr = [lower_intra_non_h,upper_intra_non_h], #uplims=True, lolims=True,
                 label='intra_nonh') 

    plt.legend(loc='upper left')
    plt.xlabel('Probability of hunting ph')
    plt.ylabel('Number of arcs')
    plt.savefig('mean_arcs_F_nh_'+str(F_nh_greater_half)+'_opt_'+str(type_opt)+'.eps',dpi=300)
    plt.show()
    plt.close()
    
    return all_data

def compute_pbb_eat_h_nh(all_data,n_bits,N,direct,receive_int,i):
    
    nw_ID = all_data.loc[i,'ID']
    hunters_indexes= all_data.loc[i,'nh']
    ph= all_data.loc[i,'ph']
    F= all_data.loc[i,'F']
    
    if receive_int:
        
        #convert to binary, first least binary is at the right extreme
        bin_i = bin(nw_ID)[2:]
        # get strings of uniform length
        bin_i = bin_i.rjust(int(n_bits),"0")
        
        # get adjacency matrix
        if direct==False:
            A = ev.fill_matrix(bin_i,N)
            A = np.tril(A, -1)
        else:
            A = ev.fill_D_matrix(bin_i,N)
            
    else: # input is a list from evolution
        
        A = ev.fill_matrix_from_list(nw_ID,N,direct)
        
    pbb_vector = ev.compute_vector_pbb_eat(A,hunters_indexes,ph,F,N,direct)

    return np.mean([pbb_vector[i] for i in range(N) if i in hunters_indexes]), np.mean([pbb_vector[i] for i in range(N) if not i in hunters_indexes])

def compute_recipr_pairs_not_hunters(A,N,h_ind):
    
    pairs = 0
    
    recipr_pairs = 0
    
    for i in range(N):
        for j in range(i):
            
            if i in h_ind or j in h_ind:
                continue
            
            if A[i,j]==0 and A[j,i]==0:
                continue
            
            pairs += 1
            
            if A[i,j]==A[j,i]:
                recipr_pairs += 1
                
    if pairs == 0:
        return 0
    else:        
        return recipr_pairs/pairs
    
def compute_medn_ppb_eat_line(N,direct,all_data,i):
    
    nw_ID = all_data.loc[i,'ID']
    hunters_indexes = all_data.loc[i,'nh']
    ph = all_data.loc[i,'ph']
    F = all_data.loc[i,'F']
    
    if direct==False:
        n_bits = N*(N-1)/2
    else:
        n_bits = N*(N-1)
    
    #convert to binary, first least binary is at the right extreme
    bin_i = bin(nw_ID)[2:]
    # get strings of uniform length
    bin_i = bin_i.rjust(int(n_bits),"0")
    
    # get adjacency matrix
    if direct==False:
        A = ev.fill_matrix(bin_i,N)
        A = np.tril(A, -1)
    else:
        A = ev.fill_D_matrix(bin_i,N)
        
    return np.median(ev.compute_vector_pbb_eat(A,hunters_indexes,ph,F,N,direct))

def compute_pbb_eat_non_h(all_data,n_bits,direct,N,i):
    
        nw_code = all_data.loc[i,'ID']
        h_ind = all_data.loc[i,'nh']
        ph = all_data.loc[i,'ph']
        F = all_data.loc[i,'F']
            
        #convert to binary, first least binary is at the right extreme
        bin_i = bin(nw_code)[2:]
        # get strings of uniform length
        bin_i = bin_i.rjust(int(n_bits),"0")
        
        # get adjacency matrix
        if direct==False:
                A = ev.fill_matrix(bin_i,N)
        else:
                A = ev.fill_D_matrix(bin_i,N)
        
        temp_p = ev.compute_vector_pbb_eat(A,h_ind,ph,F,N,direct)
        
        return np.mean([temp_p[i] for i in range(N) if i not in h_ind])
       
def compare_var_betw_opt(root):
    
    # open data of each variable and additional variable
    with open(root + "extra_vars_all_optima.pickle", "rb") as file:
        [df_1, type_min_1, nw_IDs_1, nw_mv_1, 
         df_2, type_min_2, nw_IDs_2, nw_mv_2,
         df_3, type_min_3, nw_IDs_3, nw_mv_3] = pickle.load(file)
        
    outp_df_1, type_min_1, nw_IDs_1, nw_mv_1 = preprocess_datafr_1_opt(opt=1,s=400,
                                            N=12,FS=None,
                                            root=root,
                                            direct=True,n_sample=None,phs=None,proc_NA=True)
    
    outp_df_2, type_min_2, nw_IDs_2, nw_mv_2 = preprocess_datafr_1_opt(opt=2,s=400,
                                            N=12,FS=None,
                                            root=root,
                                            direct=True,n_sample=None,phs=None,proc_NA=True)
    
    outp_df_3, type_min_3, nw_IDs_3, nw_mv_3 = preprocess_datafr_1_opt(opt=3,s=400,
                                            N=12,FS=None,
                                            root=root,
                                            direct=True,n_sample=None,phs=None,proc_NA=True)
    
    #phs = [0.02,0.08,0.15,0.3,0.6]
    
    # paste data
    all_data_1 = pd.concat([outp_df_1,df_1,nw_IDs_1,nw_mv_1],axis=1)
    all_data_1.columns = list(outp_df_1.columns) + list(df_1.columns) + ['ID','F','nh','ph']
    
    outp_df_2 = outp_df_2.drop(['out_assort','in_assort'],axis=1)
    all_data_2 = pd.concat([outp_df_2,df_2,nw_IDs_2,nw_mv_2],axis=1)
    all_data_2.columns = list(outp_df_2.columns) + list(df_2.columns) + ['ID','F','nh','ph']
    
    all_data_3 = pd.concat([outp_df_3,df_3,nw_IDs_3,nw_mv_3],axis=1)
    all_data_3.columns = list(outp_df_3.columns) + list(df_3.columns) + ['ID','F','nh','ph']
    
    # add new varibles
    direct=True
    N=12
    
    func1 = partial(compute_medn_ppb_eat_line,N,direct,all_data_1)
    func2 = partial(compute_medn_ppb_eat_line,N,direct,all_data_2)
    func3 = partial(compute_medn_ppb_eat_line,N,direct,all_data_3)
    pool = Pool()
    
    all_data_1.insert(all_data_1.shape[1],'med_pbb_eat',pool.map(func1,all_data_1.index))
    all_data_2.insert(all_data_2.shape[1],'med_pbb_eat',pool.map(func2,all_data_2.index))
    all_data_3.insert(all_data_3.shape[1],'med_pbb_eat',pool.map(func3,all_data_3.index))
    
    all_data_3.insert(all_data_3.shape[1],'pbb_eat',np.multiply(all_data_3['WEF_cost'],all_data_3['pbb_std']))
    all_data_2.insert(all_data_2.shape[1],'pbb_eat',np.multiply(all_data_2['WEF_cost'],all_data_2['pbb_std']))
    all_data_1.insert(all_data_1.shape[1],'pbb_eat',np.multiply(all_data_1['WEF_cost'],all_data_1['pbb_std']))
    
    all_data_3.insert(all_data_3.shape[1],'std_pbb',np.reciprocal(all_data_3['pbb_std']))
    all_data_2.insert(all_data_2.shape[1],'std_pbb',np.reciprocal(all_data_2['pbb_std']))
    all_data_1.insert(all_data_1.shape[1],'std_pbb',np.reciprocal(all_data_1['pbb_std']))
    direct=True
    N=12
    n_bits=12*11
    #func1 = partial(compute_pbb_eat_non_h,all_data_1,n_bits,direct,N)
    #func2 = partial(compute_pbb_eat_non_h,all_data_2,n_bits,direct,N)
    #func3 = partial(compute_pbb_eat_non_h,all_data_3,n_bits,direct,N)
    
    func1 = partial(compute_pbb_eat_h_nh,all_data_1,n_bits,N,direct,True)
    func2 = partial(compute_pbb_eat_h_nh,all_data_2,n_bits,N,direct,True)
    func3 = partial(compute_pbb_eat_h_nh,all_data_3,n_bits,N,direct,True)
    
    outp1 = pool.map(func1,all_data_1.index)
    outp2 = pool.map(func2,all_data_2.index)
    outp3 = pool.map(func3,all_data_3.index)

    all_data_1.insert(all_data_1.shape[1],'pbb_non_h',[x[1] for x in outp1])
    all_data_2.insert(all_data_2.shape[1],'pbb_non_h',[x[1] for x in outp2])
    all_data_3.insert(all_data_3.shape[1],'pbb_non_h',[x[1] for x in outp3])
    
    all_data_1.insert(all_data_1.shape[1],'delta_pbb',[x[0] - x[1] for x in outp1])
    all_data_2.insert(all_data_2.shape[1],'delta_pbb',[x[0] -x[1] for x in outp2])
    all_data_3.insert(all_data_3.shape[1],'delta_pbb',[x[0] -x[1] for x in outp3])
    
    pool.close()
    pool.join()
    all_data_3.insert(all_data_3.shape[1],'F*nh',np.multiply(all_data_3['F'],[len(x) for x in all_data_3['nh']]))
    all_data_2.insert(all_data_2.shape[1],'F*nh',np.multiply(all_data_2['F'],[len(x) for x in all_data_2['nh']]))
    all_data_1.insert(all_data_1.shape[1],'F*nh',np.multiply(all_data_1['F'],[len(x) for x in all_data_1['nh']]))
    """ """
    
    # sample and graph
    #boots_sample_and_graph(var,all_data_1, all_data_2, all_data_3,phs,bt_n_iter,seed,alpha,without_single_opt=False,nh_greater_half=None,Fnh_small=None)
    
    return all_data_1, all_data_2, all_data_3
    
def boots_sample_and_graph(var,all_data_1, all_data_2, all_data_3,phs=[0.02,0.08,0.15,0.3,0.6],bt_n_iter=500,seed=10,alpha=0.05,without_single_opt=False,nh_greater_half=None,Fnh_small=None,n_part_great=0,perc=None):
    
    # filter data
    if not Fnh_small is None:
        
        if Fnh_small:
            all_data_3 = all_data_3[all_data_3['F*nh']<=12] 
            all_data_2 = all_data_2[all_data_2['F*nh']<=12] 
            all_data_1 = all_data_1[all_data_1['F*nh']<=12] 
        else:
            all_data_3 = all_data_3[all_data_3['F*nh']>12] 
            all_data_2 = all_data_2[all_data_2['F*nh']>12] 
            all_data_1 = all_data_1[all_data_1['F*nh']>12] 
    
    if n_part_great > 0:
        
        all_data_3 = all_data_3[all_data_3['n_partitions']>n_part_great]
        all_data_2 = all_data_2[all_data_2['n_partitions']>n_part_great]
        all_data_1 = all_data_1[all_data_1['n_partitions']>n_part_great]
    
    if without_single_opt==True:
        all_data_3 = all_data_3[(all_data_3['is_RV']==False)&(all_data_3['is_WEF']==False)]    
    
    if not nh_greater_half is None:
        
        if nh_greater_half:
            indexes_1=[i for i in all_data_1.index if len(all_data_1['nh'].loc[i])>6]
            all_data_1 = all_data_1.loc[indexes_1]
            
            indexes_2=[i for i in all_data_2.index if len(all_data_2['nh'].loc[i])>6]
            all_data_2 = all_data_2.loc[indexes_2]
            
            indexes_3=[i for i in all_data_3.index if len(all_data_3['nh'].loc[i])>6]
            all_data_3 = all_data_3.loc[indexes_3]

        else:
            indexes_1=[i for i in all_data_1.index if len(all_data_1['nh'].loc[i])<=6]
            all_data_1 = all_data_1.loc[indexes_1]
            
            indexes_2=[i for i in all_data_2.index if len(all_data_2['nh'].loc[i])<=6]
            all_data_2 = all_data_2.loc[indexes_2]
            
            indexes_3=[i for i in all_data_3.index if len(all_data_3['nh'].loc[i])<=6]
            all_data_3 = all_data_3.loc[indexes_3]
    
    mean_RV = []
    lower_RV=[]
    upper_RV=[]
    mean_WEF = []
    lower_WEF=[]
    upper_WEF=[]
    mean_PF = []
    lower_PF=[]
    upper_PF=[]
    
    for p in phs:
        
        # to use a bootstrap percentile interval as error
        sample_means_RV = list()
        sample_means_WEF = list()
        sample_means_PF = list()
        
        for i in range(bt_n_iter):
            
            data_sample_1=all_data_1[all_data_1['ph']==p].sample(frac=0.7,replace=True,random_state=seed+i)
            data_sample_2=all_data_2[all_data_2['ph']==p].sample(frac=0.7,replace=True,random_state=seed+i)
            data_sample_3=all_data_3[all_data_3['ph']==p].sample(frac=0.7,replace=True,random_state=seed+i)
    	
            if not perc is None:
                sample_means_RV.append(np.percentile(data_sample_1[data_sample_1.columns[var]],perc))
                sample_means_WEF.append(np.percentile(data_sample_2[data_sample_2.columns[var]],perc))
                sample_means_PF.append(np.percentile(data_sample_3[data_sample_3.columns[var]],perc))
            else:
                sample_means_RV.append(np.mean(data_sample_1[data_sample_1.columns[var]]))
                sample_means_WEF.append(np.mean(data_sample_2[data_sample_2.columns[var]]))
                sample_means_PF.append(np.mean(data_sample_3[data_sample_3.columns[var]]))
            
        l, u, m = get_upper_lower_bootstr_interval(alpha,sample_means_RV,bt_n_iter)
        mean_RV.append(m)
        lower_RV.append(m-l)
        upper_RV.append(u-m)
        
        l, u, m = get_upper_lower_bootstr_interval(alpha,sample_means_WEF,bt_n_iter)
        mean_WEF.append(m)
        lower_WEF.append(m-l)
        upper_WEF.append(u-m)
        
        l, u, m = get_upper_lower_bootstr_interval(alpha,sample_means_PF,bt_n_iter)
        mean_PF.append(m)
        lower_PF.append(m-l)
        upper_PF.append(u-m)
        
    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(phs, mean_RV, 
                 yerr = [lower_RV,upper_RV], #uplims=True, lolims=True,
                 label='RV')    
    plt.errorbar(phs, mean_WEF, 
                 yerr = [lower_WEF,upper_WEF], #uplims=True, lolims=True,
                 label='WEF')
    plt.errorbar(phs, mean_PF, 
                 yerr = [lower_PF,upper_PF], #uplims=True, lolims=True,
                 label='PF') 

    plt.legend(loc='lower right')
    plt.xlabel('Probability of hunting ph')
    plt.ylabel(all_data_1.columns[var])
    if without_single_opt==True:
        name = './graphs/' + all_data_1.columns[var]+'_'+str(nh_greater_half)+'_wso'
    else:
        name = './graphs/' + all_data_1.columns[var]+'_'+str(nh_greater_half)
    
    if not Fnh_small is None:
        
        if Fnh_small:
            name = name + '_Fnhsmall'
        else:
            name = name + '_Fnhgreat'
            
    if n_part_great > 0:
        name = name + '_' + str(n_part_great) + 'npartgreat'
        
    if not perc is None:
        name = name + '_perc' +str(perc)
        
    name= name + '_3opt.eps'
    plt.savefig(name,dpi=300)
    plt.show()
    plt.close()

def creates_graph_and_computes_costs(direct,N,n_bits,option,ph,i_fn,x):
    
        h_ind = list(range(x[1][0])) 
        F = x[1][1]
        netw = x[0]
    
        if option==1: # input is a int key of dictionary 
            
            #convert to binary, first least binary is at the right extreme
            bin_i = bin(netw)[2:]
            # get strings of uniform length
            bin_i = bin_i.rjust(int(n_bits),"0")
        
            # get adjacency matrix
            if direct==False:
                A = ev.fill_matrix(bin_i,N)
            else:
                A = ev.fill_D_matrix(bin_i,N)
        
        else: # input i is a list
            
            A = ev.fill_matrix_from_list(netw,N,direct)    
        
        nw_costs = ev.nw_cost(A,h_ind,ph,F,N,i_fn,direct,receive_binary=False)
        """
        # compute recipr pairs with hunters
        recipr_pair_w_h = 0
        pairs= 0
    
        for i in range(N):
            for j in range(i):
            
                if A[i,j]==0 and A[j,i]==0:
                    continue
                
                # total pairs are the entire set of pairs in the graph
                pairs+=1
            
                # at least one of these symmetric coefficients is equal to 1
                # if they are equal we have a reciprocated pair
                if A[i,j]==A[j,i]:
                    if i in h_ind or j in h_ind: # a reciprocated pair with at least a hunter 
                        recipr_pair_w_h += 1
                        
        # with at least a non hunter
        pairs=0
        recipr_pairs_at_l_non_h = 0
        for i in range(N):
            for j in range(i):
                
                if A[i,j]==0 and A[j,i]==0:
                    continue
                
                pairs += 1
            
                if i in h_ind and j in h_ind:
                    continue
            
                # we have a pair of nodes (i,j), with at least 1 arc, and at least a non hunter
                # we hace a reciprocated pair
                if A[i,j]==A[j,i]:
                    recipr_pairs_at_l_non_h += 1
                
        if pairs == 0:
            rep_at_least_non_h = 0
            ans =0
        else:
            rep_at_least_non_h = recipr_pairs_at_l_non_h/pairs
            ans = recipr_pair_w_h/pairs"""
        
        
        return nw_costs[0], nw_costs[1] #, df_m.compute_recipr_pairs(A,N), compute_recipr_pairs_not_hunters(A,N,h_ind), ans, rep_at_least_non_h

def produce_corr_by_opt_type(all_data_1, all_data_2, all_data_3,
                             phs=[0.02,0.08,0.15,0.3,0.6],bt_n_iter=500,
                             seed=10,alpha=0.05,prop_sampl=0.7,without_single_opt=False,
                             nh_greater_half=None,Fnh_small=None,
                             n_part_great=0):
    
    # filter data
    if not Fnh_small is None:
        
        if Fnh_small:
            all_data_3 = all_data_3[all_data_3['F*nh']<=12] 
            all_data_2 = all_data_2[all_data_2['F*nh']<=12] 
            all_data_1 = all_data_1[all_data_1['F*nh']<=12] 
        else:
            all_data_3 = all_data_3[all_data_3['F*nh']>12] 
            all_data_2 = all_data_2[all_data_2['F*nh']>12] 
            all_data_1 = all_data_1[all_data_1['F*nh']>12] 
    
    if n_part_great > 0:
        
        all_data_3 = all_data_3[all_data_3['n_partitions']>n_part_great]
        all_data_2 = all_data_2[all_data_2['n_partitions']>n_part_great]
        all_data_1 = all_data_1[all_data_1['n_partitions']>n_part_great]
    
    if without_single_opt==True:
        all_data_3 = all_data_3[(all_data_3['is_RV']==False)&(all_data_3['is_WEF']==False)]    
    
    if not nh_greater_half is None:
        
        if nh_greater_half:
            indexes_1=[i for i in all_data_1.index if len(all_data_1['nh'].loc[i])>6]
            all_data_1 = all_data_1.loc[indexes_1]
            
            indexes_2=[i for i in all_data_2.index if len(all_data_2['nh'].loc[i])>6]
            all_data_2 = all_data_2.loc[indexes_2]
            
            indexes_3=[i for i in all_data_3.index if len(all_data_3['nh'].loc[i])>6]
            all_data_3 = all_data_3.loc[indexes_3]

        else:
            indexes_1=[i for i in all_data_1.index if len(all_data_1['nh'].loc[i])<=6]
            all_data_1 = all_data_1.loc[indexes_1]
            
            indexes_2=[i for i in all_data_2.index if len(all_data_2['nh'].loc[i])<=6]
            all_data_2 = all_data_2.loc[indexes_2]
            
            indexes_3=[i for i in all_data_3.index if len(all_data_3['nh'].loc[i])<=6]
            all_data_3 = all_data_3.loc[indexes_3]
    
    means_and_errors = []
    # 3 type of opt
    n_cond = 3
    
    # for each condition: series of (mean,lower_error,upper_error)     
    for i in range(n_cond):
        
        means_and_errors.append(([],[],[]))
    
    for p in phs:     
        
        RV_corr = []
        WEF_corr = []
        PF_corr = []
        
        all_data_1_p = all_data_1[all_data_1['ph']==p]
        all_data_2_p = all_data_2[all_data_2['ph']==p]
        all_data_3_p = all_data_3[all_data_3['ph']==p]
        
        for i in range(bt_n_iter):
            
            sample_RV = all_data_1_p.sample(n=None, frac=prop_sampl, replace=True, 
                                          weights=None, random_state=seed+i, 
                                          axis=0, ignore_index=False)
            
            sample_WEF = all_data_2_p.sample(n=None, frac=prop_sampl, replace=True, 
                                          weights=None, random_state=seed+i, 
                                          axis=0, ignore_index=False)
            sample_PF = all_data_3_p.sample(n=None, frac=prop_sampl, replace=True, 
                                          weights=None, random_state=seed+i, 
                                          axis=0, ignore_index=False)
            
            RV_corr.append(spearmanr(sample_RV['RV_cost'],sample_RV['WEF_cost'])[0])
            WEF_corr.append(spearmanr(sample_WEF['RV_cost'],sample_WEF['WEF_cost'])[0])
            PF_corr.append(spearmanr(sample_PF['RV_cost'],sample_PF['WEF_cost'])[0])
        
        l, u, m = get_upper_lower_bootstr_interval(alpha,RV_corr,bt_n_iter)
        means_and_errors[0][0].append(m)
        means_and_errors[0][1].append(m-l)
        means_and_errors[0][2].append(u-m)
        
        l, u, m = get_upper_lower_bootstr_interval(alpha,WEF_corr,bt_n_iter)
        means_and_errors[1][0].append(m)
        means_and_errors[1][1].append(m-l)
        means_and_errors[1][2].append(u-m)
        
        l, u, m = get_upper_lower_bootstr_interval(alpha,PF_corr,bt_n_iter)
        means_and_errors[2][0].append(m)
        means_and_errors[2][1].append(m-l)
        means_and_errors[2][2].append(u-m)
        
    fig = plt.figure(figsize=(8, 6))
    
    labels = ["RV","WEF","PF"]
    
    for i in range(n_cond):
        
        plt.errorbar(phs, means_and_errors[i][0], yerr = [means_and_errors[i][1],means_and_errors[i][2]],
                 label=labels[i]) 
    
    plt.legend(loc='lower right')
    plt.xlabel('Probability of hunting ph')
    plt.ylabel('Mean Spearman corr(RV,WEF)')
    plt.savefig('corr_by_opt_type.eps',dpi=300)
    plt.show()
    plt.close()
        
    return means_and_errors
    
# cond is a list of pairs (nh,F)
def graph_costcorr_p_II(cond,perc_conf_interv=True,N=12,direct=True,seed=10,sample_size=30,n_pbb_points=50,bt_n_iter=50,alpha=0.05,n_cores=4):
    
    # constants
    if direct==False:
        n_bits = N*(N-1)/2
    else:
        n_bits = N*(N-1)
        
    #df = pd.DataFrame()
    
    pool = Pool(processes=n_cores)
    
    # 50
    p_vector = np.linspace(0.02, 0.98, num=n_pbb_points, endpoint=True)
    
    random.seed(seed)
    
    with open("interp_fn_70000_norm.pickle", "rb") as f:
        i_fn = pickle.load(f)
    
    means_and_errors = []
    
    n_cond = len(cond)
    
    # for each condition: (mean,lower_error,upper_error)     
    for i in range(n_cond):
        
        means_and_errors.append(([],[],[]))
    
    for p in p_vector:
        
        # output: RV,WEF,rep,rep_n_h,rep_some_h,rep_at_least_non_h
        func = partial(creates_graph_and_computes_costs, direct,N,n_bits,1,p,i_fn)
        
        # to use a bootstrap percentile interval as error
        sample_corr = [[] for x in range(n_cond)]
        
        #netw_2 = sample_non_optima({},n_bits,N,direct,hunters_indexes,F,sample_size,pool,1,seed,True)
        sampled_networks = [random.getrandbits(n_bits) for x in range(sample_size*bt_n_iter)]
        
        output = pool.map(func,list(product(sampled_networks,cond)))
        
        for i in range(bt_n_iter):
            for j in range(n_cond):
            
                req_out = [output[j+ n_cond*l] for l in range(i*sample_size,(i+1)*sample_size)]
                sample_corr[j].append(spearmanr([x[0] for x in req_out],[x[1] for x in req_out])[0])
    	
        # lower error, upper error and mean, by condition
        l, u, m = get_upper_lower_bootstr_interval_II(alpha,sample_corr,bt_n_iter,perc_conf_interv)
        
        for i in range(n_cond):
            
            means_and_errors[i][0].append(m[i])
            means_and_errors[i][1].append(l[i])
            means_and_errors[i][2].append(u[i])
        
    fig = plt.figure(figsize=(8, 6))
    
    for i in range(n_cond):
        
        plt.errorbar(p_vector, means_and_errors[i][0], yerr = [means_and_errors[i][1],means_and_errors[i][2]],
                 label='(nh,F) = (' + str(cond[i][0]) + ", " + str(cond[i][1]) + ")") 
    
    plt.legend(loc='lower right')
    plt.xlabel('Probability of hunting ph')
    plt.ylabel('Mean Spearman corr(RV,WEF)')
    if perc_conf_interv:
        plt.savefig('sp_corr_sev_cond_perc.eps',dpi=300)
    else:
        plt.savefig('sp_corr_sev_cond_SEM_'+str(sample_size)+'samplsize_'+str(bt_n_iter)+'it.eps',dpi=300)
    plt.show()
    plt.close()
    
    if perc_conf_interv:
        with open("./corr_sev_cond_data_perc.pickle", "wb") as wr:
            wr.write(pickle.dumps([means_and_errors]))
    else:
        with open("./corr_sev_cond_data_SEM_"+str(sample_size)+"samplsize_"+str(bt_n_iter)+"it.pickle", "wb") as wr:
            wr.write(pickle.dumps([means_and_errors]))
    
    return means_and_errors

        
def test_corr_costs_II(perc_conf_interv=False):
    import time
    st= time.time()
    cond=[(1,4),(2,3),(3,3),(3,4),(6,3)]
    means_and_errors = graph_costcorr_p_II(cond,perc_conf_interv,N=12,direct=True,seed=10,sample_size=100,n_pbb_points=25,bt_n_iter=100,alpha=0.05,n_cores=8)
    #means_and_errors = graph_costcorr_p_II(cond,perc_conf_interv,N=12,direct=True,seed=10,sample_size=100,n_pbb_points=5,bt_n_iter=100,alpha=0.05,n_cores=4)
    print("took "+str(time.time() - st)+" secs.")
    
    #st= time.time()        
    #means_and_errors = graph_costcorr_p_II(cond,perc_conf_interv,N=12,direct=True,seed=10,sample_size=10,n_pbb_points=25,bt_n_iter=100,alpha=0.05,n_cores=8)
    #means_and_errors = graph_costcorr_p_II(cond,perc_conf_interv,N=12,direct=True,seed=10,sample_size=100,n_pbb_points=5,bt_n_iter=100,alpha=0.05,n_cores=4)
    #print("took "+str(time.time() - st)+" secs.")
    
    #st= time.time()        
    #means_and_errors = graph_costcorr_p_II(cond,perc_conf_interv,N=12,direct=True,seed=10,sample_size=100,n_pbb_points=25,bt_n_iter=1000,alpha=0.05,n_cores=8)
    #means_and_errors = graph_costcorr_p_II(cond,perc_conf_interv,N=12,direct=True,seed=10,sample_size=100,n_pbb_points=5,bt_n_iter=100,alpha=0.05,n_cores=4)
    #print("took "+str(time.time() - st)+" secs.")

def draw_a_sample(all_data_3,ind,n_sample=30):
    
    for i in sample(list(ind),n_sample):
        print(all_data_3.loc[i])
        draw_netw(all_data_3.loc[i,'ID'],12*11,True,all_data_3.loc[i,'nh'],12,
          test_layout=False,receive_int=True)
        

def compute_n_rep_pairs_type_intergroup_pbb_SS_and_n_hunt_std(direct,N,n_bits,all_data,i):
    
    h_ind = all_data.loc[i,'nh']
    nw_ID = all_data.loc[i,'ID']
    partition = all_data.loc[i,'partition']
    
    partitions = np.unique(partition)
    
    # count number of hunters per partition
    n_hunters_per_partition = []
    for l in partitions:
        count = 0
        for h in h_ind:
            if partition[h]==l:
                count += 1
        n_hunters_per_partition.append(count)
        
    # compute interpartition sum of squares of probability of eating
    
    ph= all_data.loc[i,'ph']
    F= all_data.loc[i,'F']
    #convert to binary, first least binary is at the right extreme
    bin_i = bin(nw_ID)[2:]
    # get strings of uniform length
    bin_i = bin_i.rjust(int(n_bits),"0")
        
    # get adjacency matrix
    if direct==False:
        A = ev.fill_matrix(bin_i,N)
        A = np.tril(A, -1)
    else:
        A = ev.fill_D_matrix(bin_i,N)
 
    pbb_vector = np.array(ev.compute_vector_pbb_eat(A,h_ind,ph,F,N,direct))
    grand_avg = np.mean(pbb_vector)
    
    n_per_group = []
    avg_per_group = []
    #compute average per group, and number of elements per group
    for l in partitions:
        indexes_l = (partition==l)
        n_per_group.append(np.sum(indexes_l))
        avg_per_group.append(np.mean(pbb_vector[indexes_l]))
    
    # finally compute the interpartition sum of squares
    interp_SS = sum([n_per_group[i]*avg_per_group[i]**2 for i in range(partitions.shape[0])]) - N*grand_avg**2
    
    rep_out = target_count_pair_type(direct,N,n_bits,2,h_ind,A)
    
    return rep_out[0], rep_out[1], rep_out[2], rep_out[3], np.std(n_hunters_per_partition), interp_SS  

def target_count_pair_type(direct,N,n_bits,option,h_ind,nw_ID):
        
    if option==1: # input is a int key of dictionary 
            
            #convert to binary, first least binary is at the right extreme
            bin_i = bin(nw_ID)[2:]
            # get strings of uniform length
            bin_i = bin_i.rjust(int(n_bits),"0")
        
            # get adjacency matrix
            if direct==False:
                A = ev.fill_matrix(bin_i,N)
            else:
                A = ev.fill_D_matrix(bin_i,N)
        
    else: # input i is the adjacency matrix
            
            A = nw_ID    
    
    recipr_pair_non_h = 0
    recipr_pair_only_h = 0
    recipr_pair_h_nh = 0
    non_recipr_pair = 0
    
    for i in range(N):
        for j in range(i):
            
            if A[i,j]==0 and A[j,i]==0:
                continue
            
            # at least one of these symmetric coefficients is equal to 1
            # if they are equal we have a reciprocated pair
            if A[i,j]==A[j,i]:
                
                if i in h_ind or j in h_ind: # a reciprocated pair with at least a hunter 
                    
                    if i in h_ind and j in h_ind:
                        recipr_pair_only_h += 1
                    else:
                        recipr_pair_h_nh += 1
                        
                else:
                    
                    recipr_pair_non_h += 1
                    
            else: # non reciprocated pair
                non_recipr_pair += 1
            
    return non_recipr_pair, recipr_pair_only_h, recipr_pair_h_nh, recipr_pair_non_h           
        
def add_more_variables(path):
    
    with open(path + "all_var_all_optima.pickle", "rb") as file:
        all_data = pickle.load(file)
        
    func = partial(compute_n_rep_pairs_type_intergroup_pbb_SS_and_n_hunt_std,True,12,12*11,all_data)
    
    pool = Pool(processes=cpu_count())
    
    output = pool.map(func,all_data.index)
    
    all_data.insert(all_data.shape[1],'n_rep_pairs_non_h',[x[3] for x in output])
    all_data.insert(all_data.shape[1],'n_rep_pairs_only_h',[x[1] for x in output])
    all_data.insert(all_data.shape[1],'n_rep_pairs_h_nh',[x[2] for x in output])
    all_data.insert(all_data.shape[1],'n_non_rep_pairs',[x[0] for x in output])
    all_data.insert(all_data.shape[1],'std_n_hunt',[x[4] for x in output])
    all_data.insert(all_data.shape[1],'interp_pbb_SS',[x[5] for x in output])
                                     
    pool.close()
    pool.join()
    
    return all_data
 
def produce_big_dataframe(dir_addr):
    
    # to produce the dataframe with all fields

    all_data_1, all_data_2, all_data_3 = compare_var_betw_opt(root=dir_addr)

    all_data = pd.concat([all_data_1,all_data_2,all_data_3],axis=0)
    all_data = all_data.reset_index(drop=True)

    with open(dir_addr + "extra_vars_all_optima.pickle", "rb") as file:
        [df_1, type_min_1, nw_IDs_1, nw_mv_1, 
         df_2, type_min_2, nw_IDs_2, nw_mv_2,
         df_3, type_min_3, nw_IDs_3, nw_mv_3] = pickle.load(file)
        
    type_min = type_min_1 + type_min_2 + type_min_3

    all_data.insert(all_data.shape[1],'type_min',type_min)

    with open(dir_addr + "all_var_all_optima.pickle", "wb") as wr:
        wr.write(pickle.dumps(all_data))
        
    # add more variables
    all_data = add_more_variables(path=dir_addr)

    with open(dir_addr + "all_var_all_optima.pickle", "wb") as wr:
        wr.write(pickle.dumps(all_data))

def compute_encoding(n_bits,all_data,i):
    
    nw_ID = all_data.loc[i,'ID']
 
    #convert to binary, first least binary is at the right extreme
    bin_i = bin(nw_ID)[2:]
    # get strings of uniform length
    bin_i = bin_i.rjust(int(n_bits),"0")
    
    return bin_i
    

def export_binary_encodings(path,n_cores=None,direct=True,N=12):
    
    with open(path + "all_var_all_optima.pickle", "rb") as file:
        all_data = pickle.load(file)
        
    if direct==False:
        n_bits = N*(N-1)/2
    else:
        n_bits = N*(N-1)
        
    func = partial(compute_encoding,n_bits,all_data)
    
    if n_cores is None:
        pool = Pool(processes=cpu_count())
    else:
        pool = Pool(processes=n_cores)
    
    output = pool.map(func,all_data.index)
    
    df = pd.DataFrame(output,columns =['nw_bin_encod'])
                                     
    pool.close()
    pool.join()
    
    df.to_csv('encod_all_optima.csv')
    
    return df

def compute_pl_vars_II(direct,N,n_bits,all_data,i):
    
    nw_ID = all_data.loc[i,'ID']
 
    #convert to binary, first least binary is at the right extreme
    bin_i = bin(nw_ID)[2:]
    # get strings of uniform length
    bin_i = bin_i.rjust(int(n_bits),"0")
        
    # get adjacency matrix
    if direct==False:
        A = ev.fill_matrix(bin_i,N)
        A = np.tril(A, -1)
    else:
        A = ev.fill_D_matrix(bin_i,N)
        
    # build graph
    g = Graph(directed=False)
    g.add_vertex(N)
    g.add_edge_list(np.transpose(A.nonzero()))
    
    # get degree sequence
    deg = g.get_out_degrees(range(N))
    
    # get deg-deg distance
    deg_deg_dist = []
    
    if np.shape(np.transpose(A.nonzero()))[0]==0: # no edges
    
        return 0, 0, 0#, 0, 0, 0, 0, 0, 0 # power law does not deliver an appropriate fit
    
    for s,t in g.iter_edges():
        deg_1 = deg[s]
        deg_2 = deg[t]
        deg_deg_dist.append(np.maximum(deg_1,deg_2)/np.minimum(deg_1,deg_2))
    
    deg_deg_dist.sort(reverse=True)
    ordered_data=np.array(deg_deg_dist)
    
    # fit tail exponent estimator
    try:
        results_h=tail_estimation.hill_estimator(ordered_data)
        #results_k=tail_estimation.kernel_type_estimator(ordered_data,int(0.3*len(ordered_data)))
    except ValueError:
        print("Hill estimator error")
        results_h=[0,0,0,0]
 
    try:
         results_m=tail_estimation.moments_estimator(ordered_data)
         #results_k=tail_estimation.kernel_type_estimator(ordered_data,int(0.3*len(ordered_data)))
    except (IndexError, ValueError) as error:
        print("Moments estimator error")
        results_m=[0,0,0,0]  
         
    try:
        results_k=tail_estimation.kernel_type_estimator(ordered_data,int(0.3*len(ordered_data)))
    except (IndexError, ValueError,ZeroDivisionError) as error:
        print("Kernel estimator error")
        results_k=[0,0,0,0]
 
    return results_h[3], results_m[3], results_k[3] 

def compute_pl_vars_III(direct,N,n_bits,all_data,i):
    
    nw_ID = all_data.loc[i,'ID']
 
    #convert to binary, first least binary is at the right extreme
    bin_i = bin(nw_ID)[2:]
    # get strings of uniform length
    bin_i = bin_i.rjust(int(n_bits),"0")
        
    # get adjacency matrix
    if direct==False:
        A = ev.fill_matrix(bin_i,N)
        A = np.tril(A, -1)
    else:
        A = ev.fill_D_matrix(bin_i,N)
        
    if np.shape(np.transpose(A.nonzero()))[0]==0: # no edges
    
        return 0, 0, 0, 0, 0, 0#, 0, 0, 0 # power law does not deliver an appropriate fit
        
    # build graph
    g = Graph(directed=False)
    g.add_vertex(N)
    g.add_edge_list(np.transpose(A.nonzero()))
    
    # get degree sequence
    out_deg = list(tail_estimation.add_uniform_noise(g.get_out_degrees(range(N))))
    in_deg = list(tail_estimation.add_uniform_noise(g.get_in_degrees(range(N))))
    out_deg.sort(reverse=True)
    ordered_data_out=np.array(out_deg)
    in_deg.sort(reverse=True)
    ordered_data_in=np.array(in_deg)
    
    
    # fit tail exponent estimator
    try:
        results_h_out=tail_estimation.hill_estimator(ordered_data_out)
    except ValueError:
        results_h_out=[0,0,0,0]
    
    try:
        results_m_out=tail_estimation.moments_estimator(ordered_data_out)
    except (IndexError, ValueError) as error:
        print("Moments estimator error")
        results_m_out=[0,0,0,0]
    
    try:
        results_k_out=tail_estimation.kernel_type_estimator(ordered_data_out,int(0.3*len(ordered_data_out)))
    except (IndexError, ValueError,ZeroDivisionError) as error:
        print("Moments estimator error")
        results_k_out=[0,0,0,0]
        
    
    try:
        results_h_in=tail_estimation.hill_estimator(ordered_data_in)
    except ValueError:
        results_h_in=[0,0,0,0]
    
    try:
        results_m_in=tail_estimation.moments_estimator(ordered_data_in)
    except (IndexError, ValueError) as error:
        print("Moments estimator error")
        results_m_in=[0,0,0,0]
        
    try:
        results_k_in=tail_estimation.kernel_type_estimator(ordered_data_in,int(0.3*len(ordered_data_in)))
    except (IndexError, ValueError,ZeroDivisionError) as error:
        print("Moments estimator error")
        results_k_in=[0,0,0,0]
 
    return results_h_out[3], results_m_out[3], results_k_out[3], results_h_in[3], results_m_in[3], results_k_in[3]


def add_pl_variables_II(path):
    
    with open(path + "all_var_all_optima.pickle", "rb") as file:
        all_data = pickle.load(file)
        
    func = partial(compute_pl_vars_II,True,12,12*11,all_data)
    
    pool = Pool(processes=14)
    
    output = pool.map(func,all_data.index)
    
    # deg deg distance, out-degree
    all_data.insert(all_data.shape[1],'gamma_hill',[x[0] for x in output])
    all_data.insert(all_data.shape[1],'gamma_mom',[x[1] for x in output])
    all_data.insert(all_data.shape[1],'gamma_kern',[x[2] for x in output])
                                     
    pool.close()
    pool.join()
    
    with open(path + "all_var_all_optima_w_pl_degdegDist.pickle", "wb") as wr:
        wr.write(pickle.dumps(all_data))
    
    return all_data

def add_pl_variables_III(path):
    
    with open(path + "all_var_all_optima.pickle", "rb") as file:
        all_data = pickle.load(file)
        
    func = partial(compute_pl_vars_III,True,12,12*11,all_data)
    
    pool = Pool(processes=4)
    
    output = pool.map(func,all_data.index)
    
    # deg deg distance, out-degree
    all_data.insert(all_data.shape[1],'gamma_hill_out',[x[0] for x in output])
    all_data.insert(all_data.shape[1],'gamma_mom_out',[x[1] for x in output])
    all_data.insert(all_data.shape[1],'gamma_kern_out',[x[2] for x in output])
    all_data.insert(all_data.shape[1],'gamma_hill_in',[x[3] for x in output])
    all_data.insert(all_data.shape[1],'gamma_mom_in',[x[4] for x in output])
    all_data.insert(all_data.shape[1],'gamma_kern_in',[x[5] for x in output])    
                                 
    pool.close()
    pool.join()
    
    with open(path + "all_var_all_optima_w_pl_degDistr.pickle", "wb") as wr:
        wr.write(pickle.dumps(all_data))
    
    return all_data

