"""
Copyright (c) 2022, Francisco Plana
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


from scipy.stats import uniform, norm, ttest_ind
import time
from sklearn.model_selection import ParameterSampler
import numpy as np
from multiprocessing import Pool, cpu_count 
from functools import partial
from itertools import product, combinations
from apted.helpers import Tree
from apted import APTED
from sklearn.metrics import balanced_accuracy_score
from numpy.random import default_rng
from math import floor, ceil
from graph_tool.all import *
import matplotlib.pyplot as plt
from adjustText import adjust_text
import classif_tree as ct
import pandas as pd
from direpack.preprocessing import gsspp
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, cluster_optics_dbscan
#from sklearnex import patch_sklearn
import pickle
from copy import deepcopy
import matplotlib.gridspec as gridspec 
import os
from collections import OrderedDict
import evol_methods as ev
import scipy.sparse as sp


# code based on https://github.com/nogueirs/JMLR2018/blob/master/python/stabilityDemo.ipynb  
def getBootstrapSample(data,labels,r=1,seed=10):
    '''
    This function takes as input the data and labels and returns 
    a bootstrap sample of the data, as well as its out-of-bag (OOB) data
    
    INPUTS:
    - data is a 2-dimensional numpy.ndarray where rows are examples and columns are features
    - labels is a 1-dimansional numpy.ndarray giving the label of each example in data
    - r is the proportion (float between 0 and 1) of data we will sample
    
    OUPUT:
    - a dictionnary where:
          - key 'bootData' gives a 2-dimensional numpy.ndarray which is a bootstrap sample of data
          - key 'bootLabels' is a 1-dimansional numpy.ndarray giving the label of each example in bootData
          - key 'OOBData' gives a 2-dimensional numpy.ndarray the OOB examples
          - key 'OOBLabels' is a 1-dimansional numpy.ndarray giving the label of each example in OOBData
    '''
    rg = default_rng(seed)
    m,d=data.shape
    if m!= len(labels):
        raise ValueError('The data and labels should have a same number of rows.')
    # np.unique since then assuming indexes are unique in setdiff1d
    ind=rg.choice(range(m), size= floor(r * m), replace=True)
    OOBind=np.setdiff1d(range(m),np.unique(ind), assume_unique=True)
    bootData=data[ind,]
    bootLabels=labels[ind]
    OOBData=data[OOBind,]
    OOBLabels=labels[OOBind]
    return {'bootData':bootData,'bootLabels':bootLabels,'OOBData':OOBData,'OOBLabels':OOBLabels}

def get_TE_sim_II(Tree_models,feature_names,pair):
    
    DT1_br = Tree_models[pair[0]].DT_to_bracket(feature_names)[0]
    DT2_br = Tree_models[pair[1]].DT_to_bracket(feature_names)[0]
    
    return -APTED(Tree.from_text(DT1_br), 
                  Tree.from_text(DT2_br)).compute_edit_distance()

def sample_and_fit_DT_II(param_list,X,Y,r,seed,feature_names,int_I,int1st,n_budget_sample):
    
    l = n_budget_sample[0]
    i = n_budget_sample[1]
    
    mss_l = param_list[l]['min_samples_split']
    msl_l = param_list[l]['min_samples_leaf'] 
    md_l = param_list[l]['max_depth']
    mid_l = param_list[l]['min_impurity_decrease']
    
    newData=getBootstrapSample(X,Y,r,seed + i) ## we get bootstrap samples
    
    DT = ct.LinearTreeClassifier(base_estimator=None,min_samples_split=mss_l, 
                                min_samples_leaf=msl_l,min_impurity_decrease=mid_l,
                                max_depth = md_l)
    
    if int_I:
        DT.fit(newData['bootData'],newData['bootLabels'])
        predLabels_DT = DT.predict(newData['OOBData'])
    else:    
        DT.fit_II(newData['bootData'],newData['bootLabels'],inter1st=int1st)
        predLabels_DT = DT.predict_II(newData['OOBData'])
    
    acc = balanced_accuracy_score(newData['OOBLabels'], predLabels_DT, adjusted=True)
            
    return DT, acc

# Pareto Front Search of accuracy and stability, with Decision Tree 
def PFsearch_DT_stbacc_II(X,Y,seed,feature_names,n_cores=None,par_distr=dict(min_samples_split = list(range(2,40)), 
                                               min_samples_leaf = list(range(1,20)),
                                               max_depth = list(range(3,9)),
                                               min_impurity_decrease = uniform(0,0.05)),
                       HP_budget=100,M = 30,r=1,alpha_conf=0.05,int_I=True,int1st=False):
    
    t1 = time.time()
    
    # sample hyper parameters
    param_list = list(ParameterSampler(par_distr, n_iter=HP_budget, random_state=seed))
    
    number_of_pairs = int(M * (M - 1)/2)
    
    # initialize data containers    
    acc_net=np.zeros((HP_budget,M))
    models = []
    pairw_sim_val = np.zeros((HP_budget,number_of_pairs))
    
    # fit the HP_budget x M models to perform random search, in parallel
    if n_cores is None:
        pool = Pool(processes=cpu_count()) 
    else:
        pool = Pool(processes=n_cores)
    
    fun = partial(sample_and_fit_DT_II,param_list,X,Y,r,seed,feature_names,int_I,int1st)
 
    models_and_accuracy = pool.map(fun,list(product(range(HP_budget),range(M))))
    
    all_distinct_pairs = list(combinations(list(range(M)),2))
    
    # recover each model and its accuracy
    # also compute pairwise model similarities for each hyper parameter set
    for l in range(HP_budget):
        
        models.append([])
        
        for i in range(M):
            
            models[l].append( models_and_accuracy[l*M + i][0])
            acc_net[l,i] = models_and_accuracy[l*M + i][1]
        #print("l: "+ str(l))    
        fun = partial(get_TE_sim_II, models[l],feature_names)
        pairw_sim_val[l,] = pool.map(fun,all_distinct_pairs)
    
    PF_points = compute_non_dominated_stab_acc_DT(HP_budget,acc_net,pairw_sim_val,M,alpha_conf)

    print("time: " + str(time.time() - t1))
    
    pool.close()
    pool.join()
    
    return acc_net, pairw_sim_val, PF_points, param_list, models

# returns True if element l1-th dominates l2-th
def max_dominates_DT(l1,l2,acc_net,pairw_sim_val,alpha=0.05):
    
    # compare accuracy of 2 samples, 1-tail
    stat, pval = ttest_ind(acc_net[l1,:], acc_net[l2,:], equal_var=False, nan_policy='raise',alternative='less')
    
    if pval < alpha: # reject 1-sided t-test => element l1-th does not dominate l2
        return False
    
    # compare stability of samples, 1-tail
    stat, pval = ttest_ind(pairw_sim_val[l1,:], pairw_sim_val[l2,:], equal_var=False, nan_policy='raise',alternative='less')
    
    if pval < alpha: # reject 1-sided t-test => element l1-th does not dominate l2
        return False
    
    # compare accuracy of 2 samples, 2-tail
    stat, pval = ttest_ind(acc_net[l1,:], acc_net[l2,:],  equal_var=False, nan_policy='raise', alternative='two-sided')
    
    if pval < alpha: # reject 2-sided t-test => element l1-th does dominate l2 since components are distinct
        return True
    
    # compare stability of samples, 2-tail
    stat, pval = ttest_ind(pairw_sim_val[l1,:], pairw_sim_val[l2,:],  equal_var=False, nan_policy='raise', alternative='two-sided')
    
    if pval < alpha: # reject 2-sided t-test => element l1-th does dominate l2 since components are distinct
        return True
    
    return False

def compute_non_dominated_stab_acc_DT(HP_budget,acc_net,pairw_sim_val,M,alpha=0.05):
    
    # perform PF search over stability and error values
    archive = {}
    
    number_of_pairs = M * (M - 1)/2
    
    # we iterate over all models (all l1_ratios and alphas)
    for l in range(HP_budget):
        
        # archive initialization
        if len(archive)==0:
                
            avg, stddv = np.mean(pairw_sim_val[l,]), np.std(pairw_sim_val[l,])
            conf_int_stb = norm.interval(1 - alpha, loc=avg, scale=stddv/np.sqrt(number_of_pairs))
                
            mu, sigma = np.mean(acc_net[l,]), np.std(acc_net[l,])
            conf_int = norm.interval(1 - alpha, loc=mu, scale=sigma/np.sqrt(M))
                
            archive[l] = ({'stability': avg, 'lower': conf_int_stb[0], 'upper': conf_int_stb[1]}, 
                           {'accuracy': mu, 'lower': conf_int[0], 'upper': conf_int[1]})
            continue
        
        # pair l is max dominated by archive[key] => next
        if any( max_dominates_DT(key, l, acc_net, pairw_sim_val, alpha)  for key in archive):
            continue
            
        # if k,l is not in archive, add it and check for dominated to erase
        is_in_archive = [is_HP_on_DB_DT(key,archive,l,pairw_sim_val,acc_net) for key in archive]
            
        # stb and accuracy of model l is not in archive  
        if sum(is_in_archive)==0:
                
            avg, stddv = np.mean(pairw_sim_val[l,]), np.std(pairw_sim_val[l,])
            conf_int_stb = norm.interval(1 - alpha, loc=avg, scale=stddv/np.sqrt(number_of_pairs))
                
            mu, sigma = np.mean(acc_net[l,]), np.std(acc_net[l,])
            conf_int = norm.interval(1 - alpha, loc=mu, scale=sigma/np.sqrt(M))
                
            archive[l] = ({'stability': avg, 'lower': conf_int_stb[0], 'upper': conf_int_stb[1]}, 
                           {'accuracy': mu, 'lower': conf_int[0], 'upper': conf_int[1]})
            
            # since it is not dominated, erase those keys dominated
            for q in [key for key in archive if max_dominates_DT(l,key,acc_net, pairw_sim_val,alpha) ]:
                archive.pop(q)
                  
    return archive

def is_HP_on_DB_DT(key,archive,l,pairw_sim_val,acc_net):
    
    new_stb = np.mean(pairw_sim_val[l,])
    new_acc = np.mean(acc_net[l,])
    
    dict_stb = archive[key][0]
    dict_acc = archive[key][1]
    
    return new_stb >= dict_stb['lower'] and new_stb <= dict_stb['upper'] and new_acc >= dict_acc['lower'] and new_acc <= dict_acc['upper']

# produce OrderedDict where key is summary DT, and value a list of DTs with attributes
# like feature set, accuracy, etc
def print_PF_stb_acc_DT_IV(dict_dist_DTs,depth,count,dict_repr_DT,M, PF_points, param_list, models, feature_names, X,Y,seed,r=1,folder = None, sort_by_acc=True, int_I=True):
    
    acc = []
    stb = []
    acc_lower = []
    acc_upper = []
    stb_lower = []
    stb_upper = []
    
    if sort_by_acc:
    
        sorted_keys = sorted(PF_points.keys(),key= lambda x: PF_points[x][1]['accuracy'])
        
    else:
        
        sorted_keys = sorted(PF_points.keys(),key= lambda x: PF_points[x][0]['stability'])
    
    for key in sorted_keys:
        
        acc.append(PF_points[key][1]['accuracy'])
        acc_lower.append(PF_points[key][1]['accuracy'] - PF_points[key][1]['lower'])
        acc_upper.append(PF_points[key][1]['upper'] - PF_points[key][1]['accuracy'])
        
        stb.append(PF_points[key][0]['stability'])
        stb_lower.append(PF_points[key][0]['stability'] - PF_points[key][0]['lower'])
        stb_upper.append(PF_points[key][0]['upper'] - PF_points[key][0]['stability'])
    
    # dict to store distinct trees found 
    # where key = DT_str, value = [(key_from_PF_points,DT_HP),..]
    
    print("Sorted list of hyperparameters and respective DT")
    #print(sorted_keys)
    for l, key in enumerate(sorted_keys):
        
        if models is None:
            
            mss_l = param_list[key]['min_samples_split']
            msl_l = param_list[key]['min_samples_leaf'] 
            md_l = param_list[key]['max_depth']
            mid_l = param_list[key]['min_impurity_decrease']
        
        # to reproduce same DT's produced in PFsearch_DT_stbacc_II()
        for i in range(M): 
            #print("key: "+str(key)+", i: "+str(i))
            if models is None:
                
                DT = ct.LinearTreeClassifier(base_estimator=None,min_samples_split=mss_l, 
                                min_samples_leaf=msl_l,min_impurity_decrease=mid_l,
                                max_depth = md_l)
    
                newData=getBootstrapSample(X,Y,r,seed + i) ## we get bootstrap samples
    
                if int_I:
                    DT.fit(newData['bootData'],newData['bootLabels'])
                else:    
                    DT.fit_II(newData['bootData'],newData['bootLabels'])
                    
                DT_str, DT_str_summ, max_depth = DT.DT_to_bracket(feature_names)
                    
            else:
                DT = models[key][i]
                DT_str, DT_str_summ, max_depth = DT.DT_to_bracket(feature_names)
            #print("key: "+str(key)+", i: "+str(i)+", 2nd mark")
            # if DT is already on archive (ie, it was printed)
            # add HP producing it, only if it has not been added before  
            if DT_str_summ in dict_dist_DTs:
                
                dict_dist_DTs[DT_str_summ].append([acc[l],acc_lower[l],acc_upper[l],
                                                 stb[l],stb_lower[l],stb_upper[l],
                                                 param_list[key],seed+i,DT_str])
                count[DT_str_summ] += 1
            else:
                dict_dist_DTs[DT_str_summ] = []
                dict_dist_DTs[DT_str_summ].append([acc[l],acc_lower[l],acc_upper[l],
                                                 stb[l],stb_lower[l],stb_upper[l],
                                                 param_list[key],seed+i,DT_str])
                depth[DT_str_summ] = max_depth
                count[DT_str_summ] = 1
                dict_repr_DT[DT_str_summ] = DT
                
                # print the DT since it is found by 1st time
        
                if folder is None:
                    filename = "DT_" + str(len(list(dict_dist_DTs.keys())) - 1)        
                else:  
                    filename = folder + "/DT_" + str(len(list(dict_dist_DTs.keys())) - 1) 
                
                if int_I:
                    DT.plot_model(feature_names = feature_names,
                              file=filename)
                else:
                    DT.plot_model_II(feature_names = feature_names,
                              file=filename)
        
    return dict_dist_DTs, count, depth, dict_repr_DT

# receive the output of cv.print_PF_stb_acc_DT()
def obtain_graph_DT_inclusions(archive, counter, depth, path=None, PF_points_l=None):
    
    # depth is in the same order as archive, here we produce a list with the max depth of every DT
    DT_max_depths = [depth[k] for k in depth]
    
    sorted_ind = [ix for ix, v in sorted(enumerate(DT_max_depths), key=lambda x: x[1])]
    
    sorted_keys_by_max_depth = [list(archive.keys())[i] for i in sorted_ind]
    sorted_max_depths = [DT_max_depths[i] for i in sorted_ind] 
    
    unique_max_depths = list(set(sorted_max_depths))
    
    # build a graph of correlations
    g = Graph(directed=True)
    g.add_vertex(len(archive.keys()))
    g.vp.labels = g.new_vertex_property("int", vals= list(range(len(archive.keys()))) )
    g.vp.DT = g.new_vertex_property("string", vals=list(archive.keys()) )
    g.vp.DT_freq = g.new_vertex_property("double", vals= [counter[k]/10 for k in counter] )
    #g.vp.accur = g.new_vertex_property("double", vals= [np.log(1000/(1 - PF_points_l[archive[k][0][0]][1]['accuracy'])) for k in archive] ) 
    
    # check DT inclusions by traversing pairs of ascending FS sizes
    for val1, val2 in zip(unique_max_depths[0:-1], unique_max_depths[1:]):
            
        ind_val1 = [i for i, x in enumerate(sorted_max_depths) if x == val1]
        ind_val2 = [i for i, x in enumerate(sorted_max_depths) if x == val2]
        
        for ind1 in ind_val1:
            for ind2 in ind_val2:
            
                is_ind1_subset_of_ind2 = is_subtree(sorted_keys_by_max_depth[ind1],
                                                       sorted_keys_by_max_depth[ind2])
                
                if is_ind1_subset_of_ind2:
                    g.add_edge(g.vertex(sorted_ind[ind1]), 
                               g.vertex(sorted_ind[ind2]))
                    
    #graph_draw(g,pos=arf_layout(g, max_iter=50),vertex_text=g.vp.labels,vertex_size=g.vp.DT_freq,
    #          vertex_text_position=0.5)
    
    if path is None:
        
        filename = "DT_inclusions.pdf"
        
    else:
        
        filename = path + "/DT_inclusions.pdf"
    
    if len(archive.keys()) > 1:
    
        graph_draw(g,pos=fruchterman_reingold_layout(g),
               vertex_text=g.vp.labels,
               vertex_size=g.vp.DT_freq,
               vertex_text_position=0.2,
               output=filename)
        
# T1 and T2 are decision trees in string bracket notation as produced by DT_to_bracket()
# returns True if T1 is subtree of T2
def is_subtree(T1, T2):
    
    if T1 == T2:
        return True
    
    # parse two trees
    DT1 = Tree.from_text(T1)
    DT2 = Tree.from_text(T2)
    
    return compare_two_trees_is_subtree(DT1,DT2)

# recursive function to test if DT1 is subtree of DT2
def compare_two_trees_is_subtree(DT1,DT2):

    if len(DT1.children)==len(DT2.children)==0:
        return DT1.name==DT2.name
    
    if len(DT1.children) > 0: # DT1 is not a leaf
        
        if len(DT1.children) > len(DT2.children):
            
            return False
        
        else:
            
            if DT1.name!=DT2.name:
                return False
            else:
                return compare_two_trees_is_subtree(DT1.children[0],DT2.children[0]) and compare_two_trees_is_subtree(DT1.children[1],DT2.children[1])
                
    else: # since DT1 is a leaf, and DT2 is not a leaf
        # returns True if leaf DT1 is substituted by Tree DT2
        if type(DT2.children[0]) == Tree and type(DT2.children[1]) == Tree and type(DT2.children) == list and len(DT2.children)==2:
            
            return True
    
    return False
        
def draw_stb_acc_set_of_trees(dict_dist_DTs,folder=None):
    #import matplotlib.transforms as mtransforms
    
    stb = [np.mean(np.array(list(x[3] for x in dict_dist_DTs[key]))) for key in dict_dist_DTs]
    acc = [np.mean(np.array(list(x[0] for x in dict_dist_DTs[key]))) for key in dict_dist_DTs]
    
    acc_lower = [np.mean(np.array(list(x[1] for x in dict_dist_DTs[key]))) for key in dict_dist_DTs]
    acc_upper = [np.mean(np.array(list(x[2] for x in dict_dist_DTs[key]))) for key in dict_dist_DTs]
    stb_lower = [np.mean(np.array(list(x[4] for x in dict_dist_DTs[key]))) for key in dict_dist_DTs]
    stb_upper = [np.mean(np.array(list(x[5] for x in dict_dist_DTs[key]))) for key in dict_dist_DTs]
    #stb_sd = np.std(stb)
    #acc_sd = np.std(acc)
    
    fig = plt.figure(figsize=(10, 6))
    axes = plt.axes()
    axes.set_title('Models with largest accuracy-stability efficiency')
    #ax.legend(loc='lower right')
    axes.set_xlabel('Stability')
    axes.set_ylabel('Accuracy')
    
    #trans_offset = mtransforms.offset_copy(axes.transData, fig=fig,
    #                                   x=0.05, y=0.10, units='inches')
    #plt.errorbar(stb, acc, yerr=acc_sd, xerr=stb_sd, fmt='.')
    plt.errorbar(stb, acc, yerr=[acc_lower,acc_upper], xerr=[stb_lower, stb_upper], fmt='.')
    #for i, (x, y) in enumerate(zip(stb, acc)):
    #    plt.text(x, y, '%d' % i, transform=trans_offset)
    texts = [plt.text(x, y, '%d' % i) for i, (x, y) in enumerate(zip(stb, acc))]
    adjust_text(texts)
    #plt.errorbar(stb, acc,fmt='-o')
    if folder is None:
        plt.savefig("Acc-Stb_efficiency_DT.pdf")
    else:
        plt.savefig(folder + "/Acc-Stb_efficiency_DT.pdf")
    plt.show()
    plt.close()
    
def compute_generalized_sign(outp_df):
    
    rad_fun='quad'
    t0=time.time()
    X = gsspp.gen_ss_pp( outp_df, fun = rad_fun, center = 'kstepLTS')
    datafr = pd.DataFrame(X, columns = outp_df.columns, index=outp_df.index)
    n_rows,n_feat=datafr.shape
    print("time of %d secs. in scaling dataset of size %d x %d" % (time.time() - t0,n_rows,n_feat))
    
    return datafr
    
# tSNE with HP tuned to work with large datasets
# as such associated to 1 type of optimum  
def tSNE_and_OPTICS_one_opt(opt,seed,n_s=None,path="/home/fplana/evol_datafr_analysis/",dim=2,ite=1000,prop_min_sampl=.03,proc_NA=False):
    
    outp_df, type_min, nw_IDs, nw_mv = preprocess_datafr_1_opt(opt,s=seed,root=path,n_sample=n_s,proc_NA=proc_NA)
    
    datafr = compute_generalized_sign(outp_df)
    n_samples = datafr.shape[0]
    
    perp = n_samples * 0.01
    EE = 4
    LR= max(200,n_samples/EE)
    
    patch_sklearn("TSNE")
    
    t0 = time.time()
    tsne = TSNE(n_components=dim, init='pca',
                     random_state=seed, perplexity=perp, 
                     n_iter=ite,early_exaggeration=EE,
                     learning_rate=LR)
    tsne.fit(datafr)
    t1 = time.time()
    print("TSNE computed in %.2g sec and %d iters" % ( t1 - t0,tsne.n_iter_))    
    
    clust = OPTICS(min_samples=prop_min_sampl, 
                   cluster_method="dbscan")
    
    X=tsne.embedding_
    t0=time.time()
    clust.fit(X)
    print("OPTICS time fitting map data: "+str(time.time()-t0)+" secs.")
    
    if not proc_NA:
        name= path + "clusters/tSNE_"+str(opt)+"opt_"+str(seed)+ "seed.pickle"
    else:
        name = path + "clusters/tSNE_"+str(opt)+"opt_NA_"+str(seed)+ "seed.pickle"
    
    with open(name, "wb") as wr:
        wr.write(pickle.dumps([outp_df, type_min, nw_IDs, nw_mv, tsne, clust]))
        
def produce_new_PF_tSNEs(dir_addr):

    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.6,F_nh_small=True,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.3,F_nh_small=True,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.08,F_nh_small=True,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.15,F_nh_small=True,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.02,F_nh_small=True,path=dir_addr)
    
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.6,F_nh_small=False,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.3,F_nh_small=False,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.08,F_nh_small=False,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.15,F_nh_small=False,path=dir_addr)
    tSNE_and_OPTICS_PF_ph(seed=50,ph=0.02,F_nh_small=False,path=dir_addr)
    
def filter_df(fs,ph,F_nh_small,path):
    
    with open(path + "all_var_all_optima.pickle", "rb") as file:
        all_data = pickle.load(file)
     
    # filter PF data, the ph, F, nh region
    all_data_3 = all_data[all_data['type_min']==3]
    
    all_data_3 = all_data_3[all_data_3['ph']==ph]
    
    del all_data
    
    if F_nh_small:
        all_data_3 = all_data_3[all_data_3['F*nh']<=12]
    else:
        all_data_3 = all_data_3[all_data_3['F*nh']>12]
        
    # filter required features
    nw_IDs = all_data_3['ID']
    nw_mvs = all_data_3[['F', 'nh', 'ph']]
    all_data_3 = all_data_3[fs]
    
    return all_data_3, nw_mvs, nw_IDs

def produce_several_PF_ph_DTs(dir_addr):
    
    produce_PF_ph_DTs(ph=0.3,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=True,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.02,rst=4,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.08,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=True,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.02,rst=3,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.15,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=True,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.03,rst=3,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.02,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=True,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.02,rst=4,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.6,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=True,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.03,rst=7.5,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.3,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=False,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.11,rst=3,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.08,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=False,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.03,rst=4.5,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.15,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=False,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.07,rst=4.5,inter1st=False)
    
    produce_PF_ph_DTs(ph=0.02,seed=50,n_cores=18,HP=500,nM=15,F_nh_small=False,
                          root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.02,rst=3,inter1st=False)

# to produce DTs of the lusters from tSNE_and_OPTICS_one_opt()
# or tSNE_and_OPTICS_one_ph()    
def produce_PF_ph_DTs(ph,seed,n_cores,HP,nM,F_nh_small=True,
                      root='/home/fplana/evol_datafr_analysis/',
                   inter_I=False,prop_min_sampl=.03,
                   mnp=0.02,rst=1.5,inter1st=False):  
    
    name=root + "clusters/tSNE_PF_"+str(seed)+ "seed_"+str(ph)+"ph_"+str(F_nh_small)+"Fnhsmall.pickle"
    
    with open(name, "rb") as file:
        [tsne, clust, fs_TSNE] = pickle.load(file)      
        
    path = root + "clusters/DTs_PF_"+str(seed)+"seed_" + str(ph) + "ph_"  + str(F_nh_small) + "_F_nh_small" 
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        
    
    features_to_select = fs_TSNE + ['pbb_non_h']
      
    outp_df, nw_mvs, nw_IDs = filter_df(features_to_select,ph,F_nh_small,root)
    
    # find least-feature-variance eps
    l_v_eps, l_std = find_epsilon_III(clust,outp_df,max_nois_prop=mnp,ratio_std_thr=rst)

    labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=l_v_eps)
    
    # it is not necessary to preprocess features
    # since we want to compute DTs on original feat scale
    # not necessary to drop na, already done when tSNE was computed
    # and it may produce differences with labels vector
    
    # add met_variables
    outp_df.insert(outp_df.shape[1],'F',nw_mvs['F'])
    outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mvs['nh']])
    outp_df.insert(outp_df.shape[1],'ph',nw_mvs['ph'])
    outp_df.insert(outp_df.shape[1],'F*nh',np.multiply(outp_df['F'],outp_df['nh']))
    outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
    

    features_to_select = features_to_select + ['F','nh','ph','F*nh','F*nh*ph']

    X = np.array(outp_df[features_to_select])
        
    # produce possible DTs
    start = time.time()    
    acc_net_l, pairw_sim_val_l, PF_points_l, param_list_l, models_l = PFsearch_DT_stbacc_II(X,labels,seed,features_to_select,n_cores,
                                            par_distr=dict(min_samples_split = uniform(3*10**-5,10**-3), 
                                            min_samples_leaf = uniform(3*10**-5,10**-3),
                                            max_depth = list(range(3,9)),
                                            min_impurity_decrease = uniform(0,0.01)),
                                            HP_budget=HP,M = nM,r=0.7,alpha_conf=0.05,int_I=inter_I,int1st=inter1st)
        
    end = time.time()
    print("Last DT search took "+ str(end-start)+ " secs.")
            
    # produce the distinct DT's
    # instantiate these dicts here since there is
    # a processsing and folder for each FS
    distinct_DTs = OrderedDict()
    depth = OrderedDict()
    counter = OrderedDict()
    dict_repr_DT = OrderedDict()
    distinct_DTs, counter, depth, dict_repr_DT = print_PF_stb_acc_DT_IV(distinct_DTs,depth,counter, dict_repr_DT,nM,
                                                                PF_points_l, param_list_l, models_l,
                                                                features_to_select,X,labels,seed,r=1, folder= path, 
                                                                sort_by_acc=True, int_I=inter_I)
                
    with open(path +"/" + "distinct_DT.pickle", "wb") as wr:
        wr.write(pickle.dumps([distinct_DTs, counter, depth, 
                               dict_repr_DT,labels,l_v_eps,
                               features_to_select,HP,nM]))
    
    # obtain DT inclusions
    obtain_graph_DT_inclusions(distinct_DTs, counter, depth, path)
    draw_stb_acc_set_of_trees(distinct_DTs,path)
    
def graph_DT_PF_ph(ph,seed,root,n_DT,F_nh_small=True,
                   is_dir=True,lay=True,p_vr=None,to_file=False,extra_DTfeat=None):   
        
    name= root + "clusters/tSNE_PF_"+str(seed)+ "seed_"+str(ph)+"ph_"+str(F_nh_small)+"Fnhsmall.pickle"
    
    with open(name, "rb") as file:
        [tsne, clust, fs] = pickle.load(file)      
        
    path = root + "clusters/DTs_PF_"+str(seed)+"seed_" + str(ph) + "ph_"  + str(F_nh_small) + "_F_nh_small/distinct_DT.pickle"
    
    # open DTs data
    with open(path, "rb") as file:
        [distinct_DTs, counter, depth, 
         dict_repr_DT,labels,l_v_eps,
         features_to_select,HP,nM] = pickle.load(file) 
    
    n_key = list(dict_repr_DT.keys())[n_DT]
    DT = dict_repr_DT[n_key]
    
    constr_dict = extract_constraints_from_DT(DT)
    
    features_to_select = fs + ['pbb_non_h']

    outp_df, nw_mv, nw_IDs = filter_df(features_to_select,ph,F_nh_small,root)
    
    outp_df.insert(outp_df.shape[1],'F',nw_mv['F'])
    outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
    outp_df.insert(outp_df.shape[1],'ph',nw_mv['ph'])
    outp_df.insert(outp_df.shape[1],'F*nh',np.multiply(outp_df['F'],outp_df['nh']))
    outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
    
    features_to_select = features_to_select + ['F','nh','ph','F*nh','F*nh*ph']
    
    outp_df = outp_df[features_to_select]
    type_min = [3]*outp_df.shape[0]
    
    if not extra_DTfeat is None:
        add_data, nw_mv, nw_IDs = filter_df(extra_DTfeat,ph,F_nh_small,root)
        outp_df = pd.concat([outp_df,add_data],axis=1)
    
    list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf_PF_ph(ph,DT,constr_dict,outp_df,type_min,nw_IDs,
                     nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                     N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file)
    
    return DT, constr_dict, outp_df, ph, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv

def graph_DT_plus_pl_vars(opt,seed,root,n_DT,ph=0,F_nh_small=True,proc_NA=False,mnp=0.02,
                   is_dir=True,lay=True,p_vr=None,to_file=False,extra_DTfeat=None):
    
    if opt < 3: # single optima, ph is not used
    
        # tSNE path
        if not proc_NA:
            name= root + "clusters/tSNE_"+str(opt)+"opt_"+str(seed)+ "seed.pickle"
        else:
            name = root + "clusters/tSNE_"+str(opt)+"opt_NA_"+str(seed)+ "seed.pickle"
        
        # DTs path    
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_"  + str(mnp) + "mnp/distinct_DT.pickle"
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_NA_"  + str(mnp) + "mnp/distinct_DT.pickle" 
        
        # open tSNE
        with open(name, "rb") as file:
            [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file) 
        
        # open DTs data
        with open(path, "rb") as file:
            [distinct_DTs, counter, depth, 
             dict_repr_DT,labels,l_v_eps,
             features_to_select,HP,nM] = pickle.load(file) 
        
        # open required DT and its constraints
        n_key = list(dict_repr_DT.keys())[n_DT]
        DT = dict_repr_DT[n_key]
        constr_dict = extract_constraints_from_DT(DT)
        
        # add extra variables
        # open the master file
        with open(root + "all_var_all_optima_w_pl.pickle", "rb") as file:
            all_data = pickle.load(file)
        
        outp_df.insert(outp_df.shape[1],'F',nw_mv['F'])
        outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
        outp_df.insert(outp_df.shape[1],'ph',nw_mv['ph'])    
        outp_df.insert(outp_df.shape[1],'F*nh',all_data.loc[all_data['type_min']==opt,'F*nh'])
        outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
        outp_df.insert(outp_df.shape[1],'pbb_non_h',all_data.loc[all_data['type_min']==opt,'pbb_non_h'])
        outp_df.insert(outp_df.shape[1],'pval_ddd',all_data.loc[all_data['type_min']==opt,'pval_ddd'])
        outp_df.insert(outp_df.shape[1],'pval_dddi',all_data.loc[all_data['type_min']==opt,'pval_dddi'])
        outp_df.insert(outp_df.shape[1],'alpha_ddd',all_data.loc[all_data['type_min']==opt,'alpha_ddd'])
        outp_df.insert(outp_df.shape[1],'alpha_dddi',all_data.loc[all_data['type_min']==opt,'alpha_dddi'])
        #'pval_ddd', 'alpha_ddd', 'xmin_ddd', 'pval_dddi', 'alpha_dddi','xmin_dddi'
        
        del all_data
        list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf(opt,DT,constr_dict,outp_df,type_min,nw_IDs,
                         nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                         N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file)
        
        return DT, constr_dict, outp_df, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv    
    
    else: # multiobjective optima
        # open tSNE
        name= root + "clusters/tSNE_PF_"+str(seed)+ "seed_"+str(ph)+"ph_"+str(F_nh_small)+"Fnhsmall.pickle"
        
        with open(name, "rb") as file:
            [tsne, clust, fs] = pickle.load(file)      
        
        # open DTs    
        path = root + "clusters/DTs_PF_"+str(seed)+"seed_" + str(ph) + "ph_"  + str(F_nh_small) + "_F_nh_small/distinct_DT.pickle"
        
        with open(path, "rb") as file:
            [distinct_DTs, counter, depth, 
             dict_repr_DT,labels,l_v_eps,
             features_to_select,HP,nM] = pickle.load(file) 
        
        # open required DT and its constraints
        n_key = list(dict_repr_DT.keys())[n_DT]
        DT = dict_repr_DT[n_key]
        constr_dict = extract_constraints_from_DT(DT)
        
        # add extra variables
        # open the master file
        with open(root + "all_var_all_optima_w_pl.pickle", "rb") as file:
            all_data = pickle.load(file)
        
        features_to_select = fs + ['pbb_non_h']

        #outp_df, nw_mv, nw_IDs = filter_df(features_to_select,ph,F_nh_small,root)
        
        # filter PF data, the ph, F, nh region
        all_data_3 = all_data[all_data['type_min']==3]
        
        all_data_3 = all_data_3[all_data_3['ph']==ph]
        
        del all_data
        
        if F_nh_small:
            all_data_3 = all_data_3[all_data_3['F*nh']<=12]
        else:
            all_data_3 = all_data_3[all_data_3['F*nh']>12]
            
        # recover required features
        nw_IDs = all_data_3['ID']
        nw_mv = all_data_3[['F', 'nh', 'ph']]
        type_min = [3]*all_data_3.shape[0]
        
        # select final features
        features_to_select = features_to_select + ['F','ph','F*nh','F*nh*ph','pval_ddd','pval_dddi']
        #'pval_ddd', 'alpha_ddd', 'xmin_ddd', 'pval_dddi', 'alpha_dddi','xmin_dddi'
        all_data_3 = all_data_3[features_to_select]
        outp_df=all_data_3
        outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
        
        if not extra_DTfeat is None:
            add_data, nw_mv, nw_IDs = filter_df(extra_DTfeat,ph,F_nh_small,root)
            outp_df = pd.concat([outp_df,add_data],axis=1)
        
        list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf_PF_ph(ph,DT,constr_dict,outp_df,type_min,nw_IDs,
                         nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                         N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file)
           
        return DT, constr_dict, outp_df, ph, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv

def graph_DT_plus_pl_ddd(opt,seed,root,n_DT,ph=0,F_nh_small=True,proc_NA=False,mnp=0.02,
                   is_dir=True,lay=True,p_vr=None,to_file=False,extra_DTfeat=None):
    
    if opt < 3: # single optima, ph is not used
    
        # tSNE path
        if not proc_NA:
            name= root + "clusters/tSNE_"+str(opt)+"opt_"+str(seed)+ "seed.pickle"
        else:
            name = root + "clusters/tSNE_"+str(opt)+"opt_NA_"+str(seed)+ "seed.pickle"
        
        # DTs path    
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_"  + str(mnp) + "mnp/distinct_DT.pickle"
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_NA_"  + str(mnp) + "mnp/distinct_DT.pickle" 
        
        # open tSNE
        with open(name, "rb") as file:
            [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file) 
        
        # open DTs data
        with open(path, "rb") as file:
            [distinct_DTs, counter, depth, 
             dict_repr_DT,labels,l_v_eps,
             features_to_select,HP,nM] = pickle.load(file) 
        
        # open required DT and its constraints
        n_key = list(dict_repr_DT.keys())[n_DT]
        DT = dict_repr_DT[n_key]
        constr_dict = extract_constraints_from_DT(DT)
        
        # add extra variables
        # open the master file
        with open(root + "all_var_all_optima.pickle", "rb") as file:
            all_data = pickle.load(file)
            
        # open file with pl fits info
        df=pd.read_csv(root +'all_data_pl_fit_ddd.csv')
        df.reset_index(drop=True, inplace=True)
        
        # paste these variables
        all_data.insert(all_data.shape[1],'pval',df['V1'])
        all_data.insert(all_data.shape[1],'xmin',df['V2'])
        all_data.insert(all_data.shape[1],'alpha',df['V3'])
        del df
        
        # select final variables
        outp_df.insert(outp_df.shape[1],'F',nw_mv['F'])
        outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
        outp_df.insert(outp_df.shape[1],'ph',nw_mv['ph'])    
        outp_df.insert(outp_df.shape[1],'F*nh',all_data.loc[all_data['type_min']==opt,'F*nh'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
        outp_df.insert(outp_df.shape[1],'pbb_non_h',all_data.loc[all_data['type_min']==opt,'pbb_non_h'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'pval',all_data.loc[all_data['type_min']==opt,'pval'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'xmin',all_data.loc[all_data['type_min']==opt,'xmin'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'alpha',all_data.loc[all_data['type_min']==opt,'alpha'].reset_index(drop=True))
        
        del all_data
        list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf(opt,DT,constr_dict,outp_df,type_min,nw_IDs,
                         nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                         N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file)
        
        return DT, constr_dict, outp_df, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv    
    
    else: # multiobjective optima
        # open tSNE
        name= root + "clusters/tSNE_PF_"+str(seed)+ "seed_"+str(ph)+"ph_"+str(F_nh_small)+"Fnhsmall.pickle"
        
        with open(name, "rb") as file:
            [tsne, clust, fs] = pickle.load(file)      
        
        # open DTs    
        path = root + "clusters/DTs_PF_"+str(seed)+"seed_" + str(ph) + "ph_"  + str(F_nh_small) + "_F_nh_small/distinct_DT.pickle"
        
        with open(path, "rb") as file:
            [distinct_DTs, counter, depth, 
             dict_repr_DT,labels,l_v_eps,
             features_to_select,HP,nM] = pickle.load(file) 
        
        # open required DT and its constraints
        n_key = list(dict_repr_DT.keys())[n_DT]
        DT = dict_repr_DT[n_key]
        constr_dict = extract_constraints_from_DT(DT)
        
        # add extra variables
        # open the master file
        with open(root + "all_var_all_optima.pickle", "rb") as file:
            all_data = pickle.load(file)
        
        features_to_select = fs + ['pbb_non_h']

        #outp_df, nw_mv, nw_IDs = filter_df(features_to_select,ph,F_nh_small,root)
        
        #open file with pl fits info
        df=pd.read_csv(root +'all_data_pl_fit_ddd.csv')
        df.reset_index(drop=True, inplace=True)
        all_data.insert(all_data.shape[1],'pval',df['V1'])
        all_data.insert(all_data.shape[1],'xmin',df['V2'])
        all_data.insert(all_data.shape[1],'alpha',df['V3'])
        del df
        
        # filter PF data, the ph, F, nh region
        all_data_3 = all_data[all_data['type_min']==3]
        
        all_data_3 = all_data_3[all_data_3['ph']==ph]
        
        del all_data
        
        if F_nh_small:
            all_data_3 = all_data_3[all_data_3['F*nh']<=12]
        else:
            all_data_3 = all_data_3[all_data_3['F*nh']>12]
            
        # recover required features
        nw_IDs = all_data_3['ID']
        nw_mv = all_data_3[['F', 'nh', 'ph']]
        type_min = [3]*all_data_3.shape[0]
        
        # select final features
        features_to_select = features_to_select + ['F','ph','F*nh','pval','xmin','alpha']
        #'pval_ddd', 'alpha_ddd', 'xmin_ddd', 'pval_dddi', 'alpha_dddi','xmin_dddi'
        all_data_3 = all_data_3[features_to_select]
        outp_df=all_data_3
        outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
        outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
        
        if not extra_DTfeat is None:
            add_data, nw_mv, nw_IDs = filter_df(extra_DTfeat,ph,F_nh_small,root)
            outp_df = pd.concat([outp_df,add_data],axis=1)
        
        list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf_PF_ph(ph,DT,constr_dict,outp_df,type_min,nw_IDs,
                         nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                         N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file)
           
        return DT, constr_dict, outp_df, ph, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv

# since uses apply_constraints() 
# table requires 'F' and 'nh' fields if used to build the DT
def graph_every_leaf_with_pos_fit(ph,DT,constr_dict,table,type_min,nw_IDs,
                     nw_mv,direct=True,lay_var=True,inter_I=False,
                     N_dir=12,n_bits=12*11,print_var=None,to_file=False,
                     n_sample=None,seed=10,r_n=2):
    
    # get keys of leaf nodes
    split_nodes, leaf_nodes = DT._get_node_ids()
    idbr_leaf_nodes = dict(zip(leaf_nodes.values(), leaf_nodes.keys()))
    
    masks = []
    list_m_r_inds = []
    
    if ph >= 1:
        #table_without_metavars = table.drop(['F','nh','ph','F*nh'],axis=1)
        #table_without_metavars = table.drop(['ph','F*nh'],axis=1)
        #table_without_metavars = table.drop(['ph','F*nh','F','nh','F*nh*ph','pbb_non_h'],axis=1)
        table_without_metavars = table
    else:
        #table_without_metavars = table.drop(['F','nh','F*nh'],axis=1)
        #table_without_metavars = table.drop(['F*nh'],axis=1)
        table_without_metavars = table
    
    for k in idbr_leaf_nodes:
        
        # extract constraints of node ID k, and filtering those with positive fit 
        mask = apply_constraints(constr_dict,table,k,inter_I)
        masks.append(mask)
        mask_filt = np.logical_and(mask,
                              table[table.columns[print_var]]>0.5)
        
        if np.sum(mask_filt)==0:
            continue
        
        if ph < 1:
        
            # extract respective mask for each type of min
            resp_mask = [(mask & [type_min[n]==i for n in range(len(type_min))]) for i in range(1,4)]
        
            for i in range(3):
                # there are nw of this opt type in the leaf => draw the most representative
                print("number of samples in leaf: "+str(sum(resp_mask[i])))
                if sum(resp_mask[i])>0: 
                
                    print(("RV" if i==0 else ("WEF" if i==1 else "PF"))+" min")
                    if to_file:
                        ind_mr = draw_the_most_representative(table_without_metavars[resp_mask[i]],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,k)
                    else:
                        ind_mr = draw_the_most_representative(table_without_metavars[resp_mask[i]],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,None)
                    list_m_r_inds.append(ind_mr)
            print()
            
        else:
           
            # extract respective mask for each type of min
            #resp_mask = (mask & [type_min[n]==ph for n in range(len(type_min))]) 
            
            print("Leaf "+str(k))
            print(("RV" if ph==1 else ("WEF" if ph==2 else "PF"))+" min")
            print("number of samples in leaf: "+str(sum(mask)))
            if n_sample is None:
                if to_file:
                    ind_mr = draw_the_most_representative(table_without_metavars[mask_filt],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,k)
                else:
                    ind_mr = draw_the_most_representative(table_without_metavars[mask_filt],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,None)
                list_m_r_inds.append(ind_mr)    
            else:
                selected_df = table[mask_filt].sample(n=n_sample,random_state=seed)
                selected_rows =selected_df.index    
                
                for i in selected_rows:
                        print(i)
                        if not to_file:
                            draw_netw(nw_IDs.loc[i],n_bits,direct,list(range(int(table['nh'].loc[i]))),N_dir,
                                  test_layout=lay_var,receive_int=True)
                        else: 
                            draw_netw(nw_IDs.loc[i],n_bits,direct,list(range(int(table['nh'].loc[i]))),N_dir,
                                  test_layout=lay_var,receive_int=True,file='leaf_'+str(k)+'_row_'+str(i)+'.pdf')
                        print(str(table.loc[i]))
                        print()
                        
                list_m_r_inds.append(selected_rows)
            print()
            
    if not print_var is None:
        
        print_var_stats(print_var,table,idbr_leaf_nodes,masks,r_n)
            
    return list_m_r_inds, idbr_leaf_nodes, masks

def graph_DT_plus_pl_degDistr(opt,seed,root,n_DT,ph=0,F_nh_small=True,proc_NA=False,mnp=0.02,
                   is_dir=True,lay=True,p_vr=None,to_file=False,extra_DTfeat=None,n_s=None,r_n=2):
    
    if opt < 3: # single optima, ph is not used
    
        # tSNE path
        if not proc_NA:
            name= root + "clusters/tSNE_"+str(opt)+"opt_"+str(seed)+ "seed.pickle"
        else:
            name = root + "clusters/tSNE_"+str(opt)+"opt_NA_"+str(seed)+ "seed.pickle"
        
        # DTs path    
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_"  + str(mnp) + "mnp/distinct_DT.pickle"
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_NA_"  + str(mnp) + "mnp/distinct_DT.pickle" 
        
        # open tSNE
        with open(name, "rb") as file:
            [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file) 
        
        # open DTs data
        with open(path, "rb") as file:
            [distinct_DTs, counter, depth, 
             dict_repr_DT,labels,l_v_eps,
             features_to_select,HP,nM] = pickle.load(file) 
        
        # open required DT and its constraints
        n_key = list(dict_repr_DT.keys())[n_DT]
        DT = dict_repr_DT[n_key]
        constr_dict = extract_constraints_from_DT(DT)
        
        # add extra variables
        # open the master file
        #with open(root + "all_var_all_optima.pickle", "rb") as file:
        #    all_data = pickle.load(file)
            
        # open file with pl fits info
        with open(root + "all_var_all_optima_w_pl_degDistr.pickle", "rb") as file:
            all_data = pickle.load(file)
        
        # open file with pl fits info
        df=pd.read_csv(root +'all_data_pl_fit_degreeDistr.csv')
        df.reset_index(drop=True, inplace=True)
        
        # paste these variables
        all_data.insert(all_data.shape[1],'pval_out',df['V1'])
        all_data.insert(all_data.shape[1],'xmin_out',df['V2'])
        all_data.insert(all_data.shape[1],'alpha_out',df['V3'])
        all_data.insert(all_data.shape[1],'pval_in',df['V4'])
        all_data.insert(all_data.shape[1],'xmin_in',df['V5'])
        all_data.insert(all_data.shape[1],'alpha_in',df['V6'])
        del df
        
        # select final variables
        outp_df.insert(outp_df.shape[1],'F',nw_mv['F'])
        outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
        outp_df.insert(outp_df.shape[1],'ph',nw_mv['ph'])    
        outp_df.insert(outp_df.shape[1],'F*nh',all_data.loc[all_data['type_min']==opt,'F*nh'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
        outp_df.insert(outp_df.shape[1],'pbb_non_h',all_data.loc[all_data['type_min']==opt,'pbb_non_h'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'gamma_hill_out',all_data.loc[all_data['type_min']==opt,'gamma_hill_out'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'gamma_mom_out',all_data.loc[all_data['type_min']==opt,'gamma_mom_out'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'gamma_kern_out',all_data.loc[all_data['type_min']==opt,'gamma_kern_out'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'gamma_hill_in',all_data.loc[all_data['type_min']==opt,'gamma_hill_in'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'gamma_mom_in',all_data.loc[all_data['type_min']==opt,'gamma_mom_in'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'gamma_kern_in',all_data.loc[all_data['type_min']==opt,'gamma_kern_in'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'pval_out',all_data.loc[all_data['type_min']==opt,'pval_out'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'xmin_out',all_data.loc[all_data['type_min']==opt,'xmin_out'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'alpha_out',all_data.loc[all_data['type_min']==opt,'alpha_out'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'pval_in',all_data.loc[all_data['type_min']==opt,'pval_in'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'xmin_in',all_data.loc[all_data['type_min']==opt,'xmin_in'].reset_index(drop=True))
        outp_df.insert(outp_df.shape[1],'alpha_in',all_data.loc[all_data['type_min']==opt,'alpha_in'].reset_index(drop=True))
        
        del all_data
        
        list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf_with_pos_fit(opt,DT,constr_dict,outp_df,type_min,nw_IDs,
                         nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                         N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file,n_sample=n_s,r_n=r_n)
        
        return DT, constr_dict, outp_df, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv    
    
    else: # multiobjective optima
        # open tSNE
        name= root + "clusters/tSNE_PF_"+str(seed)+ "seed_"+str(ph)+"ph_"+str(F_nh_small)+"Fnhsmall.pickle"
        
        with open(name, "rb") as file:
            [tsne, clust, fs] = pickle.load(file)      
        
        # open DTs    
        path = root + "clusters/DTs_PF_"+str(seed)+"seed_" + str(ph) + "ph_"  + str(F_nh_small) + "_F_nh_small/distinct_DT.pickle"
        
        with open(path, "rb") as file:
            [distinct_DTs, counter, depth, 
             dict_repr_DT,labels,l_v_eps,
             features_to_select,HP,nM] = pickle.load(file) 
        
        # open required DT and its constraints
        n_key = list(dict_repr_DT.keys())[n_DT]
        DT = dict_repr_DT[n_key]
        constr_dict = extract_constraints_from_DT(DT)
        
        # add extra variables
        # open the master file
        with open(root + "all_var_all_optima_w_pl_degDistr.pickle", "rb") as file:
            all_data = pickle.load(file)
        
        features_to_select = fs + ['pbb_non_h']

        # open file with pl fits info
        df=pd.read_csv(root +'all_data_pl_fit_degreeDistr.csv')
        df.reset_index(drop=True, inplace=True)
        
        # paste these variables
        all_data.insert(all_data.shape[1],'pval_out',df['V1'])
        all_data.insert(all_data.shape[1],'xmin_out',df['V2'])
        all_data.insert(all_data.shape[1],'alpha_out',df['V3'])
        all_data.insert(all_data.shape[1],'pval_in',df['V4'])
        all_data.insert(all_data.shape[1],'xmin_in',df['V5'])
        all_data.insert(all_data.shape[1],'alpha_in',df['V6'])
        del df
        
        # filter PF data, the ph, F, nh region
        all_data_3 = all_data[all_data['type_min']==3]
        
        all_data_3 = all_data_3[all_data_3['ph']==ph]
        
        del all_data
        
        if F_nh_small:
            all_data_3 = all_data_3[all_data_3['F*nh']<=12]
        else:
            all_data_3 = all_data_3[all_data_3['F*nh']>12]
            
        # recover required features
        nw_IDs = all_data_3['ID']
        nw_mv = all_data_3[['F', 'nh', 'ph']]
        type_min = [3]*all_data_3.shape[0]
        
        # select final features
        features_to_select = features_to_select + ['F','ph','F*nh','gamma_hill_out','gamma_mom_out','gamma_kern_out',
                                                   'gamma_hill_in','gamma_mom_in','gamma_kern_in','pval_out','xmin_out',
                                                   'alpha_out','pval_in','xmin_in','alpha_in']
        #'pval_ddd', 'alpha_ddd', 'xmin_ddd', 'pval_dddi', 'alpha_dddi','xmin_dddi'
        all_data_3 = all_data_3[features_to_select]
        outp_df=all_data_3
        outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
        outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
        
        if not extra_DTfeat is None:
            add_data, nw_mv, nw_IDs = filter_df(extra_DTfeat,ph,F_nh_small,root)
            outp_df = pd.concat([outp_df,add_data],axis=1)
        
        list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf_with_pos_fit(3,DT,constr_dict,outp_df,type_min,nw_IDs,
                         nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                         N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file,n_sample=n_s,r_n=r_n)
           
        return DT, constr_dict, outp_df, ph, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv    

# since uses apply_constraints() 
# table requires 'F' and 'nh' fields if used to build the DT
def graph_every_leaf_PF_ph(ph,DT,constr_dict,table,type_min,nw_IDs,
                     nw_mv,direct=True,lay_var=True,inter_I=False,
                     N_dir=12,n_bits=12*11,print_var=None,to_file=False):
    
    # get keys of leaf nodes
    split_nodes, leaf_nodes = DT._get_node_ids()
    idbr_leaf_nodes = dict(zip(leaf_nodes.values(), leaf_nodes.keys()))
    
    masks = []
    list_m_r_inds = []
    
    #table_without_metavars = table.drop(['F','nh','ph','F*nh','F*nh*ph'],axis=1)
    table_without_metavars = table
    
    for k in idbr_leaf_nodes:
        
        print("Leaf "+str(k))
        
        # extract constraints of node ID k
        mask = apply_constraints(constr_dict,table,k,inter_I)
        masks.append(mask)
        
        print("number of samples in leaf: "+str(sum(mask)))
        if to_file:
            ind_mr = draw_the_most_representative(table_without_metavars[mask],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,k)
        else:
            ind_mr = draw_the_most_representative(table_without_metavars[mask],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,None)
        list_m_r_inds.append(ind_mr)    
        print()
            
    if not print_var is None:
        
        print_var_stats(print_var,table,idbr_leaf_nodes,masks)
            
    return list_m_r_inds, idbr_leaf_nodes, masks
        
# tSNE with HP tuned to work with the large dataframe
# built from the 3 type of optima
def tSNE_and_OPTICS_PF_ph(seed,ph,F_nh_small=True,
                                  fs=['out_deg_var', 'in_deg_var', 'n_conn_comp', 'recipr_pairs',
                                         'mean_local_clust', 'nw_mod', 'mean_out_deg', 'h_in_deg', 'RV_cost',
                                         'WEF_cost'],
                                  path="/home/fplana/evol_datafr_analysis/",dim=2,ite=1000,prop_min_sampl=.03):
    
    """
    outp_df_1, type_min_1, nw_IDs_1, nw_mv_1 = preprocess_datafr_1_opt(opt=1,s=seed,root=path,n_sample=n_s,proc_NA=True)
    outp_df_2, type_min_2, nw_IDs_2, nw_mv_2 = preprocess_datafr_1_opt(opt=2,s=seed,root=path,n_sample=n_s,proc_NA=True)
    outp_df_3, type_min_3, nw_IDs_3, nw_mv_3 = preprocess_datafr_1_opt(opt=3,s=seed,root=path,n_sample=n_s,proc_NA=True)
    
    outp_df_2 = outp_df_2.drop(['out_assort','in_assort'], axis=1)
    
    outp_df = pd.concat([outp_df_1,outp_df_2,outp_df_3])
    type_min = type_min_1 + type_min_2 + type_min_3
    nw_IDs = pd.concat([nw_IDs_1,nw_IDs_2,nw_IDs_3])
    nw_mv = pd.concat([nw_mv_1,nw_mv_2,nw_mv_3])
    
    outp_df = outp_df.reset_index(drop=True)
    nw_IDs = nw_IDs.reset_index(drop=True)
    nw_mv = nw_mv.reset_index(drop=True)"""
    
    all_data_3, nw_mvs, nw_IDs = filter_df(fs,ph,F_nh_small,path)
    
    datafr = compute_generalized_sign(all_data_3)
    n_samples = datafr.shape[0]
    
    print("data was standardized")
    
    perp = n_samples * 0.01
    EE = 4
    LR= max(200,n_samples/EE)
    
    patch_sklearn("TSNE")
    
    t0 = time.time()
    tsne = TSNE(n_components=dim, init='pca',
                     random_state=seed, perplexity=perp, 
                     n_iter=ite,early_exaggeration=EE,
                     learning_rate=LR)
    tsne.fit(datafr)
    t1 = time.time()
    print("TSNE computed in %.2g sec and %d iters" % ( t1 - t0,tsne.n_iter_))    
    
    clust = OPTICS(min_samples=prop_min_sampl, 
                   cluster_method="dbscan")
    
    X=tsne.embedding_
    t0=time.time()
    clust.fit(X)
    print("OPTICS time fitting map data: "+str(time.time()-t0)+" secs.")
    
    with open(path + "clusters/tSNE_PF_"+str(seed)+ "seed_"
              +str(ph)+"ph_"+str(F_nh_small)+"Fnhsmall.pickle", "wb") as wr:
        wr.write(pickle.dumps([tsne, clust, fs]))
        
# visualize output from tSNE_and_OPTICS_one_opt()
# or tSNE_and_OPTICS_one_ph()
def visualize_TSNE_PF_ph(ph,seed,F_nh_small=True,
                         path='/home/fplana/evol_datafr_analysis/',n_ite=50,wait_th=10,verb=False,mnp=0.02,rst=1.5):
            
    name=path + "clusters/tSNE_PF_"+str(seed)+ "seed_"+str(ph)+"ph_"+str(F_nh_small)+"Fnhsmall.pickle"
    
    with open(name, "rb") as file:
        [tsne, clust, fs] = pickle.load(file) 
    
    X=tsne.embedding_
    
    stb_eps = find_epsilon(clust,verbose=verb,ite=n_ite,waiting_thresh=wait_th) 
    
    outp_df, nw_mvs, nw_IDs = filter_df(fs,ph,F_nh_small,path)
    
    l_v_eps, l_std = find_epsilon_III(clust,outp_df,verbose=verb,max_nois_prop=mnp,ratio_std_thr=rst)
    labels_I = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=l_v_eps)
    
    labels_II = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=stb_eps)

    draw_optics_II(labels_I,labels_II,X,stb_eps,l_v_eps,clust) 
    
    std_sum = obtain_std_sum_labels(X,labels_I)
    
    # compute variance of each cluster, standardize it first
    uniq_labels = np.unique(labels_I)
    rad_fun='quad'
    X = gsspp.gen_ss_pp( outp_df, fun = rad_fun, center = 'kstepLTS')
    datafr = pd.DataFrame(X, columns = outp_df.columns, index=outp_df.index)
    ind_each_cluster = [[i for i in range(outp_df.shape[0]) if labels_I[i]==l] for l in uniq_labels]
    max_sd_per_cluster = [np.amax(np.std(datafr.iloc[ind_each_cluster[l]])) for l in uniq_labels]
    which_feat_max_sd = [np.argmax(np.std(datafr.iloc[ind_each_cluster[l]])) for l in uniq_labels]
    
    print()
    print("Avg sum feature sd: "+str(std_sum))
    print()
    print("Max (normalized) sd per cluster")
    for i in range(uniq_labels.shape[0]):
        print("Label "+str(uniq_labels[i])+": "
              +str(max_sd_per_cluster[i]) + ", feature: "
              + outp_df.columns[which_feat_max_sd[i]]
              + ", cluster size: " + str(len(ind_each_cluster[i])))

    return stb_eps, l_v_eps, outp_df, nw_IDs, clust, nw_mvs, labels_I, labels_II, X
 
        
# opt=1 means "RV", opt=2 means "WEF", opt=3 means "PF"
def preprocess_datafr_1_opt(opt,s=10,N=12,FS=None,root="/home/fplana/evol_datafr_analysis/",
                          direct=True,n_sample=None,phs=None,proc_NA=False):
    
    if phs is None:
        phs = [0.02,0.08,0.15,0.3,0.6]  
    
    parameters_to_explore = []
    # (ph,F,h_ind)
    # F*nh >=12
    parameters_to_explore.append((12,[0]))
    parameters_to_explore.append((4,[0,1,2]))
    parameters_to_explore.append((3,[0,1,2,3,4]))
    parameters_to_explore.append((3,[0,1,2,3,4,5,6]))
    parameters_to_explore.append((4,[0,1,2,3,4,5,6,7,8]))
    
    # F*nh < 12
    parameters_to_explore.append((4,[0]))
    parameters_to_explore.append((4,[0,1]))
    parameters_to_explore.append((5,[0,1]))
    parameters_to_explore.append((3,[0,1,2]))
    parameters_to_explore.append((2,[0,1,2,3,4]))
    
    if FS is None:
        
        FS = ['out_deg_var', 'in_deg_var', 'n_conn_comp', 
              'recipr_pairs','mean_local_clust','out_assort',
              'in_assort','nw_mod','mean_out_deg','h_in_deg',
              'RV_cost', 'WEF_cost']    
    """    
    else:
        
        FS = ['out_deg_var', 'in_deg_var', 'n_conn_comp', 
              'recipr_pairs','glob_clust','out_assort',
              'in_assort','nw_mod','mean_out_deg','h_in_deg',
              'RV_cost', 'WEF_cost','mean_loc_clust',
              'non_h_in_deg_var','rep_at_least_non_h']"""
    
    df = pd.DataFrame()
    
    # iterate over all ph's, nh's, F's
    for x in product(phs,parameters_to_explore):
        
        ph = x[0]
        F = x[1][0]
        nh = x[1][1]
        
        #if opt==1 and ph < 0.1 and len(nh)<=6:
        #    continue
        
        if direct:
            name = "d_" + ("RV" if opt==1 else ("WEF" if opt ==2 else "PF"))+"_" + str(ph) + "ph_" +  str(F) + "F_"  +str(len(nh))+ "nh.pickle"
        else:
            name = "und_" + ("RV" if opt==1 else ("WEF" if opt ==2 else "PF"))+"_" + str(ph) + "ph_" +  str(F) + "F_"  +str(len(nh))+ "nh.pickle"
        
        df_curr = pd.read_pickle(root + "dataframes/df_" + name)
    
        # add metavars
        df_curr.insert(df_curr.shape[1],'F',[F]*df_curr.shape[0])
        df_curr.insert(df_curr.shape[1],'nh',[nh]*df_curr.shape[0])
        df_curr.insert(df_curr.shape[1],'ph',[ph]*df_curr.shape[0])
        
        # obtain a sample
        if not n_sample is None:
            
            #if df_curr.shape[0] > n_sample:    
            #    df_curr = df_curr.sample(n=n_sample, replace=False, random_state=s)
            df_curr = df_curr.sample(n=n_sample, replace=True, random_state=s)
        
        df = pd.concat([df,df_curr])
    
    # usual processsing, removing NA rows!
    if not proc_NA:
    
        # process dataframes
        df = df[FS + ['nw_ID','F','nh','ph']].dropna().reset_index(drop=True)
    
        # extract nw ID
        nw_IDs = df['nw_ID']
        df = df.drop(['nw_ID'], axis=1)
    
        # extract metavariables
        nw_mv = df[['F','nh','ph']]    
        outp_df = df.drop(['F','nh','ph'], axis=1) 
    
        # produce the identifier for the type of optimum
        type_min = [opt]*outp_df.shape[0] 
   
        return outp_df, type_min, nw_IDs, nw_mv
    
    # process NA, removing attributes with NA's
    else:
        
        # process dataframes
        df = df[FS + ['nw_ID','F','nh','ph']].reset_index(drop=True)
        
        # detect NA, remove arguments having them,
        # solution borrowed from https://stackoverflow.com/questions/33641231/retrieve-indices-of-nan-values-in-a-pandas-dataframe
        #x,y = sp.coo_matrix(df.isnull()).nonzero()
        #ind_x = np.unique(x)
        #ind_y= np.unique(y)
        
        ind_y = np.where(df.isnull().sum()>0)[0]
        df = df.drop(df.columns[ind_y],axis=1)
    
        # extract nw ID
        nw_IDs = df['nw_ID']
        df = df.drop(['nw_ID'], axis=1)
    
        # extract metavariables
        nw_mv = df[['F','nh','ph']]    
        outp_df = df.drop(['F','nh','ph'], axis=1) 
    
        # produce the identifier for the type of optimum
        type_min = [opt]*outp_df.shape[0] 
   
        return outp_df, type_min, nw_IDs, nw_mv

# receives vector of reachabilities, output of OPTICS()
def find_epsilon(clust,ite=50,waiting_thresh=10,verbose=False):
    
    reach = deepcopy(clust.reachability_)
    prev_max=0
    #size_array = np.shape(clust.reachability_)[0]
    stb_ct = 0
    eps_stb=-1
    prev_n_labels = -1
    n_labels_this_eps = -1
    first_app_this_eps = -1
    
    for i in range(ite):
    
        curr_max = np.amax(reach)
        pos_curr_max = np.where(reach == curr_max)[0]
        
        if i > 0:
            
            curr_eps = (curr_max + prev_max)/2
            labels_ = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=curr_eps)
            prop_noise = np.sum(labels_==-1)
            n_labels_this_eps=np.unique(labels_).shape[0] - 1
            if verbose:
                print("n_labels: "+str(n_labels_this_eps) + ", curr max: " + str(curr_max) + ", noise: "+str(prop_noise))
        
            if prev_n_labels == n_labels_this_eps:
                stb_ct += 1
            else:
                stb_ct = 0  
                first_app_this_eps=curr_eps
            if stb_ct >= waiting_thresh:
                return first_app_this_eps
                
        prev_max = curr_max
        prev_n_labels = n_labels_this_eps
        reach = np.delete(reach,pos_curr_max)
        
    return eps_stb    

def obtain_std_sum_labels(df,labels):
    the_labels=np.unique(labels)
    return sum([np.sum(np.std(df[labels==lab])) for lab in the_labels])/the_labels.shape[0]

# receives vector of reachabilities, output of OPTICS()
def find_epsilon_III(clust,df,verbose=False,max_nois_prop=0.02,ratio_std_thr=1.5):
    
    reach = deepcopy(clust.reachability_)
    prev_max=0
    ite = ceil(df.shape[0] * max_nois_prop)
    curr_min_eps = np.inf
    curr_min_std = np.inf
    
    for i in range(ite):
    
        curr_max = np.amax(reach)
        pos_curr_max = np.where(reach == curr_max)[0]
        
        if i > 0:
        
            curr_eps = (curr_max + prev_max)/2
            labels_ = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=curr_eps)
        
            curr_std = obtain_std_sum_labels(df,labels_)
            
            std_noise = np.sum(np.std(df[labels_==-1]))
            the_labels=np.unique(labels_)[1:]
            if the_labels.shape[0]!=0:
                std_non_noise = sum([np.sum(np.std(df[labels_==lab])) for lab in the_labels])/the_labels.shape[0]
            
                # update minimum
                if curr_min_std > curr_std and std_noise/std_non_noise < ratio_std_thr:
                
                    curr_min_std = curr_std
                    curr_min_eps = curr_eps
                
                    if verbose:
                        prop_noise = np.sum(labels_==-1)
                        n_labels_this_eps=np.unique(labels_).shape[0] - 1
                        print("n_labels: "+str(n_labels_this_eps) + 
                              ", curr max: " + str(curr_max) + 
                              ", noise: "+str(prop_noise) + 
                              ", std: " +str(curr_std))
                   
        prev_max = curr_max

        reach = np.delete(reach,pos_curr_max)
        
    return curr_min_eps, curr_min_std 

# visualize output from tSNE_and_OPTICS_one_opt()
# or tSNE_and_OPTICS_one_ph()
def visualize_TSNE(ph,seed,path='/home/fplana/evol_datafr_analysis/',n_ite=50,wait_th=10,verb=False,mnp=0.02,rst=1.5,proc_NA=False):
            
    # open tSNE data
    # 1 opt
    if ph == floor(ph):
        if ph <4:
            if not proc_NA:
                name= path + "clusters/tSNE_"+str(ph)+"opt_"+str(seed)+ "seed.pickle"
            else:
                name = path + "clusters/tSNE_"+str(ph)+"opt_NA_"+str(seed)+ "seed.pickle"
        else:
            
            name = path + "clusters/tSNE_big_"+str(seed)+ "seed.pickle"
                     
    else: # 1 ph
    
        if not proc_NA:
            name= path + "clusters/tSNE_"+str(ph)+"opt_"+str(seed)+ "seed.pickle"
        else:
            name = path + "clusters/tSNE_"+str(ph)+"opt_NA_"+str(seed)+ "seed.pickle"
    
    with open(name, "rb") as file:
        [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file) 
    
    X=tsne.embedding_
    
    stb_eps = find_epsilon(clust,verbose=verb,ite=n_ite,waiting_thresh=wait_th) 
    
    
    l_v_eps, l_std = find_epsilon_III(clust,outp_df,verbose=verb,max_nois_prop=mnp,ratio_std_thr=rst)
    labels_I = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=l_v_eps)
    
    labels_II = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=stb_eps)

    draw_optics_II(labels_I,labels_II,X,stb_eps,l_v_eps,clust) 
    
    std_sum = obtain_std_sum_labels(X,labels_I)
    
    # compute variance of each cluster, standardize it first
    uniq_labels = np.unique(labels_I)
    rad_fun='quad'
    X = gsspp.gen_ss_pp( outp_df, fun = rad_fun, center = 'kstepLTS')
    datafr = pd.DataFrame(X, columns = outp_df.columns, index=outp_df.index)
    ind_each_cluster = [[i for i in range(outp_df.shape[0]) if labels_I[i]==l] for l in uniq_labels]
    max_sd_per_cluster = [np.amax(np.std(datafr.iloc[ind_each_cluster[l]])) for l in uniq_labels]
    which_feat_max_sd = [np.argmax(np.std(datafr.iloc[ind_each_cluster[l]])) for l in uniq_labels]
    
    print()
    print("Avg sum feature sd: "+str(std_sum))
    print()
    print("Max (normalized) sd per cluster")
    for i in range(uniq_labels.shape[0]):
        print("Label "+str(uniq_labels[i])+": "
              +str(max_sd_per_cluster[i]) + ", feature: "
              + outp_df.columns[which_feat_max_sd[i]]
              + ", cluster size: " + str(len(ind_each_cluster[i])))

    return stb_eps, l_v_eps, outp_df, type_min, nw_IDs, clust, nw_mv, labels_I, labels_II, X

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# following method is based on https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
# Authors: Shane Grigsby <refuge@rocktalus.com>
#          Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause

def draw_optics_II(labels_I,labels_II,X,stb_eps,epsilon,clust):
    
    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    n_labels = np.unique(labels).shape[0]
    
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])

    # Reachability plot
    cmap = get_cmap(n_labels)
    colors = [cmap(i) for i in range(n_labels)]
    for klass, color in zip(range(0, n_labels), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, stb_eps, dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, epsilon, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot on tSNE map')

    # DBSCAN I
    n_labels_I = np.unique(labels_I).shape[0]
    print("Total clusters, least std eps: " + str(n_labels_I))
    #colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    cmap = get_cmap(n_labels_I)
    colors_I = [cmap(i) for i in range(n_labels_I)]
    for klass, color in zip(range(0, n_labels_I), colors_I):
        Xk = X[labels_I == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    ax2.plot(X[labels_I == -1, 0], X[labels_I == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Clustering at '+str(epsilon)+' epsilon cut (least std)\nDBSCAN')

    # DBSCAN II
    n_labels_II = np.unique(labels_II).shape[0]
    print("Total clusters, stable eps: " + str(n_labels_II))
    #colors = ['g.', 'm.', 'y.', 'c.']
    cmap = get_cmap(n_labels_II)
    colors_II = [cmap(i) for i in range(n_labels_II)]
    for klass, color in zip(range(0, n_labels_II), colors_II):
        Xk = X[labels_II == klass]
        ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    ax3.plot(X[labels_II == -1, 0], X[labels_II == -1, 1], 'k+', alpha=0.1)
    ax3.set_title('Clustering at '+str(round(stb_eps,2))+' epsilon (stable)\nDBSCAN')

    plt.tight_layout()
    plt.show()
    

def compute_pbb_eat_non_h(nw_IDs,nw_mv,n_bits,direct,N,i):
    
        nw_code = nw_IDs.iloc[i]
            
        #convert to binary, first least binary is at the right extreme
        bin_i = bin(nw_code)[2:]
        # get strings of uniform length
        bin_i = bin_i.rjust(int(n_bits),"0")
        
        # get adjacency matrix
        if direct==False:
                A = ev.fill_matrix(bin_i,N)
        else:
                A = ev.fill_D_matrix(bin_i,N)

        h_ind = nw_mv['nh'].iloc[i]
        ph = nw_mv['ph'].iloc[i]
        F = nw_mv['F'].iloc[i]
        
        temp_p = ev.compute_vector_pbb_eat(A,h_ind,ph,F,N,direct)
        
        return np.mean([temp_p[i] for i in range(N) if i not in h_ind])
    
# to produce DTs of the lusters from tSNE_and_OPTICS_one_opt()
# or tSNE_and_OPTICS_one_ph()    
def produce_PF_DTs(ph,seed,n_cores,HP,nM,root='/home/fplana/evol_datafr_analysis/',
                   inter_I=False,prop_min_sampl=.03,
                   mnp=0.02,rst=1.5,inter1st=False,proc_NA=False):  
    
    #rg = default_rng(seed)
    
    # 1 opt
    if ph == floor(ph):
        if ph < 4:
            if not proc_NA:
                name= root + "clusters/tSNE_"+str(ph)+"opt_"+str(seed)+ "seed.pickle"
            else:
                name = root + "clusters/tSNE_"+str(ph)+"opt_NA_"+str(seed)+ "seed.pickle"
        else:
            
            name = root + "clusters/tSNE_big_"+str(seed)+ "seed.pickle"
            
        
        with open(name, "rb") as file:
            [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file)      
        
        # create folder to store all outputs of this dataset
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "opt_"  + str(mnp) + "mnp" 
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "opt_NA_"  + str(mnp) + "mnp" 
            
    else: # 1 ph
    
        if not proc_NA:
            name= root + "clusters/tSNE_"+str(ph)+"opt_"+str(seed)+ "seed.pickle"
        else:
            name = root + "clusters/tSNE_"+str(ph)+"opt_NA_"+str(seed)+ "seed.pickle"
        
        with open(name, "rb") as file:
            [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file)
        
        # create folder to store all outputs of this dataset
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "ph" 
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "ph_NA" 
    
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        
    # define feature set to use
    
    if not proc_NA:
        
        features_to_select = ['out_deg_var', 'in_deg_var', 'n_conn_comp', 
              'recipr_pairs','mean_local_clust','out_assort',
              'in_assort','nw_mod','mean_out_deg','h_in_deg',
              'RV_cost', 'WEF_cost']    
        
    else:
            
        features_to_select = list(outp_df.columns)
    
    # find least-feature-variance eps
    l_v_eps, l_std = find_epsilon_III(clust,outp_df,max_nois_prop=mnp,ratio_std_thr=rst)

    labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=l_v_eps)
    
    # it is not necessary to preprocess features
    # since we want to compute DTs on original feat scale
    # not necessary to drop na, already done when tSNE was computed
    # and it may produce differences with labels vector
    # features_to_select can only select a SUBSET
    # of set of features used to compute tSNE
    
    direct=True
    N=12
    n_bits=12*11
    func = partial(compute_pbb_eat_non_h,nw_IDs,nw_mv,n_bits,direct,N)
    pool = Pool(processes=n_cores)
    
    # add met_variables
    outp_df.insert(outp_df.shape[1],'F',nw_mv['F'])
    outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
    outp_df.insert(outp_df.shape[1],'ph',nw_mv['ph'])
    outp_df.insert(outp_df.shape[1],'F*nh',np.multiply(outp_df['F'],outp_df['nh']))
    outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
    outp_df.insert(outp_df.shape[1],'pbb_non_h',pool.map(func,nw_IDs.index))
    
    pool.close()
    pool.join()
    
    features_to_select = features_to_select + ['F','nh','ph','F*nh','F*nh*ph','pbb_non_h']
    #features_to_select = features_to_select + ['ph','F*nh']
    X = np.array(outp_df[features_to_select])
        
    # produce possible DTs
    start = time.time()    
    acc_net_l, pairw_sim_val_l, PF_points_l, param_list_l, models_l = PFsearch_DT_stbacc_II(X,labels,seed,features_to_select,n_cores,
                                            par_distr=dict(min_samples_split = uniform(3*10**-5,10**-3), 
                                            min_samples_leaf = uniform(3*10**-5,10**-3),
                                            max_depth = list(range(3,9)),
                                            min_impurity_decrease = uniform(0,0.01)),
                                            HP_budget=HP,M = nM,r=0.7,alpha_conf=0.05,int_I=inter_I,int1st=inter1st)
        
    end = time.time()
    print("Last DT search took "+ str(end-start)+ " secs.")
    
    # ESTO ES SOLO PARA PROBAR
    #with open(path +"/" + "test_pa_cachar_q_vola.pickle", "wb") as wr:
    #    wr.write(pickle.dumps([acc_net_l, pairw_sim_val_l, PF_points_l, param_list_l, models_l]))
        
    # produce the distinct DT's
    # instantiate these dicts here since there is
    # a processsing and folder for each FS
    distinct_DTs = OrderedDict()
    depth = OrderedDict()
    counter = OrderedDict()
    dict_repr_DT = OrderedDict()
    distinct_DTs, counter, depth, dict_repr_DT = print_PF_stb_acc_DT_IV(distinct_DTs,depth,counter, dict_repr_DT,nM,
                                                                PF_points_l, param_list_l, models_l,
                                                                features_to_select,X,labels,seed,r=1, folder= path, 
                                                                sort_by_acc=True, int_I=inter_I)
                
    with open(path +"/" + "distinct_DT.pickle", "wb") as wr:
        wr.write(pickle.dumps([distinct_DTs, counter, depth, 
                               dict_repr_DT,labels,l_v_eps,
                               features_to_select,HP,nM]))
    
    # obtain DT inclusions
    obtain_graph_DT_inclusions(distinct_DTs, counter, depth, path)
    draw_stb_acc_set_of_trees(distinct_DTs,path)
    
def graph_DT(ph,seed,n_DT,root,mnp=0.02,is_dir=True,lay=True,p_vr=None,to_file=False,proc_NA=False):   
        
    # open tSNE data
    # 1 opt
    if ph == floor(ph):
        if ph < 4:
            if not proc_NA:
                name= root + "clusters/tSNE_"+str(ph)+"opt_"+str(seed)+ "seed.pickle"
            else:
                name = root + "clusters/tSNE_"+str(ph)+"opt_NA_"+str(seed)+ "seed.pickle"
        else:
            
            name = root + "clusters/tSNE_big_"+str(seed)+ "seed.pickle"
        
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "opt_"  + str(mnp) + "mnp/distinct_DT.pickle"
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "opt_NA_"  + str(mnp) + "mnp/distinct_DT.pickle" 
                     
    else: # 1 ph
    
        if not proc_NA:
            name= root + "clusters/tSNE_"+str(ph)+"opt_"+str(seed)+ "seed.pickle"
        else:
            name = root + "clusters/tSNE_"+str(ph)+"opt_NA_"+str(seed)+ "seed.pickle"
    
        # create folder to store all outputs of this dataset
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "ph/distinct_DT.pickle"
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(ph) + "ph_NA/distinct_DT.pickle"
    
    with open(name, "rb") as file:
        [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file) 
    
    # open DTs data
    with open(path, "rb") as file:
        [distinct_DTs, counter, depth, 
         dict_repr_DT,labels,l_v_eps,
         features_to_select,HP,nM] = pickle.load(file) 
    
    n_key = list(dict_repr_DT.keys())[n_DT]
    DT = dict_repr_DT[n_key]
    
    constr_dict = extract_constraints_from_DT(DT)
    
    direct=True
    N=12
    n_bits=12*11
    func = partial(compute_pbb_eat_non_h,nw_IDs,nw_mv,n_bits,direct,N)
    pool = Pool(processes=cpu_count())
    
    outp_df.insert(outp_df.shape[1],'F',nw_mv['F'])
    outp_df.insert(outp_df.shape[1],'nh',[len(x) for x in nw_mv['nh']])
    if ph>=1:
        outp_df.insert(outp_df.shape[1],'ph',nw_mv['ph'])    
    outp_df.insert(outp_df.shape[1],'F*nh',np.multiply(nw_mv['F'],[len(x) for x in nw_mv['nh']]))
    outp_df.insert(outp_df.shape[1],'F*nh*ph',np.multiply(outp_df['F*nh'],outp_df['ph']))
    outp_df.insert(outp_df.shape[1],'pbb_non_h',pool.map(func,nw_IDs.index))
    
    pool.close()
    pool.join()
    
    list_m_r_inds, idbr_leaf_nodes, masks = graph_every_leaf(ph,DT,constr_dict,outp_df,type_min,nw_IDs,
                     nw_mv,direct=is_dir,lay_var=lay,inter_I=False,
                     N_dir=12,n_bits=12*11, print_var=p_vr,to_file=to_file)
    
    return DT, constr_dict, outp_df, ph, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv

# since uses apply_constraints() 
# table requires 'F' and 'nh' fields if used to build the DT
def graph_every_leaf(ph,DT,constr_dict,table,type_min,nw_IDs,
                     nw_mv,direct=True,lay_var=True,inter_I=False,
                     N_dir=12,n_bits=12*11,print_var=None,to_file=False):
    
    # get keys of leaf nodes
    split_nodes, leaf_nodes = DT._get_node_ids()
    idbr_leaf_nodes = dict(zip(leaf_nodes.values(), leaf_nodes.keys()))
    
    masks = []
    list_m_r_inds = []
    
    if ph >= 1:
        #table_without_metavars = table.drop(['F','nh','ph','F*nh'],axis=1)
        #table_without_metavars = table.drop(['ph','F*nh'],axis=1)
        #table_without_metavars = table.drop(['ph','F*nh','F','nh','F*nh*ph','pbb_non_h'],axis=1)
        table_without_metavars = table
    else:
        #table_without_metavars = table.drop(['F','nh','F*nh'],axis=1)
        #table_without_metavars = table.drop(['F*nh'],axis=1)
        table_without_metavars = table
    
    for k in idbr_leaf_nodes:
        
        print("Leaf "+str(k))
        
        # extract constraints of node ID k
        mask = apply_constraints(constr_dict,table,k,inter_I)
        masks.append(mask)
        
        if ph < 1:
        
            # extract respective mask for each type of min
            resp_mask = [(mask & [type_min[n]==i for n in range(len(type_min))]) for i in range(1,4)]
        
            for i in range(3):
                # there are nw of this opt type in the leaf => draw the most representative
                print("number of samples in leaf: "+str(sum(resp_mask[i])))
                if sum(resp_mask[i])>0: 
                
                    print(("RV" if i==0 else ("WEF" if i==1 else "PF"))+" min")
                    if to_file:
                        ind_mr = draw_the_most_representative(table_without_metavars[resp_mask[i]],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,k)
                    else:
                        ind_mr = draw_the_most_representative(table_without_metavars[resp_mask[i]],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,None)
                    list_m_r_inds.append(ind_mr)
            print()
            
        else:
           
            # extract respective mask for each type of min
            #resp_mask = (mask & [type_min[n]==ph for n in range(len(type_min))]) 
        
            print(("RV" if ph==1 else ("WEF" if ph==2 else "PF"))+" min")
            print("number of samples in leaf: "+str(sum(mask)))
            if to_file:
                ind_mr = draw_the_most_representative(table_without_metavars[mask],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,k)
            else:
                ind_mr = draw_the_most_representative(table_without_metavars[mask],table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,None)
            list_m_r_inds.append(ind_mr)    
            print()
            
    if not print_var is None:
        
        print_var_stats(print_var,table,idbr_leaf_nodes,masks)
            
    return list_m_r_inds, idbr_leaf_nodes, masks

def print_var_stats(print_var,table,idbr_leaf_nodes,masks,r_n=2):
    
    print(table.columns[print_var])
    node_IDs = list(idbr_leaf_nodes.keys())
    
    for i, mask in enumerate(masks):
        
        print("node "+str(node_IDs[i]) + str(", perct:"))
        selected_table = table.loc[mask, table.columns[print_var] ]
        #print("median: " + str(np.median(selected_table,axis=0)) + ", sd: " + str(np.std(selected_table,axis=0)))
        print("0.5: " + str(round(np.nanpercentile(selected_table, 0.5,axis=0),r_n)) +
              " 25: " + str(round(np.nanpercentile(selected_table, 25,axis=0),r_n)) +
              " 50: " + str(round(np.nanpercentile(selected_table, 50,axis=0),r_n)) +
              " 75: " + str(round(np.nanpercentile(selected_table, 75,axis=0),r_n)) +
              " 99.5: " + str(round(np.nanpercentile(selected_table, 99.5,axis=0),r_n)) +
              " 100: " + str(round(np.nanpercentile(selected_table, 100,axis=0),r_n)))


# receive dataframe with only descriptors of networks (not metavariables)   
# if to_file is not None, draw to a file of name 'leaf_to_file'             
def draw_the_most_representative(selected_df,table,nw_IDs,N_dir,n_bits,nw_mv,direct,lay_var,to_file,i_f=None):
    
    # obtain centroid 
    centroid = np.mean(selected_df)
    # obtain nearest representative
    distances = [np.sum(np.absolute(centroid - selected_df.loc[l,:])) for l in selected_df.index]
    
    if len(distances)>0:
        arg_min_dist = np.argmin(distances)
        i = selected_df.index[arg_min_dist]
        # draw it
        if to_file is None:
            if i_f is None:
                draw_netw(nw_IDs.loc[i],n_bits,direct,nw_mv.loc[i,'nh'],N_dir,
                          test_layout=lay_var,receive_int=True)
            else:
                draw_netw(nw_IDs.loc[i],n_bits,direct,nw_mv.loc[i,'nh'],N_dir,
              test_layout=lay_var,receive_int=True,i_f=i_f,F=nw_mv.loc[i,'F'],ph=nw_mv.loc[i,'ph'])
        else:
            if i_f is None:
                draw_netw(nw_IDs.loc[i],n_bits,direct,nw_mv.loc[i,'nh'],N_dir,
              test_layout=lay_var,receive_int=True,file='leaf_'+str(to_file)+'.eps')
            else:
                draw_netw(nw_IDs.loc[i],n_bits,direct,nw_mv.loc[i,'nh'],N_dir,
              test_layout=lay_var,receive_int=True,file='leaf_'+str(to_file)+'.eps',i_f=i_f,F=nw_mv.loc[i,'F'],ph=nw_mv.loc[i,'ph'])
        print(table.loc[i])
    
        return i
    else:
        print("empty set")
        return -1

def apply_constraints(constr_dict,df,node_id,inter_I=False):
    
    X = np.array(df)
    n_sample, n_feat = df.shape
    
    mask = np.repeat(True, n_sample)
    
    for i in range(len(constr_dict[node_id])):
    
        if inter_I:
            mask = np.logical_and(
                        mask, ct._map_node(X, *constr_dict[node_id][i]))
        else:    
            mask = np.logical_and(
                        mask, ct._map_node_II(X, *constr_dict[node_id][i]))

    return mask

# return a dict with (node_ID,list_of_constraints)
def extract_constraints_from_DT(DT):
    
    split_nodes, leaf_nodes = DT._get_node_ids()
    idbr_split_nodes = dict(zip(split_nodes.values(), split_nodes.keys()))
    idbr_leaf_nodes = dict(zip(leaf_nodes.values(), leaf_nodes.keys()))
    
    output = {}
    
    for node_id in idbr_leaf_nodes:
    
        the_dict = idbr_leaf_nodes
       
        n_levels = len(the_dict[node_id])
        constraints = []
    
        for i in range(n_levels):
        
            constraints.append(DT._thresholds[the_dict[node_id][:i+1] ])
        
        output[node_id] = constraints
        
    for node_id in idbr_split_nodes:
    
        the_dict = idbr_split_nodes
       
        n_levels = len(the_dict[node_id])
        constraints = []
    
        for i in range(n_levels):
        
            constraints.append(DT._thresholds[the_dict[node_id][:i+1] ])
        
        output[node_id] = constraints
    
    return output

# assumes df has meta-variables, as the processing of  graph_DT()          
def draw_a_sample(node_ID,df,constr_dict,n_sample,seed,nw_IDs,nw_mv,
                  lay_var=True,direct=True,N_dir=12,inter_I=False):
        
        if direct==False:
                n_bits = N_dir * (N_dir - 1)/2
        else:
                n_bits = N_dir * (N_dir - 1)
        
        mask = apply_constraints(constr_dict,df,node_ID,inter_I)
        selected_df = df[mask].sample(n=n_sample,random_state=seed)
        selected_rows =selected_df.index    
        
        for i in selected_rows:
                print(i)
                draw_netw(nw_IDs.loc[i],n_bits,direct,list(range(int(df['nh'].loc[i]))),N_dir,
                          test_layout=lay_var,receive_int=True)
                print(str(df.loc[i]))
                print()
                print()

# this method receives int elem
# coding network, and it draws it
def draw_netw(elem,n_bits,direct,hunters_indexes,N,test_layout=True,
              receive_int=True,i_f=None,F=None,ph=None,file=None):
    
    if receive_int:
        
        #convert to binary, first least binary is at the right extreme
        bin_i = bin(elem)[2:]
        # get strings of uniform length
        bin_i = bin_i.rjust(int(n_bits),"0")
        
        # get adjacency matrix
        if direct==False:
            A = ev.fill_matrix(bin_i,N)
            A = np.tril(A, -1)
        else:
            A = ev.fill_D_matrix(bin_i,N)
            
    else: # input is a list from evolution
        
        A = ev.fill_matrix_from_list(elem,N,direct)
        
                
    g = Graph(directed=direct)
    g.add_vertex(N)
    g.add_edge_list(np.transpose(A.nonzero()))
    g.vp.hunters = g.new_vertex_property("bool",vals= list(map(lambda x: True if x in hunters_indexes else False, range(N))))           
          
        # print graph
    if test_layout==True:
            graph_draw(g,vertex_fill_color= g.vp.hunters,pos=arf_layout(g, max_iter=50),vertex_text=g.vertex_index,inline=True,output=file)
    else:
            graph_draw(g,vertex_fill_color= g.vp.hunters,vertex_text=g.vertex_index,inline=True,output=file)
       
    if not i_f is None:
        
        pbb_eat_1_step = ev.compute_vector_pbb_eat(A,hunters_indexes,ph,F,N,direct)
        
        red_var_cost_by_node = list(map(lambda p: i_f(1 - p),pbb_eat_1_step))
      
        for i in range(N):
            print("node "+ str(i)+ ", pbb: " + str(pbb_eat_1_step[i])
                  + ", RV: " + str(red_var_cost_by_node[i]))  
      
        return pbb_eat_1_step, red_var_cost_by_node

    else:
        
        return None, None

def print_RV_values(list_m_r_inds,idbr_leaf_nodes,nw_IDs,nw_mv,
                    root,
                    direct=True,n_bits=12*11,N_dir=12,lay_var=True):
    
    with open(root + "interp_fn_70000_norm.pickle", "rb") as file:
        i_fn = pickle.load(file) 
        
    node_IDs = list(idbr_leaf_nodes.keys())
    
    for i, ind in enumerate(list_m_r_inds):
        
        print("DT node "+str(node_IDs[i]))
        print("nw ID " + str(nw_IDs.loc[ind]))
        draw_netw(nw_IDs.loc[ind],n_bits,direct,nw_mv.loc[ind,'nh'],N_dir,
                  test_layout=lay_var,receive_int=True,i_f=i_fn,
                  F=nw_mv.loc[ind,'F'],ph=nw_mv.loc[ind,'ph'])

def compute_pbb_indicator(nw_ID,hunters_indexes,ph,F,N=12,direct=True):

    pbb_vector = ev.compute_vector_pbb_eat(nw_ID,hunters_indexes,ph,F,N,direct)
    
    return np.mean(pbb_vector)/np.std(pbb_vector)

def clust_per_partition(A,part_labels,labels,direct,N):
    
    clust_per_partition = []
    
    #main_graph
    g = Graph(directed=direct)
    g.add_vertex(N)
    g.add_edge_list(np.transpose(A.nonzero()))
    #graph_draw(g,pos=arf_layout(g, max_iter=0),vertex_text=g.vertex_index,inline=True)
    
    for x in part_labels:
        #obtain subgraph
        u=Graph(GraphView(g,vfilt=labels==x),prune=True)
        #graph_draw(u,pos=arf_layout(u, max_iter=0),vertex_text=u.vertex_index,inline=True)
        clust_per_partition.append( np.mean(graph_tool.clustering.local_clustering(u, undirected=not direct).get_array()) )
    
    return clust_per_partition 

# this method is the target function 
# for the processes invoked in search_loc_min
# i is an int we aim to check whether it is a local minimum
def check_loc_min(n_bits,N,hunters_indexes,ph,F,interp_fn,directed,check_is_min,i):
    
    if check_is_min == 1: # it is RV local min
    
        return True, False, 0
    
    elif check_is_min == 2:
        
        return False, True, 0
    
    else:
    
        #convert to binary, first least binary is at the right extreme
        bin_i = bin(i)[2:]
        # get strings of uniform length
        bin_i = bin_i.rjust(int(n_bits),"0")
        
        # we get the adjacency matrix 
        if directed == False:
            A = ev.fill_matrix(bin_i,N)
        else:
            A = ev.fill_D_matrix(bin_i,N)
        
        # [reduction of variability cost, welfare cost]
        current_minimum_cand = ev.nw_cost(A,hunters_indexes,ph,F,N,interp_fn,directed,receive_binary=False)
        
        # only initialization values
        is_red_var_loc_min = True
        is_wef_loc_min = True
        
        # we search over all neighbor networks to check if current_graph is a local minimum
        for j in range(int(n_bits)):
            
            if bin_i[j]=='0':
                curr_neighbor = bin_i[0:j] + '1' + bin_i[(j+1):]
            else:
                curr_neighbor = bin_i[0:j] + '0' + bin_i[(j+1):]
                
            current_neighb_val = ev.nw_cost(curr_neighbor,hunters_indexes,ph,F,N,interp_fn,directed,receive_binary=True)
            
            if current_neighb_val[0] < current_minimum_cand[0]:
                # current_graph associated to bin_i is not a variability reduction local minimum
                is_red_var_loc_min = False
                
            if current_neighb_val[1] < current_minimum_cand[1]:
                # current_graph associated to bin_i is not a welfare local minimum
                is_wef_loc_min = False
            
            if  is_red_var_loc_min == False and is_wef_loc_min==False:
                # we are sure current_minimum_cand is not a local minimum for neither criterion
                return False, False, current_minimum_cand
        
        return is_red_var_loc_min, is_wef_loc_min, current_minimum_cand

def compute_pbb_ind_line_mod_part(N,direct,nw_mv,nw_IDs,outp_df,interp_fn,check_is_min,i):
    
    nw_ID = nw_IDs.loc[i]
    hunters_indexes = nw_mv.loc[i,'nh']
    ph = nw_mv.loc[i,'ph']
    F = nw_mv.loc[i,'F']
    labels= outp_df.loc[i,'partition']
    #print(labels)
    part_labels = np.unique(labels)
    n_part = part_labels.shape[0]
    
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
        
    # obtain average of local clustering over each partition
    avg_clust = np.mean(clust_per_partition(A,part_labels,labels,direct,N))
    
    is_RV, is_WEF, cost_cand = check_loc_min(n_bits,N,hunters_indexes,ph,F,interp_fn,direct,check_is_min,i)
    
    return compute_pbb_indicator(A,hunters_indexes,ph,F,N,direct), n_part, avg_clust, is_RV, is_WEF
        
def add_indicators(opt,seed,n_cores,path,pr_NA=True,check_is_min=False):
    """
    if ph == floor(ph):
        if ph < 4:
            if not proc_NA:
                name= root + "clusters/tSNE_"+str(ph)+"opt_"+str(seed)+ "seed.pickle"
            else:
                name = root + "clusters/tSNE_"+str(ph)+"opt_NA_"+str(seed)+ "seed.pickle"
        else:
            
            name = root + "clusters/tSNE_big_"+str(seed)+ "seed.pickle"
            
        
        with open(name, "rb") as file:
            [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file)      
    else: # 1 ph
        with open(root + "tSNE_"+str(ph)+"ph_"+str(seed)+ "seed.pickle", "rb") as file:
            [outp_df, type_min, nw_IDs, nw_mv, tsne, clust] = pickle.load(file)
    """
    
    fs = ['partition','out_assort','in_assort', 'nw_mod','mean_local_clust']
    
    with open(path + "interp_fn_70000_norm.pickle", "rb") as file:
        i_fn = pickle.load(file) 
    
    if opt < 4:
        
        outp_df, type_min, nw_IDs, nw_mv = preprocess_datafr_1_opt(opt,s=seed,N=12,FS=fs,root=path,
                              direct=True,n_sample=None,phs=None,proc_NA=pr_NA) 
    else:
        outp_df_1, type_min_1, nw_IDs_1, nw_mv_1 = preprocess_datafr_1_opt(opt=1,s=seed,FS=fs,root=path,n_sample=None,proc_NA=True)
        outp_df_2, type_min_2, nw_IDs_2, nw_mv_2 = preprocess_datafr_1_opt(opt=2,s=seed,FS=fs,root=path,n_sample=None,proc_NA=True)
        outp_df_3, type_min_3, nw_IDs_3, nw_mv_3 = preprocess_datafr_1_opt(opt=3,s=seed,FS=fs,root=path,n_sample=None,proc_NA=True)
        
        outp_df = pd.concat([outp_df_1,outp_df_2,outp_df_3])
        type_min = type_min_1 + type_min_2 + type_min_3
        nw_IDs = pd.concat([nw_IDs_1,nw_IDs_2,nw_IDs_3])
        nw_mv = pd.concat([nw_mv_1,nw_mv_2,nw_mv_3])
        
        outp_df = outp_df.reset_index(drop=True)
        nw_IDs = nw_IDs.reset_index(drop=True)
        nw_mv = nw_mv.reset_index(drop=True)
    
    func1 = partial(compute_pbb_ind_line_mod_part,12,True,nw_mv,nw_IDs,outp_df,i_fn,check_is_min) 
    
    pool = Pool(processes=n_cores)
    
    output = pool.map(func1, outp_df.index)
    
    outp_df.insert(outp_df.shape[1],'pbb_std',[x[0] for x in output])
    
    #outp_df_all, type_min, nw_IDs, nw_mv = preprocess_datafr_1_opt(ph,s=seed,root=root,n_sample=None,features_to_select=2)
    
    outp_df.insert(outp_df.shape[1],'n_partitions',[x[1] for x in output])
    outp_df.insert(outp_df.shape[1],'mod*clust',np.multiply(outp_df['mean_local_clust'],outp_df['nw_mod'])) 
    #outp_df.insert(outp_df.shape[1],'mod*out_ass',np.multiply(outp_df['out_assort'],outp_df['nw_mod']))
    #outp_df.insert(outp_df.shape[1],'mod*in_ass',np.multiply(outp_df['in_assort'],outp_df['nw_mod']))
    outp_df.insert(outp_df.shape[1],'avg_cl_per_partition',[x[2] for x in output])
    outp_df.insert(outp_df.shape[1],'is_RV',[x[3] for x in output])
    outp_df.insert(outp_df.shape[1],'is_WEF',[x[4] for x in output])
    
    if 'out_assort' in outp_df.columns:
        outp_df = outp_df.drop(['out_assort','in_assort', 'nw_mod','mean_local_clust'],axis=1)
    else:
        outp_df = outp_df.drop(['nw_mod','mean_local_clust'],axis=1)
    
    pool.close()
    pool.join()
    
    return outp_df, type_min, nw_IDs, nw_mv


    
#if __name__ == '__main__':
def main(dir_addr="/home/fplana/evol_datafr_analysis/"):
    
    tSNE_and_OPTICS_one_opt(1,seed=400,n_s=None,path=dir_addr,dim=2,ite=1000,prop_min_sampl=.03,proc_NA=True)
    produce_PF_DTs(ph=1,seed=400,n_cores=12,HP=1000,nM=15,root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.02,rst=1.5,inter1st=False,proc_NA=True)
    tSNE_and_OPTICS_one_opt(2,seed=400,n_s=None,path=dir_addr,dim=2,ite=1000,prop_min_sampl=.03)
    produce_PF_DTs(ph=2,seed=400,n_cores=12,HP=1000,nM=15,root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.02,rst=1.5,inter1st=False)
    """
    produce_PF_DTs(ph=3,seed=400,n_cores=20,HP=300,nM=15,root=dir_addr,
                       inter_I=False,prop_min_sampl=.03,
                       mnp=0.02,rst=1.5,inter1st=False,proc_NA=True)
    """
    
def check_DT_acc(n_DT,opt,seed,root,ph=None,proc_NA=None,mnp=None,F_nh_small=None):
    
    if opt < 3:
        
        if not proc_NA:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_"  + str(mnp) + "mnp" 
        else:
            path = root + "clusters/DTs_"+str(seed)+"seed_" + str(opt) + "opt_NA_"  + str(mnp) + "mnp" 
            
    else:
        
        path = root + "clusters/DTs_PF_"+str(seed)+"seed_" + str(ph) + "ph_"  + str(F_nh_small) + "_F_nh_small"
        
    with open(path + "/distinct_DT.pickle", "rb") as file:
        [distinct_DTs, counter, depth, 
         dict_repr_DT,labels,l_v_eps,
         features_to_select,HP,nM] = pickle.load(file)
        
    # to obtain accuracy
    list_keys=list(distinct_DTs.keys()) 
    acc_list=[x[0] for x in distinct_DTs[list_keys[n_DT]]]
    
    print("mean accuracy: "+ str(np.mean(acc_list)))
    
    return acc_list

def produce_extra_vars(nc,root):
    
    df_1, type_min_1, nw_IDs_1, nw_mv_1 = add_indicators(opt=1,seed=400,n_cores=nc,path=root,pr_NA=True,check_is_min=1)
    df_2, type_min_2, nw_IDs_2, nw_mv_2 = add_indicators(opt=2,seed=400,n_cores=nc,path=root,pr_NA=True,check_is_min=2)
    df_3, type_min_3, nw_IDs_3, nw_mv_3 = add_indicators(opt=3,seed=400,n_cores=nc,path=root,pr_NA=True,check_is_min=3)
    
    with open(root + "extra_vars_all_optima.pickle", "wb") as wr:
        wr.write(pickle.dumps([df_1, type_min_1, nw_IDs_1, nw_mv_1, 
                               df_2, type_min_2, nw_IDs_2, nw_mv_2,
                               df_3, type_min_3, nw_IDs_3, nw_mv_3]))
        

