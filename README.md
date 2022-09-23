# Scale_free_food_sharing
Supporting code and datasets for the Appendix "Are food-sharing networks scale-free?" of the PhD thesis "Computational models for network societies".

The code shared here requires to download the code and datasets of the companion repository, associated to the submitted paper "Modularity of food-sharing networks minimises the risk for individual and group starvation in hunter-gatherer societies" to PLoS One.

https://figshare.com/articles/software/food_sharing_data_and_code/19203926

Once downloaded, the files 

produce_graphics.py
characterize_clusters.py

must be overwritten by the files with the same name from this repository. All the code is written for Python 3 and R. The dependencies of the Python code in this repository are included in the dependencies required by the code in the companion repo. Hence, consult in the documentation.pdf file therein to check for Python code dependencies. 

===================================================
Brief description of the main contents of this repo
===================================================

1) produce_graphics.py 

It has the methods to produce the datasets. In these methods, path is the address of the directory storing all the datasets and codes you download from here. These methods are

add_pl_variables_III(path) to compute the estimators of the tail exponent, which makes use of tail_estimation.py for Python 3 shared in 
https://github.com/ivanvoitalov/tail-estimation

export_binary_encodings(path,n_cores=None,direct=True,N=12) to compute a binary encoding of all the networks, stored in encod_all_optima.csv, which is the input for the code in test_power_law_ddd_opt_nw.R which is the R script that computes the power-law fitting described in the analyses. The R script makes use of the following libraries,

igraph        https://igraph.org/r/

poweRlaw      https://cran.r-project.org/web/packages/poweRlaw/index.html

foreach       https://cran.r-project.org/web/packages/foreach/index.html

doParallel    https://cran.r-project.org/web/packages/doParallel/index.html

This script produced the files all_data_pl_fit_ddd.csv, all_data_pl_fit_degreeDistr.csv, which are then opened in Python and analyzed by the following script. 

2) characterize_clusters.py

It has the methods to visualize main statistics of the datasets. The main method to use here is 

DT, constr_dict, outp_df, ph, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv = graph_DT_plus_pl_degDistr(opt,seed,root,n_DT,ph=0,F_nh_small=True,proc_NA=False,mnp=0.02,
                   is_dir=True,lay=True,p_vr=None,to_file=False,extra_DTfeat=None,n_s=None,r_n=2)
                   
which prints summary statistics (some percentile values of required variable p_vr) on each leaf of tree n_DT, and prints a number n_s of randomly sampled networks from the subset of each tree leaf where the variable p_vr has a value greater than 0.5 (designed for the p-value). This method recovers in the dataframe outp_df either the estimators of the tail exponent (from Voitalov et al 2019) of in, outdegree distribution, and the p-value, xmin and alpha estimated by poweRlaw. The outputs of these method have the same semantics as graph_DT_PF_ph() in the companion repo. Read the documentation.pdf file on the companion repo for more info about the outputs of this method. The arguments of this method are the following.

opt:           the type of optima (1: RV, 2: WEF, 3: PF)

seed:          the seed used to produce the clustering and decision tree datasets. Read the documentation.pdf file on the companion repo.

root:          the address of the directory storing all the datasets and codes you download from here.

n_DT:          ID of the decision tree to visualize the dataset. Read the documentation.pdf file on the companion repo.

ph:            specific ph value used to compute network optima. Only has effect if opt==3.

F_nh_small:    True if we want to visualize F*nh<=N. Only has effect if opt==3.

proc_NA:       removing attributes -if True- or networks -if False- with NA to produce the clustering datasets. Read the documentation.pdf file on the companion repo.

mnp:           hyperparameter necessary to produce the clustering datasets. Only has effect if opt==1 or 2. Read the documentation.pdf file on the companion repo.

is_dir:        whether we work with directed networks. True on all these datasets.

lay:           boolean indicating whether to use ARF layout or not to display the network. Read the documentation.pdf file on the companion repo.

p_vr:          Column index of the variable in the dataframe outp_df which you want to print summary statistics. These indexes vary depending on the value of opt.

to_file:       True if each network image is stored in a .pdf. 

extra_DTfeat:  list of names of additional features to be included in outp_df. Read the documentation.pdf file on the companion repo.

n_s:           number of networks to randomly sample from each tree leaf.

r_n:           number of decimals to print in the statistics summary.

There is an analog method graph_DT_plus_pl_ddd(), which can be used to print the summary statistics of the degree-degree-distance.

==================================================
Code to reproduce the data printed in the Appendix
==================================================

Table C.1 and Figure C.1 (a), (b)

DT, constr_dict, outp_df_WEF, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv=graph_DT_plus_pl_degDistr(opt=2,seed=400,root="",n_DT=2,ph=0,F_nh_small=True,proc_NA=False,mnp=0.02,is_dir=True,lay=False,p_vr=24,to_file=False,extra_DTfeat=None,n_s=2)
also use p_vr=26, 27, 29

Table C.2

DT, constr_dict, outp_df_WEF, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv=graph_DT_plus_pl_degDistr(opt=2,seed=400,root="",n_DT=2,ph=0,F_nh_small=True,proc_NA=False,mnp=0.02,is_dir=True,lay=False,p_vr=0,to_file=False,extra_DTfeat=None,n_s=2) 
also use p_vr=1

and lines (c=0.5,25,50,75,99.5)

str(round(np.percentile(outp_df_WEF['in_deg_var'], c,axis=0),2))

str(round(np.percentile(outp_df_WEF['out_deg_var'], c,axis=0),2))

Table C.3 and Figure C.1 (c) 

DT, constr_dict, outp_df_RV, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv=graph_DT_plus_pl_degDistr(opt=1,seed=400,root="",n_DT=14,ph=0,F_nh_small=True,proc_NA=True,mnp=0.02,is_dir=True,lay=False,p_vr=22,to_file=False,extra_DTfeat=None,n_s=3)
also use p_vr=24, 25, 27 

Table C.4 and Figure C.2 

DT, constr_dict, outp_df_PF, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv=graph_DT_plus_pl_degDistr(opt=3,seed=50,root="",n_DT=20,ph=0.02,F_nh_small=True,proc_NA=True,mnp=0.02,is_dir=True,lay=False,p_vr=20,to_file=False,extra_DTfeat=None,n_s=2) 
also use p_vr=22, 23, 25

Table C.5

DT, constr_dict, outp_df_PF, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv=graph_DT_plus_pl_degDistr(opt=3,seed=50,root="",n_DT=20,ph=0.02,F_nh_small=True,proc_NA=True,mnp=0.02,is_dir=True,lay=False,p_vr=0,to_file=False,extra_DTfeat=None,n_s=2)
also use p_vr=6

and lines (c=0.5,25,50,75,99.5)

str(round(np.percentile(outp_df_PF['mean_out_deg'], c,axis=0),2))

str(round(np.percentile(outp_df_PF['out_deg_var'], c,axis=0),2))

Other multiobjective networks

DT, constr_dict, outp_df_PF, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv=graph_DT_plus_pl_degDistr(opt=3,seed=50,root="",n_DT=12,ph=0.08,F_nh_small=True,proc_NA=True,mnp=0.02,is_dir=True,lay=False,p_vr=20,to_file=False,extra_DTfeat=None,n_s=2)

DT, constr_dict, outp_df_PF, opt, type_min, list_m_r_inds, idbr_leaf_nodes, masks, nw_IDs, nw_mv=graph_DT_plus_pl_degDistr(opt=3,seed=50,root="",n_DT=7,ph=0.15,F_nh_small=True,proc_NA=True,mnp=0.02,is_dir=True,lay=False,p_vr=20,to_file=False,extra_DTfeat=None,n_s=2)
