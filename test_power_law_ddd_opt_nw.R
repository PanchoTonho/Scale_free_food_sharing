library(igraph)
library(poweRlaw)
library(foreach)
library(doParallel)

# by using library(bit64) 
# and as.bitstring(as.integer64(x))
# you can read integers representable by 64 bits

#toBits <- function (x, nBits = 8){
#       tmp = rev(as.numeric(intToBits(x)))
#       l = length(tmp)
#       if(nBits>l){
#         return(c(rep(0,nBits-l),tmp))
#       }else if(nBits<l){
#         return(tmp[(l-nBits+1):l])
#       }
#       return(tmp)
#}

# receives the string with the binary encoding of every optimal network
compute_pl_vars_II <- function(bin_i,s=10,n_it_boot=500,max_prop=0.2,N=12){
  
  # decompose string into a vector of characters
  bin_i = strsplit(bin_i, "")[[1]]
  
  # create graph by adding edges
  edges <- c()
  for(i in 1:N){
    for (j in 1:N){
      if (i==j) next();
      
      if (j>i){
        jj=j-1
      }else{
        jj= j
      }
      
      if (bin_i[(i-1)*(N-1) + jj]=="1"){
        edges <- c(edges,i);
        edges <- c(edges,j); 
      }
    }
  }
  # g is a simple and undirected graph
  g <- as.undirected(add_edges(make_empty_graph(n = N),edges),mode="collapse")
  #plot(g)
  
  # get network degrees
  #in.deg<-degree(g,mode="in");
  #out.deg<-degree(g,mode="out");
  deg <- degree(g)
  
  #in_ddd <- c()
  out_ddd <- c()
  
  #i = 1
  
  # compute degree degree distances
  #while(i<length(edges)){
  # traversing edges in this way preserve the graph property of being simple
  for(e in E(g)){
    
    #ind1 = in.deg[edges[i]]
    #ind2 = in.deg[edges[i+1]]
    outd2 = deg[head_of(g,e)]
    outd1 = deg[tail_of(g,e)]
    
    #in_ddd <- c(in_ddd,max(ind1,ind2)/min(ind1,ind2))
    out_ddd <- c(out_ddd,max(outd1,outd2)/min(outd1,outd2))
    
    #i=i+2
  }
  
  m_m_0 <- conpl$new(out_ddd)
  
  # bootstrap to get significance of the gof 
  xmin_upper_bound= quantile(out_ddd,max_prop)
  est <- estimate_xmin(m_m_0,xmax = xmin_upper_bound)
  xmin =est$xmin;
  alpha=est$pars;
  bs_p = bootstrap_p(m_m_0,seed=s, no_of_sims=n_it_boot, threads=1,xmax = xmin_upper_bound)

  return(c(bs_p$p,xmin,alpha))
}

# receives the string with the binary encoding of every optimal network
# fit power law to the degree distributions
compute_pl_vars_III <- function(bin_i,s=10,n_it_boot=500,max_prop=0.2,N=12){
  
  # decompose string into a vector of characters
  bin_i = strsplit(bin_i, "")[[1]]
  
  # create graph by adding edges
  edges <- c()
  for(i in 1:N){
    for (j in 1:N){
      if (i==j) next();
      
      if (j>i){
        jj=j-1
      }else{
        jj= j
      }
      
      if (bin_i[(i-1)*(N-1) + jj]=="1"){
        edges <- c(edges,i);
        edges <- c(edges,j); 
      }
    }
  }
  # g is a simple, directed graph
  g <- add_edges(make_empty_graph(n = N),edges)
  #plot(g)
  
  # get network degrees
  in.deg<-degree(g,mode="in");
  out.deg<-degree(g,mode="out");
  #deg <- degree(g)
  
  # discrete power law fit to out-degree
  m_m_0 <- displ$new(out.deg)
  
  # bootstrap to get significance of the gof 
  xmin_upper_bound= quantile(out.deg,max_prop)
  est <- estimate_xmin(m_m_0,xmax = xmin_upper_bound)
  xmin =est$xmin;
  alpha=est$pars;
  bs_p = bootstrap_p(m_m_0,seed=s, no_of_sims=n_it_boot, threads=1,xmax = xmin_upper_bound)
  
  # discrete power law fit to in-degree
  m_m_0_in <- displ$new(in.deg)
  
  # bootstrap to get significance of the gof 
  xmin_upper_bound = quantile(in.deg,max_prop)
  est <- estimate_xmin(m_m_0_in,xmax = xmin_upper_bound)
  xmin_in = est$xmin;
  alpha_in = est$pars;
  bs_p_in = bootstrap_p(m_m_0_in,seed=s, no_of_sims=n_it_boot, threads=1,xmax = xmin_upper_bound)
  
  return(c(bs_p$p,xmin,alpha,bs_p_in$p,xmin_in,alpha_in))
}

processing_deg_deg_distance <- function(){
  # read encodings as strings
  all_encodings <- read.csv("encod_all_optima.csv",colClasses = c("numeric","character"))
  n_networks <- dim(all_encodings)[1]
  
  #numCores <- detectCores()
  registerDoParallel(6) 
  
  system.time({
    # return a dataframe
    r <- foreach(i=1:n_networks, .combine=rbind) %dopar% {
      out <- tryCatch(compute_pl_vars_II(all_encodings[i,2],s=i),
                      error=function(cond) {return(c(0,0,0))})
    }
  })
  
  stopImplicitCluster()
  
  write.table(r, file = "all_data_pl_fit_ddd.csv", sep = ",")
}

processing_degree_distr <- function(){
  # read encodings as strings
  all_encodings <- read.csv("encod_all_optima.csv",colClasses = c("numeric","character"))
  n_networks <- dim(all_encodings)[1]
  
  #numCores <- detectCores()
  registerDoParallel(6) 
  
  system.time({
    # return a dataframe
    r <- foreach(i=1:n_networks, .combine=rbind) %dopar% {
      out <- tryCatch(compute_pl_vars_III(all_encodings[i,2],s=i),
                      error=function(cond) {return(c(0,0,0,0,0,0))})
    }
  })
  
  stopImplicitCluster()
  
  write.table(r, file = "all_data_pl_fit_degreeDistr.csv", sep = ",")
}

processing_deg_deg_distance()
processing_degree_distr()