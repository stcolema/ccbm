permuteData <- function(X, 
                        replace_items = F,
                        replace_features = F,
                        N_samp = if (replace_items) nrow(X) else ceiling(.632*nrow(X)), 
                        P_samp =  if (replace_features) ncol(X) else ceiling(.632*ncol(X))
) {
  N <- nrow(X)
  P <- ncol(X)
  row_ind <- sample(1:N, size = N_samp, replace = replace_items)
  col_ind <- sample(1:P, size = P_samp, replace = replace_features)
  
  list(data = as.matrix(X[row_ind, col_ind]), items = row_ind)
}

consensusClusteringPerm <- function(X, S, R,
                                initial_labels = NULL,
                                K_max = 50,
                                alpha = 1,
                                dataType = "G",
                                replace_items = F,
                                replace_features = F,
                                N_samp = if (replace_items) nrow(X) else ceiling(.632*nrow(X)), 
                                P_samp =  if (replace_features) ncol(X) else ceiling(.632*ncol(X))) {
  
  # cl_samples <- matrix(NA, nrow = S, ncol = nrow(X))
  
  cl_samples <- foreach::foreach(
    s = 1:S,
    .export = c("X", 
  "initial_labels",
  "R", 
  "K_max",
  "alpha", 
  "dataType",
  "replace_items",
  "replace_features",
  "N_samp",
  "P_samp"),
    .packages = c("ccbm", "Rcpp")
  ) %dopar% {
    set.seed(s)
    
    cl <- rep(NA, nrow(X))
    
    if (is.null(initial_labels)) {
      initial_labels <- priorLabels(alpha, K_max, nrow(X))
    }
    
    X_perm <- permuteData(X)
    samples <- mixtureModel(
      X_perm$data,
      K_max,
      initial_labels[X_perm$items],
      dataType,
      R,
      R,
      rep(alpha, K_max),
      s
    )$samples
    
    cl[X_perm$items] <- samples
  
    cl
  }
  
  cl_samples <- matrix(unlist(cl_samples), nrow = S, byrow = T)
  cl_samples
  
  # for(s in 1:S){
  #   set.seed(s)
  #   
  #   if (is.null(initial_labels)) {
  #     initial_labels <- priorLabels(alpha, K_max, nrow(X))
  #   }
  #   
  #   X_perm <- permuteData(X)
  #   cl <- mixtureModel(
  #     X_perm$data,
  #     K_max,
  #     initial_labels[X_perm$items],
  #     dataType,
  #     R,
  #     R,
  #     rep(alpha, K_max),
  #     s
  #   )$samples
  # 
  #   cl_samples[s, X_perm$items] <- cl[1, ]
  # }
  # cl_samples
}

createCM <- function(samples){
  
  N <- ncol(samples)
  consensus_matrix <- matrix(0, N, N)
  
  inclusion_mat <- ! is.na(samples)
  inclusion_count <- colSums(inclusion_mat)
  
  for(i in 1:(N-1)){
    for(j in (i+1):N){
      in_same_sample <- sum(inclusion_mat[, i] & inclusion_mat[, j])
      coclust <- sum(samples[, i] == samples[, j], na.rm = T)
      consensus_matrix[i, j] <- consensus_matrix[j, i] <- coclust / in_same_sample
    }
  }
  diag(consensus_matrix) <- 1
  
  consensus_matrix
}


library(mdiHelpR)
library(magrittr)
library(doFuture) # install.packages("doFuture")
library(pheatmap)
library(mcclust)
library(ccbm)
library(ggplot2)


set.seed(1)
setMyTheme()

registerDoFuture()
plan(multiprocess)

K <- 5
N <- 200
P <- 20
S <- 500
R <- 10
K_max <- 50
delta_mu <- 1
P_n <- 100
cluster_sd <- 1

x <- generateSimulationDataset(K, N, P, 
                               delta_mu = delta_mu,
                               cluster_sd = cluster_sd,
                               p_n = P_n
)

X <- my_data <- as.matrix(scale(x$data))
pheatmap(X, cluster_cols = F)

samples <- consensusClusteringPerm(X, S, R, 
                                   initial_labels = NULL, 
                                   K_max = ceiling(N/8),
                                   alpha = 1, 
                                   dataType = "G" #,
                                   # N_samp = ceiling(.632*nrow(X)),
                                   # P_samp =  ceiling(0.2*.632*nrow(X))
)

cm <- createCM(samples)
# diag(cm) <- 0
pheatmap(cm)

true_cc <- x$cluster_IDs %>% 
  matrix(nrow = 1) %>% 
  createSimilarityMat()

compareSimilarityMatricesAnnotated(true_cc, cm, cluster_IDs = x$cluster_IDs)
compareSimilarityMatricesAnnotated(cm, true_cc, cluster_IDs = x$cluster_IDs)

cm <- bin_cc %>%
  createSimilarityMat()
pheatmap(cm, color = simColPal())
