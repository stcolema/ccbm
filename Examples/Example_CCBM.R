
library(mdiHelpR)
library(magrittr)
library(doFuture) # install.packages("doFuture")
library(pheatmap)
library(mcclust)
library(ccbm)

set.seed(1)
setMyTheme()

# registerDoFuture()
# plan(multiprocess)

K <- 5
N <- 200
P <- 20
S <- 250
R <- 50
K_max <- 50
delta_mu <- 1
cluster_sd <- 1

x <- generateSimulationDataset(K, N, P, 
                               delta_mu = delta_mu,
                               cluster_sd = cluster_sd)
annotatedHeatmap(x$data, x$cluster_IDs)

X <- my_data <- as.matrix(scale(x$data))

mclust_samples <- bayesMclust(X, 200, 20, K_max = 7)
psm <- mclust_samples %>% 
  createSimilarityMat()

my_labels <- sample(1:K, N, replace = T)
binary_data <- matrix(0, nrow = N, ncol = P)
phis <- c(0.2, 0.4, 0.6, 0.8, 1.0)
for(p in 1:P){
  k_phis <- sample(phis)
  for(n in 1:N){
    binary_data[n, p] <- sample(0:1, 1, prob = c(1 - k_phis[my_labels[n]], k_phis[my_labels[n]]))
  }
}
row.names(binary_data) <- paste0("Person_1", 1:N)
pheatmap(binary_data)

bin_cl <- gaussianMixtureModel(binary_data, 10000, 50, initial_labels = NULL, K_max = 50, alpha = 1, dataType = "MVN")
psm <- bin_cl %>% 
  createSimilarityMat()

pheatmap(psm, color = simColPal())
row.names(psm) <- colnames(psm) <- paste0("Person_", 1:N)
annotatedHeatmap(psm, my_labels)

pred_cl <- mcclust::maxpear(psm)$cl

annotatedHeatmap(binary_data, my_labels)
annotatedHeatmap(binary_data, pred_cl)

g_samples <- gaussianMixtureModel(X, 10000, 50, initial_labels = NULL, K_max = 50, alpha = 1, dataType = "MVN")
psm <- g_samples %>% 
  createSimilarityMat()

pheatmap(psm, color = simColPal())


cc_samples <- consensusClustering(my_data, S, R)

cm <- cc_samples %>% 
  createSimilarityMat()

pheatmap(cm, color = simColPal())

row.names(cm) <- paste0("Person", 1:N)
colnames(cm) <- paste0("Person", 1:N)

annotatedHeatmap(cm, x$cluster_IDs, col_pal = simColPal())
c_star <- maxpear(cm)$cl
annotatedHeatmap(cm, c_star, col_pal = simColPal())
arandi(c_star, x$cluster_IDs)
