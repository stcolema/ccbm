
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

mclust_samples <- bayesMclust(X, 10000, 20, K_max = 15)
psm <- mclust_samples %>% 
  createSimilarityMat()

pheatmap(psm, color = simColPal())

g_samples <- gaussianMixtureModel(X, 2000, 50, initial_labels = NULL, K_max = 50, alpha = 1)
psm <- g_samples %>% 
  createSimilarityMat()

pheatmap(psm, color = simColPal())


cm <- cc_samples %>% 
  createSimilarityMat()

pheatmap(cm, color = simColPal())




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
