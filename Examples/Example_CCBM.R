
library(mdiHelpR)
library(magrittr)
library(doFuture) # install.packages("doFuture")
library(pheatmap)
library(mcclust)
library(ccbm)

generateBinaryData <- function(N, P, K) {
  my_labels <- sample(1:K, N, replace = T)
  binary_data <- matrix(0, nrow = N, ncol = P)
  phis <- seq(from = 0, to = 1, by = 1 / (K - 1))
  for (p in 1:P) {
    k_phis <- sample(phis)
    for (n in 1:N) {
      binary_data[n, p] <- sample(0:1, 1, prob = c(1 - k_phis[my_labels[n]], k_phis[my_labels[n]]))
    }
  }
  row.names(binary_data) <- paste0("Person_1", 1:N)
  list(data = binary_data, labels = my_labels)
}

set.seed(1)
setMyTheme()

registerDoFuture()
plan(multiprocess)

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
  cluster_sd = cluster_sd
)

x2 <- generateSimulationDataset(K, N, P,
  delta_mu = delta_mu,
  cluster_sd = 3
)

annotatedHeatmap(x$data, x$cluster_IDs)

X <- my_data <- as.matrix(scale(x$data))
X2 <- as.matrix(scale(x2$data))

library(ggplot2)

# my_mclust <- bayesMclust(X, 2000, 50, K = 3:15)
#
# my_mclust$bestK
#
# my_mclust$BIC %>%
#   ggplot(aes(x = K, y = BIC, colour = K)) +
#   geom_boxplot()
#
# psmK7 <- my_mclust$sample_list[[4]]$samples %>%
#   createSimilarityMat()
#
# psmK6 <- my_mclust$sample_list[[3]]$samples %>%
#   createSimilarityMat()
#
# psmK8 <- my_mclust$sample_list[[5]]$samples %>%
#   createSimilarityMat()
#
# pheatmap(psmK6, color = simColPal())
# pheatmap(psmK7, color = simColPal())
# pheatmap(psmK8, color = simColPal())
#
# mclust_samples <- lapply(3:20, function(x){
#   bayesMclust(X, 200, 1, K_max = x)
# }
# )
#
# my_bic_df <- data.frame(K = rep(3:20, each = 200/1), BIC = 0, R = rep(1:200, by = 17))
# for(i in 1:17){
#   my_bic_df$BIC[((i-1)*200 + 1):(i*200)] <- mclust_samples[[i]]$BIC
# }
#
#
# my_bic_df %>%
#   ggplot(aes(x = R, y = BIC, colour = factor(K))) +
#   geom_line()
#
# my_bic_df %>%
#   ggplot(aes(x = K, y = BIC, colour = factor(K))) +
#   geom_boxplot()
#
# mclust_samples$BIC
# psm <- mclust_samples$samples %>%
#   createSimilarityMat()
#
# pheatmap(psm, color = simColPal())
#
# my_labels <- sample(1:K, N, replace = T)

# Easy clustering ==============================================================

gaussian_cl <- mixtureModel(X, 200, 5,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 0,
  seed = 1
)

psm <- gaussian_cl$samples[-c(1:10), ] %>%
  createSimilarityMat()

pheatmap(psm)

cl_star <- maxpear(psm)$cl
annotatedHeatmap(X, cl_star)
arandi(cl_star, x$cluster_IDs)

mvn_samples <- mixtureModel(X, 200, 5,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 1,
  seed = 1
)

psm2 <- mvn_samples$samples %>%
  createSimilarityMat()

pheatmap(psm2)

mvn_cl <- maxpear(psm2)$cl
annotatedHeatmap(X, mvn_cl)
arandi(mvn_cl, x$cluster_IDs)

tagm_samples <- mixtureModel(X, 200, 5,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 3,
  seed = 1
)

psm3 <- tagm_samples$samples %>%
  createSimilarityMat()

pheatmap(psm3)

tagm_cl <- maxpear(psm3)$cl
annotatedHeatmap(X, tagm_cl)
arandi(tagm_cl, x$cluster_IDs)

tagmInd_samples <- mixtureModel(X, 200, 5,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 4,
  seed = 1
)

psm4 <- tagmInd_samples$samples %>%
  createSimilarityMat()

pheatmap(psm4)

tagmInd_cl <- maxpear(psm4)$cl
annotatedHeatmap(X, tagmInd_cl)
arandi(tagmInd_cl, x$cluster_IDs)

# # Difficult clustering =========================================================
#
# mvn_samples_hard <- mixtureModel(X2, 2000, 50,
#                                     initial_labels = NULL,
#                                     K_max = 50,
#                                     alpha = 1,
#                                     dataType = 1)
#
# psm2_2 <- mvn_samples_hard$samples %>%
#   createSimilarityMat()
#
# pheatmap(psm2_2)
#
# mvn_cl <- maxpear(psm2_2)$cl
# annotatedHeatmap(X2, mvn_cl)
# arandi(mvn_cl, x2$cluster_IDs)
#
# tagm_samples_hard <- mixtureModel(X2, 2000, 50,
#                                      initial_labels = NULL,
#                                      K_max = 50,
#                                      alpha = 1,
#                                      dataType = 3)
#
# psm3_2 <- tagm_samples_hard$samples %>%
#   createSimilarityMat()
#
# pheatmap(psm3_2)
#
# tagm_cl <- maxpear(psm3_2)$cl
# annotatedHeatmap(X2, tagm_cl)
# arandi(tagm_cl, x$cluster_IDs)

# Binary data ==================================================================


binary_data <- generateBinaryData(N, P, K)

pheatmap(binary_data$data)

gaussian_cl <- mixtureModel(binary_data$data, 100, 5,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 0
)


bin_samples <- mixtureModel(binary_data$data, 500, 5,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 2,
  seed = 1
)

psm_bin <- bin_cl$samples %>%
  createSimilarityMat()

row.names(psm_bin) <- colnames(psm_bin) <- row.names(X)


bin_cl <- maxpear(psm_bin)$cl

pheatmap(psm_bin, color = simColPal())
annotatedHeatmap(binary_data$data, bin_cl)
annotatedHeatmap(psm_bin, binary_data$labels, col_pal = simColPal())
arandi(bin_cl, binary_data$labels)

# === MVN Test =================================================================

mvn_test <- generateSimulationDataset(K = 2, n = 100, p = 2,
                               delta_mu = 3,
                               cluster_sd = 1
)

annotatedHeatmap(mvn_test$data, mvn_test$cluster_IDs)

mvn_test_samples <- mixtureModel(mvn_test$data, 5000, 50,
                             K_max = 2,
                             dataType = 1,
                             seed = 1
)

psm_mvn_test <- mvn_test_samples$samples[-c(1:20), ] %>%
  createSimilarityMat() %>% 
  set_rownames(row.names(mvn_test$data)) %>% 
  set_colnames(row.names(mvn_test$data))

cl_mvn_test <- maxpear(psm_mvn_test)$cl

pheatmap(psm_mvn_test, color = simColPal())
annotatedHeatmap(mvn_test$data, cl_mvn_test)
annotatedHeatmap(psm_mvn_test, mvn_test$cluster_IDs, col_pal = simColPal())

# === Semi-supervised ==========================================================

K_max <- 8
# initial_labels <- sample(1:K_max, size = N, replace = T)
initial_labels <- priorLabels(1, K_max, nrow(X)) - 1
fixed <- sample(0:1, size = N, replace = T, prob = c(0.8, 0.2))
initial_labels[which(fixed == 1)] <- x$cluster_IDs[which(fixed == 1)]

my_samples <- semisupervisedMixtureModel(X, 200, 5, initial_labels - 1, fixed,
  K = K_max,
  dataType = 1,
  seed = 1
)

my_samples_2 <- mixtureModel(X, 200, 5,
  initial_labels = initial_labels - 1,
  K = K_max,
  dataType = 1,
  seed = 1
)

psm_semi <- my_samples$samples[-c(1:10), ] %>%
  createSimilarityMat() %>% 
  set_rownames(row.names(X)) %>% 
  set_colnames(row.names(X))

psm_un <- my_samples_2$samples[-c(1:10), ] %>%
  createSimilarityMat() %>% 
  set_rownames(row.names(X)) %>% 
  set_colnames(row.names(X))

cl_semi <- maxpear(psm_semi)$cl
cl_un <- maxpear(psm_un)$cl

pheatmap(psm_semi, color = simColPal())
annotatedHeatmap(X, cl_semi)
annotatedHeatmap(psm_semi, x$cluster_IDs, col_pal = simColPal())

pheatmap(psm_un, color = simColPal())
annotatedHeatmap(X, cl_un)
annotatedHeatmap(psm_un, x$cluster_IDs, col_pal = simColPal())

arandi(cl_semi, x$cluster_IDs)
arandi(cl_un, x$cluster_IDs)


# my_samples$BIC %>% boxplot()
#
# my_samples$samples[, 1:25]
#
# means <- matrix(0, nrow = P, ncol = 5)
# for(k in 1:5){
#   means[, k] <- X[x$cluster_IDs == k,] %>% colMeans() %>% round(3)
# }
# means

# === Consensus clustering =====================================================

bin_cc <- consensusClustering(binary_data$data, 500, 100,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 2
)
cm <- bin_cc %>%
  createSimilarityMat()
pheatmap(cm, color = simColPal())

bin_cc_2 <- consensusClustering(binary_data$data, 5000, 1,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = 2
)
cm_2 <- bin_cc_2 %>%
  createSimilarityMat()
pheatmap(cm_2, color = simColPal())


row.names(psm) <- colnames(psm) <- paste0("Person_", 1:N)
row.names(cm) <- colnames(cm) <- paste0("Person_", 1:N)
row.names(cm_2) <- colnames(cm_2) <- paste0("Person_", 1:N)

annotatedHeatmap(psm, binary_data$labels, col_pal = simColPal())
annotatedHeatmap(cm, binary_data$labels, col_pal = simColPal())
annotatedHeatmap(cm_2, binary_data$labels, col_pal = simColPal())

library(mcclust)
bayes_cl <- maxpear(psm, max.k = 100)$cl
cc_cl <- maxpear(cm, max.k = 100)$cl
cc_cl_2 <- maxpear(cm_2, max.k = 12)$cl
arandi(binary_data$labels, bayes_cl)
arandi(binary_data$labels, cc_cl)
arandi(binary_data$labels, cc_cl_2)

annotatedHeatmap(psm, bayes_cl, col_pal = simColPal())
annotatedHeatmap(cm, cc_cl, col_pal = simColPal())
annotatedHeatmap(cm_2, cc_cl_2, col_pal = simColPal())

pred_cl <- mcclust::maxpear(psm)$cl

annotatedHeatmap(binary_data, my_labels)
annotatedHeatmap(binary_data, pred_cl)

g_samples <- mixtureModel(X, 10, 1, initial_labels = NULL, K_max = 50, alpha = 1, dataType = 2)
psm <- g_samples %>%
  createSimilarityMat()

pheatmap(psm, color = simColPal())
row.names(psm) <- colnames(psm) <- paste0("Person_", 1:N)
annotatedHeatmap(psm, x$cluster_IDs)


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
