
library(mdiHelpR)
library(magrittr)
library(doFuture) # install.packages("doFuture")
library(pheatmap)
library(mcclust)
library(ccbm)

generateBinaryData <- function(N, P, K) {
  my_labels <- sample(1:K, N, replace = T)
  binary_data <- matrix(0, nrow = N, ncol = P)
  phis <- seq(from = 0, to = 1, by = 1/(K- 1))
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

annotatedHeatmap(x$data, x$cluster_IDs)

X <- my_data <- as.matrix(scale(x$data))

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

gaussian_cl <- gaussianMixtureModel(X, 50, 1,
                                    initial_labels = NULL, 
                                    K_max = 50, 
                                    alpha = 1, 
                                    dataType = 0)

psm <- gaussian_cl$samples[-c(1:10), ] %>% 
  createSimilarityMat()

pheatmap(psm)

cl_star <- maxpear(psm)$cl
annotatedHeatmap(X, cl_star)
arandi(cl_star, x$cluster_IDs)

mvn_cl <- gaussianMixtureModel(X, 200, 5,
                                    initial_labels = NULL, 
                                    K_max = 50, 
                                    alpha = 1, 
                                    dataType = 1)

psm2 <- mvn_cl$samples %>% 
  createSimilarityMat()

pheatmap(psm2)

mvn_cl <- maxpear(psm2)$cl
annotatedHeatmap(X, mvn_cl)
arandi(mvn_cl, x$cluster_IDs)


binary_data <- generateBinaryData(N, P, K)

pheatmap(binary_data$data)

gaussian_cl <- gaussianMixtureModel(binary_data$data, 100, 5,
                     initial_labels = NULL, 
                     K_max = 50, 
                     alpha = 1, 
                     dataType = 0)


bin_cl <- gaussianMixtureModel(binary_data$data, 5000, 50,
                               initial_labels = NULL, 
                               K_max = 50, 
                               alpha = 1, 
                               dataType = 2)



psm_bin <- bin_cl$samples %>%
  createSimilarityMat()

bin_cl <- maxpear(psm_bin)$cl

annotatedHeatmap(binary_data$data, bin_cl)
arandi(bin_cl, binary_data$labels)

pheatmap(psm_bin, color = simColPal())

bin_cc <- consensusClustering(binary_data$data, 500, 100, initial_labels = NULL, K_max = 50, alpha = 1, dataType = "G")
cm <- bin_cc %>%
  createSimilarityMat()
pheatmap(cm, color = simColPal())

bin_cc_2 <- consensusClustering(binary_data$data, 5000, 1, initial_labels = NULL, K_max = 50, alpha = 1, dataType = "G")
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

g_samples <- gaussianMixtureModel(X, 10, 1, initial_labels = NULL, K_max = 50, alpha = 1, dataType = 2)
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
