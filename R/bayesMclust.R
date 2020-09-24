#!/usr/bin/Rscript
#' @title Bayesian Mclust
#' @description A range of Bayesian mixture model initialised using ``hclust`` with a cut for each value of the vector ``K`` given.
#' Named after the ``mclust::Mclust`` function as it follows a similar logic.
#' @param X Data to cluster as a matrix (items to cluster in rows).
#' @param R The number of iterations in the sampler.
#' @param thin The factor by which the samples generated are thinned, e.g. ig ``thin=50`` only every 50th sample is kept.
#' @param K A vector of the number of components to include (the upper bound on the number of clusters found) in each model.
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
#' @param method The linkage used in ``hclust``.
#' @param metric The measure of distance used in creating a distance matrix for ``hclust``.
#' @return Matrix of MCMC samples generated (each row corresponds to a different sample).
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Sampling parameters
#' R <- 1000
#' thin <- 50
#'
#' # Number of components
#' K_max <- 15
#'
#' # MCMC samples
#' m <- bayesMclust(X, R, thin, K_max = K_max)
#'
#' library(ggplot2)
#' library(magrittr)
#' m$BIC %>%
#'   ggplot(aes(x = K, y = BIC, colour = K)) +
#'   geom_boxplot()
#'
#' psmKbest <- m$sample_list[[m$bestModel]]$samples %>%
#'   createSimilarityMat()
#' @export
bayesMclust <- function(X, R, thin, K,
                        alpha = 1,
                        dataType = "G",
                        method = "complete",
                        metric = "euclidean") {

  # Create the list of samples and BIC vectors for each choice of K
  mclust_samples <- lapply(K, function(x) {
    bayesmclust(X, R, thin,
      K_max = x,
      alpha = alpha,
      dataType = dataType,
      method = method,
      metric = metric
    )
  })

  # The BIC vector for all models
  bic <- unlist(do.call(rbind, mclust_samples)[, 2])

  # Create a data.frame of the BIC for each K and iteration
  bic_df <- data.frame(
    K = factor(rep(K, each = floor(R / thin))),
    BIC = bic,
    R = rep(1:floor(R / thin), by = length(K))
  )

  # Find the median BIC for each K
  median_bic <- by(bic_df, bic_df$K, function(x) {
    c(median_bic = median(x$BIC))
  })

  # The best K (the actual best choice of K is the name)
  bestK <- which(median_bic == min(median_bic))

  list(
    BIC = bic_df,
    bestK = names(bestK),
    bestModel = bestK,
    sample_list = mclust_samples
  )
}
