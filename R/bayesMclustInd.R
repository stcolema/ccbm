#!/usr/bin/Rscript
#' @title Bayesian mclust
#' @description A Bayesian mixture model initialised using ``hclust`` and a cut of ``K_max`` clusters.
#' Named after the ``mclust::mclust`` function as it follows a similar logic.
#' @param X Data to cluster as a matrix (items to cluster in rows).
#' @param R The number of iterations in the sampler.
#' @param thin The factor by which the samples generated are thinned, e.g. ig ``thin=50`` only every 50th sample is kept.
#' @param K_max The number of components to include (the upper bound on the number of clusters found).
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
#' @param method The linkage used in ``hclust``.
#' @param metric The measure of distance used in creating a distance matrix for ``hclust``.
#' @return Named list of the matrix of MCMC samples generated (each row
#' corresponds to a different sample) and BIC for each saved iteration.
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
#' samples <- bayesmclust(X, R, thin, K_max = K_max)
#'
#' # Predicted clustering and PSM
#' pred_cl <- mcclust::maxpear(samples)
#' psm <- createSimilarityMatrix(pred_cl)
bayesMclustInd <- function(X, R, thin,
                           K_max = 50,
                           alpha = 1,
                           dataType = "G",
                           method = "complete",
                           metric = "euclidean",
                           seed = sample.int(.Machine$integer.max, 1)) {

  # Initialise labels using hclust
  hc <- hclust(dist(X, method = metric), method = method)
  intial_labels <- cutree(hc, k = K_max)

  samples <- gaussianMixtureModel(X, R, thin, intial_labels, K_max, alpha, dataType, seed)

  samples
}
