#!/usr/bin/Rscript
#' @title Mixture model
#' @description A Bayesian mixture model.
#' @param X Data to cluster as a matrix (items to cluster in rows).
#' @param R The number of iterations in the sampler.
#' @param thin The factor by which the samples generated are thinned, e.g. ig ``thin=50`` only every 50th sample is kept.
#' @param initial_labels Labels to begin from (if ``NULL`` defaults to a stick-breaking prior).
#' @param K_max The number of components to include (the upper bound on the number of clusters found).
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
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
#' # MCMC samples and BIC vector
#' samples <- gaussianMixtureModel(X, R, thin)
#'
#' # Predicted clustering and PSM
#' pred_cl <- mcclust::maxpear(samples$samples)
#' psm <- createSimilarityMatrix(pred_cl)
#' @export
mixtureModel <- function(X, R, thin,
                                 initial_labels = NULL,
                                 K_max = 50,
                                 alpha = 1,
                                 dataType = "G",
                                 seed = sample.int(.Machine$integer.max, 1)) {
  if (is.null(initial_labels)) {
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  }

  samples <- sampleMixtureModel(
    X,
    K_max,
    initial_labels,
    dataType,
    R,
    thin,
    rep(alpha, K_max),
    seed
  )

  samples
}