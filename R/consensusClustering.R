#!/usr/bin/Rscript
#' @title Consensus clustering
#' @description Consensus clustering of Gaussian mixture models.
#' @param X Data to cluster as a matrix (items to cluster in rows).
#' @param W The width of the ensemble.
#' @param D The chain depth in each learner.
#' @param initial_labels Labels to begin from (if ``NULL`` defaults to a stick-breaking prior).
#' @param K_max The number of components to include (the upper bound on the number of clusters found).
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
#' @param anchor_seed The value added to the looped over seeds (this allows some differentiation between runs).
#' @return matrix of samples generated from each learner (each row corresponds to a different sample).
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Ensemble parameters
#' D <- 100
#' W <- 100
#'
#' # Samples from each model
#' samples <- consensusClustering(X, S, R)
#'
#' # Predicted clustering and consensus matrix
#' pred_cl <- mcclust::maxpear(samples)
#' cm <- createSimilarityMatrix(pred_cl)
#' @importFrom foreach foreach
#' @export
consensusClustering <- function(X, D, W,
                                initial_labels = NULL,
                                K_max = 50,
                                alpha = 1,
                                dataType = 0,
                                anchor_seed = 0) {
  if (is.null(initial_labels)) {
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  }

  samples <- foreach::foreach(
    d = 1:D,
    .export = c("X", "initial_labels", "D", "K_max", "alpha", "dataType"),
    .packages = c("ccbm", "Rcpp")
  ) %dorng% {
    set.seed(d + anchor_seed)
    sampleMixtureModel(
      X,
      K_max,
      initial_labels,
      dataType,
      W,
      W,
      rep(alpha, K_max),
      d
    )$samples
  }

  cl_samples <- matrix(unlist(samples), nrow = D, byrow = T)
  colnames(cl_samples) <- row.names(X)
  
  cl_samples
}
