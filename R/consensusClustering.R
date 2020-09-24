#!/usr/bin/Rscript
#' @title Consensus clustering
#' @description Consensus clustering of Gaussian mixture models.
#' @param X Data to cluster as a matrix (items to cluster in rows).
#' @param S The size of the ensemble.
#' @param R The chain depth in each learner.
#' @param initial_labels Labels to begin from (if ``NULL`` defaults to a stick-breaking prior).
#' @param K_max The number of components to include (the upper bound on the number of clusters found).
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
#' @return matrix of samples generated from each learner (each row corresponds to a different sample).
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Ensemble parameters
#' S <- 100
#' R <- 100
#'
#' # Samples from each model
#' samples <- consensusClustering(X, S, R)
#'
#' # Predicted clustering and consensus matrix
#' pred_cl <- mcclust::maxpear(samples)
#' cm <- createSimilarityMatrix(pred_cl)
#' @importFrom foreach foreach
#' @export
consensusClustering <- function(X, S, R,
                                initial_labels = NULL,
                                K_max = 50,
                                alpha = 1,
                                dataType = "G") {
  if (is.null(initial_labels)) {
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  }

  samples <- foreach::foreach(
    s = 1:S,
    .export = c("X", "initial_labels", "R", "K_max", "alpha", "dataType"),
    .packages = c("ccbm", "Rcpp")
  ) %dopar% {
    mixtureModel(
      X,
      K_max,
      initial_labels,
      dataType,
      R,
      R,
      rep(alpha, K_max),
      s
    )$samples
  }

  cl_samples <- matrix(unlist(samples), nrow = S, byrow = T)
  cl_samples
}
