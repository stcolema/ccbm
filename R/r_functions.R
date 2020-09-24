
#' @title Stick breaking prior
#' @description Draw weights from the stick-breaking prior.
#' @param alpha The concentration parameter.
#' @param K The number of weights to generate.
#' @return A vector of weights.
#' @examples
#' weights <- stickBreakingPrior(1, 50)
stickBreakingPrior <- function(alpha, K) {
  v <- rbeta(K, alpha, 1)
  stick <- 1
  w <- rep(0, K)

  for (i in 1:K) {
    w[i] <- v[i] * stick
    stick <- stick - w[i]
  }
  w
}

#' @title Prior labels
#' @description Generate labels from the stick-breaking prior.
#' @param alpha The concentration parameter for the stick-breaking prior.
#' @param K The number of components to include (the upper bound on the number of unique labels generated).
#' @param N The number of labels to generate.
#' @return A vector of labels.
#' @examples
#' initial_labels <- priorLabels(1, 50, 100)
priorLabels <- function(alpha, K, N) {
  w <- stickBreakingPrior(alpha, K)
  initial_labels <- sample(1:K, N, replace = T)
}

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
consensusClustering <- function(X, S, R,
                                initial_labels = NULL,
                                K_max = 50,
                                alpha = 1,
                                dataType = "G") {
  if (is.null(initial_labels)) {
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  }

  samples <- foreach(s = 1:S, .export = c("X", "initial_labels", "R", "K_max", "alpha", "dataType")) %dopar% {
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


#' @title Gaussian mixture model
#' @description A Bayesian mixture model using independent Gaussians. The priors
#' are empirical and follow the suggestions of Richardson and Green <https://doi.org/10.1111/1467-9868.00095>.
#' @param X Data to cluster as a matrix (items to cluster in rows).
#' @param R The number of iterations in the sampler.
#' @param thin The factor by which the samples generated are thinned, e.g. ig ``thin=50`` only every 50th sample is kept.
#' @param initial_labels Labels to begin from (if ``NULL`` defaults to a stick-breaking prior).
#' @param K_max The number of components to include (the upper bound on the number of clusters found).
#' @param alpha The concentration parameter for the stick-breaking prior and the weights in the model.
#' @return Matrix of MCMC samples generated (each row corresponds to a different sample).
#' @examples
#' # Convert data to matrix format
#' X <- as.matrix(my_data)
#'
#' # Sampling parameters
#' R <- 1000
#' thin <- 50
#'
#' # MCMC samples
#' samples <- gaussianMixtureModel(X, R, thin)
#'
#' # Predicted clustering and PSM
#' pred_cl <- mcclust::maxpear(samples)
#' psm <- createSimilarityMatrix(pred_cl)
gaussianMixtureModel <- function(X, R, thin, 
                                 initial_labels = NULL, 
                                 K_max = 50, 
                                 alpha = 1, 
                                 dataType = "G",
                                 seed = sample.int(.Machine$integer.max, 1)) {
  if (is.null(initial_labels)) {
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  }

  samples <- mixtureModel(
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
#' samples <- bayesmclust(X, R, thin, K_max = K_max)
#'
#' # Predicted clustering and PSM
#' pred_cl <- mcclust::maxpear(samples)
#' psm <- createSimilarityMatrix(pred_cl)
bayesmclust <- function(X, R, thin,
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
#' ggplot(aes(x = K, y = BIC, colour = K)) +
#'  geom_boxplot()
#' 
#' psmKbest <- m$sample_list[[m$bestModel]]$samples %>% 
#'  createSimilarityMat()
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
