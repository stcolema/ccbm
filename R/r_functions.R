
#' @title Stick breaking prior
#' @description Draw weights from the stick-breaking prior.
#' @param alpha The concentration parameter.
#' @param K The number of weights to generate.
#' @return A vector of weights.
#' @examples 
#'  weights <- stickBreakingPrior(1, 50)
stickBreakingPrior <- function(alpha, K){
  v <- rbeta(K, alpha, 1)
  stick <- 1
  w <- rep(0, K)
  
  for(i in 1:K){
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
#'  initial_labels <- priorLabels(1, 50, 100)
priorLabels <- function(alpha, K, N){
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
#'   # Convert data to matrix format
#'   X <- as.matrix(my_data)
#'   
#'   # Ensemble parameters
#'   S <- 100
#'   R <- 100
#'   
#'   # Samples from each model
#'   samples <- consensusClustering(X, S, R)
#'   
#'   # Predicted clustering and consensus matrix
#'   pred_cl <- mcclust::maxpear(samples)
#'   cm <- createSimilarityMatrix(pred_cl)
consensusClustering <- function(X, S, R, initial_labels = NULL, K_max = 50, alpha = 1, dataType = "G"){
  
  if(is.null(initial_labels)){
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  }
  
  samples <- foreach(s = 1:S, .export = c("X", "initial_labels", "R", "K_max", "alpha", "dataType")) %dopar% {
    set.seed(s)
    mixtureModel (
      X,
      K_max,
      initial_labels,
      dataType,
      R,
      R,
      rep(alpha, K_max)
    )
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
#'   # Convert data to matrix format
#'   X <- as.matrix(my_data)
#'   
#'   # Sampling parameters
#'   R <- 1000
#'   thin <- 50
#'   
#'   # MCMC samples
#'   samples <- gaussianMixtureModel(X, R, thin)
#'   
#'   # Predicted clustering and PSM
#'   pred_cl <- mcclust::maxpear(samples)
#'   psm <- createSimilarityMatrix(pred_cl)
gaussianMixtureModel <- function(X, R, thin, initial_labels = NULL, K_max = 50, alpha = 1, dataType = "G"){
  
  if(is.null(initial_labels)){
    initial_labels <- priorLabels(alpha, K_max, nrow(X))
  }
  
  samples <- mixtureModel (
      X,
      K_max,
      initial_labels,
      dataType,
      R,
      thin,
      rep(alpha, K_max)
    )
  
  samples
}


bayesMclust <- function(X, R, thin, 
                        initial_labels = NULL,
                        K_max = 50, 
                        alpha = 1, 
                        dataType = "G", 
                        method = "complete", 
                        metric = "euclidean"){
  
  # Initialise labels using hclust
  hc <- hclust(dist(X, method = metric), method = method)
  intial_labels <- cutree(hc, k = K_max)
  
  samples <- gaussianMixtureModel(X, R, thin, intial_labels, K_max, alpha, dataType)
  
  samples
}

