% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mixtureModel.R
\name{mixtureModel}
\alias{mixtureModel}
\title{Mixture model}
\usage{
mixtureModel(
  X,
  R,
  thin,
  initial_labels = NULL,
  K_max = 50,
  alpha = 1,
  dataType = "G",
  seed = sample.int(.Machine$integer.max, 1)
)
}
\arguments{
\item{X}{Data to cluster as a matrix (items to cluster in rows).}

\item{R}{The number of iterations in the sampler.}

\item{thin}{The factor by which the samples generated are thinned, e.g. if ``thin=50`` only every 50th sample is kept.}

\item{initial_labels}{Labels to begin from (if ``NULL`` defaults to a stick-breaking prior).}

\item{K_max}{The number of components to include (the upper bound on the number of clusters found).}

\item{alpha}{The concentration parameter for the stick-breaking prior and the weights in the model.}

\item{dataType}{The type of data to model, one of "G" for independent Gaussians, "MVN" for a multivariate normal, or "C" for categorical}

\item{seed}{The random seed for reproducibility.}
}
\value{
Named list of the matrix of MCMC samples generated (each row
corresponds to a different sample) and BIC for each saved iteration.
}
\description{
A Bayesian mixture model.
}
\examples{
# Convert data to matrix format
X <- as.matrix(my_data)

# Sampling parameters
R <- 1000
thin <- 50

# MCMC samples and BIC vector
samples <- mixtureModel(X, R, thin)

# Predicted clustering and PSM
pred_cl <- mcclust::maxpear(samples$samples)
psm <- createSimilarityMatrix(pred_cl)
}
