% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{gaussianSampler}
\alias{gaussianSampler}
\title{Gaussian mixture type}
\description{
The sampler for a mixture of Gaussians, where each feature is
assumed to be independent (i.e. a multivariate Normal with a diagonal 
covariance matrix).
}
\section{Fields}{

\describe{
\item{\code{new}}{Constructor \itemize{
\item Parameter: K - the number of components to model
\item Parameter: labels - the initial clustering of the data
\item Parameter: concentration - the vector for the prior concentration of 
the Dirichlet distribution of the component weights
\item Parameter: X - the data to model
}}

\item{\code{printType}}{Print the sampler type called.}

\item{\code{updateWeights}}{Update the weights of each component based on current 
clustering.}

\item{\code{updateAllocation}}{Sample a new clustering.}

\item{\code{sampleFromPrior}}{Sample from the priors for the Gaussian density.}

\item{\code{calcBIC}}{Calculate the BIC of the model.}

\item{\code{logLikelihood}}{Calculate the likelihood of a given data point in each
component. \itemize{
\item Parameter: point - a data point.
}}
}}

