#!/usr/bin/Rscript
#
# Example of consensus clustering of Bayesian mixture models applied to
# simulated data.

# Library setup
library(ccbm)
library(mdiHelpR)
library(magrittr)
library(doRNG)

# Set a seed fro reproducibility
set.seed(1)
setMyTheme()

# Plan for parallelisation using the doRNG package
registerDoRNG()
plan(multiprocess)

# Parameters of the simulated data
N <- 200
P <- 20
K <- 5
my_data <- generateSimulationDataset(K, N, P)

# Look at the data
annotatedHeatmap(my_data$data, my_data$cluster_IDs)

# Having simulated some data, we now apply consensus clustering.
D <- 100
W <- 100

# Perform consensus clustering
my_samples <- consensusClustering(my_data$data, D, W, dataType = 0)

# Create a consensus matrix
cm <- my_samples %>%
  createSimilarityMat() %>%
  set_colnames(colnames(my_samples)) %>%
  set_rownames(colnames(my_samples))

# Look at the consensus matrix annotated by the generating labels
annotatedHeatmap(cm, my_data$cluster_IDs,
  col_pal = simColPal(),
  main = "Consensus matrix"
)
