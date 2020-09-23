# include <RcppArmadillo.h>
// # include "CommonFunctions.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;

//' @title Create Similarity Matrix
//' @description Constructs a similarity matrix comparing all points clustering across the 
//' iterations.
//' @param samples Matrix of label assignment for data across iterations.
//' @return A symmetric n x n matrix (for n rows in cluster record) describing 
//' the fraction of samples for which each pairwise combination of points are
//' assigned the same label.
//' @export
// [[Rcpp::export]]
arma::mat createSimilarityMat(arma::umat samples){
  
  double entry = 0.0;                           // Hold current value
  arma::uword N = samples.n_cols;        // Number of items clustered
  arma::uword R = samples.n_rows;   // Number of MCMC samples taken
  arma::mat out = arma::ones<arma::mat>(N, N);  // Output similarity matrix 
  
  // Compare every entry to every other entry. As symmetric and diagonal is I
  // do not need to compare points with self and only need to calculate (i, j) 
  // entry
  for (arma::uword i = 0; i < N - 1; i++){ 
    for (arma::uword j = i + 1; j < N; j++){
      entry = (double)sum(samples.col(i) == samples.col(j)) / R ;
      out(i, j) = entry;
      out(j, i) = entry;
    }
  }
  return out;
}
