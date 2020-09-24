# include <RcppArmadillo.h>
# include <math.h> 
# include <string>
#include "RcppThread.h"
// #include <iostream>
// # include "CommonFunctions.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp ;

//' @title The Beta Distribution
//' @description Random generation from the Beta distribution. 
//' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
//' Samples from a Beta distribution based using two independent gamma 
//' distributions. 
//' @param a Shape parameter.
//' @param b Shape parameter.
//' @return Sample from Beta(a, b).
double rBeta(double a, double b) { // double theta = 1.0) {
  double X = arma::randg( arma::distr_param(a, 1.0) );
  double Y = arma::randg( arma::distr_param(b, 1.0) );
  double beta = X / (double)(X + Y);
  return(beta);
}

//' @title The Beta Distribution
//' @description Random generation from the Beta distribution. 
//' See https://en.wikipedia.org/wiki/Beta_distribution#Related_distributions.
//' Samples from a Beta distribution based using two independent gamma 
//' distributions. 
//' @param n The number of samples to draw.
//' @param a Shape parameter.
//' @param b Shape parameter.
//' @return Sample from Beta(a, b).
arma::vec rBeta(arma::uword n, double a, double b) {
  arma::vec X = arma::randg(n, arma::distr_param(a, 1.0) );
  arma::vec Y = arma::randg(n, arma::distr_param(b, 1.0) );
  arma::vec beta = X / (X + Y);
  return(beta);
}

//' @title Calculate sample covariance
//' @description Returns the unnormalised sample covariance. Required as 
//' arma::cov() does not work for singletons.
//' @param data Data in matrix format
//' @param sample_mean Sample mean for data
//' @param n The number of samples in data
//' @param n_col The number of columns in data
//' @return One of the parameters required to calculate the posterior of the
//'  Multivariate normal with uknown mean and covariance (the unnormalised 
//'  sample covariance).
arma::mat calcSampleCov(arma::mat data,
                        arma::vec sample_mean,
                        arma::uword N,
                        arma::uword P
) {
  
  arma::mat sample_covariance = arma::zeros<arma::mat>(P, P);
  // sample_covariance.zeros();
  
  // If n > 0 (as this would crash for empty clusters)
  if(N > 0){
    data.each_row() -= arma::trans(sample_mean);
    sample_covariance = arma::trans(data) * data;
  }
  return sample_covariance;
}


class sampler {
  
private:
  double logLikelihood(arma::vec x) { return 0.0; }
  
public:
  arma::uword K, N, P, K_occ;
  double model_likelihood = 0.0, BIC = 0.0;
  arma::uvec labels, N_k;
  arma::vec concentration, w, ll, likelihood;
  arma::umat members;
  arma::mat X, alloc;
  
  // Parametrised class
  sampler(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X) 
  {
    
    K = _K;
    labels = _labels;
    concentration = _concentration;
    X = _X;
    
    // Dimensions
    N = X.n_rows;
    P = X.n_cols;
    
    // std::cout << "\nN: " << N << "\nP: " << P << "\n\n";
    
    // Class populations
    N_k = arma::zeros<arma::uvec>(_K);
    
    // Weights
    // double x, y;
    w = arma::zeros<arma::vec>(_K);
    
    // for(arma::uword k = 0; k < K; k++){
    //   x = arma::randg(1, concentration(k));
    //   y = arma::randg(1, concentration(k));
    //   w(k) = (1 - sum(w)) * x/(x + y);
    // }
    
    // Log likelihood (individual and model)
    ll = arma::zeros<arma::vec>(_K);
    likelihood = arma::zeros<arma::vec>(N);
    
    // Class members
    members.set_size(N, _K);
    members.zeros();
    
    // Allocation probability matrix (only makes sense in predictive models)
    alloc.set_size(N, _K);
    alloc.zeros();
  };
  
  // Functions required of all mixture models
  void updateWeights(){
    
    double a = 0.0;
    
    for (arma::uword k = 0; k < K; k++) {
      
      // Find how many labels have the value
      members.col(k) = labels == k;
      N_k(k) = arma::sum(members.col(k));
      
      // Update weights by sampling from a Gamma distribution
      a  = concentration(k) + N_k(k);
      w(k) = arma::randg( arma::distr_param(a, 1.0) );
    }
    
    // Convert the cluster weights (previously gamma distributed) to Beta 
    // distributed by normalising
    w = w / arma::sum(w);
  };
  
  void updateAllocation() {

    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);

    for(arma::uword n = 0; n < N; n++){

      ll = logLikelihood(X.row(n).t());

      // Update with weights
      comp_prob = ll + log(w);

      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);

      // Prediction and update
      u = arma::randu<double>( );
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();

      // Record the likelihood of the item in it's allocated component
      likelihood(n) = ll(labels(n));
    }

    // The model log likelihood
    model_likelihood = arma::accu(likelihood);

    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
  // void predictLabel(arma::uword n, arma::vec ll) {
  //   
  //   arma::uword u = 0;
  //   arma::vec comp_prob = ll + log(w);
  //     
  //   // Normalise and overflow
  //   comp_prob = exp(comp_prob - max(comp_prob));
  //   comp_prob = comp_prob / sum(comp_prob);
  //   
  //   // Prediction and update
  //   u = arma::randu<double>( );
  //   labels(n) = sum(u > cumsum(comp_prob));
  //   alloc.row(n) = comp_prob.t();
  //   
  // };
    
  void sampleFromPriors() {};
  void sampleParameters(){};
  void calcBIC(){};
  
};

class gaussianSampler: public sampler {
  
public:
  
  double xi, kappa, alpha, g, h, a;
  arma::vec beta;
  arma::mat mu, tau;
  
  using sampler::sampler;
  
  gaussianSampler(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K, _labels, _concentration, _X) {
    
    double data_range_inv = pow(1.0 / (X.max() - X.min()), 2);
    
    alpha = 2.0;
    g = 0.2;
    a = 10;
    
    xi = arma::accu(X)/(N * P);
    
    kappa = data_range_inv;
    h = a * data_range_inv;
    
    beta.set_size(P);
    beta.zeros(); 
    
    mu.set_size(P, K);
    mu.zeros();
    
    tau.set_size(P, K);
    tau.zeros();
    
  }
  
  gaussianSampler(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X,
    double _xi,
    double _kappa,
    double _alpha,
    double _g,
    double _h,
    double _a
  ) : sampler {_K, _labels, _concentration, _X} {
    
    xi = _xi;
    kappa = _kappa;
    alpha = _alpha;
    g = _g;
    h = _h;
    a = _a;
    
    beta.set_size(P);
    beta.zeros(); 
    
    mu.set_size(P, K);
    mu.zeros();
    
    tau.set_size(P, K);
    tau.zeros();
    
  }
  
  
  // parallelFor(0, x.size(), [&x] (unsigned int i) {x[i] = i;});
  
  // Parameters for the mixture model
  void sampleFromPriors() {
    for(arma::uword p = 0; p < P; p++){
      beta(p) = arma::randg<double>( arma::distr_param(g, 1.0 / h) );
      for(arma::uword k = 0; k < K; k++){
        mu(p, k) = (arma::randn<double>() / kappa) + xi;
        tau(p, k) = arma::randg<double>( arma::distr_param(alpha, 1.0 / arma::as_scalar(beta(p))) );
      }
      
    }
  }
  
  
  // Sample beta
  void updateBeta(){
    
    double a = g + K * alpha;
    double b = 0.0;
    
    // parallelFor(0, P, [&beta] (arma::uword p) {
    //   double b = h + arma::accu(tau.row(p));
    //   beta(p) = arma::randg<double>( arma::distr_param(a, 1.0 / b) );
    // });
    for(arma::uword p = 0; p < P; p++){
      b = h + arma::accu(tau.row(p));
      beta(p) = arma::randg<double>( arma::distr_param(a, 1.0 / b) );
    }
  }
  
  // Sample mu
  void updateMuTau() {
    
    arma::uword n_k = 0;
    double _sd = 0, _mean = 0;
    
    double a, b;
    arma::vec mu_k(P);
    // arma::vec b_star;
    
    for (arma::uword k = 0; k < K; k++) {
      
      // Find how many labels have the value
      n_k = N_k(k);
      if(n_k > 0){
        
        arma::mat component_data = X.rows( arma::find(members.col(k) == 1) );
        
        for (arma::uword p = 0; p < P; p++){
          
          // The updated parameters for mu
          _sd = 1.0/(tau(p, k) * n_k + kappa);
          _mean = (tau(p, k) * arma::sum(component_data.col(p)) + kappa * xi) / (1.0/_sd) ;
          
          
          // Sample a new value
          mu(p, k) = arma::randn<double>() * _sd + _mean;
          
          // Parameters of the distribution for tau
          a = alpha + 0.5 * n_k;
          
          arma::vec b_star = component_data.col(p) - mu(p, k);
          b = beta(p) + 0.5 * arma::accu(b_star % b_star);
          
          // The updated parameters
          tau(p, k) = arma::randg<double>(arma::distr_param(a, 1.0 / b) );
        }
      } else {
        for (arma::uword p = 0; p < P; p++){
          // Sample a new value from the priors
          mu(p, k) = arma::randn<double>() * (1.0/kappa) + xi;
          tau(p, k) = arma::randg<double>(arma::distr_param(alpha, 1.0 / beta(p)) );
        }
      }
    }
  };
  
  void sampleParameters() {
    updateMuTau();
    updateBeta();
  }
  
  arma::vec logLikelihood(arma::vec item) {
    
    arma::vec ll(K);
    ll.zeros();
    
    for(arma::uword k = 0; k < K; k++){
      for (arma::uword p = 0; p < P; p++){
        ll(k) += -0.5*(std::log(2) + std::log(PI) - std::log(arma::as_scalar(tau(p, k)))) - arma::as_scalar(0.5 * tau(p, k) * pow(item(p) - mu(p, k), 2)); 
      }
    }
    return ll;
  };
  
  void calcBIC(){
    
    arma::uword n_param = (P + P) * K_occ;
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  }
  
  void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    for(arma::uword n = 0; n < N; n++){
      
      ll = logLikelihood(X.row(n).t());
      
      // Update with weights
      comp_prob = ll + log(w);
      
      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Prediction and update
      u = arma::randu<double>( );
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      
      // Record the likelihood of the item in it's allocated component
      likelihood(n) = ll(labels(n));
    }
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
  // void updateAllocation() {
  //   
  //   // double u = 0.0;
  //   arma::uvec uniqueK;
  //   arma::vec comp_prob(K);
  //   
  //   for(arma::uword n = 0; n < N; n++){
  //     
  //     ll = logLikelihood(X.row(n).t());
  //     
  //     // Predict the new label for the current item
  //     predictLabel(n, ll);
  //     
  //     // Record the likelihood of the item in it's allocated component
  //     likelihood(n) = ll(labels(n));
  //   }
  //   
  //   // std::cout << "Model likeihood\n";
  //   
  //   // The model log likelihood
  //   model_likelihood = arma::accu(likelihood);
  //   
  //   // std::cout << "Unique K\n";
  //   
  //   // Number of occupied components (used in BIC calculation)
  //   uniqueK = arma::unique(labels);
  //   K_occ = uniqueK.n_elem;
  // };
  
  
  // void updateAllocation() {
  //   
  //   double u = 0.0;
  //   arma::vec ll(K);
  //   
  //   for(arma::uword n = 0; n < N; n++){
  //     
  //     ll = logLikelihood(X.row(n).t());
  //     
  //     // Update with weights
  //     ll = ll + log(w);
  //     
  //     // Normalise and overflow
  //     ll = exp(ll - max(ll));
  //     ll = ll / sum(ll);
  //     
  //     // Prediction and update
  //     u = arma::randu<double>( );
  //     labels(n) = sum(u > cumsum(ll));
  //     // alloc.row(n) = ll.t();
  //   }
  // };
};

class mvnSampler: public sampler {
  
public:
  double kappa, nu;
  arma::vec xi;
  arma::mat scale, mu;
  arma::cube cov;
  
  using sampler::sampler;
  
  mvnSampler(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K, _labels, _concentration, _X) {
    
    // Default values for hyperparameters
    kappa = 0.01;
    nu = P + 2;
    
    arma::mat mean_mat = arma::mean(_X, 0).t();
    xi = mean_mat.col(0);
    
    arma::mat scale_param = _X.each_row() - xi.t();
    arma::rowvec diag_entries = arma::sum(scale_param % scale_param, 0) / N * pow(_K, 1.0 / (double)P);
    scale = arma::diagmat( diag_entries );
    
    // Set the size of the objects to hold the component specific parameters
    mu.set_size(P, K);
    mu.zeros();
    
    cov.set_size(P, P, K);
    cov.zeros();
  }
  
  void sampleFromPriors() {
    for(arma::uword k = 0; k < K; k++){
      cov.slice(k) = arma::iwishrnd(scale, nu);
      mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
    }
  }
  
  void sampleParameters(){
    arma::vec mu_n(P);
    
    arma::uword n_k = 0;
    arma::vec mu_k(P), sample_mean(P);
    arma::mat sample_cov(P, P), dist_from_prior(P, P), scale_n(P, P);
    
    for (arma::uword k = 0; k < K; k++) {
      
      // Find how many labels have the value
      n_k = N_k(k);
      if(n_k > 0){
        
        // Component data
        arma::mat component_data = X.rows( arma::find(members.col(k) == 1) );
        
        // Sample mean in the component data
        sample_mean = arma::mean(component_data).t();
        
        // The weighted average of the prior mean and sample mean
        mu_k = (kappa * xi + n_k * sample_mean) / (double)(kappa + N);
        
        sample_cov = calcSampleCov(component_data, sample_mean, n_k, P);
        
        // Calculate the distance of the sample mean from the prior
        dist_from_prior = (sample_mean - xi) * (sample_mean - xi).t();
        
        // Update the scale hyperparameter
        scale_n = scale + sample_cov + ((kappa * N) / (double)(kappa + N)) * dist_from_prior;
        
        cov.slice(k) = arma::iwishrnd(scale_n, nu + n_k);
        
        mu.col(k) = arma::mvnrnd(mu_k, (1.0/kappa) * cov.slice(k), 1);
        
      } else{
        
        cov.slice(k) = arma::iwishrnd(scale, nu);
        mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
        
      }
      
    }
  }
  
  arma::vec logLikelihood(arma::vec point) {
    
    double log_det = 0.0, exponent = 0.0;
    
    arma::vec ll(K), mu_k(P);
    ll.zeros();
    
    arma::vec dist_to_mean(P);
    
    arma::mat cov_k(P, P);
    
    for(arma::uword k = 0; k < K; k++){
      // Exponent in normal PDF
      mu_k = mu.col(k);
      cov_k = cov.slice(k);
      
      // The exponent part of the MVN pdf
      dist_to_mean = point - mu_k;
      exponent = arma::as_scalar(dist_to_mean.t() * arma::inv(cov_k) * dist_to_mean);
      
      // Log determinant of the variance
      log_det = arma::log_det(cov_k).real();
      
      // Normal log likelihood
      ll(k) = -0.5 *(log_det + exponent + (double) P * log(2.0 * M_PI)); 
      
    }
    
    return(ll);
  }
  
  void calcBIC(){
    
    arma::uword n_param = (P + P * (P + 1) * 0.5) * K_occ;
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  }
  
  void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    for(arma::uword n = 0; n < N; n++){
      
      ll = logLikelihood(X.row(n).t());
      
      // Update with weights
      
      std::cout << "Move yo probs\n";
      comp_prob = ll + log(w);
      
      // Normalise and overflow
      std::cout << "Normalise and handle overflow\n";
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Prediction and update
      std::cout << "Predict class\n";
      u = arma::randu<double>( );
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      
      // Record the likelihood of the item in it's allocated component
      likelihood(n) = ll(labels(n));
    }
    
    std::cout << "Model likeihood\n";
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    std::cout << "Unique K\n";
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
  // void updateAllocation() {
  //   
  //   double u = 0.0;
  //   arma::vec ll(K);
  //   
  //   for(arma::uword n = 0; n < N; n++){
  //     
  //     ll = logLikelihood(X.row(n).t());
  //     
  //     // Update with weights
  //     ll = ll + log(w);
  //     
  //     // Normalise and overflow
  //     ll = exp(ll - max(ll));
  //     ll = ll / sum(ll);
  //     
  //     // Prediction and update
  //     u = arma::randu<double>( );
  //     labels(n) = sum(u > cumsum(ll));
  //     // alloc.row(n) = ll.t();
  //   }
  // };
};


class categoricalSampler: public sampler {

public:

  double phi;
  arma::mat prob;

  using sampler::sampler;

  categoricalSampler(
    arma::uword _K,
    arma::uvec _labels,
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K, _labels, _concentration, _X) {
    phi = arma::accu(_X);

    // Probability for each class
    prob.set_size(P, _K);
    prob.zeros();
  }

  void sampleFromPriors(){
    for(arma::uword k = 0; k < K; k++){
      prob.col(k) = rBeta(P, 1 - phi, phi);
    }
  }
  
  void sampleParameters(){

    arma::uword n_k = 0;
    arma::rowvec component_column_prop;

    for(arma::uword k = 0; k < K; k++){

      // Find how many labels have the value
      n_k = N_k(k);
      if(n_k > 0){

        arma::mat component_data = X.rows( arma::find(members.col(k) == 1) );
        component_column_prop = arma::sum(component_data);

        for(arma::uword p = 0; p < P; p++){

          prob(p, k) = rBeta((1 - component_column_prop(p)) + (1 - phi), component_column_prop(p) + phi);
        }
      } else {
        prob.col(k) = rBeta(P, 1 - phi, phi);
      }
    }
  }
  
  
  arma::vec logLikelihood(arma::vec point) {
    
    arma::vec ll = arma::zeros<arma::vec>(K);
    arma::vec class_prob(P);
    
    for(arma::uword k = 0; k < K; k++) {
      
      class_prob = prob.col(k);
      
      for(arma::uword p = 0; p < P; p++) {
        ll(k) += std::log( (class_prob(p) * point(p) ) + (1 - class_prob(p)) * (1 - point(p)) );
      }
    }
  return(ll); 
  }

  
  void calcBIC(){
    
    arma::uword n_param = P * K_occ;
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  }
  
  void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    for(arma::uword n = 0; n < N; n++){
      
      ll = logLikelihood(X.row(n).t());
      
      // Update with weights
      comp_prob = ll + log(w);
      
      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Prediction and update
      u = arma::randu<double>( );
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      
      // Record the likelihood of the item in it's allocated component
      likelihood(n) = ll(labels(n));
    }
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
};

//' @title Mixture model
//' @description Performs MCMC sampling for a mixture model.
//' @param X The data matrix to perform clustering upon (items to cluster in rows).
//' @param K The number of components to model (upper limit on the number of clusters found).
//' @param labels Vector item labels to initialise from.
//' @param dataType String, "G", "MVN" or "C" for independent Gaussians, Multivariate Normal or Categorical distributions.
//' @param R The number of iterations to run for.
//' @param thin thinning factor for samples recorded.
//' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
//' @return Named list of the matrix of MCMC samples generated (each row 
//' corresponds to a different sample) and BIC for each saved iteration.
// [[Rcpp::export]]
Rcpp::List mixtureModel (
    arma::mat X,
    arma::uword K,
    arma::uvec labels,
    std::string dataType,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    arma::uword seed
) {
  
  // Set the random number
  std::default_random_engine generator(seed);
  
  // sampler _my_sampler(K, labels, concentration, X);
  
  // Declare the sampler to be used
  // if(dataType.compare("G") == 0){
  //   gaussianSampler _my_sampler(K, labels, concentration, X);
  // }

  // if(dataType.compare("MVN") == 0){
  //   mvnSampler _my_sampler(K, labels, concentration, X);
  // }
  // 
  // if(dataType.compare("C") == 0){
  //   categoricalSampler _my_sampler(K, labels, concentration, X);
  // }
  // 
  // auto my_sampler = _my_sampler;
  
  gaussianSampler my_sampler(K, labels, concentration, X);
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  
  arma::uword save_int=0;
  
  // Sampler from priors (this is unnecessary)
  my_sampler.sampleFromPriors();
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    my_sampler.updateWeights();
    my_sampler.sampleParameters();
    my_sampler.updateAllocation();
    
    // Record results
    if((r + 1) % thin == 0){
      
      my_sampler.calcBIC();
      BIC_record( save_int ) = my_sampler.BIC;
      
      class_record.row( save_int ) = my_sampler.labels.t();
      save_int++;
    }
  }
  return(List::create(Named("samples") = class_record, Named("BIC") = BIC_record));
};