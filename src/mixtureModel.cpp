# include <RcppArmadillo.h>
# include <math.h> 
# include <string>
# include <memory>
# include "RcppThread.h"
# include <iostream>

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

//' @name sampler
//' @title Generic mixture type
//' @description The class that the specific sampler types inherit from. Used as
//' the generic mixture type.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field updateWeights Update the weights of each component based on current
//' clustering.
//' @field updateAllocation Sample a new clustering.
//' @field sampleFromPrior Virtual placeholder for sampling from the priors of
//' the specific mixtures.
//' @field calcBIC Virtual placeholder for the function that calculates the BIC
//' of specific mixture models.
//' @field logLikelihood Virtual placeholder for the function that calculates
//' the likelihood of a given point in each component of specific mixture models.
class sampler {

private:
  // virtual arma::vec logLikelihood(arma::vec x) { return 0.0; }
  // void updateAllocation() { return; }

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

  // Destructor
  virtual ~sampler() { };

  // Virtual functions are those that should actual point to the sub-class
  // version of the function.
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: NULL.\n";
  }

  // Functions required of all mixture models
  virtual void updateWeights(){

    double a = 0.0;

    for (arma::uword k = 0; k < K; k++) {

      // Find how many labels have the value
      members.col(k) = labels == k;
      N_k(k) = arma::sum(members.col(k));

      // Update weights by sampling from a Gamma distribution
      a  = concentration(k) + N_k(k);
      w(k) = arma::randg( arma::distr_param(a, 1.0) );
    }

    // std::cout << "N_k: \n" << N_k << "\n\n";
    
    // Convert the cluster weights (previously gamma distributed) to Beta
    // distributed by normalising
    w = w / arma::sum(w);
  };

  virtual void updateAllocation() {

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

  virtual void sampleFromPriors() {};
  virtual void sampleParameters(){};
  virtual void calcBIC(){};
  virtual arma::vec logLikelihood(arma::vec x) { return arma::vec(); }

};

//' @name gaussianSampler
//' @title Gaussian mixture type
//' @description The sampler for a mixture of Gaussians, where each feature is
//' assumed to be independent (i.e. a multivariate Normal with a diagonal 
//' covariance matrix).
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field updateWeights Update the weights of each component based on current 
//' clustering.
//' @field updateAllocation Sample a new clustering. 
//' @field sampleFromPrior Sample from the priors for the Gaussian density.
//' @field calcBIC Calculate the BIC of the model.
//' @field logLikelihood Calculate the likelihood of a given data point in each
//' component. \itemize{
//' \item Parameter: point - a data point.
//' }
class gaussianSampler: virtual public sampler {
  
public:
  
  double xi, kappa, alpha, g, h, a;
  arma::vec beta;
  arma::mat mu, tau;
  
  using sampler::sampler;
  
  // Parametrised
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
  
  // Destructor
  virtual ~gaussianSampler() { };
  
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: Gaussian.\n";
  }
  
  // Parameters for the mixture model. The priors are empirical and follow the
  // suggestions of Richardson and Green <https://doi.org/10.1111/1467-9868.00095>.
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
  
};

//' @name mvnSampler
//' @title Multivariate Normal mixture type
//' @description The sampler for the Multivariate Normal mixture model.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field updateWeights Update the weights of each component based on current 
//' clustering.
//' @field updateAllocation Sample a new clustering. 
//' @field sampleFromPrior Sample from the priors for the multivariate normal
//' density.
//' @field calcBIC Calculate the BIC of the model.
//' @field logLikelihood Calculate the likelihood of a given data point in each
//' component. \itemize{
//' \item Parameter: point - a data point.
//' }
class mvnSampler: virtual public sampler {
  
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
  
  
  // Destructor
  virtual ~mvnSampler() { };
  
  // Print the sampler type.
  virtual void printType() {
    std::cout << "\nType: MVN.\n";
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
    
    // std::cout << "\nComponent means:\n" <<  mu << "\n\n";
    
    // std::cout << "Members:\n" << members << "\n\n";
    
    for (arma::uword k = 0; k < K; k++) {
      
      // Find how many labels have the value
      n_k = N_k(k);
      if(n_k > 0){
        
        // Component data
        arma::mat component_data = X.rows( arma::find(members.col(k) == 1) );
        // std::cout << "\n\nComponent data:\n" << component_data << "\n\n";
        
        // Sample mean in the component data
        sample_mean = arma::mean(component_data).t();
        // std::cout << "\n\nComponent mean:\n" << sample_mean << "\n\n";
        
        // The weighted average of the prior mean and sample mean
        mu_k = (kappa * xi + n_k * sample_mean) / (double)(kappa + n_k);
        
        sample_cov = calcSampleCov(component_data, sample_mean, n_k, P);
        
        // Calculate the distance of the sample mean from the prior
        dist_from_prior = (sample_mean - xi) * (sample_mean - xi).t();
        
        // Update the scale hyperparameter
        scale_n = scale + sample_cov + ((kappa * n_k) / (double)(kappa + n_k)) * dist_from_prior;
        
        cov.slice(k) = arma::iwishrnd(scale_n, nu + n_k);
        
        mu.col(k) = arma::mvnrnd(mu_k, (1.0/(kappa + n_k)) * cov.slice(k), 1);
        
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
  
};

//' @name categoricalSampler
//' @title Categorical mixture type
//' @description The sampler for the Categorical mixture model.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field updateWeights Update the weights of each component based on current 
//' clustering.
//' @field updateAllocation Sample a new clustering. 
//' @field sampleFromPrior Sample from the priors for the Categorical density.
//' @field calcBIC Calculate the BIC of the model.
//' @field logLikelihood Calculate the likelihood of a given data point in each
//' component. \itemize{
//' \item Parameter: point - a data point.
//' }
class categoricalSampler: public sampler {

public:

  double phi = 0.0;
  arma::mat prob;

  using sampler::sampler;

  categoricalSampler(
    arma::uword _K,
    arma::uvec _labels,
    arma::vec _concentration,
    arma::mat _X
  ) : sampler(_K, _labels, _concentration, _X) {
    phi = arma::accu(_X) / (N * P);

    // Probability for each class
    prob.set_size(P, _K);
    prob.zeros();
  }
  
  // Destructor
  virtual ~categoricalSampler() { };

  // Print the sampler type.
  void printType() {
    std::cout << "\nType: Categorical.\n";
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
        component_column_prop = arma::sum(component_data) / n_k;
        
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
  
};


//' @name tAdjustedSampler
//' @title Base class for adding a t-distribution to sweep up outliers in the 
//' model.
//' @description The class that the specific TAGM types inherit from. 
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field updateOutlierWeights Updates the weight of the outlier component.
//' @field updateWeights Update the weights of each component based on current 
//' clustering, excluding the outliers.
//' @field sampleOutlier Sample is the nth item is an outlier. \itemize{
//' \item Parameter n: the index of the individual in the data matrix and 
//' allocation vector.
//' }
//' @field updateAllocation Sample a new clustering and simultaneously allocate 
//' items to the outlier distribution.
//' @field calcTdistnLikelihood Virtual placeholder for the function that calculates 
//' the likelihood of a given point in a t-distribution. \itemize{
//' \item Parameter: point - a data point.
//' }
class tAdjustedSampler : virtual public sampler {
  
private:
  
public:
  // for use in the outlier distribution
  arma::uvec outlier;
  arma::vec global_mean;
  arma::mat global_cov;
  double df = 4.0, u = 2.0, v = 10.0, b = 0.0, outlier_weight = 0.0;
  
  using sampler::sampler;
  
  tAdjustedSampler(arma::uword _K,
          arma::uvec _labels, 
          arma::vec _concentration,
          arma::mat _X
  ) : sampler(_K, _labels, _concentration, _X) {
    
    // for use in the outlier distribution
    // global_cov = 0.5 * arma::cov(X); 
    // global_mean = arma::trans(arma::mean(X, 0));
    
    outlier = arma::zeros<arma::uvec>(N);
    
    b = (double) sum(outlier);
    outlier_weight = rBeta(b + u, N + v - b);
    
  };
  
  // Destructor
  virtual ~tAdjustedSampler() { };
  
  virtual double calcTdistnLikelihood(arma::vec point) {
    
    double log_det = 0.0;
    double exponent = 0.0;
    double log_likelihood = 0.0;
    
    exponent = arma::as_scalar(
      arma::trans(point - global_mean) 
      * arma::inv(global_cov)
      * (point - global_mean)
    );
    
    log_det = arma::log_det(global_cov).real();
    
    log_likelihood = lgamma((df + P) / 2.0) 
      - lgamma(df / 2.0) 
      - (double)P / 2.0 * log(df * M_PI) 
      - 0.5 * log_det 
      - ((df + (double) P) / 2.0) * log(1.0 + (1.0 / P) * exponent);
      
      return log_likelihood;
  };
  
  void updateOutlierWeights(){
    b = (double) sum(outlier);
    outlier_weight = rBeta(b + u, N + v - b);
  };
  
  void updateWeights(){
    
    double a = 0.0;
    
    for (arma::uword k = 0; k < K; k++) {
      
      // Find how many labels have the value
      members.col(k) = (labels == k) % outlier;
      N_k(k) = arma::sum(members.col(k));
      
      // Update weights by sampling from a Gamma distribution
      a  = concentration(k) + N_k(k);
      w(k) = arma::randg( arma::distr_param(a, 1.0) );
    }
    
    // Convert the cluster weights (previously gamma distributed) to Beta 
    // distributed by normalising
    w = w / arma::sum(w);
  };
  
  arma::uword sampleOutlier(int n) {
    
    arma::vec point = X.row(n).t();
    // arma::uword k = labels(n);
    
    double out_likelihood = 0.0;
    arma::vec outlier_prob(2);
    outlier_prob.zeros();
    
    // The likelihood of the point in the current cluster
    outlier_prob(0) = likelihood(n) + log(1 - outlier_weight);
    
    // Calculate outlier likelihood
    out_likelihood = calcTdistnLikelihood(point);
    out_likelihood += log(out_likelihood);
    outlier_prob(1) = out_likelihood;
    
    // Normalise and overflow
    outlier_prob = exp(outlier_prob - max(outlier_prob));
    outlier_prob = outlier_prob / sum(outlier_prob);
    
    // Prediction and update
    u = arma::randu<double>( );
    return sum(u > cumsum(outlier_prob));
  };
  
  virtual void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    // First update the outlier parameters
    updateOutlierWeights();
    
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
      
      // Update if the point is an outlier or not
      outlier(n) = sampleOutlier(n);
    }
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
  // void sampleParameters(){
  //   arma::vec mu_n(P);
  //   
  //   arma::uword n_k = 0;
  //   arma::vec mu_k(P), sample_mean(P);
  //   arma::mat sample_cov(P, P), dist_from_prior(P, P), scale_n(P, P);
  //   
  //   for (arma::uword k = 0; k < K; k++) {
  //     
  //     // Find how many labels have the value
  //     n_k = N_k(k);
  //     if(n_k > 0){
  //       
  //       // Component data
  //       arma::mat component_data = X.rows( arma::find(members.col(k) == 1) );
  //       
  //       // Sample mean in the component data
  //       sample_mean = arma::mean(component_data).t();
  //       
  //       // The weighted average of the prior mean and sample mean
  //       mu_k = (kappa * xi + n_k * sample_mean) / (double)(kappa + N);
  //       
  //       sample_cov = calcSampleCov(component_data, sample_mean, n_k, P);
  //       
  //       // Calculate the distance of the sample mean from the prior
  //       dist_from_prior = (sample_mean - xi) * (sample_mean - xi).t();
  //       
  //       // Update the scale hyperparameter
  //       scale_n = scale + sample_cov + ((kappa * N) / (double)(kappa + N)) * dist_from_prior;
  //       
  //       cov.slice(k) = arma::iwishrnd(scale_n, nu + n_k);
  //       
  //       mu.col(k) = arma::mvnrnd(mu_k, (1.0/kappa) * cov.slice(k), 1);
  //       
  //     } else{
  //       
  //       cov.slice(k) = arma::iwishrnd(scale, nu);
  //       mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
  //       
  //     }
  //     
  //   }
  // }
  
  void calcBIC(){
    
    arma::uword n_param = (P + P * (P + 1) * 0.5) * (K_occ + 1);
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  }
  
};

//' @name tagmMVN
//' @title T-ADjusted Gaussian Mixture (TAGM) type
//' @description The sampler for the TAGM mixture model.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field calcBIC Calculate the BIC of the model.
//' @field calcTdistnLikelihood Calculate the likelihood of a given data point 
//' the gloabl t-distirbution. \itemize{
//' \item Parameter: point - a data point.
//' }
class tagmMVN : public tAdjustedSampler, public mvnSampler {
  
private:
  
public:
  // for use in the outlier distribution
  // arma::uvec outlier;
  // arma::vec global_mean;
  // arma::mat global_cov;
  // double df = 4.0, u = 2.0, v = 10.0, b = 0.0, outlier_weight = 0.0;
  
  using tAdjustedSampler::tAdjustedSampler;
  
  tagmMVN(arma::uword _K,
          arma::uvec _labels, 
          arma::vec _concentration,
          arma::mat _X
  ) : tAdjustedSampler(_K, _labels, _concentration, _X),
      mvnSampler(_K, _labels, _concentration, _X),
      sampler(_K, _labels, _concentration, _X){
    
    // for use in the outlier distribution
    global_cov = 0.5 * arma::cov(X); 
    global_mean = arma::trans(arma::mean(X, 0));
    
    outlier = arma::zeros<arma::uvec>(N);
    
    b = (double) sum(outlier);
    outlier_weight = rBeta(b + u, N + v - b);
    
  };
  
  // Destructor
  virtual ~tagmMVN() { };
  
  void printType() {
    std::cout << "Type: TAGM.\n";
  }
  
  double calcTdistnLikelihood(arma::vec point) {
    
    double log_det = 0.0;
    double exponent = 0.0;
    double log_likelihood = 0.0;
    
    exponent = arma::as_scalar(
      arma::trans(point - global_mean) 
      * arma::inv(global_cov)
      * (point - global_mean)
    );
    
    log_det = arma::log_det(global_cov).real();
    
    log_likelihood = lgamma((df + P) / 2.0) 
      - lgamma(df / 2.0) 
      - (double)P / 2.0 * log(df * M_PI) 
      - 0.5 * log_det 
      - ((df + (double) P) / 2.0) * log(1.0 + (1.0 / P) * exponent);
      
      return log_likelihood;
  };
  
  // void updateOutlierWeights(){
  //   b = (double) sum(outlier);
  //   outlier_weight = rBeta(b + u, N + v - b);
  // };
  // 
  // void updateWeights(){
  //   
  //   double a = 0.0;
  //   
  //   for (arma::uword k = 0; k < K; k++) {
  //     
  //     // Find how many labels have the value
  //     members.col(k) = (labels == k) % outlier;
  //     N_k(k) = arma::sum(members.col(k));
  //     
  //     // Update weights by sampling from a Gamma distribution
  //     a  = concentration(k) + N_k(k);
  //     w(k) = arma::randg( arma::distr_param(a, 1.0) );
  //   }
  //   
  //   // Convert the cluster weights (previously gamma distributed) to Beta 
  //   // distributed by normalising
  //   w = w / arma::sum(w);
  // };
  // 
  // arma::uword sampleOutlier(int n) {
  //   
  //   arma::vec point = X.row(n).t();
  //   // arma::uword k = labels(n);
  //   
  //   double out_likelihood = 0.0;
  //   arma::vec outlier_prob(2);
  //   outlier_prob.zeros();
  //   
  //   // The likelihood of the point in the current cluster
  //   outlier_prob(0) = likelihood(n) + log(1 - outlier_weight);
  //   
  //   // Calculate outlier likelihood
  //   out_likelihood = calcTdistnLikelihood(point);
  //   out_likelihood += log(out_likelihood);
  //   outlier_prob(1) = out_likelihood;
  //   
  //   // Normalise and overflow
  //   outlier_prob = exp(outlier_prob - max(outlier_prob));
  //   outlier_prob = outlier_prob / sum(outlier_prob);
  //   
  //   // Prediction and update
  //   u = arma::randu<double>( );
  //   return sum(u > cumsum(outlier_prob));
  // };
  // 
  // void updateAllocation() {
  //   
  //   double u = 0.0;
  //   arma::uvec uniqueK;
  //   arma::vec comp_prob(K);
  //   
  //   // First update the outlier parameters
  //   updateOutlierWeights();
  //   
  //   for(arma::uword n = 0; n < N; n++){
  //     
  //     ll = logLikelihood(X.row(n).t());
  //     
  //     // Update with weights
  //     comp_prob = ll + log(w);
  //     
  //     // Normalise and overflow
  //     comp_prob = exp(comp_prob - max(comp_prob));
  //     comp_prob = comp_prob / sum(comp_prob);
  //     
  //     // Prediction and update
  //     u = arma::randu<double>( );
  //     labels(n) = sum(u > cumsum(comp_prob));
  //     alloc.row(n) = comp_prob.t();
  //     
  //     // Record the likelihood of the item in it's allocated component
  //     likelihood(n) = ll(labels(n));
  //     
  //     // Update if the point is an outlier or not
  //     outlier(n) = sampleOutlier(n);
  //   }
  //   
  //   // The model log likelihood
  //   model_likelihood = arma::accu(likelihood);
  //   
  //   // Number of occupied components (used in BIC calculation)
  //   uniqueK = arma::unique(labels);
  //   K_occ = uniqueK.n_elem;
  // };
  
  // void sampleParameters(){
  //   arma::vec mu_n(P);
  //   
  //   arma::uword n_k = 0;
  //   arma::vec mu_k(P), sample_mean(P);
  //   arma::mat sample_cov(P, P), dist_from_prior(P, P), scale_n(P, P);
  //   
  //   for (arma::uword k = 0; k < K; k++) {
  //     
  //     // Find how many labels have the value
  //     n_k = N_k(k);
  //     if(n_k > 0){
  //       
  //       // Component data
  //       arma::mat component_data = X.rows( arma::find(members.col(k) == 1) );
  //       
  //       // Sample mean in the component data
  //       sample_mean = arma::mean(component_data).t();
  //       
  //       // The weighted average of the prior mean and sample mean
  //       mu_k = (kappa * xi + n_k * sample_mean) / (double)(kappa + N);
  //       
  //       sample_cov = calcSampleCov(component_data, sample_mean, n_k, P);
  //       
  //       // Calculate the distance of the sample mean from the prior
  //       dist_from_prior = (sample_mean - xi) * (sample_mean - xi).t();
  //       
  //       // Update the scale hyperparameter
  //       scale_n = scale + sample_cov + ((kappa * N) / (double)(kappa + N)) * dist_from_prior;
  //       
  //       cov.slice(k) = arma::iwishrnd(scale_n, nu + n_k);
  //       
  //       mu.col(k) = arma::mvnrnd(mu_k, (1.0/kappa) * cov.slice(k), 1);
  //       
  //     } else{
  //       
  //       cov.slice(k) = arma::iwishrnd(scale, nu);
  //       mu.col(k) = arma::mvnrnd(xi, (1.0/kappa) * cov.slice(k), 1);
  //       
  //     }
  //     
  //   }
  // }
  
  void calcBIC(){
    
    arma::uword n_param = (P + P * (P + 1) * 0.5) * (K_occ + 1);
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  }
  
};

//' @name tagmGaussian
//' @title T-ADjusted Gaussian Mixture (TAGM) type
//' @description The sampler for the TAGM mixture model with assumption of 
//' independent features.
//' @field new Constructor \itemize{
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
//' @field printType Print the sampler type called.
//' @field calcBIC Calculate the BIC of the model.
//' @field calcTdistnLikelihood Calculate the likelihood of a given data point 
//' the gloabl t-distribution. \itemize{
//' \item Parameter: point - a data point.
//' }
class tagmGaussian : public tAdjustedSampler, public gaussianSampler {
  
private:
  
public:
  // for use in the outlier distribution
  // arma::uvec outlier;
  // arma::vec global_mean;
  // arma::mat global_cov;
  // double df = 4.0, u = 2.0, v = 10.0, b = 0.0, outlier_weight = 0.0;
  
  using tAdjustedSampler::tAdjustedSampler;
  
  tagmGaussian(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X
  ) : tAdjustedSampler(_K, _labels, _concentration, _X),
  gaussianSampler(_K, _labels, _concentration, _X),
  sampler(_K, _labels, _concentration, _X) {
    
    // for use in the outlier distribution
    global_cov = 0.5 * arma::stddev(X, 0); 
    global_mean = arma::trans(arma::mean(X, 0));
    
    // outlier = arma::zeros<arma::uvec>(N);
    // 
    // b = (double) sum(outlier);
    // outlier_weight = rBeta(b + u, N + v - b);
    
  };
  
  // Destructor
  virtual ~tagmGaussian() { };
  
  void printType() {
    std::cout << "Type: TAGM (independent features).\n";
  }
  
  double calcTdistnLikelihood(arma::vec point) {
    
    double exponent = 0.0;
    double log_likelihood = 0.0;
    // double std_dev_p = 0;
    
    for(arma::uword p = 0; p < P; p++){
      
      // arma::as_scalar(global_cov(p));
      
      exponent = std::pow(point(p) - global_mean(p), 2.0) / global_cov(p);
      
      log_likelihood += lgamma((df + P) / 2.0) 
        - lgamma(df / 2.0) 
        - (double)P / 2.0 * log(df * M_PI) 
        - 0.5 * global_cov(p) 
        - ((df + (double) P) / 2.0) * log(1.0 + (1.0 / P) * exponent);
    }
    
      
      return log_likelihood;
  };
  
  // void updateOutlierWeights(){
  //   b = (double) sum(outlier);
  //   outlier_weight = rBeta(b + u, N + v - b);
  // };
  // 
  // void updateWeights(){
  //   
  //   double a = 0.0;
  //   
  //   for (arma::uword k = 0; k < K; k++) {
  //     
  //     // Find how many labels have the value
  //     members.col(k) = (labels == k) % outlier;
  //     N_k(k) = arma::sum(members.col(k));
  //     
  //     // Update weights by sampling from a Gamma distribution
  //     a  = concentration(k) + N_k(k);
  //     w(k) = arma::randg( arma::distr_param(a, 1.0) );
  //   }
  //   
  //   // Convert the cluster weights (previously gamma distributed) to Beta 
  //   // distributed by normalising
  //   w = w / arma::sum(w);
  // };
  // 
  // arma::uword sampleOutlier(int n) {
  //   
  //   arma::vec point = X.row(n).t();
  //   // arma::uword k = labels(n);
  //   
  //   double out_likelihood = 0.0;
  //   arma::vec outlier_prob(2);
  //   outlier_prob.zeros();
  //   
  //   // The likelihood of the point in the current cluster
  //   outlier_prob(0) = likelihood(n) + log(1 - outlier_weight);
  //   
  //   // Calculate outlier likelihood
  //   out_likelihood = calcTdistnLikelihood(point);
  //   out_likelihood += log(out_likelihood);
  //   outlier_prob(1) = out_likelihood;
  //   
  //   // Normalise and overflow
  //   outlier_prob = exp(outlier_prob - max(outlier_prob));
  //   outlier_prob = outlier_prob / sum(outlier_prob);
  //   
  //   // Prediction and update
  //   u = arma::randu<double>( );
  //   return sum(u > cumsum(outlier_prob));
  // };
  // 
  // void updateAllocation() {
  //   
  //   double u = 0.0;
  //   arma::uvec uniqueK;
  //   arma::vec comp_prob(K);
  //   
  //   // First update the outlier parameters
  //   updateOutlierWeights();
  //   
  //   for(arma::uword n = 0; n < N; n++){
  //     
  //     ll = logLikelihood(X.row(n).t());
  //     
  //     // Update with weights
  //     comp_prob = ll + log(w);
  //     
  //     // Normalise and overflow
  //     comp_prob = exp(comp_prob - max(comp_prob));
  //     comp_prob = comp_prob / sum(comp_prob);
  //     
  //     // Prediction and update
  //     u = arma::randu<double>( );
  //     labels(n) = sum(u > cumsum(comp_prob));
  //     alloc.row(n) = comp_prob.t();
  //     
  //     // Record the likelihood of the item in it's allocated component
  //     likelihood(n) = ll(labels(n));
  //     
  //     // Update if the point is an outlier or not
  //     outlier(n) = sampleOutlier(n);
  //   }
  //   
  //   // The model log likelihood
  //   model_likelihood = arma::accu(likelihood);
  //   
  //   // Number of occupied components (used in BIC calculation)
  //   uniqueK = arma::unique(labels);
  //   K_occ = uniqueK.n_elem;
  // };
  
  
  void calcBIC(){
    
    arma::uword n_param = (P + P) * (K_occ + 1);
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
    
  }
  
};

class semisupervisedSampler : public virtual sampler {
private:
  
public:
  
  arma::uword N_fixed = 0;
  arma::uvec fixed, unfixed_ind;
  arma::mat X_unfixed;
  
  using sampler::sampler;
  
  semisupervisedSampler(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X,
    arma::uvec _fixed
  ) : 
  sampler(_K, _labels, _concentration, _X) {
    
    fixed = _fixed;
    N_fixed = arma::sum(fixed);
    unfixed_ind = find(fixed == 0);
    X_unfixed = X.elem( find(fixed == 0) );
    
  };
  
  // Destructor
  virtual ~semisupervisedSampler() { };
  
  void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    for (auto& n : unfixed_ind) {
    // for(arma::uword n = 0; n < N; n++){
      
      ll = logLikelihood(X.row(n).t());
      
      // if(fixed(n) == 0) {
      
      // Update with weights
      comp_prob = ll + log(w);
      
      // Normalise and overflow
      comp_prob = exp(comp_prob - max(comp_prob));
      comp_prob = comp_prob / sum(comp_prob);
      
      // Prediction and update
      u = arma::randu<double>( );
      labels(n) = sum(u > cumsum(comp_prob));
      alloc.row(n) = comp_prob.t();
      // }
      
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


class mvnPredictive : public semisupervisedSampler, public mvnSampler {
  
private:
  
public:
  
  using mvnSampler::mvnSampler;
  
  mvnPredictive(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X,
    arma::uvec _fixed
  ) : 
  semisupervisedSampler(_K, _labels, _concentration, _X, _fixed),
  mvnSampler(_K, _labels, _concentration, _X),
  sampler(_K, _labels, _concentration, _X) {
    
    
  };
  
  virtual ~mvnPredictive() { };
  
};


class tagmPredictive : public mvnPredictive, public tAdjustedSampler {
  
private:
  
public:
  
  using mvnPredictive::mvnPredictive;
  
  tagmPredictive(
    arma::uword _K,
    arma::uvec _labels, 
    arma::vec _concentration,
    arma::mat _X,
    arma::uvec _fixed
  ) : 
    mvnPredictive(_K, _labels, _concentration, _X, _fixed),
    tAdjustedSampler(_K, _labels, _concentration, _X),
    sampler(_K, _labels, _concentration, _X) {
    
    outlier = arma::ones<arma::uvec>(N) - fixed;
    
  };
  
  virtual ~tagmPredictive() { };
  
  virtual void updateAllocation() {
    
    double u = 0.0;
    arma::uvec uniqueK;
    arma::vec comp_prob(K);
    
    // First update the outlier parameters
    updateOutlierWeights();
    
    for (auto& n : unfixed_ind) {
      
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
      
      // Update if the point is an outlier or not
      outlier(n) = sampleOutlier(n);
    }
    
    // The model log likelihood
    model_likelihood = arma::accu(likelihood);
    
    // Number of occupied components (used in BIC calculation)
    uniqueK = arma::unique(labels);
    K_occ = uniqueK.n_elem;
  };
  
  void calcBIC(){
    
    arma::uword n_param = (P + P * (P + 1) * 0.5) * (K_occ + 1);
    BIC = n_param * std::log(N) - 2 * model_likelihood;
    
  };
  
};

// Factory for creating instances of samplers
//' @name samplerFactory
//' @title Factory for different sampler subtypes.
//' @description The factory allows the type of mixture implemented to change 
//' based upon the user input.
//' @field new Constructor \itemize{
//' \item Parameter: samplerType - the density type to be modelled
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
class samplerFactory
{
  public:
  enum samplerType {
    G = 0,
    MVN = 1,
    C = 2,
    TMVN = 3,
    TG = 4
  };

  static std::unique_ptr<sampler> createSampler(samplerType type,
                                         arma::uword K,
                                         arma::uvec labels,
                                         arma::vec concentration,
                                         arma::mat X) {
  switch (type) {
    case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X);
    case MVN: return std::make_unique<mvnSampler>(K, labels, concentration, X);
    case C: return std::make_unique<categoricalSampler>(K, labels, concentration, X);
    case TMVN: return std::make_unique<tagmMVN>(K, labels, concentration, X);
    case TG: return std::make_unique<tagmGaussian>(K, labels, concentration, X);
    default: throw "invalid sampler type.";
    }
    
  }

};


// Factory for creating instances of samplers
//' @name samplerFactory
//' @title Factory for different sampler subtypes.
//' @description The factory allows the type of mixture implemented to change 
//' based upon the user input.
//' @field new Constructor \itemize{
//' \item Parameter: samplerType - the density type to be modelled
//' \item Parameter: K - the number of components to model
//' \item Parameter: labels - the initial clustering of the data
//' \item Parameter: concentration - the vector for the prior concentration of 
//' the Dirichlet distribution of the component weights
//' \item Parameter: X - the data to model
//' }
class semisupervisedSamplerFactory
{
public:
  enum samplerType {
    // G = 0,
    MVN = 1,
    // C = 2,
    TMVN = 3
    // TG = 4
  };
  
  static std::unique_ptr<semisupervisedSampler> createSemisupervisedSampler(samplerType type,
                                                arma::uword K,
                                                arma::uvec labels,
                                                arma::vec concentration,
                                                arma::mat X,
                                                arma::uvec fixed) {
    switch (type) {
    // case G: return std::make_unique<gaussianSampler>(K, labels, concentration, X, fixed);
    case MVN: return std::make_unique<mvnPredictive>(K, labels, concentration, X, fixed);
    // case C: return std::make_unique<categoricalSampler>(K, labels, concentration, X, fixed);
    case TMVN: return std::make_unique<tagmPredictive>(K, labels, concentration, X, fixed);
    // case TG: return std::make_unique<tagmGaussian>(K, labels, concentration, X, fixed);
    default: throw "invalid sampler type.";
    }
    
  }
  
};

//' @title Mixture model
//' @description Performs MCMC sampling for a mixture model.
//' @param X The data matrix to perform clustering upon (items to cluster in rows).
//' @param K The number of components to model (upper limit on the number of clusters found).
//' @param labels Vector item labels to initialise from.
//' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
//' @param R The number of iterations to run for.
//' @param thin thinning factor for samples recorded.
//' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
//' @return Named list of the matrix of MCMC samples generated (each row 
//' corresponds to a different sample) and BIC for each saved iteration.
// [[Rcpp::export]]
Rcpp::List sampleMixtureModel (
    arma::mat X,
    arma::uword K,
    arma::uvec labels,
    int dataType,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    arma::uword seed
) {
  
  // Set the random number
  std::default_random_engine generator(seed);
  
  // Declare the factory
  samplerFactory my_factory;
  
  // Convert from an int to the samplerType variable for our Factory
  samplerFactory::samplerType val = static_cast<samplerFactory::samplerType>(dataType);
  
  // Make a pointer to the correct type of sampler
  std::unique_ptr<sampler> sampler_ptr = my_factory.createSampler(val,
                                              K,
                                              labels,
                                              concentration,
                                              X);
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();

  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));

  arma::uword save_int=0;

  // Sampler from priors (this is unnecessary)
  sampler_ptr->sampleFromPriors();

  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    sampler_ptr->updateWeights();
    sampler_ptr->sampleParameters();
    sampler_ptr->updateAllocation();

    // Record results
    if((r + 1) % thin == 0){

      // Update the BIC for the current model fit
      sampler_ptr->calcBIC();
      BIC_record( save_int ) = sampler_ptr->BIC; 

      // Save the current clustering
      class_record.row( save_int ) = sampler_ptr->labels.t();
      save_int++;
    }
  }
  return(List::create(Named("samples") = class_record, Named("BIC") = BIC_record));
};


//' @title Mixture model
//' @description Performs MCMC sampling for a mixture model.
//' @param X The data matrix to perform clustering upon (items to cluster in rows).
//' @param K The number of components to model (upper limit on the number of clusters found).
//' @param labels Vector item labels to initialise from.
//' @param fixed Binary vector of the items that are fixed in their initial label.
//' @param dataType Int, 0: independent Gaussians, 1: Multivariate normal, or 2: Categorical distributions.
//' @param R The number of iterations to run for.
//' @param thin thinning factor for samples recorded.
//' @param concentration Vector of concentrations for mixture weights (recommended to be symmetric).
//' @return Named list of the matrix of MCMC samples generated (each row 
//' corresponds to a different sample) and BIC for each saved iteration.
// [[Rcpp::export]]
Rcpp::List sampleSemisupervisedMixtureModel (
    arma::mat X,
    arma::uword K,
    arma::uvec labels,
    arma::uvec fixed,
    int dataType,
    arma::uword R,
    arma::uword thin,
    arma::vec concentration,
    arma::uword seed
) {
  
  // Set the random number
  std::default_random_engine generator(seed);
  
  // Declare the factory
  semisupervisedSamplerFactory my_factory;
  
  // Convert from an int to the samplerType variable for our Factory
  semisupervisedSamplerFactory::samplerType val = static_cast<semisupervisedSamplerFactory::samplerType>(dataType);
  
  // Make a pointer to the correct type of sampler
  std::unique_ptr<sampler> sampler_ptr = my_factory.createSemisupervisedSampler(val,
                                                                  K,
                                                                  labels,
                                                                  concentration,
                                                                  X,
                                                                  fixed);
  
  // The output matrix
  arma::umat class_record(floor(R / thin), X.n_rows);
  class_record.zeros();
  
  // We save the BIC at each iteration
  arma::vec BIC_record = arma::zeros<arma::vec>(floor(R / thin));
  
  arma::uword save_int=0;
  
  // Sampler from priors (this is unnecessary)
  sampler_ptr->sampleFromPriors();
  
  // Iterate over MCMC moves
  for(arma::uword r = 0; r < R; r++){
    
    sampler_ptr->updateWeights();
    sampler_ptr->sampleParameters();
    sampler_ptr->updateAllocation();
    
    // Record results
    if((r + 1) % thin == 0){
      
      // Update the BIC for the current model fit
      sampler_ptr->calcBIC();
      BIC_record( save_int ) = sampler_ptr->BIC; 
      
      // Save the current clustering
      class_record.row( save_int ) = sampler_ptr->labels.t();
      save_int++;
    }
  }
  return(List::create(Named("samples") = class_record, Named("BIC") = BIC_record));
};
