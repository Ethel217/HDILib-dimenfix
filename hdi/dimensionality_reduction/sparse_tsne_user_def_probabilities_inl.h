/*
 *
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef SPARSE_TSNE_USER_DEF_PROBABILITIES_INL
#define SPARSE_TSNE_USER_DEF_PROBABILITIES_INL

#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/utils/math_utils.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"
#include "sptree.h"
#include <random>

#ifdef __USE_GCD__
#include <dispatch/dispatch.h>
#endif

#pragma warning( push )
#pragma warning( disable : 4267)
#pragma warning( push )
#pragma warning( disable : 4291)
#pragma warning( push )
#pragma warning( disable : 4996)
#pragma warning( push )
#pragma warning( disable : 4018)
#pragma warning( push )
#pragma warning( disable : 4244)
//#define FLANN_USE_CUDA
#include "flann/flann.h"
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )

namespace hdi{
  namespace dr{

    template <typename scalar, typename sparse_scalar_matrix>
    SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::SparseTSNEUserDefProbabilities():
      _initialized(false),
      _logger(nullptr),
      _theta(0),
      _exaggeration_baseline(1)
    {

    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::reset(){
      _initialized = false;
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::clear(){
      _embedding->clear();
      _initialized = false;
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::getEmbeddingPosition(scalar_vector_type& embedding_position, data_handle_type handle)const{
      if(!_initialized){
        throw std::logic_error("Algorithm must be initialized before ");
      }
      embedding_position.resize(_params._embedding_dimensionality);
      for(int i = 0; i < _params._embedding_dimensionality; ++i){
        (*_embedding_container)[i] = (*_embedding_container)[handle*_params._embedding_dimensionality + i];
      }
    }


  /////////////////////////////////////////////////////////////////////////


    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::initialize(const sparse_scalar_matrix& probabilities, data::Embedding<scalar_type>* embedding, TsneParameters params){
      utils::secureLog(_logger,"Initializing tSNE...");
      {//Aux data
        _params = params;
        unsigned int size = probabilities.size();
        unsigned int size_sq = probabilities.size()*probabilities.size();

        _embedding = embedding;
        _embedding_container = &(embedding->getContainer());
        _embedding->resize(_params._embedding_dimensionality,size);
        _P.resize(size);
        _Q.resize(size_sq);
        _gradient.resize(size*params._embedding_dimensionality,0);
        _previous_gradient.resize(size*params._embedding_dimensionality,0);
        _gain.resize(size*params._embedding_dimensionality,1);

        // _range_limits.resize(size * 2,0);

        // std::random_device rd; // Seed for random generator
        // std::mt19937 gen(rd()); // Standard mersenne_twister_engine
        // std::uniform_int_distribution<> dist_lower(0, 50); // Adjust range for lower bounds
        // std::uniform_int_distribution<> dist_upper(51, 100); // Adjust range for upper bounds

        // for (unsigned int i = 0; i < size; ++i) {
        //     int lower_bound = dist_lower(gen);
        //     int upper_bound = dist_upper(gen);
        //     _range_limits[i * 2] = lower_bound;
        //     _range_limits[i * 2 + 1] = upper_bound;
        // }
      }

      utils::secureLogValue(_logger,"Number of data points",_P.size());

      computeHighDimensionalDistribution(probabilities);

      if (!params._presetEmbedding) {
        initializeEmbeddingPosition(_params._seed, _params._rngRange);
      }

      _iteration = 0;

      _initialized = true;
      utils::secureLog(_logger,"Initialization complete!");
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::initializeWithJointProbabilityDistribution(const sparse_scalar_matrix& distribution, data::Embedding<scalar_type>* embedding, TsneParameters params){
      utils::secureLog(_logger,"Initializing tSNE with a user-defined joint-probability distribution...");
      {//Aux data
        _params = params;
        unsigned int size = distribution.size();
        unsigned int size_sq = distribution.size()*distribution.size();

        _embedding = embedding;
        _embedding_container = &(embedding->getContainer());
        _embedding->resize(_params._embedding_dimensionality,size);
        _P.resize(size);
        _Q.resize(size_sq);
        _gradient.resize(size*params._embedding_dimensionality,0);
        _previous_gradient.resize(size*params._embedding_dimensionality,0);
        _gain.resize(size*params._embedding_dimensionality,1);

        // _range_limits.resize(size * 2,0);

        // std::random_device rd; // Seed for random generator
        // std::mt19937 gen(rd()); // Standard mersenne_twister_engine
        // std::uniform_int_distribution<> dist_lower(0, 50); // Adjust range for lower bounds
        // std::uniform_int_distribution<> dist_upper(51, 100); // Adjust range for upper bounds

        // for (unsigned int i = 0; i < size; ++i) {
        //     int lower_bound = dist_lower(gen);
        //     int upper_bound = dist_upper(gen);
        //     _range_limits[i * 2] = lower_bound;
        //     _range_limits[i * 2 + 1] = upper_bound;
        // }
      }

      utils::secureLogValue(_logger,"Number of data points",_P.size());

      _P = distribution;

      if (!params._presetEmbedding) {
        initializeEmbeddingPosition(_params._seed, _params._rngRange);
      }

      _iteration = 0;

      _initialized = true;
      utils::secureLog(_logger,"Initialization complete!");
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::computeHighDimensionalDistribution(const sparse_scalar_matrix& probabilities){
      utils::secureLog(_logger,"Computing high-dimensional joint probability distribution...");

      const size_t n = getNumberOfDataPoints();
      //Can be improved by using the simmetry of the matrix (half the memory) //TODO
      for(size_t j = 0; j < n; ++j){
        for(auto& elem: probabilities[j]){
          scalar_type v0 = elem.second;
          auto iter = probabilities[elem.first].find(j);
          scalar_type v1 = 0.;
          if(iter != probabilities[elem.first].end())
            v1 = iter->second;

          _P[j][elem.first] = static_cast<scalar_type>((v0+v1)*0.5);
          _P[elem.first][j] = static_cast<scalar_type>((v0+v1)*0.5);
        }
      }
    }


    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::initializeEmbeddingPosition(int seed, double multiplier){
      utils::secureLog(_logger,"Initializing the embedding...");
      if(seed < 0){
        std::srand(static_cast<unsigned int>(time(NULL)));
      }
      else{
        std::srand(seed);
      }
        
      for (int i = 0; i < _embedding->numDataPoints(); ++i) {
        double x(0.);
        double y(0.);
        double radius(0.);
        do {
          x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
          y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
          radius = (x * x) + (y * y);
        } while((radius >= 1.0) || (radius == 0.0));

        radius = sqrt(-2 * log(radius) / radius);
        x *= radius * multiplier;
        y *= radius * multiplier;
        _embedding->dataAt(i, 0) = x;
        _embedding->dataAt(i, 1) = y;
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::doAnIteration(double mult){
      if(!_initialized){
        throw std::logic_error("Cannot compute a gradient descent iteration on unitialized data");
      }

      if(_iteration == _params._mom_switching_iter){
        utils::secureLog(_logger,"Switch to final momentum...");
      }
      if(_iteration == _params._remove_exaggeration_iter){
        utils::secureLog(_logger,"Remove exaggeration...");
      }

      if(_theta == 0){
        doAnIterationExact(mult);
      }else{
        doAnIterationBarnesHut(mult);
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    scalar SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::exaggerationFactor(){
      scalar_type exaggeration = _exaggeration_baseline;

      if(_iteration <= _params._remove_exaggeration_iter){
        exaggeration = _params._exaggeration_factor;
      }else if(_iteration <= (_params._remove_exaggeration_iter + _params._exponential_decay_iter)){
        //double decay = std::exp(-scalar_type(_iteration-_params._remove_exaggeration_iter)/30.);
        double decay = 1. - double(_iteration-_params._remove_exaggeration_iter)/_params._exponential_decay_iter;
        exaggeration = _exaggeration_baseline + (_params._exaggeration_factor-_exaggeration_baseline)*decay;
        //utils::secureLogValue(_logger,"Exaggeration decay...",exaggeration);
      }

      return exaggeration;
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::doAnIterationExact(double mult){
      //Compute Low-dimensional distribution
      computeLowDimensionalDistribution();

      //Compute gradient of the KL function
      computeExactGradient(exaggerationFactor());

      //Update the embedding based on the gradient
      updateTheEmbedding(mult);
    }
    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::doAnIterationBarnesHut(double mult){
      //Compute gradient of the KL function using the Barnes Hut approximation
      computeBarnesHutGradient(exaggerationFactor());

      //Update the embedding based on the gradient
      updateTheEmbedding();
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::computeLowDimensionalDistribution(){
      const size_t n = getNumberOfDataPoints();
#ifdef __USE_GCD__
      //std::cout << "GCD dispatch, sparse_tsne_user_def_probabilities 285.\n";
      dispatch_apply(n, dispatch_get_global_queue(0, 0), ^(size_t j) {
#else
      #pragma omp parallel for
      for(int64_t j = 0; j < static_cast<int64_t>(n); ++j){
#endif //__USE_GCD__
        _Q[j*n + j] = 0;
        for(int64_t i = j+1; i < n; ++i){
          const double euclidean_dist_sq(
              utils::euclideanDistanceSquared<scalar_type>(
                (*_embedding_container).begin()+j*_params._embedding_dimensionality,
                (*_embedding_container).begin()+(j+1)*_params._embedding_dimensionality,
                (*_embedding_container).begin()+i*_params._embedding_dimensionality,
                (*_embedding_container).begin()+(i+1)*_params._embedding_dimensionality
              )
            );
          const double v = 1./(1.+euclidean_dist_sq);
          _Q[j*n + i] = static_cast<scalar_type>(v);
          _Q[i*n + j] = static_cast<scalar_type>(v);
        }
      }
#ifdef __USE_GCD__
      );
#endif // __USE_GCD__
      double sum_Q = 0;
      for(auto& v : _Q){
        sum_Q += v;
      }
      _normalization_Q = static_cast<scalar_type>(sum_Q);
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::computeExactGradient(double exaggeration){
      const size_t n = getNumberOfDataPoints();
      const int dim = _params._embedding_dimensionality;

      for(size_t i = 0; i < n; ++i){
        for(int d = 0; d < dim; ++d){
          _gradient[i * dim + d] = 0;
        }
      }

      for(size_t i = 0; i < n; ++i){
        for(size_t j = 0; j < n; ++j){
          for(int d = 0; d < dim; ++d){
            const size_t idx = i*n + j;
            const double distance((*_embedding_container)[i * dim + d] - (*_embedding_container)[j * dim + d]);
            const double negative(_Q[idx] * _Q[idx] / _normalization_Q * distance);
            _gradient[i * dim + d] += static_cast<scalar_type>(-4*negative);
          }
        }
        for(auto& elem: _P[i]){
          for(int d = 0; d < dim; ++d){
            const int j = elem.first;
            const size_t idx = i*n + j;
            const double distance((*_embedding_container)[i * dim + d] - (*_embedding_container)[j * dim + d]);
            double p_ij = elem.second/n;

            const double positive(p_ij * _Q[idx] * distance);
            _gradient[i * dim + d] += static_cast<scalar_type>(4*exaggeration*positive);
          }
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::computeBarnesHutGradient(double exaggeration){
      typedef double hp_scalar_type;

      SPTree<scalar_type> sptree(_params._embedding_dimensionality,_embedding->getContainer().data(),getNumberOfDataPoints());

      scalar_type sum_Q = .0;
      std::vector<hp_scalar_type> positive_forces(getNumberOfDataPoints()*_params._embedding_dimensionality);
      /*__block*/ std::vector<hp_scalar_type> negative_forces(getNumberOfDataPoints()*_params._embedding_dimensionality);

      sptree.computeEdgeForces(_P, exaggeration, positive_forces.data());

      /*__block*/ std::vector<hp_scalar_type> sum_Q_subvalues(getNumberOfDataPoints(),0);
//#ifdef __USE_GCD__
//      std::cout << "GCD dispatch, sparse_tsne_user_def_probabilities 365.\n";
//      dispatch_apply(getNumberOfDataPoints(), dispatch_get_global_queue(0, 0), ^(size_t n) {
//#else
      #pragma omp parallel for
      for(int64_t n = 0; n < getNumberOfDataPoints(); n++){
//#endif //__USE_GCD__
        sptree.computeNonEdgeForcesOMP(n, _theta, negative_forces.data() + n * _params._embedding_dimensionality, sum_Q_subvalues[n]);
      }
//#ifdef __USE_GCD__
//      );
//#endif

      sum_Q = 0;
      for(size_t n = 0; n < getNumberOfDataPoints(); n++){
        sum_Q += sum_Q_subvalues[n];
      }

      for(size_t i = 0; i < _gradient.size(); i++){
        _gradient[i] = positive_forces[i] - (negative_forces[i] / sum_Q);
      }

    }

    //temp
    template <typename T>
    T sign(T x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

    template <typename scalar, typename sparse_scalar_matrix>
    void SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::updateTheEmbedding(double mult){
      // const int dim = _params._embedding_dimensionality;
      for(int i = 0; i < _gradient.size(); ++i){
        _gain[i] = static_cast<scalar_type>((sign(_gradient[i]) != sign(_previous_gradient[i])) ? (_gain[i] + .2) : (_gain[i] * .8));
        if(_gain[i] < _params._minimum_gain){
          _gain[i] = static_cast<scalar_type>(_params._minimum_gain);
        }
        _gradient[i] = static_cast<scalar_type>((_gradient[i]>0?1:-1)*std::abs(_gradient[i]*_params._eta* _gain[i])/(_params._eta*_gain[i]));

        _previous_gradient[i] = static_cast<scalar_type>(((_iteration<_params._mom_switching_iter)?_params._momentum:_params._final_momentum) * _previous_gradient[i] - _params._eta * _gain[i] * _gradient[i]);
        (*_embedding_container)[i] += static_cast<scalar_type>(_previous_gradient[i] * mult);
        // if (i % dim != 0) {
        //   // std::cout << _range_limits.size() << std::endl;
        //   (*_embedding_container)[i] += static_cast<scalar_type>(_previous_gradient[i] * mult);
        // }
        // else {
        //   // (*_embedding_container)[i] = static_cast<scalar_type>(_range_limits[0]); // TODO
        //   // std::cout << i << std::endl;
        //   (*_embedding_container)[i] = std::max(_range_limits[(int)(i / dim)], std::min((*_embedding_container)[i], _range_limits[(int)(i / dim + 1)]));;
        // }
      }

      //MAGIC NUMBER
      if(exaggerationFactor() > 1.2){
        _embedding->scaleIfSmallerThan(0.1f);
      }else{
        _embedding->zeroCentered();
      }

      ++_iteration;
    }

    template <typename scalar, typename sparse_scalar_matrix>
    double SparseTSNEUserDefProbabilities<scalar, sparse_scalar_matrix>::computeKullbackLeiblerDivergence(){
      assert(false);
      return 0;
    }
  }
}
#endif
