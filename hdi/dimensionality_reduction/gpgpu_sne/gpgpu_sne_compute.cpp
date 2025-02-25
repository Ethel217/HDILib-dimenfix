#ifndef __APPLE__

#include "gpgpu_sne_compute.h"
#include "compute_shaders.glsl"
#include "opencv2\opencv.hpp"

#include <vector>
#include <limits>
#include <iostream>
#include <cmath> // for sqrt
#include <fstream>
#include <algorithm>

namespace hdi {
  namespace dr {
    typedef GpgpuSneCompute::Bounds2D Bounds2D;
    typedef GpgpuSneCompute::Point2D Point2D;

    enum BufferType
    {
      POSITION,
      INTERP_FIELDS,
      SUM_Q,
      NEIGHBOUR,
      PROBABILITIES,
      INDEX,
      GRADIENTS,
      PREV_GRADIENTS,
      GAIN,
      BOUNDS,
      RANGE_LIMITS,
      LABELS,
      CLASS_BOUNDS
    };

    // Linearized sparse neighbourhood matrix
    struct LinearProbabilityMatrix
    {
      std::vector<uint32_t> neighbours;
      std::vector<float> probabilities;
      std::vector<int> indices;
    };

    GpgpuSneCompute::GpgpuSneCompute() :
      _initialized(false),
      _adaptive_resolution(true),
      _resolutionScaling(PIXEL_RATIO)
    {

    }

    Bounds2D GpgpuSneCompute::computeEmbeddingBounds(const embedding_type* embedding, float padding) {
      const float* const points = embedding->getContainer().data();

      Bounds2D bounds;
      bounds.min.x = std::numeric_limits<float>::max();
      bounds.max.x = -std::numeric_limits<float>::max();
      bounds.min.y = std::numeric_limits<float>::max();
      bounds.max.y = -std::numeric_limits<float>::max();

      for (int i = 0; i < embedding->numDataPoints(); ++i) {
        float x = points[i * 2 + 0];
        float y = points[i * 2 + 1];

        bounds.min.x = std::min<float>(x, bounds.min.x);
        bounds.max.x = std::max<float>(x, bounds.max.x);
        bounds.min.y = std::min<float>(y, bounds.min.y);
        bounds.max.y = std::max<float>(y, bounds.max.y);
      }

      // Add any extra padding if requested
      if (padding != 0) {
        float half_padding = padding / 2;

        float x_padding = (bounds.max.x - bounds.min.x) * half_padding;
        float y_padding = (bounds.max.y - bounds.min.y) * half_padding;

        bounds.min.x -= x_padding;
        bounds.max.x += x_padding;
        bounds.min.y -= y_padding;
        bounds.max.y += y_padding;
      }

      return bounds;
    }

    void GpgpuSneCompute::initialize(const embedding_type* embedding, TsneParameters params, const sparse_scalar_matrix_type& P, const std::vector<Point2D>& range_limit, std::vector<int> labels) {
      _params = params;

      unsigned int num_points = embedding->numDataPoints();

      // Linearize sparse probability matrix
      LinearProbabilityMatrix linear_P;
      unsigned int num_pnts = embedding->numDataPoints();
      for (int i = 0; i < num_pnts; ++i) {
        linear_P.indices.push_back(linear_P.neighbours.size());
        int size = 0;
        for (const auto& pij : P[i]) {
          linear_P.neighbours.push_back(pij.first);
          linear_P.probabilities.push_back(pij.second);
          size++;
        }
        linear_P.indices.push_back(size);
      }

      // Compute initial data bounds
      _bounds = computeEmbeddingBounds(embedding);

      _function_support = 6.5f;

      // Initialize all OpenGL resources
      initializeOpenGL(num_points, linear_P, range_limit, labels);

      std::cout << "dimenfix: " << _params._dimenfix << ", ";
      std::cout << "every " << _params._iters << " iters, ";
      std::cout << "mode: " << _params._mode << ", ";
      std::cout << "class order: " << _params._class_order << ", ";
      std::cout << "switch axis: " << _params._switch_axis << ", ";
      std::cout << "alpha: " << _params._alpha << ", ";
      std::cout << "fixed_axis: " << _params._fix_selection << std::endl;

      _initialized = true;
    }

    void GpgpuSneCompute::clean()
    {
      glDeleteBuffers(13, _compute_buffers.data());

      fieldComputation.clean();
    }

    void GpgpuSneCompute::initializeOpenGL(const unsigned int num_points, const LinearProbabilityMatrix& linear_P, const std::vector<Point2D>& range_limit, std::vector<int> labels) {
      glClearColor(0, 0, 0, 0);

      fieldComputation.init(num_points);

      // Load in shader programs
      try {
        _interp_program.create();
        _forces_program.create();
        _update_program.create();
        _bounds_program.create();
        _center_and_scale_program.create();

        _class_bounds_program.create();
        _dimenfix_program.create();

        _interp_program.addShader(COMPUTE, interp_fields_source);
        _forces_program.addShader(COMPUTE, compute_forces_source);
        _update_program.addShader(COMPUTE, update_source);
        _bounds_program.addShader(COMPUTE, bounds_source);
        _center_and_scale_program.addShader(COMPUTE, center_and_scale_source);

        _class_bounds_program.addShader(COMPUTE, class_bounds_source);
        _dimenfix_program.addShader(COMPUTE, dimenfix_source);

        _interp_program.build();
        _forces_program.build();
        _update_program.build();
        _bounds_program.build();
        _center_and_scale_program.build();

        _class_bounds_program.build();
        _dimenfix_program.build();
      }
      catch (const ShaderLoadingException& e) {
        std::cout << e.what() << std::endl;
      }

      // Set constant uniforms
      _interp_program.bind();
      _interp_program.uniform1ui("num_points", num_points);
      _forces_program.bind();
      _forces_program.uniform1ui("num_points", num_points);

      // Create the SSBOs
      glGenBuffers(_compute_buffers.size(), _compute_buffers.data());

      // Load up SSBOs with initial values
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), nullptr, GL_STREAM_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[INTERP_FIELDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * 4 * sizeof(float), nullptr, GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[SUM_Q]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), nullptr, GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), nullptr, GL_STREAM_READ);

      // Upload sparse probability matrix
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[NEIGHBOUR]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, linear_P.neighbours.size() * sizeof(uint32_t), linear_P.neighbours.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[PROBABILITIES]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, linear_P.probabilities.size() * sizeof(float), linear_P.probabilities.data(), GL_STATIC_DRAW);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[INDEX]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, linear_P.indices.size() * sizeof(int), linear_P.indices.data(), GL_STATIC_DRAW);

      // Initialize buffer with 0s
      std::vector<float> zeroes(num_points * 2, 0);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[PREV_GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), zeroes.data(), GL_STREAM_READ);

      // Initialize buffer with 1s
      std::vector<float> ones(num_points * 2, 1);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[GAIN]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, num_points * sizeof(Point2D), ones.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[BOUNDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(Point2D), ones.data(), GL_STREAM_READ);

      // Add new buffer for range_limit
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[RANGE_LIMITS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, range_limit.size() * sizeof(Point2D), range_limit.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[LABELS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, labels.size() * sizeof(int), labels.data(), GL_STATIC_DRAW);

      // calculate a class bound for each point
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[CLASS_BOUNDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, range_limit.size() * sizeof(Point2D), zeroes.data(), GL_STREAM_DRAW);
      // std::vector<float> data(10);
      // glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 10 * sizeof(float), data.data());
      // for (const auto& value : data) {
      //   std::cout << value << " ";
      // }
      glGenQueries(2, _timerQuery);
    }

    void GpgpuSneCompute::startTimer()
    {
      glQueryCounter(_timerQuery[0], GL_TIMESTAMP);
    }

    void GpgpuSneCompute::stopTimer()
    {
      glQueryCounter(_timerQuery[1], GL_TIMESTAMP);
    }

    double GpgpuSneCompute::getElapsed()
    {
      GLint stopTimerAvailable = 0;
      while (!stopTimerAvailable)
      {
        glGetQueryObjectiv(_timerQuery[1], GL_QUERY_RESULT_AVAILABLE, &stopTimerAvailable);
      }
      GLuint64 startTime, stopTime;
      glGetQueryObjectui64v(_timerQuery[0], GL_QUERY_RESULT, &startTime);
      glGetQueryObjectui64v(_timerQuery[1], GL_QUERY_RESULT, &stopTime);

      double elapsed = (stopTime - startTime) / 1000000.0;

      return elapsed;
    }

    void GpgpuSneCompute::compute(embedding_type* embedding, float exaggeration, float iteration, float mult) {
      float* points = embedding->getContainer().data();
      unsigned int num_points = embedding->numDataPoints();

      if (iteration < 0.5)
      {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), points);
      }

      // Compute the bounds of the given embedding and add a 10% border around it
      computeEmbeddingBounds1(num_points, points, 0.1f);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[BOUNDS]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Point2D) * 2, &_bounds);
      Point2D range = _bounds.getRange();

      float aspect = range.x / range.y;

      uint32_t width = _adaptive_resolution ? std::max((unsigned int)(range.x * _resolutionScaling), MINIMUM_FIELDS_SIZE) : (int)(FIXED_FIELDS_SIZE * aspect);
      uint32_t height = _adaptive_resolution ? std::max((unsigned int)(range.y * _resolutionScaling), MINIMUM_FIELDS_SIZE) : FIXED_FIELDS_SIZE;

      // Compute the fields texture
      fieldComputation.compute(width, height, _function_support, num_points, _compute_buffers[POSITION], _compute_buffers[BOUNDS], _bounds.min.x, _bounds.min.y, _bounds.max.x, _bounds.max.y);

      // Calculate the normalization sum and sample the field values for every point
      float sum_Q = 0;
      interpolateFields(&sum_Q);

      // If normalization sum is 0, cancel further updating
      if (sum_Q == 0) {
        return;
      }

      if (_params._switch_axis) { // disable exaggeration in switching, probably dont need this
        exaggeration = 1;
      }

      // Compute the gradients of the KL-function
      computeGradients(num_points, sum_Q, exaggeration);

      // Update the point positions
      updatePoints(num_points, points, embedding, iteration, mult); // update points positions
      computeEmbeddingBounds1(num_points, points); // compute bounds
      updateEmbedding(num_points, exaggeration, iteration, mult); // rescale and center

      // new
      if (_params._dimenfix && (unsigned int)(iteration + 1) % _params._iters == 0) {
        if (_params._class_order == "avg" && (unsigned int)iteration >= _params._remove_exaggeration_iter) {
          // edit range limit using average position of class
          updateOrder(num_points, iteration, mult);
          if (_params._mode == "rescale") {
            calcClassBounds(num_points, iteration, mult);
          }
          pushEmbedding(num_points, iteration, mult);
        }
        else if (_params._class_order != "avg") {
          if (_params._mode == "rescale") {
            calcClassBounds(num_points, iteration, mult);
          }
          pushEmbedding(num_points, iteration, mult);
        }
        
      }

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, embedding->numDataPoints() * sizeof(Point2D), points);
    }

    void GpgpuSneCompute::computeEmbeddingBounds1(unsigned int num_points, const float* points, float padding, bool square)
    {
      // Compute bounds
      _bounds_program.bind();

      _bounds_program.uniform1ui("num_points", num_points);
      _bounds_program.uniform1f("padding", padding);
      //_bounds_program.uniform1i("square", square);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[BOUNDS]);

      // Compute the bounds
      glDispatchCompute(1, 1, 1);
    }

    void GpgpuSneCompute::interpolateFields(float* sum_Q)
    {
      // Bind fields texture for bilinear sampling
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, fieldComputation.getFieldTexture());

      // Set shader and uniforms
      _interp_program.bind();
      _interp_program.uniform1i("fields", 0);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[INTERP_FIELDS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[SUM_Q]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _compute_buffers[BOUNDS]);

      // Sample the fields texture for every point
      glDispatchCompute(1, 1, 1);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      // Copy sum_Q back to CPU
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[SUM_Q]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), sum_Q);
    }

    void GpgpuSneCompute::computeGradients(unsigned int num_points, float sum_Q, double exaggeration)
    {
      _forces_program.bind();

      _forces_program.uniform1f("exaggeration", exaggeration);
      _forces_program.uniform1f("sum_Q", sum_Q);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[NEIGHBOUR]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[PROBABILITIES]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _compute_buffers[INDEX]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _compute_buffers[INTERP_FIELDS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _compute_buffers[GRADIENTS]);

      // Compute the gradients of the KL function
      unsigned int grid_size = sqrt(num_points) + 1;
      glDispatchCompute(grid_size, grid_size, 1);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    void GpgpuSneCompute::updatePoints(unsigned int num_points, float* points, embedding_type* embedding, float iteration, float mult)
    {
      // update points' positions
      _update_program.bind();

      _update_program.uniform1ui("num_points", num_points);
      _update_program.uniform1f("eta", _params._eta);
      _update_program.uniform1f("minGain", _params._minimum_gain);
      _update_program.uniform1f("iter", iteration);
      _update_program.uniform1f("mom_iter", _params._mom_switching_iter);
      _update_program.uniform1f("mom", _params._momentum);
      _update_program.uniform1f("final_mom", _params._final_momentum);
      _update_program.uniform1f("mult", mult);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[GRADIENTS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[PREV_GRADIENTS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _compute_buffers[GAIN]);

      // Update the points
      unsigned int num_workgroups = (num_points * 2 / 64) + 1;
      unsigned int grid_size = sqrt(num_workgroups) + 1;
      glDispatchCompute(grid_size, grid_size, 1);
    }

    void GpgpuSneCompute::updateEmbedding(unsigned int num_points, float exaggeration, float iteration, float mult) {
      _center_and_scale_program.bind();

      _center_and_scale_program.uniform1ui("num_points", num_points);

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[BOUNDS]);
      // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[RANGE_LIMITS]);

      if (exaggeration > 1.2)
      {
        _center_and_scale_program.uniform1ui("scale", 1);
        _center_and_scale_program.uniform1f("diameter", 0.1f);
      }
      else if (_params._switch_axis && (int)iteration % 5 == 0) { // iteration limit: input
        _center_and_scale_program.uniform1ui("scale", 2);
        _center_and_scale_program.uniform1f("diameter", 0.3f); // the scale, can add as input
      }
      else
      {
        _center_and_scale_program.uniform1ui("scale", 0);
      }

      // Compute the bounds
      unsigned int num_workgroups = (num_points / 128) + 1;
      unsigned int grid_size = sqrt(num_workgroups) + 1;
      glDispatchCompute(grid_size, grid_size, 1);
    }

    void computePCA(std::vector<Point2D>& points) {
      int num_points = points.size();
      cv::Mat data(num_points, 2, CV_32F);
  
      // Load data into OpenCV Mat
      for (int i = 0; i < num_points; ++i) {
          data.at<float>(i, 0) = points[i].x;
          data.at<float>(i, 1) = points[i].y;
      }
  
      cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 2);
      cv::Mat reduced = pca.project(data);
  
      // x-y axis are switched, so that y-axis has largest variance
      for (int i = 0; i < num_points; ++i) {
          points[i].y = reduced.at<float>(i, 0);
          points[i].x = reduced.at<float>(i, 1);
      }
    }

    void GpgpuSneCompute::updateArrays(std::vector<Point2D> range_limit, std::vector<int> labels) {
      std::vector<Point2D> rl = range_limit;
      // for (int i = 0;i < 10;i ++) {
      //   std::cout << rl[i].x << " ";
      // }
      // std::cout << rl.size();

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[RANGE_LIMITS]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, rl.size() * sizeof(Point2D), rl.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

      std::cout << "check ranges update" << std::endl;
      std::vector<Point2D> data(10);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[RANGE_LIMITS]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 10 * sizeof(Point2D), data.data());
      // TODO: why is this 0???

      for (const auto& point : data) {
          std::cout << "(" << point.x << ", " << point.y << ") ";
      }
      std::cout << std::endl;


      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[LABELS]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, labels.size() * sizeof(int), labels.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void GpgpuSneCompute::updateOrder(unsigned int num_points, float iteration, float mult) {
      // calc average positions of each class and update range limit
      std::vector<Point2D> positions(num_points);
      std::vector<int> class_labels(num_points);
      std::vector<Point2D> range_limits(num_points);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[LABELS]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(int), class_labels.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), positions.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[RANGE_LIMITS]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), range_limits.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

      // PCA on positions (embedding update)
      computePCA(positions);

      // compute avg class positions y
      std::unordered_map<int, float> class_sums_y;
      std::unordered_map<int, int> class_counts;

      for (unsigned int i = 0; i < num_points; i++) {
          class_sums_y[class_labels[i]] += positions[i].y;
          class_counts[class_labels[i]]++;
      }

      // update range limits + buffer
      // TODO: arrange ORIGINAL ranges
      // Arrange class positions within range [0, 100] based on their average y position
      float min_y = 0.0f, max_y = 100.0f;
      std::vector<std::pair<int, float>> sorted_classes;
      for (std::unordered_map<int, float>::iterator it = class_sums_y.begin(); it != class_sums_y.end(); ++it) {
          int cls = it->first;
          float sum_y = it->second;
          sorted_classes.push_back(std::make_pair(cls, sum_y / class_counts[cls]));
      }
      std::sort(sorted_classes.begin(), sorted_classes.end(), [](const auto& a, const auto& b) {
          return a.second < b.second;
      });

      // Assign range limits for each class
      float current_y = min_y;
      for (int i = 0;i < sorted_classes.size();i ++) {
        int cls = sorted_classes[i].first;
        float height = (max_y - min_y) * (class_counts[cls] / (float)num_points);
          for (unsigned int i = 0; i < num_points; i++) {
              if (class_labels[i] == cls) {
                  range_limits[i].x = current_y;
                  range_limits[i].y = current_y + height;
              }
          }
          current_y += height;
      }

      // Write updated range limits back to GPU buffer
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[RANGE_LIMITS]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), range_limits.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

      // update points positions
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), positions.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void GpgpuSneCompute::calcClassBounds(unsigned int num_points, float iteration, float mult) {

      std::vector<Point2D> positions(num_points);
      std::vector<int> class_labels(num_points);

      std::vector<Point2D> class_bounds(num_points); // output, bind to buffer

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[LABELS]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(int), class_labels.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[POSITION]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), positions.data());
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

      std::unordered_map<uint32_t, std::vector<float>> class_points;

      for (size_t i = 0; i < positions.size(); i++) {
          int class_id = class_labels[i];
          float y = positions[i].y;
          class_points[class_id].push_back(y);
      }

      std::unordered_map<uint32_t, std::pair<float, float>> class_bounds_95;

      // for (auto& [class_id, y_values] : class_points) {
      for (auto& it: class_points) {
          // Sort y-values
          std::sort(it.second.begin(), it.second.end());

          size_t n = it.second.size();
          size_t lower_index = std::ceil(n * 0.025);
          size_t upper_index = std::floor(n * 0.975);

          float min_95 = it.second[lower_index];
          float max_95 = it.second[upper_index];

          class_bounds_95[it.first] = {min_95, max_95};
      }

      for (size_t i = 0; i < positions.size(); i++) {
          int class_id = class_labels[i];
          class_bounds[i].x = class_bounds_95[class_id].first;
          class_bounds[i].y = class_bounds_95[class_id].second;
      }

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _compute_buffers[CLASS_BOUNDS]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, num_points * sizeof(Point2D), class_bounds.data());
    }

    void GpgpuSneCompute::pushEmbedding(unsigned int num_points, float iteration, float mult) {
      _dimenfix_program.bind();

      _dimenfix_program.uniform1ui("num_points", num_points);

      if (_params._mode == "clipping") {
        _dimenfix_program.uniform1ui("mode", 0);
      }
      else if (_params._mode == "gaussian") {
        _dimenfix_program.uniform1ui("mode", 1);
      }
      else if (_params._mode == "rescale") {
        _dimenfix_program.uniform1ui("mode", 2);
      }

      if (_params._switch_axis) { // TODO: add a iters limit input
        _dimenfix_program.uniform1ui("aswitch", 1);
      }
      else {
        _dimenfix_program.uniform1ui("aswitch", 0);
      }

      // Bind required buffers to shader program
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _compute_buffers[POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _compute_buffers[BOUNDS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _compute_buffers[RANGE_LIMITS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _compute_buffers[CLASS_BOUNDS]);

      unsigned int num_workgroups = (num_points / 128) + 1;
      unsigned int grid_size = sqrt(num_workgroups) + 1;
      glDispatchCompute(grid_size, grid_size, 1);
    }
  }
}


#endif // __APPLE__
