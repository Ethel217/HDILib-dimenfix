#define GLSL(version, shader)  "#version " #version "\n" #shader

const char* interp_fields_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer Val { vec4 Values[]; };
  layout(std430, binding = 2) buffer SumB { float Sum[]; };
  layout(std430, binding = 3) buffer BoundsInterface { vec2 Bounds[]; };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  shared float reduction_array[64];

  uniform sampler2D fields;
  uniform uint num_points;

  void main() {
    uint lid = gl_LocalInvocationIndex.x;
    uint groupSize = gl_WorkGroupSize.x;

    vec2 min_bounds = Bounds[0];
    vec2 max_bounds = Bounds[1];
    vec2 range = max_bounds - min_bounds;
    vec2 inv_range = 1.0 / range;

    float sum_Q = 0;
    for (uint i = lid; i < num_points; i += groupSize)
    {
      // Position of point in range 0 to 1
      vec2 point = (Positions[i] - min_bounds) * inv_range;

      // Bilinearly sample the input texture
      vec4 v = texture(fields, point);
      sum_Q += max(v.x - 1, 0.0);
      Values[i] = v;
    }

    // Reduce add sum_Q to a single value
    //uint reduction_size = 64;
    if (lid >= 64) {
      reduction_array[lid - 64] = sum_Q;
    }
    barrier();
    if (lid < 64) {
      reduction_array[lid] += sum_Q;
    }
    barrier();
    if (lid < 32) {
      reduction_array[lid] += reduction_array[lid + 32];
    }
    barrier();
    if (lid < 16) {
      reduction_array[lid] += reduction_array[lid + 16];
    }
    barrier();
    if (lid < 8) {
      reduction_array[lid] += reduction_array[lid + 8];
    }
    barrier();
    if (lid < 4) {
      reduction_array[lid] += reduction_array[lid + 4];
    }
    barrier();
    if (lid < 2) {
      reduction_array[lid] += reduction_array[lid + 2];
    }
    barrier();
    if (lid < 1) {
      Sum[0] = reduction_array[0] + reduction_array[1];
    }
  }
);

const char* compute_forces_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer Neigh { uint Neighbours[]; };
  layout(std430, binding = 2) buffer Prob { float Probabilities[]; };
  layout(std430, binding = 3) buffer Ind { int Indices[]; };
  layout(std430, binding = 4) buffer Fiel { vec4 Fields[]; };
  layout(std430, binding = 5) buffer Grad { vec2 Gradients[]; };
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

  const uint group_size = 64;
  shared vec2 sum_positive_red[group_size];

  //layout(rg32f) uniform image2D point_tex;
  uniform uint num_points;
  uniform float exaggeration;
  uniform float sum_Q;

  void main() {
    uint i = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint groupSize = gl_WorkGroupSize.x;
    uint lid = gl_LocalInvocationID.x;

    float inv_num_points = 1.0 / float(num_points);
    float inv_sum_Q = 1.0 / sum_Q;

    if (i >= num_points)
      return;

    // Get the point coordinates
    vec2 point_i = Positions[i];

    //computing positive forces
    vec2 sum_positive = vec2(0);

    int index = Indices[i * 2 + 0];
    int size = Indices[i * 2 + 1];

    vec2 positive_force = vec2(0);
    for (uint j = lid; j < size; j += group_size) {
      // Get other point coordinates
      vec2 point_j = Positions[Neighbours[index + j]];

      // Calculate 2D distance between the two points
      vec2 dist = point_i - point_j;

      // Similarity measure of the two points
      float qij = 1 / (1 + dist.x*dist.x + dist.y*dist.y);

      // Calculate the attractive force
      positive_force += Probabilities[index + j] * qij * dist * inv_num_points;
    }

    // Reduce add sum_positive_red to a single value
    if (lid >= 32) {
      sum_positive_red[lid - 32] = positive_force;
    }
    barrier();
    if (lid < 32) {
      sum_positive_red[lid] += positive_force;
    }
    for (uint reduceSize = group_size/4; reduceSize > 1; reduceSize /= 2)
    {
      barrier();
      if (lid < reduceSize) {
        sum_positive_red[lid] += sum_positive_red[lid + reduceSize];
      }
    }
    barrier();
    if (lid < 1) {
      sum_positive = sum_positive_red[0] + sum_positive_red[1];

      // Computing repulsive forces
      vec2 sum_negative = Fields[i].yz * inv_sum_Q;

      Gradients[i] = 4 * (exaggeration * sum_positive - sum_negative);
    }
  }
);

const char* update_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ float Positions[]; };
  layout(std430, binding = 1) buffer GradientLayout { float Gradients[]; };
  layout(std430, binding = 2) buffer PrevGradientLayout { float PrevGradients[]; };
  layout(std430, binding = 3) buffer GainLayout { float Gain[]; };
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

  uniform uint num_points;
  uniform float eta;
  uniform float minGain;
  uniform float iter;
  uniform float mom_iter;
  uniform float mom;
  uniform float final_mom;
  uniform float mult;

  void main() {
    uint workGroupID = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint i = workGroupID * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i >= num_points * 2)
      return;

    float grad = Gradients[i];
    float pgrad = PrevGradients[i];
    float gain = Gain[i];

    gain = sign(grad) != sign(pgrad) ? gain + 0.2 : gain * 0.8;
    gain = max(gain, minGain);

    float etaGain = eta * gain;
    grad = (grad > 0 ? 1 : -1) * abs(grad * etaGain) / etaGain;

    pgrad = (iter < mom_iter ? mom : final_mom) * pgrad - etaGain * grad;

    Gain[i] = gain;
    PrevGradients[i] = pgrad;
    Positions[i] += pgrad * mult;

    // TODO: 
    // clamp y-axis only
    // TODO: add input buffer for range_limit
    // if (i * 2 + 1 < num_points * 2) {
    //   Positions[i * 2 + 1] = clamp(Positions[i * 2 + 1], 3, 5); // Clamp y-axis only
    // }
    // Compute x-axis range
    // float x_min = positions[0];
    // float x_max = positions[0];

    // for (int i = 0; i < num_points; ++i) {
    //     float x_value = positions[i * 2]; // x-axis values are at even indices
    //     x_min = min(x_min, x_value);
    //     x_max = max(x_max, x_value);
    // }

    // // TODO: input ranges as percentages
    // float x_range = x_max - x_min;
    // float segment_size = x_range / 10.0;

    // // Step 3: Clamp the y-axis based on the segment index
    // for (int i = 0; i < num_points; ++i) {
    //     float x_value = positions[i * 2];      // x-axis value of the current point
    //     float y_value = positions[i * 2 + 1]; // y-axis value of the current point

    //     // Determine the segment index
    //     int segment_index = int((x_value - x_min) / segment_size);
    //     segment_index = clamp(segment_index, 0, 9); // Ensure the index is within [0, 9]

    //     // Calculate the clamping range for y-axis
    //     float y_min = x_min + segment_index * segment_size;
    //     float y_max = x_min + (segment_index + 1) * segment_size;

    //     // Clamp the y-axis
    //     positions[i * 2 + 1] = clamp(y_value, y_min, y_max);
    // }

  }
);

const char* bounds_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { vec2 Bounds[]; };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  shared vec2 min_reduction[64];
  shared vec2 max_reduction[64];

  uniform uint num_points;
  uniform float padding;

  void main() {
    uint lid = gl_LocalInvocationIndex.x;
    uint groupSize = gl_WorkGroupSize.x;

    vec2 minBound = vec2(1e38);//1.0 / 0.0); // inf
    vec2 maxBound = vec2(-1e38);//-1.0 / 0.0); // -inf

    for (uint i = lid; i < num_points; i += groupSize)
    {
      vec2 pos = Positions[i];

      minBound = min(pos, minBound);
      maxBound = max(pos, maxBound);
    }

    // Reduce bounds
    if (lid >= 64) {
      min_reduction[lid - 64] = minBound;
      max_reduction[lid - 64] = maxBound;
    }
    barrier();
    if (lid < 64) {
      min_reduction[lid] = min(minBound, min_reduction[lid]);
      max_reduction[lid] = max(maxBound, max_reduction[lid]);
    }
    barrier();
    if (lid < 32) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 32]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 32]);
    }
    barrier();
    if (lid < 16) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 16]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 16]);
    }
    barrier();
    if (lid < 8) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 8]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 8]);
    }
    barrier();
    if (lid < 4) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 4]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 4]);
    }
    barrier();
    if (lid < 2) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 2]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 2]);
    }
    barrier();
    if (lid == 0) {
      minBound = min(min_reduction[0], min_reduction[1]);
      maxBound = max(max_reduction[0], max_reduction[1]);

      vec2 padding = (maxBound - minBound) * padding * 0.5;

      minBound -= padding;
      maxBound += padding;

      Bounds[0] = minBound;
      Bounds[1] = maxBound;
    }
  }
);

const char* center_and_scale_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { vec2 Bounds[]; };
  // layout(std430, binding = 2) buffer RangeLimitInterface { vec2 RangeLimit[]; };  // range_limit input here (in percentages)

  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  uniform uint num_points;
  uniform bool scale;
  uniform float diameter;

  void main() {
    uint workGroupID = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint i = workGroupID * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i >= num_points)
      return;

    vec2 center = (Bounds[0] + Bounds[1]) * 0.5;

    vec2 pos = Positions[i];
    float range = Bounds[1].x - Bounds[0].x;

    if (scale)
    {
      if (range < diameter) //  || range.y < diameter
      {
        float scale_factor = diameter / range;
        pos -= center;
        pos *= scale_factor;
      }
    }
    else
    {
      pos -= center;
    }

    Positions[i] = pos;
  }
);



const char* dimenfix_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos { vec2 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { vec2 Bounds[]; };
  layout(std430, binding = 2) buffer RangeLimitInterface { vec2 RangeLimit[]; };  // range_limit input here (in percentages)
  layout(std430, binding = 3) buffer ClassBoundsInterface { vec2 ClassBounds[]; };  // class bounds input here (if mode is rescale)

  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  
  uniform uint num_points;
  uniform uint mode;
  
  void main() {
    uint workGroupID = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint i = workGroupID * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i >= num_points)
      return;

    vec2 pos = Positions[i];
    float range = Bounds[1].y - Bounds[0].y;
    vec2 range_limit = RangeLimit[i];
    float range_size = range * (range_limit.y - range_limit.x) / 100.0f;
    float scaled_l = Bounds[0].y + range * range_limit.x / 100.0f;
    float scaled_u = scaled_l + range_size;

    // to integrate rescaling, push first then align xy axis size
    if (mode == 0) { // clipping
      pos.y = clamp(pos.y, scaled_l, scaled_u);
    }
    else if (mode == 1) { // TODO: gaussian
      pos.y *= 1.0f;
    }
    else if (mode == 2) { // rescale
      vec2 class_bound = ClassBounds[i];
      float scale = range_size / (class_bound.y - class_bound.x);
      float new_center = (scaled_l + scaled_u) / 2;
      float old_center = (class_bound.x + class_bound.y) / 2;
      pos.y = (pos.y - old_center) * scale + new_center;
    }

    // resize y according to x
    float x_range = Bounds[1].x - Bounds[0].x;
    float y_range = Bounds[1].y - Bounds[0].y;
    float factor = x_range / y_range;
    pos.y *= factor;

    Positions[i] = pos;
  }
);

const char* class_bounds_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos { vec2 Positions[]; };
  layout(std430, binding = 1) buffer Labels { uint ClassLabels[]; };
  layout(std430, binding = 2) buffer BoundsInterface { vec2 ClassBounds[]; };

  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  
  uniform uint num_points;
  uniform uint num_classes;
  uniform float padding;
  
  shared vec2 min_reduction[64];
  shared vec2 max_reduction[64];
  shared float histogram[64]; // Histogram bins for y-values
  
  void main() {
    uint lid = gl_LocalInvocationIndex;
    uint groupSize = gl_WorkGroupSize.x;
    
    // for (uint class_id = 0; class_id < num_classes; class_id++) {
    //   vec2 minBound = vec2(1e38);
    //   vec2 maxBound = vec2(-1e38);
      
    //   // Initialize histogram bins
    //   for (uint i = lid; i < 64; i += groupSize) {
    //     histogram[i] = 0;
    //   }
    //   barrier();
      
    //   // First pass: Find min/max bounds per class and populate histogram
    //   for (uint i = lid; i < num_points; i += groupSize) {
    //     if (ClassLabels[i] == class_id) {
    //       vec2 pos = Positions[i];
    //       minBound = min(pos, minBound);
    //       maxBound = max(pos, maxBound);
          
    //       // Compute histogram bin (assuming normalized range -1 to 1 for simplicity)
    //       uint bin = uint(clamp((pos.y + 1.0) * 32.0, 0, 63));
    //       // uint temp = floatBitsToUint(histogram[bin]);
    //       // atomicAdd(temp, 1);
    //       // histogram[bin] = uintBitsToFloat(temp);
    //     }
    //   }
    //   barrier();
      
    //   // Reduction step for global min/max bounds
    //   min_reduction[lid] = minBound;
    //   max_reduction[lid] = maxBound;
    //   barrier();
      
    //   for (uint stride = 32; stride > 0; stride /= 2) {
    //     if (lid < stride) {
    //       min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + stride]);
    //       max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + stride]);
    //     }
    //     barrier();
    //   }
      
    //   // Compute cumulative histogram to determine 95% bounds
    //   if (lid == 0) {
    //     float total_count = 0;
    //     for (uint i = 0; i < 64; i++) total_count += histogram[i];
        
    //     float lower_thresh = total_count * 0.025;
    //     float upper_thresh = total_count * 0.975;
        
    //     float cumsum = 0;
    //     float lower_bound = -1.0;
    //     float upper_bound = 1.0;
        
    //     for (uint i = 0; i < 64; i++) {
    //       cumsum += histogram[i];
    //       if (cumsum >= lower_thresh && lower_bound == -1.0) lower_bound = (i / 32.0) - 1.0;
    //       if (cumsum >= upper_thresh) {
    //         upper_bound = (i / 32.0) - 1.0;
    //         break;
    //       }
    //     }
        
    //     vec2 minBound = min_reduction[0];
    //     vec2 maxBound = max_reduction[0];
        
    //     vec2 padding_vec = (maxBound - minBound) * padding * 0.5;
    //     minBound -= padding_vec;
    //     maxBound += padding_vec;
        
    //     ClassBounds[class_id] = vec4(minBound.y, maxBound.y, lower_bound, upper_bound);
    //   }
    // }
  }
);