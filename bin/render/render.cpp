kernel VRender : ImageComputationKernel<ePixelWise>
{
  Image<eRead, eAccessRandom> voxels;
  Image<eWrite> dst;

  param:
    int3 resolution;
    float3 bbox_min;
    float3 bbox_max;
    float density;
    int samples;
    // Camera Parameters
    float width;
    float height;
    float focal;
    float haperture;
    float4x4 cam_M;
    float4x4 transform_M;

  local:
    float3 aabb[2];
    float3 voxelHalfSize;
    float3 VoxelSizeInv;
    int grid_width;
    int _samples;
    // Camera Parameters
    float ratio;
    float3 camera;
    float3 up;
    float3 right;
    float3 forward;

  void define()
  {
    defineParam(resolution, "Resolution", int3(10,10,10));
    defineParam(density, "density", 1.0f);
    defineParam(samples, "samples", 50);
    // Camera Parameters
    defineParam(width, "Width", 1440.0f);
    defineParam(height, "Height", 810.0f);
    defineParam(focal, "Focal", 30.0f);
    defineParam(haperture, "Haperture", 24.576f);
  }

  void init()
  {
    aabb[0] = bbox_min;
    aabb[1] = bbox_max;
    for (int c = 0; c < 3; c++)
      VoxelSizeInv[c] = resolution[c] / (bbox_max[c] - bbox_min[c]);
    voxelHalfSize = 0.5f / VoxelSizeInv;
    grid_width = voxels.bounds.width();
    _samples = max(1, samples);

    float4x4 camM = transform_M.invert() * cam_M;

    // Camera Parameters
    ratio   = focal / haperture; 
    camera  = float3(camM[0][3], camM[1][3], camM[2][3]);
    up      = normalize(multVectMatrix(float3(0.0f, 1.0f, 0.0f), camM) - camera) * (height / width);
    right   = normalize(multVectMatrix(float3(1.0f, 0.0f, 0.0f), camM) - camera);
    forward = normalize(multVectMatrix(float3(0.0f, 0.0f, -1.0f), camM) - camera); 
  }

  float3 multVectMatrix(float3 vec, float4x4 M)
  {
    float3 out = float3(
      vec.x * M[0][0] + vec.y * M[0][1] + vec.z * M[0][2] + M[0][3],
      vec.x * M[1][0] + vec.y * M[1][1] + vec.z * M[1][2] + M[1][3],
      vec.x * M[2][0] + vec.y * M[2][1] + vec.z * M[2][2] + M[2][3]
    );

    return out;
  }

  float4 voxelValue(int3 voxel)
  {
    // Return Empty for values outside of grid
    if (voxel.x < 0 || resolution.x <= voxel.x ||
        voxel.y < 0 || resolution.y <= voxel.y ||
        voxel.z < 0 || resolution.z <= voxel.z )
      return 0;

    int id = (voxel.y * resolution.x + voxel.x) * resolution.z + voxel.z;
    int x = id % grid_width;
    int y = id / grid_width;

    return voxels(x, y);
  }
  

  float4 Blend(float3 curpos)
  {
    float3 weight, voxel_space;
    int3 base_voxel, offset;
    // Lower Bound of 8 adjacent voxels
    for (int c = 0; c < 3; c++)
    {
      voxel_space[c] = curpos[c] * VoxelSizeInv[c];
      base_voxel[c] = int(floor(voxel_space[c] - 0.5f));
    }

    float4 result = 0;
    for (int i = 0; i < 8; i++)
    {
      // Weighted distance on each axis for each adjacent voxel
      offset = int3(i / 4, (i / 2) % 2, i % 2);
      weight.x = 1 - fabs(offset.x - (voxel_space.x - (base_voxel.x + 0.5f)));
      weight.y = 1 - fabs(offset.y - (voxel_space.y - (base_voxel.y + 0.5f)));
      weight.z = 1 - fabs(offset.z - (voxel_space.z - (base_voxel.z + 0.5f)));

      result += voxelValue(base_voxel + offset) * weight.x * weight.y * weight.z;
    }

    return result;
  }


  float2 intersection_AABB(float3 origin, float3 inv_dir)
  {
    // Axis aligned bounding box intersection for GPU
    // aabb = min_corner, max_corner (eg, (0,0,0), (1,1,1))
    // ray =  origin, direction, inv_dir (1/dir), sign (x < 0 ? 1 : 0)
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    bool sign = inv_dir.x < 0 ? 0 : 1;
    tmin  = (aabb[1 - sign].x - origin.x) * inv_dir.x;
    tmax  = (aabb[sign].x - origin.x) * inv_dir.x;
    sign = inv_dir.y < 0 ? 0 : 1;
    tymin = (aabb[1 - sign].y - origin.y) * inv_dir.y;
    tymax = (aabb[sign].y - origin.y) * inv_dir.y;
    sign = inv_dir.z < 0 ? 0 : 1;
    tzmin = (aabb[1 - sign].z - origin.z) * inv_dir.z;
    tzmax = (aabb[sign].z - origin.z) * inv_dir.z;
    tmin = max(max(tmin, tymin), tzmin);
    tmax = min(min(tmax, tymax), tzmax);
    return float2(max(tmin, 0.0f), tmax);
  }


  void process(int2 pos)
  {
    // Calculating the eye vector
    float u = pos.x / width - 0.5f;
    float v = pos.y / height - 0.5f;

    // Ray direction
    float3 dir = normalize(forward * ratio + right * u + up * v);

    // ========== Box Intersection (Entry and Exit) ==========

    float2 tMinMax = intersection_AABB(camera, 1.0f / dir);

    bool hit = (tMinMax.x < tMinMax.y);// && (tMinMax.x >= near_clip_plane) && (tMinMax.y <= far_clip_plane)
    if (!hit)
    {
      dst() = 0;
      return;
    }

    // Maximum of 1000 samples per voxel
    float max_dist = tMinMax.y - tMinMax.x;
    float deltaT, T = 1, step = max(max_dist / _samples, 0.001f);
    float3 curpos;
    float4 result = 0, colour;

    for (float dist = tMinMax.x; dist <= tMinMax.y; dist += step)
    {
      curpos = camera + dir * dist;
      colour = Blend(curpos - bbox_min);
      deltaT = exp(-density * colour.w * step);
      T *= deltaT;
      result += (1 - deltaT) * colour * T;

      // End loop if Transmittance is near 0
      if (T < 1e-6)
        break;
    }

    dst() = result;
    dst(3) = 1 - T;
  }
};