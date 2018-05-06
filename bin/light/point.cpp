kernel VPtLight : ImageComputationKernel<ePixelWise>
{
  Image<eRead, eAccessRandom> voxels;
  Image<eWrite> dst;

  param:
    float3 bbox_min;
    float3 bbox_max;
    int3 resolution;
    float3 light;
    float intensity;
    float absorption;
    int samples;
    float4x4 transform_M;

  local:
    float3 aabb[2];
    float3 VoxelSizeInv;
    float3 _light;
    int grid_width;
    int _samples;

  void define()
  {
    defineParam(intensity, "intensity", 1.0f);
    defineParam(absorption, "absorption", 0.5f);
    defineParam(samples, "samples", 50);
  }

  void init()
  {
    aabb[0] = bbox_min;
    aabb[1] = bbox_max;

    for (int c = 0; c < 3; c++)
      VoxelSizeInv[c] = resolution[c] / (bbox_max[c] - bbox_min[c]);

    _light = multVectMatrix(light, transform_M.invert());

    grid_width = voxels.bounds.width();
    _samples = max(1, samples);
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

  // Axis aligned bounding box intersection for GPU : Shortened for exit only
  float intersection_exit_AABB(float3 origin, float3 inv_dir)
  {
    //bool sign[3] {inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0};
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    bool sign = inv_dir.x < 0 ? 0 : 1;
    tmax  = (aabb[sign].x - origin.x) * inv_dir.x;
    sign = inv_dir.y < 0 ? 0 : 1;
    tymax = (aabb[sign].y - origin.y) * inv_dir.y;
    sign = inv_dir.z < 0 ? 0 : 1;
    tzmax = (aabb[sign].z - origin.z) * inv_dir.z;
    tmax = min(min(tmax, tymax), tzmax);
    return tmax;
  }

  float voxelValue(int3 voxel)
  {
    // Return Empty for values outside of grid
    if (voxel.x < 0 || resolution.x <= voxel.x ||
        voxel.y < 0 || resolution.y <= voxel.y ||
        voxel.z < 0 || resolution.z <= voxel.z )
      return 0;

    int id = (voxel.y * resolution.x + voxel.x) * resolution.z + voxel.z;
    int x = id % grid_width;
    int y = id / grid_width;
    return voxels(x, y)[3]; // Just alpha (density)
  }

  float Blend(float3 curpos)
  {
    float3 weight, voxel_space;
    int3 base_voxel, offset;
    // Lower Bound of 8 adjacent voxels
    for (int c = 0; c < 3; c++)
    {
      voxel_space[c] = curpos[c] * VoxelSizeInv[c];
      base_voxel[c] = int(floor(voxel_space[c] - 0.5f));
    }

    float result = 0;
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

  void process(int2 pos)
  {
    // Skip empty voxels
    float4 v = voxels(pos.x, pos.y);
    if (v.w <= 0)
      return;

    // Starting values
    float3 voxel = float3(v.x, v.y, v.z); // Center of voxel
    float3 dir = voxel - _light; // from light to voxel
    float dist_to_light = length(dir);
    dir /= dist_to_light;

    // Fire ray towards light, get closest distance
    float max_dist = min(dist_to_light, intersection_exit_AABB(voxel, -1.0f / dir));

    // ========== Ray Marching ==========

    // Maximum of 1000 samples per voxel
    float deltaT, result = intensity, step = max(max_dist / _samples, 0.001f);
    float3 curpos;

    for (float dist = 0; dist <= max_dist; dist += step)
    {
      curpos = voxel - dir * dist;
      deltaT = exp(-absorption * Blend(curpos - bbox_min) * step);
      result *= deltaT;

      // End loop if Transmittance is near 0
      if (result < 1e-6)
      	break;
      //dist += (result < 1e-6) * max_dist;
    }

    dst() = result;

  }

};