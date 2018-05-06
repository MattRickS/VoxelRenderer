kernel VEnvLight : ImageComputationKernel<ePixelWise>
{
  Image<eRead, eAccessRandom> voxels;
  Image<eRead, eAccessRandom> environment;
  Image<eWrite> dst;

  param:
    float3 bbox_min;
    float3 bbox_max;
    int3 resolution;
    bool luminance_on;
    float rotate;
    float intensity;
    float absorption;
    int samples;
    float4x4 transform_M;

  local:
    float3 aabb[2];
    float3 VoxelSizeInv;
    float _absorption;
    int grid_width;
    float rad2deg;
    int _samples;

  void define()
  {
    defineParam(intensity, "intensity", 1.0f);
    defineParam(absorption, "absorption", 0.5f);                   // Absorption? Rate of Decay
    defineParam(samples, "samples", 50);
  }

  void init()
  {
    aabb[0] = bbox_min;
    aabb[1] = bbox_max;
    grid_width = voxels.bounds.width();

    for (int c = 0; c < 3; c++)
      VoxelSizeInv[c] = resolution[c] / (bbox_max[c] - bbox_min[c]);

    _absorption = 1 / (1 - absorption);
    rad2deg = 180.0f / PI;
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
  float intersection_exit_AABB(float3 origin, float3 inv_dir, float3 bbox[2])
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


    // ========== AMBIENT LIGHT ==========

    // Fire a ray from outer sphere to voxel, accumulate fog

    // Starting values
    float3 voxel = float3(v.x, v.y, v.z); // Center of voxel

    int3 voxel_i;
    for (int c = 0; c < 3; c++)
      voxel_i[c] = (voxel[c] - bbox_min[c]) * VoxelSizeInv[c];

    float3 normal = float3(
      voxelValue(voxel_i - int3(1,0,0)) - voxelValue(voxel_i + int3(1,0,0)),
      voxelValue(voxel_i - int3(0,1,0)) - voxelValue(voxel_i + int3(0,1,0)),
      voxelValue(voxel_i - int3(0,0,1)) - voxelValue(voxel_i + int3(0,0,1))
      );

    float3 dir = -normalize(normal);
    // Skip directionless voxels
    if (dir.x == 0 && dir.y == 0 && dir.z == 0) return;

    float3 dir_inv = 1.0f / dir;

    // Convert direction to latlong coordinates
    float3 env_dir = normalize(multVectMatrix(dir, transform_M));
    float x = (fmod((rad2deg * atan2(env_dir.z, env_dir.x) + rotate), 360.0f) / 360.0f + 0.5f) * environment.bounds.width();
    float y = ((rad2deg * acos(env_dir.y)) / 180.0f) * environment.bounds.height();

    float4 colour = environment(x, y);
    float luminance = (colour.x * 0.3f + colour.y * 0.59f + colour.z * 0.11f) * intensity;

    // Move starting point to whatever's closest: intersection / light
    float max_dist = intersection_exit_AABB(voxel, -dir_inv, aabb);

    // ========== Ray Marching ==========

    // Maximum of 1000 samples per voxel
    float result = luminance * luminance_on + intensity * (1 - luminance_on);
    float deltaT = 1, step = max(max_dist / _samples, 0.001f);
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

    dst() = result * environment(x, y);
    dst(3) = v.w;
    
  }

};