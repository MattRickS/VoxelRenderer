kernel VAverage : ImageComputationKernel<ePixelWise>
{
  Image<eRead, eAccessRandom> colour;
  Image<eRead, eAccessRandom> voxels;
  Image<eWrite> dst;

  param:
    float4 weight;
    float3 bbox_min;
    float3 bbox_max;
    int3 resolution;
    int range;

  local:
    float4 _weight;
    float3 VoxelSizeInv;
    int grid_width;
    float max_length;

  void init()
  {
    for (int c = 0; c < 3; c++)
    {
      VoxelSizeInv[c] = resolution[c] / (bbox_max[c] - bbox_min[c]);
      _weight[c] = clamp(weight[c], 0.0f, 1.0f);
    }
    _weight.w = clamp(weight.w, 0.0f, 1.0f);
    grid_width = voxels.bounds.width();
    max_length = range * range;
  }

  float4 voxelValue(int3 voxel)
  {
    int id = (voxel.y * resolution.x + voxel.x) * resolution.z + voxel.z;
    int x = id % grid_width;
    int y = id / grid_width;
    return colour(x, y);
  }

  void process(int2 pos)
  {
    float4 current = voxels(pos.x, pos.y);

    int3 voxel;
    for (int c = 0; c < 3; c++)
      voxel[c] = int((current[c] - bbox_min[c]) * VoxelSizeInv[c]);

    float4 value, result = 0;
    int total = 0;
    float length;

    // Maximum value within specified range
    for (int i = max(voxel.x - range, 0); i <= min(voxel.x + range, resolution.x - 1); i++)
      for (int j = max(voxel.y - range, 0); j <= min(voxel.y + range, resolution.y - 1); j++)
        for (int k = max(voxel.z - range, 0); k <= min(voxel.z + range, resolution.z - 1); k++)
        {
          length = pow(i - voxel.x, 2) + pow(j - voxel.y, 2) + pow(k - voxel.z, 2);
          value = voxelValue(int3(i, j, k)) * (length <= max_length);
          total += (value.w > 0);
          result += value * (value.w > 0);
        }
    
    current = voxelValue(voxel);
    if (total < 1)
      dst() = current;
    else
      for (int c = 0; c < 4; c++)
        dst(c) = (result[c] / total - current[c]) * _weight[c] + current[c];
  }
};