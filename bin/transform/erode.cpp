kernel VErode : ImageComputationKernel<ePixelWise>
{
  Image<eRead, eAccessRandom> voxels;
  Image<eWrite> dst;

  param:
    float3 bbox_min;
    float3 bbox_max;
    int3 resolution;
    float size;

  local:
    float3 VoxelSizeInv;
    bool negative;
    int grid_width;
    int range;
    float max_length;

  void init()
  {
    for (int c = 0; c < 3; c++)
      VoxelSizeInv[c] = resolution[c] / (bbox_max[c] - bbox_min[c]);
    negative = size < 0;
    grid_width = voxels.bounds.width();
    range = negative ? floor(size) : ceil(size);
    max_length = size * size;
  }

  float voxelValue(int3 voxel)
  {
    int id = (voxel.y * resolution.x + voxel.x) * resolution.z + voxel.z;
    int x = id % grid_width;
    int y = id / grid_width;
    return voxels(x, y)[3]; // Just alpha (density)
  }

  void process(int2 pos)
  {
    float4 current = voxels(pos.x, pos.y);
    int3 voxel;
    for (int c = 0; c < 3; c++)
      voxel[c] = int((current[c] - bbox_min[c]) * VoxelSizeInv[c]);

    float length, new_val = voxelValue(voxel);
    if (negative)
    {
      // Minimum value within specified range
      for (int i = max(voxel.x + range, 0); i <= min(voxel.x - range, resolution.x - 1); i++)
        for (int j = max(voxel.y + range, 0); j <= min(voxel.y - range, resolution.y - 1); j++)
          for (int k = max(voxel.z + range, 0); k <= min(voxel.z - range, resolution.z - 1); k++)
          {
            length = pow(i - voxel.x, 2) + pow(j - voxel.y, 2) + pow(k - voxel.z, 2);
            new_val = min(new_val, voxelValue(int3(i, j, k)) + 10000.0f * (length > max_length));
          }
    }
    else
    {
      // Maximum value within specified range
      for (int i = max(voxel.x - range, 0); i <= min(voxel.x + range, resolution.x - 1); i++)
        for (int j = max(voxel.y - range, 0); j <= min(voxel.y + range, resolution.y - 1); j++)
          for (int k = max(voxel.z - range, 0); k <= min(voxel.z + range, resolution.z - 1); k++)
          {
            length = pow(i - voxel.x, 2) + pow(j - voxel.y, 2) + pow(k - voxel.z, 2);
            new_val = max(new_val, voxelValue(int3(i, j, k)) - 10000.0f * (length > max_length));
          }
    }

    dst() = current;
    dst(3) = new_val;
  }
};