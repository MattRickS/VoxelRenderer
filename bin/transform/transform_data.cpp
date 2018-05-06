kernel VTransformData : ImageComputationKernel<ePixelWise>
{
  Image<eRead> positions;
  Image<eRead, eAccessRandom> voxels;
  Image<eWrite> dst;

  param:
    float3 bbox_min;
    float3 bbox_max;
    int3 resolution;
    float4x4 transform_M;

  local:
    float3 VoxelSizeInv;
    int grid_width;
    float4x4 transform_M_Inv;

  void init()
  {
    for (int c = 0; c < 3; c++)
      VoxelSizeInv[c] = resolution[c] / (bbox_max[c] - bbox_min[c]);
    grid_width = voxels.bounds.width();
    transform_M_Inv = transform_M.invert();
    // To transform in world space, invert the world_matrix, and multiply ... after?
    // transform_M_Inv * world_M_Inv;
  }
  
  float3 multVectMatrix(float4 vec, float4x4 M)
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

  void process()
  {
    float4 position = positions();
    float3 target = multVectMatrix(position, transform_M_Inv);
    dst() = Blend(target - bbox_min);
  } 
};