kernel VGrid : ImageComputationKernel<ePixelWise>
{
  Image<eRead> format;
  Image<eWrite> dst;

  param:
    float3 bbox_min; // from origin
    float3 bbox_max; // from origin
    int3 resolution;
    int total;

  local:
    float3 scale;

  void init()
  {
    for (int c = 0; c < 3; c++)
      scale[c] = (bbox_max[c] - bbox_min[c]) / resolution[c];
  }

  void process(int2 pos)
  {
    int id = pos.y * dst.bounds.width() + pos.x;
    if (id >= total)
    {
      dst() = 0;
      return;
    }
    int xy = id / resolution.z;
    dst(0) = ((xy % resolution.x) + 0.5f) * scale.x + bbox_min.x;
    dst(1) = ((xy / resolution.x) + 0.5f) * scale.y + bbox_min.y;
    dst(2) = ((id % resolution.z) + 0.5f) * scale.z + bbox_min.z;
    dst(3) = 1;
  }
};