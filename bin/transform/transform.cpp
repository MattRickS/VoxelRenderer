kernel VTransform : ImageComputationKernel<ePixelWise>
{
  Image<eRead> voxels;
  Image<eWrite> dst;

  param:
    float4x4 M;
  
  float4 multVectMatrix(float4 vec, float4x4 M)
  {
    float4 out = float4(
      vec.x * M[0][0] + vec.y * M[0][1] + vec.z * M[0][2] + M[0][3],
      vec.x * M[1][0] + vec.y * M[1][1] + vec.z * M[1][2] + M[1][3],
      vec.x * M[2][0] + vec.y * M[2][1] + vec.z * M[2][2] + M[2][3],
      vec.w
    );

    return out;
  }

  void process()
  {
    float4 position = voxels();
    dst() = multVectMatrix(position, M);
  } 
};