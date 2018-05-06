// C++11
const uint rand_multiplier = 48271;
const uint rand_increment  = 0;
const uint rand_modulus    = 2147483647;

static int lcgRandom( int seed )
{
  return abs( ( seed * rand_multiplier + rand_increment ) % rand_modulus );
}

// ===== Interpolation Functions =====

static float Smooth (const float &t)
{
  // Fifth degree polynomial
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float Lerp (const float &a, const float &b, const float &t)
{
  return (b - a) * clamp(t, 0.0f, 1.0f) + a;
}


// ===== Noise Function =====

static float Perlin3D(float3 point, float frequency)
{
  const int hash[512] = {
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,

    
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180
  };

  const float4 gradients3D[16] = {
    float4( 1.0f,  1.0f,  0.0f, 0.0f),
    float4(-1.0f,  1.0f,  0.0f, 0.0f),
    float4( 1.0f, -1.0f,  0.0f, 0.0f),
    float4(-1.0f, -1.0f,  0.0f, 0.0f),
    float4( 1.0f,  0.0f,  1.0f, 0.0f),
    float4(-1.0f,  0.0f,  1.0f, 0.0f),
    float4( 1.0f,  0.0f, -1.0f, 0.0f),
    float4(-1.0f,  0.0f, -1.0f, 0.0f),
    float4( 0.0f,  1.0f,  1.0f, 0.0f),
    float4( 0.0f, -1.0f,  1.0f, 0.0f),
    float4( 0.0f,  1.0f, -1.0f, 0.0f),
    float4( 0.0f, -1.0f, -1.0f, 0.0f),
    float4( 1.0f,  1.0f,  0.0f, 0.0f),
    float4(-1.0f,  1.0f,  0.0f, 0.0f),
    float4( 0.0f, -1.0f,  1.0f, 0.0f),
    float4( 0.0f, -1.0f, -1.0f, 0.0f)
  };

  const int hashMask = 255;
  const int gradientsMask3D = 15;

  point *= frequency;
  int ix0 = floor(point.x);
  int iy0 = floor(point.y);
  int iz0 = floor(point.z);
  float tx0 = point.x - ix0;
  float ty0 = point.y - iy0;
  float tz0 = point.z - iz0;
  float tx1 = tx0 - 1.0f;
  float ty1 = ty0 - 1.0f;
  float tz1 = tz0 - 1.0f;
  ix0 &= hashMask;
  iy0 &= hashMask;
  iz0 &= hashMask;
  int ix1 = ix0 + 1;
  int iy1 = iy0 + 1;
  int iz1 = iz0 + 1;

  int h0 = hash[ ix0 ];
  int h1 = hash[ ix1 ];
  int h00 = hash[ h0 + iy0 ];
  int h10 = hash[ h1 + iy0 ];
  int h01 = hash[ h0 + iy1 ];
  int h11 = hash[ h1 + iy1 ];

  // float3 fails for some unknown reason. Add an empty fourth
  float4 g000 = gradients3D[ hash[ h00 + iz0 ] & gradientsMask3D ];
  float4 g100 = gradients3D[ hash[ h10 + iz0 ] & gradientsMask3D ];
  float4 g010 = gradients3D[ hash[ h01 + iz0 ] & gradientsMask3D ];
  float4 g110 = gradients3D[ hash[ h11 + iz0 ] & gradientsMask3D ];
  float4 g001 = gradients3D[ hash[ h00 + iz1 ] & gradientsMask3D ];
  float4 g101 = gradients3D[ hash[ h10 + iz1 ] & gradientsMask3D ];
  float4 g011 = gradients3D[ hash[ h01 + iz1 ] & gradientsMask3D ];
  float4 g111 = gradients3D[ hash[ h11 + iz1 ] & gradientsMask3D ];

  float v000 = dot(g000, float4(tx0, ty0, tz0, 0.0f));
  float v100 = dot(g100, float4(tx1, ty0, tz0, 0.0f));
  float v010 = dot(g010, float4(tx0, ty1, tz0, 0.0f));
  float v110 = dot(g110, float4(tx1, ty1, tz0, 0.0f));
  float v001 = dot(g001, float4(tx0, ty0, tz1, 0.0f));
  float v101 = dot(g101, float4(tx1, ty0, tz1, 0.0f));
  float v011 = dot(g011, float4(tx0, ty1, tz1, 0.0f));
  float v111 = dot(g111, float4(tx1, ty1, tz1, 0.0f));

  float tx = Smooth(tx0);
  float ty = Smooth(ty0);
  float tz = Smooth(tz0);
  return Lerp(
    Lerp(Lerp(v000, v100, tx), Lerp(v010, v110, tx), ty),
    Lerp(Lerp(v001, v101, tx), Lerp(v011, v111, tx), ty),
    tz);// * 0.5f + 0.5f;
}


static float Fractal(float3 point, float frequency, int octaves, float lacunarity, float persistence)
{
  
  float sum = 0;
  sum = Perlin3D(point, frequency);
  float amplitude = 1.0f;
  float range = 1.0f;
  for (int o = 1; o < octaves; o++) {
    frequency *= lacunarity;
    amplitude *= persistence;
    range += amplitude;
    sum += Perlin3D(point, frequency) * amplitude;
  }
  return sum / range;
}


kernel VFNoise : ImageComputationKernel<ePixelWise>
{
  Image<eRead> voxels;
  Image<eWrite> dst;

  param:
    int seed;
    float size;
    int octaves;
    float lacunarity;
    float persistence;
    float4x4 M;

  local:
    float3 seed_offset;
    float frequency;
    float4x4 M_inv;

  void define()
  {
    defineParam(size, "Size", 30.0f);
    defineParam(octaves, "Octaves", 4);
    defineParam(lacunarity, "Lacunarity", 2.0f);
    defineParam(persistence, "Persistence", 0.5f);
  }

  void init()
  {
    int random = seed;
    for (int c = 0; c < 3; c++)
    {
      random = lcgRandom(random);
      seed_offset[c] = (random / float(rand_modulus)) * 10000;
    }
    frequency = 1.0f / size;
    M_inv = M.invert();
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


  void process()
  {
    float4 voxel = voxels();
    float3 point = multVectMatrix(voxel, M_inv);
    voxel.w = Fractal(point + seed_offset, frequency, octaves, lacunarity, persistence);
    dst() = voxel;
  }
    
};