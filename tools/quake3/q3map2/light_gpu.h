/* -------------------------------------------------------------------------------

   -gpu: traces q3map2 -light rays on a Vulkan GPU with ray queries (build with Q3MAP2_VULKAN=yes)

   the GPU gets the same triangles and tests as the -embree path (see light_trace.cpp), and rays
   are traced in batches: a caller collects many rays, traces them at once and reads the results.
   these structs are shared with the shader (light_gpu_trace.comp), keep both in sync.

   ------------------------------------------------------------------------------- */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/* a ray: the inputs of TraceLine() after SetupTrace() */
struct GpuRay
{
	float origin[ 3 ];
	float inhibitRadius;
	float end[ 3 ];
	float distance;
	float direction[ 3 ];
	int recvShadows;
	float color[ 3 ];
	int lightSkyIndex;                      /* the light's sky index if sky hits count per sky, or -1 */
	int surfacesOffset, numSurfaces;        /* the trace's surfaces (no self-shadowing) in the batch's surface list */
	uint32_t flags;                         /* GPU_RAY_* */
	int pad;
};

enum
{
	GPU_RAY_TEST_ALL = 1,                   /* trace->testAll */
	GPU_RAY_TREE_ORDER = 2,                 /* no light (dirt, floodlight): the opaque triangle TraceLine() finds first, not the nearest */
	GPU_RAY_IN_GRID = 4,                    /* inGrid */
	GPU_RAY_NO_SURFACES = 8,                /* noSurfaces */
	GPU_RAY_SKIP = 16,                      /* TraceLine() returns before tracing */
};

/* the result of a ray */
struct GpuHit
{
	float hit[ 3 ];                         /* GPU_HIT_POINT: trace->hit */
	float depth;                            /* GPU_HIT_DEPTH: trace->hit = origin + direction * depth */
	float color[ 3 ];
	float pad0;
	int compileFlags;
	uint32_t status;                        /* GPU_HIT_* */
	uint32_t skyBits;                       /* trace->skyIndices */
	uint32_t pad1;
};

enum
{
	GPU_HIT_OPAQUE = 1,
	GPU_HIT_PASS_SOLID = 2,
	GPU_HIT_POINT = 4,
	GPU_HIT_DEPTH = 8,
	GPU_HIT_FORCE_SUBSAMPLING = 16,
	GPU_HIT_OVERFLOW = 32,                  /* the shader could not finish the ray: trace it on the CPU */
};

/* scene data, built from the trace nodes by light_trace.cpp */
struct GpuTriangle
{
	float v0[ 3 ];
	int info;                               /* index into infos */
	float edge1[ 3 ];
	int pathOffset;                         /* the leaf's path from the root: GpuScene::paths[ pathOffset .. + pathLength ] */
	float edge2[ 3 ];
	int pathLength;
	float st[ 3 ][ 2 ];
	int item;                               /* position in the leaf's items */
	int pad;
};

enum
{
	GPU_INFO_OPAQUE = 0,
	GPU_INFO_SKY = 1,
	GPU_INFO_FILTER = 2,
	GPU_INFO_FILTER_COLOR = 4,              /* with GPU_INFO_FILTER: lightfilter */
	GPU_INFO_FILTER_SHADOW = 8,             /* with GPU_INFO_FILTER: alphashadow */
};

struct GpuInfo
{
	int castShadows, surfaceNum, skipGrid, compileFlags;
	int skyIndex;
	int kind;                               /* GPU_INFO_* */
	int imageOffset;                        /* in GpuScene::images (RGBA8 texels) */
	int imageSize;                          /* width << 16 | height */
};

struct GpuNode
{
	float normal[ 3 ];
	float dist;
	int type, children[ 2 ], numItems;
};

struct GpuScene
{
	std::vector<float> vertices;            /* the enlarged triangles for the acceleration structure, 9 floats each */
	std::vector<GpuTriangle> triangles;
	std::vector<GpuInfo> infos;
	std::vector<GpuNode> nodes;
	std::vector<uint32_t> paths;            /* node | side << 31, root first */
	std::vector<uint32_t> images;
	int headNode, numBspNodes;
	uint32_t skyFlag;                       /* C_SKY */
};

/* light_gpu.cpp */
bool GpuSetup( const GpuScene& scene );     /* false: no usable GPU */
void GpuTrace( const GpuRay *rays, GpuHit *hits, size_t count, const int *surfaces, size_t numSurfaces ); /* thread safe */
void GpuShutdown();

/* gathers: a set of rays per sample made by the GPU from tangent space vectors (dirt, floodlight),
   summed per sample like DirtForSample() / FloodLightForSample() do */
enum
{
	GPU_GATHER_DIRT = 0,                    /* dirtMode 0: gatherDirt */
	GPU_GATHER_FLOODLIGHT = 1,              /* gatherLight */
};

struct GpuGatherSample
{
	float origin[ 4 ], normal[ 4 ], right[ 4 ], up[ 4 ]; /* right / up: the tangent space, w unused */
};

struct GpuGatherOut
{
	float gather;                           /* the sum, before the final formula */
	uint32_t flags;                         /* GPU_GATHER_OVERFLOW: compute the sample on the CPU */
};

enum
{
	GPU_GATHER_OVERFLOW = 1,
};

struct GpuGatherParams
{
	int mode;                               /* GPU_GATHER_* */
	float rayLength;
	float inhibitRadius;
	float ooDepth;                          /* dirt: 1 / dirtDepth */
	int recvShadows;
	uint32_t rayFlags;                      /* GPU_RAY_IN_GRID, GPU_RAY_NO_SURFACES */
	uint32_t translucentFlag;               /* C_TRANSLUCENT */
	const int *surfaces;
	int numSurfaces;
	const float *vectors;                   /* xyz each */
	int numVectors;
};

void GpuGather( const GpuGatherParams& params, const GpuGatherSample *samples, size_t count, GpuGatherOut *outs ); /* thread safe */

/* about how many rays callers put in a batch */
constexpr int GPU_BATCH_RAYS = 1 << 18;

/* light_trace.cpp: rays collected from TraceLine() calls, traced on the GPU at once */
struct trace_t;
class GpuTraceBatch
{
	std::vector<GpuRay> m_rays;
	std::vector<GpuHit> m_hits;
	std::vector<int> m_surfaces;
	const int *m_lastSurfaces = nullptr;
	int m_lastNumSurfaces = -1;
	int m_lastOffset = 0;
	size_t m_next = 0;
public:
	void add( const trace_t& trace );       /* a trace ready for TraceLine() */
	void run();
	/* TraceLine() of the next ray added: trace has the inputs of the added one but may differ in
	   origin / end, which are restored from the ray */
	void apply( trace_t *trace );
	size_t size() const {
		return m_rays.size();
	}
	void clear();
};

bool GpuTraceEnabled();                    /* -gpu and a GPU was set up */
void GpuShutdownTrace();
