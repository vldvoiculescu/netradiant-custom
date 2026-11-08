/* -------------------------------------------------------------------------------

   -gpu: traces q3map2 -light rays on a Vulkan GPU with ray queries (build with Q3MAP2_VULKAN=yes)

   the Vulkan loader is loaded at run time, so q3map2 runs without it; when there is no GPU with
   ray queries, -gpu warns and the CPU traces the rays. the rays are traced by a compute shader
   (light_gpu_trace.comp, embedded as SPIR-V) that repeats the -embree path's tests, see light_trace.cpp

   ------------------------------------------------------------------------------- */

#include "light_gpu.h"

#ifdef Q3MAP2_VULKAN

#include "inout.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <string>

#include "light_gpu_trace.spv.h"

namespace
{
/* a Vulkan struct with its type and the rest zeroed */
template<typename T>
T VkStruct( VkStructureType type ){
	T t{};
	t.sType = type;
	return t;
}

/* Vulkan functions, loaded at run time */
#define GPU_INSTANCE_FUNCTIONS( X ) \
	X( vkDestroyInstance ) \
	X( vkEnumeratePhysicalDevices ) \
	X( vkGetPhysicalDeviceProperties ) \
	X( vkGetPhysicalDeviceProperties2 ) \
	X( vkGetPhysicalDeviceQueueFamilyProperties ) \
	X( vkGetPhysicalDeviceMemoryProperties ) \
	X( vkEnumerateDeviceExtensionProperties ) \
	X( vkGetPhysicalDeviceFeatures2 ) \
	X( vkCreateDevice ) \
	X( vkGetDeviceProcAddr )

#define GPU_DEVICE_FUNCTIONS( X ) \
	X( vkDestroyDevice ) \
	X( vkDeviceWaitIdle ) \
	X( vkGetDeviceQueue ) \
	X( vkCreateBuffer ) \
	X( vkDestroyBuffer ) \
	X( vkGetBufferMemoryRequirements ) \
	X( vkAllocateMemory ) \
	X( vkFreeMemory ) \
	X( vkBindBufferMemory ) \
	X( vkMapMemory ) \
	X( vkInvalidateMappedMemoryRanges ) \
	X( vkGetBufferDeviceAddress ) \
	X( vkCreateCommandPool ) \
	X( vkDestroyCommandPool ) \
	X( vkResetCommandPool ) \
	X( vkAllocateCommandBuffers ) \
	X( vkBeginCommandBuffer ) \
	X( vkEndCommandBuffer ) \
	X( vkCmdBindPipeline ) \
	X( vkCmdBindDescriptorSets ) \
	X( vkCmdPushConstants ) \
	X( vkCmdDispatch ) \
	X( vkCmdPipelineBarrier ) \
	X( vkCmdCopyBuffer ) \
	X( vkCreateQueryPool ) \
	X( vkDestroyQueryPool ) \
	X( vkCmdResetQueryPool ) \
	X( vkCmdWriteTimestamp ) \
	X( vkGetQueryPoolResults ) \
	X( vkQueueSubmit ) \
	X( vkCreateFence ) \
	X( vkDestroyFence ) \
	X( vkWaitForFences ) \
	X( vkResetFences ) \
	X( vkCreateShaderModule ) \
	X( vkDestroyShaderModule ) \
	X( vkCreateDescriptorSetLayout ) \
	X( vkDestroyDescriptorSetLayout ) \
	X( vkCreatePipelineLayout ) \
	X( vkDestroyPipelineLayout ) \
	X( vkCreateComputePipelines ) \
	X( vkDestroyPipeline ) \
	X( vkCreateDescriptorPool ) \
	X( vkDestroyDescriptorPool ) \
	X( vkAllocateDescriptorSets ) \
	X( vkUpdateDescriptorSets ) \
	X( vkGetAccelerationStructureBuildSizesKHR ) \
	X( vkCreateAccelerationStructureKHR ) \
	X( vkDestroyAccelerationStructureKHR ) \
	X( vkCmdBuildAccelerationStructuresKHR ) \
	X( vkGetAccelerationStructureDeviceAddressKHR )

#define GPU_DECLARE_FUNCTION( name ) PFN_##name name = nullptr;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;
GPU_INSTANCE_FUNCTIONS( GPU_DECLARE_FUNCTION )
GPU_DEVICE_FUNCTIONS( GPU_DECLARE_FUNCTION )
#undef GPU_DECLARE_FUNCTION

const char *const deviceExtensions[] = {
	VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	VK_KHR_RAY_QUERY_EXTENSION_NAME,
	VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

struct Buffer
{
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkDeviceSize size = 0;
	void *mapped = nullptr;
	bool coherent = true;
	VkDeviceAddress address = 0;
};

/* push constants of the shader */
struct Constants
{
	VkDeviceAddress rays, hits, surfaces, triangles, infos, nodes, paths, images;
	uint32_t count;
	int32_t headNode, numBspNodes;
	uint32_t skyFlag;
	uint32_t debugMode;
	uint32_t kernel;                            /* KERNEL_* in the shader */
	VkDeviceAddress params;                     /* GatherParams */
};
static_assert( sizeof( Constants ) == 96 );

enum { KERNEL_TRACE = 0, KERNEL_GATHER_RAYS = 1, KERNEL_GATHER_REDUCE = 2 };

/* Params in the shader */
struct GatherParams
{
	VkDeviceAddress samples, vectors, values, outs;
	uint32_t numSamples, numVectors, mode;
	float rayLength, inhibitRadius;
	int32_t recvShadows, numSurfaces;
	uint32_t rayFlags, translucentFlag;
	float ooDepth;
};
static_assert( sizeof( GatherParams ) == 72 );
static_assert( sizeof( GpuGatherSample ) == 64 );
static_assert( sizeof( GpuGatherOut ) == 8 );
static_assert( sizeof( GpuRay ) == 80 );
static_assert( sizeof( GpuHit ) == 48 );
static_assert( sizeof( GpuTriangle ) == 80 );
static_assert( sizeof( GpuInfo ) == 32 );
static_assert( sizeof( GpuNode ) == 32 );

/* a set of buffers and a command buffer for one batch on its way through the GPU */
struct Slot
{
	VkCommandPool pool = VK_NULL_HANDLE;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;
	Buffer rays, hits, surfaces;
	Buffer samples, vectors, params, outs;      /* gathers */
	Buffer values;                              /* gathers: a value per ray, on the GPU */
	VkQueryPool timestamps = VK_NULL_HANDLE;   /* shader start and end */
};

#ifdef _WIN32
HMODULE vulkanLibrary = nullptr;
#else
void *vulkanLibrary = nullptr;
#endif
VkInstance instance = VK_NULL_HANDLE;
VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
VkPhysicalDeviceMemoryProperties memoryProperties;
VkDevice device = VK_NULL_HANDLE;
uint32_t queueFamily = 0;
VkQueue queue = VK_NULL_HANDLE;
std::mutex queueMutex;
VkCommandPool setupPool = VK_NULL_HANDLE;
VkDeviceSize scratchAlignment = 256;

Buffer triangles, infos, nodes, paths, images, vertices, instances;
Buffer blasBuffer, tlasBuffer;
VkAccelerationStructureKHR blas = VK_NULL_HANDLE, tlas = VK_NULL_HANDLE;
VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
Constants sceneConstants;

std::mutex slotsMutex;
std::vector<std::unique_ptr<Slot>> allSlots;
std::vector<Slot*> freeSlots;

std::atomic<long long> statRays, statBatches, statMicroseconds, statShaderNanoseconds, statCopyMicroseconds;
float timestampPeriod = 0;                      /* nanoseconds per tick, 0 = no timestamps */

void Check( VkResult result, const char *what ){
	if ( result != VK_SUCCESS ) {
		Error( "-gpu: %s failed (Vulkan error %d)", what, int( result ) );
	}
}

bool LoadVulkan(){
#ifdef _WIN32
	vulkanLibrary = LoadLibraryA( "vulkan-1.dll" );
	if ( vulkanLibrary == nullptr ) {
		return false;
	}
	vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>( reinterpret_cast<void*>( GetProcAddress( vulkanLibrary, "vkGetInstanceProcAddr" ) ) );
#else
	vulkanLibrary = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
	if ( vulkanLibrary == nullptr ) {
		vulkanLibrary = dlopen( "libvulkan.so", RTLD_NOW | RTLD_LOCAL );
	}
	if ( vulkanLibrary == nullptr ) {
		return false;
	}
	vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>( dlsym( vulkanLibrary, "vkGetInstanceProcAddr" ) );
#endif
	if ( vkGetInstanceProcAddr == nullptr ) {
		return false;
	}
	vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>( vkGetInstanceProcAddr( nullptr, "vkCreateInstance" ) );
	return vkCreateInstance != nullptr;
}

bool LoadInstanceFunctions(){
	bool ok = true;
#define GPU_LOAD_FUNCTION( name ) ok = ok && ( name = reinterpret_cast<PFN_##name>( vkGetInstanceProcAddr( instance, #name ) ) ) != nullptr;
	GPU_INSTANCE_FUNCTIONS( GPU_LOAD_FUNCTION )
#undef GPU_LOAD_FUNCTION
	return ok;
}

bool LoadDeviceFunctions(){
	bool ok = true;
#define GPU_LOAD_FUNCTION( name ) ok = ok && ( name = reinterpret_cast<PFN_##name>( vkGetDeviceProcAddr( device, #name ) ) ) != nullptr;
	GPU_DEVICE_FUNCTIONS( GPU_LOAD_FUNCTION )
#undef GPU_LOAD_FUNCTION
	return ok;
}

/* features the shader needs, chained for vkGetPhysicalDeviceFeatures2 / vkCreateDevice */
struct Features
{
	VkPhysicalDeviceRayQueryFeaturesKHR rayQuery = VkStruct<VkPhysicalDeviceRayQueryFeaturesKHR>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR );
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure = VkStruct<VkPhysicalDeviceAccelerationStructureFeaturesKHR>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR );
	VkPhysicalDeviceVulkan12Features vulkan12 = VkStruct<VkPhysicalDeviceVulkan12Features>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES );
	VkPhysicalDeviceFeatures2 features2 = VkStruct<VkPhysicalDeviceFeatures2>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 );
	Features(){
		features2.pNext = &vulkan12;
		vulkan12.pNext = &accelerationStructure;
		accelerationStructure.pNext = &rayQuery;
	}
	Features( const Features& ) = delete;
};

/* a device with ray queries: the score to pick the best (discrete first), or -1 */
long long RateDevice( VkPhysicalDevice physical, uint32_t& family, std::string& why ){
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties( physical, &properties );
	if ( properties.apiVersion < VK_API_VERSION_1_2 ) {
		why = "Vulkan 1.2 not supported";
		return -1;
	}

	uint32_t count = 0;
	vkEnumerateDeviceExtensionProperties( physical, nullptr, &count, nullptr );
	std::vector<VkExtensionProperties> extensions( count );
	vkEnumerateDeviceExtensionProperties( physical, nullptr, &count, extensions.data() );
	for ( const char *name : deviceExtensions )
	{
		if ( std::none_of( extensions.begin(), extensions.end(), [name]( const VkExtensionProperties& e ){ return strcmp( e.extensionName, name ) == 0; } ) ) {
			why = std::string( "no " ) + name;
			return -1;
		}
	}

	Features features;
	vkGetPhysicalDeviceFeatures2( physical, &features.features2 );
	if ( !features.rayQuery.rayQuery || !features.accelerationStructure.accelerationStructure || !features.vulkan12.bufferDeviceAddress ) {
		why = "ray query features not supported";
		return -1;
	}
	if ( !features.features2.features.shaderFloat64 ) {
		why = "no 64-bit floats in shaders";
		return -1;
	}

	vkGetPhysicalDeviceQueueFamilyProperties( physical, &count, nullptr );
	std::vector<VkQueueFamilyProperties> families( count );
	vkGetPhysicalDeviceQueueFamilyProperties( physical, &count, families.data() );
	family = count;
	for ( uint32_t i = 0; i < count; ++i )
	{
		if ( families[ i ].queueFlags & VK_QUEUE_COMPUTE_BIT ) {
			family = i;
			break;
		}
	}
	if ( family == count ) {
		why = "no compute queue";
		return -1;
	}

	VkPhysicalDeviceMemoryProperties memory;
	vkGetPhysicalDeviceMemoryProperties( physical, &memory );
	VkDeviceSize localMemory = 0;
	for ( uint32_t i = 0; i < memory.memoryHeapCount; ++i )
	{
		if ( memory.memoryHeaps[ i ].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ) {
			localMemory = std::max( localMemory, memory.memoryHeaps[ i ].size );
		}
	}
	const long long type = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2
	                     : properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
	return ( type << 48 ) + ( localMemory >> 20 );
}

uint32_t FindMemoryType( uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred ){
	for ( const VkMemoryPropertyFlags flags : { required | preferred, required } )
	{
		for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i )
		{
			if ( ( typeBits & ( 1u << i ) ) && ( memoryProperties.memoryTypes[ i ].propertyFlags & flags ) == flags ) {
				return i;
			}
		}
	}
	Error( "-gpu: no suitable memory type" );
}

Buffer CreateBuffer( VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0 ){
	Buffer b;
	b.size = std::max<VkDeviceSize>( size, 16 );

	VkBufferCreateInfo info = VkStruct<VkBufferCreateInfo>( VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO );
	info.size = b.size;
	info.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	Check( vkCreateBuffer( device, &info, nullptr, &b.buffer ), "vkCreateBuffer" );

	VkMemoryRequirements requirements;
	vkGetBufferMemoryRequirements( device, b.buffer, &requirements );
	VkMemoryAllocateFlagsInfo flags = VkStruct<VkMemoryAllocateFlagsInfo>( VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO );
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo allocate = VkStruct<VkMemoryAllocateInfo>( VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO );
	allocate.pNext = &flags;
	allocate.allocationSize = requirements.size;
	allocate.memoryTypeIndex = FindMemoryType( requirements.memoryTypeBits, required, preferred );
	Check( vkAllocateMemory( device, &allocate, nullptr, &b.memory ), "vkAllocateMemory" );
	Check( vkBindBufferMemory( device, b.buffer, b.memory, 0 ), "vkBindBufferMemory" );

	const VkMemoryPropertyFlags memoryFlags = memoryProperties.memoryTypes[ allocate.memoryTypeIndex ].propertyFlags;
	if ( memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ) {
		Check( vkMapMemory( device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped ), "vkMapMemory" );
		b.coherent = memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	}

	VkBufferDeviceAddressInfo address = VkStruct<VkBufferDeviceAddressInfo>( VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO );
	address.buffer = b.buffer;
	b.address = vkGetBufferDeviceAddress( device, &address );
	return b;
}

void DestroyBuffer( Buffer& b ){
	if ( b.buffer != VK_NULL_HANDLE ) {
		vkDestroyBuffer( device, b.buffer, nullptr );
	}
	if ( b.memory != VK_NULL_HANDLE ) {
		vkFreeMemory( device, b.memory, nullptr ); /* unmaps */
	}
	b = Buffer();
}

/* host buffers for batches: written or read by the CPU */
const VkMemoryPropertyFlags hostMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

void EnsureDeviceBuffer( Buffer& b, VkDeviceSize size ){
	if ( b.size < size ) {
		const VkDeviceSize grown = std::max( size, b.size * 2 );
		DestroyBuffer( b );
		b = CreateBuffer( grown, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	}
}

void EnsureHostBuffer( Buffer& b, VkDeviceSize size, VkMemoryPropertyFlags preferred ){
	if ( b.size < size ) {
		const VkDeviceSize grown = std::max( size, b.size * 2 );
		DestroyBuffer( b );
		b = CreateBuffer( grown, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, preferred );
	}
}

/* records and runs commands, waiting for them to finish (setup only) */
template<typename Record>
void RunCommands( Record&& record ){
	VkCommandBufferAllocateInfo allocate = VkStruct<VkCommandBufferAllocateInfo>( VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO );
	allocate.commandPool = setupPool;
	allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate.commandBufferCount = 1;
	VkCommandBuffer cmd;
	Check( vkAllocateCommandBuffers( device, &allocate, &cmd ), "vkAllocateCommandBuffers" );
	VkCommandBufferBeginInfo begin = VkStruct<VkCommandBufferBeginInfo>( VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO );
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	Check( vkBeginCommandBuffer( cmd, &begin ), "vkBeginCommandBuffer" );
	record( cmd );
	Check( vkEndCommandBuffer( cmd ), "vkEndCommandBuffer" );

	VkFenceCreateInfo fenceInfo = VkStruct<VkFenceCreateInfo>( VK_STRUCTURE_TYPE_FENCE_CREATE_INFO );
	VkFence fence;
	Check( vkCreateFence( device, &fenceInfo, nullptr, &fence ), "vkCreateFence" );
	VkSubmitInfo submit = VkStruct<VkSubmitInfo>( VK_STRUCTURE_TYPE_SUBMIT_INFO );
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;
	{
		std::lock_guard lock( queueMutex );
		Check( vkQueueSubmit( queue, 1, &submit, fence ), "vkQueueSubmit" );
	}
	Check( vkWaitForFences( device, 1, &fence, VK_TRUE, UINT64_MAX ), "vkWaitForFences" );
	vkDestroyFence( device, fence, nullptr );
	Check( vkResetCommandPool( device, setupPool, 0 ), "vkResetCommandPool" );
}

/* a device local buffer with the data */
template<typename T>
Buffer Upload( const std::vector<T>& data, VkBufferUsageFlags usage ){
	const VkDeviceSize size = data.size() * sizeof( T );
	Buffer staging = CreateBuffer( size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hostMemory );
	if ( size != 0 ) {
		memcpy( staging.mapped, data.data(), size );
	}
	Buffer b = CreateBuffer( size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	RunCommands( [&]( VkCommandBuffer cmd ){
		VkBufferCopy copy{ 0, 0, staging.size };
		vkCmdCopyBuffer( cmd, staging.buffer, b.buffer, 1, &copy );
	} );
	DestroyBuffer( staging );
	return b;
}

/* builds an acceleration structure from one geometry */
VkAccelerationStructureKHR BuildAccelerationStructure( VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR& geometry, uint32_t primitives, Buffer& storage ){
	VkAccelerationStructureBuildGeometryInfoKHR build = VkStruct<VkAccelerationStructureBuildGeometryInfoKHR>( VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR );
	build.type = type;
	build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.geometryCount = 1;
	build.pGeometries = &geometry;

	VkAccelerationStructureBuildSizesInfoKHR sizes = VkStruct<VkAccelerationStructureBuildSizesInfoKHR>( VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR );
	vkGetAccelerationStructureBuildSizesKHR( device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &primitives, &sizes );

	storage = CreateBuffer( sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	VkAccelerationStructureCreateInfoKHR create = VkStruct<VkAccelerationStructureCreateInfoKHR>( VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR );
	create.buffer = storage.buffer;
	create.size = sizes.accelerationStructureSize;
	create.type = type;
	VkAccelerationStructureKHR structure;
	Check( vkCreateAccelerationStructureKHR( device, &create, nullptr, &structure ), "vkCreateAccelerationStructureKHR" );

	Buffer scratch = CreateBuffer( sizes.buildScratchSize + scratchAlignment, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	build.dstAccelerationStructure = structure;
	build.scratchData.deviceAddress = ( scratch.address + scratchAlignment - 1 ) / scratchAlignment * scratchAlignment;

	VkAccelerationStructureBuildRangeInfoKHR range{};
	range.primitiveCount = primitives;
	const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
	RunCommands( [&]( VkCommandBuffer cmd ){
		vkCmdBuildAccelerationStructuresKHR( cmd, 1, &build, &ranges );
	} );
	DestroyBuffer( scratch );
	return structure;
}

bool CreateDevice(){
	VkApplicationInfo application = VkStruct<VkApplicationInfo>( VK_STRUCTURE_TYPE_APPLICATION_INFO );
	application.pApplicationName = "q3map2";
	application.apiVersion = VK_API_VERSION_1_2;
	VkInstanceCreateInfo instanceInfo = VkStruct<VkInstanceCreateInfo>( VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO );
	instanceInfo.pApplicationInfo = &application;
	if ( vkCreateInstance( &instanceInfo, nullptr, &instance ) != VK_SUCCESS || !LoadInstanceFunctions() ) {
		Sys_Warning( "-gpu: could not start Vulkan\n" );
		return false;
	}

	uint32_t count = 0;
	vkEnumeratePhysicalDevices( instance, &count, nullptr );
	std::vector<VkPhysicalDevice> physicals( count );
	vkEnumeratePhysicalDevices( instance, &count, physicals.data() );
	long long bestScore = -1;
	for ( VkPhysicalDevice physical : physicals )
	{
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties( physical, &properties );
		uint32_t family;
		std::string why;
		const long long score = RateDevice( physical, family, why );
		if ( score < 0 ) {
			Sys_FPrintf( SYS_VRB, "-gpu: %s: %s\n", properties.deviceName, why.c_str() );
		}
		else if ( score > bestScore ) {
			bestScore = score;
			physicalDevice = physical;
			queueFamily = family;
		}
	}
	if ( physicalDevice == VK_NULL_HANDLE ) {
		Sys_Warning( "-gpu: no GPU with Vulkan ray queries found\n" );
		return false;
	}

	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties( physicalDevice, &properties );
	vkGetPhysicalDeviceMemoryProperties( physicalDevice, &memoryProperties );
	VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties = VkStruct<VkPhysicalDeviceAccelerationStructurePropertiesKHR>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR );
	VkPhysicalDeviceProperties2 properties2 = VkStruct<VkPhysicalDeviceProperties2>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 );
	properties2.pNext = &asProperties;
	vkGetPhysicalDeviceProperties2( physicalDevice, &properties2 );
	scratchAlignment = std::max<VkDeviceSize>( asProperties.minAccelerationStructureScratchOffsetAlignment, 1 );
	if ( properties.limits.timestampComputeAndGraphics ) {
		timestampPeriod = properties.limits.timestampPeriod;
	}

	const float priority = 1;
	VkDeviceQueueCreateInfo queueInfo = VkStruct<VkDeviceQueueCreateInfo>( VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO );
	queueInfo.queueFamilyIndex = queueFamily;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &priority;

	Features features; /* all off but these */
	features.rayQuery.rayQuery = VK_TRUE;
	features.accelerationStructure.accelerationStructure = VK_TRUE;
	features.vulkan12.bufferDeviceAddress = VK_TRUE;
	features.features2.features.shaderFloat64 = VK_TRUE;

	VkDeviceCreateInfo deviceInfo = VkStruct<VkDeviceCreateInfo>( VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO );
	deviceInfo.pNext = &features.features2;
	deviceInfo.queueCreateInfoCount = 1;
	deviceInfo.pQueueCreateInfos = &queueInfo;
	deviceInfo.enabledExtensionCount = uint32_t( std::size( deviceExtensions ) );
	deviceInfo.ppEnabledExtensionNames = deviceExtensions;
	if ( vkCreateDevice( physicalDevice, &deviceInfo, nullptr, &device ) != VK_SUCCESS || !LoadDeviceFunctions() ) {
		Sys_Warning( "-gpu: could not open %s\n", properties.deviceName );
		return false;
	}
	vkGetDeviceQueue( device, queueFamily, 0, &queue );

	VkCommandPoolCreateInfo poolInfo = VkStruct<VkCommandPoolCreateInfo>( VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO );
	poolInfo.queueFamilyIndex = queueFamily;
	Check( vkCreateCommandPool( device, &poolInfo, nullptr, &setupPool ), "vkCreateCommandPool" );

	Sys_Printf( "-gpu: tracing on %s\n", properties.deviceName );
	return true;
}

void CreatePipeline(){
	VkDescriptorSetLayoutBinding binding{};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	VkDescriptorSetLayoutCreateInfo setInfo = VkStruct<VkDescriptorSetLayoutCreateInfo>( VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO );
	setInfo.bindingCount = 1;
	setInfo.pBindings = &binding;
	Check( vkCreateDescriptorSetLayout( device, &setInfo, nullptr, &setLayout ), "vkCreateDescriptorSetLayout" );

	VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( Constants ) };
	VkPipelineLayoutCreateInfo layoutInfo = VkStruct<VkPipelineLayoutCreateInfo>( VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO );
	layoutInfo.setLayoutCount = 1;
	layoutInfo.pSetLayouts = &setLayout;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &pushRange;
	Check( vkCreatePipelineLayout( device, &layoutInfo, nullptr, &pipelineLayout ), "vkCreatePipelineLayout" );

	VkShaderModuleCreateInfo moduleInfo = VkStruct<VkShaderModuleCreateInfo>( VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO );
	moduleInfo.codeSize = sizeof( light_gpu_trace_spv );
	moduleInfo.pCode = light_gpu_trace_spv;
	VkShaderModule module;
	Check( vkCreateShaderModule( device, &moduleInfo, nullptr, &module ), "vkCreateShaderModule" );

	VkComputePipelineCreateInfo pipelineInfo = VkStruct<VkComputePipelineCreateInfo>( VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO );
	pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	pipelineInfo.stage.module = module;
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = pipelineLayout;
	Check( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline ), "vkCreateComputePipelines" );
	vkDestroyShaderModule( device, module, nullptr );

	VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 };
	VkDescriptorPoolCreateInfo poolInfo = VkStruct<VkDescriptorPoolCreateInfo>( VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO );
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	Check( vkCreateDescriptorPool( device, &poolInfo, nullptr, &descriptorPool ), "vkCreateDescriptorPool" );
	VkDescriptorSetAllocateInfo allocate = VkStruct<VkDescriptorSetAllocateInfo>( VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO );
	allocate.descriptorPool = descriptorPool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts = &setLayout;
	Check( vkAllocateDescriptorSets( device, &allocate, &descriptorSet ), "vkAllocateDescriptorSets" );

	VkWriteDescriptorSetAccelerationStructureKHR asWrite = VkStruct<VkWriteDescriptorSetAccelerationStructureKHR>( VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR );
	asWrite.accelerationStructureCount = 1;
	asWrite.pAccelerationStructures = &tlas;
	VkWriteDescriptorSet write = VkStruct<VkWriteDescriptorSet>( VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET );
	write.pNext = &asWrite;
	write.dstSet = descriptorSet;
	write.dstBinding = 0;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	vkUpdateDescriptorSets( device, 1, &write, 0, nullptr );
}

Slot *AcquireSlot(){
	{
		std::lock_guard lock( slotsMutex );
		if ( !freeSlots.empty() ) {
			Slot *slot = freeSlots.back();
			freeSlots.pop_back();
			return slot;
		}
	}
	auto slot = std::make_unique<Slot>();
	VkCommandPoolCreateInfo poolInfo = VkStruct<VkCommandPoolCreateInfo>( VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO );
	poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	poolInfo.queueFamilyIndex = queueFamily;
	Check( vkCreateCommandPool( device, &poolInfo, nullptr, &slot->pool ), "vkCreateCommandPool" );
	VkCommandBufferAllocateInfo allocate = VkStruct<VkCommandBufferAllocateInfo>( VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO );
	allocate.commandPool = slot->pool;
	allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate.commandBufferCount = 1;
	Check( vkAllocateCommandBuffers( device, &allocate, &slot->cmd ), "vkAllocateCommandBuffers" );
	VkFenceCreateInfo fenceInfo = VkStruct<VkFenceCreateInfo>( VK_STRUCTURE_TYPE_FENCE_CREATE_INFO );
	Check( vkCreateFence( device, &fenceInfo, nullptr, &slot->fence ), "vkCreateFence" );
	if ( timestampPeriod != 0 ) {
		VkQueryPoolCreateInfo queryInfo = VkStruct<VkQueryPoolCreateInfo>( VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO );
		queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		queryInfo.queryCount = 2;
		Check( vkCreateQueryPool( device, &queryInfo, nullptr, &slot->timestamps ), "vkCreateQueryPool" );
	}

	std::lock_guard lock( slotsMutex );
	allSlots.push_back( std::move( slot ) );
	return allSlots.back().get();
}

void ReleaseSlot( Slot *slot ){
	std::lock_guard lock( slotsMutex );
	freeSlots.push_back( slot );
}
}

bool GpuSetup( const GpuScene& scene ){
	const auto start = std::chrono::steady_clock::now();
	if ( scene.triangles.empty() ) {
		Sys_Warning( "-gpu: no triangles to trace, using the CPU\n" );
		return false;
	}
	if ( !LoadVulkan() ) {
		Sys_Warning( "-gpu: Vulkan is not installed, using the CPU\n" );
		return false;
	}
	if ( !CreateDevice() ) {
		Sys_Warning( "-gpu: using the CPU\n" );
		return false;
	}

	const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	triangles = Upload( scene.triangles, storage );
	infos = Upload( scene.infos, storage );
	nodes = Upload( scene.nodes, storage );
	paths = Upload( scene.paths, storage );
	images = Upload( scene.images, storage );

	/* bottom level: the triangles */
	vertices = Upload( scene.vertices, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR );
	const uint32_t numTriangles = uint32_t( scene.triangles.size() );
	VkAccelerationStructureGeometryKHR geometry = VkStruct<VkAccelerationStructureGeometryKHR>( VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR );
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR; /* not opaque: the shader tests every candidate */
	geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	geometry.geometry.triangles.vertexData.deviceAddress = vertices.address;
	geometry.geometry.triangles.vertexStride = 3 * sizeof( float );
	geometry.geometry.triangles.maxVertex = numTriangles * 3 - 1;
	geometry.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
	blas = BuildAccelerationStructure( VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometry, numTriangles, blasBuffer );

	/* top level: one instance of it */
	VkAccelerationStructureDeviceAddressInfoKHR blasAddress = VkStruct<VkAccelerationStructureDeviceAddressInfoKHR>( VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR );
	blasAddress.accelerationStructure = blas;
	std::vector<VkAccelerationStructureInstanceKHR> instance( 1 );
	instance[ 0 ] = {};
	instance[ 0 ].transform.matrix[ 0 ][ 0 ] = instance[ 0 ].transform.matrix[ 1 ][ 1 ] = instance[ 0 ].transform.matrix[ 2 ][ 2 ] = 1;
	instance[ 0 ].mask = 0xff;
	instance[ 0 ].flags = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
	instance[ 0 ].accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR( device, &blasAddress );
	instances = Upload( instance, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR );
	VkAccelerationStructureGeometryKHR top = VkStruct<VkAccelerationStructureGeometryKHR>( VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR );
	top.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	top.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	top.geometry.instances.data.deviceAddress = instances.address;
	tlas = BuildAccelerationStructure( VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top, 1, tlasBuffer );

	CreatePipeline();

	sceneConstants = {};
	sceneConstants.triangles = triangles.address;
	sceneConstants.infos = infos.address;
	sceneConstants.nodes = nodes.address;
	sceneConstants.paths = paths.address;
	sceneConstants.images = images.address;
	sceneConstants.headNode = scene.headNode;
	sceneConstants.numBspNodes = scene.numBspNodes;
	sceneConstants.skyFlag = scene.skyFlag;
	/* debugging aid: env Q3MAP2_DEBUG_GPU_MODE skips parts of the shader's work (see light_gpu_trace.comp) */
	if ( const char *mode = getenv( "Q3MAP2_DEBUG_GPU_MODE" ) ) {
		sceneConstants.debugMode = uint32_t( atoi( mode ) );
	}

	const VkDeviceSize bytes = triangles.size + infos.size + nodes.size + paths.size + images.size + vertices.size + blasBuffer.size + tlasBuffer.size;
	Sys_Printf( "%9u GPU triangles, %.1f MB on the GPU, set up in %.2f s\n", numTriangles, bytes / ( 1024.0 * 1024.0 ),
	            std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count() );
	return true;
}

namespace
{
const size_t maxPerDispatch = size_t( 65535 ) * 64;
const VkMemoryPropertyFlags cachedMemory = VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

/* records a batch's commands, runs them and waits: the shader's writes are visible to the host after */
template<typename Record>
void RunBatch( Slot *slot, Record&& record ){
	Check( vkResetCommandPool( device, slot->pool, 0 ), "vkResetCommandPool" );
	VkCommandBufferBeginInfo begin = VkStruct<VkCommandBufferBeginInfo>( VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO );
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	Check( vkBeginCommandBuffer( slot->cmd, &begin ), "vkBeginCommandBuffer" );
	if ( slot->timestamps != VK_NULL_HANDLE ) {
		vkCmdResetQueryPool( slot->cmd, slot->timestamps, 0, 2 );
		vkCmdWriteTimestamp( slot->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, slot->timestamps, 0 );
	}
	vkCmdBindPipeline( slot->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline );
	vkCmdBindDescriptorSets( slot->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr );
	record( slot->cmd );
	if ( slot->timestamps != VK_NULL_HANDLE ) {
		vkCmdWriteTimestamp( slot->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot->timestamps, 1 );
	}
	VkMemoryBarrier barrier = VkStruct<VkMemoryBarrier>( VK_STRUCTURE_TYPE_MEMORY_BARRIER );
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier( slot->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
	Check( vkEndCommandBuffer( slot->cmd ), "vkEndCommandBuffer" );

	VkSubmitInfo submit = VkStruct<VkSubmitInfo>( VK_STRUCTURE_TYPE_SUBMIT_INFO );
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &slot->cmd;
	{
		std::lock_guard lock( queueMutex );
		Check( vkQueueSubmit( queue, 1, &submit, slot->fence ), "vkQueueSubmit" );
	}
	Check( vkWaitForFences( device, 1, &slot->fence, VK_TRUE, UINT64_MAX ), "vkWaitForFences" );
	Check( vkResetFences( device, 1, &slot->fence ), "vkResetFences" );
	if ( slot->timestamps != VK_NULL_HANDLE ) {
		uint64_t ticks[ 2 ];
		if ( vkGetQueryPoolResults( device, slot->timestamps, 0, 2, sizeof( ticks ), ticks, sizeof( uint64_t ), VK_QUERY_RESULT_64_BIT ) == VK_SUCCESS ) {
			statShaderNanoseconds += (long long)( ( ticks[ 1 ] - ticks[ 0 ] ) * double( timestampPeriod ) );
		}
	}
}

void Dispatch( VkCommandBuffer cmd, const Constants& constants, size_t threads ){
	vkCmdPushConstants( cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( constants ), &constants );
	vkCmdDispatch( cmd, uint32_t( ( threads + 63 ) / 64 ), 1, 1 );
}

/* a dispatch's writes are seen by the next one */
void ComputeBarrier( VkCommandBuffer cmd ){
	VkMemoryBarrier barrier = VkStruct<VkMemoryBarrier>( VK_STRUCTURE_TYPE_MEMORY_BARRIER );
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier( cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
}

/* the host reads what the GPU wrote */
void ReadBuffer( const Buffer& b, void *destination, size_t size ){
	if ( !b.coherent ) {
		VkMappedMemoryRange range = VkStruct<VkMappedMemoryRange>( VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE );
		range.memory = b.memory;
		range.size = VK_WHOLE_SIZE;
		Check( vkInvalidateMappedMemoryRanges( device, 1, &range ), "vkInvalidateMappedMemoryRanges" );
	}
	const auto start = std::chrono::steady_clock::now();
	memcpy( destination, b.mapped, size );
	statCopyMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start ).count();
}

void WriteBuffer( Buffer& b, const void *source, size_t size, VkMemoryPropertyFlags preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ){
	EnsureHostBuffer( b, std::max<size_t>( size, 4 ), preferred );
	const auto start = std::chrono::steady_clock::now();
	if ( size != 0 ) {
		memcpy( b.mapped, source, size );
	}
	statCopyMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start ).count();
}
}

void GpuTrace( const GpuRay *rays, GpuHit *hits, size_t count, const int *surfaces, size_t numSurfaces ){
	if ( count == 0 ) {
		return;
	}
	const auto start = std::chrono::steady_clock::now();
	Slot *slot = AcquireSlot();
	WriteBuffer( slot->surfaces, surfaces, numSurfaces * sizeof( int ) );

	for ( size_t first = 0; first < count; first += maxPerDispatch )
	{
		const size_t n = std::min( count - first, maxPerDispatch );
		WriteBuffer( slot->rays, rays + first, n * sizeof( GpuRay ) );
		EnsureHostBuffer( slot->hits, n * sizeof( GpuHit ), cachedMemory );

		Constants constants = sceneConstants;
		constants.rays = slot->rays.address;
		constants.hits = slot->hits.address;
		constants.surfaces = slot->surfaces.address;
		constants.count = uint32_t( n );
		constants.kernel = KERNEL_TRACE;
		RunBatch( slot, [&]( VkCommandBuffer cmd ){
			Dispatch( cmd, constants, n );
		} );
		ReadBuffer( slot->hits, hits + first, n * sizeof( GpuHit ) );
	}

	ReleaseSlot( slot );
	statRays += count;
	++statBatches;
	statMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start ).count();
}

void GpuGather( const GpuGatherParams& params, const GpuGatherSample *samples, size_t count, GpuGatherOut *outs ){
	if ( count == 0 || params.numVectors <= 0 ) {
		return;
	}
	const auto start = std::chrono::steady_clock::now();
	Slot *slot = AcquireSlot();
	WriteBuffer( slot->surfaces, params.surfaces, params.numSurfaces * sizeof( int ) );
	std::vector<float> vectors( params.numVectors * 4, 0.0f );
	for ( int i = 0; i < params.numVectors; ++i )
		std::copy_n( params.vectors + i * 3, 3, vectors.begin() + i * 4 );
	WriteBuffer( slot->vectors, vectors.data(), vectors.size() * sizeof( float ) );

	const size_t maxSamples = std::max<size_t>( 1, maxPerDispatch / params.numVectors );
	for ( size_t first = 0; first < count; first += maxSamples )
	{
		const size_t n = std::min( count - first, maxSamples );
		WriteBuffer( slot->samples, samples + first, n * sizeof( GpuGatherSample ) );
		EnsureDeviceBuffer( slot->values, n * params.numVectors * 16 );
		EnsureHostBuffer( slot->outs, n * sizeof( GpuGatherOut ), cachedMemory );

		GatherParams gather;
		gather.samples = slot->samples.address;
		gather.vectors = slot->vectors.address;
		gather.values = slot->values.address;
		gather.outs = slot->outs.address;
		gather.numSamples = uint32_t( n );
		gather.numVectors = uint32_t( params.numVectors );
		gather.mode = uint32_t( params.mode );
		gather.rayLength = params.rayLength;
		gather.inhibitRadius = params.inhibitRadius;
		gather.recvShadows = params.recvShadows;
		gather.numSurfaces = params.numSurfaces;
		gather.rayFlags = params.rayFlags;
		gather.translucentFlag = params.translucentFlag;
		gather.ooDepth = params.ooDepth;
		WriteBuffer( slot->params, &gather, sizeof( gather ) );

		Constants constants = sceneConstants;
		constants.surfaces = slot->surfaces.address;
		constants.params = slot->params.address;
		RunBatch( slot, [&]( VkCommandBuffer cmd ){
			constants.kernel = KERNEL_GATHER_RAYS;
			Dispatch( cmd, constants, n * params.numVectors );
			ComputeBarrier( cmd );
			constants.kernel = KERNEL_GATHER_REDUCE;
			Dispatch( cmd, constants, n );
		} );
		ReadBuffer( slot->outs, outs + first, n * sizeof( GpuGatherOut ) );
	}

	ReleaseSlot( slot );
	statRays += count * params.numVectors;
	++statBatches;
	statMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - start ).count();
}

void GpuShutdown(){
	if ( device == VK_NULL_HANDLE ) {
		return;
	}
	Sys_Printf( "%9lld GPU rays in %lld batches: %.2f s in the shader, %.1f s in batches and %.1f s copying rays and hits (summed over threads)\n",
	            statRays.load(), statBatches.load(), statShaderNanoseconds.load() * 1e-9, statMicroseconds.load() * 1e-6, statCopyMicroseconds.load() * 1e-6 );
	vkDeviceWaitIdle( device );
	for ( auto& slot : allSlots )
	{
		DestroyBuffer( slot->rays );
		DestroyBuffer( slot->hits );
		DestroyBuffer( slot->surfaces );
		for ( Buffer *b : { &slot->samples, &slot->vectors, &slot->params, &slot->outs, &slot->values } )
			DestroyBuffer( *b );
		vkDestroyFence( device, slot->fence, nullptr );
		if ( slot->timestamps != VK_NULL_HANDLE ) {
			vkDestroyQueryPool( device, slot->timestamps, nullptr );
		}
		vkDestroyCommandPool( device, slot->pool, nullptr );
	}
	allSlots.clear();
	freeSlots.clear();
	vkDestroyDescriptorPool( device, descriptorPool, nullptr );
	vkDestroyPipeline( device, pipeline, nullptr );
	vkDestroyPipelineLayout( device, pipelineLayout, nullptr );
	vkDestroyDescriptorSetLayout( device, setLayout, nullptr );
	vkDestroyAccelerationStructureKHR( device, tlas, nullptr );
	vkDestroyAccelerationStructureKHR( device, blas, nullptr );
	for ( Buffer *b : { &triangles, &infos, &nodes, &paths, &images, &vertices, &instances, &blasBuffer, &tlasBuffer } )
		DestroyBuffer( *b );
	vkDestroyCommandPool( device, setupPool, nullptr );
	vkDestroyDevice( device, nullptr );
	device = VK_NULL_HANDLE;
	vkDestroyInstance( instance, nullptr );
	instance = VK_NULL_HANDLE;
}

#else

/* built without Vulkan: -gpu is never enabled */
bool GpuSetup( const GpuScene& ){
	return false;
}
void GpuTrace( const GpuRay *, GpuHit *, size_t, const int *, size_t ){
}
void GpuGather( const GpuGatherParams&, const GpuGatherSample *, size_t, GpuGatherOut * ){
}
void GpuShutdown(){
}

#endif
