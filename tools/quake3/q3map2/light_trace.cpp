/* -------------------------------------------------------------------------------

   Copyright (C) 1999-2007 id Software, Inc. and contributors.
   For a list of contributors, see the accompanying CONTRIBUTORS file.

   This file is part of GtkRadiant.

   GtkRadiant is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   GtkRadiant is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GtkRadiant; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

   ----------------------------------------------------------------------------------

   This code has been altered significantly from its original form, to support
   several games based on the Quake III Arena engine, in the form of "Q3Map2."

   ------------------------------------------------------------------------------- */



/* dependencies */
#include "q3map2.h"
#include "light_gpu.h"
#include <map>
#include <chrono>
#include <atomic>
#if defined( __x86_64__ ) || defined( __i386__ )
#include <x86intrin.h>
#endif


#define MAX_NODE_ITEMS          5
#define MAX_NODE_TRIANGLES      5
#define MAX_TRACE_DEPTH         32
#define MIN_NODE_SIZE           32.0f

#define GROW_TRACE_INFOS        32768       //%	4096
#define GROW_TRACE_WINDINGS     65536       //%	32768
#define GROW_TRACE_TRIANGLES    131072      //%	32768
#define GROW_TRACE_NODES        16384       //%	16384
#define GROW_NODE_ITEMS         16          //%	256

#define MAX_TW_VERTS            24 // vortex: increased from 12 to 24 for ability co compile some insane maps with large curve count

#define TRACE_ON_EPSILON        0.1f

#define TRACE_LEAF              -1
#define TRACE_LEAF_SOLID        -2

struct traceVert_t
{
	Vector3 xyz;
	Vector2 st;
};

struct traceInfo_t
{
	const shaderInfo_t          *si;
	int surfaceNum, castShadows;
	bool skipGrid;
};

struct traceWinding_t
{
	Plane3f plane;
	int infoNum, numVerts;
	traceVert_t v[ MAX_TW_VERTS ];
};

struct traceTriangle_t
{
	Vector3 edge1, edge2;
	int infoNum;
	traceVert_t v[ 3 ];
};

struct traceNode_t
{
	int type;
	Plane3f plane;
	MinMax minmax;
	int children[ 2 ];
	int numItems, maxItems;
	int                         *items;
};


namespace
{
int noDrawContentFlags, noDrawSurfaceFlags, noDrawCompileFlags;

int numTraceInfos = 0, maxTraceInfos = 0, firstTraceInfo = 0;
traceInfo_t                     *traceInfos = nullptr;

int numTraceWindings = 0, maxTraceWindings = 0, deadWinding = -1;
traceWinding_t                  *traceWindings = nullptr;

int numTraceTriangles = 0, maxTraceTriangles = 0, deadTriangle = -1;
traceTriangle_t                 *traceTriangles = nullptr;

int headNodeNum = 0, skyboxNodeNum = 0, maxTraceDepth = 0, numTraceLeafNodes = 0;
int numTraceNodes = 0, maxTraceNodes = 0;
int numBspTraceNodes = 0;                       /* trace nodes made from the bsp tree come first */
traceNode_t                     *traceNodes = nullptr;
}



/* -------------------------------------------------------------------------------

   allocation and list management

   ------------------------------------------------------------------------------- */

/*
   AddTraceInfo() - ydnar
   adds a trace info structure to the pool
 */

static int AddTraceInfo( const traceInfo_t& ti ){
	int num;

	/* find an existing info */
	for ( num = firstTraceInfo; num < numTraceInfos; ++num )
	{
		if ( traceInfos[ num ].si          == ti.si &&
		     traceInfos[ num ].surfaceNum  == ti.surfaceNum &&
		     traceInfos[ num ].castShadows == ti.castShadows &&
		     traceInfos[ num ].skipGrid    == ti.skipGrid ) {
			return num;
		}
	}

	/* enough space? */
	AUTOEXPAND_BY_REALLOC_ADD( traceInfos, numTraceInfos, maxTraceInfos, GROW_TRACE_INFOS );

	/* add the info */
	traceInfos[ num ] = ti;
	if ( num == numTraceInfos ) {
		numTraceInfos++;
	}

	/* return the ti number */
	return num;
}



/*
   AllocTraceNode() - ydnar
   allocates a new trace node
 */

static int AllocTraceNode(){
	/* enough space? */
	AUTOEXPAND_BY_REALLOC_ADD( traceNodes, numTraceNodes, maxTraceNodes, GROW_TRACE_NODES );

	/* add the node */
	memset( &traceNodes[ numTraceNodes ], 0, sizeof( traceNode_t ) );
	traceNodes[ numTraceNodes ].type = TRACE_LEAF;
	traceNodes[ numTraceNodes ].minmax.clear();

	/* Sys_Printf( "alloc node %d\n", numTraceNodes ); */

	numTraceNodes++;

	/* return the count */
	return ( numTraceNodes - 1 );
}



/*
   AddTraceWinding() - ydnar
   adds a winding to the raytracing pool
 */

static int AddTraceWinding( const traceWinding_t& tw ){
	int num;

	/* check for a dead winding */
	if ( deadWinding >= 0 && deadWinding < numTraceWindings ) {
		num = deadWinding;
	}
	else
	{
		/* put winding at the end of the list */
		num = numTraceWindings;

		/* enough space? */
		AUTOEXPAND_BY_REALLOC_ADD( traceWindings, numTraceWindings, maxTraceWindings, GROW_TRACE_WINDINGS );
	}

	/* add the winding */
	traceWindings[ num ] = tw;
	if ( num == numTraceWindings ) {
		numTraceWindings++;
	}
	deadWinding = -1;

	/* return the winding number */
	return num;
}



/*
   AddTraceTriangle() - ydnar
   adds a triangle to the raytracing pool
 */

static int AddTraceTriangle( const traceTriangle_t& tt ){
	int num;

	/* check for a dead triangle */
	if ( deadTriangle >= 0 && deadTriangle < numTraceTriangles ) {
		num = deadTriangle;
	}
	else
	{
		/* put triangle at the end of the list */
		num = numTraceTriangles;

		/* enough space? */
		AUTOEXPAND_BY_REALLOC_ADD( traceTriangles, numTraceTriangles, maxTraceTriangles, GROW_TRACE_TRIANGLES );
	}

	/* add the triangle */
	traceTriangles[ num ] = tt;
	if ( num == numTraceTriangles ) {
		numTraceTriangles++;
	}
	deadTriangle = -1;

	/* return the triangle number */
	return num;
}



/*
   AddItemToTraceNode() - ydnar
   adds an item reference (winding or triangle) to a trace node
 */

static int AddItemToTraceNode( traceNode_t& node, int num ){
	/* dummy check */
	if ( num < 0 ) {
		return -1;
	}

	/* enough space? */
	if ( node.numItems >= node.maxItems ) {
		/* allocate more room */
		if ( &node == traceNodes ) {
			node.maxItems *= 2;
		}
		else{
			node.maxItems += GROW_NODE_ITEMS;
		}
		if ( node.maxItems <= 0 ) {
			node.maxItems = GROW_NODE_ITEMS;
		}
		node.items = void_ptr( realloc( node.items, node.maxItems * sizeof( *node.items ) ) );
		if ( !node.items ) {
			Error( "node.items out of memory" );
		}
	}

	/* add the poly */
	node.items[ node.numItems ] = num;
	node.numItems++;

	/* return the count */
	return ( node.numItems - 1 );
}




/* -------------------------------------------------------------------------------

   trace node setup

   ------------------------------------------------------------------------------- */

/*
   SetupTraceNodes_r() - ydnar
   recursively create the initial trace node structure from the bsp tree
 */

static int SetupTraceNodes_r( int bspNodeNum ){
	/* get bsp node and plane */
	const bspNode_t& bspNode = bspNodes[ bspNodeNum ];
	const bspPlane_t& plane = bspPlanes[ bspNode.planeNum ];

	/* allocate a new trace node */
	const int nodeNum = AllocTraceNode();

	/* setup trace node */
	traceNodes[ nodeNum ].type = PlaneTypeForNormal( plane.normal() );
	traceNodes[ nodeNum ].plane = plane;

	/* setup children */
	for ( int i = 0; i < 2; ++i )
	{
		/* leafnode */
		if ( bspNode.children[ i ] < 0 ) {
			const int bspLeafNum = -bspNode.children[ i ] - 1;

			/* new code */
			const int newNode = AllocTraceNode();
			traceNodes[ nodeNum ].children[ i ] = newNode;
			/* have to do this separately, as gcc first executes LHS, then RHS, and if a realloc took place, this fails */

			if ( bspLeafs[ bspLeafNum ].cluster == CLUSTER_OPAQUE ) {
				traceNodes[ traceNodes[ nodeNum ].children[ i ] ].type = TRACE_LEAF_SOLID;
			}
		}

		/* normal node */
		else
		{
			const int newNode = SetupTraceNodes_r( bspNode.children[ i ] );
			traceNodes[ nodeNum ].children[ i ] = newNode;
		}

		if ( traceNodes[ nodeNum ].children[ i ] == 0 ) {
			Error( "Invalid tracenode allocated" );
		}
	}

	/* Sys_Printf( "node %d children: %d %d\n", nodeNum, traceNodes[ nodeNum ].children[0], traceNodes[ nodeNum ].children[1] ); */

	/* return node number */
	return nodeNum;
}



/*
   ClipTraceWinding() - ydnar
   clips a trace winding against a plane into one or two parts
 */

#define TW_ON_EPSILON   0.25f

void ClipTraceWinding( const traceWinding_t& tw, const Plane3f& plane, traceWinding_t& front, traceWinding_t& back ){
	EPlaneSide sides[ MAX_TW_VERTS ];
	int counts[ 3 ] = { 0, 0, 0 };
	float dists[ MAX_TW_VERTS ];
	float frac;
	traceVert_t mid;


	/* clear front and back */
	front.numVerts = 0;
	back.numVerts = 0;

	/* classify points */
	for ( int i = 0; i < tw.numVerts; ++i )
	{
		dists[ i ] = plane3_distance_to_point( plane, tw.v[ i ].xyz );
		if ( dists[ i ] < -TW_ON_EPSILON ) {
			sides[ i ] = eSideBack;
		}
		else if ( dists[ i ] > TW_ON_EPSILON ) {
			sides[ i ] = eSideFront;
		}
		else{
			sides[ i ] = eSideOn;
		}
		counts[ sides[ i ] ]++;
	}

	/* entirely on front? */
	if ( counts[ eSideBack ] == 0 ) {
		front = tw;
	}

	/* entirely on back? */
	else if ( counts[ eSideFront ] == 0 ) {
		back = tw;
	}

	/* straddles the plane */
	else
	{
		/* setup front and back */
		front = tw;
		front.numVerts = 0;
		back = tw;
		back.numVerts = 0;

		/* split the winding */
		for ( int i = 0; i < tw.numVerts; ++i )
		{
			/* radix */
			const int j = ( i + 1 ) % tw.numVerts;

			/* get verts */
			const traceVert_t& a = tw.v[ i ];
			const traceVert_t& b = tw.v[ j ];

			/* handle points on the splitting plane */
			switch ( sides[ i ] )
			{
			case eSideFront:
				if ( front.numVerts >= MAX_TW_VERTS ) {
					Error( "MAX_TW_VERTS (%d) exceeded", MAX_TW_VERTS );
				}
				front.v[ front.numVerts++ ] = a;
				break;

			case eSideBack:
				if ( back.numVerts >= MAX_TW_VERTS ) {
					Error( "MAX_TW_VERTS (%d) exceeded", MAX_TW_VERTS );
				}
				back.v[ back.numVerts++ ] = a;
				break;

			case eSideOn:
				if ( front.numVerts >= MAX_TW_VERTS || back.numVerts >= MAX_TW_VERTS ) {
					Error( "MAX_TW_VERTS (%d) exceeded", MAX_TW_VERTS );
				}
				front.v[ front.numVerts++ ] = a;
				back.v[ back.numVerts++ ] = a;
				continue;
			case eSideCross: // unused here, suppress warning
				break;
			}

			/* check next point to see if we need to split the edge */
			if ( sides[ j ] == eSideOn || sides[ j ] == sides[ i ] ) {
				continue;
			}

			/* check limit */
			if ( front.numVerts >= MAX_TW_VERTS || back.numVerts >= MAX_TW_VERTS ) {
				Error( "MAX_TW_VERTS (%d) exceeded", MAX_TW_VERTS );
			}

			/* generate a split point */
			frac = dists[ i ] / ( dists[ i ] - dists[ j ] );
			for ( int k = 0; k < 3; ++k )
			{
				/* minimize fp precision errors */
				if ( plane.normal()[ k ] == 1 ) {
					mid.xyz[ k ] = plane.dist();
				}
				else if ( plane.normal()[ k ] == -1 ) {
					mid.xyz[ k ] = -plane.dist();
				}
				else{
					mid.xyz[ k ] = a.xyz[ k ] + frac * ( b.xyz[ k ] - a.xyz[ k ] );
				}
			}
			/* set texture coordinates */
			mid.st = a.st + ( b.st - a.st ) * frac;

			/* copy midpoint to front and back polygons */
			front.v[ front.numVerts++ ] = mid;
			back.v[ back.numVerts++ ] = mid;
		}
	}
}



/*
   FilterTraceWindingIntoNodes_r() - ydnar
   filters a trace winding into the raytracing tree
 */

static void FilterTraceWindingIntoNodes_r( traceWinding_t& tw, int nodeNum ){
	/* don't filter if passed a bogus node (solid, etc) */
	if ( nodeNum < 0 || nodeNum >= numTraceNodes ) {
		return;
	}

	/* get node */
	traceNode_t& node = traceNodes[ nodeNum ];

	/* is this a decision node? */
	if ( node.type >= 0 ) {
		/* create winding plane if necessary, filtering out bogus windings as well */
		if ( nodeNum == headNodeNum ) {
			if ( !PlaneFromPoints( tw.plane, tw.v[ 0 ].xyz, tw.v[ 1 ].xyz, tw.v[ 2 ].xyz ) ) {
				return;
			}
		}

		/* validate the node */
		if ( node.children[ 0 ] == 0 || node.children[ 1 ] == 0 ) {
			Error( "Invalid tracenode: %d", nodeNum );
		}

		/* get node plane */
		const Plane3f plane1 = node.plane;

		/* get winding plane */
		const Plane3f plane2 = tw.plane;

		/* invert surface plane */
		const Plane3f reverse = plane3_flipped( plane2 );

		/* front only */
		if ( vector3_dot( plane1.normal(), plane2.normal() ) > 0.999f && std::fabs( plane1.dist() - plane2.dist() ) < 0.001f ) {
			FilterTraceWindingIntoNodes_r( tw, node.children[ 0 ] );
			return;
		}

		/* back only */
		if ( vector3_dot( plane1.normal(), reverse.normal() ) > 0.999f && std::fabs( plane1.dist() - reverse.dist() ) < 0.001f ) {
			FilterTraceWindingIntoNodes_r( tw, node.children[ 1 ] );
			return;
		}

		/* clip the winding by node plane */
		traceWinding_t front, back;
		ClipTraceWinding( tw, plane1, front, back );

		/* filter by node plane */
		if ( front.numVerts >= 3 ) {
			FilterTraceWindingIntoNodes_r( front, node.children[ 0 ] );
		}
		if ( back.numVerts >= 3 ) {
			FilterTraceWindingIntoNodes_r( back, node.children[ 1 ] );
		}

		/* return to caller */
		return;
	}

	/* add winding to leaf node */
	const int num = AddTraceWinding( tw );
	AddItemToTraceNode( node, num );
}



/*
   SubdivideTraceNode_r() - ydnar
   recursively subdivides a tracing node until it meets certain size and complexity criteria
 */

static void SubdivideTraceNode_r( int nodeNum, int depth ){
	int count, frontNum, backNum, type;
	float dist;
	traceNode_t     *node;


	/* dummy check */
	if ( nodeNum < 0 || nodeNum >= numTraceNodes ) {
		return;
	}

	/* get node */
	node = &traceNodes[ nodeNum ];

	/* runaway recursion check */
	if ( depth >= MAX_TRACE_DEPTH ) {
		//%	Sys_Printf( "Depth: (%d items)\n", node->numItems );
		numTraceLeafNodes++;
		return;
	}
	depth++;

	/* is this a decision node? */
	if ( node->type >= 0 ) {
		/* subdivide children */
		frontNum = node->children[ 0 ];
		backNum = node->children[ 1 ];
		SubdivideTraceNode_r( frontNum, depth );
		SubdivideTraceNode_r( backNum, depth );
		return;
	}

	/* bound the node */
	node->minmax.clear();
	DoubleVector3 average( 0 );
	count = 0;
	for ( int i = 0; i < node->numItems; ++i )
	{
		/* get winding */
		const traceWinding_t& tw = traceWindings[ node->items[ i ] ];

		/* walk its verts */
		for ( int j = 0; j < tw.numVerts; ++j )
		{
			node->minmax.extend( tw.v[ j ].xyz );
			average += tw.v[ j ].xyz;
			count++;
		}
	}

	/* check triangle limit */
	//%	if( node->numItems <= MAX_NODE_ITEMS )
	if ( ( count - ( node->numItems * 2 ) ) < MAX_NODE_TRIANGLES ) {
		//%	Sys_Printf( "Limit: (%d triangles)\n", ( count - ( node->numItems * 2 ) ) );
		numTraceLeafNodes++;
		return;
	}

	/* the largest dimension of the bounding box will be the split axis */
	const Vector3 size = node->minmax.maxs - node->minmax.mins;
	if ( size[ 0 ] >= size[ 1 ] && size[ 0 ] >= size[ 2 ] ) {
		type = ePlaneX;
	}
	else if ( size[ 1 ] >= size[ 0 ] && size[ 1 ] >= size[ 2 ] ) {
		type = ePlaneY;
	}
	else{
		type = ePlaneZ;
	}

	/* don't split small nodes */
	if ( size[ type ] <= MIN_NODE_SIZE ) {
		//%	Sys_Printf( "Limit: %f %f %f (%d items)\n", size[ 0 ], size[ 1 ], size[ 2 ], node->numItems );
		numTraceLeafNodes++;
		return;
	}

	/* set max trace depth */
	value_maximize( maxTraceDepth, depth );

	/* snap the average */
	dist = floor( average[ type ] / count );

	/* dummy check it */
	if ( dist <= node->minmax.mins[ type ] || dist >= node->minmax.maxs[ type ] ) {
		dist = floor( 0.5f * ( node->minmax.mins[ type ] + node->minmax.maxs[ type ] ) );
	}

	/* allocate child nodes */
	frontNum = AllocTraceNode();
	backNum = AllocTraceNode();

	/* reset pointers */
	node = &traceNodes[ nodeNum ];
	traceNode_t& frontNode = traceNodes[ frontNum ];
	traceNode_t& backNode = traceNodes[ backNum ];

	/* attach children */
	node->type = type;
	node->plane.normal()[ type ] = 1;
	node->plane.dist() = dist;
	node->children[ 0 ] = frontNum;
	node->children[ 1 ] = backNum;

	/* setup front node */
	frontNode.maxItems = ( node->maxItems >> 1 );
	frontNode.items = safe_malloc( frontNode.maxItems * sizeof( *frontNode.items ) );

	/* setup back node */
	backNode.maxItems = ( node->maxItems >> 1 );
	backNode.items = safe_malloc( backNode.maxItems * sizeof( *backNode.items ) );

	/* filter windings into child nodes */
	for ( int i = 0; i < node->numItems; ++i )
	{
		/* get winding */
		const traceWinding_t& tw = traceWindings[ node->items[ i ] ];

		/* clip the winding by the new split plane */
		traceWinding_t front, back;
		ClipTraceWinding( tw, node->plane, front, back );

		/* kill the existing winding */
		if ( front.numVerts >= 3 || back.numVerts >= 3 ) {
			deadWinding = node->items[ i ];
		}

		/* add front winding */
		if ( front.numVerts >= 3 ) {
			const int num = AddTraceWinding( front );
			AddItemToTraceNode( frontNode, num );
		}

		/* add back winding */
		if ( back.numVerts >= 3 ) {
			const int num = AddTraceWinding( back );
			AddItemToTraceNode( backNode, num );
		}
	}

	/* free original node winding list */
	node->numItems = 0;
	node->maxItems = 0;
	free( std::exchange( node->items, nullptr ) );

	/* check children */
	if ( frontNode.numItems <= 0 ) {
		frontNode.maxItems = 0;
		free( std::exchange( frontNode.items, nullptr ) );
	}

	if ( backNode.numItems <= 0 ) {
		backNode.maxItems = 0;
		free( std::exchange( backNode.items, nullptr ) );
	}

	/* subdivide children */
	SubdivideTraceNode_r( frontNum, depth );
	SubdivideTraceNode_r( backNum, depth );
}



/*
   TriangulateTraceNode_r()
   optimizes the tracing data by changing trace windings into triangles
 */

static int TriangulateTraceNode_r( int nodeNum ){
	/* dummy check */
	if ( nodeNum < 0 || nodeNum >= numTraceNodes ) {
		return 0;
	}

	/* get node */
	traceNode_t& node = traceNodes[ nodeNum ];

	/* is this a decision node? */
	if ( node.type >= 0 ) {
		/* triangulate children */
		const int frontNum = node.children[ 0 ];
		const int backNum = node.children[ 1 ];
		node.numItems = TriangulateTraceNode_r( frontNum );
		node.numItems += TriangulateTraceNode_r( backNum );
		return node.numItems;
	}

	/* empty node? */
	if ( node.numItems == 0 ) {
		node.maxItems = 0;
		free( node.items );
		return node.numItems;
	}

	/* store off winding data */
	const int numWindings = node.numItems;
	int *windings = node.items;

	/* clear it */
	node.numItems = 0;
	node.maxItems = numWindings * 2;
	node.items = safe_malloc( node.maxItems * sizeof( *node.items ) );

	/* walk winding list */
	for ( int i = 0; i < numWindings; ++i )
	{
		/* get winding */
		traceWinding_t& tw = traceWindings[ windings[ i ] ];

		/* initial setup */
		traceTriangle_t tt;
		tt.infoNum = tw.infoNum;
		tt.v[ 0 ] = tw.v[ 0 ];

		/* walk vertex list */
		for ( int j = 1; j + 1 < tw.numVerts; ++j )
		{
			/* set verts */
			tt.v[ 1 ] = tw.v[ j ];
			tt.v[ 2 ] = tw.v[ j + 1 ];

			/* find vectors for two edges sharing the first vert */
			tt.edge1 = tt.v[ 1 ].xyz - tt.v[ 0 ].xyz;
			tt.edge2 = tt.v[ 2 ].xyz - tt.v[ 0 ].xyz;

			/* add it to the node */
			const int num = AddTraceTriangle( tt );
			AddItemToTraceNode( node, num );
		}
	}

	/* free windings */
	free( windings );

	/* return item count */
	return node.numItems;
}



/* -------------------------------------------------------------------------------

   shadow casting item setup (triangles, patches, entities)

   ------------------------------------------------------------------------------- */

/*
   PopulateWithBSPModel() - ydnar
   filters a bsp model's surfaces into the raytracing tree
 */

static void PopulateWithBSPModel( const bspModel_t& model, const Matrix4& transform ){
	int nodeNum;
	traceInfo_t ti;
	traceWinding_t tw;


	/* walk the list of surfaces in this model and fill out the info structs */
	for ( int i = 0; i < model.numBSPSurfaces; ++i )
	{
		/* get surface and info */
		const bspDrawSurface_t& ds = bspDrawSurfaces[ model.firstBSPSurface + i ];
		const surfaceInfo_t& info = surfaceInfos[ model.firstBSPSurface + i ];
		if ( info.si == nullptr ) {
			continue;
		}

		/* no shadows */
		if ( !info.castShadows ) {
			continue;
		}

		/* patchshadows? */
		if ( ds.surfaceType == MST_PATCH && !patchShadows ) {
			continue;
		}

		/* some surfaces in the bsp might have been tagged as nodraw, with a bogus shader */
		if ( ( bspShaders[ ds.shaderNum ].contentFlags & noDrawContentFlags ) ||
		     ( bspShaders[ ds.shaderNum ].surfaceFlags & noDrawSurfaceFlags ) ) {
			continue;
		}

		/* translucent surfaces that are neither alphashadow or lightfilter don't cast shadows */
		if ( ( info.si->compileFlags & C_NODRAW ) ) {
			continue;
		}
		if ( ( info.si->compileFlags & C_TRANSLUCENT ) &&
		    !( info.si->compileFlags & C_ALPHASHADOW ) &&
		    !( info.si->compileFlags & C_LIGHTFILTER ) ) {
			continue;
		}

		/* setup trace info */
		ti.si = info.si;
		ti.castShadows = info.castShadows;
		ti.surfaceNum = model.firstBSPBrush + i;
		ti.skipGrid = ( ds.surfaceType == MST_PATCH );

		/* choose which node (normal or skybox) */
		if ( info.parentSurfaceNum >= 0 ) {
			nodeNum = skyboxNodeNum;

			/* sky surfaces in portal skies are ignored */
			if ( info.si->compileFlags & C_SKY ) {
				continue;
			}
		}
		else{
			nodeNum = headNodeNum;
		}

		/* setup trace winding */
		memset( &tw, 0, sizeof( tw ) );
		tw.infoNum = AddTraceInfo( ti );
		tw.numVerts = 3;

		/* switch on type */
		switch ( ds.surfaceType )
		{
		/* handle patches */
		case MST_PATCH:
		{
			/* subdivide the surface */
			const mesh_t mesh = TessellatedMesh( mesh_view_t( ds.patchWidth, ds.patchHeight, &bspDrawVerts[ ds.firstVert ] ), info.patchIterations );

			/* subdivide each quad to place the models */
			for( MeshQuadIterator it( mesh ); it; ++it ){
				for( const TriRef& tri : it.tris() ){
					for( int i = 0; i < 3; ++i )
					{
						tw.v[ i ].xyz = matrix4_transformed_point( transform, tri[ i ]->xyz );
						tw.v[ i ].st  = tri[ i ]->st;
					}
					FilterTraceWindingIntoNodes_r( tw, nodeNum );
				}
			}
			break;
		}
		/* handle triangle surfaces */
		case MST_TRIANGLE_SOUP:
		case MST_PLANAR:
		{
			/* set verts and indexes */
			const bspDrawVert_t *verts = &bspDrawVerts[ ds.firstVert ];
			const int *indexes = &bspDrawIndexes[ ds.firstIndex ];

			/* walk the triangle list */
			for ( int j = 0; j < ds.numIndexes; j += 3 )
			{
				for( int i = 0; i < 3; ++i )
				{
					const bspDrawVert_t& v = verts[ indexes[ j + i ] ];
					tw.v[ i ].xyz = matrix4_transformed_point( transform, v.xyz );
					tw.v[ i ].st  = v.st;
				}
				FilterTraceWindingIntoNodes_r( tw, nodeNum );
			}
			break;
		}
		/* other surface types do not cast shadows */
		default:
			break;
		}
	}
}

#include "model.h"

/*
   PopulateWithPicoModel() - ydnar
   filters a picomodel's surfaces into the raytracing tree
 */

static void PopulateWithPicoModel( int castShadows, const std::vector<const AssMeshWalker*>& model, const Matrix4& transform ){
	traceInfo_t ti;
	traceWinding_t tw;


	/* dummy check */
	if ( model.empty() ) {
		return;
	}

	/* walk the list of surfaces in this model and fill out the info structs */
	for ( const auto *mesh : model )
	{
		/* get shader (fixme: support shader remapping) */
		ti.si = ShaderInfoForShaderNull( mesh->getShaderName() );
		if ( ti.si == nullptr ) {
			continue;
		}

		/* translucent surfaces that are neither alphashadow or lightfilter don't cast shadows */
		if ( ( ti.si->compileFlags & C_NODRAW ) ) {
			continue;
		}
		if ( ( ti.si->compileFlags & C_TRANSLUCENT ) &&
		    !( ti.si->compileFlags & C_ALPHASHADOW ) &&
		    !( ti.si->compileFlags & C_LIGHTFILTER ) ) {
			continue;
		}

		/* setup trace info */
		ti.castShadows = castShadows;
		ti.surfaceNum = -1;
		ti.skipGrid = true; // also ignore picomodels when skipping patches

		/* setup trace winding */
		memset( &tw, 0, sizeof( tw ) );
		tw.infoNum = AddTraceInfo( ti );
		tw.numVerts = 3;

		/* walk the triangle list */
		mesh->forEachFace( [&tw, &transform]( const Vector3 ( &xyz )[3], const Vector2 ( &st )[3] ){
			for( int i = 0; i < 3; ++i ){
				tw.v[ i ].xyz = matrix4_transformed_point( transform, xyz[ i ] );
				tw.v[ i ].st = st[ i ];
			}
			FilterTraceWindingIntoNodes_r( tw, headNodeNum );
		} );
	}
}



/*
   PopulateTraceNodes() - ydnar
   fills the raytracing tree with world and entity occluders
 */

static void PopulateTraceNodes(){
	size_t m;
	const char      *value;


	/* add worldspawn triangles */
	PopulateWithBSPModel( bspModels[ 0 ], g_matrix4_identity );

	/* walk each entity list */
	for ( std::size_t i = 1; i < entities.size(); ++i )
	{
		/* get entity */
		const entity_t& e = entities[ i ];

		/* get shadow flags */
		int castShadows = ENTITY_CAST_SHADOWS;
		GetEntityShadowFlags( &e, nullptr, &castShadows, nullptr );

		/* early out? */
		if ( !castShadows ) {
			continue;
		}

		/* get model */
		value = e.valueForKey( "model" );

		/* switch on model type */
		switch ( value[ 0 ] )
		{
		/* no model */
		case '\0':
			break;

		/* bsp model */
		case '*':
			m = atoi( &value[ 1 ] );
			if ( m <= 0 || m >= bspModels.size() ) {
				continue;
			}
			// using only origin for bmodels transform; only exception in vq3 is func_pendulum transform determined by angle/angles
			PopulateWithBSPModel( bspModels[ m ], matrix4_translation_for_vec3( e.vectorForKey( "origin" ) ) );
			break;

		/* external model */
		default:
			PopulateWithPicoModel( castShadows, LoadModelWalker( value, e.intForKey( "_frame", "frame" ) ), ModelGetTransform( e ) );
			continue;
		}

		/* get model2 */
		value = e.valueForKey( "model2" );

		/* switch on model type */
		switch ( value[ 0 ] )
		{
		/* no model */
		case '\0':
			break;

		/* bsp model */
		case '*':
			m = atoi( &value[ 1 ] );
			if ( m <= 0 || m >= bspModels.size() ) {
				continue;
			}
			PopulateWithBSPModel( bspModels[ m ], matrix4_translation_for_vec3( e.vectorForKey( "origin" ) ) );
			break;

		/* external model */
		default:
			PopulateWithPicoModel( castShadows, LoadModelWalker( value, e.intForKey( "_frame2" ) ), ModelGetTransform( e ) );
			continue;
		}
	}
}




/* -------------------------------------------------------------------------------

   trace initialization

   ------------------------------------------------------------------------------- */

/*
   SetupTraceNodes() - ydnar
   creates a balanced bsp with axis-aligned splits for efficient raytracing
 */

#ifdef Q3MAP2_EMBREE
static void SetupEmbree();
#endif
#ifdef Q3MAP2_VULKAN
namespace
{
void SetupGpu();
}
#endif

void SetupTraceNodes(){
	/* note it */
	Sys_FPrintf( SYS_VRB, "--- SetupTraceNodes ---\n" );

	/* find nodraw bit */
	noDrawContentFlags = noDrawSurfaceFlags = noDrawCompileFlags = 0;
	ApplySurfaceParm( "nodraw", &noDrawContentFlags, &noDrawSurfaceFlags, &noDrawCompileFlags );

	/* create the baseline raytracing tree from the bsp tree */
	headNodeNum = SetupTraceNodes_r( 0 );
	numBspTraceNodes = numTraceNodes;

	/* create outside node for skybox surfaces */
	skyboxNodeNum = AllocTraceNode();

	/* populate the tree with triangles from the world and shadow casting entities */
	PopulateTraceNodes();

	/* create the raytracing bsp */
	if ( !loMem ) {
		SubdivideTraceNode_r( headNodeNum, 0 );
		SubdivideTraceNode_r( skyboxNodeNum, 0 );
	}

	/* create triangles from the trace windings */
	TriangulateTraceNode_r( headNodeNum );
	TriangulateTraceNode_r( skyboxNodeNum );

	/* emit some stats */
	//%	Sys_FPrintf( SYS_VRB, "%9d original triangles\n", numOriginalTriangles );
	Sys_FPrintf( SYS_VRB, "%9d trace windings (%.2fMB)\n", numTraceWindings, (float) ( numTraceWindings * sizeof( *traceWindings ) ) / ( 1024.0f * 1024.0f ) );
	Sys_FPrintf( SYS_VRB, "%9d trace triangles (%.2fMB)\n", numTraceTriangles, (float) ( numTraceTriangles * sizeof( *traceTriangles ) ) / ( 1024.0f * 1024.0f ) );
	Sys_FPrintf( SYS_VRB, "%9d trace nodes (%.2fMB)\n", numTraceNodes, (float) ( numTraceNodes * sizeof( *traceNodes ) ) / ( 1024.0f * 1024.0f ) );
	Sys_FPrintf( SYS_VRB, "%9d leaf nodes (%.2fMB)\n", numTraceLeafNodes, (float) ( numTraceLeafNodes * sizeof( *traceNodes ) ) / ( 1024.0f * 1024.0f ) );
	//%	Sys_FPrintf( SYS_VRB, "%9d average triangles per leaf node\n", numTraceTriangles / numTraceLeafNodes );
	Sys_FPrintf( SYS_VRB, "%9d average windings per leaf node\n", numTraceWindings / ( numTraceLeafNodes + 1 ) );
	Sys_FPrintf( SYS_VRB, "%9d max trace depth\n", maxTraceDepth );

	/* free trace windings */
	free( traceWindings );
	numTraceWindings = 0;
	maxTraceWindings = 0;
	deadWinding = -1;

#ifdef Q3MAP2_EMBREE
	if ( lightEmbree ) {
		SetupEmbree();
	}
#endif
#ifdef Q3MAP2_VULKAN
	if ( lightGpu ) {
		SetupGpu();
	}
#endif

	/* debug code: write out trace triangles to an alias obj file */
	#if 0
	{
		int i, j;
		FILE            *file;
		char filename[ 1024 ];
		traceWinding_t  *tw;


		/* open the file */
		strcpy( filename, source );
		path_set_extension( filename, ".lin" );
		Sys_Printf( "Opening light trace file %s...\n", filename );
		file = SafeOpenWrite( filename, "wt" );

		/* walk node list */
		for ( i = 0; i < numTraceWindings; ++i )
		{
			tw = &traceWindings[ i ];
			for ( j = 0; j < tw->numVerts + 1; ++j )
				fprintf( file, "%f %f %f\n",
				         tw->v[ j % tw->numVerts ].xyz[ 0 ], tw->v[ j % tw->numVerts ].xyz[ 1 ], tw->v[ j % tw->numVerts ].xyz[ 2 ] );
		}

		/* close it */
		fclose( file );
	}
	#endif
}



/* -------------------------------------------------------------------------------

   raytracer

   ------------------------------------------------------------------------------- */

/*
   TraceTriangle()
   based on code written by william 'spog' joseph
   based on code originally written by tomas moller and ben trumbore, journal of graphics tools, 2(1):21-28, 1997
 */

#define BARY_EPSILON            0.01f
#define ASLF_EPSILON            0.0001f /* so to not get double shadows */
#define COPLANAR_EPSILON        0.25f   //%	0.000001f
#define NEAR_SHADOW_EPSILON     1.5f    //%	1.25f
#define SELF_SHADOW_EPSILON     0.5f

static bool TraceTriangle( const traceInfo_t& ti, const traceTriangle_t& tt, trace_t *trace ){
	Vector3 tvec, pvec, qvec;
	float det, invDet, depth;
	float u, v, w, s, t;
	int is, it;
	const byte            *pixel;
	const shaderInfo_t    *si;


	/* don't double-trace against sky */
	si = ti.si;
	if ( trace->compileFlags & si->compileFlags & C_SKY ) {
		return false;
	}

	/* worldspawn group only receives shadows from positive groups */
	if ( trace->recvShadows == 1 ) {
		if ( ti.castShadows <= 0 ) {
			return false;
		}
	}

	/* receive shadows from same group and worldspawn group */
	else if ( trace->recvShadows > 1 ) {
		if ( ti.castShadows != 1 && abs( ti.castShadows ) != abs( trace->recvShadows ) ) {
			return false;
		}
		//%	Sys_Printf( "%d:%d ", tt->castShadows, trace->recvShadows );
	}

	/* receive shadows from the same group only (< 0) */
	else
	{
		if ( abs( ti.castShadows ) != abs( trace->recvShadows ) ) {
			return false;
		}
	}

	/* skip patches when doing the grid (FIXME this is an ugly hack) */
	if ( inGrid ) {
		if ( ti.skipGrid ) {
			return false;
		}
	}

	/* begin calculating determinant - also used to calculate u parameter */
	pvec = vector3_cross( trace->direction, tt.edge2 );

	/* if determinant is near zero, trace lies in plane of triangle */
	det = vector3_dot( tt.edge1, pvec );

	/* the non-culling branch */
	if ( std::fabs( det ) < COPLANAR_EPSILON ) {
		return false;
	}
	invDet = 1.0f / det;

	/* calculate distance from first vertex to ray origin */
	tvec = trace->origin - tt.v[ 0 ].xyz;

	/* calculate u parameter and test bounds */
	u = vector3_dot( tvec, pvec ) * invDet;
	if ( u < -BARY_EPSILON || u > ( 1.0f + BARY_EPSILON ) ) {
		return false;
	}

	/* prepare to test v parameter */
	qvec = vector3_cross( tvec, tt.edge1 );

	/* calculate v parameter and test bounds */
	v = vector3_dot( trace->direction, qvec ) * invDet;
	if ( v < -BARY_EPSILON || ( u + v ) > ( 1.0f + BARY_EPSILON ) ) {
		return false;
	}

	/* calculate t (depth) */
	depth = vector3_dot( tt.edge2, qvec ) * invDet;
	if ( depth <= trace->inhibitRadius || depth >= trace->distance ) {
		return false;
	}

	/* if hitpoint is really close to trace origin (sample point), then check for self-shadowing */
	if ( depth <= SELF_SHADOW_EPSILON ) {
		/* don't self-shadow */
		for ( int i : Span( trace->surfaces, trace->numSurfaces ) )
		{
			if ( ti.surfaceNum == i ) {
				return false;
			}
		}
	}

	/* stack compile flags */
	trace->compileFlags |= si->compileFlags;

	/* don't trace against sky */
	if ( si->compileFlags & C_SKY ) {
		if( !g_oneSky && trace->light != nullptr /* 0 for -dirty */ && trace->light->skyIndex != -1 && si->skyIndex != -1 )
			bit_enable( trace->skyIndices, si->skyIndex );
		return false;
	}

	/* most surfaces are completely opaque */
	if ( !( si->compileFlags & ( C_ALPHASHADOW | C_LIGHTFILTER ) ) ||
	     si->lightImage == nullptr || si->lightImage->pixels == nullptr ) {
		trace->hit = trace->origin + trace->direction * depth;
		trace->color.set( 0 );
		trace->opaque = true;
		return true;
	}

	/* force subsampling because the lighting is texture dependent */
	trace->forceSubsampling = 1;

	/* try to avoid double shadows near triangle seams */
	if ( u < -ASLF_EPSILON || u > ( 1.0f + ASLF_EPSILON ) ||
	     v < -ASLF_EPSILON || ( u + v ) > ( 1.0f + ASLF_EPSILON ) ) {
		return false;
	}

	/* calculate w parameter */
	w = 1.0f - ( u + v );

	/* calculate st from uvw (barycentric) coordinates */
	s = w * tt.v[ 0 ].st[ 0 ] + u * tt.v[ 1 ].st[ 0 ] + v * tt.v[ 2 ].st[ 0 ];
	t = w * tt.v[ 0 ].st[ 1 ] + u * tt.v[ 1 ].st[ 1 ] + v * tt.v[ 2 ].st[ 1 ];
	s = s - floor( s );
	t = t - floor( t );
	is = std::clamp( int( s * si->lightImage->width ), 0, si->lightImage->width - 1 );
	it = std::clamp( int( t * si->lightImage->height ), 0, si->lightImage->height - 1 );

	/* get pixel */
	pixel = si->lightImage->pixels + 4 * ( it * si->lightImage->width + is );

	/* ydnar: color filter */
	if ( si->compileFlags & C_LIGHTFILTER ) {
		/* filter by texture color */
		trace->color *= Vector3( pixel[0], pixel[1], pixel[2] ) * ( 1.0f / 255.0f );
	}

	/* ydnar: alpha filter */
	if ( si->compileFlags & C_ALPHASHADOW ) {
		/* filter by inverse texture alpha */
		const float shadow = ( 1.0f / 255.0f ) * ( 255 - pixel[ 3 ] );
		trace->color *= shadow;
	}

	/* check filter for opaque */
	if ( trace->color[ 0 ] <= 0.001f && trace->color[ 1 ] <= 0.001f && trace->color[ 2 ] <= 0.001f ) {
		trace->color.set( 0 );
		trace->hit = trace->origin + trace->direction * depth;
		trace->opaque = true;
		return true;
	}

	/* continue tracing */
	return false;
}



/*
   TraceWinding() - ydnar
   temporary hack
 */

static bool TraceWinding( traceWinding_t *tw, trace_t *trace ){
	/* initial setup */
	traceTriangle_t tt;
	tt.infoNum = tw->infoNum;
	tt.v[ 0 ] = tw->v[ 0 ];

	/* walk vertex list */
	for ( int i = 1; i + 1 < tw->numVerts; ++i )
	{
		/* set verts */
		tt.v[ 1 ] = tw->v[ i ];
		tt.v[ 2 ] = tw->v[ i + 1 ];

		/* find vectors for two edges sharing the first vert */
		tt.edge1 = tt.v[ 1 ].xyz - tt.v[ 0 ].xyz;
		tt.edge2 = tt.v[ 2 ].xyz - tt.v[ 0 ].xyz;

		/* trace it */
		if ( TraceTriangle( traceInfos[ tt.infoNum ], tt, trace ) ) {
			return true;
		}
	}

	/* done */
	return false;
}




/*
   TraceLine_r()
   returns true if something is hit and tracing can stop

   SmileTheory: made half-iterative
 */

#define TRACELINE_R_HALF_ITERATIVE 1

#if TRACELINE_R_HALF_ITERATIVE
static bool TraceLine_r( int nodeNum, const Vector3& start, const Vector3& end, trace_t *trace )
#else
static bool TraceLine_r( int nodeNum, const Vector3& origin, const Vector3& end, trace_t *trace )
#endif
{
	traceNode_t     *node;
	int side;
	float front, back, frac;
	Vector3 mid;

#if TRACELINE_R_HALF_ITERATIVE
	Vector3 origin( start );

	while ( true )
#endif
	{
		/* bogus node number means solid, end tracing unless testing all */
		if ( nodeNum < 0 ) {
			trace->hit = origin;
			trace->passSolid = true;
			return true;
		}

		/* get node */
		node = &traceNodes[ nodeNum ];

		/* solid? */
		if ( node->type == TRACE_LEAF_SOLID ) {
			trace->hit = origin;
			trace->passSolid = true;
			return true;
		}

		/* leafnode? */
		if ( node->type < 0 ) {
			/* note leaf and return */
			if ( node->numItems > 0 && trace->numTestNodes < MAX_TRACE_TEST_NODES ) {
				trace->testNodes[ trace->numTestNodes++ ] = nodeNum;
			}
			return false;
		}

		/* ydnar 2003-09-07: don't test branches of the bsp with nothing in them when testall is enabled */
		if ( trace->testAll && node->numItems == 0 ) {
			return false;
		}

		/* classify beginning and end points */
		switch ( node->type )
		{
		case ePlaneX:
			front = origin[ 0 ] - node->plane.dist();
			back = end[ 0 ] - node->plane.dist();
			break;

		case ePlaneY:
			front = origin[ 1 ] - node->plane.dist();
			back = end[ 1 ] - node->plane.dist();
			break;

		case ePlaneZ:
			front = origin[ 2 ] - node->plane.dist();
			back = end[ 2 ] - node->plane.dist();
			break;

		default:
			front = plane3_distance_to_point( node->plane, origin );
			back = plane3_distance_to_point( node->plane, end );
			break;
		}

		/* entirely in front side? */
		if ( front >= -TRACE_ON_EPSILON && back >= -TRACE_ON_EPSILON ) {
#if TRACELINE_R_HALF_ITERATIVE
			nodeNum = node->children[ 0 ];
			continue;
#else
			return TraceLine_r( node->children[ 0 ], origin, end, trace );
#endif
		}

		/* entirely on back side? */
		if ( front < TRACE_ON_EPSILON && back < TRACE_ON_EPSILON ) {
#if TRACELINE_R_HALF_ITERATIVE
			nodeNum = node->children[ 1 ];
			continue;
#else
			return TraceLine_r( node->children[ 1 ], origin, end, trace );
#endif
		}

		/* select side */
		side = front < 0;

		/* calculate intercept point */
		frac = front / ( front - back );
		mid = origin + ( end - origin ) * frac;

		/* fixme: check inhibit radius, then solid nodes and ignore */

		/* set trace hit here */
		//%	trace->hit = mid;

		/* trace first side */
		if ( TraceLine_r( node->children[ side ], origin, mid, trace ) ) {
			return true;
		}

		/* trace other side */
#if TRACELINE_R_HALF_ITERATIVE
		nodeNum = node->children[ !side ];
		origin = mid;
#else
		return TraceLine_r( node->children[ !side ], mid, end, trace );
#endif
	}
}



/*
   TraceLine() - ydnar
   rewrote this function a bit :)
 */

static void TraceLineInner( trace_t *trace );

/* thread profiling debug counter: time spent in one function (Q3MAP2_DEBUG_COUNTER = trace (default) | sample),
   measured with the cpu timestamp counter where available (cheap enough per call) */
namespace
{
unsigned long long DebugCounterNow(){
#if defined( __x86_64__ ) || defined( __i386__ )
	return __rdtsc();
#else
	return std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

thread_local unsigned long long debugCounterTicks = 0;
thread_local long long debugCounterCalls = 0;
unsigned long long debugCounterTicksStart;
std::chrono::steady_clock::time_point debugCounterClockStart;

void DebugCounterFlush(){
	const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - debugCounterClockStart ).count();
	const double ticksPerSecond = ( DebugCounterNow() - debugCounterTicksStart ) / std::max( seconds, 1e-6 );
	ThreadProfileAddCounter( debugCounterTicks / ticksPerSecond, debugCounterCalls );
	debugCounterTicks = 0;
	debugCounterCalls = 0;
}

const char *DebugCounterSetup(){
	if ( !ThreadProfileEnabled() ) {
		return "";
	}
	debugCounterTicksStart = DebugCounterNow();
	debugCounterClockStart = std::chrono::steady_clock::now();
	threadProfileFlush = DebugCounterFlush;
	const char *name = getenv( "Q3MAP2_DEBUG_COUNTER" );
	return name != nullptr ? name : "trace";
}
}

bool DebugCounterEnabled( const char *name ){
	static const char *counter = DebugCounterSetup();
	return strEqual( counter, name );
}

unsigned long long DebugCounterStart(){
	return DebugCounterNow();
}

void DebugCounterStop( unsigned long long start ){
	debugCounterTicks += DebugCounterNow() - start;
	++debugCounterCalls;
}

void TraceLine( trace_t *trace ){
	static const bool counted = DebugCounterEnabled( "trace" );
	if ( counted ) {
		const unsigned long long start = DebugCounterStart();
		TraceLineInner( trace );
		DebugCounterStop( start );
	}
	else{
		TraceLineInner( trace );
	}
}

static void TraceLineTree( trace_t *trace );
#ifdef Q3MAP2_EMBREE
static void TraceLineEmbree( trace_t *trace );
static void EmbreeCheck( const trace_t& reference, const trace_t& trace );
#endif

static void TraceLineInner( trace_t *trace ){
	/* setup output (note: this code assumes the input data is completely filled out) */
	trace->passSolid = false;
	trace->opaque = false;
	trace->compileFlags = 0;
	memset( trace->skyIndices, 0, sizeof( trace->skyIndices ) );
	trace->numTestNodes = 0;

	/* early outs */
	if ( !trace->recvShadows || !trace->testOcclusion || trace->distance <= 0.00001f ) {
		return;
	}

#ifdef Q3MAP2_EMBREE
	if ( lightEmbree ) {
		/* debugging aid: env Q3MAP2_DEBUG_EMBREE_CHECK runs both tracers on every ray and reports differences */
		if ( static const bool check = getenv( "Q3MAP2_DEBUG_EMBREE_CHECK" ) != nullptr; check ) {
			trace_t reference = *trace;
			TraceLineEmbree( trace );
			TraceLineTree( &reference );
			EmbreeCheck( reference, *trace );
		}
		else{
			TraceLineEmbree( trace );
		}
		return;
	}
#endif

	TraceLineTree( trace );
}

/* TraceLine() with the trace node tree, after its setup and early outs */
static void TraceLineTree( trace_t *trace ){
	/* trace through nodes */
	TraceLine_r( headNodeNum, trace->origin, trace->end, trace );

	/* debugging aid: env Q3MAP2_DEBUG_TESTNODES counts rays whose node list is full, so later nodes' triangles are not tested */
	if ( static const bool debugTestNodes = getenv( "Q3MAP2_DEBUG_TESTNODES" ) != nullptr; debugTestNodes ) {
		static std::atomic<long long> rays, full, fullSun;
		static const struct Report { ~Report(){ fprintf( stderr, "testnodes: %lld rays, %lld with full node list (%lld testAll)\n", rays.load(), full.load(), fullSun.load() ); } } report;
		++rays;
		if ( trace->numTestNodes >= MAX_TRACE_TEST_NODES ) {
			++full;
			if ( trace->testAll ) {
				++fullSun;
			}
		}
	}

	if ( trace->passSolid && !trace->testAll ) {
		trace->opaque = true;
		return;
	}

	/* skip surfaces? */
	if ( noSurfaces ) {
		return;
	}

	/* testall means trace through sky */
	if ( trace->testAll && trace->numTestNodes < MAX_TRACE_TEST_NODES &&
	     trace->compileFlags & C_SKY &&
	     ( trace->numSurfaces == 0 || surfaceInfos[ trace->surfaces[ 0 ] ].childSurfaceNum < 0 ) ) {
		//%	trace->testNodes[ trace->numTestNodes++ ] = skyboxNodeNum;
		TraceLine_r( skyboxNodeNum, trace->origin, trace->end, trace );
	}

	/* walk node list */
	for ( int i = 0; i < trace->numTestNodes; ++i )
	{
		/* get node */
		traceNode_t& node = traceNodes[ trace->testNodes[ i ] ];

		/* walk node item list */
		for ( int j = 0; j < node.numItems; ++j )
		{
			const traceTriangle_t& tt = traceTriangles[ node.items[ j ] ];
			traceInfo_t& ti = traceInfos[ tt.infoNum ];
			if ( TraceTriangle( ti, tt, trace ) ) {
				return;
			}
			//%	if( TraceWinding( &traceWindings[ node.items[ j ] ], trace ) )
			//%		return;
		}
	}
}



#ifdef Q3MAP2_EMBREE
/* -------------------------------------------------------------------------------

   -embree: Intel Embree finds the triangles a ray hits, instead of walking the trace node tree.

   the result is meant to match TraceLine(): Embree is given the same triangles (the clipped pieces
   from the trace nodes), slightly enlarged so it reports every triangle that TraceTriangle() could
   accept with its epsilons, and a filter callback runs the exact TraceTriangle() tests on each one.
   the bsp tree is still walked for solid space, like TraceLine_r() does, but not its subdivision.
   what differs is the order hits are found in: TraceLine() stops at the first opaque triangle in
   tree order, this uses the nearest one, and applies alpha/filter shadows in distance order.

   ------------------------------------------------------------------------------- */

#include <embree4/rtcore.h>

namespace
{
RTCDevice embreeDevice = nullptr;
RTCScene embreeScene = nullptr;
std::vector<int> embreeTriangles;               /* embree primitive id -> traceTriangles index */
std::vector<int> embreeTriangleLeaves;          /* embree primitive id -> trace node (leaf) holding the triangle */
std::vector<int> embreeTriangleItems;           /* embree primitive id -> position in the leaf's items */
std::vector<int> traceNodeParents;

/* how much the triangles are enlarged around their center. TraceTriangle() accepts barycentric
   coordinates up to BARY_EPSILON outside the triangle, which equals scaling it by 1 + 3 * BARY_EPSILON */
const float embreeTriangleScale = 1.06f;

/* outcome of the TraceTriangle() tests for one triangle */
enum class TriangleTest { Ignored, Flags, Sky, Filter, Opaque };

struct TriangleResult
{
	TriangleTest test = TriangleTest::Ignored;
	float depth;
	int tri;
	bool inMargin;                              /* hit within BARY_EPSILON outside the triangle */
	float leafEntry;                            /* TraceLine() tests leaves in the order the ray enters them, */
	int item;                                   /* then the triangles in a leaf in item order */
	Vector3 filterColor;                        /* Filter: lightfilter color, or 1 */
	float filterShadow;                         /* Filter: alphashadow factor, or 1 */
	bool hasFilterColor, hasFilterShadow;
};

/* TraceTriangle() without changing the trace: classifies the triangle */
TriangleResult TestTraceTriangle( int tri, const trace_t& trace ){
	TriangleResult r;
	r.tri = tri;
	const traceTriangle_t& tt = traceTriangles[ tri ];
	const traceInfo_t& ti = traceInfos[ tt.infoNum ];
	const shaderInfo_t *si = ti.si;

	/* worldspawn group only receives shadows from positive groups */
	if ( trace.recvShadows == 1 ) {
		if ( ti.castShadows <= 0 ) {
			return r;
		}
	}
	/* receive shadows from same group and worldspawn group */
	else if ( trace.recvShadows > 1 ) {
		if ( ti.castShadows != 1 && abs( ti.castShadows ) != abs( trace.recvShadows ) ) {
			return r;
		}
	}
	/* receive shadows from the same group only (< 0) */
	else
	{
		if ( abs( ti.castShadows ) != abs( trace.recvShadows ) ) {
			return r;
		}
	}

	/* skip patches when doing the grid (FIXME this is an ugly hack) */
	if ( inGrid && ti.skipGrid ) {
		return r;
	}

	/* begin calculating determinant - also used to calculate u parameter */
	const Vector3 pvec = vector3_cross( trace.direction, tt.edge2 );

	/* if determinant is near zero, trace lies in plane of triangle */
	const float det = vector3_dot( tt.edge1, pvec );
	if ( std::fabs( det ) < COPLANAR_EPSILON ) {
		return r;
	}
	const float invDet = 1.0f / det;

	/* calculate distance from first vertex to ray origin */
	const Vector3 tvec = trace.origin - tt.v[ 0 ].xyz;

	/* calculate u parameter and test bounds */
	const float u = vector3_dot( tvec, pvec ) * invDet;
	if ( u < -BARY_EPSILON || u > ( 1.0f + BARY_EPSILON ) ) {
		return r;
	}

	/* prepare to test v parameter */
	const Vector3 qvec = vector3_cross( tvec, tt.edge1 );

	/* calculate v parameter and test bounds */
	const float v = vector3_dot( trace.direction, qvec ) * invDet;
	if ( v < -BARY_EPSILON || ( u + v ) > ( 1.0f + BARY_EPSILON ) ) {
		return r;
	}

	r.inMargin = u < 0 || v < 0 || ( u + v ) > 1;

	/* calculate t (depth) */
	r.depth = vector3_dot( tt.edge2, qvec ) * invDet;
	if ( r.depth <= trace.inhibitRadius || r.depth >= trace.distance ) {
		return r;
	}

	/* if hitpoint is really close to trace origin (sample point), then check for self-shadowing */
	if ( r.depth <= SELF_SHADOW_EPSILON ) {
		for ( int i : Span( trace.surfaces, trace.numSurfaces ) )
		{
			if ( ti.surfaceNum == i ) {
				return r;
			}
		}
	}

	/* from here on the triangle's compile flags are stacked */
	r.test = TriangleTest::Flags;

	/* don't trace against sky */
	if ( si->compileFlags & C_SKY ) {
		r.test = TriangleTest::Sky;
		return r;
	}

	/* most surfaces are completely opaque */
	if ( !( si->compileFlags & ( C_ALPHASHADOW | C_LIGHTFILTER ) ) ||
	     si->lightImage == nullptr || si->lightImage->pixels == nullptr ) {
		r.test = TriangleTest::Opaque;
		return r;
	}

	/* force subsampling because the lighting is texture dependent: stays TriangleTest::Flags + filter shader */

	/* try to avoid double shadows near triangle seams */
	if ( u < -ASLF_EPSILON || u > ( 1.0f + ASLF_EPSILON ) ||
	     v < -ASLF_EPSILON || ( u + v ) > ( 1.0f + ASLF_EPSILON ) ) {
		return r;
	}

	/* calculate w parameter */
	const float w = 1.0f - ( u + v );

	/* calculate st from uvw (barycentric) coordinates */
	float s = w * tt.v[ 0 ].st[ 0 ] + u * tt.v[ 1 ].st[ 0 ] + v * tt.v[ 2 ].st[ 0 ];
	float t = w * tt.v[ 0 ].st[ 1 ] + u * tt.v[ 1 ].st[ 1 ] + v * tt.v[ 2 ].st[ 1 ];
	s = s - floor( s );
	t = t - floor( t );
	const int is = std::clamp( int( s * si->lightImage->width ), 0, si->lightImage->width - 1 );
	const int it = std::clamp( int( t * si->lightImage->height ), 0, si->lightImage->height - 1 );

	/* get pixel */
	const byte *pixel = si->lightImage->pixels + 4 * ( it * si->lightImage->width + is );

	r.test = TriangleTest::Filter;
	r.hasFilterColor = si->compileFlags & C_LIGHTFILTER;
	r.hasFilterShadow = si->compileFlags & C_ALPHASHADOW;
	r.filterColor = Vector3( pixel[0], pixel[1], pixel[2] ) * ( 1.0f / 255.0f );
	r.filterShadow = ( 1.0f / 255.0f ) * ( 255 - pixel[ 3 ] );
	return r;
}

/* whether TraceLine_r() visits the leaf when walking from origin to end (same plane tests and splits);
   entry is set to where the ray enters it */
bool RayPassesLeaf( int leaf, Vector3 origin, Vector3 end, Vector3& entry ){
	int path[ 512 ];
	int depth = 0;
	for ( int n = leaf; n >= 0 && depth < int( std::size( path ) ); n = traceNodeParents[ n ] )
		path[ depth++ ] = n;

	for ( int i = depth - 1; i > 0; --i )
	{
		const traceNode_t& node = traceNodes[ path[ i ] ];
		const int side = node.children[ 0 ] == path[ i - 1 ] ? 0 : 1;

		float front, back;
		switch ( node.type )
		{
		case ePlaneX:
			front = origin[ 0 ] - node.plane.dist();
			back = end[ 0 ] - node.plane.dist();
			break;
		case ePlaneY:
			front = origin[ 1 ] - node.plane.dist();
			back = end[ 1 ] - node.plane.dist();
			break;
		case ePlaneZ:
			front = origin[ 2 ] - node.plane.dist();
			back = end[ 2 ] - node.plane.dist();
			break;
		default:
			front = plane3_distance_to_point( node.plane, origin );
			back = plane3_distance_to_point( node.plane, end );
			break;
		}

		if ( front >= -TRACE_ON_EPSILON && back >= -TRACE_ON_EPSILON ) {
			if ( side != 0 ) {
				return false;
			}
		}
		else if ( front < TRACE_ON_EPSILON && back < TRACE_ON_EPSILON ) {
			if ( side != 1 ) {
				return false;
			}
		}
		else
		{
			const int first = front < 0;
			const Vector3 mid = origin + ( end - origin ) * ( front / ( front - back ) );
			if ( side == first ) {
				end = mid;
			}
			else{
				origin = mid;
			}
		}
	}
	entry = origin;
	return true;
}

/* TestTraceTriangle() for an embree primitive: TraceLine() only tests a triangle when the ray passes its leaf.
   that limits the BARY_EPSILON margin of triangles clipped by the trace nodes to their own leaf, and with
   TRACE_ON_EPSILON it skips leaves that a ray running along a node plane only touches */
TriangleResult ClassifyTriangle( unsigned primID, const trace_t& trace ){
	TriangleResult r = TestTraceTriangle( embreeTriangles[ primID ], trace );
	if ( r.test != TriangleTest::Ignored ) {
		Vector3 entry;
		if ( RayPassesLeaf( embreeTriangleLeaves[ primID ], trace.origin, trace.end, entry ) ) {
			r.leafEntry = vector3_length( entry - trace.origin );
			r.item = embreeTriangleItems[ primID ];
		}
		else{
			r.test = TriangleTest::Ignored;
		}
	}
	return r;
}

/* the order TraceLine() tests triangles in */
bool TreeOrderLess( const TriangleResult& a, const TriangleResult& b ){
	return a.leafEntry < b.leafEntry || ( a.leafEntry == b.leafEntry && a.item < b.item );
}

/* the ray query context: embree passes it to the filter callback */
struct EmbreeQuery
{
	RTCRayQueryContext context;                 /* must be first */
	const trace_t *trace;
	bool treeOrder;                             /* find the opaque triangle TraceLine() finds first, not the nearest */
};

/* triangles that don't stop the ray but count for its result, collected during a query */
thread_local std::vector<TriangleResult> embreePassedTriangles;

void EmbreeFilter( const RTCFilterFunctionNArguments *args ){
	const EmbreeQuery& query = *reinterpret_cast<const EmbreeQuery*>( args->context );
	const TriangleResult r = ClassifyTriangle( RTCHitN_primID( args->hit, args->N, 0 ), *query.trace );
	if ( r.test == TriangleTest::Opaque && !query.treeOrder ) {
		return; /* accept: embree keeps the nearest */
	}
	args->valid[ 0 ] = 0;
	if ( r.test != TriangleTest::Ignored ) {
		/* embree may report a triangle more than once */
		for ( const TriangleResult& passed : embreePassedTriangles )
		{
			if ( passed.tri == r.tri ) {
				return;
			}
		}
		embreePassedTriangles.push_back( r );
	}
}

/* walks the bsp part of the trace node tree like TraceLine_r(), only to find solid space;
   nodes from SubdivideTraceNode_r() can't be solid */
bool TraceSolid_r( int nodeNum, const Vector3& origin, const Vector3& end, trace_t *trace ){
	/* bogus node number means solid, end tracing unless testing all */
	if ( nodeNum < 0 ) {
		trace->hit = origin;
		trace->passSolid = true;
		return true;
	}
	if ( nodeNum >= numBspTraceNodes ) {
		return false;
	}

	const traceNode_t& node = traceNodes[ nodeNum ];

	/* solid? */
	if ( node.type == TRACE_LEAF_SOLID ) {
		trace->hit = origin;
		trace->passSolid = true;
		return true;
	}

	/* leafnode? */
	if ( node.type < 0 ) {
		return false;
	}

	/* ydnar 2003-09-07: don't test branches of the bsp with nothing in them when testall is enabled */
	if ( trace->testAll && node.numItems == 0 ) {
		return false;
	}

	/* classify beginning and end points */
	float front, back;
	switch ( node.type )
	{
	case ePlaneX:
		front = origin[ 0 ] - node.plane.dist();
		back = end[ 0 ] - node.plane.dist();
		break;

	case ePlaneY:
		front = origin[ 1 ] - node.plane.dist();
		back = end[ 1 ] - node.plane.dist();
		break;

	case ePlaneZ:
		front = origin[ 2 ] - node.plane.dist();
		back = end[ 2 ] - node.plane.dist();
		break;

	default:
		front = plane3_distance_to_point( node.plane, origin );
		back = plane3_distance_to_point( node.plane, end );
		break;
	}

	/* entirely in front side? */
	if ( front >= -TRACE_ON_EPSILON && back >= -TRACE_ON_EPSILON ) {
		return TraceSolid_r( node.children[ 0 ], origin, end, trace );
	}

	/* entirely on back side? */
	if ( front < TRACE_ON_EPSILON && back < TRACE_ON_EPSILON ) {
		return TraceSolid_r( node.children[ 1 ], origin, end, trace );
	}

	/* select side */
	const int side = front < 0;

	/* calculate intercept point */
	const float frac = front / ( front - back );
	const Vector3 mid = origin + ( end - origin ) * frac;

	/* trace first side, then the other */
	return TraceSolid_r( node.children[ side ], origin, mid, trace )
	    || TraceSolid_r( node.children[ !side ], mid, end, trace );
}

void EmbreeError( void *, RTCError code, const char *message ){
	Error( "Embree error %d: %s", int( code ), message );
}

/* adds the triangles in the leaves of a trace node tree */
void CollectTraceTriangles_r( int nodeNum ){
	if ( nodeNum < 0 || nodeNum >= numTraceNodes ) {
		return;
	}
	const traceNode_t& node = traceNodes[ nodeNum ];
	if ( node.type >= 0 ) {
		traceNodeParents[ node.children[ 0 ] ] = nodeNum;
		traceNodeParents[ node.children[ 1 ] ] = nodeNum;
		CollectTraceTriangles_r( node.children[ 0 ] );
		CollectTraceTriangles_r( node.children[ 1 ] );
	}
	else if ( node.type == TRACE_LEAF ) {
		embreeTriangles.insert( embreeTriangles.end(), node.items, node.items + node.numItems );
		embreeTriangleLeaves.insert( embreeTriangleLeaves.end(), node.numItems, nodeNum );
		for ( int i = 0; i < node.numItems; ++i )
			embreeTriangleItems.push_back( i );
	}
}
}

/*
   SetupEmbree()
   builds the embree scene from the trace triangles, after SetupTraceNodes()
 */

static void SetupEmbree(){
	Sys_Printf( "--- SetupEmbree ---\n" );

	embreeDevice = rtcNewDevice( nullptr );
	if ( embreeDevice == nullptr ) {
		Error( "Embree: failed to create device (error %d)", int( rtcGetDeviceError( nullptr ) ) );
	}
	rtcSetDeviceErrorFunction( embreeDevice, EmbreeError, nullptr );

	embreeTriangles.clear();
	embreeTriangleLeaves.clear();
	embreeTriangleItems.clear();
	traceNodeParents.assign( numTraceNodes, -1 );
	CollectTraceTriangles_r( headNodeNum );

	RTCGeometry geometry = rtcNewGeometry( embreeDevice, RTC_GEOMETRY_TYPE_TRIANGLE );
	float *vertices = static_cast<float*>( rtcSetNewGeometryBuffer( geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof( float ), embreeTriangles.size() * 3 ) );
	unsigned *indexes = static_cast<unsigned*>( rtcSetNewGeometryBuffer( geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof( unsigned ), embreeTriangles.size() ) );
	for ( size_t i = 0; i < embreeTriangles.size(); ++i )
	{
		const traceTriangle_t& tt = traceTriangles[ embreeTriangles[ i ] ];
		const Vector3 center = ( tt.v[ 0 ].xyz + tt.v[ 1 ].xyz + tt.v[ 2 ].xyz ) * ( 1.0f / 3.0f );
		for ( int j = 0; j < 3; ++j )
		{
			const Vector3 xyz = center + ( tt.v[ j ].xyz - center ) * embreeTriangleScale;
			vertices[ i * 9 + j * 3 + 0 ] = xyz[ 0 ];
			vertices[ i * 9 + j * 3 + 1 ] = xyz[ 1 ];
			vertices[ i * 9 + j * 3 + 2 ] = xyz[ 2 ];
			indexes[ i * 3 + j ] = i * 3 + j;
		}
	}
	rtcSetGeometryEnableFilterFunctionFromArguments( geometry, true );
	rtcCommitGeometry( geometry );

	embreeScene = rtcNewScene( embreeDevice );
	rtcSetSceneFlags( embreeScene, RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS );
	rtcAttachGeometry( embreeScene, geometry );
	rtcReleaseGeometry( geometry );
	rtcCommitScene( embreeScene );

	Sys_Printf( "%9zu embree triangles\n", embreeTriangles.size() );
}

/*
   TraceLineEmbree()
   TraceLine() for -embree, called after its setup and early outs
 */

static void TraceLineEmbree( trace_t *trace ){
	/* solid space */
	float segment = trace->distance;
	if ( TraceSolid_r( headNodeNum, trace->origin, trace->end, trace ) ) {
		if ( !trace->testAll ) {
			trace->opaque = true;
			return;
		}
		/* TraceLine_r() stops collecting nodes at solid space: surfaces up to its edge (e.g. sky) count */
		segment = std::min( segment, float( vector3_length( trace->hit - trace->origin ) ) + 1.0f );
	}

	/* skip surfaces? */
	if ( noSurfaces ) {
		return;
	}

	/* find the nearest opaque triangle, collecting the sky and filter ones on the way */
	embreePassedTriangles.clear();

	EmbreeQuery query;
	rtcInitRayQueryContext( &query.context );
	query.trace = trace;
	/* -dirty and floodlight rays (no light) use the hit distance: that depends on which opaque triangle is found */
	query.treeOrder = trace->light == nullptr;

	RTCIntersectArguments args;
	rtcInitIntersectArguments( &args );
	args.context = &query.context;
	args.filter = EmbreeFilter;
	args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;

	RTCRayHit rayhit;
	rayhit.ray.org_x = trace->origin[ 0 ];
	rayhit.ray.org_y = trace->origin[ 1 ];
	rayhit.ray.org_z = trace->origin[ 2 ];
	rayhit.ray.dir_x = trace->direction[ 0 ];
	rayhit.ray.dir_y = trace->direction[ 1 ];
	rayhit.ray.dir_z = trace->direction[ 2 ];
	rayhit.ray.tnear = 0;
	rayhit.ray.tfar = segment;
	rayhit.ray.time = 0;
	rayhit.ray.mask = -1;
	rayhit.ray.id = 0;
	rayhit.ray.flags = 0;
	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[ 0 ] = RTC_INVALID_GEOMETRY_ID;
	rtcIntersect1( embreeScene, &rayhit, &args );

	TriangleResult opaque;
	if ( rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID ) {
		opaque = ClassifyTriangle( rayhit.hit.primID, *trace );
	}

	/* apply the passed triangles that TraceLine() tests before the opaque one, in its order */
	std::sort( embreePassedTriangles.begin(), embreePassedTriangles.end(), TreeOrderLess );
	if ( query.treeOrder ) {
		for ( const TriangleResult& r : embreePassedTriangles )
		{
			if ( r.test == TriangleTest::Opaque ) {
				opaque = r;
				break;
			}
		}
	}
	for ( const TriangleResult& r : embreePassedTriangles )
	{
		if ( opaque.test == TriangleTest::Opaque && !TreeOrderLess( r, opaque ) ) {
			break;
		}

		const shaderInfo_t *si = traceInfos[ traceTriangles[ r.tri ].infoNum ].si;

		/* don't double-trace against sky */
		if ( trace->compileFlags & si->compileFlags & C_SKY ) {
			continue;
		}

		/* stack compile flags */
		trace->compileFlags |= si->compileFlags;

		if ( r.test == TriangleTest::Sky ) {
			if( !g_oneSky && trace->light != nullptr /* 0 for -dirty */ && trace->light->skyIndex != -1 && si->skyIndex != -1 )
				bit_enable( trace->skyIndices, si->skyIndex );
			continue;
		}

		/* alphashadow / lightfilter shader */
		trace->forceSubsampling = 1;
		if ( r.test != TriangleTest::Filter ) {
			continue;
		}

		/* ydnar: color filter */
		if ( r.hasFilterColor ) {
			trace->color *= r.filterColor;
		}

		/* ydnar: alpha filter */
		if ( r.hasFilterShadow ) {
			trace->color *= r.filterShadow;
		}

		/* check filter for opaque */
		if ( trace->color[ 0 ] <= 0.001f && trace->color[ 1 ] <= 0.001f && trace->color[ 2 ] <= 0.001f ) {
			trace->color.set( 0 );
			trace->hit = trace->origin + trace->direction * r.depth;
			trace->opaque = true;
			return;
		}
	}

	/* opaque triangle */
	if ( opaque.test == TriangleTest::Opaque ) {
		trace->compileFlags |= traceInfos[ traceTriangles[ opaque.tri ].infoNum ].si->compileFlags;
		trace->hit = trace->origin + trace->direction * opaque.depth;
		trace->color.set( 0 );
		trace->opaque = true;
	}
}

#endif

/* debugging aid: counts how the results of a tracer differ from the reference tracer's,
   printed at exit (Q3MAP2_DEBUG_EMBREE_CHECK, Q3MAP2_DEBUG_GPU_CHECK) */
namespace
{
struct TraceChecker
{
	enum { Rays, OpaqueOnlyTested, OpaqueOnlyReference, Color, Flags, Sky, ForceSubsampling, HitDistance, LitOnlyTested, LitOnlyReference, LitOnlyReferenceFullNodes, Count };
	const char *name, *referenceName;
	std::atomic<long long> counts[ Count ] = {};
	std::atomic<int> printed = 0;

	TraceChecker( const char *name, const char *referenceName ) : name( name ), referenceName( referenceName ){
	}
	~TraceChecker(){
		if ( counts[ Rays ] != 0 ) {
			fprintf( stderr, "%s check: %lld rays; lit only with %s %lld, only with %s %lld (%lld with full node list); opaque only with %s %lld, only with %s %lld; color %lld; compile flags %lld; sky %lld; forceSubsampling %lld; hit distance %lld\n",
			         name, counts[ Rays ].load(), name, counts[ LitOnlyTested ].load(), referenceName, counts[ LitOnlyReference ].load(), counts[ LitOnlyReferenceFullNodes ].load(),
			         name, counts[ OpaqueOnlyTested ].load(), referenceName, counts[ OpaqueOnlyReference ].load(), counts[ Color ].load(),
			         counts[ Flags ].load(), counts[ Sky ].load(), counts[ ForceSubsampling ].load(), counts[ HitDistance ].load() );
		}
	}

	void check( const trace_t& reference, const trace_t& trace ){
		++counts[ Rays ];

		/* what the caller makes of it: sun / sky light (testAll) needs to reach the sky */
		const auto lit = []( const trace_t& t ){
			return !t.opaque && ( !t.testAll || ( t.compileFlags & C_SKY ) );
		};
		if ( lit( trace ) != lit( reference ) ) {
			++counts[ lit( trace ) ? LitOnlyTested : LitOnlyReference ];
			if ( !lit( trace ) && reference.numTestNodes >= MAX_TRACE_TEST_NODES ) {
				++counts[ LitOnlyReferenceFullNodes ];
			}
			if ( printed++ < 12 ) {
				fprintf( stderr, "%s check lit only with %s: origin ( %g %g %g ) end ( %g %g %g ) testAll %d; %s opaque %d hit %g flags %x; %s opaque %d hit %g flags %x\n",
				         name, lit( trace ) ? name : referenceName, reference.origin[ 0 ], reference.origin[ 1 ], reference.origin[ 2 ], reference.end[ 0 ], reference.end[ 1 ], reference.end[ 2 ], reference.testAll,
				         referenceName, reference.opaque, vector3_length( reference.hit - reference.origin ), reference.compileFlags,
				         name, trace.opaque, vector3_length( trace.hit - trace.origin ), trace.compileFlags );
			}
		}
		if ( trace.opaque != reference.opaque ) {
			++counts[ trace.opaque ? OpaqueOnlyTested : OpaqueOnlyReference ];
			return;
		}
		if ( trace.color != reference.color ) {
			++counts[ Color ];
		}
		if ( trace.compileFlags != reference.compileFlags ) {
			++counts[ Flags ];
		}
		if ( memcmp( trace.skyIndices, reference.skyIndices, sizeof( trace.skyIndices ) ) ) {
			++counts[ Sky ];
		}
		if ( trace.forceSubsampling != reference.forceSubsampling ) {
			++counts[ ForceSubsampling ];
		}
		if ( trace.opaque && std::fabs( vector3_length( trace.hit - trace.origin ) - vector3_length( reference.hit - reference.origin ) ) > 0.01f ) {
			++counts[ HitDistance ];
		}
	}
};
}

#ifdef Q3MAP2_EMBREE
/* Q3MAP2_DEBUG_EMBREE_CHECK: counts how the -embree result differs from TraceLine_r()'s */
static void EmbreeCheck( const trace_t& reference, const trace_t& trace ){
	static TraceChecker checker( "embree", "tree" );
	checker.check( reference, trace );
}
#endif



#ifdef Q3MAP2_VULKAN
/* -------------------------------------------------------------------------------

   -gpu: the trace triangles and nodes for the GPU (light_gpu.cpp), and rays traced in batches.
   the GPU repeats TraceLineEmbree(): the same enlarged triangles, TestTraceTriangle() and
   RayPassesLeaf() tests, tree order for rays without a light, solid space from the bsp nodes.

   ------------------------------------------------------------------------------- */

namespace
{
bool gpuReady = false;
std::atomic<long long> gpuOverflowRays;

struct GpuSceneBuilder
{
	GpuScene& scene;
	std::vector<int> parents;

	/* adds the triangles in the leaves, like CollectTraceTriangles_r() */
	void collect( int nodeNum ){
		if ( nodeNum < 0 || nodeNum >= numTraceNodes ) {
			return;
		}
		const traceNode_t& node = traceNodes[ nodeNum ];
		if ( node.type >= 0 ) {
			parents[ node.children[ 0 ] ] = nodeNum;
			parents[ node.children[ 1 ] ] = nodeNum;
			collect( node.children[ 0 ] );
			collect( node.children[ 1 ] );
			return;
		}
		if ( node.type != TRACE_LEAF || node.numItems == 0 ) {
			return;
		}

		/* the path from the root to the leaf: RayPassesLeaf() */
		const int pathOffset = int( scene.paths.size() );
		for ( int child = nodeNum, n = parents[ nodeNum ]; n >= 0; child = n, n = parents[ n ] )
			scene.paths.push_back( uint32_t( n ) | ( traceNodes[ n ].children[ 0 ] == child ? 0u : 1u << 31 ) );
		std::reverse( scene.paths.begin() + pathOffset, scene.paths.end() );
		const int pathLength = int( scene.paths.size() ) - pathOffset;

		for ( int i = 0; i < node.numItems; ++i )
		{
			const traceTriangle_t& tt = traceTriangles[ node.items[ i ] ];
			GpuTriangle& t = scene.triangles.emplace_back();
			for ( int j = 0; j < 3; ++j )
			{
				t.v0[ j ] = tt.v[ 0 ].xyz[ j ];
				t.edge1[ j ] = tt.edge1[ j ];
				t.edge2[ j ] = tt.edge2[ j ];
				t.st[ j ][ 0 ] = tt.v[ j ].st[ 0 ];
				t.st[ j ][ 1 ] = tt.v[ j ].st[ 1 ];
			}
			t.info = tt.infoNum;
			t.pathOffset = pathOffset;
			t.pathLength = pathLength;
			t.item = i;
			t.pad = 0;

			/* enlarged like the embree triangles */
			const Vector3 center = ( tt.v[ 0 ].xyz + tt.v[ 1 ].xyz + tt.v[ 2 ].xyz ) * ( 1.0f / 3.0f );
			for ( int j = 0; j < 3; ++j )
			{
				const Vector3 xyz = center + ( tt.v[ j ].xyz - center ) * 1.06f;
				scene.vertices.insert( scene.vertices.end(), { xyz[ 0 ], xyz[ 1 ], xyz[ 2 ] } );
			}
		}
	}
};

/* builds the GPU scene after SetupTraceNodes() */
void SetupGpu(){
	Sys_Printf( "--- SetupGpu ---\n" );

	GpuScene scene;
	GpuSceneBuilder builder{ scene, std::vector<int>( numTraceNodes, -1 ) };
	builder.collect( headNodeNum );

	for ( int i = 0; i < numTraceNodes; ++i )
	{
		const traceNode_t& node = traceNodes[ i ];
		GpuNode& n = scene.nodes.emplace_back();
		n.normal[ 0 ] = node.plane.normal()[ 0 ];
		n.normal[ 1 ] = node.plane.normal()[ 1 ];
		n.normal[ 2 ] = node.plane.normal()[ 2 ];
		n.dist = node.plane.dist();
		n.type = node.type;
		n.children[ 0 ] = node.children[ 0 ];
		n.children[ 1 ] = node.children[ 1 ];
		n.numItems = node.numItems;
	}

	std::map<const image_t*, int> imageOffsets;
	for ( int i = 0; i < numTraceInfos; ++i )
	{
		const traceInfo_t& ti = traceInfos[ i ];
		const shaderInfo_t *si = ti.si;
		GpuInfo& info = scene.infos.emplace_back();
		info.castShadows = ti.castShadows;
		info.surfaceNum = ti.surfaceNum;
		info.skipGrid = ti.skipGrid;
		info.compileFlags = si->compileFlags;
		info.skyIndex = si->skyIndex;
		info.imageOffset = 0;
		info.imageSize = 0;
		/* TestTraceTriangle() */
		if ( si->compileFlags & C_SKY ) {
			info.kind = GPU_INFO_SKY;
		}
		else if ( !( si->compileFlags & ( C_ALPHASHADOW | C_LIGHTFILTER ) ) || si->lightImage == nullptr || si->lightImage->pixels == nullptr ) {
			info.kind = GPU_INFO_OPAQUE;
		}
		else
		{
			const image_t *image = si->lightImage;
			info.kind = GPU_INFO_FILTER | ( ( si->compileFlags & C_LIGHTFILTER ) ? GPU_INFO_FILTER_COLOR : 0 ) | ( ( si->compileFlags & C_ALPHASHADOW ) ? GPU_INFO_FILTER_SHADOW : 0 );
			if ( image->width > 0xffff || image->height > 0xffff ) {
				Error( "-gpu: image %s is too large", image->name.c_str() );
			}
			const auto [ it, added ] = imageOffsets.emplace( image, int( scene.images.size() ) );
			if ( added ) {
				const size_t texels = size_t( image->width ) * image->height;
				scene.images.resize( scene.images.size() + texels );
				memcpy( scene.images.data() + it->second, image->pixels, texels * 4 );
			}
			info.imageOffset = it->second;
			info.imageSize = image->width << 16 | image->height;
		}
	}

	scene.headNode = headNodeNum;
	scene.numBspNodes = numBspTraceNodes;
	scene.skyFlag = C_SKY;

	gpuReady = GpuSetup( scene );
}
}

bool GpuTraceEnabled(){
	return gpuReady;
}

void GpuTraceBatch::add( const trace_t& trace ){
	GpuRay& ray = m_rays.emplace_back();
	for ( int j = 0; j < 3; ++j )
	{
		ray.origin[ j ] = trace.origin[ j ];
		ray.end[ j ] = trace.end[ j ];
		ray.direction[ j ] = trace.direction[ j ];
		ray.color[ j ] = trace.color[ j ];
	}
	ray.inhibitRadius = trace.inhibitRadius;
	ray.distance = trace.distance;
	ray.recvShadows = trace.recvShadows;
	ray.lightSkyIndex = !g_oneSky && trace.light != nullptr ? trace.light->skyIndex : -1;

	if ( trace.surfaces != m_lastSurfaces || trace.numSurfaces != m_lastNumSurfaces ) {
		m_lastSurfaces = trace.surfaces;
		m_lastNumSurfaces = trace.numSurfaces;
		m_lastOffset = int( m_surfaces.size() );
		m_surfaces.insert( m_surfaces.end(), trace.surfaces, trace.surfaces + trace.numSurfaces );
	}
	ray.surfacesOffset = m_lastOffset;
	ray.numSurfaces = trace.numSurfaces;

	/* debugging aid: env Q3MAP2_DEBUG_GPU_NEAREST traces rays without a light to the nearest opaque triangle */
	static const bool nearest = getenv( "Q3MAP2_DEBUG_GPU_NEAREST" ) != nullptr;
	ray.flags = ( trace.testAll ? GPU_RAY_TEST_ALL : 0 )
	          | ( trace.light == nullptr && !nearest ? GPU_RAY_TREE_ORDER : 0 )
	          | ( inGrid ? GPU_RAY_IN_GRID : 0 )
	          | ( noSurfaces ? GPU_RAY_NO_SURFACES : 0 )
	          /* TraceLine()'s early outs */
	          | ( !trace.recvShadows || !trace.testOcclusion || trace.distance <= 0.00001f ? GPU_RAY_SKIP : 0 );
	ray.pad = 0;
}

void GpuTraceBatch::run(){
	m_hits.resize( m_rays.size() );
	GpuTrace( m_rays.data(), m_hits.data(), m_rays.size(), m_surfaces.data(), m_surfaces.size() );
	m_next = 0;
}

void GpuTraceBatch::apply( trace_t *trace ){
	const GpuRay& ray = m_rays[ m_next ];
	const GpuHit& hit = m_hits[ m_next ];
	++m_next;

	/* the ray as added */
	trace->origin = Vector3( ray.origin[ 0 ], ray.origin[ 1 ], ray.origin[ 2 ] );
	trace->end = Vector3( ray.end[ 0 ], ray.end[ 1 ], ray.end[ 2 ] );
	SetupTrace( trace );
	trace->color = Vector3( ray.color[ 0 ], ray.color[ 1 ], ray.color[ 2 ] );

	/* debugging aid: env Q3MAP2_DEBUG_GPU_CHECK compares each ray with the CPU tracer's result */
	static const bool check = getenv( "Q3MAP2_DEBUG_GPU_CHECK" ) != nullptr;
	trace_t reference;
	if ( check ) {
		reference = *trace;
		TraceLine( &reference );
	}

	if ( hit.status & GPU_HIT_OVERFLOW ) {
		++gpuOverflowRays;
		TraceLine( trace );
	}
	else
	{
		/* TraceLineInner() */
		trace->passSolid = hit.status & GPU_HIT_PASS_SOLID;
		trace->opaque = hit.status & GPU_HIT_OPAQUE;
		trace->compileFlags = hit.compileFlags;
		memset( trace->skyIndices, 0, sizeof( trace->skyIndices ) );
		for ( int i = 0; i < MAX_SKIES; ++i )
		{
			if ( hit.skyBits & ( 1u << i ) ) {
				bit_enable( trace->skyIndices, i );
			}
		}
		trace->numTestNodes = 0;
		if ( hit.status & GPU_HIT_POINT ) {
			trace->hit = Vector3( hit.hit[ 0 ], hit.hit[ 1 ], hit.hit[ 2 ] );
		}
		if ( hit.status & GPU_HIT_DEPTH ) {
			trace->hit = trace->origin + trace->direction * hit.depth;
		}
		if ( hit.status & GPU_HIT_FORCE_SUBSAMPLING ) {
			trace->forceSubsampling = 1;
		}
		trace->color = Vector3( hit.color[ 0 ], hit.color[ 1 ], hit.color[ 2 ] );
	}

	if ( check ) {
		static TraceChecker checker( "gpu", lightEmbree ? "embree" : "tree" );
		checker.check( reference, *trace );
	}
}

void GpuTraceBatch::clear(){
	m_rays.clear();
	m_hits.clear();
	m_surfaces.clear();
	m_lastSurfaces = nullptr;
	m_lastNumSurfaces = -1;
	m_lastOffset = 0;
	m_next = 0;
}

void GpuShutdownTrace(){
	if ( gpuReady ) {
		Sys_Printf( "%9lld GPU rays traced on the CPU (too complex for the shader)\n", gpuOverflowRays.load() );
		GpuShutdown();
		gpuReady = false;
	}
}
#else
bool GpuTraceEnabled(){
	return false;
}
void GpuTraceBatch::add( const trace_t& ){
}
void GpuTraceBatch::run(){
}
void GpuTraceBatch::apply( trace_t *trace ){
	TraceLine( trace );
}
void GpuTraceBatch::clear(){
}
void GpuShutdownTrace(){
}
#endif



/*
   SetupTrace() - ydnar
   sets up certain trace values
 */

float SetupTrace( trace_t *trace ){
	trace->displacement = trace->end - trace->origin;
	trace->direction = trace->displacement;
	trace->distance = VectorFastNormalize( trace->direction );
	trace->hit = trace->origin;
	return trace->distance;
}
