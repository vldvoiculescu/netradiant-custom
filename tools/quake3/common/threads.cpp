/*
   Copyright (C) 1999-2006 Id Software, Inc. and contributors.
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
 */

#include "cmdlib.h"
#include "inout.h"
#include "qthreads.h"
#include "timer.h"

#define MAX_THREADS 64
int numthreads = -1;
bool threaded;

int dispatch;
int workcount;
int oldf;
bool pacifier;

/*
   =============
   GetThreadWork

   =============
 */
int GetThreadWork(){
	ThreadLock();

	if ( dispatch == workcount ) {
		ThreadUnlock();
		return -1;
	}

	const int f = 40 * dispatch / workcount;
	if ( f < oldf ) {
		Sys_Warning( "progress went backwards (should never happen)\n" );
		oldf = f;
	}
	while ( f > oldf )
	{
		++oldf;
		if ( pacifier ) {
			if ( oldf % 4 == 0 ) {
				Sys_Printf( "%i", f / 4 );
			}
			else{
				Sys_Printf( "." );
			}
			fflush( stdout );   /* ydnar */
		}
	}

	const int r = dispatch++;
	ThreadUnlock();

	return r;
}


/*
   thread profiling - debugging aid
   when env Q3MAP2_DEBUG_THREADPROFILE names a file, every RunThreadsOnIndividual call writes
   its wall time, the time spent in work items, the serial time since the previous call,
   when the last item was handed out, and the slowest items.
 */

#include <chrono>
#include <vector>
#include <algorithm>

namespace
{
using ProfileClock = std::chrono::steady_clock;

struct ProfileItem
{
	double start, end;
	int thread;
};

struct ThreadProfile
{
	FILE *file = nullptr;
	ProfileClock::time_point origin = ProfileClock::now();
	double lastEnd = 0;
	int call = 0;
	std::vector<ProfileItem> items;

	ThreadProfile(){
		if ( const char *path = getenv( "Q3MAP2_DEBUG_THREADPROFILE" ) ) {
			file = fopen( path, "wt" );
		}
	}
	~ThreadProfile(){
		if ( file != nullptr ) {
			fprintf( file, "end %.4f serial_before %.4f\n", now(), now() - lastEnd );
			fclose( file );
		}
	}
	double now() const {
		return std::chrono::duration<double>( ProfileClock::now() - origin ).count();
	}
};

ThreadProfile g_threadProfile;
}


/*
   ParallelFor jobs
   a RunThreadsOnIndividual item can split its own work with ParallelFor. threads that have no
   items left help with those jobs (like a task scheduler's work sharing), so a single huge
   item no longer keeps one core busy while the others sit idle.
 */

#include <atomic>
#include <mutex>
#include <condition_variable>

void ( *threadProfileFlush )() = nullptr;

namespace
{
std::atomic<double> g_counterSeconds;
std::atomic<long long> g_counterCalls;
}

bool ThreadProfileEnabled(){
	return g_threadProfile.file != nullptr;
}

void ThreadProfileAddCounter( double seconds, long long calls ){
	g_counterSeconds.fetch_add( seconds );
	g_counterCalls.fetch_add( calls );
}

bool threadsHelpWithJobs = false;

namespace
{
struct ParallelJob
{
	const std::function<void( int )> *fn;
	int count;
	std::atomic<int> next{ 0 };
	std::atomic<int> done{ 0 };
};

std::mutex g_jobMutex;
std::condition_variable g_jobCond;
std::vector<ParallelJob*> g_jobs;               /* active jobs, guarded by g_jobMutex */
int g_itemsDone;                                /* finished RunThreadsOnIndividual items, guarded by g_jobMutex */
std::atomic<double> g_helpSeconds;              /* profiling: time idle workers spent on job indexes */

/* runs one pending index of any active job; lock is held on entry and exit */
bool HelpWithJob( std::unique_lock<std::mutex>& lock, bool profile ){
	for ( ParallelJob *job : g_jobs )
	{
		const int i = job->next.fetch_add( 1 );
		if ( i < job->count ) {
			lock.unlock();
			const double start = profile ? g_threadProfile.now() : 0;
			( *job->fn )( i );
			if ( profile ) {
				g_helpSeconds.fetch_add( g_threadProfile.now() - start );
			}
			lock.lock();
			if ( ++job->done == job->count ) {
				g_jobCond.notify_all();
			}
			return true;
		}
	}
	return false;
}
}

void ParallelFor( int count, const std::function<void( int )>& fn ){
	if ( !threadsHelpWithJobs || !threaded || numthreads <= 1 || count <= 1 ) {
		for ( int i = 0; i < count; ++i )
			fn( i );
		return;
	}

	ParallelJob job;
	job.fn = &fn;
	job.count = count;

	std::unique_lock<std::mutex> lock( g_jobMutex );
	g_jobs.push_back( &job );
	g_jobCond.notify_all();

	while ( job.done < count )
	{
		/* own indexes first */
		const int i = job.next.fetch_add( 1 );
		if ( i < count ) {
			lock.unlock();
			fn( i );
			lock.lock();
			++job.done;
			continue;
		}
		/* the rest are being run by helpers: help others meanwhile, or wait */
		if ( !HelpWithJob( lock, false ) ) {
			g_jobCond.wait( lock );
		}
	}

	g_jobs.erase( std::find( g_jobs.begin(), g_jobs.end(), &job ) );
}

void ( *workfunction )( int );

void ThreadWorkerFunction( int threadnum ){
	while ( true )
	{
		const int work = GetThreadWork();
		if ( work == -1 ) {
			break;
		}
//Sys_Printf ("thread %i, work %i\n", threadnum, work);
		if ( g_threadProfile.file != nullptr ) {
			ProfileItem& item = g_threadProfile.items[ work ];
			item.thread = threadnum;
			item.start = g_threadProfile.now();
			workfunction( work );
			item.end = g_threadProfile.now();
		}
		else{
			workfunction( work );
		}
		if ( threadsHelpWithJobs ) {
			std::lock_guard<std::mutex> lock( g_jobMutex );
			if ( ++g_itemsDone == workcount ) {
				g_jobCond.notify_all();
			}
		}
	}

	/* no items left to start: help the items still running until all of them are done */
	if ( threadsHelpWithJobs ) {
		std::unique_lock<std::mutex> lock( g_jobMutex );
		while ( true )
		{
			if ( HelpWithJob( lock, g_threadProfile.file != nullptr ) ) {
				continue;
			}
			if ( g_itemsDone == workcount ) {
				break;
			}
			g_jobCond.wait( lock );
		}
	}

	if ( threadProfileFlush != nullptr && g_threadProfile.file != nullptr ) {
		threadProfileFlush();
	}
}

static void ThreadProfileReport( int workcnt, bool showpacifier, double start ){
	ThreadProfile& p = g_threadProfile;
	const double end = p.now();
	const double wall = end - start;
	double busy = 0, lastStart = start, maxItem = 0;
	std::vector<double> threadBusy( std::max( numthreads, 1 ), 0 );
	std::vector<int> order( workcnt );
	busy += g_helpSeconds.load();
	for ( int i = 0; i < workcnt; ++i )
	{
		const ProfileItem& item = p.items[ i ];
		busy += item.end - item.start;
		threadBusy[ std::clamp( item.thread, 0, int( threadBusy.size() ) - 1 ) ] += item.end - item.start;
		lastStart = std::max( lastStart, item.start );
		maxItem = std::max( maxItem, item.end - item.start );
		order[ i ] = i;
	}
	const int top = std::min( workcnt, 8 );
	std::partial_sort( order.begin(), order.begin() + top, order.end(), [&p]( int a, int b ){
		return p.items[ a ].end - p.items[ a ].start > p.items[ b ].end - p.items[ b ].start;
	} );
	fprintf( p.file, "call %d pacifier %d items %d threads %d start %.4f serial_before %.4f wall %.4f busy %.4f efficiency %.3f last_dispatch %.4f tail %.4f max_item %.4f\n",
	         p.call, showpacifier, workcnt, numthreads, start, start - p.lastEnd, wall, busy,
	         wall > 0 ? busy / ( wall * std::max( numthreads, 1 ) ) : 1.0, lastStart - start, end - lastStart, maxItem );
	if ( g_counterCalls != 0 ) {
		fprintf( p.file, "  counter %.4f s, %lld calls, %.1f%% of busy\n", g_counterSeconds.load(), g_counterCalls.load(), busy > 0 ? 100 * g_counterSeconds.load() / busy : 0.0 );
	}
	fprintf( p.file, "  slowest" );
	for ( int i = 0; i < top; ++i )
		fprintf( p.file, " %d:%.4f@%.4f", order[ i ], p.items[ order[ i ] ].end - p.items[ order[ i ] ].start, p.items[ order[ i ] ].start - start );
	fprintf( p.file, "\n  thread_busy" );
	for ( const double b : threadBusy )
		fprintf( p.file, " %.3f", b );
	fprintf( p.file, "\n" );
	fflush( p.file );
	p.lastEnd = end;
	p.call++;
}

void RunThreadsOn( void ( *func )( int ) );

void RunThreadsOnIndividual( int workcnt, bool showpacifier, void ( *func )( int ) ){
	if ( numthreads == -1 ) {
		ThreadSetDefault();
	}
	Timer timer;
	const double profileStart = g_threadProfile.file != nullptr ? g_threadProfile.now() : 0;
	if ( g_threadProfile.file != nullptr ) {
		g_threadProfile.items.assign( workcnt, ProfileItem{ profileStart, profileStart, -1 } );
		g_helpSeconds = 0;
		g_counterSeconds = 0;
		g_counterCalls = 0;
	}
	g_itemsDone = 0;

	dispatch = 0;
	workcount = workcnt;
	oldf = -1;
	pacifier = showpacifier;

	workfunction = func;
	RunThreadsOn( ThreadWorkerFunction );

	if ( g_threadProfile.file != nullptr ) {
		ThreadProfileReport( workcnt, showpacifier, profileStart );
	}

	if ( pacifier ) {
		Sys_Printf( " (%i)\n", int( timer.elapsed_sec() ) );
	}
}


#if 1

#include <thread>
#include <mutex>

std::mutex crit;
static bool enter;

void ThreadSetDefault(){
	if ( numthreads == -1 ) { // not set manually
		numthreads = std::max( std::thread::hardware_concurrency(), 1u );
	}

	Sys_Printf( "%i threads\n", numthreads );
}


void ThreadLock(){
	if ( threaded ) {
		crit.lock();
		if ( enter ) {
			Error( "Recursive ThreadLock\n" );
		}
		enter = true;
	}
}

void ThreadUnlock(){
	if ( threaded ) {
		if ( !enter ) {
			Error( "ThreadUnlock without lock\n" );
		}
		enter = false;
		crit.unlock();
	}
}

/*
   =============
   RunThreadsOn
   =============
 */
void RunThreadsOn( void ( *func )( int ) ){
	if ( numthreads == 1 ) { // use same thread
		func( 0 );
	}
	else
	{
		threaded = true;

		std::thread threads[MAX_THREADS];

		for ( int i = 0; i < numthreads; ++i )
			threads[i] = std::thread( func, i );

		for ( int i = 0; i < numthreads; ++i )
			threads[i].join();

		threaded = false;
	}
}

#else

#ifndef WIN32
// The below define is necessary to use
// pthreads extensions like pthread_mutexattr_settype
#define _GNU_SOURCE
#include <pthread.h>
#endif

/*
   ===================================================================

   WIN32

   ===================================================================
 */
#ifdef WIN32

#define USED

#include <windows.h>

CRITICAL_SECTION crit;
static int enter;

void ThreadSetDefault(){
	SYSTEM_INFO info;

	if ( numthreads == -1 ) { // not set manually
		GetSystemInfo( &info );
		numthreads = info.dwNumberOfProcessors;
		if ( numthreads < 1 || numthreads > 32 ) {
			numthreads = 1;
		}
	}

	Sys_Printf( "%i threads\n", numthreads );
}


void ThreadLock(){
	if ( !threaded ) {
		return;
	}
	EnterCriticalSection( &crit );
	if ( enter ) {
		Error( "Recursive ThreadLock\n" );
	}
	enter = 1;
}

void ThreadUnlock(){
	if ( !threaded ) {
		return;
	}
	if ( !enter ) {
		Error( "ThreadUnlock without lock\n" );
	}
	enter = 0;
	LeaveCriticalSection( &crit );
}

/*
   =============
   RunThreadsOn
   =============
 */
void RunThreadsOn( void ( *func )( int ) ){
	HANDLE threadhandle[MAX_THREADS];
	int i;

	threaded = true;

	//
	// run threads in parallel
	//
	InitializeCriticalSection( &crit );

	if ( numthreads == 1 ) { // use same thread
		func( 0 );
	}
	else
	{
		for ( i = 0; i < numthreads; ++i )
		{
			threadhandle[i] = CreateThread(
			                      NULL,                         // LPSECURITY_ATTRIBUTES  lpThreadAttributes,
			                      //0,                          // SIZE_T                 dwStackSize,

			                      /* ydnar: cranking stack size to eliminate radiosity crash with 1MB stack on win32 */
			                      ( 4096 * 1024 ),

			                      (LPTHREAD_START_ROUTINE)func, // LPTHREAD_START_ROUTINE lpStartAddress,
			                      (LPVOID)i,                    // LPVOID                 lpParameter,
			                      0,                            // DWORD                  dwCreationFlags,
			                      NULL );                       // LPDWORD                lpThreadId
		}

		for ( i = 0; i < numthreads; ++i )
			WaitForSingleObject( threadhandle[i], INFINITE );
	}
	DeleteCriticalSection( &crit );

	threaded = false;
}


#endif

/*
   ===================================================================

   OSF1

   ===================================================================
 */

#ifdef __osf__
#define USED

void ThreadSetDefault(){
	if ( numthreads == -1 ) { // not set manually
		numthreads = 4;
	}
}


#include <pthread.h>

pthread_mutex_t *my_mutex;

void ThreadLock(){
	if ( my_mutex ) {
		pthread_mutex_lock( my_mutex );
	}
}

void ThreadUnlock(){
	if ( my_mutex ) {
		pthread_mutex_unlock( my_mutex );
	}
}


/*
   =============
   RunThreadsOn
   =============
 */
void RunThreadsOn( void ( *func )( int ) ){
	int i;
	pthread_t work_threads[MAX_THREADS];
	pthread_addr_t status;
	pthread_attr_t attrib;
	pthread_mutexattr_t mattrib;

	threaded = true;

	if ( pacifier ) {
		setbuf( stdout, NULL );
	}

	if ( !my_mutex ) {
		my_mutex = safe_malloc( sizeof( *my_mutex ) );
		if ( pthread_mutexattr_create( &mattrib ) == -1 ) {
			Error( "pthread_mutex_attr_create failed" );
		}
		if ( pthread_mutexattr_setkind_np( &mattrib, MUTEX_FAST_NP ) == -1 ) {
			Error( "pthread_mutexattr_setkind_np failed" );
		}
		if ( pthread_mutex_init( my_mutex, mattrib ) == -1 ) {
			Error( "pthread_mutex_init failed" );
		}
	}

	if ( pthread_attr_create( &attrib ) == -1 ) {
		Error( "pthread_attr_create failed" );
	}
	if ( pthread_attr_setstacksize( &attrib, 0x100000 ) == -1 ) {
		Error( "pthread_attr_setstacksize failed" );
	}

	for ( i = 0; i < numthreads; ++i )
	{
		if ( pthread_create( &work_threads[i], attrib
		                     , (pthread_startroutine_t)func, (pthread_addr_t)i ) == -1 ) {
			Error( "pthread_create failed" );
		}
	}

	for ( i = 0; i < numthreads; ++i )
	{
		if ( pthread_join( work_threads[i], &status ) == -1 ) {
			Error( "pthread_join failed" );
		}
	}

	threaded = false;
}


#endif

/*
   ===================================================================

   IRIX

   ===================================================================
 */

#ifdef _MIPS_ISA
#define USED

#include <task.h>
#include <abi_mutex.h>
#include <sys/types.h>
#include <sys/prctl.h>


abilock_t lck;

void ThreadSetDefault(){
	if ( numthreads == -1 ) {
		numthreads = prctl( PR_MAXPPROCS );
	}
	Sys_Printf( "%i threads\n", numthreads );
	usconfig( CONF_INITUSERS, numthreads );
}


void ThreadLock(){
	spin_lock( &lck );
}

void ThreadUnlock(){
	release_lock( &lck );
}


/*
   =============
   RunThreadsOn
   =============
 */
void RunThreadsOn( void ( *func )( int ) ){
	int i;
	int pid[MAX_THREADS];

	threaded = true;

	if ( pacifier ) {
		setbuf( stdout, NULL );
	}

	init_lock( &lck );

	for ( i = 0; i < numthreads - 1; ++i )
	{
		pid[i] = sprocsp( ( void ( * )( void *, size_t ) )func, PR_SALL, (void *)i
		                  , NULL, 0x200000 ); // 2 meg stacks
		if ( pid[i] == -1 ) {
			perror( "sproc" );
			Error( "sproc failed" );
		}
	}

	func( i );

	for ( i = 0; i < numthreads - 1; ++i )
		wait( NULL );

	threaded = false;
}


#endif


/*
   =======================================================================

   Linux pthreads

   =======================================================================
 */

#if defined( __linux__ ) || defined( __APPLE__ )
#define USED

#include <unistd.h>

void ThreadSetDefault(){
	if ( numthreads == -1 ) { // not set manually
#ifdef _SC_NPROCESSORS_ONLN
		long cpus = sysconf( _SC_NPROCESSORS_ONLN );
		if ( cpus > 0 ) {
			numthreads = cpus;
		}
		else
#endif
		/* can't detect, so default to four threads */
		numthreads = 4;
	}

	if ( numthreads > 1 ) {
		Sys_Printf( "threads: %d\n", numthreads );
	}
}

#include <pthread.h>

struct pt_mutex_t
{
	pthread_t       *owner;
	pthread_mutex_t a_mutex;
	pthread_cond_t cond;
	unsigned int lock;
};

pt_mutex_t global_lock;

void ThreadLock(){
	pt_mutex_t *pt_mutex = &global_lock;

	if ( !threaded ) {
		return;
	}

	pthread_mutex_lock( &pt_mutex->a_mutex );
	if ( pthread_equal( pthread_self(), (pthread_t)&pt_mutex->owner ) ) {
		pt_mutex->lock++;
	}
	else
	{
		if ( ( !pt_mutex->owner ) && ( pt_mutex->lock == 0 ) ) {
			pt_mutex->owner = (pthread_t *)pthread_self();
			pt_mutex->lock  = 1;
		}
		else
		{
			while ( true )
			{
				pthread_cond_wait( &pt_mutex->cond, &pt_mutex->a_mutex );
				if ( ( !pt_mutex->owner ) && ( pt_mutex->lock == 0 ) ) {
					pt_mutex->owner = (pthread_t *)pthread_self();
					pt_mutex->lock  = 1;
					break;
				}
			}
		}
	}
	pthread_mutex_unlock( &pt_mutex->a_mutex );
}

void ThreadUnlock(){
	pt_mutex_t *pt_mutex = &global_lock;

	if ( !threaded ) {
		return;
	}

	pthread_mutex_lock( &pt_mutex->a_mutex );
	pt_mutex->lock--;

	if ( pt_mutex->lock == 0 ) {
		pt_mutex->owner = NULL;
		pthread_cond_signal( &pt_mutex->cond );
	}

	pthread_mutex_unlock( &pt_mutex->a_mutex );
}

void recursive_mutex_init( pthread_mutexattr_t attribs ){
	pt_mutex_t *pt_mutex = &global_lock;

	pt_mutex->owner = NULL;
	if ( pthread_mutex_init( &pt_mutex->a_mutex, &attribs ) != 0 ) {
		Error( "pthread_mutex_init failed\n" );
	}
	if ( pthread_cond_init( &pt_mutex->cond, NULL ) != 0 ) {
		Error( "pthread_cond_init failed\n" );
	}

	pt_mutex->lock = 0;
}

/*
   =============
   RunThreadsOn
   =============
 */
void RunThreadsOn( void ( *func )( int ) ){
	pthread_mutexattr_t mattrib;
	pthread_attr_t attr;
	pthread_t work_threads[MAX_THREADS];
	size_t stacksize;

	int i = 0;

	pthread_attr_init( &attr );
	if ( pthread_attr_setstacksize( &attr, 8388608 ) != 0 ) {
		stacksize = 0;
		pthread_attr_getstacksize( &attr, &stacksize );
		Sys_Printf( "Could not set a per-thread stack size of 8 MB, using only %.2f MB\n", stacksize / 1048576.0 );
	}

	if ( numthreads == 1 ) {
		func( 0 );
	}
	else
	{
		threaded  = true;

		if ( pacifier ) {
			setbuf( stdout, NULL );
		}

		if ( pthread_mutexattr_init( &mattrib ) != 0 ) {
			Error( "pthread_mutexattr_init failed" );
		}
		if ( pthread_mutexattr_settype( &mattrib, PTHREAD_MUTEX_ERRORCHECK ) != 0 ) {
			Error( "pthread_mutexattr_settype failed" );
		}
		recursive_mutex_init( mattrib );

		for ( i = 0; i < numthreads; ++i )
		{
			/* Default pthread attributes: joinable & non-realtime scheduling */
			if ( pthread_create( &work_threads[i], &attr, (void *(*)(void *)) func, (void*)(size_t)i ) != 0 ) {
				Error( "pthread_create failed" );
			}
		}
		for ( i = 0; i < numthreads; ++i )
		{
			if ( pthread_join( work_threads[i], NULL ) != 0 ) {
				Error( "pthread_join failed" );
			}
		}
		pthread_mutexattr_destroy( &mattrib );
		threaded = false;
	}
}
#endif // ifdef __linux__


/*
   =======================================================================

   SINGLE THREAD

   =======================================================================
 */

#ifndef USED

void ThreadSetDefault(){
	numthreads = 1;
}

void ThreadLock(){
}

void ThreadUnlock(){
}

/*
   =============
   RunThreadsOn
   =============
 */
void RunThreadsOn( int workcnt, bool showpacifier, void ( *func )( int ) ){
	func( 0 );
}

#endif

#endif


/*
   ThreadSetLowPriority
   lets the whole process run below normal priority, so interactive programs stay responsive
   while compiling; it still gets all of the cpu that they leave idle.
   (deliberately not the stronger background/efficiency modes, which would also limit
   the process to efficiency cores and throttle its io on hybrid cpus)
 */

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

bool ThreadSetLowPriority(){
	return SetPriorityClass( GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS );
}
#else
#include <sys/resource.h>

bool ThreadSetLowPriority(){
	return setpriority( PRIO_PROCESS, 0, 10 ) == 0;
}
#endif
