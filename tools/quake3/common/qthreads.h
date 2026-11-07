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

#pragma once

#include <functional>

extern int numthreads;

void ThreadSetDefault();
void RunThreadsOnIndividual( int workcnt, bool showpacifier, void ( *func )( int ) );
void ThreadLock();
void ThreadUnlock();

/* when enabled, worker threads that run out of RunThreadsOnIndividual items stay alive
   and help with ParallelFor jobs started by the remaining busy items, instead of exiting */
extern bool threadsHelpWithJobs;

/* runs fn( 0 .. count-1 ), sharing the indexes with idle worker threads when threadsHelpWithJobs
   is enabled and this is called from inside RunThreadsOnIndividual; otherwise runs them in order
   on the calling thread. returns when all indexes are done. */
void ParallelFor( int count, const std::function<void( int )>& fn );

/* runs the process below normal priority; returns false if that failed */
bool ThreadSetLowPriority();
