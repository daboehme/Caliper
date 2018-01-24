// Copyright (c) 2015, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
//
// This file is part of Caliper.
// Written by Alfredo Gimenez, gimenez1@llnl.gov.
// LLNL-CODE-678900
// All rights reserved.
//
// For details, see https://github.com/scalability-llnl/Caliper.
// Please also see the LICENSE file for our additional BSD notice.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the disclaimer below.
//  * Redistributions in binary form must reproduce the above copyright notice, this list of
//    conditions and the disclaimer (as noted below) in the documentation and/or other materials
//    provided with the distribution.
//  * Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse
//    or promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/// \file DataTracker.h
/// \brief Caliper C++ data tracking interface

#pragma once

#include "common/util/callback.hpp"

#include <vector>
#include <string>
#include <cstdlib>
#include <cinttypes>

namespace cali
{

namespace DataTracker
{

struct Events {
    util::callback<void(void* ptr, const char* label, size_t elem_size, size_t ndim, const size_t dims[])>
    track_memory_evt;
    
    util::callback<void(void* ptr)>
    untrack_memory_evt;
};

Events* events();

void* Allocate(const char*        label,
               const size_t       size);

void* Allocate(const char*        label,
               const size_t       elem_size,
               const size_t       ndims,
               const size_t       dimensions[]);

void Free(void *ptr);

void TrackAllocation(void         *ptr,
                     const char*  label,
                     size_t       size);

void TrackAllocation(void *ptr,
                     const char*  label,
                     const size_t elem_size,
                     const size_t ndims,
                     const size_t dimensions[]);

void UntrackAllocation(void *ptr);

} // namespace DataTracker

} // namespace cali
