// Copyright (c) 2017, Lawrence Livermore National Security, LLC.  
// Produced at the Lawrence Livermore National Laboratory.
//
// This file is part of Caliper.
// Written by David Boehme, boehme3@llnl.gov.
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

/// \file  OmptService.cpp
/// \brief Service for OpenMP Tools interface 

#include "../CaliperService.h"

#include "caliper/Caliper.h"

#include "caliper/common/Log.h"
#include "caliper/common/RuntimeConfig.h"

#include "caliper/common/util/spinlock.hpp"

#include <map>
#include <mutex>

#include <ompt.h>


using namespace cali;
using namespace std;

namespace 
{

//
// --- Data
//

const ConfigSet::Entry configdata[] = {
    { "capture_state", CALI_TYPE_BOOL, "true",
      "Capture the OpenMP runtime state on context queries",
      "Capture the OpenMP runtime state on context queries"
    },
    { "capture_events", CALI_TYPE_BOOL, "false",
      "Capture OpenMP events (enter/exit parallel regions, barriers, etc.)",
      "Capture OpenMP events (enter/exit parallel regions, barriers, etc.)"
    },
    ConfigSet::Terminator
};

volatile bool          finished    { false };
bool                   enable_ompt { false };
bool		       perm_off    { false };

Attribute              thread_attr { Attribute::invalid };
Attribute              state_attr  { Attribute::invalid };
Attribute	       region_attr { Attribute::invalid };

std::map<int, string>  runtime_states;

ConfigSet              config;

int                    thread_id;
util::spinlock         thread_id_lock;

// The OMPT interface function pointers

struct OmptAPI {
    ompt_set_callback_t     set_callback     { nullptr };
    ompt_get_state_t        get_state        { nullptr };
    ompt_enumerate_states_t enumerate_states { nullptr };

    bool
    init(ompt_function_lookup_t lookup) {
        set_callback     = (ompt_set_callback_t)     (*lookup)("ompt_set_callback");
        get_state        = (ompt_get_state_t)        (*lookup)("ompt_get_state");
        enumerate_states = (ompt_enumerate_states_t) (*lookup)("ompt_enumerate_states");

        if (!set_callback || !get_state || !enumerate_states)
            return false;

        return true;
    }
} api;


//
// --- OMPT Callbacks
//

// ompt_event_thread_begin

void
cb_event_thread_begin(ompt_thread_type_t type, ompt_data_t*)
{
    // Set the thread id in the new environment. We have to make our own since OMPT abandoned it :-(

    std::lock_guard<util::spinlock>
        g(thread_id_lock);

    Caliper().set(thread_attr, Variant(++thread_id));
}

// ompt_event_thread_end

void
cb_event_thread_end(ompt_data_t*)
{
}

// ompt_event_parallel_begin

void
cb_event_parallel_begin(ompt_thread_type_t type)
{
    if (enable_ompt == true && !finished)
        Caliper().begin(region_attr, Variant(CALI_TYPE_STRING, "parallel", 8));
}

// ompt_event_parallel_end

void
cb_event_parallel_end(ompt_thread_type_t type)
{
    if (enable_ompt == true && !finished)
        Caliper().end(region_attr);
}

// ompt_event_idle_begin

void
cb_event_idle(ompt_scope_endpoint_t endpoint)
{
    if (enable_ompt == true && !finished) {
        if (endpoint      == ompt_scope_begin)
            Caliper().begin(region_attr, Variant(CALI_TYPE_STRING, "idle", 4));       
        else if (endpoint == ompt_scope_end)
            Caliper().end(region_attr);
    }
}

// ompt_event_wait_barrier_begin

void
cb_event_sync_region(ompt_sync_region_kind_t kind,
                     ompt_scope_endpoint_t   endpoint,
                     ompt_data_t*            /*parallel_data*/,
                     ompt_data_t*            /*task_data*/,
                     const void*             /*codeptr_ra*/)
{
    if (enable_ompt == true && !finished) {
        if (endpoint == ompt_scope_begin)
            switch (kind) {
            case ompt_sync_region_barrier:
                Caliper().begin(region_attr, Variant(CALI_TYPE_STRING, "barrier",   7));
                break;
            case ompt_sync_region_taskwait:
                Caliper().begin(region_attr, Variant(CALI_TYPE_STRING, "taskwait",  8));
                break;
            case ompt_sync_region_taskgroup:
                Caliper().begin(region_attr, Variant(CALI_TYPE_STRING, "taskgroup", 9));
                break;
            default:
                Caliper().begin(region_attr, Variant(CALI_TYPE_STRING, "UNKNOWN",   7));
                break;
            }
        else if (endpoint == ompt_scope_end)
            Caliper().end(region_attr);
    }
}

void
cb_event_sync_region_wait(ompt_sync_region_kind_t /*kind*/,
                          ompt_scope_endpoint_t   endpoint,
                          ompt_data_t*            /*parallel_data*/,
                          ompt_data_t*            /*task_data*/,
                          const void*             /*codeptr_ra*/)
{
    if (enable_ompt && !finished) {
        if (endpoint == ompt_scope_begin)
            Caliper().begin(region_attr, Variant(CALI_TYPE_STRING, "idle", 4));
        else
            Caliper().end(region_attr);
    }
}

// ompt_event_control

void
cb_event_control_tool(uint64_t command, uint64_t modifier, void* /*arg*/)
{
    // Should react to enable / disable measurement commands.
    switch (command)
    {
    case 1 : // Start or restart monitoring
        if ( perm_off == false && enable_ompt == false) {
            enable_ompt = true;
        }
        break;
    case 2 : // Pause monitoring
        if ( enable_ompt == true ) {
            enable_ompt = false;
        }
        break;
    case 3 : // Flush buffers and continue monitoring
        // To be iplemented if a case arises where we would want to do this.
        break;
    case 4 : // Permanently turn off monitoring
        perm_off = true;
        enable_ompt = false;
        break;
    default :
        break;
    }		    
}

// ompt_event_runtime_shutdown

void
cb_event_runtime_shutdown(void)
{
    // This seems to be called after the Caliper static object has been destroyed.
    // Hence, we can't do much here.
}


//
// -- Caliper callbacks
//

void
finish_cb(Caliper*)
{
    finished = true;
}

void
snapshot_cb(Caliper* c, int scope, const SnapshotRecord*, SnapshotRecord*)
{
    if (!api.get_state || !(scope & CALI_SCOPE_THREAD))
        return;

    auto it = runtime_states.find((*api.get_state)(NULL));

    if (it != runtime_states.end())
        c->set(state_attr, Variant(CALI_TYPE_STRING, it->second.data(), it->second.size()));
}


//
// --- Management
//

/// Register our callbacks with the OpenMP runtime

bool
register_ompt_callbacks(bool capture_events)
{
    if (!api.set_callback)
        return false;

    struct callback_info_t { 
        ompt_callbacks_t event;
        ompt_callback_t  cbptr;
        const char*      name;
    } basic_callbacks[] = {
        { ompt_callback_thread_begin,     (ompt_callback_t) &cb_event_thread_begin,     "thread_begin"     },
        { ompt_callback_thread_end,       (ompt_callback_t) &cb_event_thread_end,       "thread_end"       },
        { ompt_callback_control_tool,     (ompt_callback_t) &cb_event_control_tool,     "control_tool"     }
    }, event_callbacks[] = {
	{ ompt_callback_parallel_begin,   (ompt_callback_t) &cb_event_parallel_begin,   "parallel_begin"   },
	{ ompt_callback_parallel_end,     (ompt_callback_t) &cb_event_parallel_end,     "parallel_end"     },
        { ompt_callback_sync_region,      (ompt_callback_t) &cb_event_sync_region,      "sync_region"      },
        { ompt_callback_sync_region_wait, (ompt_callback_t) &cb_event_sync_region_wait, "sync_region_wait" },
        { ompt_callback_idle,             (ompt_callback_t) &cb_event_idle,              "idle"            }
    };

    for ( auto cb : basic_callbacks ) 
        if ((*api.set_callback)(cb.event, cb.cbptr) == 0) {
            Log(0).stream() << "ompt: Error: Unable to register callback \"" << cb.name << "\"" << std::endl;
            return false;
        }
    if (capture_events)
        for ( auto cb : event_callbacks ) 
            if ((*api.set_callback)(cb.event, cb.cbptr) == 0) {
                Log(0).stream() << "ompt: Error: Unable to register callback \"" << cb.name << "\"" << std::endl;
                return false;
            }

    return true;
}

bool 
register_ompt_states()
{
    if (!api.enumerate_states)
        return false;

    int         state = omp_state_undefined;
    const char* state_name;

    while ((*api.enumerate_states)(state, &state, &state_name))
        runtime_states[state] = state_name;

    return true;
}


/// The Caliper service initialization callback.
/// Register attributes and set Caliper callbacks here.

void 
omptservice_initialize(Caliper* c) 
{
    config      = RuntimeConfig::init("ompt", configdata);

    enable_ompt = true;

    thread_attr = 
        c->create_attribute("ompt.thread.id", CALI_TYPE_INT, CALI_ATTR_SCOPE_THREAD);
    
    state_attr  =
        c->create_attribute("ompt.state",     CALI_TYPE_STRING, 
                            CALI_ATTR_SCOPE_THREAD | CALI_ATTR_SKIP_EVENTS);
    region_attr =
	c->create_attribute("ompt.region",    CALI_TYPE_STRING,
			    CALI_ATTR_SCOPE_THREAD | CALI_ATTR_NESTED);

    c->events().finish_evt.connect(&finish_cb);

    Log(1).stream() << "Registered OMPT service." << std::endl;
}

// The OpenMP tools interface intialization function, called by the OpenMP 
// runtime. We must register our callbacks w/ the OpenMP runtime here.

int
cali_ompt_initialize(ompt_function_lookup_t lookup, ompt_fns_t*)
{
    Log(1).stream() << "ompt: Initializing OMPT interface." << std::endl;

    Caliper c;
    
    // register callbacks

    if (!::api.init(lookup) || !::register_ompt_callbacks(::config.get("capture_events").to_bool())) {
        Log(0).stream() << "ompt: Callback registration error: OMPT interface disabled" << endl;
        return 0;
    }

    if (::config.get("capture_state").to_bool() == true) {
        register_ompt_states();
        c.events().snapshot.connect(&snapshot_cb);
    }

    // set default thread ID
    c.set(thread_attr, Variant(0));
        
    Log(1).stream() << "ompt: OMPT interface enabled." << endl;

    return 1;
}

void
cali_ompt_finalize(ompt_fns_t*)
{
    finished = true;
}

ompt_fns_t cali_ompt_fns { cali_ompt_initialize, cali_ompt_finalize };

}  // namespace [ anonymous ]


extern "C" {

// The tool entry point called by OMPT. Return our initialize/finalize
//   functions here if ompt service is enabled.
ompt_fns_t*
ompt_start_tool(unsigned int ompt_version, const char* runtime_version)
{
    Log(1).stream() << "Caliper OMPT entry function invoked from " << runtime_version << std::endl;

    // Make sure Caliper is initialized & OMPT service is enabled    
    Caliper::instance();

    if (!::enable_ompt)
        return NULL;
    
    return ::enable_ompt ? &::cali_ompt_fns : NULL;
}
    
} // extern "C"

namespace cali
{
    CaliperService ompt_service = { "ompt", ::omptservice_initialize };
}
