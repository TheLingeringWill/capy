//
// Copyright (c) 2026 Michael Vandeberg
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Official repository: https://github.com/cppalliance/capy
//

#ifndef BOOST_CAPY_EX_FRAME_ALLOC_MIXIN_HPP
#define BOOST_CAPY_EX_FRAME_ALLOC_MIXIN_HPP

#include <boost/capy/detail/config.hpp>
#include <boost/capy/detail/frame_memory_resource.hpp>
#include <boost/capy/ex/frame_allocator.hpp>
#include <boost/capy/ex/recycling_memory_resource.hpp>

#include <cstddef>
#include <cstring>
#include <memory_resource>

// The recycling fast path is forced inline into each coroutine's ramp and
// destroy functions, where the frame size is a constant, so the size class
// folds away. Not on MSVC, which may keep an inlined thread_local's address
// in a coroutine frame across a suspension (see current_frame_allocator_ref).
#if BOOST_CAPY_WORKAROUND(BOOST_CAPY_MSVC, >= 1)
# define BOOST_CAPY_FRAME_INLINE inline
#else
# define BOOST_CAPY_FRAME_INLINE BOOST_CAPY_FORCEINLINE
#endif

namespace boost {
namespace capy {

namespace detail {

// Frames from any resource other than the recycling one, out of line so
// the inlined recycling path stays small. A null resource means "not set".
BOOST_CAPY_NOINLINE inline void*
allocate_frame_other(std::pmr::memory_resource* mr, std::size_t size)
{
    if(!mr)
        mr = std::pmr::get_default_resource();
    void* raw = mr->allocate(size + sizeof(mr), frame_alignment);
    std::memcpy(static_cast<char*>(raw) + size, &mr, sizeof(mr));
    return raw;
}

BOOST_CAPY_NOINLINE inline void
deallocate_frame_other(
    std::pmr::memory_resource* mr, void* ptr, std::size_t size) noexcept
{
    mr->deallocate(ptr, size + sizeof(mr), frame_alignment);
}

} // namespace detail

/** Mixin that adds frame-allocator-aware allocation to a promise type.

    Inherit from this class in any coroutine promise type to opt into
    TLS-based frame allocation with the recycling memory resource
    fast path. The mixin provides `operator new` and `operator delete`
    that:

    1. Read the thread-local frame allocator set by `run_async` or `run`.
    2. Bypass virtual dispatch when the allocator is the default
       recycling memory resource.
    3. Store the allocator pointer at the end of each frame for
       correct deallocation even when TLS changes between allocation
       and deallocation.

    This is the same allocation strategy used by @ref
    context_awaitable_promise_base. Use this mixin directly when your
    promise type does not need the full environment and continuation
    support that `context_awaitable_promise_base` provides.

    @par Example
    @par !example example


    @par Thread Safety
    The allocation fast path uses thread-local storage and requires
    no synchronization. The global pool fallback is mutex-protected.

    @see @ref context_awaitable_promise_base, @ref recycling_memory_resource
*/
struct frame_alloc_mixin
{
    /** Allocate a coroutine frame.

        Uses the thread-local frame allocator set by run_async.
        Falls back to default memory resource if not set.
        Stores the allocator pointer at the end of each frame for
        correct deallocation even when TLS changes. Uses memcpy
        to avoid alignment requirements on the trailing pointer.
        Bypasses virtual dispatch for the recycling allocator.

        @param size The size, in bytes, of the coroutine frame.

        @return A pointer to storage for the frame.

        @par Exception Safety
        Propagates any exception thrown by the underlying memory
        resource's `allocate`, for example `std::bad_alloc` from
        `::operator new`.
    */
    BOOST_CAPY_FRAME_INLINE static void* operator new(std::size_t size)
    {
        auto* const rmr = &detail::recycling_instance.mr;

        std::pmr::memory_resource* mr = get_current_frame_allocator();
        if(mr != rmr) [[unlikely]]
            return detail::allocate_frame_other(mr, size);

        void* raw = rmr->allocate_fast(
            size + sizeof(mr), detail::frame_alignment);
        std::memcpy(static_cast<char*>(raw) + size, &mr, sizeof(mr));
        return raw;
    }

    /** Deallocate a coroutine frame.

        Reads the allocator pointer stored at the end of the frame
        to ensure correct deallocation regardless of current TLS.
        Bypasses virtual dispatch for the recycling allocator.

        @param ptr The frame storage returned by `operator new`.

        @param size The size, in bytes, that was passed to `operator new`.
        The allocator pointer is read from `ptr + size`, which is where
        `operator new` wrote it, so this value must match.
    */
    BOOST_CAPY_FRAME_INLINE static void
    operator delete(void* ptr, std::size_t size) noexcept
    {
        auto* const rmr = &detail::recycling_instance.mr;

        std::pmr::memory_resource* mr;
        std::memcpy(&mr, static_cast<char*>(ptr) + size, sizeof(mr));
        if(mr != rmr) [[unlikely]]
        {
            detail::deallocate_frame_other(mr, ptr, size);
            return;
        }
        rmr->deallocate_fast(
            ptr, size + sizeof(mr), detail::frame_alignment);
    }
};

namespace detail {

// For capy's own frames whose teardown can release the chain that owns
// the frame allocator: they are never allocated from it, so freeing them
// afterwards never reaches a resource that died with that chain.
struct process_frame_mixin
{
    BOOST_CAPY_FRAME_INLINE static void* operator new(std::size_t size)
    {
        return recycling_instance.mr.allocate_fast(size, frame_alignment);
    }

    BOOST_CAPY_FRAME_INLINE static void
    operator delete(void* ptr, std::size_t size) noexcept
    {
        recycling_instance.mr.deallocate_fast(ptr, size, frame_alignment);
    }
};

} // namespace detail

} // namespace capy
} // namespace boost

#endif
