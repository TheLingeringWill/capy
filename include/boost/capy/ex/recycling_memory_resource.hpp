//
// Copyright (c) 2025 Vinnie Falco (vinnie.falco@gmail.com)
// Copyright (c) 2026 Michael Vandeberg
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Official repository: https://github.com/cppalliance/capy
//

#ifndef BOOST_CAPY_RECYCLING_MEMORY_RESOURCE_HPP
#define BOOST_CAPY_RECYCLING_MEMORY_RESOURCE_HPP

#include <boost/capy/detail/config.hpp>

#include <bit>
#include <cstddef>
#include <memory_resource>
#include <mutex>

namespace boost {
namespace capy {

/** Recycles freed blocks through per-thread pools, with a shared pool for cross-thread reuse.

    This memory resource recycles memory blocks using power-of-two
    size classes for O(1) allocation lookup. It maintains a thread-local
    pool for fast lock-free access and a global pool for cross-thread
    block sharing.

    Size classes: 64, 128, 256, 512, 1024, 2048 bytes.
    Allocations larger than 2048 bytes bypass the pools entirely.

    This is the default allocator used by run_async when no allocator
    is specified.

    @par Thread Safety
    Thread-safe. The thread-local pool requires no synchronization.
    The global pool uses a mutex for cross-thread access.

    @par Example
    @par !example example


    @see @ref get_recycling_memory_resource
    @see @ref run_async
*/
BOOST_CAPY_MSVC_WARNING_PUSH
BOOST_CAPY_MSVC_WARNING_DISABLE(4275) // non dll-interface base class
class BOOST_CAPY_DECL recycling_memory_resource : public std::pmr::memory_resource
{
    static constexpr std::size_t num_classes = 6;
    static constexpr std::size_t min_class_size = 64;   // 2^6
    static constexpr std::size_t max_class_size = 2048; // 2^11
    static constexpr std::size_t bucket_capacity = 16;

    // Size class of an n-byte request: 0 for up to 64 bytes, 1 for up
    // to 128, ..., 5 for up to 2048, and num_classes or more beyond (n
    // == 0 too, on allocate and deallocate alike). Four operations.
    static std::size_t
    class_index(std::size_t n) noexcept
    {
        return static_cast<std::size_t>(
            std::bit_width((n - 1) | (min_class_size - 1))) - 6;
    }

    static std::size_t
    class_size(std::size_t idx) noexcept
    {
        return min_class_size << idx;
    }

    struct bucket
    {
        std::size_t count = 0;
        void* ptrs[bucket_capacity] = {};

        void* pop() noexcept
        {
            if(count == 0)
                return nullptr;
            return ptrs[--count];
        }

        // Peter Dimov's idea
        void* pop(bucket& b) noexcept
        {
            if(count == 0)
                return nullptr;
            for(std::size_t i = 0; i < count; ++i)
                b.ptrs[i] = ptrs[i];
            b.count = count - 1;
            count = 0;
            return b.ptrs[b.count];
        }

        bool push(void* p) noexcept
        {
            if(count >= bucket_capacity)
                return false;
            ptrs[count++] = p;
            return true;
        }
    };

    struct pool
    {
        // Blocks a bucket may cache: 0 until this thread arms its exit
        // drain, bucket_capacity once armed, and 0 again after the drain
        // ran. A free that finds its bucket at the limit takes the slow
        // path, which arms, so the hot path needs no separate flag.
        std::size_t limit = 0;
        bucket buckets[num_classes];

        // No destructor: a non-trivial dtor forces a guard variable on the
        // thread_local in local(), checked on every alloc/free. Constant
        // initialization plus a trivial dtor makes that access a bare TLS
        // load. Cached blocks are instead reclaimed explicitly: per-thread
        // by arm_thread_cleanup() at thread exit, and the global pool by
        // global()'s holder destructor at process exit.
    };

#if (defined(BOOST_CAPY_DYN_LINK) || defined(BOOST_ALL_DYN_LINK)) && \
    !defined(BOOST_CAPY_STATIC_LINK)
    // A shared build defines this thread's pool in
    // the library: inline, each module would cache in a pool of its own,
    // which the library's thread-exit cleanup never drains.
    static pool& local() noexcept;
#else
    static pool& local() noexcept
    {
        static thread_local pool p;
        return p;
    }
#endif

    static pool& global() noexcept;
    static std::mutex& global_mutex() noexcept;

    void* allocate_slow(std::size_t idx);
    void deallocate_slow(void* p, std::size_t idx);

    // Register a thread-exit callback that drains this thread's local
    // pool back to the OS, and open the pool (limit = bucket_capacity).
    // Called only from the slow paths, while the limit is 0. Returns
    // false once that callback has run: nothing drains the local pool
    // after it, so frees must bypass it.
    static bool arm_thread_cleanup() noexcept;

public:
    /** Destroy the resource.

        No cached block is released here. Every pool is static, so an
        instance holds no state of its own. The thread-local pool is
        drained at thread exit, and the global pool at process exit.
    */
    ~recycling_memory_resource();

    /** Allocate without virtual dispatch.

        Handles the fast path inline (thread-local bucket pop)
        and falls through to the slow path for global pool or
        heap allocation.

        A request larger than the largest size class (2048 bytes)
        bypasses the pools and goes straight to `::operator new`.

        The second parameter is the requested alignment, and it is ignored.
        Every block comes from `::operator new`, so blocks carry the
        implementation's default new alignment and no more.

        @param bytes The number of bytes to allocate.

        @return A pointer to a block of at least `bytes` bytes. A pooled
        block is rounded up to its size class, so it may be larger than
        requested.

        @throws std::bad_alloc If the underlying `::operator new` fails.
    */
    void*
    allocate_fast(std::size_t bytes, std::size_t)
    {
        std::size_t idx = class_index(bytes);
        if(idx >= num_classes)
            return ::operator new(bytes);
        auto& b = local().buckets[idx];
        if(b.count != 0)
            return b.ptrs[--b.count];
        return allocate_slow(idx);
    }

    /** Deallocate without virtual dispatch.

        Handles the fast path inline (thread-local bucket push)
        and falls through to the slow path for global pool or
        heap deallocation.

        The block is cached in the pool of the thread that frees it, not
        the thread that allocated it.

        The third parameter is the alignment the block was allocated with,
        and it is ignored, as it is on allocation.

        @param p The block to return. It must have come from
        @ref allocate_fast or @ref do_allocate on this resource.

        @param bytes The size the block was allocated with. The size class
        is recomputed from it, so passing a different value puts the block
        in the wrong bucket.
    */
    void
    deallocate_fast(void* p, std::size_t bytes, std::size_t)
    {
        std::size_t idx = class_index(bytes);
        if(idx >= num_classes)
        {
            ::operator delete(p);
            return;
        }
        // A pool not yet armed has limit 0, so its first free takes the
        // slow path, which arms thread-exit cleanup (including consumer
        // threads that never allocate); after that cleanup the limit is
        // 0 again and the block bypasses the local pool.
        auto& lp = local();
        auto& b = lp.buckets[idx];
        if(b.count < lp.limit)
        {
            b.ptrs[b.count++] = p;
            return;
        }
        deallocate_slow(p, idx);
    }

protected:
    /** Allocate through the `std::pmr::memory_resource` interface.

        Forwards to @ref allocate_fast, so it has that function's contract.
        Call `allocate_fast` directly to skip the virtual dispatch.

        @param bytes The number of bytes to allocate.

        @param alignment The requested alignment. It is ignored.

        @return A pointer to a block of at least `bytes` bytes.

        @throws std::bad_alloc If the underlying `::operator new` fails.
    */
    void*
    do_allocate(std::size_t bytes, std::size_t alignment) override;

    /** Deallocate through the `std::pmr::memory_resource` interface.

        Forwards to @ref deallocate_fast, so it has that function's
        contract.

        @param p The block to return, as obtained from this resource.

        @param bytes The size the block was allocated with.

        @param alignment The alignment the block was allocated with. It is
        ignored.
    */
    void
    do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;

    /** Compare this resource with another for equality.

        Equality is object identity: two distinct
        `recycling_memory_resource` objects compare unequal, even though the
        pools they draw from are static and therefore shared.

        @param other The resource to compare against.

        @return `true` if `other` is the same object as `*this`; otherwise
        `false`.
    */
    bool
    do_is_equal(const memory_resource& other) const noexcept override
    {
        return this == &other;
    }
};
BOOST_CAPY_MSVC_WARNING_POP

namespace detail {

// Storage for the process-wide instance. Constant-initialized and never
// destroyed: its address is a link-time constant, so comparing against it
// needs no static-init guard, and it outlives every static that frees a
// frame. The resource is stateless, so skipping its destructor leaks
// nothing.
union recycling_storage
{
    recycling_memory_resource mr;

    constexpr recycling_storage() noexcept : mr() {}
    ~recycling_storage() {}
};

BOOST_CAPY_DECL extern recycling_storage recycling_instance;

} // namespace detail

/** Returns pointer to the default recycling memory resource.

    The returned pointer is valid for the lifetime of the program.
    This is the default allocator used by run_async.

    @return Pointer to the recycling memory resource.

    @see @ref recycling_memory_resource
    @see @ref run_async
*/
inline
std::pmr::memory_resource*
get_recycling_memory_resource() noexcept
{
    return &detail::recycling_instance.mr;
}

} // namespace capy
} // namespace boost

#endif
