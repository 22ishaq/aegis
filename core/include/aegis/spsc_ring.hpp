// Lock-free single-producer / single-consumer ring buffer.
//
// One thread pushes, one thread pops, and they never take a lock. Correctness
// rests on two ideas:
//
//  1. Ownership. Only the producer ever writes `head_`; only the consumer ever
//     writes `tail_`. Each index has exactly one writer, so no compare-and-swap
//     is needed - plain atomic loads and stores are enough.
//
//  2. Acquire/release ordering. The producer writes the slot and *then*
//     publishes it with a release-store of `head_`. The consumer reads `head_`
//     with an acquire-load, which guarantees it also sees the slot contents
//     written before that store. The same handshake runs in reverse via
//     `tail_` so the producer never overwrites a slot that is still being read.
//
// Performance details:
//  - Indices grow forever (uint64 will not wrap in practice) and are mapped to a
//    slot with `index & mask`, so capacity must be a power of two. That avoids a
//    division on every operation and makes "full" vs "empty" unambiguous.
//  - Producer and consumer state sit on separate cache lines. Without this,
//    the two cores would keep stealing the same line from each other ("false
//    sharing"), which can cost more than the work itself.
//  - Each side keeps a private cached copy of the other side's index and only
//    re-reads the shared atomic when the cached value says full/empty. Most
//    operations therefore touch no shared cache line at all.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace aegis {

// 128 bytes: Apple Silicon uses 128-byte lines, and on x86 the adjacent-line
// prefetcher pulls 64-byte lines in pairs, so 128 is the safe padding on both.
inline constexpr std::size_t kCacheLine = 128;

template <typename T>
class SpscRing {
    static_assert(std::is_trivially_copyable_v<T>, "ring slots are copied with plain assignment");

public:
    explicit SpscRing(std::size_t capacity)
        : capacity_(capacity), mask_(capacity - 1), slots_(std::make_unique<T[]>(capacity)) {
        if (capacity < 2 || (capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("SpscRing capacity must be a power of two >= 2");
        }
    }

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    // Producer thread only. Returns false if the ring is full.
    bool try_push(const T& value) noexcept {
        const uint64_t head = producer_.head.load(std::memory_order_relaxed);
        if (head - producer_.cached_tail == capacity_) {
            producer_.cached_tail = consumer_.tail.load(std::memory_order_acquire);
            if (head - producer_.cached_tail == capacity_) return false;
        }
        slots_[head & mask_] = value;
        producer_.head.store(head + 1, std::memory_order_release);  // publish the slot
        return true;
    }

    // Consumer thread only. Returns false if the ring is empty.
    bool try_pop(T& out) noexcept {
        const uint64_t tail = consumer_.tail.load(std::memory_order_relaxed);
        if (tail == consumer_.cached_head) {
            consumer_.cached_head = producer_.head.load(std::memory_order_acquire);
            if (tail == consumer_.cached_head) return false;
        }
        out = slots_[tail & mask_];
        consumer_.tail.store(tail + 1, std::memory_order_release);  // hand the slot back
        return true;
    }

    // Approximate when called while the other thread is active.
    std::size_t size_approx() const noexcept {
        const uint64_t head = producer_.head.load(std::memory_order_acquire);
        const uint64_t tail = consumer_.tail.load(std::memory_order_acquire);
        return static_cast<std::size_t>(head - tail);
    }

    std::size_t capacity() const noexcept { return capacity_; }

private:
    struct alignas(kCacheLine) ProducerSide {
        std::atomic<uint64_t> head{0};
        uint64_t cached_tail = 0;
    };
    struct alignas(kCacheLine) ConsumerSide {
        std::atomic<uint64_t> tail{0};
        uint64_t cached_head = 0;
    };

    const std::size_t capacity_;
    const std::size_t mask_;
    std::unique_ptr<T[]> slots_;
    ProducerSide producer_;
    ConsumerSide consumer_;
};

}  // namespace aegis
