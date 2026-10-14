// Open-addressing hash map from uint64 keys to values, used for per-account state.
//
// Why not std::unordered_map? It allocates one heap node per entry and chains
// collisions through linked lists, so every lookup is a pointer chase to a
// random address (a likely cache miss). Here every entry lives in one
// contiguous array: a lookup hashes the key, jumps to a slot, and on collision
// steps to the *next* slot ("linear probing"), which is usually already in the
// same or the adjacent cache line.
//
// Key 0 marks an empty slot, so 0 is not a valid key (enforced by
// aegis::validate). References returned by get_or_insert are invalidated by a
// later insertion that triggers growth - callers use them immediately.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aegis {

// Final mixing step of SplitMix64: a fast bijection that spreads sequential
// account ids across the whole table so they do not cluster.
inline constexpr uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

template <typename V>
class FlatMap {
public:
    explicit FlatMap(std::size_t expected = 1024) {
        std::size_t cap = 16;
        while (cap * 7 < expected * 10) cap <<= 1;  // keep load factor below 0.7
        slots_.assign(cap, Slot{});
        mask_ = cap - 1;
    }

    // Returns the value for `key`, default-constructing it if absent.
    V& get_or_insert(uint64_t key, bool& inserted) {
        if (key == 0) throw std::invalid_argument("FlatMap key 0 is reserved");
        if ((size_ + 1) * 10 > slots_.size() * 7) grow();
        std::size_t i = mix64(key) & mask_;
        while (true) {
            Slot& s = slots_[i];
            if (s.key == key) {
                inserted = false;
                return s.value;
            }
            if (s.key == 0) {
                s.key = key;
                s.value = V{};
                ++size_;
                inserted = true;
                return s.value;
            }
            i = (i + 1) & mask_;
        }
    }

    const V* find(uint64_t key) const noexcept {
        if (key == 0) return nullptr;
        std::size_t i = mix64(key) & mask_;
        while (true) {
            const Slot& s = slots_[i];
            if (s.key == key) return &s.value;
            if (s.key == 0) return nullptr;
            i = (i + 1) & mask_;
        }
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return slots_.size(); }

    void clear() {
        for (auto& s : slots_) s = Slot{};
        size_ = 0;
    }

private:
    struct Slot {
        uint64_t key = 0;
        V value{};
    };

    void grow() {
        std::vector<Slot> old;
        old.swap(slots_);
        slots_.assign(old.size() * 2, Slot{});
        mask_ = slots_.size() - 1;
        for (const Slot& s : old) {
            if (s.key == 0) continue;
            std::size_t i = mix64(s.key) & mask_;
            while (slots_[i].key != 0) i = (i + 1) & mask_;
            slots_[i] = s;
        }
    }

    std::vector<Slot> slots_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
};

}  // namespace aegis
