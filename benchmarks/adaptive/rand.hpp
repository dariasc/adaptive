#pragma once

#include <cstdint>
#include <limits>

namespace adaptive::bench {

class pcg32_rng {
  public:
    explicit pcg32_rng(uint64_t seed) {
        init(seed);
    }

    void init(uint64_t seed) {
        state_ = seed + increment;
        (void)next();
    }

    uint32_t next() {
        uint64_t x = state_;
        const unsigned count = static_cast<unsigned>(x >> 59);

        state_ = x * multiplier + increment;
        x ^= x >> 18;
        return rotr32(static_cast<uint32_t>(x >> 27), count);
    }

    uint32_t index(uint32_t limit) {
        return next() % limit;
    }

    bool probability(float p) {
        return static_cast<float>(next()) < static_cast<float>(std::numeric_limits<uint32_t>::max()) * p;
    }

  private:
    static constexpr uint64_t multiplier = 6364136223846793005ULL;
    static constexpr uint64_t increment = 1442695040888963407ULL;

    uint64_t state_ = 0x4d595df4d0f33173ULL;

    static uint32_t rotr32(uint32_t x, unsigned r) {
        return x >> r | x << (-r & 31);
    }
};

} // namespace adaptive::bench
