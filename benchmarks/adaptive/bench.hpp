#pragma once

#include <adaptive/adaptive.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/times.h>
#include <unistd.h>
#include <vector>

#include "rand.hpp"

#ifndef BENCH_SEED
#define BENCH_SEED 36
#endif

namespace adaptive::bench {

enum class bench_op { access, rank, select };

inline void print_usage_and_exit(const char* program_name) {
    std::fprintf(stderr,
                 "Usage: %s <log_2 n> <alpha> <1/q> <theta>\n"
                 "Creates a bitvector of length n and applies alpha*n operations on it.\n"
                 "A fraction 1/q of the operations are indels (split evenly between\n"
                 "insertions and deletions); the remaining operations target the selected\n"
                 "query. Theta controls the flattening threshold and is passed directly\n"
                 "as a fraction such as 0.1, 0.01, or 0.001.\n",
                 program_name);
    std::exit(1);
}

inline void run(int argc, char** argv, bench_op op) {
    if (argc < 5) {
        print_usage_and_exit(argv[0]);
    }

    pcg32_rng rng(BENCH_SEED);

    const uint64_t initial_n = uint64_t{1} << std::atoi(argv[1]);
    uint64_t n = initial_n;
    const uint64_t m = initial_n * static_cast<uint64_t>(std::atoi(argv[2]));
    const float update_probability = std::atof(argv[3]);
    const uint64_t theta = std::strchr(argv[4], '.') ? static_cast<uint64_t>(1.0 / std::strtod(argv[4], nullptr) + 0.5)
                                                     : std::strtoull(argv[4], nullptr, 10);

    std::vector<uint64_t> data((n + 63) / 64);
    for (uint64_t i = 0; i < n / 64; ++i) {
        data[i] = rng.next() | (static_cast<uint64_t>(rng.next()) << 32);
    }

    struct tms start_cpu = {};
    struct tms end_cpu = {};
    times(&start_cpu);

    volatile uint64_t checksum = 0;
    bit_vector<> bv(data.data(), n);
    bv.set_theta(theta);

    for (uint64_t i = 0; i < m; ++i) {
        if (rng.probability(update_probability)) {
            if (rng.index(2) == 0) {
                bv.insert(rng.index(static_cast<uint32_t>(++n)), rng.index(2) != 0);
            } else {
                bv.erase(rng.index(static_cast<uint32_t>(n--)));
            }
        } else {
            switch (op) {
            case bench_op::access:
                checksum += bv.access(rng.index(static_cast<uint32_t>(n)));
                break;
            case bench_op::rank:
                checksum += bv.rank1(rng.index(static_cast<uint32_t>(n)) + 1);
                break;
            case bench_op::select:
                checksum += bv.select1(1 + rng.index(static_cast<uint32_t>(bv.ones())));
                break;
            }
        }
    }

    times(&end_cpu);

    const double elapsed_ticks = static_cast<double>(end_cpu.tms_utime - start_cpu.tms_utime);
    const double ticks_per_second = static_cast<double>(sysconf(_SC_CLK_TCK));
    const double microseconds_per_operation = (elapsed_ticks / ticks_per_second) * 1000000.0 / static_cast<double>(m);

    const double total_space_bits = 8.0 * static_cast<double>(bv.space_bytes());
    const double current_length_bits = static_cast<double>(n);

    std::printf("%f,%f,%lu\n", microseconds_per_operation, total_space_bits / current_length_bits, checksum);
}

} // namespace adaptive::bench
