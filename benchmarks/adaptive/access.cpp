#include "bench.hpp"

int main(int argc, char** argv) {
    adaptive::bench::run(argc, argv, adaptive::bench::bench_op::access);
    return 0;
}
