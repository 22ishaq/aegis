// A deliberately tiny test harness (~40 lines) so the core keeps zero
// third-party dependencies. Each TEST registers itself in a global list
// before main() runs, using a static object's constructor.
#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

namespace check {

struct Case {
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

inline int& failures() {
    static int f = 0;
    return f;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

}  // namespace check

#define TEST(name)                                             \
    static void name();                                        \
    static ::check::Registrar name##_registrar(#name, name);   \
    static void name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            ++::check::failures();                                                    \
            std::fprintf(stderr, "    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                             \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                          \
    do {                                                                                               \
        const double check_a_ = static_cast<double>(a), check_b_ = static_cast<double>(b);             \
        if (!(std::fabs(check_a_ - check_b_) <= (tol))) {                                              \
            ++::check::failures();                                                                     \
            std::fprintf(stderr, "    FAIL %s:%d: %s = %.9g, expected %.9g (tol %g)\n", __FILE__,      \
                         __LINE__, #a, check_a_, check_b_, static_cast<double>(tol));                  \
        }                                                                                              \
    } while (0)

#define CHECK_THROWS(expr)                                                                       \
    do {                                                                                         \
        bool check_threw_ = false;                                                               \
        try {                                                                                    \
            expr;                                                                                \
        } catch (...) {                                                                          \
            check_threw_ = true;                                                                 \
        }                                                                                        \
        if (!check_threw_) {                                                                     \
            ++::check::failures();                                                               \
            std::fprintf(stderr, "    FAIL %s:%d: expected exception from %s\n", __FILE__,       \
                         __LINE__, #expr);                                                       \
        }                                                                                        \
    } while (0)
