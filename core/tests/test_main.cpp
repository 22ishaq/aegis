#include <cstring>
#include <exception>

#include "check.hpp"

// Usage: aegis_tests [substring]   - runs every test whose name contains it.
int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int ran = 0;
    for (const auto& c : check::registry()) {
        if (filter && !std::strstr(c.name, filter)) continue;
        const int before = check::failures();
        std::printf("%-48s", c.name);
        try {
            c.fn();
        } catch (const std::exception& e) {
            ++check::failures();
            std::fprintf(stderr, "    FAIL unexpected exception: %s\n", e.what());
        }
        std::printf("%s\n", check::failures() == before ? "ok" : "FAILED");
        ++ran;
    }
    std::printf("\n%d tests, %d failed checks\n", ran, check::failures());
    return check::failures() == 0 ? 0 : 1;
}
