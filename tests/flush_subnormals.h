#pragma once
/*
 * flush_subnormals.h -- the arithmetic of a process that flushes subnormal
 * numbers to zero, for the tests of what only such a process meets.
 *
 * Compiling with -ffast-math does not flush subnormal numbers: linking an
 * executable with it does (GCC and Clang link crtfastmath.o, whose start-up
 * code sets the flush-to-zero and denormals-are-zero bits of the
 * floating-point control register). A test sets those bits itself, while a
 * FlushSubnormals lives, and so reaches that arithmetic whatever the flags
 * of the test and of the code it calls. The register is per thread: the
 * code under test runs on the thread that holds the object.
 *
 * active() is false where this header knows no such control (neither x86
 * with SSE nor AArch64); the arithmetic is then left as it was.
 */

#include <cstdint>

#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#define TADAH_TEST_FTZ_X86 1
#elif defined(__aarch64__)
#define TADAH_TEST_FTZ_AARCH64 1
#endif

namespace tadah::hpo::test {

class FlushSubnormals {
  public:
    FlushSubnormals() {
#if defined(TADAH_TEST_FTZ_X86)
        saved_ = _mm_getcsr();
        /* FTZ (bit 15) flushes subnormal results, DAZ (bit 6) reads a
         * subnormal operand as zero. */
        _mm_setcsr(saved_ | 0x8040u);
        active_ = true;
#elif defined(TADAH_TEST_FTZ_AARCH64)
        saved_ = __builtin_aarch64_get_fpcr64();
        /* FZ (bit 24) flushes subnormal inputs and results. */
        __builtin_aarch64_set_fpcr64(saved_ | (std::uint64_t{1} << 24));
        active_ = true;
#endif
    }
    ~FlushSubnormals() {
#if defined(TADAH_TEST_FTZ_X86)
        _mm_setcsr(saved_);
#elif defined(TADAH_TEST_FTZ_AARCH64)
        __builtin_aarch64_set_fpcr64(saved_);
#endif
    }
    FlushSubnormals(const FlushSubnormals&) = delete;
    FlushSubnormals& operator=(const FlushSubnormals&) = delete;

    /* The bits are set: subnormal numbers are flushed to zero. */
    bool active() const { return active_; }

  private:
#if defined(TADAH_TEST_FTZ_X86)
    unsigned int saved_ = 0;
#elif defined(TADAH_TEST_FTZ_AARCH64)
    std::uint64_t saved_ = 0;
#endif
    bool active_ = false;
};

/* Is a subnormal result flushed to zero here? Read from the bits of the
 * result of a product computed at run time: not inlined, and behind a
 * compiler barrier, so that no call is merged with another or moved across
 * a change of the control register (the compiler does not know that the
 * register changes the result). */
__attribute__((noinline)) inline bool subnormals_are_flushed() {
    asm volatile("" ::: "memory");
    volatile double tiny = 1e-300;
    volatile double scale = 1e-10;
    volatile double r = tiny * scale;   /* 1e-310: subnormal in IEEE */
    asm volatile("" ::: "memory");
    const double v = r;
    std::uint64_t bits = 0;
    static_assert(sizeof bits == sizeof v, "double is 64 bits");
    __builtin_memcpy(&bits, &v, sizeof bits);
    return (bits & ~(std::uint64_t{1} << 63)) == 0;
}

}  // namespace tadah::hpo::test
