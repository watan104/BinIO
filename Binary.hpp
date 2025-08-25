/*
MIT License

Copyright (c) 2025 Andre Watan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <type_traits>
#include <memory>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define BINARY_IO_X86

#if defined(__AVX2__)
#define BINARY_IO_AVX2_AVAILABLE
#include <immintrin.h>
#elif defined(__SSE2__)
#define BINARY_IO_SSE2_AVAILABLE
#include <emmintrin.h>
#endif

#if defined(__SSE4_1__)
#define BINARY_IO_SSE41_AVAILABLE
#include <smmintrin.h>
#endif

#include <xmmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_MSC_VER) && defined(BINARY_IO_X86)
#ifndef BINARY_IO_MSVC_SHIMS_DEFINED
#define BINARY_IO_MSVC_SHIMS_DEFINED

static __forceinline __m128i _binaryio_loadu_si16(const void* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return _mm_cvtsi32_si128((int)v);
}
static __forceinline void _binaryio_storeu_si16(void* p, __m128i a) {
    uint16_t v = (uint16_t)_mm_cvtsi128_si32(a);
    std::memcpy(p, &v, 2);
}
static __forceinline __m128i _binaryio_loadu_si32(const void* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return _mm_cvtsi32_si128((int)v);
}
static __forceinline void _binaryio_storeu_si32(void* p, __m128i a) {
    uint32_t v = (uint32_t)_mm_cvtsi128_si32(a);
    std::memcpy(p, &v, 4);
}
static __forceinline __m128i _binaryio_loadu_si64(const void* p) {
#if defined(_M_X64)
    long long v;
    std::memcpy(&v, p, 8);
    return _mm_cvtsi64_si128(v);
#else
    __m128i v = _mm_setzero_si128();
    uint64_t tmp;
    std::memcpy(&tmp, p, 8);
    return _mm_unpacklo_epi64(_mm_cvtsi32_si128((int)(tmp & 0xFFFFFFFFULL)),
        _mm_cvtsi32_si128((int)((tmp >> 32) & 0xFFFFFFFFULL)));
#endif
}
static __forceinline void _binaryio_storeu_si64(void* p, __m128i a) {
#if defined(_M_X64)
    long long v = _mm_cvtsi128_si64(a);
    std::memcpy(p, &v, 8);
#else
    uint64_t lo = (uint32_t)_mm_cvtsi128_si32(a);
    __m128i hiShift = _mm_srli_si128(a, 4);
    uint64_t hi = (uint32_t)_mm_cvtsi128_si32(hiShift);
    uint64_t v = lo | (hi << 32);
    std::memcpy(p, &v, 8);
#endif
}
#ifndef _mm_loadu_si16
#define _mm_loadu_si16(p)  _binaryio_loadu_si16((p))
#endif
#ifndef _mm_storeu_si16
#define _mm_storeu_si16(p,a)  _binaryio_storeu_si16((p),(a))
#endif
#ifndef _mm_loadu_si32
#define _mm_loadu_si32(p)  _binaryio_loadu_si32((p))
#endif
#ifndef _mm_storeu_si32
#define _mm_storeu_si32(p,a)  _binaryio_storeu_si32((p),(a))
#endif
#ifndef _mm_loadu_si64
#define _mm_loadu_si64(p)  _binaryio_loadu_si64((p))
#endif
#ifndef _mm_storeu_si64
#define _mm_storeu_si64(p,a)  _binaryio_storeu_si64((p),(a))
#endif
#endif
#endif

#elif defined(__aarch64__) || defined(_M_ARM64)
#define BINARY_IO_ARM64
#if defined(__ARM_NEON) || defined(_MSC_VER)
#define BINARY_IO_NEON_AVAILABLE
#include <arm_neon.h>
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BINARY_IO_FORCE_INLINE __attribute__((always_inline)) inline
#define BINARY_IO_RESTRICT __restrict__
#define BINARY_IO_LIKELY(x) __builtin_expect(!!(x), 1)
#define BINARY_IO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define BINARY_IO_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)

#elif defined(_MSC_VER)
#define BINARY_IO_FORCE_INLINE __forceinline
#define BINARY_IO_RESTRICT __restrict
#define BINARY_IO_LIKELY(x) (x)
#define BINARY_IO_UNLIKELY(x) (x)
#define BINARY_IO_PREFETCH(addr, rw, locality) _mm_prefetch((const char*)(addr), _MM_HINT_T0)

#else
#define BINARY_IO_FORCE_INLINE inline
#define BINARY_IO_RESTRICT
#define BINARY_IO_LIKELY(x) (x)
#define BINARY_IO_UNLIKELY(x) (x)
#define BINARY_IO_PREFETCH(addr, rw, locality)
#endif

namespace BinIO {
    namespace BinaryIOConfig {
        constexpr std::size_t CACHE_LINE_SIZE = 64;
        constexpr std::size_t SIMD_THRESHOLD = 32;
        constexpr std::size_t PREFETCH_DISTANCE = 128;

        constexpr bool is_little_endian() noexcept {
            constexpr uint32_t test = 0x01020304;
            return static_cast<const uint8_t*>(static_cast<const void*>(&test))[0] == 0x04;
        }
    }

    namespace BinaryIOSIMD {

        BINARY_IO_FORCE_INLINE void zero_memory(uint8_t* ptr, std::size_t size) noexcept {
#if defined(BINARY_IO_AVX2_AVAILABLE)
            const __m256i zero = _mm256_setzero_si256();
            std::size_t simd_size = size & ~31;

            for (std::size_t i = 0; i < simd_size; i += 32) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i), zero);
            }

            if (std::size_t remaining = size - simd_size; remaining > 0) {
                std::memset(ptr + simd_size, 0, remaining);
            }
#elif defined(BINARY_IO_SSE2_AVAILABLE)
            const __m128i zero = _mm_setzero_si128();
            std::size_t simd_size = size & ~15;

            for (std::size_t i = 0; i < simd_size; i += 16) {
                _mm_store_si128(reinterpret_cast<__m128i*>(ptr + i), zero);
            }

            if (std::size_t remaining = size - simd_size; remaining > 0) {
                std::memset(ptr + simd_size, 0, remaining);
            }
#elif defined(BINARY_IO_NEON_AVAILABLE)
            uint8x16_t zero = vdupq_n_u8(0);
            std::size_t simd_size = size & ~15;

            for (std::size_t i = 0; i < simd_size; i += 16) {
                vst1q_u8(ptr + i, zero);
            }

            if (std::size_t remaining = size - simd_size; remaining > 0) {
                std::memset(ptr + simd_size, 0, remaining);
            }
#else
            std::memset(ptr, 0, size);
#endif
        }

        BINARY_IO_FORCE_INLINE void copy_memory(uint8_t* BINARY_IO_RESTRICT dst,
            const uint8_t* BINARY_IO_RESTRICT src,
            std::size_t size) noexcept {
#if defined(BINARY_IO_AVX2_AVAILABLE)
            std::size_t simd_size = size & ~31;

            for (std::size_t i = 0; i < simd_size; i += 32) {
                __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), data);
            }

            if (std::size_t remaining = size - simd_size; remaining > 0) {
                std::memcpy(dst + simd_size, src + simd_size, remaining);
            }
#elif defined(BINARY_IO_SSE2_AVAILABLE)
            std::size_t simd_size = size & ~15;

            for (std::size_t i = 0; i < simd_size; i += 16) {
                __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), data);
            }

            if (std::size_t remaining = size - simd_size; remaining > 0) {
                std::memcpy(dst + simd_size, src + simd_size, remaining);
            }
#elif defined(BINARY_IO_NEON_AVAILABLE)
            std::size_t simd_size = size & ~15;

            for (std::size_t i = 0; i < simd_size; i += 16) {
                uint8x16_t data = vld1q_u8(src + i);
                vst1q_u8(dst + i, data);
            }

            if (std::size_t remaining = size - simd_size; remaining > 0) {
                std::memcpy(dst + simd_size, src + simd_size, remaining);
            }
#else
            std::memcpy(dst, src, size);
#endif
        }

        template<typename T>
        BINARY_IO_FORCE_INLINE constexpr T byte_swap(T val) noexcept {
            static_assert(std::is_integral_v<T>, "Type must be integral");

#if defined(__GNUC__) || defined(__clang__)
            if constexpr (sizeof(T) == 2) return __builtin_bswap16(static_cast<uint16_t>(val));
            else if constexpr (sizeof(T) == 4) return __builtin_bswap32(static_cast<uint32_t>(val));
            else if constexpr (sizeof(T) == 8) return __builtin_bswap64(static_cast<uint64_t>(val));
            else return val;
#elif defined(_MSC_VER)
            if constexpr (sizeof(T) == 2) return _byteswap_ushort(static_cast<uint16_t>(val));
            else if constexpr (sizeof(T) == 4) return _byteswap_ulong(static_cast<uint32_t>(val));
            else if constexpr (sizeof(T) == 8) return _byteswap_uint64(static_cast<uint64_t>(val));
            else return val;
#else
            if constexpr (sizeof(T) == 2) {
                uint16_t v = static_cast<uint16_t>(val);
                return static_cast<T>((v << 8) | (v >> 8));
            }
            else if constexpr (sizeof(T) == 4) {
                uint32_t v = static_cast<uint32_t>(val);
                return static_cast<T>(((v & 0xFF000000) >> 24) |
                    ((v & 0x00FF0000) >> 8) |
                    ((v & 0x0000FF00) << 8) |
                    ((v & 0x000000FF) << 24));
            }
            else if constexpr (sizeof(T) == 8) {
                uint64_t v = static_cast<uint64_t>(val);
                return static_cast<T>(
                    ((v & 0xFF00000000000000ULL) >> 56) |
                    ((v & 0x00FF000000000000ULL) >> 40) |
                    ((v & 0x0000FF0000000000ULL) >> 24) |
                    ((v & 0x000000FF00000000ULL) >> 8) |
                    ((v & 0x00000000FF000000ULL) << 8) |
                    ((v & 0x0000000000FF0000ULL) << 24) |
                    ((v & 0x000000000000FF00ULL) << 40) |
                    ((v & 0x00000000000000FFULL) << 56));
            }
            else {
                return val;
            }
#endif
        }

        template<typename T>
        BINARY_IO_FORCE_INLINE T load_unaligned(const uint8_t* ptr) noexcept {
            static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");

#if defined(BINARY_IO_X86)
            if constexpr (sizeof(T) == 1) {
                return static_cast<T>(*ptr);
            }
            else if constexpr (sizeof(T) == 2) {
                __m128i xmm = _mm_loadu_si16(ptr);
                return static_cast<T>(_mm_cvtsi128_si32(xmm));
            }
            else if constexpr (sizeof(T) == 4) {
                if constexpr (std::is_floating_point_v<T>) {
                    return _mm_cvtss_f32(_mm_load_ss(reinterpret_cast<const float*>(ptr)));
                }
                else {
                    __m128i xmm = _mm_loadu_si32(ptr);
                    return static_cast<T>(_mm_cvtsi128_si32(xmm));
                }
            }
            else if constexpr (sizeof(T) == 8) {
                if constexpr (std::is_floating_point_v<T>) {
                    return _mm_cvtsd_f64(_mm_load_sd(reinterpret_cast<const double*>(ptr)));
                }
                else {
                    __m128i xmm = _mm_loadu_si64(ptr);
#if defined(_M_X64) || defined(__x86_64__)
                    return static_cast<T>(_mm_cvtsi128_si64(xmm));
#else
                    uint64_t lo = (uint32_t)_mm_cvtsi128_si32(xmm);
                    __m128i hiShift = _mm_srli_si128(xmm, 4);
                    uint64_t hi = (uint32_t)_mm_cvtsi128_si32(hiShift);
                    return static_cast<T>(lo | (hi << 32));
#endif
                }
            }
            else {
                T val;
                std::memcpy(&val, ptr, sizeof(T));
                return val;
            }
#else
            T val;
            std::memcpy(&val, ptr, sizeof(T));
            return val;
#endif
        }

        template<typename T>
        BINARY_IO_FORCE_INLINE void store_unaligned(uint8_t* ptr, T val) noexcept {
            static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");

#if defined(BINARY_IO_X86)
            if constexpr (sizeof(T) == 1) {
                *ptr = static_cast<uint8_t>(val);
            }
            else if constexpr (sizeof(T) == 2) {
                __m128i xmm = _mm_cvtsi32_si128(static_cast<uint16_t>(val));
                _mm_storeu_si16(ptr, xmm);
            }
            else if constexpr (sizeof(T) == 4) {
                if constexpr (std::is_floating_point_v<T>) {
                    _mm_store_ss(reinterpret_cast<float*>(ptr), _mm_set_ss(static_cast<float>(val)));
                }
                else {
                    __m128i xmm = _mm_cvtsi32_si128(static_cast<uint32_t>(val));
                    _mm_storeu_si32(ptr, xmm);
                }
            }
            else if constexpr (sizeof(T) == 8) {
                if constexpr (std::is_floating_point_v<T>) {
                    _mm_store_sd(reinterpret_cast<double*>(ptr), _mm_set_sd(static_cast<double>(val)));
                }
                else {
                    __m128i xmm;
#if defined(_M_X64) || defined(__x86_64__)
                    xmm = _mm_cvtsi64_si128(static_cast<long long>(val));
#else
                    uint64_t u = static_cast<uint64_t>(val);
                    __m128i lo = _mm_cvtsi32_si128((int)(u & 0xFFFFFFFFULL));
                    __m128i hi = _mm_slli_si128(_mm_cvtsi32_si128((int)((u >> 32) & 0xFFFFFFFFULL)), 4);
                    xmm = _mm_or_si128(lo, hi);
#endif
                    _mm_storeu_si64(ptr, xmm);
                }
            }
            else {
                std::memcpy(ptr, &val, sizeof(T));
            }
#else
            std::memcpy(ptr, &val, sizeof(T));
#endif
        }
    }

    namespace BinaryIOMemory {
        BINARY_IO_FORCE_INLINE void* aligned_alloc(std::size_t alignment, std::size_t size) noexcept {
#if defined(_MSC_VER)
            return _aligned_malloc(size, alignment);
#elif defined(__MINGW32__) || defined(__MINGW64__)
            return __mingw_aligned_malloc(size, alignment);
#else
            return std::aligned_alloc(alignment, size);
#endif
        }

        BINARY_IO_FORCE_INLINE void aligned_free(void* ptr) noexcept {
            if (!ptr) return;

#if defined(_MSC_VER)
            _aligned_free(ptr);
#elif defined(__MINGW32__) || defined(__MINGW64__)
            __mingw_aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }

    class alignas(BinaryIOConfig::CACHE_LINE_SIZE) Writer {
    public:
        explicit Writer(uint8_t* BINARY_IO_RESTRICT data, std::size_t pos = 0) noexcept
            : m_data(data), m_pos(pos), m_size(0), m_owns_memory(false) {
        }

        explicit Writer(std::size_t size) noexcept
            : m_pos(0), m_size(size), m_owns_memory(true) {
            const std::size_t aligned_size = (size + BinaryIOConfig::CACHE_LINE_SIZE - 1) &
                ~(BinaryIOConfig::CACHE_LINE_SIZE - 1);

            m_data = static_cast<uint8_t*>(
                BinaryIOMemory::aligned_alloc(BinaryIOConfig::CACHE_LINE_SIZE, aligned_size));

            if (m_data) {
                BinaryIOSIMD::zero_memory(m_data, aligned_size);
            }
        }

        ~Writer() noexcept {
            if (m_owns_memory && m_data) {
                BinaryIOMemory::aligned_free(m_data);
            }
        }

        Writer(Writer&& other) noexcept
            : m_data(other.m_data), m_pos(other.m_pos),
            m_size(other.m_size), m_owns_memory(other.m_owns_memory) {
            other.m_data = nullptr;
            other.m_owns_memory = false;
        }

        Writer& operator=(Writer&& other) noexcept {
            if (this != &other) {
                if (m_owns_memory && m_data) {
                    BinaryIOMemory::aligned_free(m_data);
                }

                m_data = other.m_data;
                m_pos = other.m_pos;
                m_size = other.m_size;
                m_owns_memory = other.m_owns_memory;

                other.m_data = nullptr;
                other.m_owns_memory = false;
            }
            return *this;
        }

        Writer(const Writer&) = delete;
        Writer& operator=(const Writer&) = delete;

        template<typename T>
        BINARY_IO_FORCE_INLINE void write(T val) noexcept {
            static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");

            if (BINARY_IO_UNLIKELY((m_pos % BinaryIOConfig::CACHE_LINE_SIZE) ==
                (BinaryIOConfig::CACHE_LINE_SIZE - sizeof(T) - 8))) {
                BINARY_IO_PREFETCH(m_data + m_pos + BinaryIOConfig::PREFETCH_DISTANCE, 1, 3);
            }

            BinaryIOSIMD::store_unaligned(m_data + m_pos, val);
            m_pos += sizeof(T);
        }

        BINARY_IO_FORCE_INLINE void write_string(const std::string& val, uint16_t len_size = 2) noexcept {
            const auto len = val.length();

            if (BINARY_IO_LIKELY(len_size == 2)) {
                write(static_cast<uint16_t>(len));
            }
            else if (len_size == 4) {
                write(static_cast<uint32_t>(len));
            }
            else {
                std::memcpy(m_data + m_pos, &len, len_size);
                m_pos += len_size;
            }

            write_bytes(reinterpret_cast<const uint8_t*>(val.data()), len);
        }

        BINARY_IO_FORCE_INLINE void write_bytes(const uint8_t* BINARY_IO_RESTRICT data, std::size_t len) noexcept {
            if (BINARY_IO_UNLIKELY(len > BinaryIOConfig::PREFETCH_DISTANCE)) {
                BINARY_IO_PREFETCH(data, 0, 3);
                BINARY_IO_PREFETCH(m_data + m_pos + BinaryIOConfig::PREFETCH_DISTANCE, 1, 3);
            }

            if (BINARY_IO_UNLIKELY(len >= BinaryIOConfig::SIMD_THRESHOLD)) {
                BinaryIOSIMD::copy_memory(m_data + m_pos, data, len);
            }
            else {
                std::memcpy(m_data + m_pos, data, len);
            }

            m_pos += len;
        }

        BINARY_IO_FORCE_INLINE void write_bytes(const char* BINARY_IO_RESTRICT data, std::size_t len) noexcept {
            write_bytes(reinterpret_cast<const uint8_t*>(data), len);
        }

        BINARY_IO_FORCE_INLINE void set_pos(std::size_t pos) noexcept { m_pos = pos; }
        BINARY_IO_FORCE_INLINE void skip_pos(std::size_t len) noexcept { m_pos += len; }
        BINARY_IO_FORCE_INLINE void advance_pos(std::size_t len) noexcept { m_pos += len; }

        [[nodiscard]] BINARY_IO_FORCE_INLINE uint8_t* get() noexcept { return m_data; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE const uint8_t* get() const noexcept { return m_data; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE std::size_t get_pos() const noexcept { return m_pos; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE std::size_t get_size() const noexcept { return m_size; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE std::size_t remaining() const noexcept { return m_size - m_pos; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE bool owns_memory() const noexcept { return m_owns_memory; }

        BINARY_IO_FORCE_INLINE void reset() noexcept { m_pos = 0; }

        [[nodiscard]] BINARY_IO_FORCE_INLINE bool can_write(std::size_t bytes) const noexcept {
            return m_pos + bytes <= m_size;
        }

    private:
        uint8_t* BINARY_IO_RESTRICT m_data;
        std::size_t m_pos;
        std::size_t m_size;
        bool m_owns_memory;
    };

    class alignas(BinaryIOConfig::CACHE_LINE_SIZE) Reader {
    public:
        explicit Reader(const uint8_t* BINARY_IO_RESTRICT data, std::size_t size = 0) noexcept
            : m_data(data), m_pos(0), m_size(size), m_owns_memory(false) {
        }

        explicit Reader(const std::vector<uint8_t>& data) noexcept
            : m_pos(0), m_size(data.size()), m_owns_memory(true) {

            const std::size_t aligned_size = (m_size + BinaryIOConfig::CACHE_LINE_SIZE - 1) &
                ~(BinaryIOConfig::CACHE_LINE_SIZE - 1);

            m_data = static_cast<uint8_t*>(
                BinaryIOMemory::aligned_alloc(BinaryIOConfig::CACHE_LINE_SIZE, aligned_size));

            if (m_data && !data.empty()) {
                if (BINARY_IO_UNLIKELY(m_size >= BinaryIOConfig::SIMD_THRESHOLD)) {
                    BinaryIOSIMD::copy_memory(const_cast<uint8_t*>(m_data), data.data(), m_size);
                }
                else {
                    std::memcpy(const_cast<uint8_t*>(m_data), data.data(), m_size);
                }
            }
        }

        ~Reader() noexcept {
            if (m_owns_memory && m_data) {
                BinaryIOMemory::aligned_free(const_cast<uint8_t*>(m_data));
            }
        }

        Reader(Reader&& other) noexcept
            : m_data(other.m_data), m_pos(other.m_pos),
            m_size(other.m_size), m_owns_memory(other.m_owns_memory) {
            other.m_data = nullptr;
            other.m_owns_memory = false;
        }

        Reader& operator=(Reader&& other) noexcept {
            if (this != &other) {
                if (m_owns_memory && m_data) {
                    BinaryIOMemory::aligned_free(const_cast<uint8_t*>(m_data));
                }

                m_data = other.m_data;
                m_pos = other.m_pos;
                m_size = other.m_size;
                m_owns_memory = other.m_owns_memory;

                other.m_data = nullptr;
                other.m_owns_memory = false;
            }
            return *this;
        }

        Reader(const Reader&) = delete;
        Reader& operator=(const Reader&) = delete;

        template<typename T>
        [[nodiscard]] BINARY_IO_FORCE_INLINE T read() noexcept {
            static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");

            if (BINARY_IO_UNLIKELY((m_pos % BinaryIOConfig::CACHE_LINE_SIZE) ==
                (BinaryIOConfig::CACHE_LINE_SIZE - sizeof(T) - 8))) {
                BINARY_IO_PREFETCH(m_data + m_pos + BinaryIOConfig::PREFETCH_DISTANCE, 0, 3);
            }

            T val = BinaryIOSIMD::load_unaligned<T>(m_data + m_pos);
            m_pos += sizeof(T);
            return val;
        }

        [[nodiscard]] BINARY_IO_FORCE_INLINE std::string read_string() noexcept {
            const uint16_t str_len = read<uint16_t>();
            return read_string_data(str_len);
        }

        [[nodiscard]] BINARY_IO_FORCE_INLINE std::string read_string(uint16_t len_size) noexcept {
            std::size_t str_len;

            if (BINARY_IO_LIKELY(len_size == 2)) {
                str_len = read<uint16_t>();
            }
            else if (len_size == 4) {
                str_len = read<uint32_t>();
            }
            else {
                str_len = 0;
                std::memcpy(&str_len, m_data + m_pos, len_size);
                m_pos += len_size;
            }

            return read_string_data(str_len);
        }

        [[nodiscard]] BINARY_IO_FORCE_INLINE std::vector<uint8_t> read_bytes(std::size_t len) noexcept {
            std::vector<uint8_t> result(len);

            if (BINARY_IO_UNLIKELY(len >= BinaryIOConfig::SIMD_THRESHOLD)) {
                BinaryIOSIMD::copy_memory(result.data(), m_data + m_pos, len);
            }
            else {
                std::memcpy(result.data(), m_data + m_pos, len);
            }

            m_pos += len;
            return result;
        }

        BINARY_IO_FORCE_INLINE void read_bytes(uint8_t* BINARY_IO_RESTRICT buffer, std::size_t len) noexcept {
            if (BINARY_IO_UNLIKELY(len >= BinaryIOConfig::SIMD_THRESHOLD)) {
                BinaryIOSIMD::copy_memory(buffer, m_data + m_pos, len);
            }
            else {
                std::memcpy(buffer, m_data + m_pos, len);
            }
            m_pos += len;
        }

        BINARY_IO_FORCE_INLINE void skip(std::size_t len) noexcept { m_pos += len; }
        BINARY_IO_FORCE_INLINE void set_pos(std::size_t pos) noexcept { m_pos = pos; }
        BINARY_IO_FORCE_INLINE void reset() noexcept { m_pos = 0; }

        [[nodiscard]] BINARY_IO_FORCE_INLINE std::size_t get_pos() const noexcept { return m_pos; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE std::size_t get_size() const noexcept { return m_size; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE std::size_t remaining() const noexcept { return m_size - m_pos; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE const uint8_t* get_data() const noexcept { return m_data; }
        [[nodiscard]] BINARY_IO_FORCE_INLINE bool owns_memory() const noexcept { return m_owns_memory; }

        template<typename T>
        [[nodiscard]] BINARY_IO_FORCE_INLINE T peek() const noexcept {
            static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");
            return BinaryIOSIMD::load_unaligned<T>(m_data + m_pos);
        }

        template<typename T>
        [[nodiscard]] BINARY_IO_FORCE_INLINE T peek_at(std::size_t offset) const noexcept {
            static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");
            return BinaryIOSIMD::load_unaligned<T>(m_data + offset);
        }

        [[nodiscard]] BINARY_IO_FORCE_INLINE bool can_read(std::size_t bytes) const noexcept {
            return m_pos + bytes <= m_size;
        }

        [[nodiscard]] BINARY_IO_FORCE_INLINE bool at_end() const noexcept {
            return m_pos >= m_size;
        }

    private:
        BINARY_IO_FORCE_INLINE std::string read_string_data(std::size_t len) noexcept {
            if (BINARY_IO_UNLIKELY(len == 0)) {
                return std::string{};
            }

            std::string result;
            result.reserve(len);
            result.assign(reinterpret_cast<const char*>(m_data + m_pos), len);

            m_pos += len;
            return result;
        }

        const uint8_t* BINARY_IO_RESTRICT m_data;
        std::size_t m_pos;
        std::size_t m_size;
        bool m_owns_memory;
    };
}