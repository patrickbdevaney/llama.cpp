#pragma once

// DeepSeek-V4.1 Engram O_DIRECT row store (Thor patch, deepseek-v4.1-flash-reap-thor/llamacpp).
//
// The two Engram tables are 101.5 GB each in FP8. On a unified-memory device (Jetson Thor, DGX Spark)
// holding them in host memory or mmap competes with the GPU pool, and mmap pulls ~64 KiB of page cache
// per 264 B row touched. Instead each table lives in a packed side file and every lookup is a direct
// pread of the aligned window that holds the row. File format: llamacpp/ENGRAM_ROWFILE.md.
//
// Enable with LLAMA_ENGRAM_ROWFILE_DIR=<dir> holding engram_L<layer>.v41rows; the GGUF then needs no
// engram_embd tensors (and ignores them if present). Optional: LLAMA_ENGRAM_THREADS (default 16),
// LLAMA_ENGRAM_ALIGN (default 4096; 512 matches the 1 KiB windows measured on the Thor NVMe).

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct deepseek41_engram_rowstore {
    static constexpr char     MAGIC[8]    = {'V', '4', '1', 'E', 'N', 'G', 'R', '1'};
    static constexpr uint32_t VERSION     = 1;
    static constexpr size_t   HEADER_SIZE = 4096;

    int      fd         = -1;
    bool     direct     = false;
    size_t   align      = 4096;
    int      n_threads  = 16;
    uint32_t layer_id   = 0;
    uint64_t n_rows     = 0;
    uint32_t key_len    = 0;   // values per row (256)
    uint32_t n_scales   = 0;   // E8M0 scales per row (8, one per 32 values)
    uint32_t row_bytes  = 0;   // key_len + n_scales (264)
    uint64_t data_off   = 0;
    std::string path;
    float fp8_lut[256];

    deepseek41_engram_rowstore(const std::string & p, uint32_t expect_layer) : path(p) {
        if (const char * e = getenv("LLAMA_ENGRAM_ALIGN"))   { align     = (size_t) std::max(512L, atol(e)); }
        if (const char * e = getenv("LLAMA_ENGRAM_THREADS")) { n_threads = std::max(1, atoi(e)); }

        fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
        direct = fd >= 0;
        if (fd < 0) {
            fd = ::open(p.c_str(), O_RDONLY);   // e.g. a filesystem without O_DIRECT; correct, just slower
        }
        if (fd < 0) {
            throw std::runtime_error("engram rowstore: cannot open " + p + ": " + strerror(errno));
        }

        std::vector<uint8_t> hdr = read_window(0, HEADER_SIZE);
        if (memcmp(hdr.data(), MAGIC, 8) != 0) {
            throw std::runtime_error("engram rowstore: bad magic in " + p);
        }
        uint32_t version;
        memcpy(&version,   hdr.data() +  8, 4);
        memcpy(&layer_id,  hdr.data() + 12, 4);
        memcpy(&n_rows,    hdr.data() + 16, 8);
        memcpy(&key_len,   hdr.data() + 24, 4);
        memcpy(&n_scales,  hdr.data() + 28, 4);
        memcpy(&row_bytes, hdr.data() + 32, 4);
        memcpy(&data_off,  hdr.data() + 44, 8);
        if (version != VERSION || layer_id != expect_layer || row_bytes != key_len + n_scales ||
                n_scales == 0 || key_len % n_scales != 0 || data_off % 4096 != 0) {
            throw std::runtime_error("engram rowstore: header mismatch in " + p);
        }
        for (int i = 0; i < 256; ++i) {
            fp8_lut[i] = fp8_e4m3fn_to_f32((uint8_t) i);
        }
    }

    ~deepseek41_engram_rowstore() {
        if (fd >= 0) {
            ::close(fd);
        }
    }

    deepseek41_engram_rowstore(const deepseek41_engram_rowstore &) = delete;
    deepseek41_engram_rowstore & operator=(const deepseek41_engram_rowstore &) = delete;

    static float fp8_e4m3fn_to_f32(uint8_t v) {
        const int s = v >> 7, e = (v >> 3) & 0xF, m = v & 0x7;
        float r;
        if (e == 0xF && m == 0x7) {
            r = NAN;
        } else if (e == 0) {
            r = std::ldexp((float) m / 8.0f, -6);
        } else {
            r = std::ldexp(1.0f + (float) m / 8.0f, e - 7);
        }
        return s ? -r : r;
    }

    // the reference dequantizes in fp32 and then casts to bfloat16 (ParallelEngramEmbedding.forward)
    static float round_bf16(float x) {
        uint32_t u;
        memcpy(&u, &x, 4);
        if ((u & 0x7f800000u) != 0x7f800000u) {
            u += 0x7fffu + ((u >> 16) & 1u);
        }
        u &= 0xffff0000u;
        memcpy(&x, &u, 4);
        return x;
    }

    std::vector<uint8_t> read_window(uint64_t off, size_t len) const {
        // aligned buffer, offset and length, as O_DIRECT requires
        const uint64_t start = off & ~(uint64_t) (align - 1);
        const size_t   span  = (size_t) (((off + len - start) + align - 1) & ~(uint64_t) (align - 1));
        void * buf = nullptr;
        if (posix_memalign(&buf, align, span) != 0) {
            throw std::runtime_error("engram rowstore: posix_memalign failed");
        }
        size_t got = 0;
        while (got < span) {
            const ssize_t r = pread(fd, (uint8_t *) buf + got, span - got, (off_t) (start + got));
            if (r <= 0) {
                if (r < 0 && errno == EINTR) {
                    continue;
                }
                if (got >= (off - start) + len) {
                    break;   // short read at end of file, the row itself is complete
                }
                free(buf);
                throw std::runtime_error("engram rowstore: pread failed on " + path + ": " + strerror(errno));
            }
            got += (size_t) r;
        }
        std::vector<uint8_t> out((uint8_t *) buf + (off - start), (uint8_t *) buf + (off - start) + len);
        free(buf);
        return out;
    }

    void dequant_row(const uint8_t * row, float * dst) const {
        const uint32_t group = key_len / n_scales;
        for (uint32_t g = 0; g < n_scales; ++g) {
            const uint8_t e = row[key_len + g];
            const float scale = e == 0xFF ? NAN : std::ldexp(1.0f, (int) e - 127);
            for (uint32_t c = 0; c < group; ++c) {
                dst[g*group + c] = round_bf16(fp8_lut[row[g*group + c]] * scale);
            }
        }
    }

    // out[i*key_len .. +key_len) = dequantized row ids[i]; reads run on n_threads threads
    void gather(const int32_t * ids, size_t n, float * out) const {
        auto work = [&](size_t t0, size_t step) {
            for (size_t i = t0; i < n; i += step) {
                const int64_t id = ids[i];
                if (id < 0 || (uint64_t) id >= n_rows) {
                    throw std::runtime_error("engram rowstore: row id out of range");
                }
                const std::vector<uint8_t> row = read_window(data_off + (uint64_t) id * row_bytes, row_bytes);
                dequant_row(row.data(), out + i*key_len);
            }
        };
        const size_t nt = std::min<size_t>((size_t) n_threads, std::max<size_t>(1, n / 4));
        if (nt <= 1) {
            work(0, 1);
            return;
        }
        std::vector<std::thread> pool;
        std::vector<std::string> errs(nt);
        for (size_t t = 0; t < nt; ++t) {
            pool.emplace_back([&, t] {
                try { work(t, nt); } catch (const std::exception & ex) { errs[t] = ex.what(); }
            });
        }
        for (auto & th : pool) {
            th.join();
        }
        for (const auto & e : errs) {
            if (!e.empty()) {
                throw std::runtime_error(e);
            }
        }
    }
};
