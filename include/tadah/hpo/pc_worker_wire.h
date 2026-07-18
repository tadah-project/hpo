#pragma once
#ifndef TADAH_HPO_PC_WORKER_WIRE_H
#define TADAH_HPO_PC_WORKER_WIRE_H

/* Wire codec for the PC_ISOLATION worker-process protocol
 * (pc_worker_pool.h). Deliberately tiny and dependency-free so it is
 * unit-testable in isolation (test_pc_frame_codec.cpp).
 *
 * Frames: header {u32 magic 'TPCW', u32 type, u64 payload_len} followed by
 * payload_len bytes. Host-endian (same machine, same binary -- the worker is
 * a re-exec of the parent). Payload primitives: u8/u32/u64/f64; str = u64
 * length + bytes; vec<T> = u64 count + elements. All socket writes use
 * MSG_NOSIGNAL so a dead peer yields EPIPE instead of SIGPIPE. */

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace tadah {
namespace hpo {
namespace pcwire {

constexpr std::uint32_t MAGIC = 0x54504357u;              /* 'TPCW' */
constexpr std::uint64_t MAX_PAYLOAD = 256ull * 1024 * 1024; /* 256 MiB cap */

enum class FrameType : std::uint32_t {
  HELLO = 1,
  HELLO_ACK = 2,
  TASK = 3,
  RESULT = 4,
  SHUTDOWN = 5,
};

/* ---- payload writer --------------------------------------------------- */
struct Writer {
  std::string buf;
  void u8(std::uint8_t v)   { buf.push_back(static_cast<char>(v)); }
  void u32(std::uint32_t v) { buf.append(reinterpret_cast<const char*>(&v), 4); }
  void u64(std::uint64_t v) { buf.append(reinterpret_cast<const char*>(&v), 8); }
  void f64(double v)        { buf.append(reinterpret_cast<const char*>(&v), 8); }
  void str(const std::string& s) { u64(s.size()); buf.append(s); }
  void bytes(const std::vector<char>& b) { u64(b.size()); buf.append(b.data(), b.size()); }
  void vstr(const std::vector<std::string>& v) {
    u64(v.size());
    for (const auto& s : v) str(s);
  }
};

/* ---- payload reader (throws std::runtime_error on truncation) --------- */
struct Reader {
  const char* p;
  const char* end;
  explicit Reader(const std::string& b) : p(b.data()), end(b.data() + b.size()) {}
  void need(std::size_t n) const {
    if (static_cast<std::size_t>(end - p) < n)
      throw std::runtime_error("pcwire: truncated payload");
  }
  std::uint8_t u8()   { need(1); return static_cast<std::uint8_t>(*p++); }
  std::uint32_t u32() { need(4); std::uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
  std::uint64_t u64() { need(8); std::uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
  double f64()        { need(8); double v; std::memcpy(&v, p, 8); p += 8; return v; }
  std::string str() {
    const std::uint64_t n = u64();
    if (n > MAX_PAYLOAD) throw std::runtime_error("pcwire: oversized string");
    need(n);
    std::string s(p, p + n);
    p += n;
    return s;
  }
  std::vector<char> bytes() {
    const std::uint64_t n = u64();
    if (n > MAX_PAYLOAD) throw std::runtime_error("pcwire: oversized bytes");
    need(n);
    std::vector<char> b(p, p + n);
    p += n;
    return b;
  }
  std::vector<std::string> vstr() {
    const std::uint64_t n = u64();
    if (n > MAX_PAYLOAD) throw std::runtime_error("pcwire: oversized vector");
    std::vector<std::string> v;
    v.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) v.push_back(str());
    return v;
  }
};

/* ---- blocking-with-deadline socket I/O -------------------------------- */
/* deadline_ms < 0 : wait indefinitely. Returns false on timeout; throws on
 * hard I/O errors and EOF (message names the condition). EINTR retried. */
inline bool readn(int fd, void* dst, std::size_t n, long deadline_ms) {
  char* out = static_cast<char*>(dst);
  while (n > 0) {
    if (deadline_ms >= 0) {
      struct pollfd pf{fd, POLLIN, 0};
      int pr;
      do { pr = ::poll(&pf, 1, static_cast<int>(deadline_ms)); }
      while (pr < 0 && errno == EINTR);
      if (pr == 0) return false;                       /* deadline expired */
      if (pr < 0) throw std::runtime_error("pcwire: poll failed");
    }
    ssize_t k;
    do { k = ::recv(fd, out, n, 0); } while (k < 0 && errno == EINTR);
    if (k == 0) throw std::runtime_error("pcwire: peer closed (EOF)");
    if (k < 0)  throw std::runtime_error(std::string("pcwire: recv failed: ")
                                         + std::strerror(errno));
    out += k;
    n -= static_cast<std::size_t>(k);
  }
  return true;
}

inline void writen(int fd, const void* src, std::size_t n) {
  const char* in = static_cast<const char*>(src);
  while (n > 0) {
    ssize_t k;
    do { k = ::send(fd, in, n, MSG_NOSIGNAL); } while (k < 0 && errno == EINTR);
    if (k <= 0) throw std::runtime_error(std::string("pcwire: send failed: ")
                                         + std::strerror(errno));
    in += k;
    n -= static_cast<std::size_t>(k);
  }
}

struct FrameHeader {
  std::uint32_t magic;
  std::uint32_t type;
  std::uint64_t len;
};

inline void send_frame(int fd, FrameType type, const std::string& payload) {
  FrameHeader h{MAGIC, static_cast<std::uint32_t>(type), payload.size()};
  writen(fd, &h, sizeof h);
  if (!payload.empty()) writen(fd, payload.data(), payload.size());
}

/* Returns false on deadline expiry BEFORE the frame header arrived (the
 * caller decides: kill a hung worker). Once a header has started arriving,
 * the body read uses a generous fixed I/O deadline so a wedged peer cannot
 * stall the parent forever. Throws on EOF/garbage/oversize. */
inline bool recv_frame(int fd, FrameType& type, std::string& payload,
                       long deadline_ms) {
  FrameHeader h{};
  if (!readn(fd, &h, sizeof h, deadline_ms)) return false;
  if (h.magic != MAGIC) throw std::runtime_error("pcwire: bad frame magic");
  if (h.len > MAX_PAYLOAD) throw std::runtime_error("pcwire: oversized frame");
  type = static_cast<FrameType>(h.type);
  payload.resize(static_cast<std::size_t>(h.len));
  if (h.len > 0) {
    constexpr long IO_DEADLINE_MS = 60'000;
    if (!readn(fd, payload.data(), payload.size(), IO_DEADLINE_MS))
      throw std::runtime_error("pcwire: frame body stalled");
  }
  return true;
}

}  // namespace pcwire
}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_PC_WORKER_WIRE_H
