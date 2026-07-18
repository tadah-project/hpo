/* PC_ISOLATION wire codec unit tests (no process spawning, no sockets for
 * the pure payload tests; a socketpair for the frame-level tests).
 * Design: INVESTIGATIONS/PC_ISOLATION_process_pool.md section 4. */

#include "catch2/catch.hpp"

#include <tadah/hpo/pc_worker_wire.h>

#include <cstring>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

using namespace tadah::hpo;

TEST_CASE("pcwire payload primitives round-trip", "[pc_wire]") {
  pcwire::Writer w;
  w.u8(7);
  w.u32(0xDEADBEEFu);
  w.u64(0x0123456789ABCDEFull);
  w.f64(-1.5e-300);
  w.str("hello \x01 world");
  w.str("");                       /* empty string round-trips */
  w.bytes(std::vector<char>{'\0', 'x', '\n'});
  w.vstr({"a", "", "ccc"});

  pcwire::Reader r(w.buf);
  CHECK(r.u8() == 7);
  CHECK(r.u32() == 0xDEADBEEFu);
  CHECK(r.u64() == 0x0123456789ABCDEFull);
  CHECK(r.f64() == -1.5e-300);
  CHECK(r.str() == "hello \x01 world");
  CHECK(r.str().empty());
  const auto b = r.bytes();
  REQUIRE(b.size() == 3);
  CHECK(b[0] == '\0');
  CHECK(b[2] == '\n');
  const auto v = r.vstr();
  REQUIRE(v.size() == 3);
  CHECK(v[0] == "a");
  CHECK(v[1].empty());
  CHECK(v[2] == "ccc");
}

TEST_CASE("pcwire reader rejects truncation and oversize", "[pc_wire]") {
  SECTION("truncated fixed-width read") {
    std::string buf = "ab";                 /* 2 bytes, u32 needs 4 */
    pcwire::Reader r(buf);
    CHECK_THROWS_AS(r.u32(), std::runtime_error);
  }
  SECTION("string length larger than remaining payload") {
    pcwire::Writer w;
    w.u64(1000);                             /* claims 1000 bytes... */
    w.buf += "short";                        /* ...delivers 5 */
    pcwire::Reader r(w.buf);
    CHECK_THROWS_AS(r.str(), std::runtime_error);
  }
  SECTION("absurd length is rejected before allocation") {
    pcwire::Writer w;
    w.u64(pcwire::MAX_PAYLOAD + 1);
    pcwire::Reader r(w.buf);
    CHECK_THROWS_AS(r.str(), std::runtime_error);
  }
}

TEST_CASE("pcwire frames over a socketpair", "[pc_wire]") {
  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

  SECTION("frame round-trip") {
    pcwire::Writer w;
    w.u64(42);
    w.str("payload");
    pcwire::send_frame(sv[0], pcwire::FrameType::TASK, w.buf);

    pcwire::FrameType t{};
    std::string payload;
    REQUIRE(pcwire::recv_frame(sv[1], t, payload, 1000));
    CHECK(t == pcwire::FrameType::TASK);
    pcwire::Reader r(payload);
    CHECK(r.u64() == 42);
    CHECK(r.str() == "payload");
  }

  SECTION("deadline expiry before any frame -> returns false (no throw)") {
    pcwire::FrameType t{};
    std::string payload;
    CHECK_FALSE(pcwire::recv_frame(sv[1], t, payload, 50));
  }

  SECTION("bad magic -> throws") {
    pcwire::FrameHeader h{0x0BADF00Du, 1, 0};
    pcwire::writen(sv[0], &h, sizeof h);
    pcwire::FrameType t{};
    std::string payload;
    CHECK_THROWS_AS(pcwire::recv_frame(sv[1], t, payload, 1000),
                    std::runtime_error);
  }

  SECTION("oversized frame -> throws") {
    pcwire::FrameHeader h{pcwire::MAGIC, 1, pcwire::MAX_PAYLOAD + 1};
    pcwire::writen(sv[0], &h, sizeof h);
    pcwire::FrameType t{};
    std::string payload;
    CHECK_THROWS_AS(pcwire::recv_frame(sv[1], t, payload, 1000),
                    std::runtime_error);
  }

  SECTION("peer close mid-frame -> throws EOF") {
    pcwire::FrameHeader h{pcwire::MAGIC, 1, 100};   /* body never sent */
    pcwire::writen(sv[0], &h, sizeof h);
    ::close(sv[0]);
    sv[0] = -1;
    pcwire::FrameType t{};
    std::string payload;
    CHECK_THROWS_AS(pcwire::recv_frame(sv[1], t, payload, 1000),
                    std::runtime_error);
  }

  SECTION("send to closed peer -> throws (EPIPE, not SIGPIPE)") {
    ::close(sv[1]);
    sv[1] = -1;
    std::string big(1 << 20, 'x');   /* larger than the socket buffer */
    CHECK_THROWS_AS(pcwire::send_frame(sv[0], pcwire::FrameType::TASK, big),
                    std::runtime_error);
  }

  if (sv[0] >= 0) ::close(sv[0]);
  if (sv[1] >= 0) ::close(sv[1]);
}
