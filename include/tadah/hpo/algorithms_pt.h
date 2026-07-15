#pragma once
/* ----- ANNEAL_PT  (parallel tempering / replica exchange) ------------
 *
 * Refs: Swendsen & Wang, PRL 57 (1986); Geyer (1991); Earl & Deem,
 *       PCCP 7 (2005) [review]; Corana et al., ACM TOMS 13:262 (1987)
 *       for the per-replica adaptive stepper; the deterministic
 *       even/odd (DEO) pairing follows Okabe et al., CPL 335 (2001).
 *
 * K replicas run the shared annealing rung (anneal_core.h) at fixed
 * temperatures T_0 < T_1 < ... < T_{K-1} (slot 0 coldest). Every
 * SWAP_EVERY epochs, adjacent pairs attempt a state exchange with
 * Metropolis probability
 *
 *     p = min(1, exp((beta_i - beta_j) * (f_i - f_j)))     (i colder)
 *
 * so a better (lower-f) state always migrates cold-ward. Swaps move
 * STATES (x, f), not temperatures: the Corana step vector V is
 * calibrated to the acceptance rate AT its slot's temperature and
 * therefore stays with the slot.
 *
 * Execution: logical replicas are statically assigned to a persistent
 * ReplicaPool (THREADS workers); one host, many threads -- HPO_Host's
 * operator() is thread-safe by contract. Same SEED => identical
 * results for ANY thread count (per-replica RNG streams, driver-side
 * swap coins keyed by epoch, deterministic per-epoch budget slicing).
 * Documented exceptions: MAXTIME truncation and the tails of runs cut
 * short by STOPVAL/exceptions are timing-dependent.
 *
 * Ladder construction: T_HOT from PARAM T_HOT or the shared T0
 * calibration (target initial acceptance T0_PACCEPT at the hot end);
 * T_COLD from PARAM T_COLD or T_HOT / LADDER_SPAN; geometric spacing
 * in between (adjacent ratio (T_HOT/T_COLD)^(1/(K-1)) -- aim for
 * 20-40% swap acceptance; the per-pair rates are logged so LADDER_SPAN
 * can be tuned). RT (default 1.0) < 1 additionally cools the WHOLE
 * ladder each epoch (PT -> SA hybrid); the run then ends when the hot
 * end crosses T_MIN.
 *
 * PT-specific PARAMs (shared stepper knobs as in ANNEAL):
 *   REPLICAS     max(THREADS,2)  ladder size K
 *   T_HOT        auto            hottest slot
 *   T_COLD       T_HOT/LADDER_SPAN  coldest slot
 *   LADDER_SPAN  1e3             T_HOT/T_COLD when T_COLD unset
 *   SWAP_EVERY   1               exchange attempt every N epochs
 *   RT           1.0             ladder cooling per epoch ((0,1])
 *   INIT_MODE    0               0: chain0=seed, others sigma-jittered
 *                                1: chains>0 drawn uniformly in the box
 *   RESET_TO_BEST 0              when 1, snaps ONLY the coldest slot to
 *                                the global best after each epoch
 */
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_parallel.h>
#include <tadah/hpo/warmstart_cache.h>   // current_chain() bridge

#ifdef TADAH_ENABLE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tadah::hpo {

template<typename Vec, typename HostT>
  void run_anneal_pt_impl(const OptimizerSpec& sp, HostT& host)
  {
    using anneal::Knobs;
    using anneal::AlgoKind;
    using anneal::RungStop;

    const std::size_t dim = sp.low.size();
    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());

    const Knobs kb = Knobs::parse(sp, AlgoKind::PT);

    const std::size_t threads = sp.num_threads.value_or(1);
    const std::size_t K = kb.replicas ? kb.replicas
                                      : std::max<std::size_t>(threads, 2);
    if (K < 2)
      throw std::runtime_error("TADAH:ANNEAL_PT: needs REPLICAS >= 2");

    /* --- MPI layout (no-op when built without TADAH_HPO_MPI or run on
     * one rank): contiguous ladder blocks per rank, slot 0 (coldest) on
     * rank 0. Chain RNG streams are keyed by GLOBAL slot index, budget
     * slices are computed from the GLOBAL remaining budget, and swap
     * coins come from per-pair streams -- so the evaluated set is
     * identical for any rank count (and any THREADS). */
    const int world = hpo_size();
    const int rank  = hpo_rank();
    if (static_cast<std::size_t>(world) > K)
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: REPLICAS (" + std::to_string(K) +
          ") must be >= the MPI world size (" + std::to_string(world) + ")");
    auto slot_count = [&](int r) -> std::size_t {
      return K / world + (static_cast<std::size_t>(r) < K % world ? 1 : 0);
    };
    auto slot_offset = [&](int r) -> std::size_t {
      std::size_t off = 0;
      for (int q = 0; q < r; ++q) off += slot_count(q);
      return off;
    };
    const std::size_t n_local  = slot_count(rank);
    const std::size_t off_local = slot_offset(rank);

    if (!sp.num_threads) {
      const std::string hint =
          "ANNEAL_PT: THREADS not set -- running " + std::to_string(K) +
          " replicas on 1 worker thread. Set THREADS <n> in the STRATEGY "
          "block to evaluate replicas concurrently.";
      std::cout << "[HPO] " << hint << "\n";
      HPOLog::instance().warning(hint);
    }
#ifdef TADAH_ENABLE_OPENMP
    {
      const unsigned hw = std::thread::hardware_concurrency();
      const std::size_t omp_w = static_cast<std::size_t>(omp_get_max_threads());
      if (hw && threads * omp_w > hw) {
        std::ostringstream oss;
        oss << "ANNEAL_PT: THREADS(" << threads << ") x OMP_NUM_THREADS("
            << omp_w << ") = " << threads * omp_w
            << " exceeds hardware_concurrency(" << hw
            << "); expect oversubscription. Aim for THREADS x OMP <= cores.";
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }
#endif

    /* per-dim sigma (transformed space) */
    std::vector<double> sigma(dim);
    if (sp.step.has_value()) {
      for (std::size_t i = 0; i < dim; ++i) sigma[i] = *sp.step;
    } else {
      for (std::size_t i = 0; i < dim; ++i)
        sigma[i] = kb.step_frac * (sp.high[i] - sp.low[i]);
    }

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t max_eval_global = sp.stop.maxeval.value_or(HUGE_UNSIGNED);
    /* Multi-rank runs enforce MAXEVAL through GLOBAL accounting (spent
     * counts allreduced each epoch feed the deterministic slicing), so
     * the local reserve counter stays uncapped there. */
    anneal::SharedBudget budget(world > 1 ? HUGE_UNSIGNED : max_eval_global);

    const auto t0_wall = std::chrono::steady_clock::now();
    const double max_time = sp.stop.maxtime.value_or(0.0);
    const std::function<bool()> time_exceeded = [&]() {
      if (max_time <= 0.0) return false;
      using Sec = std::chrono::duration<double>;
      return std::chrono::duration_cast<Sec>(
               std::chrono::steady_clock::now() - t0_wall).count() >= max_time;
    };

    const double stopval =
        sp.stop.stopval.value_or(-std::numeric_limits<double>::infinity());

    std::atomic<bool> stop_flag{false};
    anneal::StopCtx ctx{budget, stop_flag, time_exceeded, stopval};

    anneal::CoranaPlateau plateau(sp.stop, kb.neps);

    enum class StopReason {
      Done, MaxEval, MaxTime, StopVal, TMin, Plateau,
      StopValOnProbe, MaxTimeOnProbe, NoDims, NoBudget
    };
    StopReason stop_reason = StopReason::Done;

    double      T_hot = 0.0, T_cold = 0.0;
    std::size_t epoch_count    = 0;
    std::size_t exchange_rounds = 0;
    std::size_t total_proposed = 0, total_accepted = 0, total_oob = 0;
    std::size_t reanneal_count = 0;  (void)reanneal_count;
    /* g_best is the GLOBAL best value (allreduced for logging/stops);
     * (l_best, l_best_x) is the best THIS RANK evaluated -- ownership
     * questions (who broadcasts the winning x) are always resolved
     * from l_best, never from the reduced value (after a reduction
     * every rank holds the same number, so a MINLOC over it would tie
     * onto rank 0 which need not own the matching x). */
    double      g_best = std::numeric_limits<double>::infinity();
    double      l_best = std::numeric_limits<double>::infinity();
    Vec         g_best_x(dim);
    Vec         l_best_x(dim);
    bool        ran_sa = false;

    {
      std::ostringstream oss;
      oss << "ANNEAL_PT[" << (kb.corana ? "CORANA" : "CLASSIC") << "]: "
          << "seed=" << seed
          << "  REPLICAS=" << K
          << "  THREADS=" << threads
          << "  SWAP_EVERY=" << kb.swap_every
          << "  RT=" << kb.rt;
      if (kb.t_cold)      oss << "  T_COLD=" << *kb.t_cold;
      else                oss << "  LADDER_SPAN=" << kb.ladder_span;
      if (kb.corana)
        oss << "  NS=" << kb.NS << "  NT=" << kb.NT << "  C=" << kb.C;
      else
        oss << "  L=" << kb.L;
      oss << "  INIT_MODE=" << kb.init_mode;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    if (dim == 0) {
      if (!sp.optim_init.empty() && budget.left() > 0) {
        const auto v = sp.in(sp.optim_init.front());
        Vec xv(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) xv(i) = v[i];
        host(sp.out(xv));
      }
      stop_reason = StopReason::NoDims;
    }

    bool proceed = (stop_reason == StopReason::Done);

    /* --- seed evaluation + T0 calibration --------------------------------
     * Rank 0 evaluates the seed and runs the probe calibration; the
     * outcome (T_hot, f0, x0, or an abort code) is broadcast so every
     * rank proceeds from identical state without duplicate probe evals. */
    Vec x0(dim);
    double f0 = std::numeric_limits<double>::infinity();
    if (proceed && rank == 0) {
      if (!sp.optim_init.empty()) {
        const auto v = sp.in(sp.optim_init.front());
        for (std::size_t i = 0; i < v.size(); ++i) x0(i) = v[i];
      } else {
        for (std::size_t i = 0; i < dim; ++i)
          x0(i) = 0.5 * (sp.low[i] + sp.high[i]);
      }
      if (!budget.try_reserve() ||
          (world == 1 && max_eval_global == 0)) {
        stop_reason = StopReason::NoBudget;
        proceed = false;
      } else {
        f0 = host(sp.out(x0));
        ran_sa = true;
        g_best = l_best = f0;
        g_best_x = l_best_x = x0;
        if (f0 <= stopval) {
          stop_reason = StopReason::StopVal; proceed = false;
        } else if (time_exceeded()) {
          stop_reason = StopReason::MaxTime; proceed = false;
        }
      }
    }

    bool calib_threw = false;
    std::string calib_what;
    if (proceed && rank == 0) {
      std::mt19937_64 probe_rng = anneal::make_stream(seed, anneal::SALT_PROBES);
      try {
        const auto t0out = anneal::calibrate_t0(
            sp, kb, sigma, x0, f0, l_best_x, l_best,
            host, probe_rng, ctx, "ANNEAL_PT");
        g_best = l_best;
        T_hot = t0out.T0;
        if (t0out.aborted) {
          stop_reason = t0out.stopval_hit ? StopReason::StopValOnProbe
                                          : StopReason::MaxTimeOnProbe;
          proceed = false;
        }
      } catch (const std::exception& e) {
        calib_threw = true;
        calib_what  = e.what();
      }
    }
    if (world > 1) {
      /* Unconditional handshake: every rank participates, so a rank-0
       * abort can never leave the others hanging on the broadcast. */
      int code = 0;
      if (rank == 0)
        code = calib_threw ? -1
                           : (proceed ? 0 : 1 + static_cast<int>(stop_reason));
      hpo_bcast(code, 0);
      if (code == -1) {
        if (rank == 0) throw std::runtime_error(calib_what);
        throw std::runtime_error(
            "TADAH:ANNEAL_PT: rank 0 aborted during seed/T0 calibration");
      }
      std::vector<double> pkt(dim + 2, 0.0);
      if (rank == 0) {
        pkt[0] = T_hot; pkt[1] = f0;
        for (std::size_t i = 0; i < dim; ++i) pkt[2 + i] = x0(i);
      }
      hpo_bcast(pkt, 0);
      if (rank != 0) {
        T_hot = pkt[0];
        f0    = pkt[1];
        for (std::size_t i = 0; i < dim; ++i) x0(i) = pkt[2 + i];
        if (code != 0) {
          stop_reason = static_cast<StopReason>(code - 1);
          proceed = false;
        } else {
          proceed  = true;
          ran_sa   = true;
          g_best   = f0;
          g_best_x = x0;
        }
      }
    } else if (calib_threw) {
      throw std::runtime_error(calib_what);
    }
    T_cold = kb.t_cold ? *kb.t_cold : T_hot / kb.ladder_span;
    if (proceed && !(T_cold > 0.0 && T_cold < T_hot))
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: resolved ladder is invalid (T_COLD=" +
          std::to_string(T_cold) + ", T_HOT=" + std::to_string(T_hot) +
          "); set PARAM T_COLD/T_HOT or LADDER_SPAN consistently");

    /* Hybrid cooling floor (only meaningful when RT < 1). */
    const double t_min = kb.t_min ? *kb.t_min
                                  : (T_hot > 0.0 ? T_hot * 1e-6 : 1e-12);

    /* --- ladder + chains ----------------------------------------------
     * Every rank builds the FULL temperature ladder (identical, cheap)
     * but only its own contiguous slot block of chains. Chain RNG
     * streams are keyed by the GLOBAL slot index. */
    std::vector<double> T(K);
    for (std::size_t k = 0; k < K; ++k)
      T[k] = T_cold * std::pow(T_hot / T_cold,
                               static_cast<double>(k) /
                               static_cast<double>(K - 1));

    std::vector<anneal::ChainState<Vec>> chains(n_local);
    anneal::ReplicaPool pool(std::min(threads, n_local ? n_local : 1),
                             stop_flag);

    if (proceed) {
      for (std::size_t li = 0; li < n_local; ++li) {
        const std::size_t k = off_local + li;   /* global slot */
        auto& c = chains[li];
        c.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, k);
        c.V   = sigma;
        c.x   = Vec(dim);
        if (k == 0 || kb.init_mode == 0) {
          /* chain 0 starts at the seed exactly; higher slots jitter
           * around it from their own stream (uniform-box starts are
           * cliff bait on real landscapes). */
          std::normal_distribution<double> nd(0.0, 1.0);
          for (std::size_t i = 0; i < dim; ++i) {
            double v = x0(i);
            if (k > 0) v += sigma[i] * nd(c.rng);
            anneal::reflect_into(v, sp.low[i], sp.high[i]);
            c.x(i) = v;
          }
        } else {
          for (std::size_t i = 0; i < dim; ++i)
            c.x(i) = std::uniform_real_distribution<>(sp.low[i], sp.high[i])(c.rng);
        }
        c.best_x = c.x;
      }
      if (rank == 0 && n_local > 0) {
        chains[0].f      = f0;
        chains[0].best_f = f0;
      }

      /* Evaluate the non-seed starting points on the pool. */
      pool.run_epoch(n_local, [&](std::size_t li) {
        /* key this chain's warm-start by its GLOBAL slot (off_local+li) so
         * results are independent of which worker runs it. */
        warmstart::current_chain() = off_local + li;
        if (rank == 0 && li == 0) return;   /* global slot 0 = seed */
        auto& c = chains[li];
        if (stop_flag.load(std::memory_order_relaxed)) return;
        if (!ctx.budget.try_reserve()) return;   /* f stays inf */
        c.f = host(sp.out(c.x));
        c.best_f = c.f;
        c.best_x = c.x;
        if (c.f <= stopval) stop_flag.store(true, std::memory_order_relaxed);
      });
      pool.rethrow_if_error();
      for (std::size_t li = 0; li < n_local; ++li) {
        if (chains[li].best_f < l_best) {
          l_best   = chains[li].best_f;
          l_best_x = chains[li].best_x;
        }
      }
      g_best = std::min(g_best, l_best);
      if (world > 1) {
        double gb = std::isfinite(l_best) ? l_best
                                          : std::numeric_limits<double>::max();
        int owner = 0;
        hpo_allreduce_minloc(gb, owner);
        if (gb < std::numeric_limits<double>::max())
          g_best = std::min(g_best, gb);
        int code = (stop_flag.load(std::memory_order_relaxed) &&
                    g_best <= stopval) ? 1 : 0;
        code = hpo_allreduce_max(code);
        if (code) { stop_reason = StopReason::StopVal; proceed = false; }
      } else if (stop_flag.load(std::memory_order_relaxed) &&
                 g_best <= stopval) {
        stop_reason = StopReason::StopVal;
        proceed = false;
      }
    }

    /* Per-pair swap-acceptance bookkeeping. */
    std::vector<std::size_t> swap_attempt(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept (K > 1 ? K - 1 : 0, 0);

    const std::uint64_t rung_cost =
        kb.corana ? static_cast<std::uint64_t>(kb.NT) * kb.NS * dim
                  : static_cast<std::uint64_t>(kb.L);

    /* --- epoch loop ------------------------------------------------------
     * Fixed per-epoch collective order (every rank, every epoch):
     * [allreduce spent] -> local rungs -> [allreduce stats+stopcode] ->
     * boundary sendrecvs -> [minloc best] -> [bcast rank-0 stop code]
     * (-> [bcast best-x] when RESET_TO_BEST). Ranks therefore always
     * agree on loop exits and no collective can be left unmatched. */
    if (proceed) {
      std::vector<anneal::RungStats> rstats(n_local);
      std::uniform_real_distribution<double> udist(0.0, 1.0);
      /* Planned total for the epoch=current/total display: min of the
       * budget bound (all replicas rung in lockstep) and, in hybrid
       * mode (RT < 1), the ladder's cooling bound on the hot end.
       * 0 = unbounded (pure PT with no MAXEVAL) -> plain counter. */
      std::size_t planned_epochs = 0;
      for (;;) {
        const std::uint64_t spent_global =
            world > 1 ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const std::uint64_t remaining =
            max_eval_global > spent_global ? max_eval_global - spent_global : 0;
        if (remaining == 0) { stop_reason = StopReason::MaxEval; break; }
        if (planned_epochs == 0 &&
            (max_eval_global != HUGE_UNSIGNED || kb.rt < 1.0)) {
          planned_epochs = anneal::planned_rungs(
              T[K - 1], t_min, kb.rt,
              max_eval_global == HUGE_UNSIGNED
                  ? std::numeric_limits<std::uint64_t>::max()
                  : remaining,
              static_cast<std::uint64_t>(K) * rung_cost);
        }
        if (time_exceeded() && world == 1) {
          stop_reason = StopReason::MaxTime; break;
        }

        /* Deterministic per-epoch budget slicing: full rungs while the
         * budget allows, else floor(remaining/K) with the remainder
         * given to the coldest slots. Computed identically on every
         * rank from the GLOBAL remaining count. */
        std::vector<std::uint64_t> allowance(K, rung_cost);
        if (remaining < static_cast<std::uint64_t>(K) * rung_cost) {
          const std::uint64_t base = remaining / K;
          const std::uint64_t rem  = remaining % K;
          for (std::size_t k = 0; k < K; ++k)
            allowance[k] = base + (k < rem ? 1 : 0);
        }

        pool.run_epoch(n_local, [&](std::size_t li) {
          const std::size_t k = off_local + li;
          warmstart::current_chain() = k;   /* global slot for warm-start key */
          rstats[li] = anneal::run_rung(chains[li], T[k], sp, kb, host, ctx,
                                        allowance[k]);
        });
        pool.rethrow_if_error();

        std::size_t e_prop = 0, e_acc = 0, e_oob = 0;
        bool any_stopval = false, any_time = false;
        for (std::size_t li = 0; li < n_local; ++li) {
          e_prop += rstats[li].proposed;
          e_acc  += rstats[li].accepted;
          e_oob  += rstats[li].oob;
          if (rstats[li].stop == RungStop::StopVal) any_stopval = true;
          if (rstats[li].stop == RungStop::Time)    any_time    = true;
          if (chains[li].best_f < l_best) {
            l_best   = chains[li].best_f;
            l_best_x = chains[li].best_x;
          }
        }
        g_best = std::min(g_best, l_best);
        if (world > 1) {
          e_prop = hpo_allreduce_sum_sz(e_prop);
          e_acc  = hpo_allreduce_sum_sz(e_acc);
          e_oob  = hpo_allreduce_sum_sz(e_oob);
          int code = any_stopval ? 2 : (any_time ? 1 : 0);
          code = hpo_allreduce_max(code);
          any_stopval = (code == 2);
          any_time    = (code == 1);
        }
        total_proposed += e_prop;
        total_accepted += e_acc;
        total_oob      += e_oob;
        ++epoch_count;

        if (any_stopval) { stop_reason = StopReason::StopVal; break; }
        if (any_time)    { stop_reason = StopReason::MaxTime; break; }
        if (e_prop == 0) { stop_reason = StopReason::MaxEval; break; }

        /* --- exchange phase (deterministic; per-pair coin streams so
         * the decision is identical no matter which rank(s) own the
         * pair or how many ranks there are) --------------------------- */
        double ep_swap_rate = -1.0;
        if (epoch_count % kb.swap_every == 0) {
          const std::size_t parity =
              (epoch_count / kb.swap_every) % 2;  /* DEO alternation */
          std::size_t att = 0, acc = 0;
          for (std::size_t i = parity; i + 1 < K; i += 2) {
            const bool i_mine = (i >= off_local && i < off_local + n_local);
            const bool j_mine = (i + 1 >= off_local && i + 1 < off_local + n_local);
            if (!i_mine && !j_mine) continue;
            std::mt19937_64 pair_rng = anneal::make_stream(
                seed, anneal::SALT_PT_SWAP,
                static_cast<std::uint64_t>(epoch_count) * K + i);
            const double coin = udist(pair_rng);
            const double beta_i = 1.0 / T[i];      /* colder */
            const double beta_j = 1.0 / T[i + 1];  /* hotter */

            if (i_mine && j_mine) {
              auto& ci = chains[i - off_local];
              auto& cj = chains[i + 1 - off_local];
              const double delta = (beta_i - beta_j) * (ci.f - cj.f);
              ++swap_attempt[i];
              ++att;
              if (delta >= 0.0 || coin < std::exp(delta)) {
                std::swap(ci.x, cj.x);
                std::swap(ci.f, cj.f);
                ++swap_accept[i];
                ++acc;
              }
            } else {
              /* Boundary pair: I own exactly one side; my peer owns the
               * other. Both sides draw the SAME coin from the pair
               * stream and compute the SAME decision -- only f (and x
               * on accept) travel. */
              const std::size_t mine_g = i_mine ? i : i + 1;
              auto& c = chains[mine_g - off_local];
              const int peer = i_mine ? rank + 1 : rank - 1;
              std::vector<double> sendf{c.f}, recvf(1, 0.0);
              hpo_sendrecv(sendf, recvf, peer, static_cast<int>(i % 1024) * 2);
              const double f_i = i_mine ? c.f : recvf[0];
              const double f_j = i_mine ? recvf[0] : c.f;
              const double delta = (beta_i - beta_j) * (f_i - f_j);
              const bool accepted = (delta >= 0.0) || coin < std::exp(delta);
              if (i_mine) {
                ++swap_attempt[i];
                ++att;
                if (accepted) { ++swap_accept[i]; ++acc; }
              }
              if (accepted) {
                std::vector<double> sx(dim), rx(dim, 0.0);
                for (std::size_t d = 0; d < dim; ++d) sx[d] = c.x(d);
                hpo_sendrecv(sx, rx, peer, static_cast<int>(i % 1024) * 2 + 1);
                for (std::size_t d = 0; d < dim; ++d) c.x(d) = rx[d];
                c.f = i_mine ? f_j : f_i;
              }
            }
          }
          ++exchange_rounds;
          ep_swap_rate = att > 0 ? static_cast<double>(acc) /
                                   static_cast<double>(att) : 0.0;
        }

        /* --- global best + rank-0 stop decisions ----------------------
         * Ownership resolved from the rank-LOCAL best (see l_best). */
        int best_owner = 0;
        if (world > 1) {
          double gb = std::isfinite(l_best) ? l_best
                                            : std::numeric_limits<double>::max();
          hpo_allreduce_minloc(gb, best_owner);
          if (gb < std::numeric_limits<double>::max())
            g_best = std::min(g_best, gb);
        }

        int rank0_code = 0;   /* 0 none, 1 plateau */
        if (rank == 0) {
          if (plateau.update(chains[0].f, g_best)) rank0_code = 1;
        }
        if (world > 1) hpo_bcast(rank0_code, 0);
        if (rank0_code == 1) { stop_reason = StopReason::Plateau; break; }

        /* --- optional ladder cooling (PT -> SA hybrid) ---------------- */
        if (kb.rt < 1.0) {
          for (auto& t : T) t *= kb.rt;
          if (T[K - 1] < t_min) { stop_reason = StopReason::TMin; break; }
        }

        /* --- reset-to-best: coldest slot only ------------------------- */
        if (kb.reset_to_best) {
          std::vector<double> bx(dim, 0.0);
          if (world > 1) {
            if (rank == best_owner)
              for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
            hpo_bcast(bx, best_owner);
          } else {
            for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
          }
          if (rank == 0 && n_local > 0 && chains[0].f > g_best) {
            for (std::size_t d = 0; d < dim; ++d) chains[0].x(d) = bx[d];
            chains[0].f = g_best;
          }
        }

        /* --- one journal line per epoch (rank 0 carries the global
         * numbers; other ranks stay quiet) ----------------------------- */
        if (rank == 0) {
          const double rate = e_prop > 0
              ? static_cast<double>(e_acc) / static_cast<double>(e_prop) : 0.0;
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL_PT epoch=" << epoch_count;
          if (planned_epochs > 0) oss << "/" << planned_epochs;
          oss << "  T=[" << T[0] << ", " << T[K - 1] << "]"
              << "  proposed=" << e_prop
              << "  accepted=" << std::fixed << std::setprecision(1)
              << (rate * 100.0) << "%";
          if (ep_swap_rate >= 0.0)
            oss << "  swap_acc=" << (ep_swap_rate * 100.0) << "%";
          oss << std::scientific << std::setprecision(3)
              << "  f_cold=" << chains[0].f
              << "  best=" << g_best;
          HPOLog::instance().info(oss.str());
        }
      }
    }

    /* --- multi-rank finalisation ------------------------------------------
     * Rank 0's best_* output files must carry the GLOBAL winner: the
     * owning rank broadcasts its best x and rank 0 re-evaluates it once
     * (one eval) so the host-side writers record it. Guarded by dim>0
     * (symmetric on every rank) rather than per-rank flags so the
     * collectives can never be left unmatched. */
    std::uint64_t spent_report = budget.spent();
    if (world > 1 && dim > 0) {
      double gb = std::isfinite(l_best) ? l_best
                                        : std::numeric_limits<double>::max();
      int owner = 0;
      hpo_allreduce_minloc(gb, owner);
      if (gb < std::numeric_limits<double>::max()) {
        g_best = gb;
        std::vector<double> bx(dim, 0.0);
        if (rank == owner)
          for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
        hpo_bcast(bx, owner);
        for (std::size_t d = 0; d < dim; ++d) g_best_x(d) = bx[d];
        if (rank == 0 && owner != 0) {
          budget.try_reserve();   /* accounting only */
          (void)host(sp.out(g_best_x));
        }
      }
      spent_report = hpo_allreduce_sum(budget.spent());
    }

    /* --- final summary --------------------------------------------------- */
    if (ran_sa) {
      const double overall_rate = total_proposed > 0
          ? static_cast<double>(total_accepted) /
            static_cast<double>(total_proposed) : 0.0;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(3)
          << "ANNEAL_PT summary: epochs=" << epoch_count
          << "  replicas=" << K
          << "  proposed=" << total_proposed
          << "  accepted=" << std::fixed << std::setprecision(1)
          << (overall_rate * 100.0) << "%"
          << std::scientific << std::setprecision(3)
          << "  oob=" << total_oob
          << "  T_hot=" << T_hot
          << "  T_cold=" << T_cold
          << "  best=" << g_best;
      HPOLog::instance().info(oss.str());
      /* Per-pair swap-acceptance table -- THE diagnostic for tuning
       * LADDER_SPAN/REPLICAS (aim 20-40% per pair). */
      if (exchange_rounds > 0) {
        std::ostringstream ossp;
        ossp << "ANNEAL_PT swap acceptance per pair (cold->hot):";
        for (std::size_t i = 0; i + 1 < K; ++i) {
          const double r = swap_attempt[i] > 0
              ? 100.0 * static_cast<double>(swap_accept[i]) /
                        static_cast<double>(swap_attempt[i]) : 0.0;
          ossp << "  " << i << ":" << std::fixed << std::setprecision(0)
               << r << "%";
        }
        HPOLog::instance().info(ossp.str());
      }
    }

    {
      const char* reason = nullptr;
      switch (stop_reason) {
        case StopReason::MaxEval:        reason = "MAXEVAL exhausted"; break;
        case StopReason::MaxTime:        reason = "MAXTIME exceeded"; break;
        case StopReason::StopVal:        reason = "STOPVAL reached"; break;
        case StopReason::StopValOnProbe: reason = "STOPVAL reached during T0 probes"; break;
        case StopReason::MaxTimeOnProbe: reason = "MAXTIME exceeded during T0 probes"; break;
        case StopReason::TMin:           reason = "T < T_MIN (hot end)"; break;
        case StopReason::Plateau:        reason = "FTOL plateau (Corana eps-NEPS rule, coldest slot)"; break;
        case StopReason::NoDims:         reason = "no optimisable dimensions"; break;
        case StopReason::NoBudget:       reason = "MAXEVAL exhausted before SA proposal"; break;
        case StopReason::Done:           reason = "loop completed normally"; break;
      }
      std::ostringstream oss;
      oss << "ANNEAL_PT stopped: " << reason
          << " after " << spent_report << " host eval(s)"
          << " (" << epoch_count << " epoch(s), "
          << exchange_rounds << " exchange round(s))";
      HPOLog::instance().info(oss.str());
    }
  }

template<typename Vec, typename Runner>
  void run_anneal_pt(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_pt_impl<Vec>(sp, host);
  }

}  // namespace tadah::hpo
