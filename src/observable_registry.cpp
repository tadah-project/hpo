#include <tadah/hpo/observable_registry.h>

#include <tadah/core/aeds_type.h>

#include <cctype>
#include <charconv>
#include <stdexcept>
#include <string>

namespace tadah::hpo {

namespace {

inline std::string strip_outer_paren_and_dollar(std::string s)
{
    /* Trim whitespace, then strip a single leading `$(` … `)` immediate-
     * eval wrapper if present. Inner content is not re-trimmed of paren
     * pairs (a recursive descent into "(a+b)" is left to the parser). */
    auto trim = [](std::string& t) {
        const auto a = t.find_first_not_of(" \t");
        const auto b = t.find_last_not_of(" \t");
        if (a == std::string::npos) { t.clear(); return; }
        t = t.substr(a, b - a + 1);
    };
    trim(s);
    if (s.size() >= 3 && s[0] == '$' && s[1] == '(' && s.back() == ')') {
        s = s.substr(2, s.size() - 3);
        trim(s);
    }
    return s;
}

inline bool starts_with(const std::string& s, const char* p)
{
    const std::size_t n = std::char_traits<char>::length(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

/* Find the outermost +/- operator (lowest precedence, left-to-right
 * scan honouring parentheses). Returns std::string::npos if none. */
std::size_t find_outermost_pm(const std::string& s)
{
    int depth = 0;
    /* Skip a leading unary sign if any. */
    std::size_t start = 0;
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) start = 1;

    for (std::size_t i = start; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '(') ++depth;
        else if (c == ')') --depth;
        else if (depth == 0 && (c == '+' || c == '-')) {
            /* Reject sign-as-exponent (`1e-3`, `2E+4`) — the previous
             * char is a digit or '.' and the one before that is an
             * 'e'/'E'. */
            if (i >= 2) {
                const char p1 = s[i - 1];
                if ((p1 == 'e' || p1 == 'E')) {
                    /* The 'e' must be preceded by a digit or '.' to be
                     * an exponent marker (e.g. "1e-3" yes; "v_eatoms-3"
                     * no — the 's' before 'e' isn't digit/dot). */
                    if (i >= 2) {
                        const char p2 = s[i - 2];
                        if (std::isdigit(static_cast<unsigned char>(p2)) || p2 == '.') {
                            continue;
                        }
                    }
                }
            }
            return i;
        }
    }
    return std::string::npos;
}

/* Parse `lx/<num>`, `ly/<num>`, `lz/<num>`, returning axis ∈ {0,1,2}
 * and divisor on success. Plain "lx" / "ly" / "lz" → divisor 1. */
struct LatticeMatch {
    int    axis;
    double divisor;
};

bool try_match_lattice(const std::string& s, LatticeMatch& out)
{
    int axis = -1;
    if      (starts_with(s, "lx")) axis = 0;
    else if (starts_with(s, "ly")) axis = 1;
    else if (starts_with(s, "lz")) axis = 2;
    else return false;

    if (s.size() == 2) {
        out = LatticeMatch{axis, 1.0};
        return true;
    }
    if (s.size() <= 3 || s[2] != '/') return false;

    std::string num = s.substr(3);
    /* Strip surrounding parens once (e.g. "lx/(4.0)" tolerated). */
    if (num.size() >= 2 && num.front() == '(' && num.back() == ')')
        num = num.substr(1, num.size() - 2);

    /* std::stod accepts leading whitespace and trailing junk; require a
     * full-string parse so "lx/4.0e" or "lx/4.0+5" doesn't slip through. */
    try {
        std::size_t pos = 0;
        const double v = std::stod(num, &pos);
        if (pos != num.size()) return false;
        if (!(v > 0.0 || v < 0.0)) return false;     /* reject zero divisor */
        out = LatticeMatch{axis, v};
        return true;
    } catch (...) {
        return false;
    }
}

/* Sum descriptors over local atoms: out[c] = Σ_a aeds(c, a). */
void accumulate_aeds_columns(const tadah::core::aeds_type2& aeds,
                             std::size_t nlocal,
                             Eigen::VectorXd& out)
{
    const std::size_t n_desc = aeds.rows();
    out = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_desc));
    /* Guard against an upstream resize mismatch — descriptor matrix
     * columns are the storage allocation (atom->nmax), but logical
     * content tops out at nlocal. */
    const std::size_t cap = std::min<std::size_t>(nlocal, aeds.cols());
    for (std::size_t a = 0; a < cap; ++a) {
        for (std::size_t c = 0; c < n_desc; ++c) {
            out(static_cast<Eigen::Index>(c)) += aeds(c, a);
        }
    }
}

ObservableHandle make_zero_handle(const std::string& kind)
{
    ObservableHandle h;
    h.kind = kind;
    h.dg_dx_is_zero = true;
    h.dg_dx = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(3 * ctx.nlocal + ctx.box_dofs);
    };
    h.dg_dW = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(ctx.n_desc);
    };
    return h;
}

ObservableHandle make_cohesive_e_per_atom()
{
    ObservableHandle h;
    h.kind = "cohesive_e_per_atom";
    /* At the relaxed state x*, ∂E/∂x = -F = 0 to minimiser tolerance;
     * skip the Hessian solve. Phase 3.4 routes λ_k = 0 in that case. */
    h.dg_dx_is_zero = true;
    h.dg_dx = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(3 * ctx.nlocal + ctx.box_dofs);
    };
    h.dg_dW = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        if (!ctx.aeds || ctx.natoms <= 0) {
            throw std::runtime_error(
                "cohesive_e_per_atom: ObservableContext missing aeds or natoms");
        }
        accumulate_aeds_columns(*ctx.aeds, static_cast<std::size_t>(ctx.nlocal), out);
        out /= static_cast<double>(ctx.natoms);
    };
    return h;
}

ObservableHandle make_pe()
{
    ObservableHandle h;
    h.kind = "pe";
    h.dg_dx_is_zero = true;
    h.dg_dx = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(3 * ctx.nlocal + ctx.box_dofs);
    };
    h.dg_dW = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        if (!ctx.aeds) {
            throw std::runtime_error("pe: ObservableContext missing aeds");
        }
        accumulate_aeds_columns(*ctx.aeds, static_cast<std::size_t>(ctx.nlocal), out);
    };
    return h;
}

ObservableHandle make_lattice_constant(int axis, double divisor)
{
    ObservableHandle h;
    h.kind = "lattice_constant";
    /* lx (and friends) doesn't depend on atom positions; the strain
     * DOF carries it. dg_dx is identically zero on the atom block. */
    h.dg_dx_is_zero = false;
    h.dg_dx = [axis, divisor](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(3 * ctx.nlocal + ctx.box_dofs);
        if (ctx.box_dofs <= 0) return;
        const double L =
            (axis == 0) ? ctx.lx :
            (axis == 1) ? ctx.ly :
                          ctx.lz;
        const int row = 3 * ctx.nlocal;
        if (ctx.box_dofs == 1) {
            /* Iso strain ε: L_new = (1+ε)·L, so ∂L/∂ε = L. */
            out(row) = L / divisor;
        } else if (ctx.box_dofs == 6) {
            /* Voigt aniso: ε_xx → axis 0, ε_yy → 1, ε_zz → 2. */
            out(row + axis) = L / divisor;
        }
    };
    h.dg_dW = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(ctx.n_desc);
    };
    return h;
}

ObservableHandle make_vol()
{
    ObservableHandle h;
    h.kind = "vol";
    h.dg_dx_is_zero = false;
    h.dg_dx = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(3 * ctx.nlocal + ctx.box_dofs);
        if (ctx.box_dofs <= 0) return;
        const double V = ctx.lx * ctx.ly * ctx.lz;
        const int row = 3 * ctx.nlocal;
        if (ctx.box_dofs == 1) {
            /* Iso: V_new = (1+ε)³·V ⇒ ∂V/∂ε = 3·V at ε=0. */
            out(row) = 3.0 * V;
        } else if (ctx.box_dofs == 6) {
            /* Aniso diag (Voigt 0..2): ∂V/∂ε_αα = V (since V = lx·ly·lz
             * and ε_αα stretches one edge by V/L_α · L_α = V). Shear
             * Voigt modes (3..5) preserve V to leading order. */
            out(row + 0) = V;
            out(row + 1) = V;
            out(row + 2) = V;
        }
    };
    h.dg_dW = [](const ObservableContext& ctx, Eigen::VectorXd& out) {
        out = Eigen::VectorXd::Zero(ctx.n_desc);
    };
    return h;
}

ObservableHandle make_linear_combo(ObservableHandle lhs, ObservableHandle rhs,
                                   double sign_rhs)
{
    ObservableHandle h;
    h.kind = lhs.kind + (sign_rhs > 0 ? "+" : "-") + rhs.kind;
    h.dg_dx_is_zero = lhs.dg_dx_is_zero && rhs.dg_dx_is_zero;
    /* Capture by value so the combo owns its leaves regardless of
     * try_register-stack lifetime. */
    h.dg_dx = [lhs, rhs, sign_rhs](const ObservableContext& ctx,
                                    Eigen::VectorXd& out) {
        Eigen::VectorXd ol, orv;
        lhs.dg_dx(ctx, ol);
        rhs.dg_dx(ctx, orv);
        out = ol + sign_rhs * orv;
    };
    h.dg_dW = [lhs, rhs, sign_rhs](const ObservableContext& ctx,
                                    Eigen::VectorXd& out) {
        Eigen::VectorXd ol, orv;
        lhs.dg_dW(ctx, ol);
        rhs.dg_dW(ctx, orv);
        out = ol + sign_rhs * orv;
    };
    return h;
}

bool try_register_primitive(const std::string& s, ObservableHandle& out)
{
    if (s == "c_eatoms/atoms") { out = make_cohesive_e_per_atom(); return true; }
    if (s == "pe")             { out = make_pe();                  return true; }
    if (s == "vol")            { out = make_vol();                 return true; }
    if (s == "atoms")          { out = make_zero_handle("atoms");  return true; }

    LatticeMatch lm{};
    if (try_match_lattice(s, lm)) {
        out = make_lattice_constant(lm.axis, lm.divisor);
        return true;
    }
    return false;
}

bool try_register_inner(const std::string& expr,
                        const VariableResolver& resolver,
                        ObservableHandle& out,
                        int depth);

bool try_register_inner(const std::string& expr,
                        const VariableResolver& resolver,
                        ObservableHandle& out,
                        int depth)
{
    if (depth > 16) return false;  /* recursion guard against cycles */

    std::string s = strip_outer_paren_and_dollar(expr);
    if (s.empty()) return false;

    /* Strip a single leading unary sign — `(-pe)` would otherwise stay
     * parenthesised; the strip above already handled `$(…)`. */
    /* No special handling here; binary -/+ at offset 0 falls through. */

    /* 1. Linear combination — split on the outermost +/-. */
    const std::size_t op = find_outermost_pm(s);
    if (op != std::string::npos) {
        const std::string lhs_s = s.substr(0, op);
        const std::string rhs_s = s.substr(op + 1);
        const double sign = (s[op] == '+') ? +1.0 : -1.0;

        ObservableHandle hl, hr;
        if (!try_register_inner(lhs_s, resolver, hl, depth + 1)) return false;
        if (!try_register_inner(rhs_s, resolver, hr, depth + 1)) return false;
        out = make_linear_combo(hl, hr, sign);
        return true;
    }

    /* 2. v_<name> variable reference — defer to the host resolver. */
    if (s.size() > 2 && s[0] == 'v' && s[1] == '_') {
        if (!resolver) return false;
        const std::string vname = s.substr(2);
        const std::string vexpr = resolver(vname);
        if (vexpr.empty()) return false;
        /* Don't reuse the same name twice in the recursion frame —
         * caller-side resolver is responsible for cycle prevention. */
        return try_register_inner(vexpr, resolver, out, depth + 1);
    }

    /* 3. Primitive forms. */
    return try_register_primitive(s, out);
}

}  // anonymous namespace

bool try_register(const std::string& expr,
                  const VariableResolver& resolver,
                  ObservableHandle& out)
{
    return try_register_inner(expr, resolver, out, 0);
}

bool try_register(const std::string& expr, ObservableHandle& out)
{
    return try_register_inner(expr, VariableResolver{}, out, 0);
}

}  // namespace tadah::hpo
