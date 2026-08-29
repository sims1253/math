#ifndef STAN_MATH_PRIM_PROB_NEG_BINOMIAL_2_LPMF_HPP
#define STAN_MATH_PRIM_PROB_NEG_BINOMIAL_2_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/multiply_log.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/select.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <cmath>


namespace stan {
namespace math {
namespace internal {

// Fused per-element worker for neg_binomial_2_lpmf (W-123 restructure):
// evaluates one element's logp term, location partial, and precision
// partial in stock's per-element order. always_inline: the pass loop must
// not pay a per-element call (GCC otherwise outlines a capturing lambda).
// phi_scalar = phi_val is an arithmetic scalar (broadcast); then log_phi
// is the precomputed scalar log(phi) and digamma_phi_scalar the one
// per-call digamma(phi).
template <bool include_precision, bool include_location, bool mu_autodiff,
          bool phi_autodiff, bool phi_scalar, typename TN, typename TMU,
          typename TPHI, typename TLOGPHI>
__attribute__((always_inline)) inline double nb2_element_work(
    Eigen::Index i, const TN& n_val, const TMU& mu_val, const TPHI& phi_val,
    const TLOGPHI& log_phi, double digamma_phi_scalar, double& p_mu_i,
    double& p_phi_i) {
  auto at = [](const auto& x, Eigen::Index k) {
    using X = std::decay_t<decltype(x)>;
    if constexpr (std::is_arithmetic_v<X>) {
      return x;
    } else {
      return x[k];
    }
  };
  const auto n_o = at(n_val, i);
  const double n_d = static_cast<double>(n_o);
  const double mu_v = at(mu_val, i);
  const double phi_v = at(phi_val, i);
  const double mu_plus_phi = mu_v + phi_v;
  const double log_mu_plus_phi = log(mu_plus_phi);
  const double n_plus_phi = n_d + phi_v;
  if constexpr (include_precision || include_location) {
    // logp_calc(): the subtract's fused product is pinned explicitly.
    // Stock's compiled expression evaluates fl(fl(-phi*log1p(mu/phi)) -
    // (n*lmp)_exact) — the n*lmp product is the FMA-fused one
    // (vfnmadd231sd, disassembly-verified in both the worktree and the
    // bundle environments at -mfma levels; mulsd+mulsd+subsd at -O2).
    // Writing the same expression in plain C++ leaves GCC free to fuse
    // EITHER product (TU-scheduling dependent — observed fusing
    // phi*log1p instead, differing in ~21% of elements by 1 ULP), so
    // the stock form is pinned: one explicit std::fma (the same
    // instruction) when FMA is enabled, the unfused shape otherwise.
    const double neg_phi_log1p = -phi_v * log1p(mu_v / phi_v);
#ifdef __FMA__
    const double calc = std::fma(-n_d, log_mu_plus_phi, neg_phi_log1p);
#else
    const double calc = neg_phi_log1p - n_d * log_mu_plus_phi;
#endif
    double term = 0;
    if constexpr (include_precision && include_location) {
      term = binomial_coefficient_log(n_plus_phi - 1.0, n_o)
             + multiply_log(n_o, mu_v) + calc;
    } else if constexpr (include_precision) {
      term = binomial_coefficient_log(n_plus_phi - 1.0, n_o) + calc;
    } else {
      term = multiply_log(n_o, mu_v) + calc;
    }
    if constexpr (mu_autodiff) {
      p_mu_i = n_d / mu_v - n_plus_phi / mu_plus_phi;
    }
    if constexpr (phi_autodiff) {
      const double digamma_n_plus_phi = digamma(n_plus_phi);
      double log_term;
      if (mu_v < phi_v) {
        log_term = log1p(-mu_v / mu_plus_phi);
      } else {
        if constexpr (phi_scalar) {
          log_term = log_phi - log_mu_plus_phi;
        } else {
          log_term = log_phi[i] - log_mu_plus_phi;
        }
      }
      double digamma_phi;
      if constexpr (phi_scalar) {
        digamma_phi = digamma_phi_scalar;
      } else {
        digamma_phi = digamma(at(phi_val, i));
      }
      p_phi_i = (mu_v - n_d) / mu_plus_phi + log_term - digamma_phi
                + digamma_n_plus_phi;
    }
    return term;
  } else {
    return 0;
  }
}

}  // namespace internal

// NegBinomial(n|mu, phi)  [mu >= 0; phi > 0;  n >= 0]
//
// The interior is a single fused scalar-sequential pass over the
// max-consistent size (W-123 restructure of the scalar-loop-era Eigen
// expression chain). Per-element operation order, the transcendental call
// sequence (binomial_coefficient_log -> lgamma, log, log1p, digamma,
// multiply_log), every check, and the left-fold accumulations (the logp
// sum; the scalar-edge partial sums) are preserved exactly as the stock
// expression evaluation produced them: Eigen's DefaultTraversal redux
// folds from element 0, so element 0 seeds the accumulators and the loop
// runs from 1. Redundant recomputations of identical values (log(mu+phi)
// evaluated twice per element, mu+phi and n+phi recomputed per pass, the
// discarded operand of the select) are computed once; each is the same
// libm/boost function on the same inputs, so values are bit-identical.
template <bool propto, typename T_n, typename T_location, typename T_precision,
          require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_n, T_location, T_precision>* = nullptr>
inline return_type_t<T_location, T_precision> neg_binomial_2_lpmf(
    const T_n& n, const T_location& mu, const T_precision& phi) {
  using T_partials_return = partials_return_t<T_n, T_location, T_precision>;
  using std::log;
  using T_n_ref = ref_type_t<T_n>;
  using T_mu_ref = ref_type_t<T_location>;
  using T_phi_ref = ref_type_t<T_precision>;
  static constexpr const char* function = "neg_binomial_2_lpmf";
  check_consistent_sizes(function, "Failures variable", n, "Location parameter",
                         mu, "Precision parameter", phi);

  T_n_ref n_ref = n;
  T_mu_ref mu_ref = mu;
  T_phi_ref phi_ref = phi;

  check_nonnegative(function, "Failures variable", n_ref);
  check_positive_finite(function, "Location parameter", mu_ref);
  check_positive_finite(function, "Precision parameter", phi_ref);

  if (size_zero(n, mu, phi)) {
    return 0.0;
  }
  if constexpr (!include_summand<propto, T_location, T_precision>::value) {
    return 0.0;
  }

  T_partials_return logp(0.0);
  auto ops_partials = make_partials_propagator(mu_ref, phi_ref);

  auto n_vec = as_array_or_scalar(as_column_vector_or_scalar(n_ref));
  auto mu_vec = as_array_or_scalar(as_column_vector_or_scalar(mu_ref));
  auto phi_vec = as_array_or_scalar(as_column_vector_or_scalar(phi_ref));
  decltype(auto) mu_val = value_of(mu_vec);
  decltype(auto) phi_val = value_of(phi_vec);
  auto n_val = value_of(n_vec);
  auto log_phi = log(phi_val);
  constexpr bool include_precision
      = include_summand<propto, T_precision>::value;
  constexpr bool include_location = include_summand<propto, T_location>::value;
  constexpr bool phi_is_scalar
      = std::is_arithmetic_v<std::decay_t<decltype(phi_val)>>;
  // digamma(phi) for a scalar phi is one call per lpmf evaluation (stock
  // evaluates it once at the precision-partials statement); a vector phi
  // evaluates it per element inside the pass, as stock's lazy expression
  // node did.
  double digamma_phi_scalar = 0.0;
  if constexpr (is_autodiff_v<T_precision> && phi_is_scalar) {
    digamma_phi_scalar = digamma(phi_val);
  }

  T_partials_return p_mu_i = 0.0;
  T_partials_return p_phi_i = 0.0;
  auto element_work = [&](Eigen::Index i) -> T_partials_return {
    return internal::nb2_element_work<include_precision, include_location,
                                      is_autodiff_v<T_location>,
                                      is_autodiff_v<T_precision>,
                                      phi_is_scalar>(
        i, n_val, mu_val, phi_val, log_phi, digamma_phi_scalar, p_mu_i,
        p_phi_i);
  };

  const Eigen::Index size = max_size(n_ref, mu_ref, phi_ref);
  // DefaultTraversal redux: element 0 seeds each accumulation, the loop
  // folds from 1 (stock's unpeeled prologue + loop shape).
  T_partials_return lp_acc = element_work(0);
  T_partials_return mu_acc = p_mu_i;
  T_partials_return phi_acc = p_phi_i;
  if constexpr (is_autodiff_v<T_location>) {
    if constexpr (!is_stan_scalar_v<std::decay_t<T_mu_ref>>) {
      partials<0>(ops_partials).coeffRef(0) = p_mu_i;
    }
  }
  if constexpr (is_autodiff_v<T_precision>) {
    if constexpr (!is_stan_scalar_v<std::decay_t<T_phi_ref>>) {
      partials<1>(ops_partials).coeffRef(0) = p_phi_i;
    }
  }
  for (Eigen::Index i = 1; i < size; ++i) {
    const T_partials_return term = element_work(i);
    lp_acc = lp_acc + term;
    if constexpr (is_autodiff_v<T_location>) {
      if constexpr (is_stan_scalar_v<std::decay_t<T_mu_ref>>) {
        mu_acc = mu_acc + p_mu_i;
      } else {
        partials<0>(ops_partials).coeffRef(i) = p_mu_i;
      }
    }
    if constexpr (is_autodiff_v<T_precision>) {
      if constexpr (is_stan_scalar_v<std::decay_t<T_phi_ref>>) {
        phi_acc = phi_acc + p_phi_i;
      } else {
        partials<1>(ops_partials).coeffRef(i) = p_phi_i;
      }
    }
  }
  logp += lp_acc;
  if constexpr (is_autodiff_v<T_location>) {
    if constexpr (is_stan_scalar_v<std::decay_t<T_mu_ref>>) {
      partials<0>(ops_partials)[0] = mu_acc;
    }
  }
  if constexpr (is_autodiff_v<T_precision>) {
    if constexpr (is_stan_scalar_v<std::decay_t<T_phi_ref>>) {
      partials<1>(ops_partials)[0] = phi_acc;
    }
  }
  return ops_partials.build(logp);
}

template <typename T_n, typename T_location, typename T_precision>
inline return_type_t<T_location, T_precision> neg_binomial_2_lpmf(
    const T_n& n, const T_location& mu, const T_precision& phi) {
  return neg_binomial_2_lpmf<false>(n, mu, phi);
}

}  // namespace math
}  // namespace stan
#endif
