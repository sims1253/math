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
  // digamma(phi) for a scalar phi is one call per lpmf evaluation (stock
  // evaluates it once at the precision-partials statement); a vector phi
  // evaluates it per element inside the pass below, as stock's lazy
  // expression node did.
  T_partials_return digamma_phi_scalar = 0.0;
  if constexpr (is_autodiff_v<T_precision>
                && std::is_arithmetic_v<std::decay_t<decltype(phi_val)>>) {
    digamma_phi_scalar = digamma(phi_val);
  }
  // element accessor: Eigen containers index, arithmetic scalars broadcast
  auto elem = [](const auto& x, Eigen::Index i) {
    using X = std::decay_t<decltype(x)>;
    if constexpr (std::is_arithmetic_v<X>) {
      return x;
    } else {
      return x[i];
    }
  };
  auto log_phi_at = [&](Eigen::Index i) -> T_partials_return {
    if constexpr (std::is_arithmetic_v<std::decay_t<decltype(log_phi)>>) {
      return log_phi;
    } else {
      return static_cast<T_partials_return>(log_phi[i]);
    }
  };
  auto digamma_phi_at = [&](Eigen::Index i) -> T_partials_return {
    if constexpr (std::is_arithmetic_v<std::decay_t<decltype(phi_val)>>) {
      return digamma_phi_scalar;
    } else {
      return digamma(elem(phi_val, i));
    }
  };

  T_partials_return p_mu_i = 0.0;
  T_partials_return p_phi_i = 0.0;
  // One fused scalar-sequential pass. Per element this evaluates, in stock
  // order: the logp term (binomial_coefficient_log(n+phi-1, n) +
  // multiply_log(n, mu) + (-phi)*log1p(mu/phi) - n*log(mu+phi)), then the
  // location partial (n/mu - (n+phi)/(mu+phi)), then the precision partial
  // ((mu-n)/(mu+phi) + select(mu<phi, log1p(-mu/(mu+phi)), log(phi) -
  // log(mu+phi)) - digamma(phi) + digamma(n+phi)).
  auto element_work = [&](Eigen::Index i) -> T_partials_return {
    const auto n_o = elem(n_val, i);
    const T_partials_return n_d = static_cast<T_partials_return>(n_o);
    const T_partials_return mu_v = elem(mu_val, i);
    const T_partials_return phi_v = elem(phi_val, i);
    const T_partials_return mu_plus_phi = mu_v + phi_v;
    const T_partials_return log_mu_plus_phi = log(mu_plus_phi);
    const T_partials_return n_plus_phi = n_d + phi_v;
    if constexpr (include_precision || include_location) {
      // logp_calc(), with the (-phi)*log1p product in its own statement so
      // that the only FMA-fusable product at -mfma levels is n*log(mu+phi)
      // — the product stock's compiled expression fuses (vfnmadd231sd:
      // fl(fl(-phi*log1p) - (n*lmp)_exact), verified in stock disassembly
      // of both the worktree and bundle environments; at -O2 the shape is
      // mulsd+mulsd+subsd in both).
      const T_partials_return neg_phi_log1p
          = -phi_v * log1p(mu_v / phi_v);
      const T_partials_return calc
          = neg_phi_log1p - n_d * log_mu_plus_phi;
      T_partials_return term = 0;
      if constexpr (include_precision && include_location) {
        term = binomial_coefficient_log(n_plus_phi - 1.0, n_o)
               + multiply_log(n_o, mu_v) + calc;
      } else if constexpr (include_precision) {
        term = binomial_coefficient_log(n_plus_phi - 1.0, n_o) + calc;
      } else {
        term = multiply_log(n_o, mu_v) + calc;
      }
      if constexpr (is_autodiff_v<T_location>) {
        p_mu_i = n_d / mu_v - n_plus_phi / mu_plus_phi;
      }
      if constexpr (is_autodiff_v<T_precision>) {
        const T_partials_return digamma_n_plus_phi = digamma(n_plus_phi);
        T_partials_return log_term;
        if (mu_v < phi_v) {
          log_term = log1p(-mu_v / mu_plus_phi);
        } else {
          log_term = log_phi_at(i) - log_mu_plus_phi;
        }
        p_phi_i = (mu_v - n_d) / mu_plus_phi + log_term
                  - digamma_phi_at(i) + digamma_n_plus_phi;
      }
      return term;
    } else {
      return 0;
    }
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
