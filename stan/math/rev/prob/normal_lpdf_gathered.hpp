#ifndef STAN_MATH_REV_PROB_NORMAL_LPDF_GATHERED_HPP
#define STAN_MATH_REV_PROB_NORMAL_LPDF_GATHERED_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/inv.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta/is_rev_matrix.hpp>
#include <cmath>
#include <vector>

namespace stan {
namespace math {

namespace internal {

/** \ingroup prob_dists
 * Shared core of the gathered normal likelihoods (not part of the public
 * API). Given the already-gathered linear predictor values `mu_val` (built
 * by the public entry points with the per-element op order of the generated
 * loop), computes one log-density term per observation with EXACTLY the
 * floating-point operation order of the SCALAR instantiation
 * `normal_lpdf<propto>(double, var, var)` that the stanc-generated loop
 * calls per element, and installs ONE reverse-mode callback that performs
 * the same adjoint accumulation, in the same (reverse-n) order, as the
 * per-element ops-partials callbacks of that loop.
 *
 * The returned vector holds one `var` per observation (the loop's per-term
 * `lp_accum__.add(t_n)` targets); each wraps a no-chain `vari_value<double>`
 * storing the term value and, at gradient time, the term's adjoint. This
 * keeps the model-side accumulation schedule (stan::math::accumulator's
 * chunked buffer) bit-identical to the stock loop.
 */
template <bool propto, bool HasBeta, typename T_alpha, typename T_beta,
          typename T_sigma>
std::vector<var> normal_lpdf_gathered_impl(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& y_d,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& mu_val,
    const arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>>& ii_arena,
    const arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>>& x_arena,
    const arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>>& ii2_arena,
    const T_alpha& alpha, const T_beta& beta, const T_sigma& sigma) {
  static constexpr const char* function = "normal_lpdf_gathered";
  const Eigen::Index n_obs = y_d.size();
  const double sigma_val = value_of(sigma);
  check_positive(function, "Scale parameter", sigma_val);
  if (unlikely(n_obs == 0)) {
    return std::vector<var>{};
  }

  // Adjoint routes. No gathered Matrix<var> is ever built: Matrix<var>
  // operands get one vari* per coefficient (the loop's mu elements alias
  // exactly these varis), var_value<> (SoA) operands keep their single
  // matrix vari.
  using vari_vec = Eigen::Matrix<vari*, Eigen::Dynamic, 1>;
  using soa_vec_vari = vari_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
  [[maybe_unused]] arena_t<vari_vec> alpha_vi(0);
  [[maybe_unused]] arena_t<vari_vec> beta_vi(0);
  [[maybe_unused]] soa_vec_vari* alpha_soa = nullptr;
  [[maybe_unused]] soa_vec_vari* beta_soa = nullptr;
  if constexpr (is_var_v<std::decay_t<T_alpha>>) {
    alpha_soa = alpha.vi_;
  } else {
    alpha_vi = arena_t<vari_vec>(alpha.size());
    for (Eigen::Index j = 0; j < alpha.size(); ++j) {
      alpha_vi.coeffRef(j) = alpha.coeff(j).vi_;
    }
  }
  if constexpr (HasBeta) {
    if constexpr (is_var_v<std::decay_t<T_beta>>) {
      beta_soa = beta.vi_;
    } else {
      beta_vi = arena_t<vari_vec>(beta.size());
      for (Eigen::Index j = 0; j < beta.size(); ++j) {
        beta_vi.coeffRef(j) = beta.coeff(j).vi_;
      }
    }
  }
  [[maybe_unused]] vari* sigma_vi = nullptr;
  if constexpr (is_var_v<std::decay_t<T_sigma>>) {
    sigma_vi = sigma.vi_;
  }

  // Scalar normal_lpdf computes these once per call; the values are
  // deterministic in sigma_val, so computing them once here and reusing the
  // double is bit-identical.
  const double inv_sigma_val = inv(sigma_val);
  const double log_sigma_val = [sigma_val]() {
    if constexpr (include_summand<propto, T_sigma>::value) {
      return math::log(sigma_val);
    } else {
      return 0.0;
    }
  }();

  // Per-element terms (stock op order: (y - mu) * inv_sigma, squared,
  // -0.5 * sum(...) for the scalar, then the constant additions in stock's
  // statement order) and the edge partials stock stores at forward time.
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> d_mu(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> d_sigma(n_obs);
  arena_t<vari_vec> term_vi(n_obs);
  std::vector<var> terms;
  terms.reserve(n_obs);
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    const double y_scaled = (y_d.coeff(k) - mu_val.coeff(k)) * inv_sigma_val;
    const double y_scaled_sq = y_scaled * y_scaled;
    double lp_k = -0.5 * y_scaled_sq;
    if constexpr (include_summand<propto>::value) {
      lp_k += NEG_LOG_SQRT_TWO_PI;
    }
    if constexpr (include_summand<propto, T_sigma>::value) {
      lp_k -= log_sigma_val;
    }
    terms.emplace_back(lp_k);
    term_vi.coeffRef(k) = terms.back().vi_;
    d_mu.coeffRef(k) = inv_sigma_val * y_scaled;
    d_sigma.coeffRef(k) = inv_sigma_val * y_scaled_sq - inv_sigma_val;
  }

  // ONE reverse callback replacing the per-element ops-partials callbacks
  // (and, for HasBeta, the per-element multiply/add callbacks). grad() runs
  // the stack in reverse creation order, so the loop's per-element varis
  // accumulate into the shared sigma vari and the indexed alpha/beta varis
  // in REVERSE n order; the same order is reproduced here.
  reverse_pass_callback(
      [alpha_vi, beta_vi, alpha_soa, beta_soa, sigma_vi, d_mu, d_sigma,
       ii_arena, ii2_arena, x_arena, term_vi]() {
        const Eigen::Index n = d_mu.size();
        for (Eigen::Index k = n - 1; k >= 0; --k) {
          // the term's adjoint (set by the accumulation of the returned
          // terms upstream) and the scalar lpdf's mu-edge add
          const double m = term_vi.coeff(k)->adj_ * d_mu.coeff(k);
          const int ik = ii_arena.coeff(k);
          if constexpr (is_var_v<std::decay_t<T_alpha>>) {
            alpha_soa->adj_.coeffRef(ik) += m;
          } else {
            alpha_vi.coeff(ik)->adj_ += m;
          }
          if constexpr (HasBeta) {
            const int i2 = ii2_arena.coeff(k);
            if constexpr (is_var_v<std::decay_t<T_beta>>) {
              // SoA route: stock's rvalue on a var_value<> creates a read
              // vari whose callback adds the already-rounded m * x into the
              // matrix adjoint with a PLAIN add (no multiply to contract),
              // so the product is kept un-contracted here.
              volatile double mbx = m * x_arena.coeff(k);
              beta_soa->adj_.coeffRef(i2) += mbx;
            } else {
              // AoS route: stock's multiply node chain does
              // adj += (add-node adjoint) * x -- a pointer read-modify-write
              // the compiler FMA-contracts exactly like this statement
              // (verified by gate (a) and disassembly of
              // internal::multiply_vd_vari::chain).
              const double mbx = m * x_arena.coeff(k);
              beta_vi.coeff(i2)->adj_ += mbx;
            }
          }
          if constexpr (is_var_v<std::decay_t<T_sigma>>) {
            sigma_vi->adj_ += term_vi.coeff(k)->adj_ * d_sigma.coeff(k);
          }
        }
      });
  return terms;
}

}  // namespace internal

/** \ingroup prob_dists
 * Gathered normal likelihood, loop form: the per-element log densities of
 *
 *   y[n] ~ normal(alpha[ii[n]], sigma),  n = 1..N
 *
 * returned one `var` per observation, without materializing the gathered
 * mean vector. This is the pattern the Stan compiler emits for the
 * stereotyped likelihood loop
 *
 *   for (n in 1:N) {
 *     mu[n] = alpha[county_idx[n]];
 *     target += normal_lpdf(y[n] | mu[n], sigma);
 *   }
 *
 * Each returned term performs exactly the same floating-point operations,
 * in the same order, as the SCALAR `normal_lpdf<propto>(y_n, mu_n, sigma)`
 * call of that loop: `(y - mu) * inv(sigma)` with `inv(sigma) = 1/sigma`,
 * the square, `-0.5 *` of the scalar sum, then the constant terms in
 * stock's statement order (`+= NEG_LOG_SQRT_TWO_PI`, `-= log(sigma)` when
 * `propto` includes them). `inv(sigma)` and `log(sigma)` are computed once
 * and the doubles reused (their values are deterministic in sigma).
 *
 * The intent is for the caller to add the returned terms to the model's
 * accumulator one element at a time, exactly as the stock loop does; the
 * values are those of the stock terms, so any such accumulation is
 * bit-identical. The reverse pass is ONE callback performing the same
 * adjoint accumulation as the loop's per-element autodiff nodes, in the
 * same reverse-n order: alpha's (and sigma's) partials are added through
 * the index in the order grad() visits the loop's nodes.
 *
 * @tparam propto if `true`, drop constant terms.
 * @tparam T_y type of the random variable (data vector of doubles)
 * @tparam T_alpha type of the coefficient vector (`Matrix<var>` or
 * `var_value<Matrix<double,-1,1>>`)
 * @tparam T_ii type of the index vector (integer vector-like, 1-based)
 * @tparam T_sigma type of the scale (`var` or `double`)
 * @param y random variable, one entry per observation
 * @param alpha coefficient vector, indexed by `ii`
 * @param ii index for each observation (1-based)
 * @param sigma scale parameter (scalar)
 * @return one log-density term (var) per observation
 * @throw std::domain_error if sigma is not positive
 * @throw std::out_of_range if an index is out of range
 */
template <bool propto, typename T_y, typename T_alpha, typename T_ii,
          typename T_sigma,
          require_eigen_vt<std::is_arithmetic, T_y>* = nullptr,
          require_st_var<T_alpha>* = nullptr,
          require_vector_like_vt<std::is_integral, T_ii>* = nullptr,
          require_stan_scalar_t<T_sigma>* = nullptr>
inline std::vector<var> normal_lpdf_gathered(const T_y& y, const T_alpha& alpha,
                                             const T_ii& ii,
                                             const T_sigma& sigma) {
  static constexpr const char* function = "normal_lpdf_gathered";
  const Eigen::Index n_obs = stan::math::size(ii);
  check_size_match(function, "Random variable size", stan::math::size(y),
                   "Index vector size", n_obs);
  const Eigen::Matrix<double, Eigen::Dynamic, 1> y_d = value_of(y);
  const Eigen::Matrix<double, Eigen::Dynamic, 1> alpha_d = value_of(alpha);
  const Eigen::Index alpha_size = alpha.size();
  const int* ii_data = ii.data();
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> ii_arena(n_obs);
  Eigen::Matrix<double, Eigen::Dynamic, 1> mu_val(n_obs);
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    if (unlikely(ii_data[k] < 1 || ii_data[k] > alpha_size)) {
      check_range("vector[uni] indexing", "alpha", alpha_size, ii_data[k]);
    }
    const int ik = ii_data[k] - 1;
    ii_arena.coeffRef(k) = ik;
    mu_val.coeffRef(k) = alpha_d.coeff(ik);
  }
  return internal::normal_lpdf_gathered_impl<propto, false>(
      y_d, mu_val, ii_arena,
      arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>>(0), ii_arena, alpha,
      0.0, sigma);
}

/** \ingroup prob_dists
 * Gathered normal likelihood, loop form with a data slope: the
 * per-element log densities of
 *
 *   y[n] ~ normal(alpha[ii[n]] + x[n] * beta[ii2[n]], sigma),  n = 1..N
 *
 * returned one `var` per observation (see the (y, alpha, ii, sigma)
 * overload for the bit-identity contract). The linear predictor values are
 * assembled per element with the generated loop's op order: the
 * multiplication `x[n] * beta[ii2[n]]` first, then the addition of
 * `alpha[ii[n]]`. In the reverse pass, beta's adjoint contribution for
 * element n is the loop's two-step propagation
 * `(w * dlp/dmu_n) * x[n]` (the scalar lpdf's mu-edge add followed by the
 * multiply node's chain), in the same order.
 *
 * @tparam propto if `true`, drop constant terms.
 * @tparam T_y type of the random variable (data vector of doubles)
 * @tparam T_alpha type of the intercept vector (`Matrix<var>` or SoA)
 * @tparam T_ii type of the intercept index vector (1-based)
 * @tparam T_x type of the data vector x (vector of doubles)
 * @tparam T_beta type of the slope vector (`Matrix<var>` or SoA)
 * @tparam T_ii2 type of the slope index vector (1-based; may be the same
 * vector as `ii`, as in the varying-intercept-slope models)
 * @tparam T_sigma type of the scale (`var` or `double`)
 */
template <bool propto, typename T_y, typename T_alpha, typename T_ii,
          typename T_x, typename T_beta, typename T_ii2, typename T_sigma,
          require_eigen_vt<std::is_arithmetic, T_y>* = nullptr,
          require_st_var<T_alpha>* = nullptr,
          require_vector_like_vt<std::is_integral, T_ii>* = nullptr,
          require_eigen_vt<std::is_arithmetic, T_x>* = nullptr,
          require_st_var<T_beta>* = nullptr,
          require_vector_like_vt<std::is_integral, T_ii2>* = nullptr,
          require_stan_scalar_t<T_sigma>* = nullptr>
inline std::vector<var> normal_lpdf_gathered(
    const T_y& y, const T_alpha& alpha, const T_ii& ii, const T_x& x,
    const T_beta& beta, const T_ii2& ii2, const T_sigma& sigma) {
  static constexpr const char* function = "normal_lpdf_gathered";
  const Eigen::Index n_obs = stan::math::size(ii);
  check_size_match(function, "Random variable size", stan::math::size(y),
                   "Index vector size", n_obs);
  check_size_match(function, "Data vector size", stan::math::size(x),
                   "Index vector size", stan::math::size(ii2));
  const Eigen::Matrix<double, Eigen::Dynamic, 1> y_d = value_of(y);
  const Eigen::Matrix<double, Eigen::Dynamic, 1> alpha_d = value_of(alpha);
  const Eigen::Matrix<double, Eigen::Dynamic, 1> beta_d = value_of(beta);
  const Eigen::Index alpha_size = alpha.size();
  const Eigen::Index beta_size = beta.size();
  const int* ii_data = ii.data();
  const int* ii2_data = ii2.data();
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> ii_arena(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> ii2_arena(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_arena(n_obs);
  Eigen::Matrix<double, Eigen::Dynamic, 1> mu_val(n_obs);
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    if (unlikely(ii_data[k] < 1 || ii_data[k] > alpha_size)) {
      check_range("vector[uni] indexing", "alpha", alpha_size, ii_data[k]);
    }
    if (unlikely(ii2_data[k] < 1 || ii2_data[k] > beta_size)) {
      check_range("vector[uni] indexing", "beta", beta_size, ii2_data[k]);
    }
    const int ik = ii_data[k] - 1;
    const int i2 = ii2_data[k] - 1;
    ii_arena.coeffRef(k) = ik;
    ii2_arena.coeffRef(k) = i2;
    x_arena.coeffRef(k) = x.coeff(k);
    // stock: multiply(x[n], beta[ii2[n]]) evaluated first (its value is
    // stored to the multiply vari), then add(alpha[ii[n]], ...). The two
    // results round-trip through memory in stock, so no FMA contraction of
    // the multiply-add is possible there; the volatile forces the same
    // un-contracted schedule here.
    volatile double prod = x.coeff(k) * beta_d.coeff(i2);
    mu_val.coeffRef(k) = alpha_d.coeff(ik) + prod;
  }
  return internal::normal_lpdf_gathered_impl<propto, true>(
      y_d, mu_val, ii_arena, x_arena, ii2_arena, alpha, beta, sigma);
}

/** \ingroup prob_dists
 * Gathered normal likelihood (see the propto overload); drops constant
 * terms.
 */
template <typename T_y, typename T_alpha, typename T_ii, typename T_sigma,
          require_eigen_vt<std::is_arithmetic, T_y>* = nullptr,
          require_st_var<T_alpha>* = nullptr,
          require_vector_like_vt<std::is_integral, T_ii>* = nullptr,
          require_stan_scalar_t<T_sigma>* = nullptr>
inline std::vector<var> normal_lpdf_gathered(const T_y& y,
                                             const T_alpha& alpha,
                                             const T_ii& ii,
                                             const T_sigma& sigma) {
  return normal_lpdf_gathered<false>(y, alpha, ii, sigma);
}

/** \ingroup prob_dists
 * Gathered normal likelihood with data slope (see the propto overload);
 * drops constant terms.
 */
template <typename T_y, typename T_alpha, typename T_ii, typename T_x,
          typename T_beta, typename T_ii2, typename T_sigma,
          require_eigen_vt<std::is_arithmetic, T_y>* = nullptr,
          require_st_var<T_alpha>* = nullptr,
          require_vector_like_vt<std::is_integral, T_ii>* = nullptr,
          require_eigen_vt<std::is_arithmetic, T_x>* = nullptr,
          require_st_var<T_beta>* = nullptr,
          require_vector_like_vt<std::is_integral, T_ii2>* = nullptr,
          require_stan_scalar_t<T_sigma>* = nullptr>
inline std::vector<var> normal_lpdf_gathered(const T_y& y,
                                             const T_alpha& alpha,
                                             const T_ii& ii, const T_x& x,
                                             const T_beta& beta,
                                             const T_ii2& ii2,
                                             const T_sigma& sigma) {
  return normal_lpdf_gathered<false>(y, alpha, ii, x, beta, ii2, sigma);
}

}  // namespace math
}  // namespace stan
#endif
