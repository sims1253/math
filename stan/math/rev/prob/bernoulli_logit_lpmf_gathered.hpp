#ifndef STAN_MATH_REV_PROB_BERNOULLI_LOGIT_LPMF_GATHERED_HPP
#define STAN_MATH_REV_PROB_BERNOULLI_LOGIT_LPMF_GATHERED_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/as_array_or_scalar.hpp>
#include <stan/math/prim/fun/as_value_column_array_or_scalar.hpp>
#include <stan/math/prim/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/log1p.hpp>
#include <stan/math/prim/fun/promote_scalar.hpp>
#include <stan/math/prim/fun/sum.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta/is_rev_matrix.hpp>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Gathered 2PL likelihood: the log PMF of the logit-parametrized Bernoulli
 * distribution over an index-assembled linear predictor
 *
 *   eta[n] = alpha[ii[n]] * (theta[jj[n]] - beta[ii[n]])
 *
 * without materializing any gathered `Matrix<var>`. This is the pattern the
 * Stan compiler emits for the IRT likelihood
 * `y ~ bernoulli_logit(alpha[ii] .* (theta[jj] - beta[ii]))`, which
 * `bernoulli_logit_glm_lpmf` cannot express (its predictor must be linear in
 * a dense design matrix; this one is bilinear in `(alpha, theta)`).
 *
 * The value path performs exactly the same floating point operations, in the
 * same per-element order, as the composed stock expression
 * `bernoulli_logit_lpmf(y, elt_multiply(rvalue(alpha, index_multi(ii)),
 * subtract(rvalue(theta, index_multi(jj)), rvalue(beta, index_multi(ii)))))`:
 * per element one subtraction (`theta - beta`) then one multiplication
 * (`alpha * ...`), followed by the `bernoulli_logit_lpmf` interior verbatim
 * (signs, `ntheta`, `exp(-ntheta)`, the two nested `Select` trees for the
 * value and the partials, Eigen's `sum` over the same expression types). The
 * reverse pass is ONE callback vari that performs the same scatter-adds, in
 * the same k order, that stock's elt_multiply / subtract callbacks perform
 * through their aliased gathered records:
 *
 *   alpha[ii[k]]_adj += (theta[jj[k]] - beta[ii[k]]) * (w * dtheta[k])
 *   theta[jj[k]]_adj +=  alpha[ii[k]]              * (w * dtheta[k])
 *   beta[ii[k]]_adj  -=  alpha[ii[k]]              * (w * dtheta[k])
 *
 * with `w` the adjoint of the returned log probability and `dtheta` the
 * elementwise partial of `bernoulli_logit_lpmf` (including its branch
 * behavior at `|ntheta| > 20`). Values and every gradient component are
 * bit-identical to the composed stock path.
 *
 * @tparam propto if `true`, normalize out constant terms (there are none for
 * this distribution; the flag is kept for drop-in symmetry with
 * `bernoulli_logit_lpmf`).
 * @tparam T_n type of the random variable (integer vector-like)
 * @tparam T_theta type of the ability vector (var vector: `Matrix<var>` or
 * `var_value<Matrix<double>>`)
 * @tparam T_jj type of the person index vector (integer vector-like)
 * @tparam T_alpha type of the discrimination vector (vector of vars)
 * @tparam T_beta type of the difficulty vector (vector of vars)
 * @tparam T_ii type of the item index vector (integer vector-like)
 * @param n random variable (0 or 1), one entry per observation
 * @param theta ability vector, indexed by `jj`
 * @param jj person index for each observation (1-based)
 * @param alpha discrimination vector, indexed by `ii`
 * @param beta difficulty vector, indexed by `ii`
 * @param ii item index for each observation (1-based)
 * @return var holding the log probability mass
 * @throw std::domain_error if any assembled predictor value is NaN
 * @throw std::invalid_argument if the container sizes mismatch
 * @throw std::out_of_range if an index is out of range
 */
template <
    bool propto, typename T_n, typename T_theta, typename T_jj,
    typename T_alpha, typename T_beta, typename T_ii,
    require_all_st_var<T_theta, T_alpha, T_beta>* = nullptr,
    require_vector_like_vt<std::is_integral, T_n>* = nullptr,
    require_vector_like_vt<std::is_integral, T_jj>* = nullptr,
    require_vector_like_vt<std::is_integral, T_ii>* = nullptr>
inline var bernoulli_logit_lpmf_gathered(const T_n& n, const T_theta& theta,
                                         const T_jj& jj, const T_alpha& alpha,
                                         const T_beta& beta, const T_ii& ii) {
  using T_partials_array = Eigen::Array<double, Eigen::Dynamic, 1>;
  using std::exp;
  static constexpr const char* function = "bernoulli_logit_lpmf_gathered";

  const Eigen::Index n_obs = ii.size();
  check_size_match(function, "Person index vector size", jj.size(),
                   "Item index vector size", n_obs);
  if (unlikely(n_obs == 0 || size_zero(n))) {
    return var(0.0);
  }
  check_size_match(function, "Random variable size", stan::math::size(n),
                   "Item index vector size", n_obs);
  check_bounded(function, "n", n, 0, 1);

  // Coefficient values (forward pass only) and adjoint routes. No gathered
  // Matrix<var> is ever built: Matrix<var> operands get one vari* per
  // element (their gathered records alias these varis), var_value<>
  // (SoA) operands keep just their single matrix vari; observations cost
  // two doubles and two ints each.
  const Eigen::Matrix<double, Eigen::Dynamic, 1> theta_d = value_of(theta);
  const Eigen::Matrix<double, Eigen::Dynamic, 1> alpha_d = value_of(alpha);
  const Eigen::Matrix<double, Eigen::Dynamic, 1> beta_d = value_of(beta);
  using vari_vec = Eigen::Matrix<vari*, Eigen::Dynamic, 1>;
  using soa_vec_vari = vari_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
  [[maybe_unused]] arena_t<vari_vec> theta_vi(0);
  [[maybe_unused]] arena_t<vari_vec> alpha_vi(0);
  [[maybe_unused]] arena_t<vari_vec> beta_vi(0);
  [[maybe_unused]] soa_vec_vari* theta_soa = nullptr;
  [[maybe_unused]] soa_vec_vari* alpha_soa = nullptr;
  [[maybe_unused]] soa_vec_vari* beta_soa = nullptr;
  if constexpr (is_var_v<std::decay_t<T_theta>>) {
    theta_soa = theta.vi_;
  } else {
    theta_vi = arena_t<vari_vec>(theta_d.size());
    for (Eigen::Index j = 0; j < theta_d.size(); ++j) {
      theta_vi.coeffRef(j) = theta.coeff(j).vi_;
    }
  }
  if constexpr (is_var_v<std::decay_t<T_alpha>>) {
    alpha_soa = alpha.vi_;
  } else {
    alpha_vi = arena_t<vari_vec>(alpha_d.size());
    for (Eigen::Index i = 0; i < alpha_d.size(); ++i) {
      alpha_vi.coeffRef(i) = alpha.coeff(i).vi_;
    }
  }
  if constexpr (is_var_v<std::decay_t<T_beta>>) {
    beta_soa = beta.vi_;
  } else {
    beta_vi = arena_t<vari_vec>(beta_d.size());
    for (Eigen::Index i = 0; i < beta_d.size(); ++i) {
      beta_vi.coeffRef(i) = beta.coeff(i).vi_;
    }
  }
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> sub_val(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> a_val(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> ii_arena(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> jj_arena(n_obs);

  const Eigen::Index theta_size = theta.size();
  const Eigen::Index alpha_size = alpha.size();
  const Eigen::Index beta_size = beta.size();
  const int* ii_data = as_array_or_scalar(ii).data();
  const int* jj_data = as_array_or_scalar(jj).data();
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    const int ik = ii_data[k] - 1;
    const int jk = jj_data[k] - 1;
    if (unlikely(ik < 0 || ik >= alpha_size)) {
      check_range("vector[multi] indexing", "alpha", alpha_size, ii_data[k]);
    }
    if (unlikely(ik >= beta_size)) {
      check_range("vector[multi] indexing", "beta", beta_size, ii_data[k]);
    }
    if (unlikely(jk < 0 || jk >= theta_size)) {
      check_range("vector[multi] indexing", "theta", theta_size, jj_data[k]);
    }
    ii_arena.coeffRef(k) = ik;
    jj_arena.coeffRef(k) = jk;
    // stock: subtract evaluates theta[jj] - beta[ii] elementwise ...
    sub_val.coeffRef(k) = theta_d.coeff(jk) - beta_d.coeff(ik);
    // ... and elt_multiply multiplies it by alpha[ii] elementwise.
    a_val.coeffRef(k) = alpha_d.coeff(ik);
  }

  // The linear predictor values, in stock's operand order.
  T_partials_array eta(n_obs);
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    eta.coeffRef(k) = a_val.coeff(k) * sub_val.coeff(k);
  }
  check_not_nan(function, "Logit transformed probability parameter", eta);
  if constexpr (!include_summand<propto, T_theta>::value) {
    return var(0.0);
  }

  // ---- bernoulli_logit_lpmf interior, verbatim (its double-partial
  // instantiation), over the same expression types as the composed call. ----
  const auto& n_col = as_column_vector_or_scalar(n);
  const auto& n_double = value_of_rec(n_col);
  auto signs = to_ref(2 * as_array_or_scalar(n_double) - 1);
  T_partials_array ntheta = signs * eta;

  T_partials_array exp_m_ntheta = exp(-ntheta);
  static constexpr double cutoff = 20.0;
  double logp = sum(
      (ntheta > cutoff)
          .select(-exp_m_ntheta,
                  (ntheta < -cutoff).select(ntheta, -log1p(exp_m_ntheta))));

  arena_t<T_partials_array> dtheta =
      (ntheta > cutoff)
          .select(-exp_m_ntheta,
                  (ntheta >= -cutoff)
                      .select(promote_scalar<double>(
                                  signs * exp_m_ntheta / (exp_m_ntheta + 1)),
                              promote_scalar<double>(signs)));
  // ---- end of interior ------------------------------------------------------

  return make_callback_var(
      logp,
      [theta_vi, alpha_vi, beta_vi, theta_soa, alpha_soa, beta_soa, sub_val,
       a_val, dtheta, ii_arena, jj_arena](const auto& vi) {
        const double w = vi.adj_;
        for (Eigen::Index k = 0; k < dtheta.size(); ++k) {
          const double e = w * dtheta.coeff(k);
          const int ik = ii_arena.coeff(k);
          const int jk = jj_arena.coeff(k);
          // elt_multiply callback: alpha gets sub_val * e, the subtraction
          // output gets a_val * e ...
          if constexpr (is_var_v<std::decay_t<T_alpha>>) {
            alpha_soa->adj_.coeffRef(ik) += sub_val.coeff(k) * e;
          } else {
            alpha_vi.coeff(ik)->adj_ += sub_val.coeff(k) * e;
          }
          // ... subtract callback: theta gets the subtraction output's
          // adjoint, beta gets its negation.
          if constexpr (is_var_v<std::decay_t<T_theta>>) {
            theta_soa->adj_.coeffRef(jk) += a_val.coeff(k) * e;
          } else {
            theta_vi.coeff(jk)->adj_ += a_val.coeff(k) * e;
          }
          if constexpr (is_var_v<std::decay_t<T_beta>>) {
            beta_soa->adj_.coeffRef(ik) -= a_val.coeff(k) * e;
          } else {
            beta_vi.coeff(ik)->adj_ -= a_val.coeff(k) * e;
          }
        }
      });
}

/** \ingroup prob_dists
 * Gathered 2PL likelihood (see the propto overload). Drops constant terms;
 * there are none for this distribution, so this matches
 * `bernoulli_logit_lpmf_gathered<true>` exactly.
 */
template <typename T_n, typename T_theta, typename T_jj, typename T_alpha,
          typename T_beta, typename T_ii,
          require_all_st_var<T_theta, T_alpha, T_beta>* = nullptr,
          require_vector_like_vt<std::is_integral, T_n>* = nullptr,
          require_vector_like_vt<std::is_integral, T_jj>* = nullptr,
          require_vector_like_vt<std::is_integral, T_ii>* = nullptr>
inline var bernoulli_logit_lpmf_gathered(const T_n& n, const T_theta& theta,
                                         const T_jj& jj, const T_alpha& alpha,
                                         const T_beta& beta, const T_ii& ii) {
  return bernoulli_logit_lpmf_gathered<true>(n, theta, jj, alpha, beta, ii);
}

}  // namespace math
}  // namespace stan
#endif
