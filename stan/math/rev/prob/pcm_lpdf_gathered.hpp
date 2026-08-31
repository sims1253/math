#ifndef STAN_MATH_REV_PROB_PCM_LPDF_GATHERED_HPP
#define STAN_MATH_REV_PROB_PCM_LPDF_GATHERED_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/softmax.hpp>
#include <stan/math/prim/prob/categorical_lpmf.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta/is_rev_matrix.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace stan {
namespace math {

namespace internal {

/** \ingroup prob_dists
 * Shared core of the gathered pcm likelihood (not part of the public API).
 * All index/route resolution and the per-observation buffers are prepared by
 * the public entry point; this function runs the value pass (stock op order)
 * and installs the ONE reverse callback.
 *
 * Bit-identity contract (against the composed stock path the stanc loop
 * generates), per observation:
 * - value: t = theta_j * alpha_i; c = [0; t - beta_steps] cumulated
 *   sequentially; softmax under Eigen's SCALAR traversal (stock feeds the
 *   prim softmax a non-packet-accessible `val()` view, so the interior is
 *   glibc exp per element, a sequential ascending sum, and a per-element
 *   divide); lp_n = log(p[y_n]).
 * - checks: `check_bounded` then `check_simplex` (categorical_lpmf's own
 *   order, same function strings) after the value pass.
 * - reverse: r = 0 except r[y] = e / p[y] (division); dot = sum p*r
 *   (sequential scalar redux); A = p * (r - dot); the cumulative_sum relay
 *   adj(u_k) = A_k + (A_{k+1} + ... + A_m) right-nested; adj_t =
 *   ((adj(u_0) + adj(u_1)) + ...) ascending (the subtract node's loop);
 *   theta/alpha increments = the multiply node's single-statement chains;
 *   beta increments = pure subtracts. Observations are visited in reverse-n
 *   order (the stock sweep order).
 */
template <bool propto, typename T_theta, typename T_alpha, typename T_beta>
std::vector<var> pcm_lpdf_gathered_impl(
    const arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>>& theta_d,
    const arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>>& alpha_d,
    const arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>>& beta_d,
    const arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>>& j_idx,
    const arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>>& i_idx,
    const arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>>& K_vec,
    const arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>>& y_vec,
    const arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>>& pos_idx,
    Eigen::Index total_cats, Eigen::Index max_K, const T_theta& theta,
    const T_alpha& alpha, const T_beta& beta) {
  using vari_vec = Eigen::Matrix<vari*, Eigen::Dynamic, 1>;
  using soa_vec_vari = vari_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
  [[maybe_unused]] vari_vec theta_vi(0);
  [[maybe_unused]] vari_vec alpha_vi(0);
  [[maybe_unused]] vari_vec beta_vi(0);
  [[maybe_unused]] soa_vec_vari* theta_soa = nullptr;
  [[maybe_unused]] soa_vec_vari* alpha_soa = nullptr;
  [[maybe_unused]] soa_vec_vari* beta_soa = nullptr;
  const Eigen::Index J = theta.size();
  const Eigen::Index I = alpha.size();
  const Eigen::Index B = beta.size();
  if constexpr (is_var_v<std::decay_t<T_theta>>) {
    theta_soa = theta.vi_;
  } else {
    theta_vi = vari_vec(J);
    for (Eigen::Index j = 0; j < J; ++j) {
      theta_vi.coeffRef(j) = theta.coeff(j).vi_;
    }
  }
  if constexpr (is_var_v<std::decay_t<T_alpha>>) {
    alpha_soa = alpha.vi_;
  } else {
    alpha_vi = vari_vec(I);
    for (Eigen::Index i = 0; i < I; ++i) {
      alpha_vi.coeffRef(i) = alpha.coeff(i).vi_;
    }
  }
  if constexpr (is_var_v<std::decay_t<T_beta>>) {
    beta_soa = beta.vi_;
  } else {
    beta_vi = vari_vec(B);
    for (Eigen::Index k = 0; k < B; ++k) {
      beta_vi.coeffRef(k) = beta.coeff(k).vi_;
    }
  }

  const Eigen::Index n_obs = K_vec.size();
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> p_all(total_cats);
  arena_t<vari_vec> term_vi(n_obs);
  std::vector<var> terms;
  terms.reserve(n_obs);

  Eigen::Index cat_off = 0;
  for (Eigen::Index n = 0; n < n_obs; ++n) {
    const int K = K_vec.coeff(n);
    const int y_n = y_vec.coeff(n);
    const Eigen::Index item = i_idx.coeff(n);
    // value pass (stock op order)
    const double t = theta_d.coeff(j_idx.coeff(n)) * alpha_d.coeff(item);
    Eigen::Matrix<double, Eigen::Dynamic, 1> c(K);
    c.coeffRef(0) = 0.0;
    for (int k = 1; k < K; ++k) {
      const double u = t - beta_d.coeff(pos_idx.coeff(item) + k - 1);
      c.coeffRef(k) = c.coeff(k - 1) + u;
    }
    // softmax through the SAME prim instantiation the stock rev softmax
    // calls: a val() view over an arena AoS var matrix (probe-verified
    // bit-identical to the composed stock path on every math stack; the
    // direct dense call is NOT -- Eigen's traversal of the interior
    // depends on the input expression type).
    arena_matrix<Eigen::Matrix<var, Eigen::Dynamic, 1>> c_var(K);
    for (int k = 0; k < K; ++k) {
      c_var.coeffRef(k) = var(c.coeff(k));
    }
    Eigen::Matrix<double, Eigen::Dynamic, 1> p = softmax(c_var.val());
    // stock checks (categorical_lpmf's order)
    check_bounded("categorical_lpmf", "Number of categories", y_n + 1, 1, K);
    check_simplex("categorical_lpmf", "Probabilities parameter", p);
    terms.emplace_back(std::log(p.coeff(y_n)));
    term_vi.coeffRef(n) = terms.back().vi_;
    for (int k = 0; k < K; ++k) {
      p_all.coeffRef(cat_off + k) = p.coeff(k);
    }
    cat_off += K;
  }

  // ONE reverse callback replacing every per-observation node of the stock
  // graph (log, softmax, cumulative_sum, append_row, subtract, multiply),
  // reproducing its adjoint accumulation in the same reverse-n order.
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> adj_u_buf(max_K);
  reverse_pass_callback([theta_vi, alpha_vi, beta_vi, theta_soa, alpha_soa,
                         beta_soa, j_idx, i_idx, K_vec, y_vec, pos_idx,
                         p_all, term_vi, theta_d, alpha_d, adj_u_buf]() mutable {
    const Eigen::Index n_obs = term_vi.size();
    Eigen::Index cat_off = 0;
    for (Eigen::Index n = 0; n < n_obs; ++n) {
      cat_off += K_vec.coeff(n);
    }
    for (Eigen::Index n = n_obs - 1; n >= 0; --n) {
      const int K = K_vec.coeff(n);
      cat_off -= K;
      const int y_n = y_vec.coeff(n);
      const Eigen::Index jn = j_idx.coeff(n);
      const Eigen::Index item = i_idx.coeff(n);
      const double* p = p_all.data() + cat_off;
      const double e = term_vi.coeff(n)->adj_;
      // log node: adjoint of the selected probability is a DIVISION
      const double r_y = e / p[y_n];
      // softmax node: dot = p . r (sequential scalar redux)
      double dot = 0.0;
      for (int k = 0; k < K; ++k) {
        double r_k = 0.0;
        if (k == y_n) {
          r_k = r_y;
        }
        dot = dot + p[k] * r_k;
      }
      // cumulative_sum relay (right-nested suffix accumulation of A) plus
      // the subtract node's ascending adjoint accumulation
      double suf = 0.0;
      double* adj_u_stack = adj_u_buf.data();
      for (int k = K - 1; k >= 1; --k) {
        double r_k = 0.0;
        if (k == y_n) {
          r_k = r_y;
        }
        const double A_k = p[k] * (r_k - dot);
        suf = A_k + suf;
        adj_u_stack[k - 1] = suf;
      }
      double adj_t = 0.0;
      for (int k = 0; k < K - 1; ++k) {
        adj_t = adj_t + adj_u_stack[k];
      }
      // multiply node: single-statement chains (FMA-contractible on the
      // AoS route, like stock's compiled op_vv chain; the SoA route keeps
      // the product un-contracted -- stock's per-read view node schedule)
      const double a_val = alpha_d.coeff(item);
      const double t_val = theta_d.coeff(jn);
      if constexpr (is_var_v<std::decay_t<T_theta>>) {
        volatile double inc = adj_t * a_val;
        theta_soa->adj_.coeffRef(jn) += inc;
      } else {
        theta_vi.coeff(jn)->adj_ += adj_t * a_val;
      }
      if constexpr (is_var_v<std::decay_t<T_alpha>>) {
        volatile double inc = adj_t * t_val;
        alpha_soa->adj_.coeffRef(item) += inc;
      } else {
        alpha_vi.coeff(item)->adj_ += adj_t * t_val;
      }
      // subtract node: pure subtracts into the beta slots
      for (int k = 0; k < K - 1; ++k) {
        const Eigen::Index slot = pos_idx.coeff(item) + k;
        if constexpr (is_var_v<std::decay_t<T_beta>>) {
          beta_soa->adj_.coeffRef(slot) -= adj_u_stack[k];
        } else {
          beta_vi.coeff(slot)->adj_ -= adj_u_stack[k];
        }
      }
    }
  });
  return terms;
}

}  // namespace internal

/** \ingroup prob_dists
 * Gathered partial-credit (PCM) likelihood, loop form: the per-observation
 * log probability masses of the stereotyped generated loop
 *
 *   for (n in 1:N)
 *     target += pcm(y[n], theta[jj[n]] * alpha[ii[n]],
 *                   segment(beta, pos[ii[n]], m[ii[n]]));
 *
 * with the user function
 *
 *   real pcm(int y, real theta, vector beta) {
 *     vector[rows(beta) + 1] unsummed
 *         = append_row(rep_vector(0.0, 1), theta - beta);
 *     probs = softmax(cumulative_sum(unsummed));
 *     return categorical_lpmf(y + 1 | probs);
 *   }
 *
 * returned one `var` per observation (the loop's per-term `lp_accum__.add`
 * targets), without materializing the per-observation autodiff nodes. The
 * caller adds the returned terms to the model's accumulator one element at
 * a time, exactly as the stock loop does.
 *
 * Bit-identity contract, operand routes, and check parity: see
 * `internal::pcm_lpdf_gathered_impl` (values, every gradient component,
 * throw set and messages are those of the composed stock path).
 *
 * @tparam propto present for family API consistency; the pcm interior has
 * no constant terms (both values behave identically).
 * @tparam T_theta person ability vector (`Matrix<var>`, `Eigen::Map`, or
 * `var_value<vector<double>>`)
 * @tparam T_alpha item discrimination vector (same routes)
 * @tparam T_beta concatenated item step parameter vector (same routes)
 * @param y category responses, `y[n]` in `0..m[ii[n]]` (0-based)
 * @param theta person abilities, indexed by `jj` (1-based)
 * @param jj person index per observation (1-based)
 * @param alpha item discriminations, indexed by `ii` (1-based)
 * @param ii item index per observation (1-based)
 * @param beta concatenated item step parameters (`sum(m)` entries)
 * @param pos first position of each item's steps in `beta` (1-based)
 * @param m number of step parameters per item (`m[i]` categories minus 1)
 * @return one log-probability term (var) per observation
 * @throw std::out_of_range if `jj[n]` or `ii[n]` is out of range
 * @throw std::domain_error if `y[n] + 1` is outside `1..m[ii[n]] + 1` or
 * the implied probabilities are not a simplex (non-finite parameters)
 */
template <bool propto, typename T_theta, typename T_alpha, typename T_beta,
          require_st_var<T_theta>* = nullptr,
          require_st_var<T_alpha>* = nullptr,
          require_st_var<T_beta>* = nullptr>
inline std::vector<var> pcm_lpdf_gathered(
    const std::vector<int>& y, const T_theta& theta,
    const std::vector<int>& jj, const T_alpha& alpha,
    const std::vector<int>& ii, const T_beta& beta,
    const std::vector<int>& pos, const std::vector<int>& m) {
  static constexpr const char* function = "pcm_lpdf_gathered";
  const Eigen::Index n_obs = jj.size();
  check_size_match(function, "Responses size", y.size(), "Person index size",
                   n_obs);
  check_size_match(function, "Responses size", y.size(), "Item index size",
                   static_cast<Eigen::Index>(ii.size()));
  const Eigen::Index J = theta.size();
  const Eigen::Index I = alpha.size();
  const Eigen::Index B = beta.size();
  if (unlikely(static_cast<Eigen::Index>(pos.size()) != I
               || static_cast<Eigen::Index>(m.size()) != I)) {
    std::invalid_argument e(
        "pcm_lpdf_gathered: pos/m must have one entry per item");
    throw e;
  }

  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> theta_d = value_of(theta);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> alpha_d = value_of(alpha);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> beta_d = value_of(beta);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> j_idx(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> i_idx(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> K_vec(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> y_vec(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> pos_idx(I);
  Eigen::Index total_cats = 0;
  Eigen::Index max_K = 1;
  for (Eigen::Index i = 0; i < I; ++i) {
    pos_idx.coeffRef(i) = pos[i] - 1;  // to 0-based
    if (pos[i] < 1 || pos[i] + m[i] - 1 > B) {
      std::invalid_argument e(
          "pcm_lpdf_gathered: item steps out of range of beta");
      throw e;
    }
    max_K = std::max(max_K, static_cast<Eigen::Index>(m[i]) + 1);
  }

  // Index resolution in the stock evaluation order: theta's rvalue (jj)
  // first, then alpha's rvalue (ii). Invalid entries throw here with
  // stock's `vector[uni] indexing` messages.
  for (Eigen::Index n = 0; n < n_obs; ++n) {
    const int jn = jj[n];
    if (unlikely(jn < 1 || jn > J)) {
      check_range("vector[uni] indexing", "theta", J, jn);
    }
    const int in = ii[n];
    if (unlikely(in < 1 || in > I)) {
      check_range("vector[uni] indexing", "alpha", I, in);
    }
    j_idx.coeffRef(n) = jn - 1;
    i_idx.coeffRef(n) = in - 1;
    const int K = m[in - 1] + 1;
    K_vec.coeffRef(n) = K;
    y_vec.coeffRef(n) = y[n];
    total_cats += K;
  }

  return internal::pcm_lpdf_gathered_impl<propto>(
      theta_d, alpha_d, beta_d, j_idx, i_idx, K_vec, y_vec, pos_idx,
      total_cats, max_K, theta, alpha, beta);
}

/** \ingroup prob_dists
 * Gathered pcm likelihood (see the propto overload); the pcm interior has
 * no constant terms, so this is identical to the `propto = true` form.
 */
template <typename T_theta, typename T_alpha, typename T_beta,
          require_st_var<T_theta>* = nullptr,
          require_st_var<T_alpha>* = nullptr,
          require_st_var<T_beta>* = nullptr>
inline std::vector<var> pcm_lpdf_gathered(
    const std::vector<int>& y, const T_theta& theta,
    const std::vector<int>& jj, const T_alpha& alpha,
    const std::vector<int>& ii, const T_beta& beta,
    const std::vector<int>& pos, const std::vector<int>& m) {
  return pcm_lpdf_gathered<false>(y, theta, jj, alpha, ii, beta, pos, m);
}

}  // namespace math
}  // namespace stan
#endif
