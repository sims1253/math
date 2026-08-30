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
 * through their aliased gathered records, AT THE MACHINE-CODE SCHEDULE
 * level (W-108.1): stock's elt_multiply chain accumulates a Matrix<var>
 * alpha with a fused multiply-add, computes the subtraction output's
 * increment as ONE rounded product, and subtract's chain applies that
 * product to theta with a pure add and to beta with a pure subtract:
 *
 *   alpha[ii[k]]_adj  += fma((theta[jj[k]] - beta[ii[k]]), (w*dtheta[k]),
 *                            alpha[ii[k]]_adj)          [AoS alpha, fused]
 *   inc                =  round(alpha[ii[k]] * (w*dtheta[k]))
 *   theta[jj[k]]_adj  += inc
 *   beta[ii[k]]_adj   -= inc
 *
 * (SoA `var_value<>` operands reach their adjoints through rvalue_varmat's
 * gather, whose scatter is a pure add, so every SoA-route increment is a
 * rounded product followed by an unfused add/sub.) `w` is the adjoint of the
 * returned log probability and `dtheta` the elementwise partial of
 * `bernoulli_logit_lpmf` (including its branch behavior at `|ntheta| > 20`).
 * Values and every gradient component are bit-identical to the composed
 * stock path in EVERY operand layout the stanc deserializer produces
 * (`Matrix<var>`, `var_value<Matrix<double>>`, and `Map<const Matrix<var>>`
 * - the default-level deserializer layout).
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
          // The adjoint arithmetic replicates the machine-code schedule of
          // the composed stock expression (W-108.1, decoded from the
          // elt_multiply / subtract reverse chains):
          //  - elt_multiply's chain accumulates a Matrix<var> alpha with a
          //    FUSED multiply-add (vfmadd213sd: alpha_adj += sub_val * e in
          //    one expression, one rounding of the sum);
          //  - the subtraction output's increment is ONE ROUNDED PRODUCT
          //    (the chain's fma into its zero-initialized record), which
          //    subtract's chain then applies to theta with a PURE ADD and to
          //    beta with a PURE SUB (vaddsd / vsubsd - two roundings total).
          //  - var_value<> (SoA) operands flow through rvalue_varmat's
          //    gather, whose scatter is a pure add, so their alpha increment
          //    is also a rounded product followed by an add.
          // The `volatile` barriers force the unfused forms (GCC's
          // fp-contract=fast re-fuses plain statement splits); keeping AoS
          // alpha's statement in one expression lets the compiler fuse it
          // exactly as stock's chain does.
          if constexpr (is_var_v<std::decay_t<T_alpha>>) {
            volatile const double ainc = sub_val.coeff(k) * e;
            alpha_soa->adj_.coeffRef(ik) += ainc;
          } else {
            alpha_vi.coeff(ik)->adj_ += sub_val.coeff(k) * e;
          }
          volatile const double inc = a_val.coeff(k) * e;
          if constexpr (is_var_v<std::decay_t<T_theta>>) {
            theta_soa->adj_.coeffRef(jk) += inc;
          } else {
            theta_vi.coeff(jk)->adj_ += inc;
          }
          if constexpr (is_var_v<std::decay_t<T_beta>>) {
            beta_soa->adj_.coeffRef(ik) -= inc;
          } else {
            beta_vi.coeff(ik)->adj_ -= inc;
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

// ---------------------------------------------------------------------------
// Additive multi-gather predictor (W-127): eta[n] = intercept +
// sum(data-product terms) + sum(gathered coefficient terms), the election88
// class `y ~ bernoulli_logit(beta[1] + beta[2]*black + ... + a[age] + ...)`.
// ---------------------------------------------------------------------------

/** \ingroup prob_dists
 * Leaf tag: a gathered coefficient term `coefs[idx[n]]` of the additive
 * predictor. `coefs` is a vector of vars (`Eigen::Matrix<var,-1,1>` /
 * `Map<const Matrix<var>>` for AoS layouts, `var_value<Matrix<double>>` for
 * the SoA deserializer layout); `idx` holds 1-based observation indices.
 * `name` is the Stan variable name used in range-check messages (parity with
 * `rvalue`'s "vector[uni] indexing" text on the composed path).
 */
template <typename T_vec>
struct gather_term {
  const char* name;
  const T_vec& coefs;
  const std::vector<int>& idx;
  gather_term(const char* name, const T_vec& coefs,
              const std::vector<int>& idx)
      : name(name), coefs(coefs), idx(idx) {}
};
template <typename T_vec>
gather_term(const char*, const T_vec&, const std::vector<int>&)
    -> gather_term<T_vec>;

/** \ingroup prob_dists
 * Leaf tag: a data-slope term `coef * xd[n]` of the additive predictor, with
 * `coef` a scalar var and `xd` a contiguous data vector
 * (`std::vector<double>` or an Eigen vector). This is the composed-path
 * `rvalue(beta, index_uni(k)) * rvalue(xd, index_uni(n))`.
 */
template <typename T_coef, typename T_d>
struct slope_term {
  const T_coef& coef;
  const T_d& xd;
  slope_term(const T_coef& coef, const T_d& xd) : coef(coef), xd(xd) {}
};
template <typename T_coef, typename T_d>
slope_term(const T_coef&, const T_d&) -> slope_term<T_coef, T_d>;

/** \ingroup prob_dists
 * Leaf tag: a chained data-slope term `(coef * xd1[n]) * xd2[n]` of the
 * additive predictor (the composed `beta[5] * female[i] * black[i]` form:
 * two rounded products, left-associated).
 */
template <typename T_coef, typename T_d1, typename T_d2>
struct slope2_term {
  const T_coef& coef;
  const T_d1& xd1;
  const T_d2& xd2;
  slope2_term(const T_coef& coef, const T_d1& xd1, const T_d2& xd2)
      : coef(coef), xd1(xd1), xd2(xd2) {}
};
template <typename T_coef, typename T_d1, typename T_d2>
slope2_term(const T_coef&, const T_d1&, const T_d2&)
    -> slope2_term<T_coef, T_d1, T_d2>;

namespace internal {

// Resolved (forward-value + reverse-route) form of a `gather_term`. The
// adjoint routes follow the composed path: SoA (`var_value<>`) coefficient
// vectors scatter into the matrix vari's adjoint array (the rvalue view's
// pure adds), AoS vectors into one `vari*` per coefficient element.
template <typename T_vec>
struct resolved_gather {
  static constexpr bool is_soa = is_var_v<std::decay_t<T_vec>>;
  using soa_vec_vari = vari_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
  using vari_vec = Eigen::Matrix<vari*, Eigen::Dynamic, 1>;

  const char* name_;
  Eigen::Index csize_;
  const int* idx1_;  // 1-based indices, observation-ordered (leaf data)
  Eigen::Matrix<double, Eigen::Dynamic, 1> vals_;
  [[maybe_unused]] soa_vec_vari* vi_soa_ = nullptr;
  [[maybe_unused]] arena_t<vari_vec> vi_aos_{0};

  template <typename T>
  explicit resolved_gather(const gather_term<T>& leaf)
      : name_(leaf.name),
        csize_(leaf.coefs.size()),
        idx1_(leaf.idx.data()),
        vals_(value_of(leaf.coefs)) {
    if constexpr (is_soa) {
      vi_soa_ = leaf.coefs.vi_;
    } else {
      vi_aos_ = arena_t<vari_vec>(csize_);
      for (Eigen::Index j = 0; j < csize_; ++j) {
        vi_aos_.coeffRef(j) = leaf.coefs.coeff(j).vi_;
      }
    }
  }

  // Stock per-element order: rvalue's check_range then the coefficient read.
  inline double fwd(Eigen::Index k) const {
    const int one_based = idx1_[k];
    if (unlikely(one_based < 1 || one_based > csize_)) {
      check_range("vector[uni] indexing", name_, csize_, one_based);
    }
    return vals_.coeff(one_based - 1);
  }

  inline void rev(Eigen::Index k, double e) const {
    const Eigen::Index ik = idx1_[k] - 1;
    if constexpr (is_soa) {
      vi_soa_->adj_.coeffRef(ik) += e;
    } else {
      vi_aos_.coeff(ik)->adj_ += e;
    }
  }
};

// Resolved `slope_term`: one scalar-var coefficient, one data vector.
struct resolved_slope {
  vari* coef_vi_;
  double coef_val_;
  const double* xd_;

  template <typename T_coef, typename T_d>
  explicit resolved_slope(const slope_term<T_coef, T_d>& leaf)
      : coef_vi_(leaf.coef.vi_),
        coef_val_(leaf.coef.vi_->val_),
        xd_(leaf.xd.data()) {}

  // Stock: multiply_vd_vari's vmulsd, one rounded product. The volatile
  // barrier materializes the rounding so the caller's accumulation add
  // cannot be contracted into an fma with this product (stock's adds are
  // pure vaddsd's over already-rounded products).
  inline double fwd(Eigen::Index k) const {
    volatile const double p = coef_val_ * xd_[k];
    return p;
  }

  // Stock: multiply_vd_vari::chain()'s vfmadd132sd (adj_ += adj_*bd_),
  // fused at the model flags.
  inline void rev(Eigen::Index k, double e) const { coef_vi_->adj_ += xd_[k] * e; }
};

// Resolved `slope2_term`: one scalar-var coefficient, two chained data
// vectors ((coef*xd1)*xd2).
struct resolved_slope2 {
  vari* coef_vi_;
  double coef_val_;
  const double* xd1_;
  const double* xd2_;

  template <typename T_coef, typename T_d1, typename T_d2>
  explicit resolved_slope2(const slope2_term<T_coef, T_d1, T_d2>& leaf)
      : coef_vi_(leaf.coef.vi_),
        coef_val_(leaf.coef.vi_->val_),
        xd1_(leaf.xd1.data()),
        xd2_(leaf.xd2.data()) {}

  // Stock: two vmulsd's, left-associated ((coef*xd1)*xd2), each rounded
  // (barriers block any contraction into the caller's adds or across the
  // two products).
  inline double fwd(Eigen::Index k) const {
    volatile const double p1 = coef_val_ * xd1_[k];
    volatile const double p2 = p1 * xd2_[k];
    return p2;
  }

  // Stock reverse (m2's chain then m1's chain): m2 applies a vfmadd into its
  // operand's zero-initialized adjoint (= one rounded product e*xd2), then
  // m1 applies a fused vfmadd of that product by xd1 into the coefficient
  // adjoint. The volatile barrier forces the intermediate rounding.
  // SPECIAL CASE xd1 == 1.0: stock's operator*(var, Arith) ALIASES its var
  // operand when the multiplier is exactly 1.0 (no vari is created), so the
  // composed path's m1 does not exist there and the increment is m2's
  // single fused product fma(e, xd2, adj) — one rounding of the product,
  // not two (the 1.0-multiplier class caught by the W-127 gate). The
  // volatile copy of `e` in the generic path stops GCC from CSE-ing the two
  // branches' e*xd2 products, which would leave the aliased branch's add
  // unfused (two roundings — the exact bug the gate caught in disassembly).
  inline void rev(Eigen::Index k, double e) const {
    if (xd1_[k] == 1.0) {
      coef_vi_->adj_ += xd2_[k] * e;
      return;
    }
    volatile const double ev = e;
    volatile const double t = ev * xd2_[k];
    coef_vi_->adj_ += xd1_[k] * t;
  }
};

// Resolve a leaf tag to its forward/reverse state.
template <typename T_vec>
inline auto resolve_leaf(const gather_term<T_vec>& leaf, Eigen::Index = 0) {
  static_assert(is_var_v<std::decay_t<T_vec>> ||
                    is_eigen_v<std::decay_t<T_vec>>,
                "gather_term coefficient vectors must be vectors of vars");
  return resolved_gather<T_vec>(leaf);
}

template <typename T_coef, typename T_d>
inline auto resolve_leaf(const slope_term<T_coef, T_d>& leaf, Eigen::Index = 0) {
  static_assert(is_var_v<std::decay_t<T_coef>>,
                "slope_term coefficients must be vars");
  return resolved_slope(leaf);
}

template <typename T_coef, typename T_d1, typename T_d2>
inline auto resolve_leaf(const slope2_term<T_coef, T_d1, T_d2>& leaf,
                         Eigen::Index = 0) {
  static_assert(is_var_v<std::decay_t<T_coef>>,
                "slope2_term coefficients must be vars");
  return resolved_slope2(leaf);
}

// Per-leaf size checks (the composed path's y_hat always matches n by
// construction, so these are API-level guards).
template <typename T_vec>
inline void check_leaf_size(const char* function, Eigen::Index n_obs,
                            const gather_term<T_vec>& leaf) {
  check_size_match(function, "Random variable size", n_obs,
                   "Index vector size",
                   static_cast<Eigen::Index>(leaf.idx.size()));
}

template <typename T_coef, typename T_d>
inline void check_leaf_size(const char* function, Eigen::Index n_obs,
                            const slope_term<T_coef, T_d>& leaf) {
  check_size_match(function, "Random variable size", n_obs,
                   "Data vector size",
                   static_cast<Eigen::Index>(leaf.xd.size()));
}

template <typename T_coef, typename T_d1, typename T_d2>
inline void check_leaf_size(const char* function, Eigen::Index n_obs,
                            const slope2_term<T_coef, T_d1, T_d2>& leaf) {
  check_size_match(function, "Random variable size", n_obs,
                   "Data vector size",
                   static_cast<Eigen::Index>(leaf.xd1.size()));
  check_size_match(function, "Random variable size", n_obs,
                   "Data vector size",
                   static_cast<Eigen::Index>(leaf.xd2.size()));
}

// Reverse pass over the resolved-leaf tuple in REVERSE declaration order:
// the composed path's per-element varis are swept in reverse creation
// order, so within each element the LAST-declared leaf's increment lands
// first.
template <std::size_t I = 0, typename Tuple>
inline void rev_leaves(const Tuple& leaves, Eigen::Index k, double e) {
  if constexpr (I < std::tuple_size_v<std::decay_t<Tuple>>) {
    rev_leaves<I + 1>(leaves, k, e);
    std::get<I>(leaves).rev(k, e);
  }
}

// Shared impl for the additive overloads. `scatter=true`: the callback
// scatters the increments into the coefficient adjoints directly (the shape
// for likelihood-last models). `scatter=false`: the callback writes the
// per-observation increments into `target`'s varis with the stock edge's
// arithmetic, and the model's own transformed-parameter chains propagate
// them (the shape for prior-statements-before-likelihood models such as
// election88 — see the _tp overload's docs).
template <bool propto, bool scatter, typename T_n, typename T_target,
          typename T_intercept, typename... Leaves>
inline var additive_impl(const T_n& n, const T_target* target,
                         const T_intercept& intercept,
                         const Leaves&... leaves) {
  using T_partials_array = Eigen::Array<double, Eigen::Dynamic, 1>;
  using std::exp;
  static constexpr const char* function = "bernoulli_logit_lpmf";

  const Eigen::Index n_obs = stan::math::size(n);
  ((internal::check_leaf_size(function, n_obs, leaves)), ...);
  if constexpr (!scatter) {
    check_size_match(function, "Random variable size", n_obs,
                     "Adjoint target size", target->size());
  }
  if (unlikely(n_obs == 0 || size_zero(n))) {
    return var(0.0);
  }

  vari* intercept_vi = intercept.vi_;
  const double intercept_val = intercept.vi_->val_;

  // Resolve every leaf's routes (adjoint targets + forward values).
  const auto leaves_resolved
      = std::tuple(internal::resolve_leaf(leaves, n_obs)...);

  // Per-observation predictor in the composed path's exact op order:
  // left-associated sum over already-rounded leaf values. The gathered
  // leaves' range checks fire HERE, in the composed path's per-element leaf
  // order — i.e. BEFORE check_bounded, matching the transformed-parameter
  // loop that runs ahead of the likelihood statement in the stock model.
  arena_t<T_partials_array> eta(n_obs);
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    double t = intercept_val;
    std::apply(
        [&t, k](const auto&... ls) { ((t = t + ls.fwd(k)), ...); },
        leaves_resolved);
    eta.coeffRef(k) = t;
  }
  check_bounded(function, "n", n, 0, 1);
  check_not_nan(function, "Logit transformed probability parameter", eta);
  if constexpr (!include_summand<propto, T_intercept>::value) {
    return var(0.0);
  }

  // ---- bernoulli_logit_lpmf interior, verbatim (as in the 2PL overload) ----
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

  if constexpr (scatter) {
    return make_callback_var(
        logp,
        [intercept_vi, leaves_resolved, dtheta](const auto& vi) {
          const double w = vi.adj_;
          // Elements DESCENDING, leaves in reverse declaration order: the
          // composed path's scalar varis are swept from the stack top down.
          for (Eigen::Index k = dtheta.size() - 1; k >= 0; --k) {
            const double e = w * dtheta.coeff(k);
            internal::rev_leaves(leaves_resolved, k, e);
            intercept_vi->adj_ += e;
          }
        });
  } else {
    // Stock's edge application, bit-for-bit: one FUSED multiply-add per
    // element into the target vari's adjoint, elements ascending (the
    // composed update_adjoints loop decoded at the model flags). Written as
    // a single expression so GCC contracts it exactly as stock does.
    arena_t<Eigen::Matrix<vari*, Eigen::Dynamic, 1>> target_vi(n_obs);
    for (Eigen::Index k = 0; k < n_obs; ++k) {
      target_vi.coeffRef(k) = target->coeff(k).vi_;
    }
    return make_callback_var(
        logp, [target_vi, dtheta](const auto& vi) {
          const double w = vi.adj_;
          for (Eigen::Index k = 0; k < dtheta.size(); ++k) {
            target_vi.coeff(k)->adj_ += w * dtheta.coeff(k);
          }
        });
  }
}

}  // namespace internal

/** \ingroup prob_dists
 * Additive multi-gather Bernoulli-logit likelihood: the log PMF of the
 * logit-parametrized Bernoulli distribution over the index/data-assembled
 * linear predictor
 *
 *   eta[n] = intercept + sum_t prod_j xd_tj[n] * coef_t
 *          + sum_k coefs_k[idx_k[n]]
 *
 * built from `slope_term` / `slope2_term` / `gather_term` leaves, WITHOUT
 * materializing the per-observation predictor as autodiff intermediates.
 * This is the pattern behind the election88-class model
 *
 *   y ~ bernoulli_logit(beta[1] + beta[2]*black + beta[3]*female
 *                       + beta[5]*female*black + beta[4]*v_prev
 *                       + a[age] + b[edu] + c[age_edu] + d[state]
 *                       + e[region_full]);
 *
 * (there written through a transformed parameter, whose own chain this
 * primitive does not replace: call it from the likelihood statement with the
 * gathered operands directly and leave the transformed-parameter loop as the
 * compiler wrote it; that loop's adjoints stay zero and its values stay
 * available for output columns).
 *
 * The value path performs exactly the same floating point operations, in the
 * same per-element order, as the composed scalar expression the compiler
 * generates for that model: per element, each data-product leaf evaluates
 * its rounded product chain (vmulsd; left-associated for slope2), and the
 * terms are summed left-to-right with pure rounded adds over already-rounded
 * products (vaddsd; the `volatile` barriers in the leaf forward paths block
 * GCC's fp-contraction from fusing an accumulation add with a leaf product).
 * The interior is the `bernoulli_logit_lpmf` expression verbatim (signs,
 * `ntheta`, `exp(-ntheta)`, the two nested `Select` trees, Eigen's `sum`
 * redux).
 *
 * The reverse pass is ONE callback vari replicating the machine-code schedule
 * of the composed path (W-127 disassembly of the stock model at
 * `-O3 -mavx2 -mfma`): the likelihood edge applies its partial as one
 * multiply-add per element, the transformed-parameter expression's scalar
 * varis are swept in reverse creation order (elements DESCENDING, and within
 * each element the leaves in REVERSE declaration order), gathered terms
 * arrive through pure adds, single-product terms through a FUSED multiply-add
 * (`multiply_vd_vari::chain`'s vfmadd132sd), and two-product terms through a
 * rounded intermediate product then a fused multiply-add:
 *
 *   e                 = round(w * dtheta[k])         [the edge application]
 *   coefs_k[idx]_adj += e                            [pure add]
 *   coef_t_adj        = fma(xd_t[k], e, coef_t_adj)  [fused]
 *   t                 = round(e * xd2[k])            [m2 chain, one rounding]
 *   coef_adj          = fma(xd1[k], t, coef_adj)     [m1 chain, fused]
 *   intercept_adj     += e                            [pure add, first operand]
 *
 * with `w` the adjoint of the returned log probability and `dtheta` the
 * elementwise partial of `bernoulli_logit_lpmf` (branch behavior at
 * `|ntheta| > 20` preserved). Values and every gradient component are
 * bit-identical to the composed stock path in every operand layout the stanc
 * deserializer produces (`Matrix<var>`, `var_value<Matrix<double>>`,
 * `Map<const Matrix<var>>`).
 *
 * Checks mirror stock's `bernoulli_logit_lpmf` (same function name in
 * messages: `check_bounded` on `n`, `check_not_nan` on the assembled
 * predictor) and stock's `rvalue` range checks on every gathered index
 * ("vector[uni] indexing", in the composed path's per-element leaf order).
 *
 * @tparam propto if `true`, normalize out constant terms (there are none for
 * this distribution)
 * @tparam T_n type of the random variable (integer vector-like)
 * @tparam T_intercept scalar var type of the intercept term
 * @tparam Leaves zero or more `gather_term` / `slope_term` / `slope2_term`
 * leaves, in the composed expression's declaration order
 * @param n random variable (0 or 1), one entry per observation
 * @param intercept scalar var intercept of the predictor
 * @param leaves the predictor's product and gathered terms
 * @return var holding the log probability mass
 * @throw std::domain_error if any assembled predictor value is NaN
 * @throw std::invalid_argument if the container sizes mismatch
 * @throw std::out_of_range if a gathered index is out of range
 */
template <bool propto, typename T_n, typename T_intercept, typename... Leaves,
          require_st_var<T_intercept>* = nullptr,
          require_vector_like_vt<std::is_integral, T_n>* = nullptr>
inline var bernoulli_logit_lpmf_gathered_additive(
    const T_n& n, const T_intercept& intercept, const Leaves&... leaves) {
  return internal::additive_impl<propto, true>(
      n, static_cast<const Eigen::Matrix<var, Eigen::Dynamic, 1>*>(nullptr),
      intercept, leaves...);
}

/** \ingroup prob_dists
 * Additive multi-gather likelihood (see the propto overload). Drops constant
 * terms; there are none for this distribution, so this matches
 * `bernoulli_logit_lpmf_gathered_additive<true>` exactly.
 */
template <typename T_n, typename T_intercept, typename... Leaves,
          require_st_var<T_intercept>* = nullptr,
          require_vector_like_vt<std::is_integral, T_n>* = nullptr>
inline var bernoulli_logit_lpmf_gathered_additive(
    const T_n& n, const T_intercept& intercept, const Leaves&... leaves) {
  return internal::additive_impl<true, true>(
      n, static_cast<const Eigen::Matrix<var, Eigen::Dynamic, 1>*>(nullptr),
      intercept, leaves...);
}

/** \ingroup prob_dists
 * Transformed-parameter-writeback variant of the additive multi-gather
 * Bernoulli-logit likelihood, for models whose predictor is ALSO consumed
 * through a transformed parameter (`vector[N] y_hat` built by a per-element
 * loop and kept materialized for output, the election88 class).
 *
 * The value path is identical to `bernoulli_logit_lpmf_gathered_additive`
 * (the predictor is recomputed from the gathered operands in stock's exact
 * op order — the transformed-parameter loop is retained untouched for the
 * output columns, and its recomputation is bit-identical by construction).
 * The REVERSE pass, however, does NOT scatter into the coefficient adjoints
 * directly: it applies its per-observation increments to
 * `adjoint_target[k]`'s own vari with stock edge arithmetic (one fused
 * multiply-add per element, elements ascending — the composed
 * `bernoulli_logit_lpmf`'s edge application bit-for-bit), and the retained
 * transformed-parameter expression's own varis then propagate them at their
 * machine-code-verified stack position and schedule.
 *
 * WHY: when the model block's PRIOR statements touch the coefficient
 * vectors BEFORE the likelihood statement (election88: `a ~ normal(0,
 * sigma_a); ...; y ~ bernoulli_logit(y_hat);`), the reverse-mode stack
 * sweep applies the prior edges BETWEEN the likelihood edge and the
 * transformed-parameter chains. Direct scattering from the likelihood's
 * stack position would accumulate each coefficient's adjoint in the
 * opposite order (`sum + prior` vs stock's `prior + sum`) — a 1-ulp
 * reorder (values stay bit-identical; W-127 measured 0/100 lp mismatches
 * with 100/100 gradient rows differing by 1 ulp). The writeback variant
 * reproduces stock exactly in that layout; the direct-scatter overload is
 * the right shape when the likelihood is the last statement touching its
 * operands (the gathered-operand class of the 2PL/radon models, where the
 * composed path's own callbacks also sit at the likelihood's stack
 * position).
 *
 * @param adjoint_target the transformed-parameter vector whose per-element
 * varis receive the likelihood's adjoint increments (stock's `y_hat`; its
 * values are NOT read — the predictor is recomputed from the leaves)
 */
template <bool propto, typename T_n, typename T_target, typename T_intercept,
          typename... Leaves, require_st_var<T_intercept>* = nullptr,
          require_vector_like_vt<std::is_integral, T_n>* = nullptr,
          require_rev_matrix_t<T_target>* = nullptr>
inline var bernoulli_logit_lpmf_gathered_additive_tp(
    const T_n& n, const T_target& adjoint_target,
    const T_intercept& intercept, const Leaves&... leaves) {
  return internal::additive_impl<propto, false>(n, &adjoint_target, intercept,
                                                leaves...);
}

/** \ingroup prob_dists
 * Transformed-parameter-writeback variant (see the propto overload). Drops
 * constant terms; there are none for this distribution.
 */
template <typename T_n, typename T_target, typename T_intercept,
          typename... Leaves, require_st_var<T_intercept>* = nullptr,
          require_vector_like_vt<std::is_integral, T_n>* = nullptr,
          require_rev_matrix_t<T_target>* = nullptr>
inline var bernoulli_logit_lpmf_gathered_additive_tp(
    const T_n& n, const T_target& adjoint_target,
    const T_intercept& intercept, const Leaves&... leaves) {
  return internal::additive_impl<true, false>(n, &adjoint_target, intercept,
                                              leaves...);
}

}  // namespace math
}  // namespace stan
#endif
