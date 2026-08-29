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
#include <new>
#include <vector>

namespace stan {
namespace math {

namespace internal {

/** \ingroup prob_dists
 * W-118 (W-53-class batching): ONE nochain-stack registration covering a
 * whole batch of unregistered term records. `set_zero_all_adjoints`
 * (and its nested variant) reach the batch through this adapter's
 * virtual `set_zero_adjoint`, giving the batch the same per-record
 * zeroing the stock loop's per-element `var(double)` records receive
 * (the accumulator's `sum` chains ACCUMULATE `+=` into the term
 * adjoints, so zeroing parity is required for repeated `grad()` calls
 * on one tape). The adapter itself is arena-allocated and never
 * chained (`chain()` is empty; it lives on the nochain stack).
 */
class gathered_term_zeroer final : public vari_base {
 private:
  vari_value<double>* recs_;
  Eigen::Index n_;

 public:
  gathered_term_zeroer(vari_value<double>* recs, Eigen::Index n)
      : recs_(recs), n_(n) {
    ChainableStack::instance_->var_nochain_stack_.push_back(this);
  }
  inline void chain() final {}
  inline void set_zero_adjoint() final {
    for (Eigen::Index k = 0; k < n_; ++k) {
      recs_[k].adj_ = 0.0;
    }
  }
};

/** \ingroup prob_dists
 * The reverse-pass state of one gathered-normal call, as ONE arena
 * struct captured by pointer (no per-array closure copies). All
 * pointed-to arrays are arena-allocated and outlive the tape.
 */
struct gathered_normal_revdata {
  vari_value<double>* term_recs;  // batched term records (adjoint read here)
  const int* ii;                  // 0-based alpha index per observation
  const int* ii2;                 // 0-based beta index (shape B)
  const double* x;                // data slope (shape B)
  const double* d_mu;             // inv_sigma * y_scaled per element
  const double* d_sigma;          // inv_sigma*y_scaled_sq - inv_sigma
                                  // (nullptr when sigma is data)
  vari* const* alpha_vi;          // AoS route (nullptr on SoA)
  vari* const* beta_vi;           // AoS route (nullptr on SoA)
  vari_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>* alpha_soa;
  vari_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>* beta_soa;
  vari* sigma_vi;                 // nullptr when sigma is data
  Eigen::Index n;
};

/** \ingroup prob_dists
 * The vectorizable term pass as a STANDALONE noinline function: inside a
 * stanc-generated model TU (a single ~31KB translation unit) GCC's loop
 * vectorizer gives up on every double loop of the inlined primitive
 * (verified by whole-TU disassembly, W-118); a function boundary restores
 * it. The `__restrict__` parameters are load-bearing under the model .so's
 * -fPIC (the vectorizer refuses to version the loop for aliasing there);
 * every array is a distinct arena allocation, so the no-alias assertion
 * is true. Per-lane op order is stock's scalar sequence with stock's
 * compiler contraction points (W-112.1 table F2-F7: (y-mu)*inv_sigma
 * unfused vsub+vmul, z^2 mul, the -0.5*z^2 + const increment CONTRACTED,
 * the -= log_sigma sub, d_mu mul, and the CONTRACTED
 * inv_sigma*z^2 - inv_sigma). No horizontal operations.
 */
template <bool HasConstTerm, bool HasLogSigma, bool HasDS>
[[gnu::noinline]] void gathered_term_pass(
    const double* __restrict__ y_data, const double* __restrict__ mu_data,
    const int* __restrict__ bounds_mask, double* __restrict__ lp,
    double* __restrict__ d_mu, double* __restrict__ d_sigma,
    int* __restrict__ mask_out, const Eigen::Index n_obs,
    const double inv_sigma, const double log_sigma) {
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    const double mu_k = mu_data[k];
    const int bad = bounds_mask[k];
    const double y_k = y_data[k];
    const double y_scaled = (y_k - mu_k) * inv_sigma;
    const double y_scaled_sq = y_scaled * y_scaled;
    double lp_k = -0.5 * y_scaled_sq;
    if constexpr (HasConstTerm) {
      lp_k += NEG_LOG_SQRT_TWO_PI;
    }
    if constexpr (HasLogSigma) {
      lp_k -= log_sigma;
    }
    lp[k] = lp_k;
    d_mu[k] = inv_sigma * y_scaled;
    if constexpr (HasDS) {
      d_sigma[k] = inv_sigma * y_scaled_sq - inv_sigma;
    }
    const double mu_self = mu_k - mu_k;
    mask_out[k] = bad | static_cast<int>(y_k != y_k)
                  | static_cast<int>(mu_self != 0.0);
  }
}

/** \ingroup prob_dists
 * Shared core of the gathered normal likelihoods (not part of the public
 * API). W-118 fused interior: ONE vectorizable term traversal produces,
 * per element, the 0-based index, the log-density term, the mu-edge
 * partial (and, when sigma is a var, the sigma-edge partial) and the
 * per-element validity mask carrying stock's `check_range` +
 * `check_not_nan(y)` + `check_finite(mu)` conditions. The per-element op
 * order is EXACTLY that of the SCALAR instantiation
 * `normal_lpdf<propto>(double, var, var)` the stanc-generated loop calls,
 * compiler contraction points included (the `-0.5*z^2 + const` increment
 * and the `inv_sigma*z^2 - inv_sigma` partial are contracted forms in
 * stock; the W-112.1 disassembly table F2-F7). The loop body has no
 * horizontal operations and no control flow, so the compiler's per-lane
 * vectorization is bit-identical to its scalar codegen (each lane
 * performs stock's scalar sequence on its own element). Throw-set parity
 * is preserved by a cold per-element re-derivation in stock's exact
 * order (alpha index, beta index, y, mu) when the mask reports any bad
 * element; valid states never leave the fused loop.
 *
 * The returned vector holds one `var` per observation wrapping a
 * NO-STACK batched `vari_value<double>` record (W-53-class arena
 * batching: one arena allocation, no per-record stack push, one
 * `gathered_term_zeroer` registration for zeroing parity). The records'
 * values and zero adjoints are byte-identical to the per-element
 * `var(double)` records of the unfused primitive, keeping the
 * model-side accumulation schedule (stan::math::accumulator's chunked
 * buffer) bit-identical to the stock loop.
 *
 * The linear predictor values arrive from the public overload's gather
 * pass: shape B needs it anyway (its volatile-barrier multiply-add
 * cannot enter a vectorized loop), and shape A's random-index gather
 * cannot be auto-vectorized either (GCC's loop vectorizer produces no
 * vgather for a[b[k]] loads -- verified by probe, W-118), so both
 * shapes gather first and the term loop reads a dense mu array.
 */
template <bool propto, bool HasBeta, typename T_alpha, typename T_beta,
          typename T_sigma>
std::vector<var> normal_lpdf_gathered_impl(
    const Eigen::Index n_obs, const double* y_data, const int* ii_data,
    Eigen::Index alpha_size, const double* alpha_d, const double* mu_data,
    const int* bounds_mask, const int* ii2_data, Eigen::Index beta_size,
    const int* ii2_store, const double* x_arr, const int* ii_arr,
    const T_alpha& alpha, const T_beta& beta, const T_sigma& sigma) {
  static constexpr const char* function = "normal_lpdf_gathered";
  const double sigma_val = value_of(sigma);
  check_positive(function, "Scale parameter", sigma_val);

  // Adjoint routes. No gathered Matrix<var> is ever built: Matrix<var>
  // operands get one vari* per coefficient (the loop's mu elements alias
  // exactly these varis), var_value<> (SoA) operands keep their single
  // matrix vari.
  using vari_vec = Eigen::Matrix<vari*, Eigen::Dynamic, 1>;
  using soa_vec_vari = vari_value<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
  [[maybe_unused]] arena_t<vari_vec> alpha_vi(0);
  [[maybe_unused]] arena_t<vari_vec> beta_vi(0);
  if constexpr (!is_var_v<std::decay_t<T_alpha>>) {
    alpha_vi = arena_t<vari_vec>(alpha.size());
    for (Eigen::Index j = 0; j < alpha.size(); ++j) {
      alpha_vi.coeffRef(j) = alpha.coeff(j).vi_;
    }
  }
  if constexpr (HasBeta) {
    if constexpr (!is_var_v<std::decay_t<T_beta>>) {
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

  auto& memalloc = ChainableStack::instance_->memalloc_;
  static constexpr bool sigma_is_var_v = is_var_v<std::decay_t<T_sigma>>;
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> lp_arr(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> d_mu(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> d_sigma(
      sigma_is_var_v ? n_obs : 0);

  // ---- the vectorizable term pass (standalone noinline function: see
  // gathered_term_pass). Bad elements are recorded in a per-element int
  // mask (any of stock's per-element conditions -- the shape-specific
  // gather pass contributes the index-range bit, this pass adds y-NaN
  // and mu-non-finite) and diagnosed in stock's exact per-element order
  // by the cold path below.
  int* const bmask_out = memalloc.alloc_array<int>(n_obs);
  gathered_term_pass<include_summand<propto>::value,
                     include_summand<propto, T_sigma>::value, sigma_is_var_v>(
      y_data, mu_data, bounds_mask, lp_arr.data(), d_mu.data(),
      sigma_is_var_v ? d_sigma.data() : static_cast<double*>(nullptr),
      bmask_out, n_obs, inv_sigma_val, log_sigma_val);

  // ---- the batched term records (W-53 class): one arena allocation, no
  // per-record stack push, one zeroing registration. The record loop is
  // scalar (24-byte stride stores) and carries the cold throw-set path.
  vari_value<double>* const recs
      = memalloc.alloc_array<vari_value<double>>(n_obs);
  std::vector<var> terms;
  terms.reserve(n_obs);
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    if (unlikely(bmask_out[k] != 0)) {
      // Stock's per-element order re-derived from the ORIGINAL operands:
      // rvalue bounds (alpha, then beta for shape B -- exactly the
      // generated loop's rvalue order), then check_not_nan(y), then
      // check_finite(mu). The scalar check overloads reproduce stock's
      // exact exception type and byte-identical message text (value
      // printed, no index). THROW-SET parity is part of the
      // bit-identity contract (W-112.1/W-112.2 mechanism).
      const int iraw = ii_data[k];
      if (unlikely(iraw < 1 || iraw > alpha_size)) {
        check_range("vector[uni] indexing", "alpha", alpha_size, iraw);
      }
      if constexpr (HasBeta) {
        const int i2raw = ii2_data[k];
        if (unlikely(i2raw < 1 || i2raw > beta_size)) {
          check_range("vector[uni] indexing", "beta", beta_size, i2raw);
        }
      }
      check_not_nan("normal_lpdf", "Random variable", y_data[k]);
      if constexpr (HasBeta) {
        check_finite("normal_lpdf", "Location parameter", mu_data[k]);
      } else {
        check_finite("normal_lpdf", "Location parameter", alpha_d[iraw - 1]);
      }
    }
    ::new (static_cast<void*>(recs + k))
        vari_value<double>(lp_arr.coeff(k), vari_no_stack);
    terms.emplace_back(recs + k);
  }
  new gathered_term_zeroer(recs, n_obs);

  // ---- ONE reverse callback replacing the per-element ops-partials
  // callbacks (and, for HasBeta, the per-element multiply/add
  // callbacks). grad() runs the stack in reverse creation order, so the
  // loop's per-element varis accumulate into the shared sigma vari and
  // the indexed alpha/beta varis in REVERSE n order; the same order and
  // the same arithmetic statements as the unfused primitive (contraction
  // points included: the beta increment stays contractible on the AoS
  // route and un-contracted on the SoA route -- stock's multiply chain
  // fuses exactly the former).
  auto* rd = new (memalloc.alloc_array<gathered_normal_revdata>(1))
      gathered_normal_revdata{};
  rd->term_recs = recs;
  rd->ii = ii_arr;
  rd->ii2 = ii2_store;
  rd->x = x_arr;
  rd->d_mu = d_mu.data();
  rd->d_sigma
      = sigma_is_var_v ? d_sigma.data() : static_cast<double*>(nullptr);
  rd->alpha_vi = nullptr;
  rd->beta_vi = nullptr;
  rd->alpha_soa = nullptr;
  rd->beta_soa = nullptr;
  rd->sigma_vi = nullptr;
  rd->n = n_obs;
  if constexpr (is_var_v<std::decay_t<T_alpha>>) {
    rd->alpha_soa = alpha.vi_;
  } else {
    rd->alpha_vi = alpha_vi.data();
  }
  if constexpr (HasBeta) {
    if constexpr (is_var_v<std::decay_t<T_beta>>) {
      rd->beta_soa = beta.vi_;
    } else {
      rd->beta_vi = beta_vi.data();
    }
  }
  if constexpr (is_var_v<std::decay_t<T_sigma>>) {
    rd->sigma_vi = sigma_vi;
  }
  reverse_pass_callback([rd]() {
    const Eigen::Index n = rd->n;
    for (Eigen::Index k = n - 1; k >= 0; --k) {
      // the term's adjoint (set by the accumulation of the returned
      // terms upstream) and the scalar lpdf's mu-edge add
      const double m = rd->term_recs[k].adj_ * rd->d_mu[k];
      const int ik = rd->ii[k];
      if constexpr (is_var_v<std::decay_t<T_alpha>>) {
        rd->alpha_soa->adj_.coeffRef(ik) += m;
      } else {
        rd->alpha_vi[ik]->adj_ += m;
      }
      if constexpr (HasBeta) {
        const int i2 = rd->ii2[k];
        if constexpr (is_var_v<std::decay_t<T_beta>>) {
          // SoA route: stock's rvalue on a var_value<> creates a read
          // vari whose callback adds the already-rounded m * x into the
          // matrix adjoint with a PLAIN add (no multiply to contract),
          // so the product is kept un-contracted here.
          volatile double mbx = m * rd->x[k];
          rd->beta_soa->adj_.coeffRef(i2) += mbx;
        } else {
          // AoS route: stock's multiply node chain does
          // adj += (add-node adjoint) * x -- a pointer read-modify-write
          // the compiler FMA-contracts exactly like this statement
          // (verified by gate (a) and disassembly of
          // internal::multiply_vd_vari::chain).
          const double mbx = m * rd->x[k];
          rd->beta_vi[i2]->adj_ += mbx;
        }
      }
      if constexpr (is_var_v<std::decay_t<T_sigma>>) {
        rd->sigma_vi->adj_ += rd->term_recs[k].adj_ * rd->d_sigma[k];
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
 * @throw std::domain_error if any y[n] is NaN, any mu value is not
 * finite, or sigma is not positive (per element, in stock's order --
 * sigma once per call; the throw set matches the stock loop exactly)
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
  if (unlikely(n_obs == 0)) {
    check_positive(function, "Scale parameter", value_of(sigma));
    return std::vector<var>{};
  }
  const Eigen::Index alpha_size = alpha.size();
  const int* ii_data = ii.data();
  // stock's first rvalue throws for the empty coefficient vector (also
  // keeps the fused loop's clamped gather safe)
  if (unlikely(alpha_size == 0)) {
    check_range("vector[uni] indexing", "alpha", 0, ii_data[0]);
  }
  // y is already a dense double vector: read it in place (no copy). A
  // non-direct-access expression is evaluated once (no model hits it).
  Eigen::Matrix<double, Eigen::Dynamic, 1> y_store;
  const double* y_data;
  if constexpr ((Eigen::internal::traits<T_y>::Flags & Eigen::DirectAccessBit)
                != 0) {
    y_data = y.data();
  } else {
    y_store = y;
    y_data = y_store.data();
  }
  // alpha values: the SoA operand's val_ array is read in place; the AoS
  // operand's values are collected once (J-sized, not N-sized).
  Eigen::Matrix<double, Eigen::Dynamic, 1> alpha_store;
  const double* alpha_d;
  if constexpr (is_var_v<std::decay_t<T_alpha>>) {
    alpha_d = alpha.vi_->val_.data();
  } else {
    alpha_store = value_of(alpha);
    alpha_d = alpha_store.data();
  }
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> ii_arena(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> mu_val(n_obs);
  int* const bounds_mask
      = ChainableStack::instance_->memalloc_.alloc_array<int>(n_obs);
  // ---- the gather pass (shape A): clamped gather + stock's per-element
  // index condition in a mask bit for the term pass's cold path (the
  // random-index load cannot enter an auto-vectorized loop; the clamped
  // load's value on a bad lane is discarded by that cold path).
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    const int iraw = ii_data[k];
    const int ikc = (iraw < 1 ? 1 : (iraw > alpha_size ? (int)alpha_size : iraw)) - 1;
    const unsigned irawu = static_cast<unsigned>(iraw) - 1u;
    bounds_mask[k] = irawu >= static_cast<unsigned>(alpha_size) ? 1 : 0;
    ii_arena.coeffRef(k) = ikc;
    mu_val.coeffRef(k) = alpha_d[ikc];
  }
  return internal::normal_lpdf_gathered_impl<propto, false>(
      n_obs, y_data, ii_data, alpha_size, alpha_d, mu_val.data(),
      bounds_mask, nullptr, 0, nullptr, nullptr, ii_arena.data(), alpha,
      0.0, sigma);
}

/** \ingroup prob_dists
 * Gathered normal likelihood, loop form with a data slope: the
 * per-element log densities of
 *
 *   y[n] ~ normal(alpha[ii[n]] + x[n] * beta[ii2[n]], sigma),  n = 1..N
 *
 * returned one `var` per observation (see the (y, alpha, ii, sigma)
 * overload for the bit-identity contract). The linear predictor values
 * are assembled in one gather pass with the generated loop's op order:
 * the multiplication `x[n] * beta[ii2[n]]` first, then the addition of
 * `alpha[ii[n]]`. In the reverse pass, beta's adjoint contribution for
 * element n is the loop's two-step propagation
 * `(w * dlp/dmu_n) * x[n]` (the scalar lpdf's mu-edge add followed by
 * the multiply node's chain), in the same order.
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
  if (unlikely(n_obs == 0)) {
    check_positive(function, "Scale parameter", value_of(sigma));
    return std::vector<var>{};
  }
  const Eigen::Index alpha_size = alpha.size();
  const Eigen::Index beta_size = beta.size();
  const int* ii_data = ii.data();
  const int* ii2_data = ii2.data();
  // stock's first rvalue order: alpha's index, then (after the x
  // rvalue) beta's index; empty vectors throw before anything else (and
  // keep the gather pass's clamped loads safe)
  if (unlikely(alpha_size == 0)) {
    check_range("vector[uni] indexing", "alpha", 0, ii_data[0]);
  }
  if (unlikely(beta_size == 0)) {
    check_range("vector[uni] indexing", "beta", 0, ii2_data[0]);
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> y_store;
  const double* y_data;
  if constexpr ((Eigen::internal::traits<T_y>::Flags & Eigen::DirectAccessBit)
                != 0) {
    y_data = y.data();
  } else {
    y_store = y;
    y_data = y_store.data();
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> alpha_store;
  const double* alpha_d;
  if constexpr (is_var_v<std::decay_t<T_alpha>>) {
    alpha_d = alpha.vi_->val_.data();
  } else {
    alpha_store = value_of(alpha);
    alpha_d = alpha_store.data();
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> beta_store;
  const double* beta_d;
  if constexpr (is_var_v<std::decay_t<T_beta>>) {
    beta_d = beta.vi_->val_.data();
  } else {
    beta_store = value_of(beta);
    beta_d = beta_store.data();
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> x_store;
  const double* x_data;
  if constexpr ((Eigen::internal::traits<T_x>::Flags & Eigen::DirectAccessBit)
                != 0) {
    x_data = x.data();
  } else {
    x_store = x;
    x_data = x_store.data();
  }
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> ii_arena(n_obs);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> ii2_arena(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_arena(n_obs);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> mu_val(n_obs);
  int* const bounds_mask
      = ChainableStack::instance_->memalloc_.alloc_array<int>(n_obs);
  // ---- the gather pass (shape B only): the generated loop's linear
  // predictor assembly, with stock's per-element index conditions
  // recorded in a mask bit for the term pass's cold path (stock's throw
  // ORDER -- alpha rvalue, beta rvalue, y, mu -- is preserved by that
  // cold path; the mask only defers the throw). The clamped loads'
  // values on bad lanes are discarded by the cold path.
  for (Eigen::Index k = 0; k < n_obs; ++k) {
    const int iraw = ii_data[k];
    const int i2raw = ii2_data[k];
    const unsigned irawu = static_cast<unsigned>(iraw) - 1u;
    const unsigned i2rawu = static_cast<unsigned>(i2raw) - 1u;
    const bool a_bad = irawu >= static_cast<unsigned>(alpha_size);
    const bool b_bad = i2rawu >= static_cast<unsigned>(beta_size);
    bounds_mask[k] = (a_bad | b_bad) ? 1 : 0;
    const int ik = a_bad ? 0 : (int)irawu;
    const int i2 = b_bad ? 0 : (int)i2rawu;
    ii_arena.coeffRef(k) = ik;
    ii2_arena.coeffRef(k) = i2;
    x_arena.coeffRef(k) = x_data[k];
    // stock: multiply(x[n], beta[ii2[n]]) evaluated first (its value is
    // stored to the multiply vari), then add(alpha[ii[n]], ...). The two
    // results round-trip through memory in stock, so no FMA contraction of
    // the multiply-add is possible there; the volatile forces the same
    // un-contracted schedule here.
    volatile double prod = x_data[k] * beta_d[i2];
    mu_val.coeffRef(k) = alpha_d[ik] + prod;
  }
  return internal::normal_lpdf_gathered_impl<propto, true>(
      n_obs, y_data, ii_data, alpha_size, nullptr, mu_val.data(),
      bounds_mask, ii2_data, beta_size, ii2_arena.data(), x_arena.data(),
      ii_arena.data(), alpha, beta, sigma);
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
