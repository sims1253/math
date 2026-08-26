#ifndef STAN_MATH_PRIM_PROB_BERNOULLI_LOGIT_LPMF_HPP
#define STAN_MATH_PRIM_PROB_BERNOULLI_LOGIT_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/fun/as_value_array_or_scalar.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/log1p.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/promote_scalar.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <cmath>
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

// ===================== W-103 KERNEL (internal) =====================
// Fused forward kernel for bernoulli_logit_lpmf: per element with
// w=exp(-|x|):
//   value   = x>20 ? -w : x<-20 ? x : min(x,0) - log1p(w)
//   partial = x>20 ? -w : x<-20 ? sg : (x<0 ? sg/(1+w) : sg*w/(1+w))
// W-46 lineage: bodies byte-identical to
// harness/w46/bernoulli_logit_lpmf.hpp.patched (validated: parity
// grad rel-L2 2.37e-16, -22.8% Ir/grad on hier_2pl at stock form;
// results/log1p_ceiling_w46.md). log1p(w) via peeled Chebyshev deg-16
// (<=1 ulp vs glibc on w in [e^-20,1]; mpmath 60-dps fit,
// scratch/w46/fit_log1p.py). Baseline path is scalar (std::exp + poly);
// an AVX2+FMA island (runtime dispatch) runs where the CPU supports it.
// W-103 changes vs the W-46 file: (a) namespace placement (internal::w46
// instead of a global w46_kern), (b) the W-46 `#pragma GCC
// target("avx2,fma")` region is replaced by per-function target
// attributes (identical GCC codegen; clang-compatible), (c) the kernel
// arm is taken only when T_partials_return is double (prim/rev);
// fvar/higher-order keeps the stock expression path below verbatim.
// BUG-COMPATIBILITY: the x>20 partial replicates the stock
// `-exp_m_ntheta` (NO signs factor) bug — see the elem() note below;
// the fix is a separate one-line upstream change (W-46 section 5).
namespace stan {
namespace math {
namespace internal {
namespace w46 {

inline constexpr double kLn1pHalf = 0.4054651081081643819780131154643;
inline constexpr double kS[17] = {
       0.284829740686107820177,  -0.0444902635080327358639,
     0.00366968616228059164663, -0.000313502828368026324476,
    0.0000274490505901107177794, -0.00000244748467663048339674,
     2.21316418650173210927e-7, -2.02377436080352872464e-8,
     1.86753179590458400941e-9, -1.73645361011441549114e-10,
    1.62493634499289881827e-11, -1.52892923161186679152e-12,
    1.44543029838076305004e-13, -1.37216703518573255112e-14,
    1.30738719340475863584e-15, -1.24971776483632075309e-16,
     1.1980675245932421879e-17
};

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline double log1p_poly(double w) {
  const double a = (w >= 0.5) ? 0.5 : 0.0;
  const double u = (w - a) * ((a == 0.0) ? 1.0 : (2.0 / 3.0));
  const double z = (u - 0.25) * 4.0;
  const double zz = 2.0 * z;
  double b1 = 0.0, b2 = 0.0;
  for (int k = 16; k >= 1; --k) {
    const double t = kS[k] + zz * b1 - b2;
    b2 = b1;
    b1 = t;
  }
  const double S = kS[0] + z * b1 - b2;
  const double u2 = u * u;
  double r = (a == 0.5) ? kLn1pHalf : 0.0;
  r += u - 0.5 * u2;
  r += u * u2 * S;
  return r;
}

// scalar element: value term; EDGE partial (d lp/d theta, signs applied)
// into *pp. NOTE: the x>20 branch replicates stan-math 5.3.0's
// `-exp_m_ntheta` (NO signs factor) bug-compatibly; the mathematically
// correct partial would be signs*w. See results/log1p_ceiling_w46.md.
inline double elem(double x, double sg, double* pp) {
  const double w = std::exp(-std::fabs(x));
  if (x > 20.0) {
    *pp = -w;   // stock: -exp(-ntheta), no signs (W-46 bug-compat)
    return -w;
  }
  if (x < -20.0) {
    *pp = sg;
    return x;
  }
  const double l = log1p_poly(w);
  if (x < 0.0) {
    *pp = sg / (1.0 + w);
    return x - l;
  }
  *pp = sg * (w / (1.0 + w));
  return -l;
}

inline double fwd_scalar(const double* x, const double* sg, int n, double* p) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += elem(x[i], sg[i], p + i);
  }
  return s;
}

// ---- AVX2+FMA island (runtime-dispatched); no Eigen dependency ----
// Island entry is per-function `target("avx2,fma")` attributes (NOT
// target_clones FMV: FMV clones one body for every target and the
// default clone cannot compile AVX2 intrinsics; autovectorizing a
// scalar clone instead would discard the measured hand pipeline).
#if defined(__x86_64__) && defined(__GNUC__)
#define W46_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define W46_AVX2_TARGET
#endif
#if defined(__x86_64__) && defined(__GNUC__)
// 2^k for k in [-259, 3]: biased-exponent bit construction per 64-bit lane
W46_AVX2_TARGET inline __m256d w46_pow2(__m128i k32) {
  const __m128i bias = _mm_set1_epi32(1023);
  __m256i b64 = _mm256_cvtepi32_epi64(_mm_add_epi32(k32, bias));
  return _mm256_castsi256_pd(_mm256_slli_epi64(b64, 52));
}

W46_AVX2_TARGET inline __m256d w46_exp_negabs(__m256d xin) {
  const __m256d signbit = _mm256_set1_pd(-0.0);
  __m256d x = _mm256_or_pd(signbit, _mm256_andnot_pd(signbit, xin));  // -|x|
  x = _mm256_max_pd(x, _mm256_set1_pd(-709.784));
  __m256d fx = _mm256_floor_pd(_mm256_fmadd_pd(
      _mm256_set1_pd(1.4426950408889634073599), x, _mm256_set1_pd(0.5)));
  __m256d z = _mm256_fnmadd_pd(fx, _mm256_set1_pd(0.693145751953125), x);
  z = _mm256_fnmadd_pd(fx, _mm256_set1_pd(1.42860682030941723212e-6), z);
  const __m256d z2 = _mm256_mul_pd(z, z);
  __m256d px = _mm256_set1_pd(1.26177193074810590878e-4);
  px = _mm256_fmadd_pd(px, z2, _mm256_set1_pd(3.02994407707441961300e-2));
  px = _mm256_fmadd_pd(px, z2, _mm256_set1_pd(9.99999999999999999910e-1));
  px = _mm256_mul_pd(px, z);
  __m256d qx = _mm256_set1_pd(3.00198505138664455042e-6);
  qx = _mm256_fmadd_pd(qx, z2, _mm256_set1_pd(2.52448340349684104192e-3));
  qx = _mm256_fmadd_pd(qx, z2, _mm256_set1_pd(2.27265548208155028766e-1));
  qx = _mm256_fmadd_pd(qx, z2, _mm256_set1_pd(2.00000000000000000009e0));
  __m256d r = _mm256_div_pd(px, _mm256_sub_pd(qx, px));
  r = _mm256_fmadd_pd(_mm256_set1_pd(2.0), r, _mm256_set1_pd(1.0));
  __m128i e = _mm256_cvtpd_epi32(fx);           // 4x int32, in [-1023, 0]
  __m128i b = _mm_srai_epi32(e, 2);             // floor(e/4)
  __m256d c1 = w46_pow2(b);
  __m256d c2 = w46_pow2(_mm_shuffle_epi32(b, 0x4E));  // high half lanes
  __m256d cb = _mm256_insertf128_pd(c1, _mm256_castpd256_pd128(c2), 1);
  __m128i b2i = _mm_sub_epi32(_mm_sub_epi32(_mm_sub_epi32(e, b), b), b);
  __m256d d1 = w46_pow2(b2i);
  __m256d d2 = w46_pow2(_mm_shuffle_epi32(b2i, 0x4E));
  __m256d cd = _mm256_insertf128_pd(d1, _mm256_castpd256_pd128(d2), 1);
  // r * 2^b * 2^b * 2^b * 2^(e-3b) = r * 2^e (Eigen pldexp scheme)
  return _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(r, cb), cb), cb), cd);
}

W46_AVX2_TARGET inline __m256d w46_log1p_poly(__m256d w) {
  const __m256d half = _mm256_set1_pd(0.5), zero = _mm256_setzero_pd();
  const __m256d one = _mm256_set1_pd(1.0), two3 = _mm256_set1_pd(2.0 / 3.0);
  const __m256d quar = _mm256_set1_pd(0.25), four = _mm256_set1_pd(4.0);
  const __m256d anc = _mm256_set1_pd(kLn1pHalf), cf = _mm256_set1_pd(0.5);
  const __m256d small = _mm256_cmp_pd(w, half, _CMP_LT_OQ);
  const __m256d a = _mm256_blendv_pd(half, zero, small);
  const __m256d inv = _mm256_blendv_pd(two3, one, small);
  const __m256d ac = _mm256_blendv_pd(anc, zero, small);
  const __m256d u = _mm256_mul_pd(_mm256_sub_pd(w, a), inv);
  const __m256d z = _mm256_mul_pd(_mm256_sub_pd(u, quar), four);
  const __m256d zz = _mm256_add_pd(z, z);
  __m256d b1 = zero, b2 = zero;
  for (int k = 16; k >= 1; --k) {
    __m256d t = _mm256_fmadd_pd(zz, b1, _mm256_broadcast_sd(&kS[k]));
    t = _mm256_sub_pd(t, b2);
    b2 = b1;
    b1 = t;
  }
  __m256d S = _mm256_fmadd_pd(z, b1, _mm256_set1_pd(kS[0]));
  S = _mm256_sub_pd(S, b2);
  const __m256d u2 = _mm256_mul_pd(u, u);
  const __m256d r = _mm256_add_pd(ac, _mm256_fnmadd_pd(cf, u2, u));
  return _mm256_fmadd_pd(_mm256_mul_pd(u, u2), S, r);
}

W46_AVX2_TARGET inline double fwd_avx2(const double* x,
                                       const double* sg,
                                       int n, double* p) {
  const __m256d c20 = _mm256_set1_pd(20.0), nm20 = _mm256_set1_pd(-20.0);
  const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
  double s = 0.0;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m256d px = _mm256_loadu_pd(x + i);
    const __m256d sgv = _mm256_loadu_pd(sg + i);
    const __m256d w = w46_exp_negabs(px);
    const __m256d y = _mm256_add_pd(w, one);
    const __m256d l = w46_log1p_poly(w);
    const __m256d vm = _mm256_sub_pd(_mm256_min_pd(px, zero), l);
    const __m256d gt = _mm256_cmp_pd(px, c20, _CMP_GT_OQ);
    const __m256d lt = _mm256_cmp_pd(px, nm20, _CMP_LT_OQ);
    const __m256d nw = _mm256_xor_pd(w, _mm256_set1_pd(-0.0));
    const __m256d v = _mm256_blendv_pd(
        _mm256_blendv_pd(vm, px, lt), nw, gt);
    // partials: mid = sg * (x<0 ? 1-q : q), q = w/(1+w);
    // x<-20 -> sg; x>20 -> -w (stock bug-compat, no sg)
    const __m256d q = _mm256_div_pd(w, y);
    const __m256d pmid = _mm256_mul_pd(
        sgv, _mm256_blendv_pd(q, _mm256_sub_pd(one, q),
                              _mm256_cmp_pd(px, zero, _CMP_LT_OQ)));
    const __m256d pv = _mm256_blendv_pd(
        _mm256_blendv_pd(pmid, sgv, lt), nw, gt);
    _mm256_storeu_pd(p + i, pv);
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, v);
    s += (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
  }
  for (; i < n; ++i) {
    double pp;
    s += elem(x[i], sg[i], &pp);
    p[i] = pp;
  }
  return s;
}

inline double dispatch(const double* x, const double* sg, int n, double* p) {
  if (__builtin_cpu_supports("avx2")) {
    return fwd_avx2(x, sg, n, p);
  }
  return fwd_scalar(x, sg, n, p);
}

#else  // non-x86 / non-gcc: scalar only

inline double dispatch(const double* x, const double* sg, int n, double* p) {
  return fwd_scalar(x, sg, n, p);
}

#endif  // __x86_64__

// single runtime-dispatch boundary used by the lpmf below
inline double bernoulli_logit_fwd(const double* x, const double* sg,
                                  int n, double* p) {
  return dispatch(x, sg, n, p);
}
}  // namespace w46
}  // namespace internal
}  // namespace math
}  // namespace stan

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Returns the log PMF of the logit-parametrized Bernoulli distribution. If
 * containers are supplied, returns the log sum of the probabilities.
 *
 * @tparam T_n type of integer parameter
 * @tparam T_prob type of chance of success parameter
 * @param n integer parameter
 * @param theta logit-transformed chance of success parameter
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if theta is infinite.
 * @throw std::invalid_argument if container sizes mismatch.
 */
template <bool propto, typename T_n, typename T_prob,
          require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_n, T_prob>* = nullptr>
inline return_type_t<T_prob> bernoulli_logit_lpmf(const T_n& n,
                                                  const T_prob& theta) {
  using T_partials_return = partials_return_t<T_n, T_prob>;
  using T_partials_array = Eigen::Array<T_partials_return, Eigen::Dynamic, 1>;
  using std::exp;
  using T_n_ref = ref_type_if_not_constant_t<T_n>;
  using T_theta_ref = ref_type_if_not_constant_t<T_prob>;
  static constexpr const char* function = "bernoulli_logit_lpmf";
  check_consistent_sizes(function, "Random variable", n,
                         "Probability parameter", theta);
  if (size_zero(n, theta)) {
    return 0.0;
  }
  T_n_ref n_ref = n;
  T_theta_ref theta_ref = theta;
  check_bounded(function, "n", n_ref, 0, 1);

  decltype(auto) theta_val = to_ref(as_value_column_array_or_scalar(theta_ref));

  check_not_nan(function, "Logit transformed probability parameter", theta_val);
  if constexpr (!include_summand<propto, T_prob>::value) {
    return 0.0;
  }

  const auto& n_col = as_column_vector_or_scalar(n_ref);
  const auto& n_double = value_of_rec(n_col);

  auto signs = to_ref_if<is_autodiff_v<T_prob>>(
      (2 * as_array_or_scalar(n_double) - 1));
  T_partials_array ntheta;
  if constexpr (is_vector<T_n>::value || is_vector<T_prob>::value) {
    ntheta = signs * theta_val;
  } else {
    T_partials_return ntheta_s = signs * theta_val;
    ntheta = T_partials_array::Constant(1, 1, ntheta_s);
  }
  // W-103: fused kernel arm (T_partials_return == double: prim + rev).
  // Replaces exp() + both Select expressions for the double-partial
  // instantiations; same branch cuts at |ntheta| = 20; log1p argument
  // confined to [e^-20, 1] (see internal::w46 above). The partials are
  // BUG-COMPATIBLE with the stock expression (the ntheta > 20 branch is
  // -w WITHOUT the signs factor, exactly as stock; see W-46 section 5;
  // the patch must NOT fix that silently).
  if constexpr (std::is_same_v<T_partials_return, double>) {
    T_partials_array w46_partials = T_partials_array::Zero(ntheta.size());
    T_partials_return logp = 0.0;
    if constexpr (is_vector<T_n>::value || is_vector<T_prob>::value) {
      const auto signs_ref = to_ref(as_array_or_scalar(signs));
      logp = internal::w46::bernoulli_logit_fwd(
          ntheta.data(), signs_ref.data(), ntheta.size(),
          w46_partials.data());
    } else {
      const double sg = signs;
      logp = internal::w46::bernoulli_logit_fwd(ntheta.data(), &sg,
                                                ntheta.size(),
                                                w46_partials.data());
    }
    auto ops_partials = make_partials_propagator(theta_ref);
    if constexpr (is_autodiff_v<T_prob>) {
      edge<0>(ops_partials).partials_ = w46_partials;
    }
    return ops_partials.build(logp);
  }
  T_partials_array exp_m_ntheta = exp(-ntheta);
  static constexpr double cutoff = 20.0;
  T_partials_return logp = sum(
      (ntheta > cutoff)
          .select(-exp_m_ntheta,
                  (ntheta < -cutoff).select(ntheta, -log1p(exp_m_ntheta))));

  auto ops_partials = make_partials_propagator(theta_ref);
  if constexpr (is_autodiff_v<T_prob>) {
    edge<0>(ops_partials).partials_
        = (ntheta > cutoff)
              .select(
                  -exp_m_ntheta,
                  (ntheta >= -cutoff)
                      .select(promote_scalar<T_partials_return>(
                                  signs * exp_m_ntheta / (exp_m_ntheta + 1)),
                              promote_scalar<T_partials_return>(signs)));
  }
  return ops_partials.build(logp);
}

template <typename T_n, typename T_prob>
inline return_type_t<T_prob> bernoulli_logit_lpmf(const T_n& n,
                                                  const T_prob& theta) {
  return bernoulli_logit_lpmf<false>(n, theta);
}

}  // namespace math
}  // namespace stan
#endif
