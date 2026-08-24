#ifndef STAN_MATH_REV_CORE_MAKE_NOCHAIN_VARI_ARRAY_HPP
#define STAN_MATH_REV_CORE_MAKE_NOCHAIN_VARI_ARRAY_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/vari.hpp>
#include <new>

namespace stan {
namespace math {

/**
 * W-53 research slice: batch-construct `n` nochain vari records for
 * the values of a double Eigen expression, in ONE arena allocation,
 * and register them as ONE nochain span.
 *
 * Each record is a `vari_value<double>` constructed by placement-new
 * with value `expr.coeff(i)` and adjoint 0.0 - field-for-field
 * identical to what the stock per-element path (a
 * `vari_value<double>(expr.coeff(i), false)` construction inside the
 * Eigen double-to-var assignment) produces: same vptr, same value,
 * same zero adjoint, same arena lifetime, same set_zero and recover
 * coverage. What differs (and is the point):
 *   1. ONE memalloc_.alloc call for the whole array instead of one
 *      24-byte allocation per record;
 *   2. ONE span entry on ChainableStack instead of one
 *      var_nochain_stack_ push_back per record.
 *
 * The returned pointer is the first record; element i is at recs+i.
 * W-59: one pass constructs each record AND fills the caller's output
 * Matrix<var> data pointer (`out[i] = var(recs + i)`) — the fused
 * loop removes the second pointer-fill pass the call sites used to
 * run separately (rationale + measurements: see WORKLOG W-59).
 */
template <typename Eig, require_eigen_t<Eig>* = nullptr,
          require_st_arithmetic<Eig>* = nullptr>
inline vari* make_nochain_vari_array(const Eig& expr, var* out) {
  const Eigen::Index n = expr.size();
  if (n == 0) {
    return nullptr;
  }
  vari* recs = reinterpret_cast<vari*>(
      ChainableStack::instance_->memalloc_.alloc_array<char>(
          n * static_cast<Eigen::Index>(sizeof(vari))));
  // W-58: 2D fallback for expressions without LinearAccessBit (Eigen 5
  // static-asserts linear coeff on general 2D binary ops; values identical —
  // coefficientwise per-element evaluation).
  if constexpr ((Eig::Flags & Eigen::LinearAccessBit) != 0) {
    // hot path (linear walk — the already-gated batch-0/1 path)
    for (Eigen::Index i = 0; i < n; ++i) {
      ::new (static_cast<void*>(recs + i)) vari(expr.coeff(i), vari_no_stack);
      out[i] = var(recs + i);
    }
  } else {
    // 2D col-major walk: same per-element computation as stock's Eigen
    // assignment loop; record k sits at element (k % rows, k / rows)
    const Eigen::Index nr = expr.rows();
    for (Eigen::Index j = 0; j < expr.cols(); ++j) {
      for (Eigen::Index i = 0; i < nr; ++i) {
        ::new (static_cast<void*>(recs + i + j * nr))
            vari(expr.coeff(i, j), vari_no_stack);
        out[i + j * nr] = var(recs + i + j * nr);
      }
    }
  }
  ChainableStack::instance_->var_nochain_spans_.push_back(
      ChainableStack::AutodiffStackStorage::NoChainSpan{
          recs, static_cast<size_t>(n), sizeof(vari)});
  ChainableStack::instance_->nochain_span_records_
      += static_cast<size_t>(n);
  return recs;
}

}  // namespace math
}  // namespace stan
#endif
