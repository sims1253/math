#ifndef STAN_MATH_REV_CORE_SET_ZERO_ALL_ADJOINTS_HPP
#define STAN_MATH_REV_CORE_SET_ZERO_ALL_ADJOINTS_HPP

#include <stan/math/rev/core/vari.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>

namespace stan {
namespace math {

/**
 * Reset all adjoint values in the stack to zero.
 */
static inline void set_zero_all_adjoints() {
  for (auto &x : ChainableStack::instance_->var_stack_) {
    x->set_zero_adjoint();
  }
  for (auto &x : ChainableStack::instance_->var_nochain_stack_) {
    x->set_zero_adjoint();
  }
  // W-53 research slice: batch-registered nochain records. Spans are
  // only ever created by make_nochain_vari_array, whose records are
  // vari_value<double>; the vari* cast plus the `final`
  // set_zero_adjoint devirtualizes to the same single store the
  // per-vari loop performs on stock records.
  for (auto &span : ChainableStack::instance_->var_nochain_spans_) {
    char *p = reinterpret_cast<char *>(span.begin);
    for (size_t k = 0; k < span.count; ++k) {
      reinterpret_cast<vari *>(p)->set_zero_adjoint();
      p += span.stride;
    }
  }
}

}  // namespace math
}  // namespace stan
#endif
