#ifndef STAN_MATH_REV_CORE_SET_ZERO_ALL_ADJOINTS_NESTED_HPP
#define STAN_MATH_REV_CORE_SET_ZERO_ALL_ADJOINTS_NESTED_HPP

#include <stan/math/rev/core/vari.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/empty_nested.hpp>
#include <stdexcept>

namespace stan {
namespace math {

/**
 * Reset all adjoint values in the top nested portion of the stack
 * to zero.
 *
 * It is preferred to use the <code>nested_rev_autodiff</code> class for
 * nested autodiff class as it handles recovery of memory automatically.
 */
static inline void set_zero_all_adjoints_nested() {
  if (empty_nested()) {
    throw std::logic_error(
        "empty_nested() must be false before calling"
        " set_zero_all_adjoints_nested()");
  }
  const size_t start1
      = ChainableStack::instance_->nested_var_stack_sizes_.back();
  // avoid wrap with unsigned when start1 == 0
  for (size_t i = start1; i < ChainableStack::instance_->var_stack_.size();
       ++i) {
    ChainableStack::instance_->var_stack_[i]->set_zero_adjoint();
  }

  const size_t start2
      = ChainableStack::instance_->nested_var_nochain_stack_sizes_.back();
  for (size_t i = start2;
       i < ChainableStack::instance_->var_nochain_stack_.size(); ++i) {
    ChainableStack::instance_->var_nochain_stack_[i]->set_zero_adjoint();
  }
  // W-53 research slice: nested portion of the batch-registered spans.
  const size_t span_start
      = ChainableStack::instance_->nested_var_nochain_span_sizes_.back();
  for (size_t s = span_start;
       s < ChainableStack::instance_->var_nochain_spans_.size(); ++s) {
    const auto &span = ChainableStack::instance_->var_nochain_spans_[s];
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
