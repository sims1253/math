#ifndef STAN_MATH_REV_CORE_RECOVER_MEMORY_HPP
#define STAN_MATH_REV_CORE_RECOVER_MEMORY_HPP

#include <stan/math/rev/core/vari.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/empty_nested.hpp>
#include <stdexcept>

namespace stan {
namespace math {

/**
 * Recover memory used for all variables for reuse.
 *
 * @throw std::logic_error if <code>empty_nested()</code> returns
 * <code>false</code>
 */
static inline void recover_memory() {
  if (!empty_nested()) {
    throw std::logic_error(
        "empty_nested() must be true"
        " before calling recover_memory()");
  }
  ChainableStack::instance_->var_stack_.clear();
  ChainableStack::instance_->var_nochain_stack_.clear();
  // W-53 research slice: span registry cleared with the per-vari
  // nochain stack; the records live in the arena freed by
  // recover_all().
  ChainableStack::instance_->var_nochain_spans_.clear();
  ChainableStack::instance_->nochain_span_records_ = 0;
  for (auto &x : ChainableStack::instance_->var_alloc_stack_) {
    delete x;
  }
  ChainableStack::instance_->var_alloc_stack_.clear();
  ChainableStack::instance_->memalloc_.recover_all();
}

}  // namespace math
}  // namespace stan
#endif
