#ifndef STAN_MATH_REV_FUN_DOT_SELF_GATHERED_DIFF_HPP
#define STAN_MATH_REV_FUN_DOT_SELF_GATHERED_DIFF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/as_array_or_scalar.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta/is_rev_matrix.hpp>

#include <vector>

namespace stan {
namespace math {

/** \ingroup rev_fun
 * Gathered ICAR quadratic form: the dot product with itself of the
 * index-assembled difference vector
 *
 *   d[e] = phi[node1[e]] - phi[node2[e]],  e = 0..E-1
 *
 * without materializing any gathered variable container. This is the
 * pattern the Stan compiler emits for the BYM2 ICAR prior
 * `target += -0.5 * dot_self(phi[node1] - phi[node2])`, where `node1` and
 * `node2` are data integer vectors listing the graph's edges (the scaling
 * stays OUTSIDE the primitive, exactly as in the generated code).
 *
 * The value path performs exactly the same floating point operations, in
 * the same per-element order, as the composed stock expression
 * `dot_self(subtract(rvalue(phi, index_multi(node1)),
 * rvalue(phi, index_multi(node2))))`:
 * - the gathers are plain copies of `phi`'s values,
 * - the subtraction is one `a - b` per edge,
 * - and the reduction is the one stock `dot_self` uses for the operand
 *   layout at hand: `v.val().dot(v.val())` (Eigen's redux over
 *   `binaryExpr<scalar_conj_product_op>` -- the identical template
 *   instantiation is called here for `var_value<>` operands) for
 *   `var_value<>` (SoA) vectors, and the sequential
 *   `res += x * x` accumulation loop for `Matrix<var>` (AoS) vectors.
 *
 * The reverse pass is ONE callback that performs the same scatter-adds,
 * in the same order, that the stock chain performs through its callbacks
 * (which run LIFO: dot_self's, then subtract's, then the second gather's
 * scatter, then the first gather's scatter):
 * - SoA operands: all node2 contributions first (edge order), then all
 *   node1 contributions (edge order) -- the order the two
 *   `rvalue(var_value, index_multi)` reverse callbacks fire in;
 * - AoS operands: per edge, node1 `+=` then node2 `-=` -- the order
 *   `subtract`'s callback scatters in when the gathered views alias
 *   phi's own varis (the gathered `Matrix<var>` copies share records, so
 *   no gather callback exists).
 * In both cases the per-edge adjoint increment is `2*w*d[e]` with `w` the
 * adjoint of the returned var and the `2*w` product computed once, exactly
 * as stock's `2.0 * res.adj()` is. Values and every gradient component
 * are bit-identical to the composed stock path.
 *
 * Bounds checking mirrors the stock `rvalue` loops: every `node1` index is
 * checked (ascending), then every `node2` index, and the first offender
 * throws through the same `check_range` the corresponding layout's
 * `rvalue` uses (the exception's function string matches the layout:
 * "vector[multi] assign range" for `var_value<>`, "vector[multi]
 * indexing" for `Matrix<var>`; the container name in the message is
 * "phi").
 *
 * @tparam T_phi a vector of vars: `Matrix<var,-1,1>` (AoS) or
 * `var_value<Eigen::VectorXd>` (SoA, what the deserializer produces for a
 * parameter vector)
 * @tparam T_n1 type of the first edge index vector (integer vector-like)
 * @tparam T_n2 type of the second edge index vector (integer vector-like)
 * @param phi the spatial random effect vector
 * @param node1 first endpoint of each edge (1-based)
 * @param node2 second endpoint of each edge (1-based)
 * @return var holding sum_e (phi[node1[e]] - phi[node2[e]])^2
 * @throw std::invalid_argument if the index vector sizes differ
 * @throw std::out_of_range if an index is out of range
 */
template <typename T_phi, typename T_n1, typename T_n2,
          require_any_t<is_rev_col_vector<T_phi>, is_rev_row_vector<T_phi>>* =
              nullptr,
          require_vector_like_vt<std::is_integral, T_n1>* = nullptr,
          require_vector_like_vt<std::is_integral, T_n2>* = nullptr>
inline var dot_self_gathered_diff(const T_phi& phi, const T_n1& node1,
                                  const T_n2& node2) {
  using VecD = Eigen::Matrix<double, Eigen::Dynamic, 1>;
  using vari_vec = Eigen::Matrix<vari*, Eigen::Dynamic, 1>;
  using soa_vec_vari = vari_value<VecD>;
  static constexpr const char* function = "dot_self_gathered_diff";
  // the exception function string of the rvalue overload the composed
  // stock expression would take for this operand layout
  static constexpr const char* rvalue_fn =
      is_var_v<std::decay_t<T_phi>> ? "vector[multi] assign range"
                                    : "vector[multi] indexing";

  const Eigen::Index E = stan::math::size(node1);
  check_size_match(function, "First index vector size", E,
                   "Second index vector size", stan::math::size(node2));
  if (unlikely(E == 0)) {
    return var(0.0);
  }

  // Bounds-check + gather pass, in stock rvalue order (all of node1, then
  // all of node2). The gathered values are plain copies, so this loop is
  // also where the per-edge subtraction happens (bit-exact: one a-b per
  // edge, same operands as stock's subtract).
  const VecD phi_d = value_of(phi);
  const Eigen::Index n_phi = phi.size();
  const int n_phi_int = static_cast<int>(n_phi);
  const int* n1_data = as_array_or_scalar(node1).data();
  const int* n2_data = as_array_or_scalar(node2).data();
  arena_t<VecD> d_val(E);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> n1_arena(E);
  arena_t<Eigen::Matrix<int, Eigen::Dynamic, 1>> n2_arena(E);
  for (Eigen::Index e = 0; e < E; ++e) {
    check_range(rvalue_fn, "phi", n_phi_int, n1_data[e]);
    n1_arena.coeffRef(e) = n1_data[e] - 1;
  }
  for (Eigen::Index e = 0; e < E; ++e) {
    check_range(rvalue_fn, "phi", n_phi_int, n2_data[e]);
    n2_arena.coeffRef(e) = n2_data[e] - 1;
    d_val.coeffRef(e)
        = phi_d.coeff(n1_arena.coeff(e)) - phi_d.coeff(n2_arena.coeff(e));
  }

  // Adjoint routes: one vari* per element for Matrix<var> (AoS) operands
  // (the gathered stock records alias these), or the single matrix vari
  // for var_value<> (SoA) operands.
  [[maybe_unused]] arena_t<vari_vec> phi_vi(0);
  [[maybe_unused]] soa_vec_vari* phi_soa = nullptr;
  if constexpr (is_var_v<std::decay_t<T_phi>>) {
    phi_soa = phi.vi_;
  } else {
    phi_vi = arena_t<vari_vec>(n_phi);
    for (Eigen::Index k = 0; k < n_phi; ++k) {
      phi_vi.coeffRef(k) = phi.coeff(k).vi_;
    }
  }

  if constexpr (is_var_v<std::decay_t<T_phi>>) {
    // SoA: stock's dot_self(var_value) reduces with
    // v.val().dot(v.val()); d_val is the same arena_matrix type, so this
    // is the identical Eigen template instantiation on the same values.
    var res = d_val.dot(d_val);
    return make_callback_var(
        res.val(), [phi_soa, d_val, n1_arena, n2_arena](const auto& vi) {
          const double s = 2.0 * vi.adj_;
          // stock order (LIFO): GCC evaluates subtract's arguments
          // right-to-left, so the node2 gather's reverse callback is
          // registered FIRST and runs LAST -- the node1 scatter comes first
          for (Eigen::Index e = 0; e < d_val.size(); ++e) {
            phi_soa->adj_.coeffRef(n1_arena.coeff(e)) += s * d_val.coeff(e);
          }
          for (Eigen::Index e = 0; e < d_val.size(); ++e) {
            phi_soa->adj_.coeffRef(n2_arena.coeff(e)) -= s * d_val.coeff(e);
          }
        });
  } else {
    // AoS: stock's dot_self(Matrix<var>) accumulates sequentially
    double res_val = 0;
    for (Eigen::Index i = 0; i < d_val.size(); ++i) {
      res_val += d_val.coeffRef(i) * d_val.coeffRef(i);
    }
    return make_callback_var(
        res_val, [phi_vi, d_val, n1_arena, n2_arena](const auto& vi) {
          const double s = 2.0 * vi.adj_;
          // stock subtract callback order: per edge, node1 += then
          // node2 -= (the gathered views alias phi's varis)
          for (Eigen::Index e = 0; e < d_val.size(); ++e) {
            const double d2 = s * d_val.coeff(e);
            phi_vi.coeff(n1_arena.coeff(e))->adj_ += d2;
            phi_vi.coeff(n2_arena.coeff(e))->adj_ -= d2;
          }
        });
  }
}

}  // namespace math
}  // namespace stan
#endif
