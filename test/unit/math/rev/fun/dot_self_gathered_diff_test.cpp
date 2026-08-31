#include <stan/math/rev/fun/dot_self_gathered_diff.hpp>
// the composed-stock reference path
#include <stan/math/rev/core/operator_subtraction.hpp>
#include <stan/math/rev/fun/dot_self.hpp>

#include <gtest/gtest.h>
#include <random>
#include <vector>

// The gathered ICAR quadratic form must be BIT-IDENTICAL to the expression
// the Stan compiler emits for target += -0.5 * dot_self(phi[node1] -
// phi[node2]): same per-element subtraction order, the same reduction stock
// dot_self uses for the operand layout, and the same adjoint scatter order
// (which DIFFERS by layout: interleaved per edge for Matrix<var> operands,
// two-pass -- all node2, then all node1 -- for var_value<> operands, the
// order the two rvalue reverse callbacks fire in). Values and every
// gradient component are compared with memcmp, not a tolerance.
namespace {

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::math::var;
using VecD = Eigen::Matrix<double, Dynamic, 1>;
using soa_vec = stan::math::var_value<VecD>;

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

// stand-in for stan::model::rvalue(x, name, index_multi(idx)) for a
// Matrix<var> (which lives in the Stan repo, not math): the same indexed
// view of the same container
template <typename EigVec, typename Idx>
auto gather_aos(const EigVec& x, const Idx& idx) {
  Eigen::Matrix<Eigen::Index, Dynamic, 1> idx0(idx.size());
  for (Eigen::Index k = 0; k < idx.size(); ++k) {
    idx0.coeffRef(k) = idx[k] - 1;
  }
  return x(idx0);
}

// stand-in for stan::model::rvalue(var_value, name, index_multi(idx))
// (rvalue_varmat.hpp in the Stan repo): the stock SoA multi-index gather,
// copied verbatim -- an eager copy of the gathered values plus ONE reverse
// scatter callback over the index vector
template <typename Idx>
soa_vec gather_soa(const soa_vec& x, const Idx& idx) {
  using stan::math::arena_allocator;
  using stan::arena_t;
  using stan::math::check_range;
  using stan::math::reverse_pass_callback;
  using arena_std_vec = std::vector<int, arena_allocator<int>>;
  const Eigen::Index x_size = x.size();
  const auto ret_size = idx.size();
  arena_t<VecD> x_ret_vals(ret_size);
  arena_std_vec row_idx(ret_size);
  for (int i = 0; i < ret_size; ++i) {
    check_range("vector[multi] assign range", "phi", x_size, idx[i]);
    row_idx[i] = idx[i] - 1;
    x_ret_vals.coeffRef(i) = x.vi_->val_.coeff(row_idx[i]);
  }
  soa_vec x_ret(x_ret_vals);
  reverse_pass_callback([x, x_ret, row_idx]() mutable {
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(row_idx.size());
         ++i) {
      x.adj().coeffRef(row_idx[i]) += x_ret.adj().coeff(i);
    }
  });
  return x_ret;
}

template <int layout>
var composed_stock(const soa_vec& phi_s, const Matrix<var, Dynamic, 1>& phi_a,
                   const std::vector<int>& n1, const std::vector<int>& n2) {
  if constexpr (layout == 1) {
    return stan::math::dot_self(stan::math::subtract(
        gather_soa(phi_s, n1), gather_soa(phi_s, n2)));
  } else {
    return stan::math::dot_self(
        stan::math::subtract(gather_aos(phi_a, n1), gather_aos(phi_a, n2)));
  }
}

template <int layout>
var primitive(const soa_vec& phi_s, const Matrix<var, Dynamic, 1>& phi_a,
              const std::vector<int>& n1, const std::vector<int>& n2) {
  if constexpr (layout == 1) {
    return stan::math::dot_self_gathered_diff(phi_s, n1, n2);
  } else {
    return stan::math::dot_self_gathered_diff(phi_a, n1, n2);
  }
}

template <int layout>
void expect_bit_identical(const VecD& phi_d, const std::vector<int>& n1,
                          const std::vector<int>& n2) {
  const Eigen::Index N = phi_d.size();
  double v0 = 0.0, v1 = 0.0;
  VecD g0(N), g1(N);
  {
    soa_vec phi_s(phi_d);
    Matrix<var, Dynamic, 1> phi_a(phi_d);
    var r = composed_stock<layout>(phi_s, phi_a, n1, n2);
    v0 = r.val();
    r.grad();
    for (Eigen::Index j = 0; j < N; ++j) {
      if constexpr (layout == 1) {
        g0(j) = phi_s.vi_->adj_.coeff(j);
      } else {
        g0(j) = phi_a.coeff(j).adj();
      }
    }
  }
  stan::math::recover_memory();
  {
    soa_vec phi_s(phi_d);
    Matrix<var, Dynamic, 1> phi_a(phi_d);
    var r = primitive<layout>(phi_s, phi_a, n1, n2);
    v1 = r.val();
    r.grad();
    for (Eigen::Index j = 0; j < N; ++j) {
      if constexpr (layout == 1) {
        g1(j) = phi_s.vi_->adj_.coeff(j);
      } else {
        g1(j) = phi_a.coeff(j).adj();
      }
    }
  }
  stan::math::recover_memory();

  EXPECT_TRUE(bits_equal(v0, v1)) << "value " << v0 << " vs " << v1;
  for (Eigen::Index j = 0; j < N; ++j) {
    EXPECT_TRUE(bits_equal(g0(j), g1(j)))
        << "d/dphi(" << j << "): " << g0(j) << " vs " << g1(j);
  }
}

}  // namespace

TEST(RevFunDotSelfGatheredDiff, BitIdenticalToComposedStock) {
  std::mt19937 rng(20260819);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int rep = 0; rep < 8; ++rep) {
    const int N = 2 + static_cast<int>(rng() % 300);
    const int E = 1 + static_cast<int>(rng() % 3 * N / 2);
    const double scale = (rep % 3 == 2) ? 5.0 : 1.0;
    VecD phi_d(N);
    for (int j = 0; j < N; ++j) {
      phi_d(j) = nd(rng) * scale;
    }
    std::vector<int> n1(E), n2(E);
    for (int e = 0; e < E; ++e) {
      n1[e] = 1 + static_cast<int>(rng() % N);
      n2[e] = 1 + static_cast<int>(rng() % N);
    }
    // layout 0: phi as Matrix<var>; layout 1: phi as var_value<> (the
    // layout the generated model uses for a parameter vector). The two
    // layouts have DIFFERENT stock scatter schedules, so each is compared
    // against the composed stock of its own layout.
    expect_bit_identical<0>(phi_d, n1, n2);
    expect_bit_identical<1>(phi_d, n1, n2);
  }
  // tiny graphs (E = 1..4) and self edges (d = 0)
  {
    VecD phi_d(3);
    phi_d << 0.3, -1.2, 2.5;
    expect_bit_identical<0>(phi_d, {1}, {2});
    expect_bit_identical<1>(phi_d, {1}, {2});
    expect_bit_identical<0>(phi_d, {3, 1, 2, 3}, {2, 3, 1, 3});
    expect_bit_identical<1>(phi_d, {3, 1, 2, 3}, {2, 3, 1, 3});
  }
}

TEST(RevFunDotSelfGatheredDiff, ValueMatchesHandComputed) {
  // d = (1.7 - (-0.3), 0.0, 2.5 - 1.7, 2.5 - 1.7) = (2, 0, .8, .8)
  // dot_self = 4 + 0 + .64 + .64
  VecD phi_d(3);
  phi_d << 1.7, -0.3, 2.5;
  std::vector<int> n1{1, 2, 3, 3}, n2{2, 2, 1, 1};
  {
    soa_vec phi_s(phi_d);
    var r = stan::math::dot_self_gathered_diff(phi_s, n1, n2);
    EXPECT_DOUBLE_EQ(r.val(), 4.0 + 0.64 + 0.64);
    r.grad();
    // d/dphi[j] = 2*sum_e (phi[n1]-phi[n2])*(1[n1=j] - 1[n2=j])
    EXPECT_DOUBLE_EQ(phi_s.vi_->adj_.coeff(0), 2 * 2.0 - 2 * 0.8 - 2 * 0.8);
    EXPECT_DOUBLE_EQ(phi_s.vi_->adj_.coeff(1), -4.0);
    EXPECT_DOUBLE_EQ(phi_s.vi_->adj_.coeff(2), 2 * 0.8 + 2 * 0.8);
  }
  stan::math::recover_memory();
  {
    Matrix<var, Dynamic, 1> phi_a(phi_d);
    var r = stan::math::dot_self_gathered_diff(phi_a, n1, n2);
    EXPECT_DOUBLE_EQ(r.val(), 4.0 + 0.64 + 0.64);
    r.grad();
    EXPECT_DOUBLE_EQ(phi_a.coeff(0).adj(), 2 * 2.0 - 2 * 0.8 - 2 * 0.8);
    EXPECT_DOUBLE_EQ(phi_a.coeff(1).adj(), -4.0);
    EXPECT_DOUBLE_EQ(phi_a.coeff(2).adj(), 2 * 0.8 + 2 * 0.8);
  }
  stan::math::recover_memory();
}

TEST(RevFunDotSelfGatheredDiff, SizeZeroAndBounds) {
  {
    soa_vec phi_s(VecD::Zero(3));
    var r = stan::math::dot_self_gathered_diff(
        phi_s, std::vector<int>{}, std::vector<int>{});
    EXPECT_EQ(r.val(), 0.0);
    stan::math::recover_memory();
  }
  {
    Matrix<var, Dynamic, 1> phi_a(VecD::Zero(3));
    var r = stan::math::dot_self_gathered_diff(
        phi_a, std::vector<int>{}, std::vector<int>{});
    EXPECT_EQ(r.val(), 0.0);
    stan::math::recover_memory();
  }
  // size mismatch throws invalid_argument
  {
    soa_vec phi_s(VecD::Zero(3));
    EXPECT_THROW(stan::math::dot_self_gathered_diff(
                     phi_s, std::vector<int>{1}, std::vector<int>{1, 2}),
                 std::invalid_argument);
    stan::math::recover_memory();
  }
  // first-offender bounds semantics: node1 is checked before node2,
  // ascending, exactly as the two rvalue calls would
  {
    Matrix<var, Dynamic, 1> phi_a(VecD::Zero(3));
    EXPECT_THROW(stan::math::dot_self_gathered_diff(
                     phi_a, std::vector<int>{1, 9}, std::vector<int>{2, 8}),
                 std::out_of_range);
    stan::math::recover_memory();
    soa_vec phi_s(VecD::Zero(3));
    EXPECT_THROW(stan::math::dot_self_gathered_diff(
                     phi_s, std::vector<int>{1, 2}, std::vector<int>{0, 2}),
                 std::out_of_range);
    stan::math::recover_memory();
  }
}
