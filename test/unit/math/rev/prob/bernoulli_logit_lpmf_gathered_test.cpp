#include <stan/math/rev/prob/bernoulli_logit_lpmf_gathered.hpp>
// the composed-stock reference path
#include <stan/math/prim/prob/bernoulli_logit_lpmf.hpp>
#include <stan/math/rev/fun/elt_multiply.hpp>
#include <stan/math/rev/functor/partials_propagator.hpp>
#include <stan/math/rev/core/operator_subtraction.hpp>

#include <gtest/gtest.h>
#include <random>
#include <vector>

// The gathered 2PL likelihood must be BIT-IDENTICAL to the expression the
// Stan compiler emits for y ~ bernoulli_logit(alpha[ii] .* (theta[jj] -
// beta[ii])): same per-element value ops (subtract, then multiply) and the
// same adjoint scatter order. Values and every gradient component are
// compared with memcmp, not a tolerance.
namespace {

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using stan::math::var;

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

// stand-in for stan::model::rvalue(x, name, index_multi(idx)) (which lives
// in the Stan repo, not math): the same indexed view of the same container
template <typename Vec, typename Idx>
auto gather(const Vec& x, const Idx& idx) {
  Eigen::Matrix<Eigen::Index, Dynamic, 1> idx0(idx.size());
  for (Eigen::Index k = 0; k < idx.size(); ++k) {
    idx0.coeffRef(k) = idx[k] - 1;
  }
  return x(idx0);
}

template <int layout>
struct Inputs {
  Matrix<var, Dynamic, 1> theta_a, alpha_a, beta_a;
  stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>> theta_s{
      Eigen::Matrix<double, Dynamic, 1>(0)};
  explicit Inputs(const Eigen::Matrix<double, Dynamic, 1>& t,
                  const Eigen::Matrix<double, Dynamic, 1>& a,
                  const Eigen::Matrix<double, Dynamic, 1>& b)
      : theta_a(t), alpha_a(a), beta_a(b) {
    if constexpr (layout == 1) {
      theta_s = stan::math::var_value<
          Eigen::Matrix<double, Dynamic, 1>>(t);
    }
  }
  Eigen::Matrix<double, Dynamic, 1> adj_theta() const {
    Eigen::Matrix<double, Dynamic, 1> g(theta_a.size());
    if constexpr (layout == 1) {
      for (Eigen::Index j = 0; j < g.size(); ++j) {
        g(j) = theta_s.vi_->adj_.coeff(j);
      }
    } else {
      for (Eigen::Index j = 0; j < g.size(); ++j) {
        g(j) = theta_a.coeff(j).adj();
      }
    }
    return g;
  }
  Eigen::Matrix<double, Dynamic, 1> adj_alpha() const {
    Eigen::Matrix<double, Dynamic, 1> g(alpha_a.size());
    for (Eigen::Index i = 0; i < g.size(); ++i) {
      g(i) = alpha_a.coeff(i).adj();
    }
    return g;
  }
  Eigen::Matrix<double, Dynamic, 1> adj_beta() const {
    Eigen::Matrix<double, Dynamic, 1> g(beta_a.size());
    for (Eigen::Index i = 0; i < g.size(); ++i) {
      g(i) = beta_a.coeff(i).adj();
    }
    return g;
  }
};

template <int layout>
var composed_stock(const std::vector<int>& y, const Inputs<layout>& in,
                   const std::vector<int>& ii, const std::vector<int>& jj) {
  return stan::math::bernoulli_logit_lpmf<false>(
      y, stan::math::elt_multiply(
             gather(in.alpha_a, ii),
             stan::math::subtract(gather(in.theta_a, jj),
                                  gather(in.beta_a, ii))));
}

template <int layout>
var primitive(const std::vector<int>& y, const Inputs<layout>& in,
              const std::vector<int>& ii, const std::vector<int>& jj) {
  if constexpr (layout == 1) {
    return stan::math::bernoulli_logit_lpmf_gathered<false>(
        y, in.theta_s, jj, in.alpha_a, in.beta_a, ii);
  } else {
    return stan::math::bernoulli_logit_lpmf_gathered<false>(
        y, in.theta_a, jj, in.alpha_a, in.beta_a, ii);
  }
}

template <int layout>
void expect_bit_identical(
    const Eigen::Matrix<double, Dynamic, 1>& theta,
    const Eigen::Matrix<double, Dynamic, 1>& alpha,
    const Eigen::Matrix<double, Dynamic, 1>& beta,
    const std::vector<int>& ii, const std::vector<int>& jj,
    const std::vector<int>& y) {
  double lp0 = 0.0, lp1 = 0.0;
  Eigen::Matrix<double, Dynamic, 1> gt0, ga0, gb0, gt1, ga1, gb1;
  {
    Inputs<0> in(theta, alpha, beta);  // composed stock, AoS operands
    var lp = composed_stock(y, in, ii, jj);
    lp0 = lp.val();
    lp.grad();
    gt0 = in.adj_theta();
    ga0 = in.adj_alpha();
    gb0 = in.adj_beta();
  }
  stan::math::recover_memory();
  {
    Inputs<layout> in(theta, alpha, beta);
    var lp = primitive(y, in, ii, jj);
    lp1 = lp.val();
    lp.grad();
    gt1 = in.adj_theta();
    ga1 = in.adj_alpha();
    gb1 = in.adj_beta();
  }
  stan::math::recover_memory();

  EXPECT_TRUE(bits_equal(lp0, lp1))
      << "log prob " << lp0 << " vs " << lp1;
  for (Eigen::Index j = 0; j < theta.size(); ++j) {
    EXPECT_TRUE(bits_equal(gt0(j), gt1(j)))
        << "d/dtheta(" << j << "): " << gt0(j) << " vs " << gt1(j);
  }
  for (Eigen::Index i = 0; i < alpha.size(); ++i) {
    EXPECT_TRUE(bits_equal(ga0(i), ga1(i)))
        << "d/dalpha(" << i << "): " << ga0(i) << " vs " << ga1(i);
    EXPECT_TRUE(bits_equal(gb0(i), gb1(i)))
        << "d/dbeta(" << i << "): " << gb0(i) << " vs " << gb1(i);
  }
}

}  // namespace

TEST(RevProbBernoulliLogitGathered, BitIdenticalToComposedStock) {
  std::mt19937 rng(20260819);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int rep = 0; rep < 8; ++rep) {
    const int I = 1 + static_cast<int>(rng() % 60);
    const int J = 1 + static_cast<int>(rng() % 60);
    const int N = 1 + static_cast<int>(rng() % 2000);
    const double scale = (rep % 3 == 2) ? 8.0 : 1.0;  // |ntheta| > 20 branch
    Eigen::Matrix<double, Dynamic, 1> theta(J), alpha(I), beta(I);
    for (int j = 0; j < J; ++j) {
      theta(j) = nd(rng) * scale;
    }
    for (int i = 0; i < I; ++i) {
      alpha(i) = std::fabs(nd(rng)) * scale + 0.1;
      beta(i) = nd(rng) * scale;
    }
    std::vector<int> ii(N), jj(N), y(N);
    for (int k = 0; k < N; ++k) {
      ii[k] = 1 + static_cast<int>(rng() % I);
      jj[k] = 1 + static_cast<int>(rng() % J);
      y[k] = static_cast<int>(rng() % 2);
    }
    // layout 0: all Matrix<var>; layout 1: theta as var_value<> (the
    // layout the generated hier_2pl model uses for a parameter vector)
    expect_bit_identical<0>(theta, alpha, beta, ii, jj, y);
    expect_bit_identical<1>(theta, alpha, beta, ii, jj, y);
  }
}

TEST(RevProbBernoulliLogitGathered, ScalarValueMatchesReference) {
  // hand-computed 2-point case (complete 1x2 grid), checks the assembled
  // predictor semantics eta = alpha[ii] * (theta[jj] - beta[ii])
  Eigen::Matrix<double, Dynamic, 1> theta(2), alpha(1), beta(1);
  theta << 1.7, -0.3;
  alpha << 0.8;
  beta << 0.5;
  std::vector<int> ii{1, 1}, jj{1, 2}, y{1, 0};
  Inputs<0> in(theta, alpha, beta);
  var lp = stan::math::bernoulli_logit_lpmf_gathered<true>(
      y, in.theta_a, jj, in.alpha_a, in.beta_a, ii);
  double eta1 = 0.8 * (1.7 - 0.5);
  double eta2 = 0.8 * (-0.3 - 0.5);
  // log p(y|x) = -log1p(exp(-x)) for y = 1, -log1p(exp(x)) for y = 0
  double expect = -std::log1p(std::exp(-eta1)) - std::log1p(std::exp(eta2));
  // compare at tight tolerance (the reference is a rearranged expression)
  EXPECT_NEAR(lp.val(), expect, 1e-12);
  lp.grad();
  // d log p / d eta_k = 1 - sigmoid(eta_k) for y = 1, -sigmoid(eta_k) for y = 0
  double d1 = 1.0 - 1.0 / (1.0 + std::exp(-eta1));
  double d2 = -1.0 / (1.0 + std::exp(-eta2));
  EXPECT_NEAR(in.alpha_a.coeff(0).adj(),
              d1 * (1.7 - 0.5) + d2 * (-0.3 - 0.5), 1e-12);
  EXPECT_NEAR(in.theta_a.coeff(0).adj(), d1 * 0.8, 1e-12);
  EXPECT_NEAR(in.theta_a.coeff(1).adj(), d2 * 0.8, 1e-12);
  EXPECT_NEAR(in.beta_a.coeff(0).adj(), -(d1 + d2) * 0.8, 1e-12);
  stan::math::recover_memory();
}

TEST(RevProbBernoulliLogitGathered, SizeZeroAndPropto) {
  Eigen::Matrix<double, Dynamic, 1> theta(0), alpha(1), beta(1);
  alpha << 1.0;
  beta << 0.0;
  Inputs<0> in(theta, alpha, beta);
  var lp = stan::math::bernoulli_logit_lpmf_gathered<true>(
      std::vector<int>{}, in.theta_a, std::vector<int>{}, in.alpha_a,
      in.beta_a, std::vector<int>{});
  EXPECT_EQ(lp.val(), 0.0);
  stan::math::recover_memory();
}
