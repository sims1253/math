#include <stan/math/rev/prob/bernoulli_logit_lpmf_gathered.hpp>
#include <stan/math/prim/prob/normal_lpdf.hpp>
// the composed-stock reference path
#include <stan/math/prim/prob/bernoulli_logit_lpmf.hpp>
#include <stan/math/rev/fun/elt_multiply.hpp>
#include <stan/math/rev/functor/partials_propagator.hpp>
#include <stan/math/rev/core/operator_subtraction.hpp>

#include <gtest/gtest.h>
#include <limits>
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
  // layout 2: theta viewed as the deserializer's default-level Map over the
  // AoS var buffer
  Eigen::Map<const Matrix<var, Dynamic, 1>> theta_map() const {
    return Eigen::Map<const Matrix<var, Dynamic, 1>>(theta_a.data(),
                                                     theta_a.size());
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
  if constexpr (layout == 2) {
    return stan::math::bernoulli_logit_lpmf<false>(
        y, stan::math::elt_multiply(
               gather(in.alpha_a, ii),
               stan::math::subtract(gather(in.theta_map(), jj),
                                    gather(in.beta_a, ii))));
  } else {
    return stan::math::bernoulli_logit_lpmf<false>(
        y, stan::math::elt_multiply(
               gather(in.alpha_a, ii),
               stan::math::subtract(gather(in.theta_a, jj),
                                    gather(in.beta_a, ii))));
  }
}

template <int layout>
var primitive(const std::vector<int>& y, const Inputs<layout>& in,
              const std::vector<int>& ii, const std::vector<int>& jj) {
  if constexpr (layout == 1) {
    return stan::math::bernoulli_logit_lpmf_gathered<false>(
        y, in.theta_s, jj, in.alpha_a, in.beta_a, ii);
  } else if constexpr (layout == 2) {
    return stan::math::bernoulli_logit_lpmf_gathered<false>(
        y, in.theta_map(), jj, in.alpha_a, in.beta_a, ii);
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
    // layout an O1-level hpp uses for a parameter vector); layout 2:
    // theta as Map<const Matrix<var>> (the DEFAULT-level deserializer
    // layout, the W-108.1 case)
    expect_bit_identical<0>(theta, alpha, beta, ii, jj, y);
    expect_bit_identical<1>(theta, alpha, beta, ii, jj, y);
    expect_bit_identical<2>(theta, alpha, beta, ii, jj, y);
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

// ---------------------------------------------------------------------------
// W-127: the additive multi-gather predictor (eta = intercept + data-product
// terms + gathered coefficient terms), bitwise vs the composed scalar chain
// the stanc tp-loop emits for the election88 class.
// ---------------------------------------------------------------------------
namespace {

// the composed per-element expression (the hpp's tp-loop line), AoS operands
template <typename Vec>
var add_leaf_chain(const var& intercept, const var& c2,
                   const std::vector<double>& xd2, const var& c3,
                   const std::vector<double>& xd3, const var& c5,
                   const std::vector<double>& xd31,
                   const std::vector<double>& xd32, const Vec& g0,
                   const std::vector<int>& i0, const Vec& g1,
                   const std::vector<int>& i1, const Vec& g2,
                   const std::vector<int>& i2, Eigen::Index k) {
  return ((((((intercept + (c2 * xd2[k])) + (c3 * xd3[k])) +
             ((c5 * xd31[k]) * xd32[k])) +
            g0.coeff(i0[k] - 1)) +
           g1.coeff(i1[k] - 1)) +
          g2.coeff(i2[k] - 1));
}

template <typename T>
struct AdditiveInputs {
  Matrix<var, Dynamic, 1> g0_a, g1_a, g2_a;
  stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>> g0_s{
      Eigen::Matrix<double, Dynamic, 1>(0)};
  stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>> g1_s{
      Eigen::Matrix<double, Dynamic, 1>(0)};
  stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>> g2_s{
      Eigen::Matrix<double, Dynamic, 1>(0)};
  AdditiveInputs(const Eigen::Matrix<double, Dynamic, 1>& a,
                 const Eigen::Matrix<double, Dynamic, 1>& b,
                 const Eigen::Matrix<double, Dynamic, 1>& c)
      : g0_a(a), g1_a(b), g2_a(c) {
    if constexpr (T::value >= 1) {
      g0_s = stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>>(a);
      g1_s = stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>>(b);
      g2_s = stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>>(c);
    }
  }
  auto g0() const {
    if constexpr (T::value == 1) return g0_s;
    else if constexpr (T::value == 2)
      return Eigen::Map<const Matrix<var, Dynamic, 1>>(g0_a.data(),
                                                       g0_a.size());
    else return (const Matrix<var, Dynamic, 1>&)g0_a;
  }
  auto g1() const {
    if constexpr (T::value == 1) return g1_s;
    else if constexpr (T::value == 2)
      return Eigen::Map<const Matrix<var, Dynamic, 1>>(g1_a.data(),
                                                       g1_a.size());
    else return (const Matrix<var, Dynamic, 1>&)g1_a;
  }
  auto g2() const {
    if constexpr (T::value == 1) return g2_s;
    else if constexpr (T::value == 2)
      return Eigen::Map<const Matrix<var, Dynamic, 1>>(g2_a.data(),
                                                       g2_a.size());
    else return (const Matrix<var, Dynamic, 1>&)g2_a;
  }
  Eigen::Matrix<double, Dynamic, 1> adj(int which) const {
    const auto& aos = which == 0 ? g0_a : (which == 1 ? g1_a : g2_a);
    Eigen::Matrix<double, Dynamic, 1> g(aos.size());
    if constexpr (T::value == 1) {
      const auto& soa = which == 0 ? g0_s : (which == 1 ? g1_s : g2_s);
      for (Eigen::Index j = 0; j < g.size(); ++j) {
        g(j) = soa.vi_->adj_.coeff(j);
      }
    } else {
      for (Eigen::Index j = 0; j < g.size(); ++j) {
        g(j) = aos.coeff(j).adj();
      }
    }
    return g;
  }
};

struct AdditiveCase {
  Eigen::Matrix<double, Dynamic, 1> g0, g1, g2, beta;
  std::vector<int> i0, i1, i2, y;
  std::vector<double> xd2, xd3, xd31, xd32;
};

AdditiveCase make_additive_case(std::mt19937& rng, int N, double scale) {
  AdditiveCase c;
  auto nd = [&](double s) { return std::normal_distribution<double>(0, s)(rng); };
  c.g0.resize(3);
  c.g1.resize(5);
  c.g2.resize(9);
  c.beta.resize(5);
  for (auto& x : c.g0) x = nd(scale);
  for (auto& x : c.g1) x = nd(scale);
  for (auto& x : c.g2) x = nd(scale);
  for (auto& x : c.beta) x = nd(scale);
  c.i0.resize(N);
  c.i1.resize(N);
  c.i2.resize(N);
  c.y.resize(N);
  c.xd2.resize(N);
  c.xd3.resize(N);
  c.xd31.resize(N);
  c.xd32.resize(N);
  for (int k = 0; k < N; ++k) {
    c.i0[k] = 1 + static_cast<int>(rng() % 3);
    c.i1[k] = 1 + static_cast<int>(rng() % 5);
    c.i2[k] = 1 + static_cast<int>(rng() % 9);
    c.y[k] = static_cast<int>(rng() % 2);
    c.xd2[k] = nd(scale);             // real
    c.xd3[k] = static_cast<double>(rng() % 2);  // binary (the 1.0 alias)
    c.xd31[k] = static_cast<double>(rng() % 2); // binary (alias on m1)
    c.xd32[k] = nd(scale);            // real
  }
  return c;
}

template <typename Vec>
Eigen::Matrix<double, Dynamic, 1> adj_of(const Vec& v) {
  Eigen::Matrix<double, Dynamic, 1> g(v.size());
  for (Eigen::Index j = 0; j < g.size(); ++j) {
    g(j) = v.coeff(j).adj();
  }
  return g;
}

template <int N_FRAC>
void expect_additive_bit_identical(const AdditiveCase& c, int layout) {
  double lp0 = 0.0, lp1 = 0.0;
  Eigen::Matrix<double, Dynamic, 1> a00, a01, a02, b0, a10, a11, a12, b1;
  {
    Matrix<var, Dynamic, 1> beta(c.beta);
    Matrix<var, Dynamic, 1> g0(c.g0), g1(c.g1), g2(c.g2);
    var intercept = beta.coeff(0);
    var c2 = beta.coeff(1), c3 = beta.coeff(2), c5 = beta.coeff(4);
    Matrix<var, Dynamic, 1> y_hat(c.y.size());
    for (Eigen::Index k = 0; k < y_hat.size(); ++k) {
      y_hat.coeffRef(k) = add_leaf_chain(
          intercept, c2, c.xd2, c3, c.xd3, c5, c.xd31, c.xd32, g0, c.i0, g1,
          c.i1, g2, c.i2, k);
    }
    var lp = stan::math::bernoulli_logit_lpmf<false>(c.y, y_hat);
    lp0 = lp.val();
    lp.grad();
    a00 = adj_of(g0);
    a01 = adj_of(g1);
    a02 = adj_of(g2);
    b0 = adj_of(beta);
  }
  stan::math::recover_memory();
  {
    using intc = std::integral_constant<int, N_FRAC>;
    using IN = AdditiveInputs<intc>;
    IN in(c.g0, c.g1, c.g2);
    Matrix<var, Dynamic, 1> beta(c.beta);
    auto B = [&](int slot) { return beta.coeff(slot - 1); };
    var lp = stan::math::bernoulli_logit_lpmf_gathered_additive<false>(
        c.y, B(1), stan::math::slope_term{B(2), c.xd2},
        stan::math::slope_term{B(3), c.xd3},
        stan::math::slope2_term{B(5), c.xd31, c.xd32},
        stan::math::gather_term{"g0", in.g0(), c.i0},
        stan::math::gather_term{"g1", in.g1(), c.i1},
        stan::math::gather_term{"g2", in.g2(), c.i2});
    lp1 = lp.val();
    lp.grad();
    a10 = in.adj(0);
    a11 = in.adj(1);
    a12 = in.adj(2);
    b1 = adj_of(beta);
  }
  stan::math::recover_memory();
  EXPECT_TRUE(bits_equal(lp0, lp1)) << "lp " << lp0 << " vs " << lp1;
  for (Eigen::Index j = 0; j < c.g0.size() + c.g1.size() + c.g2.size(); ++j) {
    EXPECT_TRUE(bits_equal(j < c.g0.size() ? a00(j) : (j < c.g0.size() + c.g1.size() ? a01(j - c.g0.size()) : a02(j - c.g0.size() - c.g1.size())),
                           j < c.g0.size() ? a10(j) : (j < c.g0.size() + c.g1.size() ? a11(j - c.g0.size()) : a12(j - c.g0.size() - c.g1.size()))))
        << "gathered adj " << j;
  }
  for (int j = 0; j < 5; ++j) {
    EXPECT_TRUE(bits_equal(b0(j), b1(j))) << "beta adj " << j;
  }
}

}  // namespace

TEST(RevProbBernoulliLogitGathered, AdditiveBitIdenticalToComposedStock) {
  std::mt19937 rng(20260829);
  for (int rep = 0; rep < 6; ++rep) {
    const int N = 1 + static_cast<int>(rng() % 1500);
    const double scale = (rep % 3 == 2) ? 12.0 : 1.0;  // |ntheta| > 20 branch
    AdditiveCase c = make_additive_case(rng, N, scale);
    expect_additive_bit_identical<0>(c, 0);
    expect_additive_bit_identical<1>(c, 1);
    expect_additive_bit_identical<2>(c, 2);
  }
}

TEST(RevProbBernoulliLogitGathered, AdditiveTpWritebackPriorsFirst) {
  // the sweep-order certification: priors BEFORE the likelihood, writeback
  // primitive + retained tp chain vs the composed stock chain
  std::mt19937 rng(20260830);
  for (int rep = 0; rep < 4; ++rep) {
    const int N = 1 + static_cast<int>(rng() % 800);
    AdditiveCase c = make_additive_case(rng, N, 1.0);
    double lp0 = 0.0, lp1 = 0.0;
    Eigen::Matrix<double, Dynamic, 1> a00, a01, a02, b0, a10, a11, a12, b1;
    {
      Matrix<var, Dynamic, 1> beta(c.beta);
      Matrix<var, Dynamic, 1> g0(c.g0), g1(c.g1), g2(c.g2);
      var s1(1.3), s2(0.8), s3(2.2);
      // priors first (their edge callbacks stack BELOW the likelihood's)
      var lp = stan::math::normal_lpdf<false>(g0, 0.0, s1);
      lp = lp + stan::math::normal_lpdf<false>(g1, 0.0, s2);
      lp = lp + stan::math::normal_lpdf<false>(g2, 0.0, s3);
      Matrix<var, Dynamic, 1> y_hat(c.y.size());
      for (Eigen::Index k = 0; k < y_hat.size(); ++k) {
        y_hat.coeffRef(k) = add_leaf_chain(
            beta.coeff(0), beta.coeff(1), c.xd2, beta.coeff(2), c.xd3,
            beta.coeff(4), c.xd31, c.xd32, g0, c.i0, g1, c.i1, g2, c.i2, k);
      }
      lp = lp + stan::math::bernoulli_logit_lpmf<false>(c.y, y_hat);
      lp0 = lp.val();
      lp.grad();
      a00 = adj_of(g0);
      a01 = adj_of(g1);
      a02 = adj_of(g2);
      b0 = adj_of(beta);
    }
    stan::math::recover_memory();
    {
      Matrix<var, Dynamic, 1> beta(c.beta);
      Matrix<var, Dynamic, 1> g0(c.g0), g1(c.g1), g2(c.g2);
      var s1(1.3), s2(0.8), s3(2.2);
      var lp = stan::math::normal_lpdf<false>(g0, 0.0, s1);
      lp = lp + stan::math::normal_lpdf<false>(g1, 0.0, s2);
      lp = lp + stan::math::normal_lpdf<false>(g2, 0.0, s3);
      // the RETAINED stock tp chain
      Matrix<var, Dynamic, 1> y_hat(c.y.size());
      for (Eigen::Index k = 0; k < y_hat.size(); ++k) {
        y_hat.coeffRef(k) = add_leaf_chain(
            beta.coeff(0), beta.coeff(1), c.xd2, beta.coeff(2), c.xd3,
            beta.coeff(4), c.xd31, c.xd32, g0, c.i0, g1, c.i1, g2, c.i2, k);
      }
      auto B = [&](int slot) { return beta.coeff(slot - 1); };
      lp = lp + stan::math::bernoulli_logit_lpmf_gathered_additive_tp<false>(
                    c.y, y_hat, B(1), stan::math::slope_term{B(2), c.xd2},
                    stan::math::slope_term{B(3), c.xd3},
                    stan::math::slope2_term{B(5), c.xd31, c.xd32},
                    stan::math::gather_term{"g0", g0, c.i0},
                    stan::math::gather_term{"g1", g1, c.i1},
                    stan::math::gather_term{"g2", g2, c.i2});
      lp1 = lp.val();
      lp.grad();
      a10 = adj_of(g0);
      a11 = adj_of(g1);
      a12 = adj_of(g2);
      b1 = adj_of(beta);
    }
    stan::math::recover_memory();
    EXPECT_TRUE(bits_equal(lp0, lp1)) << "tp lp " << lp0 << " vs " << lp1;
    for (int j = 0; j < c.g0.size(); ++j)
      EXPECT_TRUE(bits_equal(a00(j), a10(j))) << "tp g0 adj " << j;
    for (int j = 0; j < c.g1.size(); ++j)
      EXPECT_TRUE(bits_equal(a01(j), a11(j))) << "tp g1 adj " << j;
    for (int j = 0; j < c.g2.size(); ++j)
      EXPECT_TRUE(bits_equal(a02(j), a12(j))) << "tp g2 adj " << j;
    for (int j = 0; j < 5; ++j)
      EXPECT_TRUE(bits_equal(b0(j), b1(j))) << "tp beta adj " << j;
  }
}

TEST(RevProbBernoulliLogitGathered, AdditiveScalarValueMatchesReference) {
  // hand-computed 3-point case for the additive predictor semantics
  Matrix<var, Dynamic, 1> g0(2), beta(2);
  g0 << -0.2, 0.4;
  beta << 0.1, -0.6;
  std::vector<int> i0{1, 2, 1}, y{1, 0, 1};
  std::vector<double> xd{0.5, 2.0, -1.5};
  var lp = stan::math::bernoulli_logit_lpmf_gathered_additive<true>(
      y, beta.coeff(0), stan::math::slope_term{beta.coeff(1), xd},
      stan::math::gather_term{"g0", g0, i0});
  std::vector<double> eta(3);
  for (int k = 0; k < 3; ++k) {
    eta[k] = 0.1 + (-0.6 * xd[k]) + g0.coeff(i0[k] - 1).val();
  }
  double expect = 0.0;
  for (int k = 0; k < 3; ++k) {
    expect += y[k] ? -std::log1p(std::exp(-eta[k]))
                   : -std::log1p(std::exp(eta[k]));
  }
  EXPECT_NEAR(lp.val(), expect, 1e-12);
  lp.grad();
  // gradients: intercept = sum d; slope = sum d*xd; g0[j] = sum over hits
  double di = 0, ds = 0;
  std::vector<double> dg(2, 0.0);
  for (int k = 0; k < 3; ++k) {
    double d = y[k] ? 1.0 - 1.0 / (1.0 + std::exp(-eta[k]))
                    : -1.0 / (1.0 + std::exp(-eta[k]));
    di += d;
    ds += d * xd[k];
    dg[i0[k] - 1] += d;
  }
  EXPECT_NEAR(beta.coeff(0).adj(), di, 1e-12);
  EXPECT_NEAR(beta.coeff(1).adj(), ds, 1e-12);
  EXPECT_NEAR(g0.coeff(0).adj(), dg[0], 1e-12);
  EXPECT_NEAR(g0.coeff(1).adj(), dg[1], 1e-12);
  stan::math::recover_memory();
}

TEST(RevProbBernoulliLogitGathered, AdditiveThrowSet) {
  Matrix<var, Dynamic, 1> g0(4), beta(2);
  g0.setConstant(var(0.3));
  beta << 0.1, -0.6;
  std::vector<int> i0(6, 2), y(6, 1);
  std::vector<double> xd(6, 0.5);
  // out-of-range gathered index: rvalue's text
  {
    std::vector<int> bad = i0;
    bad[3] = 99;
    try {
      var lp = stan::math::bernoulli_logit_lpmf_gathered_additive<true>(
          y, beta.coeff(0), stan::math::slope_term{beta.coeff(1), xd},
          stan::math::gather_term{"g0", g0, bad});
      ADD_FAILURE() << "expected out_of_range";
    } catch (const std::out_of_range& e) {
      EXPECT_NE(std::string(e.what()).find("vector[uni] indexing"),
                std::string::npos);
    }
    stan::math::recover_memory();
  }
  // NaN coefficient -> stock's check_not_nan message
  {
    Matrix<var, Dynamic, 1> gn(4);
    std::vector<double> gnv(4, 0.3);
    gnv[1] = std::numeric_limits<double>::quiet_NaN();
    for (int j = 0; j < 4; ++j) gn.coeffRef(j) = var(gnv[j]);
    try {
      var lp = stan::math::bernoulli_logit_lpmf_gathered_additive<true>(
          y, beta.coeff(0), stan::math::gather_term{"g0", gn, i0});
      ADD_FAILURE() << "expected domain_error";
    } catch (const std::domain_error& e) {
      EXPECT_EQ(std::string(e.what()),
                "bernoulli_logit_lpmf: Logit transformed probability"
                " parameter[1] is nan, but must be not nan!");
    }
    stan::math::recover_memory();
  }
  // y out of bounds -> stock's check_bounded message
  {
    std::vector<int> yb = y;
    yb[2] = 2;
    try {
      var lp = stan::math::bernoulli_logit_lpmf_gathered_additive<true>(
          yb, beta.coeff(0), stan::math::gather_term{"g0", g0, i0});
      ADD_FAILURE() << "expected domain_error";
    } catch (const std::domain_error& e) {
      EXPECT_NE(std::string(e.what()).find("must be in the interval [0, 1]"),
                std::string::npos);
    }
    stan::math::recover_memory();
  }
}

// ---------------------------------------------------------------------------
// W-130: the TP-block custom vari (gathered_additive_tp) — one custom vari
// per element created at the transformed-parameters loop, the likelihood
// line stock. Bitwise vs the FULLY-STOCK composed path (tp loop + stock
// bernoulli_logit_lpmf), with the slope coefficient vector beta layout-varied
// too (in the built election88 .so every parameter vector is SoA — the
// deserializer's to_var_value — whose stock discipline is the per-element
// rvalue view + callback one; .coeff on the layout-active operand IS that
// path), plus the causal-triangle prior-position controls and the throw set.
// ---------------------------------------------------------------------------
namespace {

template <typename BetaT, typename G0, typename G1, typename G2>
var tp_vari_stock_chain(const BetaT& beta, const std::vector<double>& xd2,
                        const std::vector<double>& xd3,
                        const std::vector<double>& xd31,
                        const std::vector<double>& xd32, const G0& g0,
                        const std::vector<int>& i0, const G1& g1,
                        const std::vector<int>& i1, const G2& g2,
                        const std::vector<int>& i2, Eigen::Index k) {
  return ((((((beta.coeff(0) + (beta.coeff(1) * xd2[k])) +
              (beta.coeff(2) * xd3[k])) +
             ((beta.coeff(4) * xd31[k]) * xd32[k])) +
            g0.coeff(i0[k] - 1)) +
           g1.coeff(i1[k] - 1)) +
          g2.coeff(i2[k] - 1));
}

// beta in one of the three deserializer layouts (AoS / SoA / Map-over-AoS)
template <int layout>
struct TpBeta {
  Matrix<var, Dynamic, 1> aos;
  stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>> soa{
      Eigen::Matrix<double, Dynamic, 1>(0)};

  explicit TpBeta(const Eigen::Matrix<double, Dynamic, 1>& v) : aos(v) {
    if constexpr (layout == 1)
      soa = stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>>(v);
  }
  auto b() const {
    if constexpr (layout == 1)
      return (const stan::math::var_value<Eigen::Matrix<double, Dynamic, 1>>&)
          soa;
    else if constexpr (layout == 2)
      return Eigen::Map<const Matrix<var, Dynamic, 1>>(aos.data(),
                                                       aos.size());
    else
      return (const Matrix<var, Dynamic, 1>&)aos;
  }
  Eigen::Matrix<double, Dynamic, 1> adj() const {
    Eigen::Matrix<double, Dynamic, 1> g(aos.size());
    if constexpr (layout == 1) {
      for (Eigen::Index j = 0; j < g.size(); ++j)
        g(j) = soa.vi_->adj_.coeff(j);
    } else {
      for (Eigen::Index j = 0; j < g.size(); ++j) g(j) = aos.coeff(j).adj();
    }
    return g;
  }
};

struct TpVariResult {
  double lp;
  Eigen::Matrix<double, Dynamic, 1> b_adj, g0_adj, g1_adj, g2_adj;
  Eigen::Matrix<double, Dynamic, 1> yhat;
};

// prior_pos: 0 none, 1 BEFORE the likelihood, 2 AFTER it.
template <int beta_layout, int g_layout, int prior_pos>
TpVariResult tp_vari_composed(const AdditiveCase& c) {
  TpVariResult r;
  using intc = std::integral_constant<int, g_layout>;
  using IN = AdditiveInputs<intc>;
  IN in(c.g0, c.g1, c.g2);
  TpBeta<beta_layout> beta(c.beta);
  Matrix<var, Dynamic, 1> y_hat(c.y.size());
  for (Eigen::Index k = 0; k < y_hat.size(); ++k) {
    y_hat.coeffRef(k) = tp_vari_stock_chain(
        beta.b(), c.xd2, c.xd3, c.xd31, c.xd32, in.g0(), c.i0, in.g1(), c.i1,
        in.g2(), c.i2, k);
  }
  r.yhat.resize(y_hat.size());
  for (Eigen::Index k = 0; k < y_hat.size(); ++k) r.yhat(k) = y_hat(k).val();
  var lp = stan::math::bernoulli_logit_lpmf<false>(c.y, y_hat);
  var s1(1.3), s2(0.8), s3(2.2), s4(9.1);
  // NOTE: this math's normal_lpdf does not accept var_value vectors, so the
  // prior-position tests restrict the operands to the AoS/Map layouts (the
  // SoA+prior combinations are certified by the gate-a harness, which runs
  // on the bundle's newer math).
  if constexpr (prior_pos == 1) {
    lp = lp + stan::math::normal_lpdf<false>(in.g0(), 0.0, s1);
    lp = lp + stan::math::normal_lpdf<false>(in.g1(), 0.0, s2);
    lp = lp + stan::math::normal_lpdf<false>(in.g2(), 0.0, s3);
    lp = lp + stan::math::normal_lpdf<false>(beta.b(), 0.0, s4);
  }
  r.lp = lp.val();
  if constexpr (prior_pos == 2) {
    lp = lp + stan::math::normal_lpdf<false>(in.g0(), 0.0, s1);
    lp = lp + stan::math::normal_lpdf<false>(in.g1(), 0.0, s2);
    lp = lp + stan::math::normal_lpdf<false>(in.g2(), 0.0, s3);
    lp = lp + stan::math::normal_lpdf<false>(beta.b(), 0.0, s4);
  }
  lp.grad();
  r.b_adj = beta.adj();
  r.g0_adj = in.adj(0);
  r.g1_adj = in.adj(1);
  r.g2_adj = in.adj(2);
  stan::math::recover_memory();
  return r;
}

template <int beta_layout, int g_layout, int prior_pos>
TpVariResult tp_vari_factory(const AdditiveCase& c) {
  TpVariResult r;
  using intc = std::integral_constant<int, g_layout>;
  using IN = AdditiveInputs<intc>;
  IN in(c.g0, c.g1, c.g2);
  TpBeta<beta_layout> beta(c.beta);
  Matrix<var, Dynamic, 1> y_hat = stan::math::gathered_additive_tp(
      static_cast<Eigen::Index>(c.y.size()),
      stan::math::slot_term{"beta", beta.b(), 1},
      stan::math::slot_slope_term{"beta", beta.b(), 2, c.xd2},
      stan::math::slot_slope_term{"beta", beta.b(), 3, c.xd3},
      stan::math::slot_slope2_term{"beta", beta.b(), 5, c.xd31, c.xd32},
      stan::math::gather_term{"g0", in.g0(), c.i0},
      stan::math::gather_term{"g1", in.g1(), c.i1},
      stan::math::gather_term{"g2", in.g2(), c.i2});
  r.yhat.resize(y_hat.size());
  for (Eigen::Index k = 0; k < y_hat.size(); ++k) r.yhat(k) = y_hat(k).val();
  var lp = stan::math::bernoulli_logit_lpmf<false>(c.y, y_hat);
  var s1(1.3), s2(0.8), s3(2.2), s4(9.1);
  // NOTE: this math's normal_lpdf does not accept var_value vectors, so the
  // prior-position tests restrict the operands to the AoS/Map layouts (the
  // SoA+prior combinations are certified by the gate-a harness, which runs
  // on the bundle's newer math).
  if constexpr (prior_pos == 1) {
    lp = lp + stan::math::normal_lpdf<false>(in.g0(), 0.0, s1);
    lp = lp + stan::math::normal_lpdf<false>(in.g1(), 0.0, s2);
    lp = lp + stan::math::normal_lpdf<false>(in.g2(), 0.0, s3);
    lp = lp + stan::math::normal_lpdf<false>(beta.b(), 0.0, s4);
  }
  r.lp = lp.val();
  if constexpr (prior_pos == 2) {
    lp = lp + stan::math::normal_lpdf<false>(in.g0(), 0.0, s1);
    lp = lp + stan::math::normal_lpdf<false>(in.g1(), 0.0, s2);
    lp = lp + stan::math::normal_lpdf<false>(in.g2(), 0.0, s3);
    lp = lp + stan::math::normal_lpdf<false>(beta.b(), 0.0, s4);
  }
  lp.grad();
  r.b_adj = beta.adj();
  r.g0_adj = in.adj(0);
  r.g1_adj = in.adj(1);
  r.g2_adj = in.adj(2);
  stan::math::recover_memory();
  return r;
}

void expect_tp_vari_equal(const TpVariResult& s, const TpVariResult& p) {
  EXPECT_TRUE(bits_equal(s.lp, p.lp)) << "lp " << s.lp << " vs " << p.lp;
  for (Eigen::Index k = 0; k < s.yhat.size(); ++k)
    EXPECT_TRUE(bits_equal(s.yhat(k), p.yhat(k))) << "y_hat " << k << " "
                                                  << s.yhat(k) << " vs "
                                                  << p.yhat(k);
  for (int j = 0; j < s.b_adj.size(); ++j)
    EXPECT_TRUE(bits_equal(s.b_adj(j), p.b_adj(j))) << "beta adj " << j;
  for (int j = 0; j < s.g0_adj.size(); ++j)
    EXPECT_TRUE(bits_equal(s.g0_adj(j), p.g0_adj(j))) << "g0 adj " << j;
  for (int j = 0; j < s.g1_adj.size(); ++j)
    EXPECT_TRUE(bits_equal(s.g1_adj(j), p.g1_adj(j))) << "g1 adj " << j;
  for (int j = 0; j < s.g2_adj.size(); ++j)
    EXPECT_TRUE(bits_equal(s.g2_adj(j), p.g2_adj(j))) << "g2 adj " << j;
}

}  // namespace

TEST(RevProbBernoulliLogitGathered, TpVariBitIdenticalToComposedStock) {
  std::mt19937 rng(20260831);
  for (int rep = 0; rep < 4; ++rep) {
    const int N = 1 + static_cast<int>(rng() % 900);
    const double scale = (rep % 3 == 2) ? 14.0 : 1.0;
    AdditiveCase c = make_additive_case(rng, N, scale);
    // every beta x gather layout combination (9), no priors
    expect_tp_vari_equal(tp_vari_composed<0, 0, 0>(c),
                         tp_vari_factory<0, 0, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<0, 1, 0>(c),
                         tp_vari_factory<0, 1, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<0, 2, 0>(c),
                         tp_vari_factory<0, 2, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<1, 0, 0>(c),
                         tp_vari_factory<1, 0, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<1, 1, 0>(c),
                         tp_vari_factory<1, 1, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<1, 2, 0>(c),
                         tp_vari_factory<1, 2, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<2, 0, 0>(c),
                         tp_vari_factory<2, 0, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<2, 1, 0>(c),
                         tp_vari_factory<2, 1, 0>(c));
    expect_tp_vari_equal(tp_vari_composed<2, 2, 0>(c),
                         tp_vari_factory<2, 2, 0>(c));
  }
}

TEST(RevProbBernoulliLogitGathered, TpVariPriorsBothSides) {
  // the causal triangle: the custom varis are created at the tp loop, so
  // delivery is at stock's position regardless of where the likelihood sits
  // among the model statements — priors BEFORE (the election88 layout, the
  // W-129 failure case) and AFTER must BOTH be bitwise exact.
  std::mt19937 rng(20260901);
  for (int rep = 0; rep < 3; ++rep) {
    const int N = 1 + static_cast<int>(rng() % 600);
    AdditiveCase c = make_additive_case(rng, N, 1.0);
    expect_tp_vari_equal(tp_vari_composed<0, 0, 1>(c),
                         tp_vari_factory<0, 0, 1>(c));
    expect_tp_vari_equal(tp_vari_composed<2, 2, 1>(c),
                         tp_vari_factory<2, 2, 1>(c));
    expect_tp_vari_equal(tp_vari_composed<0, 2, 1>(c),
                         tp_vari_factory<0, 2, 1>(c));
    expect_tp_vari_equal(tp_vari_composed<2, 0, 1>(c),
                         tp_vari_factory<2, 0, 1>(c));
    expect_tp_vari_equal(tp_vari_composed<0, 0, 2>(c),
                         tp_vari_factory<0, 0, 2>(c));
    expect_tp_vari_equal(tp_vari_composed<2, 2, 2>(c),
                         tp_vari_factory<2, 2, 2>(c));
    expect_tp_vari_equal(tp_vari_composed<0, 2, 2>(c),
                         tp_vari_factory<0, 2, 2>(c));
    expect_tp_vari_equal(tp_vari_composed<2, 0, 2>(c),
                         tp_vari_factory<2, 0, 2>(c));
  }
}

TEST(RevProbBernoulliLogitGathered, TpVariValueMatchesReference) {
  // hand-computed 3-point case: eta = b1 + b2*xd2 + b3*xd3 + (b5*xd31)*xd32
  // + g0[i0] + g1[i1] + g2[i2]
  std::vector<int> y{1, 0, 1};
  std::vector<int> i0{1, 2, 3}, i1{4, 5, 1}, i2{9, 3, 7};
  std::vector<double> xd2{0.5, -1.25, 2.0};
  std::vector<double> xd3{1.0, 0.0, 1.0};   // the 1.0 alias
  std::vector<double> xd31{1.0, 1.0, 0.0};  // the m1 alias
  std::vector<double> xd32{0.25, 3.5, -2.0};
  Matrix<var, Dynamic, 1> beta(5);
  beta << var(-0.3), var(0.7), var(-1.1), var(99.0), var(0.4);
  Matrix<var, Dynamic, 1> g0(3), g1(5), g2(9);
  for (int j = 0; j < 3; ++j) g0.coeffRef(j) = var(0.1 * (j + 1));
  for (int j = 0; j < 5; ++j) g1.coeffRef(j) = var(-0.2 * (j + 1));
  for (int j = 0; j < 9; ++j) g2.coeffRef(j) = var(0.05 * (j + 1));
  Matrix<var, Dynamic, 1> y_hat = stan::math::gathered_additive_tp(
      3, stan::math::slot_term{"beta", beta, 1},
      stan::math::slot_slope_term{"beta", beta, 2, xd2},
      stan::math::slot_slope_term{"beta", beta, 3, xd3},
      stan::math::slot_slope2_term{"beta", beta, 5, xd31, xd32},
      stan::math::gather_term{"g0", g0, i0}, stan::math::gather_term{"g1", g1, i1},
      stan::math::gather_term{"g2", g2, i2});
  EXPECT_FLOAT_EQ(-0.3 + 0.7 * 0.5 + -1.1 * 1.0 + 0.4 * 1.0 * 0.25 + 0.1
                      + -0.8 + 0.45,
                  y_hat(0).val());
  EXPECT_FLOAT_EQ(-0.3 + 0.7 * -1.25 + -1.1 * 0.0 + 0.4 * 1.0 * 3.5 + 0.2
                      + -1.0 + 0.15,
                  y_hat(1).val());
  EXPECT_FLOAT_EQ(-0.3 + 0.7 * 2.0 + -1.1 * 1.0 + 0.4 * 0.0 * -2.0 + 0.3
                      + -0.2 + 0.35,
                  y_hat(2).val());
  var lp = stan::math::bernoulli_logit_lpmf<false>(y, y_hat);
  // the stock DOUBLE lpmf over the hand-computed eta values
  Eigen::Matrix<double, Dynamic, 1> eta(3);
  eta << -0.3 + 0.7 * 0.5 + -1.1 * 1.0 + 0.4 * 1.0 * 0.25 + 0.1 + -0.8 + 0.45,
      -0.3 + 0.7 * -1.25 + -1.1 * 0.0 + 0.4 * 1.0 * 3.5 + 0.2 + -1.0 + 0.15,
      -0.3 + 0.7 * 2.0 + -1.1 * 1.0 + 0.4 * 0.0 * -2.0 + 0.3 + -0.2 + 0.35;
  EXPECT_FLOAT_EQ(stan::math::bernoulli_logit_lpmf<false>(y, eta), lp.val());
  lp.grad();
  stan::math::recover_memory();
}

TEST(RevProbBernoulliLogitGathered, TpVariThrowSet) {
  std::vector<int> y{1, 1, 1};
  std::vector<int> i0{1, 2, 1}, i1{1, 1, 1}, i2{1, 1, 1};
  std::vector<double> xd2(3, 0.5), xd3(3, 1.0), xd31(3, 1.0), xd32(3, 0.5);
  Matrix<var, Dynamic, 1> beta(5);
  for (int j = 0; j < 5; ++j) beta.coeffRef(j) = var(0.2 * j);
  Matrix<var, Dynamic, 1> g0(2), g1(1), g2(1);
  g0 << var(0.3), var(-0.3);
  g1 << var(0.4);
  g2 << var(0.5);
  // out-of-range gathered index -> stock's rvalue text at the tp position
  {
    std::vector<int> ibad{1, 3, 1};
    try {
      auto yh = stan::math::gathered_additive_tp(
          3, stan::math::slot_term{"beta", beta, 1},
          stan::math::slot_slope_term{"beta", beta, 2, xd2},
          stan::math::gather_term{"g0", g0, ibad});
      (void)yh;
      ADD_FAILURE() << "expected out_of_range";
    } catch (const std::out_of_range& e) {
      EXPECT_NE(std::string(e.what()).find("vector[uni] indexing"),
                std::string::npos);
    }
    stan::math::recover_memory();
  }
  // NaN coefficient -> the STOCK likelihood's indexed check_not_nan text
  {
    Matrix<var, Dynamic, 1> gn(2);
    gn << var(0.3), var(std::numeric_limits<double>::quiet_NaN());
    Matrix<var, Dynamic, 1> yh = stan::math::gathered_additive_tp(
        3, stan::math::slot_term{"beta", beta, 1},
        stan::math::gather_term{"g0", gn, i0});
    try {
      var lp = stan::math::bernoulli_logit_lpmf<true>(y, yh);
      (void)lp;
      ADD_FAILURE() << "expected domain_error";
    } catch (const std::domain_error& e) {
      EXPECT_EQ(std::string(e.what()),
                "bernoulli_logit_lpmf: Logit transformed probability"
                " parameter[2] is nan, but must be not nan!");
    }
    stan::math::recover_memory();
  }
  // out-of-range slot (defensive API guard, stock's rvalue text)
  {
    try {
      auto yh = stan::math::gathered_additive_tp(
          3, stan::math::slot_term{"beta", beta, 9},
          stan::math::gather_term{"g0", g0, i0});
      (void)yh;
      ADD_FAILURE() << "expected out_of_range";
    } catch (const std::out_of_range& e) {
      EXPECT_NE(std::string(e.what()).find("vector[uni] indexing"),
                std::string::npos);
    }
    stan::math::recover_memory();
  }
}
