#include <stan/math/rev/prob/normal_lpdf_gathered.hpp>
// the composed-stock reference path
#include <stan/math.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

// The gathered normal likelihood must be BIT-IDENTICAL to the loop the Stan
// compiler emits for
//   for (n in 1:N) { mu[n] = alpha[ii[n]] (+ x[n] * beta[ii2[n]]);
//                    target += normal_lpdf(y[n] | mu[n], sigma); }
// including the model's accumulation (accumulator<var> pushes, one per
// term). Values and every gradient component are compared with memcmp, not
// a tolerance.
namespace {

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::math::var;
using VectorXd = Eigen::Matrix<double, Dynamic, 1>;
using soa_vec = stan::math::var_value<VectorXd>;

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

// layout: 0 = Matrix<var> (AoS); 1 = var_value<VectorXd> (SoA)
template <int layout>
struct Coeffs {
  Matrix<var, Dynamic, 1> aos;
  soa_vec soa{VectorXd(0)};
  explicit Coeffs(const VectorXd& v) : aos(v) {
    if constexpr (layout == 1) {
      soa = soa_vec(v);
    }
  }
  VectorXd adj() const {
    VectorXd g(aos.size());
    if constexpr (layout == 1) {
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

template <int layout>
const auto& get(const Coeffs<layout>& c) {
  if constexpr (layout == 1) {
    return c.soa;
  } else {
    return c.aos;
  }
}

// The stock loop for shape A (mu[n] = alpha[ii[n]]), with the model's
// accumulator semantics (one add per element).
var stock_loop(const VectorXd& y, const Matrix<var, Dynamic, 1>& alpha,
               const std::vector<int>& ii, const var& sigma) {
  const int N = (int)y.size();
  stan::math::accumulator<var> lp_accum;
  Matrix<var, Dynamic, 1> mu = Matrix<var, Dynamic, 1>::Constant(
      N, var(std::numeric_limits<double>::quiet_NaN()));
  for (int n = 1; n <= N; ++n) {
    mu.coeffRef(n - 1) = alpha.coeff(ii[n - 1] - 1);  // rvalue + assign
    lp_accum.add(stan::math::normal_lpdf<false>(
        y.coeff(n - 1), mu.coeff(n - 1), sigma));
  }
  return lp_accum.sum();
}

var stock_loop(const VectorXd& y, const Matrix<var, Dynamic, 1>& alpha,
               const std::vector<int>& ii, double sigma) {
  const int N = (int)y.size();
  stan::math::accumulator<var> lp_accum;
  Matrix<var, Dynamic, 1> mu = Matrix<var, Dynamic, 1>::Constant(
      N, var(std::numeric_limits<double>::quiet_NaN()));
  for (int n = 1; n <= N; ++n) {
    mu.coeffRef(n - 1) = alpha.coeff(ii[n - 1] - 1);
    lp_accum.add(stan::math::normal_lpdf<false>(y.coeff(n - 1),
                                                mu.coeff(n - 1), sigma));
  }
  return lp_accum.sum();
}

// The stock loop for shape B (mu[n] = alpha[ii[n]] + x[n] * beta[ii2[n]]),
// with the generated op order: multiply first, then add.
var stock_loop(const VectorXd& y, const Matrix<var, Dynamic, 1>& alpha,
               const std::vector<int>& ii, const VectorXd& x,
               const Matrix<var, Dynamic, 1>& beta,
               const std::vector<int>& ii2, const var& sigma) {
  const int N = (int)y.size();
  stan::math::accumulator<var> lp_accum;
  Matrix<var, Dynamic, 1> mu = Matrix<var, Dynamic, 1>::Constant(
      N, var(std::numeric_limits<double>::quiet_NaN()));
  for (int n = 1; n <= N; ++n) {
    mu.coeffRef(n - 1)
        = alpha.coeff(ii[n - 1] - 1) + x.coeff(n - 1) * beta.coeff(ii2[n - 1] - 1);
    lp_accum.add(stan::math::normal_lpdf<false>(
        y.coeff(n - 1), mu.coeff(n - 1), sigma));
  }
  return lp_accum.sum();
}

// shape A case runner
template <int layout>
void run_case_A(const VectorXd& y, const VectorXd& alpha_d,
                const std::vector<int>& ii, double sigma_d,
                bool sigma_is_var) {
  const Eigen::Index J = alpha_d.size();
  double lp0 = 0.0, lp1 = 0.0, s0 = 0.0, s1 = 0.0;
  VectorXd g0, g1;
  {
    Coeffs<0> alpha(alpha_d);
    var sigma(sigma_d);
    var lp = sigma_is_var ? stock_loop(y, alpha.aos, ii, sigma)
                          : stock_loop(y, alpha.aos, ii, sigma_d);
    lp0 = lp.val();
    lp.grad();
    g0 = alpha.adj();
    s0 = sigma_is_var ? sigma.adj() : 0.0;
  }
  stan::math::recover_memory();
  {
    Coeffs<layout> alpha(alpha_d);
    var sigma(sigma_d);
    stan::math::accumulator<var> lp_accum;
    std::vector<var> terms;
    if (sigma_is_var) {
      terms = stan::math::normal_lpdf_gathered<false>(y, get(alpha), ii,
                                                      sigma);
    } else {
      terms = stan::math::normal_lpdf_gathered<false>(y, get(alpha), ii,
                                                      sigma_d);
    }
    for (const auto& t : terms) {
      lp_accum.add(t);
    }
    var lp = lp_accum.sum();
    lp1 = lp.val();
    lp.grad();
    g1 = alpha.adj();
    s1 = sigma_is_var ? sigma.adj() : 0.0;
  }
  stan::math::recover_memory();
  EXPECT_TRUE(bits_equal(lp0, lp1)) << "lp " << lp0 << " vs " << lp1;
  for (Eigen::Index j = 0; j < J; ++j) {
    EXPECT_TRUE(bits_equal(g0(j), g1(j)))
        << "d/dalpha(" << j << "): " << g0(j) << " vs " << g1(j);
  }
  if (sigma_is_var) {
    EXPECT_TRUE(bits_equal(s0, s1)) << "d/dsigma: " << s0 << " vs " << s1;
  }
}

// shape B case runner
template <int layout>
void run_case_B(const VectorXd& y, const VectorXd& alpha_d,
                const std::vector<int>& ii, const VectorXd& x,
                const VectorXd& beta_d, const std::vector<int>& ii2,
                double sigma_d, bool sigma_is_var) {
  const Eigen::Index J = alpha_d.size();
  double lp0 = 0.0, lp1 = 0.0, s0 = 0.0, s1 = 0.0;
  VectorXd ga0, ga1, gb0, gb1;
  {
    Coeffs<0> alpha(alpha_d), beta(beta_d);
    var sigma(sigma_d);
    var lp = stock_loop(y, alpha.aos, ii, x, beta.aos, ii2, sigma);
    lp0 = lp.val();
    lp.grad();
    ga0 = alpha.adj();
    gb0 = beta.adj();
    s0 = sigma.adj();
  }
  stan::math::recover_memory();
  {
    Coeffs<layout> alpha(alpha_d), beta(beta_d);
    var sigma(sigma_d);
    stan::math::accumulator<var> lp_accum;
    auto terms = stan::math::normal_lpdf_gathered<false>(
        y, get(alpha), ii, x, get(beta), ii2, sigma);
    for (const auto& t : terms) {
      lp_accum.add(t);
    }
    var lp = lp_accum.sum();
    lp1 = lp.val();
    lp.grad();
    ga1 = alpha.adj();
    gb1 = beta.adj();
    s1 = sigma.adj();
  }
  stan::math::recover_memory();
  EXPECT_TRUE(bits_equal(lp0, lp1)) << "lp " << lp0 << " vs " << lp1;
  for (Eigen::Index j = 0; j < J; ++j) {
    EXPECT_TRUE(bits_equal(ga0(j), ga1(j)))
        << "B d/dalpha(" << j << "): " << ga0(j) << " vs " << ga1(j);
    EXPECT_TRUE(bits_equal(gb0(j), gb1(j)))
        << "B d/dbeta(" << j << "): " << gb0(j) << " vs " << gb1(j);
  }
  EXPECT_TRUE(bits_equal(s0, s1)) << "B d/dsigma: " << s0 << " vs " << s1;
}

}  // namespace

TEST(RevProbNormalLpdfGathered, BitIdenticalToComposedStock) {
  std::mt19937 rng(20260829);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int rep = 0; rep < 8; ++rep) {
    const int J = 1 + static_cast<int>(rng() % 120);
    const int N = 1 + static_cast<int>(rng() % 2000);
    const double sg[] = {0.5, 1.0, 1e-3, 1e3};
    const double sigma = sg[rep % 4];
    VectorXd y(N), a(J);
    for (int n = 0; n < N; ++n) {
      y(n) = nd(rng) * 2;
    }
    for (int j = 0; j < J; ++j) {
      a(j) = nd(rng);
    }
    std::vector<int> ii(N);
    for (int k = 0; k < N; ++k) {
      ii[k] = 1 + static_cast<int>(rng() % J);
    }
    run_case_A<0>(y, a, ii, sigma, /*sigma_is_var=*/rep % 2 == 0);
    run_case_A<1>(y, a, ii, sigma, /*sigma_is_var=*/rep % 2 == 0);
  }
}

TEST(RevProbNormalLpdfGathered, BitIdenticalToComposedStockSlope) {
  std::mt19937 rng(20260830);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int rep = 0; rep < 8; ++rep) {
    const int J = 1 + static_cast<int>(rng() % 100);
    const int N = 1 + static_cast<int>(rng() % 1500);
    const double sg[] = {0.5, 1.0, 1e-3, 1e3};
    const double sigma = sg[rep % 4];
    VectorXd y(N), a(J), b(J), x(N);
    for (int n = 0; n < N; ++n) {
      y(n) = nd(rng);
      const double r = nd(rng);
      x(n) = (n % 4 == 0) ? 0.0 : ((n % 3 == 0) ? -std::fabs(r) : r);
    }
    for (int j = 0; j < J; ++j) {
      a(j) = nd(rng);
      b(j) = nd(rng);
    }
    std::vector<int> ii(N), ii2(N);
    for (int k = 0; k < N; ++k) {
      ii[k] = 1 + static_cast<int>(rng() % J);
      ii2[k] = 1 + static_cast<int>(rng() % J);
    }
    const std::vector<int>& idx2 = (rep % 2 == 0) ? ii : ii2;
    run_case_B<0>(y, a, ii, x, b, idx2, sigma, true);
    run_case_B<1>(y, a, ii, x, b, idx2, sigma, false);
  }
}

TEST(RevProbNormalLpdfGathered, ScalarValueMatchesReference) {
  // hand-computed 2-point case (propto = false: constants included)
  VectorXd y(2), a(1);
  y << 1.7, -0.3;
  a << 0.8;
  std::vector<int> ii{1, 1};
  var sigma(0.7);
  auto terms = stan::math::normal_lpdf_gathered<false>(y, Matrix<var, Dynamic, 1>(a), ii, sigma);
  double inv_s = 1.0 / 0.7;
  double t1 = -0.5 * ((1.7 - 0.8) * inv_s) * ((1.7 - 0.8) * inv_s)
              - std::log(0.7) - 0.5 * std::log(2 * stan::math::pi());
  // compare at tight tolerance (the reference is a rearranged expression)
  EXPECT_NEAR(terms[0].val(), t1, 1e-12);
  stan::math::recover_memory();
}

TEST(RevProbNormalLpdfGathered, SizeZero) {
  VectorXd y(0), a(1);
  a << 1.0;
  auto terms = stan::math::normal_lpdf_gathered<false>(
      y, Matrix<var, Dynamic, 1>(a), std::vector<int>{}, var(1.0));
  EXPECT_EQ(terms.size(), 0);
  stan::math::recover_memory();
}
