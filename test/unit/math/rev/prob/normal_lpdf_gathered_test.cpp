#include <stan/math/rev/prob/normal_lpdf_gathered.hpp>
// the composed-stock reference path
#include <stan/math.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
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

// W-118: bounds-guarded stock loops for the strict-order throw cases.
// The simplified reference above indexes directly (fine for valid
// states); the out-of-range cases need the generated loop's rvalue
// semantics, reproduced here with the same check_range call the
// primitive's cold path makes (identical exception + message).
var stock_loop_guarded(const VectorXd& y, const Matrix<var, Dynamic, 1>& alpha,
                       const std::vector<int>& ii, const var& sigma) {
  const int N = (int)y.size();
  const int J = (int)alpha.size();
  stan::math::accumulator<var> lp_accum;
  for (int n = 1; n <= N; ++n) {
    stan::math::check_range("vector[uni] indexing", "alpha", J, ii[n - 1]);
    lp_accum.add(stan::math::normal_lpdf<false>(
        y.coeff(n - 1), alpha.coeff(ii[n - 1] - 1), sigma));
  }
  return lp_accum.sum();
}

var stock_loop_guarded(const VectorXd& y, const Matrix<var, Dynamic, 1>& alpha,
                       const std::vector<int>& ii, const VectorXd& x,
                       const Matrix<var, Dynamic, 1>& beta,
                       const std::vector<int>& ii2, const var& sigma) {
  const int N = (int)y.size();
  const int JA = (int)alpha.size();
  const int JB = (int)beta.size();
  stan::math::accumulator<var> lp_accum;
  for (int n = 1; n <= N; ++n) {
    stan::math::check_range("vector[uni] indexing", "alpha", JA, ii[n - 1]);
    stan::math::check_range("vector[uni] indexing", "beta", JB, ii2[n - 1]);
    lp_accum.add(stan::math::normal_lpdf<false>(
        y.coeff(n - 1),
        alpha.coeff(ii[n - 1] - 1) + x.coeff(n - 1) * beta.coeff(ii2[n - 1] - 1),
        sigma));
  }
  return lp_accum.sum();
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

// W-112.2: THROW-SET parity -- on invalid states the primitive must
// throw exactly what the composed stock loop throws (same type, same
// message, hence the same first-failing element in stock's per-element
// check order). The stock loop's check_finite(mu) converts non-finite-mu
// states into exceptions the sampler treats as (logp=-inf, grad=0); a
// silently-computed NaN/-inf lp with NaN gradients does not reproduce
// that observable behavior (the W-116b radon_var divergence mechanism).
TEST(RevProbNormalLpdfGathered, ThrowSetParity) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  struct Case {
    bool shapeB;
    int y_nan;
    int a_bad;    // alpha coefficient made +inf
    int b_bad;    // beta coefficient made NaN
    double sigma;
  };
  const Case cases[] = {
      {true, -1, 3, -1, 1.0},    // mu = +inf
      {true, -1, -1, 5, 1.0},    // mu = NaN
      {false, -1, 2, -1, 1.0},   // shape A mu = +inf
      {true, 4, 1, -1, 1.0},     // mu bad at element 1, y NaN at 4
      {true, -1, -1, -1, 0.0},   // sigma = 0
  };
  for (const auto& tc : cases) {
    const int J = 9, N = 24;
    VectorXd y(N), a(J), b(J), x(N);
    for (int n = 0; n < N; ++n) {
      y(n) = 0.3 * n;
      x(n) = 1.5;
    }
    for (int j = 0; j < J; ++j) {
      a(j) = 0.2 * j;
      b(j) = 0.1 * j;
    }
    std::vector<int> ii(N);
    for (int k = 0; k < N; ++k) {
      ii[k] = 1 + (k % J);
    }
    if (tc.y_nan >= 0) {
      y(tc.y_nan) = nan;
    }
    if (tc.a_bad >= 0) {
      a(tc.a_bad) = inf;
    }
    if (tc.b_bad >= 0) {
      b(tc.b_bad) = nan;
    }
    std::string msg0, msg1;
    {
      var sigma(tc.sigma);
      try {
        var lp = tc.shapeB ? stock_loop(y, Matrix<var, Dynamic, 1>(a), ii, x,
                                        Matrix<var, Dynamic, 1>(b), ii, sigma)
                           : stock_loop(y, Matrix<var, Dynamic, 1>(a), ii,
                                        sigma);
        (void)lp;
        msg0 = "<no-throw>";
      } catch (const std::domain_error& e) {
        msg0 = e.what();
      }
      stan::math::recover_memory();
    }
    {
      Coeffs<0> alpha(a), beta(b);
      var sigma(tc.sigma);
      stan::math::accumulator<var> lp_accum;
      try {
        std::vector<var> terms;
        if (tc.shapeB) {
          terms = stan::math::normal_lpdf_gathered<false>(
              y, alpha.aos, ii, x, beta.aos, ii, sigma);
        } else {
          terms = stan::math::normal_lpdf_gathered<false>(y, alpha.aos, ii,
                                                          sigma);
        }
        for (const auto& t : terms) {
          lp_accum.add(t);
        }
        var lp = lp_accum.sum();
        (void)lp;
        msg1 = "<no-throw>";
      } catch (const std::domain_error& e) {
        msg1 = e.what();
      }
      stan::math::recover_memory();
    }
    // The sigma<=0 message differs only in the function-name prefix
    // (normal_lpdf vs normal_lpdf_gathered): normalize before compare.
    const std::string p = "normal_lpdf: ", g = "normal_lpdf_gathered: ";
    if (msg0.rfind(p, 0) == 0) {
      msg0 = msg0.substr(p.size());
    }
    if (msg1.rfind(p, 0) == 0) {
      msg1 = msg1.substr(p.size());
    }
    if (msg1.rfind(g, 0) == 0) {
      msg1 = msg1.substr(g.size());
    }
    EXPECT_NE(msg0, "<no-throw>") << "stock must throw (case sigma=" << tc.sigma << ")";
    EXPECT_EQ(msg0, msg1) << "throw-set/message parity";
  }
}

// ================= W-118: fused-interior additions =================

// N spanning SIMD widths and remainders (AVX2: 4 double lanes): the
// vectorized term pass and its scalar epilogue must stay bit-identical
// to the composed stock loop at every width/remainder combination.
TEST(RevProbNormalLpdfGathered, FusionEdgeWidths) {
  std::mt19937 rng(20260829);
  std::normal_distribution<double> nd(0.0, 1.0);
  const long long Ns[] = {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17,
                          31, 32, 33, 100, 919, 12573};
  for (long long N : Ns) {
    const int J = (int)std::min<long long>(1 + N / 3 + 3, 400);
    VectorXd y(N), a(J), b(J), x(N);
    for (long long n = 0; n < N; ++n) {
      y(n) = nd(rng) * 1.5;
      x(n) = ((n % 7 == 0) ? 0.0 : ((n % 5 == 0) ? -1.25 : 1.0 + 0.5 * nd(rng)));
    }
    for (int j = 0; j < J; ++j) {
      a(j) = nd(rng);
      b(j) = nd(rng) * 0.5;
    }
    std::vector<int> ii(N), ii2(N), iip(N);
    for (long long k = 0; k < N; ++k) {
      ii[k] = 1 + (int)(rng() % J);
      ii2[k] = 1 + (int)(rng() % J);
      iip[k] = 1 + (int)(k % J);
    }
    run_case_A<0>(y, a, ii, 0.9, true);
    run_case_A<1>(y, a, iip, 0.5, false);
    run_case_B<0>(y, a, ii, x, b, ii, 0.7, true);
    run_case_B<1>(y, a, iip, x, b, ii2, 1e-3, false);
  }
  // single-coefficient degenerate: every lane gathers the same index
  const int N = 300;
  VectorXd y(N), a(1), b(1), x(N);
  for (int n = 0; n < N; ++n) {
    y(n) = nd(rng);
    x(n) = 1.0;
  }
  a(0) = 0.25;
  b(0) = -0.5;
  std::vector<int> ones(N, 1);
  run_case_A<0>(y, a, ones, 1.0, true);
  run_case_B<0>(y, a, ones, x, b, ones, 1.0, true);
}

// The strict per-element throw ORDER (alpha index, beta index, y, mu) on
// mixed-defect states: the cold path re-derives stock's order exactly,
// including defects at different elements and two defects at one element.
TEST(RevProbNormalLpdfGathered, StrictOrderThrowSet) {
  struct Case {
    bool shapeB;
    int y_nan_idx;
    int bad_a_idx_k;
    int a_oob;
    int bad_b_idx_k;
    int b_oob;
  };
  const Case cases[] = {
      {false, 0, 5, 0, -1, 0},   // y-NaN@0 vs bad index@5: y wins
      {false, 2, 0, 99, -1, 0},  // bad index@0 vs y-NaN@2: range wins
      {false, -1, 1, -1, -1, 0},  // negative index
      {false, -1, 3, 99, -1, 0},  // index one past the end
      {true, 2, 1, 99, -1, 0},   // B: alpha index bad at earlier element
      {true, -1, -1, 0, 0, -3},  // B: beta index bad at element 0
      {true, 0, 0, 0, -1, 0},    // same element: bounds before y
  };
  const int J = 12, N = 30;
  std::mt19937 rng(20260830);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (const auto& tc : cases) {
    VectorXd y(N), a(J), b(J), x(N);
    for (int n = 0; n < N; ++n) y(n) = nd(rng) * 0.5;
    for (int j = 0; j < J; ++j) {
      a(j) = nd(rng);
      b(j) = nd(rng) * 0.5;
    }
    for (int n = 0; n < N; ++n) x(n) = 2.0;
    std::vector<int> ii(N), ii2(N);
    for (int k = 0; k < N; ++k) {
      ii[k] = 1 + (k % J);
      ii2[k] = 1 + ((k * 5 + 3) % J);
    }
    if (tc.y_nan_idx >= 0)
      y(tc.y_nan_idx) = std::numeric_limits<double>::quiet_NaN();
    if (tc.bad_a_idx_k >= 0) ii[tc.bad_a_idx_k] = tc.a_oob;
    if (tc.bad_b_idx_k >= 0) ii2[tc.bad_b_idx_k] = tc.b_oob;
    std::string msg0, msg1;
    {
      Matrix<var, Dynamic, 1> aa(a), bb(b);
      var sigma(1.0);
      try {
        var lp = tc.shapeB ? stock_loop_guarded(y, aa, ii, x, bb, ii2, sigma)
                           : stock_loop_guarded(y, aa, ii, sigma);
        (void)lp;
        msg0 = "<no-throw>";
      } catch (const std::exception& e) {
        msg0 = e.what();
      }
      stan::math::recover_memory();
    }
    stan::math::recover_memory();
    {
      Matrix<var, Dynamic, 1> aa(a), bb(b);
      var sigma(1.0);
      stan::math::accumulator<var> lp_accum;
      try {
        std::vector<var> terms
            = tc.shapeB
                  ? stan::math::normal_lpdf_gathered<false>(y, aa, ii, x,
                                                            bb, ii2, sigma)
                  : stan::math::normal_lpdf_gathered<false>(y, aa, ii, sigma);
        for (const auto& t : terms) lp_accum.add(t);
        var lp = lp_accum.sum();
        (void)lp;
        msg1 = "<no-throw>";
      } catch (const std::exception& e) {
        msg1 = e.what();
      }
      stan::math::recover_memory();
    }
    stan::math::recover_memory();
    EXPECT_NE(msg0, "<no-throw>") << "stock must throw (case y=" << tc.y_nan_idx
                                  << " a@" << tc.bad_a_idx_k << ")";
    EXPECT_EQ(msg0, msg1) << "strict-order throw parity";
  }
}

// The W-53-class batched term records must zero-and-re-accumulate exactly
// like the stock loop's per-element var(double) records across repeated
// grad() calls on ONE tape (set_zero_all_adjoints between calls -- the
// gathered_term_zeroer path).
TEST(RevProbNormalLpdfGathered, BatchedRecordsRepeatedGrad) {
  std::mt19937 rng(20260831);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int J = 40, N = 500;
  VectorXd y(N), a(J), b(J), x(N);
  for (int n = 0; n < N; ++n) {
    y(n) = nd(rng);
    x(n) = (n % 3 == 0) ? -1.0 : 0.5;
  }
  for (int j = 0; j < J; ++j) {
    a(j) = nd(rng);
    b(j) = nd(rng) * 0.5;
  }
  std::vector<int> ii(N), ii2(N);
  for (int k = 0; k < N; ++k) {
    ii[k] = 1 + (int)(rng() % J);
    ii2[k] = 1 + (int)(rng() % J);
  }
  double lp0[2], lp1[2], s0[2], s1[2];
  VectorXd ga0[2], ga1[2], gb0[2], gb1[2];
  {
    Matrix<var, Dynamic, 1> aa(a), bb(b);
    var sigma(0.8);
    var lp = stock_loop(y, aa, ii, x, bb, ii2, sigma);
    for (int rep = 0; rep < 2; ++rep) {
      if (rep == 1) stan::math::set_zero_all_adjoints();
      lp.grad();
      lp0[rep] = lp.val();
      ga0[rep].resize(J);
      gb0[rep].resize(J);
      for (int j = 0; j < J; ++j) {
        ga0[rep](j) = aa.coeff(j).adj();
        gb0[rep](j) = bb.coeff(j).adj();
      }
      s0[rep] = sigma.adj();
    }
    stan::math::recover_memory();
  }
  stan::math::recover_memory();
  {
    Matrix<var, Dynamic, 1> aa(a), bb(b);
    var sigma(0.8);
    stan::math::accumulator<var> lp_accum;
    auto terms = stan::math::normal_lpdf_gathered<false>(y, aa, ii, x, bb,
                                                         ii2, sigma);
    for (const auto& t : terms) lp_accum.add(t);
    var lp = lp_accum.sum();
    for (int rep = 0; rep < 2; ++rep) {
      if (rep == 1) stan::math::set_zero_all_adjoints();
      lp.grad();
      lp1[rep] = lp.val();
      ga1[rep].resize(J);
      gb1[rep].resize(J);
      for (int j = 0; j < J; ++j) {
        ga1[rep](j) = aa.coeff(j).adj();
        gb1[rep](j) = bb.coeff(j).adj();
      }
      s1[rep] = sigma.adj();
    }
    stan::math::recover_memory();
  }
  stan::math::recover_memory();
  for (int rep = 0; rep < 2; ++rep) {
    EXPECT_TRUE(bits_equal(lp0[rep], lp1[rep]));
    EXPECT_TRUE(bits_equal(s0[rep], s1[rep]));
    for (int j = 0; j < J; ++j) {
      EXPECT_TRUE(bits_equal(ga0[rep](j), ga1[rep](j)));
      EXPECT_TRUE(bits_equal(gb0[rep](j), gb1[rep](j)));
    }
  }
}
