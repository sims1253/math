#include <stan/math/rev/prob/pcm_lpdf_gathered.hpp>
#include <stan/math.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// The gathered pcm likelihood must be BIT-IDENTICAL to the loop the Stan
// compiler emits for models with the user function
//   real pcm(int y, real theta, vector beta) {
//     vector[rows(beta) + 1] unsummed
//         = append_row(rep_vector(0.0, 1), theta - beta);
//     probs = softmax(cumulative_sum(unsummed));
//     return categorical_lpmf(y + 1 | probs);
//   }
// called as  for (n) target += pcm(y[n], theta[jj[n]] * alpha[ii[n]],
//                                  segment(beta, pos[ii[n]], m[ii[n]]));
// Values and every gradient component are compared with memcmp, not a
// tolerance. The softmax interior is evaluated through the same prim
// instantiation the composed path hits (a val() view over a var matrix):
// Eigen's interior traversal depends on the input expression type, and the
// view form is the one the rev softmax produces (this matters on every
// stack; see the header's implementation notes).
namespace {

using Eigen::Dynamic;
using stan::math::accumulator;
using stan::math::rep_vector;
using stan::math::append_row;
using stan::math::categorical_lpmf;
using stan::math::cumulative_sum;
using stan::math::pcm_lpdf_gathered;
using stan::math::softmax;
using stan::math::subtract;
using stan::math::to_ref;
using stan::math::var;
using stan::math::var_value;
using VectorXd = Eigen::Matrix<double, Dynamic, 1>;

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

// layout: 0 = Matrix<var> (AoS); 1 = var_value<VectorXd> (SoA)
template <int layout>
struct Coeffs {
  Eigen::Matrix<var, Dynamic, 1> aos;
  var_value<VectorXd> soa{VectorXd(0)};
  explicit Coeffs(const VectorXd& v) : aos(v) {
    if constexpr (layout == 1) {
      soa = var_value<VectorXd>(v);
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

// The composed stock loop (AoS beta; the generated model's local beta is
// always Matrix<var>, so this is the canonical stock form).
template <int LT, int LA>
var stock_loop(const std::vector<int>& y, const Coeffs<LT>& theta,
               const std::vector<int>& jj, const Coeffs<LA>& alpha,
               const std::vector<int>& ii,
               const Eigen::Matrix<var, Dynamic, 1>& beta,
               const std::vector<int>& pos, const std::vector<int>& m,
               bool priors_first = false) {
  accumulator<var> lp_accum;
  if (priors_first) {
    lp_accum.add(stan::math::normal_lpdf<false>(theta.aos, 0.0, 2.0));
  }
  const int N = y.size();
  for (int n = 0; n < N; ++n) {
    var th = get(theta).coeff(jj[n] - 1);
    var al = get(alpha).coeff(ii[n] - 1);
    Eigen::Matrix<var, Dynamic, 1> b(m[ii[n] - 1]);
    for (int k = 0; k < m[ii[n] - 1]; ++k) {
      b(k) = beta.coeff(pos[ii[n] - 1] - 1 + k);
    }
    var t = th * al;
    Eigen::Matrix<var, Dynamic, 1> unsummed =
        append_row(rep_vector(0.0, 1), subtract(t, to_ref(b)));
    Eigen::Matrix<var, Dynamic, 1> cs = cumulative_sum(unsummed);
    auto p = softmax(cs);
    Eigen::Matrix<var, Dynamic, 1> probs(p.rows());
    for (int k = 0; k < p.rows(); ++k) {
      probs(k) = p.coeff(k);
    }
    lp_accum.add(categorical_lpmf<false>(y[n] + 1, probs));
  }
  return lp_accum.sum();
}

var prim_loop(const std::vector<int>& y,
              const Eigen::Matrix<var, Dynamic, 1>& theta,
              const std::vector<int>& jj,
              const Eigen::Matrix<var, Dynamic, 1>& alpha,
              const std::vector<int>& ii,
              const Eigen::Matrix<var, Dynamic, 1>& beta,
              const std::vector<int>& pos, const std::vector<int>& m) {
  accumulator<var> lp_accum;
  auto terms = pcm_lpdf_gathered<false>(y, theta, jj, alpha, ii, beta, pos, m);
  for (const auto& t : terms) {
    lp_accum.add(t);
  }
  return lp_accum.sum();
}

template <int LT, int LA>
void check_bitwise(unsigned seed, int N, int I, int J,
                   const std::vector<int>& m_in, bool priors_first) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1.5, 1.5);
  std::vector<int> m = m_in, pos(I);
  int tot = 0;
  for (int i = 0; i < I; ++i) {
    pos[i] = tot + 1;
    tot += m[i];
  }
  std::vector<int> y(N), jj(N), ii(N);
  for (int n = 0; n < N; ++n) {
    jj[n] = 1 + static_cast<int>(rng() % J);
    ii[n] = 1 + static_cast<int>(rng() % I);
    y[n] = static_cast<int>(rng() % (m[ii[n] - 1] + 1));
  }
  VectorXd th_v(J), al_v(I), b_v(tot);
  for (int j = 0; j < J; ++j) th_v(j) = U(rng);
  for (int i = 0; i < I; ++i) al_v(i) = std::abs(U(rng)) + 0.25;
  for (int k = 0; k < tot; ++k) b_v(k) = U(rng);

  // stock arm
  double lp1;
  VectorXd g1;
  {
    Coeffs<LT> theta(th_v);
    Coeffs<LA> alpha(al_v);
    Eigen::Matrix<var, Dynamic, 1> beta(b_v);
    var lp = stock_loop(y, theta, jj, alpha, ii, beta, pos, m, priors_first);
    stan::math::grad(lp.vi_);
    lp1 = lp.val();
    VectorXd gt = theta.adj(), ga = alpha.adj(), gb(beta.size());
    for (int k = 0; k < beta.size(); ++k) gb(k) = beta.coeff(k).adj();
    g1.resize(gt.size() + ga.size() + gb.size());
    g1 << gt, ga, gb;
    stan::math::recover_memory();
  }
  // prim arm (AoS operands; the SoA routes are exercised below and in the
  // harness noted in the header)
  double lp2;
  VectorXd g2;
  {
    Coeffs<LT> theta(th_v);
    Coeffs<LA> alpha(al_v);
    Eigen::Matrix<var, Dynamic, 1> beta(b_v);
    accumulator<var> lp_accum;
    if (priors_first) {
      // the model's statement order: the prior runs BEFORE the likelihood
      // statement (which is when the primitive -- and its callback -- runs)
      lp_accum.add(stan::math::normal_lpdf<false>(theta.aos, 0.0, 2.0));
    }
    auto terms = pcm_lpdf_gathered<false>(y, get(theta), jj, get(alpha), ii,
                                          beta, pos, m);
    for (const auto& t : terms) {
      lp_accum.add(t);
    }
    var lp = lp_accum.sum();
    stan::math::grad(lp.vi_);
    lp2 = lp.val();
    VectorXd gt = theta.adj(), ga = alpha.adj(), gb(beta.size());
    for (int k = 0; k < beta.size(); ++k) gb(k) = beta.coeff(k).adj();
    g2.resize(gt.size() + ga.size() + gb.size());
    g2 << gt, ga, gb;
    stan::math::recover_memory();
  }
  EXPECT_TRUE(bits_equal(lp1, lp2));
  ASSERT_EQ(g1.size(), g2.size());
  for (Eigen::Index k = 0; k < g1.size(); ++k) {
    EXPECT_TRUE(bits_equal(g1(k), g2(k))) << "component " << k;
  }
}

}  // namespace

TEST(RevProbPcmLpdfGathered, BitIdenticalToComposedStock) {
  check_bitwise<0, 0>(1, 1, 1, 1, {1}, false);
  check_bitwise<0, 0>(2, 7, 3, 4, {1, 2, 4}, false);
  check_bitwise<0, 1>(3, 17, 4, 5, {2, 1, 3, 7}, false);
  check_bitwise<1, 0>(4, 33, 2, 6, {5, 3}, false);
  check_bitwise<1, 1>(5, 100, 5, 8, {1, 2, 3, 4, 5}, false);
  check_bitwise<0, 0>(6, 519, 3, 9, {2, 4, 6}, false);
}

TEST(RevProbPcmLpdfGathered, PriorsBeforeLikelihood) {
  // the model's statement order (priors before the likelihood): the
  // likelihood-site callback must chain at stock's position
  check_bitwise<0, 0>(7, 40, 3, 4, {2, 3, 1}, true);
  check_bitwise<0, 0>(8, 77, 2, 3, {4, 2}, true);
}

TEST(RevProbPcmLpdfGathered, ValueMatchesReference) {
  // term values == log(p[y]) with p from the same prim-softmax
  // instantiation the composed path uses, and the probs sum to 1
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> U(-2.0, 2.0);
  int I = 3, J = 2;
  std::vector<int> m{1, 3, 5}, pos{1, 2, 5};
  int tot = 9;
  std::vector<int> y{0, 2, 4}, jj{1, 2, 1}, ii{1, 2, 3};
  VectorXd th_v(J), al_v(I), b_v(tot);
  for (int j = 0; j < J; ++j) th_v(j) = U(rng);
  for (int i = 0; i < I; ++i) al_v(i) = std::abs(U(rng)) + 0.5;
  for (int k = 0; k < tot; ++k) b_v(k) = U(rng);
  Eigen::Matrix<var, Dynamic, 1> theta(th_v), alpha(al_v), beta(b_v);
  auto terms = pcm_lpdf_gathered<false>(y, theta, jj, alpha, ii, beta, pos, m);
  ASSERT_EQ(terms.size(), 3);
  for (int n = 0; n < 3; ++n) {
    int K = m[ii[n] - 1] + 1;
    double t = th_v(jj[n] - 1) * al_v(ii[n] - 1);
    VectorXd c(K);
    c(0) = 0.0;
    for (int k = 1; k < K; ++k) {
      c(k) = c(k - 1) + (t - b_v(pos[ii[n] - 1] - 1 + k - 1));
    }
    Eigen::Matrix<var, Dynamic, 1> cvar(K);
    for (int k = 0; k < K; ++k) cvar(k) = var(c(k));
    VectorXd p = softmax(cvar.val());
    EXPECT_TRUE(bits_equal(terms[n].val(), std::log(p(y[n])))) << "obs " << n;
  }
  stan::math::recover_memory();
}

TEST(RevProbPcmLpdfGathered, ThrowSet) {
  int I = 3, J = 3;
  std::vector<int> m{2, 1, 3}, pos{1, 3, 4};
  VectorXd th_v(J), al_v(I), b_v(6);
  th_v << 0.5, -0.8, 1.1;
  al_v << 1.2, 0.9, 1.5;
  b_v << 0.1, -0.2, 0.3, -0.4, 0.5, 0.6;
  std::vector<int> jj{1, 2, 3}, ii{1, 2, 3};
  auto stock_msg = [&](const std::vector<int>& y, double th_replace,
                       int th_idx) {
    // composed reference throw (same interior checks)
    try {
      VectorXd th = th_v;
      if (th_idx >= 0) th(th_idx) = th_replace;
      Eigen::Matrix<var, Dynamic, 1> theta(th), alpha(al_v), beta(b_v);
      std::vector<int> yy = y;
      auto terms = pcm_lpdf_gathered<false>(yy, theta, jj, alpha, ii, beta,
                                            pos, m);
      (void)terms;
      return std::string("(no throw)");
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
  };
  // y out of range both ends: message must be categorical_lpmf's
  // check_bounded text with the same numbers
  std::vector<int> ylow{0, 1, 2};
  ylow[0] = -1;
  std::vector<int> yhigh{0, 1, 2};
  yhigh[0] = 3;  // item 0 has K = 3
  EXPECT_NE(stock_msg(ylow, 0.0, -1).find("Number of categories"),
            std::string::npos);
  EXPECT_NE(stock_msg(yhigh, 0.0, -1).find("Number of categories"),
            std::string::npos);
  // non-finite parameters: the non-simplex message
  EXPECT_NE(stock_msg({0, 1, 2}, std::nan(""), 1).find("not a valid simplex"),
            std::string::npos);
  // in-range baseline: no throw
  EXPECT_EQ(stock_msg({0, 1, 2}, 0.0, -1), "(no throw)");
  // out-of-range indices
  try {
    Eigen::Matrix<var, Dynamic, 1> theta(th_v), alpha(al_v), beta(b_v);
    std::vector<int> jbad{1, 0, 3}, yv{0, 1, 2};
    pcm_lpdf_gathered<false>(yv, theta, jbad, alpha, ii, beta, pos, m);
    FAIL() << "expected throw for jj=0";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("out of range"), std::string::npos);
  }
  stan::math::recover_memory();
}

TEST(RevProbPcmLpdfGathered, SizeZero) {
  Eigen::Matrix<var, Dynamic, 1> theta(2), alpha(2), beta(3);
  theta.setZero();
  alpha.setOnes();
  beta.setZero();
  std::vector<int> m{2, 1}, pos{1, 3};
  auto terms = pcm_lpdf_gathered<false>({}, theta, {}, alpha, {}, beta, pos,
                                        m);
  EXPECT_EQ(terms.size(), 0);
  stan::math::recover_memory();
}
