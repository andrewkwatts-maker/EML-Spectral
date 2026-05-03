/**
 * eml_spectral.h — C API for the eml-spectral library.
 *
 * Domain layer of EML mathematics: Clifford algebras (geometric product),
 * octonions (Fano plane), Lorentz invariants, the spectral flow operator Φ,
 * and lattice helpers. Sister C API to eml_math.h; binary-compatible types.
 *
 * Build:
 *   cargo build --release -p eml_spectral_c_api
 *
 * Link in C/C++:
 *   gcc your_program.c -L./target/release -leml_spectral -lm -o your_program
 *
 * Frame-shift guard (Axiom 8): y ≤ 0 is replaced by |y| internally so every
 * function accepting a y-coordinate is well-defined on all of ℝ.
 */

#ifndef EML_SPECTRAL_H
#define EML_SPECTRAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Spectral flow (Φ operator) ────────────────────────────────────────── */

/** One Φ step: (x, y) → (|y|, exp(x) − ln|y|). Writes into *out_x, *out_y. */
void els_spectral_flow_step(double x, double y,
                             double *out_x, double *out_y);

/** Iterate Φ for n_steps. Caller pre-allocates n_steps+1 doubles for each
 *  output buffer. Index 0 = initial state. */
void els_spectral_flow(double x0, double y0, size_t n_steps,
                       double *out_xs, double *out_ys);

/** Batch flow: n_starts independent trajectories, each of length n_steps+1.
 *  Output buffers must be size n_starts * (n_steps + 1), row-major. */
void els_spectral_flow_batch(const double *xs, const double *ys,
                              size_t n_starts, size_t n_steps,
                              double *out_xs, double *out_ys);

/* ─── Octonion (Fano-plane multiplication) ──────────────────────────────── */

/** Octonion product: a[8] * b[8] → out[8]. Caller allocates out. */
void els_octonion_mul(const double a[8], const double b[8], double out[8]);

/** Octonion norm = √(Σ aᵢ²). */
double els_octonion_norm(const double a[8]);

/** Batch product: n pairs, stride 8 doubles per octonion in each buffer. */
void els_octonion_mul_batch(size_t n,
                             const double *a, const double *b, double *out);

/* ─── Lorentz invariants ─────────────────────────────────────────────────── */

/** sqrt(|exp(2x) − (c·ln y)²|) — Minkowski interval.
 *  plus_signature=1 for (+---), 0 for (-+++). */
double els_minkowski_delta(double x, double y,
                           int plus_signature, double c);

/** Rapidity φ = atanh(ln(y) / exp(x)).  φ is additive under sequential boosts.
 *  Returns NaN if the point is not timelike. */
double els_rapidity(double x, double y);

/** Lorentz boost by rapidity phi with speed of light c. */
void els_boost(double x, double y, double phi, double c,
               double *out_x, double *out_y);

/** Batch boost over n independent (xs[i], ys[i], phis[i]) tuples. */
void els_boost_batch(const double *xs, const double *ys,
                      const double *phis, double c, size_t n,
                      double *out_xs, double *out_ys);

/* ─── Schwarzschild Christoffel symbols ─────────────────────────────────── */

/** Analytic Gamma^lam_{mu nu} for Schwarzschild (radial 2D slice).
 *  Index convention: upper lam (contravariant), lower mu, nu.
 *  Signature: (-, +). Returns 0 for r ≤ rs or r ≤ 0 or unrecognised indices. */
double els_schwarzschild_christoffel(size_t lam, size_t mu, size_t nu,
                                     double r, double rs);

/* ─── Clifford geometric product ─────────────────────────────────────────── */

/** Geometric product in Cl(p, q).
 *  signature[k] is +1 or −1 for each of the p+q generators, in order.
 *  a, b, out are arrays of 2^(p+q) doubles indexed by bitmask blade-id. */
void els_geometric_product(size_t p, size_t q, const int *signature,
                            const double *a, const double *b, double *out);

/* ─── Lattice helpers ────────────────────────────────────────────────────── */

/** E₈ ‖x‖² for an 8-component point. */
double els_e8_norm_squared(const double pt[8]);

/** Leech ‖x‖² for a 24-component point. */
double els_leech_norm_squared(const double pt[24]);

#ifdef __cplusplus
}
#endif

#endif /* EML_SPECTRAL_H */
