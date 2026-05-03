//! C-compatible API for the eml-spectral library.
//!
//! Build as a static or shared library:
//!   cargo build --release -p eml_spectral_c_api
//!
//! Outputs (in target/release/):
//!   libeml_spectral.a       — static library (link with -leml_spectral)
//!   libeml_spectral.so      — shared library (Linux/macOS)
//!   eml_spectral.dll        — dynamic library (Windows)
//!
//! Include eml_spectral.h in your C/C++ project. Pure Rust stdlib —
//! no PyO3, no Rayon, no third-party dependencies.

use std::os::raw::{c_double, c_int};

const OVERFLOW_THRESHOLD: f64 = 709.78;

// ── internal helpers ──────────────────────────────────────────────────────────

#[inline]
fn y_safe(y: f64) -> f64 {
    if y <= 0.0 { y.abs().max(1e-300) } else { y }
}

#[inline]
fn xv_safe(x: f64) -> f64 {
    if x > OVERFLOW_THRESHOLD { x.ln() } else { x }
}

// ─── Spectral flow (Φ operator) ─────────────────────────────────────────────

/// One Φ step: writes (y_safe, exp(xv_safe(x)) − ln(y_safe)) into outputs.
#[no_mangle]
pub unsafe extern "C" fn els_spectral_flow_step(
    x: c_double, y: c_double,
    out_x: *mut c_double, out_y: *mut c_double,
) {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    *out_x = ys;
    *out_y = xv.exp() - ys.ln();
}

/// Iterate Φ from (x0, y0) for n_steps. Caller pre-allocates n_steps+1 doubles
/// for both out_xs and out_ys; index 0 is the initial state.
#[no_mangle]
pub unsafe extern "C" fn els_spectral_flow(
    x0: c_double, y0: c_double, n_steps: usize,
    out_xs: *mut c_double, out_ys: *mut c_double,
) {
    *out_xs.add(0) = x0;
    *out_ys.add(0) = y0;
    let mut x = x0;
    let mut y = y0;
    for i in 1..=n_steps {
        let xv = xv_safe(x);
        let ys = y_safe(y);
        let t = xv.exp() - ys.ln();
        x = ys;
        y = t;
        *out_xs.add(i) = x;
        *out_ys.add(i) = y;
    }
}

/// Batch flow: n_starts independent trajectories, each of length n_steps+1.
/// Output buffers must be size n_starts * (n_steps + 1) row-major.
#[no_mangle]
pub unsafe extern "C" fn els_spectral_flow_batch(
    xs: *const c_double, ys: *const c_double,
    n_starts: usize, n_steps: usize,
    out_xs: *mut c_double, out_ys: *mut c_double,
) {
    let stride = n_steps + 1;
    for s in 0..n_starts {
        els_spectral_flow(
            *xs.add(s), *ys.add(s), n_steps,
            out_xs.add(s * stride), out_ys.add(s * stride),
        );
    }
}

// ─── Octonion (Fano-plane multiplication) ───────────────────────────────────

const FANO_LINES: [(usize, usize, usize); 7] =
    [(1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
     (1, 5, 6), (2, 6, 7), (1, 3, 7)];

const fn build_oct_table() -> [[(i8, usize); 8]; 8] {
    let mut t = [[(0i8, 0usize); 8]; 8];
    let mut i = 0;
    while i < 8 { t[0][i] = (1, i); t[i][0] = (1, i); i += 1; }
    let mut i = 1;
    while i < 8 { t[i][i] = (-1, 0); i += 1; }
    let mut li = 0;
    while li < 7 {
        let (a, b, c) = FANO_LINES[li];
        t[a][b] = (1, c); t[b][a] = (-1, c);
        t[b][c] = (1, a); t[c][b] = (-1, a);
        t[c][a] = (1, b); t[a][c] = (-1, b);
        li += 1;
    }
    t
}

const OCT_TABLE: [[(i8, usize); 8]; 8] = build_oct_table();

/// Multiply two octonions. a, b, out are 8-element arrays.
#[no_mangle]
pub unsafe extern "C" fn els_octonion_mul(
    a: *const c_double, b: *const c_double, out: *mut c_double,
) {
    let a = std::slice::from_raw_parts(a, 8);
    let b = std::slice::from_raw_parts(b, 8);
    let out = std::slice::from_raw_parts_mut(out, 8);
    for v in out.iter_mut() { *v = 0.0; }
    for i in 0..8 {
        if a[i] == 0.0 { continue; }
        for j in 0..8 {
            if b[j] == 0.0 { continue; }
            let (sign, k) = OCT_TABLE[i][j];
            out[k] += (sign as f64) * a[i] * b[j];
        }
    }
}

/// Octonion norm = √(Σ aᵢ²).
#[no_mangle]
pub unsafe extern "C" fn els_octonion_norm(a: *const c_double) -> c_double {
    let a = std::slice::from_raw_parts(a, 8);
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Batch multiply n pairs (stride 8 doubles per octonion in each buffer).
#[no_mangle]
pub unsafe extern "C" fn els_octonion_mul_batch(
    n: usize,
    a: *const c_double, b: *const c_double, out: *mut c_double,
) {
    for i in 0..n {
        els_octonion_mul(a.add(i * 8), b.add(i * 8), out.add(i * 8));
    }
}

// ─── Lorentz (Minkowski) ────────────────────────────────────────────────────

/// √|exp(2x) − (c·ln y)²| — Minkowski interval.
/// plus_signature=1 for (+−−−), 0 for (−+++).
#[no_mangle]
pub extern "C" fn els_minkowski_delta(
    x: c_double, y: c_double, plus_signature: c_int, c: c_double,
) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = c * ys.ln();
    let ds2 = if plus_signature != 0 { t * t - s * s } else { s * s - t * t };
    ds2.abs().sqrt()
}

/// Rapidity φ = atanh(ln(y) / exp(x)). Returns NaN if not timelike.
#[no_mangle]
pub extern "C" fn els_rapidity(x: c_double, y: c_double) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = ys.ln();
    if t.abs() < 1e-300 { return f64::NAN; }
    let r = s / t;
    if r.abs() >= 1.0 { return f64::NAN; }
    r.atanh()
}

/// Lorentz boost by rapidity phi with light-speed c.
#[no_mangle]
pub unsafe extern "C" fn els_boost(
    x: c_double, y: c_double, phi: c_double, c: c_double,
    out_x: *mut c_double, out_y: *mut c_double,
) {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = ys.ln();
    let sh = phi.sinh();
    let ch = phi.cosh();
    let t_new = (t * ch - (s / c) * sh).max(1e-300);
    let s_new = (s * ch - t * c * sh).clamp(-709.0, 709.0);
    *out_x = t_new.ln();
    *out_y = s_new.exp();
}

/// Batch boost: n independent (xs[i], ys[i], phis[i]) → (out_xs[i], out_ys[i]).
#[no_mangle]
pub unsafe extern "C" fn els_boost_batch(
    xs: *const c_double, ys: *const c_double, phis: *const c_double,
    c: c_double, n: usize,
    out_xs: *mut c_double, out_ys: *mut c_double,
) {
    for i in 0..n {
        els_boost(*xs.add(i), *ys.add(i), *phis.add(i), c,
                  out_xs.add(i), out_ys.add(i));
    }
}

// ─── Schwarzschild Christoffels ──────────────────────────────────────────────

/// Analytic Γ^lam_{mu nu} for the Schwarzschild metric (radial 2D slice).
/// Returns 0 outside r > rs > 0.
#[no_mangle]
pub extern "C" fn els_schwarzschild_christoffel(
    lam: usize, mu: usize, nu: usize,
    r: c_double, rs: c_double,
) -> c_double {
    if r <= rs || r <= 0.0 { return 0.0; }
    match (lam, mu, nu) {
        (0, 0, 1) | (0, 1, 0) => rs / (2.0 * r * (r - rs)),
        (1, 0, 0) => rs * (1.0 - rs / r) / (2.0 * r * r),
        (1, 1, 1) => -rs / (2.0 * r * (r - rs)),
        _ => 0.0,
    }
}

// ─── Clifford geometric product ──────────────────────────────────────────────

fn blade_product(ia: usize, jb: usize, signature: &[i8]) -> (f64, usize) {
    let n = signature.len();
    let mut swaps: u32 = 0;
    let mut bit = 1usize;
    for k in 0..n {
        if jb & bit != 0 {
            swaps += (ia >> (k + 1)).count_ones();
        }
        bit <<= 1;
    }
    let mut sign: f64 = if swaps & 1 == 1 { -1.0 } else { 1.0 };
    let shared = ia & jb;
    let mut s = shared;
    let mut k = 0usize;
    while s != 0 {
        if s & 1 == 1 { sign *= signature[k] as f64; }
        s >>= 1;
        k += 1;
    }
    (sign, ia ^ jb)
}

/// Geometric product in Cl(p, q). signature is an array of p+q ±1 ints.
/// a, b, out are arrays of 2^(p+q) doubles indexed by bitmask blade-id.
#[no_mangle]
pub unsafe extern "C" fn els_geometric_product(
    p: usize, q: usize,
    signature: *const c_int,
    a: *const c_double, b: *const c_double, out: *mut c_double,
) {
    let n = p + q;
    let total = 1usize << n;
    let sig: Vec<i8> = (0..n).map(|i| *signature.add(i) as i8).collect();
    let a_slice = std::slice::from_raw_parts(a, total);
    let b_slice = std::slice::from_raw_parts(b, total);
    let out_slice = std::slice::from_raw_parts_mut(out, total);
    for v in out_slice.iter_mut() { *v = 0.0; }
    for ia in 0..total {
        if a_slice[ia] == 0.0 { continue; }
        for jb in 0..total {
            if b_slice[jb] == 0.0 { continue; }
            let (sign, k) = blade_product(ia, jb, &sig);
            out_slice[k] += sign * a_slice[ia] * b_slice[jb];
        }
    }
}

// ─── Lattice utilities ───────────────────────────────────────────────────────

/// E₈: ‖pt‖² for 8-component point.
#[no_mangle]
pub unsafe extern "C" fn els_e8_norm_squared(pt: *const c_double) -> c_double {
    let pt = std::slice::from_raw_parts(pt, 8);
    pt.iter().map(|c| c * c).sum()
}

/// Leech: ‖pt‖² for 24-component point.
#[no_mangle]
pub unsafe extern "C" fn els_leech_norm_squared(pt: *const c_double) -> c_double {
    let pt = std::slice::from_raw_parts(pt, 24);
    pt.iter().map(|c| c * c).sum()
}
