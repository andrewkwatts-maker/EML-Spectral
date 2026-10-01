//! C-compatible API for the eml-spectral library.
//!
//! Build as a static or shared library:
//!   cargo build --release -p eml_spectral_c_api
//!
//! Outputs (in target/release/):
//!   libeml_spectral.a       -- static library (link with -leml_spectral)
//!   libeml_spectral.so      -- shared library (Linux/macOS)
//!   eml_spectral.dll        -- dynamic library (Windows)
//!
//! Include eml_spectral.h in your C/C++ project. Pure Rust stdlib --
//! no PyO3, no Rayon, no third-party dependencies.
//!
//! # Safety contract shared by every entry point
//!
//! There is no error channel in this ABI, so each function documents the
//! buffer sizes it requires and the caller is responsible for meeting them.
//! Across the whole API:
//!
//! - every pointer parameter must be non-null, correctly aligned for
//!   `double`, and valid for the number of elements named in the `# Safety`
//!   section of that function;
//! - output buffers must not alias the inputs;
//! - non-finite inputs are not rejected. They propagate under IEEE-754 rules,
//!   exactly as in the Rust `eml_spectral_core` crate.

use std::os::raw::{c_double, c_int};

const OVERFLOW_THRESHOLD: f64 = 709.78;

// -- internal helpers --------------------------------------------------------

#[inline]
fn y_safe(y: f64) -> f64 {
    if y <= 0.0 {
        y.abs().max(1e-300)
    } else {
        y
    }
}

#[inline]
fn xv_safe(x: f64) -> f64 {
    if x > OVERFLOW_THRESHOLD {
        x.ln()
    } else {
        x
    }
}

// --- Spectral flow (Phi operator) -------------------------------------------

/// One Phi step: writes (y_safe, exp(xv_safe(x)) - ln(y_safe)) into outputs.
///
/// # Safety
///
/// `out_x` and `out_y` must each be valid for a write of one `double`, and
/// must not alias each other.
#[no_mangle]
pub unsafe extern "C" fn els_spectral_flow_step(
    x: c_double,
    y: c_double,
    out_x: *mut c_double,
    out_y: *mut c_double,
) {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    *out_x = ys;
    *out_y = xv.exp() - ys.ln();
}

/// Iterate Phi from (x0, y0) for n_steps. Caller pre-allocates n_steps+1 doubles
/// for both out_xs and out_ys; index 0 is the initial state.
///
/// # Safety
///
/// `out_xs` and `out_ys` must each be valid for `n_steps + 1` `double` writes
/// and must not overlap. `n_steps + 1` must not overflow `usize`.
#[no_mangle]
pub unsafe extern "C" fn els_spectral_flow(
    x0: c_double,
    y0: c_double,
    n_steps: usize,
    out_xs: *mut c_double,
    out_ys: *mut c_double,
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
///
/// # Safety
///
/// `xs` and `ys` must each be valid for `n_starts` `double` reads; `out_xs` and
/// `out_ys` must each be valid for `n_starts * (n_steps + 1)` `double` writes
/// and must not overlap the inputs or each other. That product must not
/// overflow `usize`.
#[no_mangle]
pub unsafe extern "C" fn els_spectral_flow_batch(
    xs: *const c_double,
    ys: *const c_double,
    n_starts: usize,
    n_steps: usize,
    out_xs: *mut c_double,
    out_ys: *mut c_double,
) {
    let stride = n_steps + 1;
    for s in 0..n_starts {
        els_spectral_flow(
            *xs.add(s),
            *ys.add(s),
            n_steps,
            out_xs.add(s * stride),
            out_ys.add(s * stride),
        );
    }
}

// --- Octonion (Fano-plane multiplication) -----------------------------------

const OCTONION_DIM: usize = 8;

const FANO_LINES: [(usize, usize, usize); 7] = [
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 7),
    (1, 5, 6),
    (2, 6, 7),
    (1, 3, 7),
];

const fn build_oct_table() -> [[(i8, usize); OCTONION_DIM]; OCTONION_DIM] {
    let mut t = [[(0i8, 0usize); OCTONION_DIM]; OCTONION_DIM];
    let mut i = 0;
    while i < OCTONION_DIM {
        t[0][i] = (1, i);
        t[i][0] = (1, i);
        i += 1;
    }
    let mut i = 1;
    while i < OCTONION_DIM {
        t[i][i] = (-1, 0);
        i += 1;
    }
    let mut li = 0;
    while li < FANO_LINES.len() {
        let (a, b, c) = FANO_LINES[li];
        t[a][b] = (1, c);
        t[b][a] = (-1, c);
        t[b][c] = (1, a);
        t[c][b] = (-1, a);
        t[c][a] = (1, b);
        t[a][c] = (-1, b);
        li += 1;
    }
    t
}

const OCT_TABLE: [[(i8, usize); OCTONION_DIM]; OCTONION_DIM] = build_oct_table();

/// Multiply two octonions. a, b, out are 8-element arrays.
///
/// # Safety
///
/// `a` and `b` must each be valid for 8 `double` reads and `out` for 8 `double`
/// writes. `out` must not alias `a` or `b`: it is zeroed before accumulation.
#[no_mangle]
pub unsafe extern "C" fn els_octonion_mul(
    a: *const c_double,
    b: *const c_double,
    out: *mut c_double,
) {
    let a = std::slice::from_raw_parts(a, OCTONION_DIM);
    let b = std::slice::from_raw_parts(b, OCTONION_DIM);
    let out = std::slice::from_raw_parts_mut(out, OCTONION_DIM);
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        let row = &OCT_TABLE[i];
        for (j, &bj) in b.iter().enumerate() {
            if bj == 0.0 {
                continue;
            }
            let (sign, k) = row[j];
            out[k] += f64::from(sign) * ai * bj;
        }
    }
}

/// Octonion norm = sqrt(sum of squared components).
///
/// # Safety
///
/// `a` must be valid for 8 `double` reads.
#[no_mangle]
pub unsafe extern "C" fn els_octonion_norm(a: *const c_double) -> c_double {
    let a = std::slice::from_raw_parts(a, OCTONION_DIM);
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Batch multiply n pairs (stride 8 doubles per octonion in each buffer).
///
/// # Safety
///
/// `a` and `b` must each be valid for `n * 8` `double` reads and `out` for
/// `n * 8` `double` writes, with `out` not aliasing either input. `n * 8` must
/// not overflow `usize`.
#[no_mangle]
pub unsafe extern "C" fn els_octonion_mul_batch(
    n: usize,
    a: *const c_double,
    b: *const c_double,
    out: *mut c_double,
) {
    for i in 0..n {
        els_octonion_mul(
            a.add(i * OCTONION_DIM),
            b.add(i * OCTONION_DIM),
            out.add(i * OCTONION_DIM),
        );
    }
}

// --- Lorentz (Minkowski) ----------------------------------------------------

/// Minkowski interval **magnitude**: `sqrt(|exp(2x) - (c ln y)^2|)`.
///
/// `plus_signature` selects (+---) when non-zero and (-+++) when zero, and
/// **has no effect on this function's result**. That is not an oversight: the
/// two conventions differ only in the sign of `ds^2`, and taking the absolute
/// value before the square root discards exactly that sign. The magnitude of a
/// spacetime interval is genuinely convention-independent.
///
/// The parameter is kept because this is a C ABI and removing it would break
/// every caller. If you need the quantity the signature actually changes, use
/// [`els_minkowski_interval_squared`], which returns the **signed** `ds^2` and
/// so distinguishes timelike from spacelike separation -- something this
/// function cannot do.
#[no_mangle]
pub extern "C" fn els_minkowski_delta(
    x: c_double,
    y: c_double,
    plus_signature: c_int,
    c: c_double,
) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = c * ys.ln();
    let ds2 = if plus_signature != 0 {
        t * t - s * s
    } else {
        s * s - t * t
    };
    ds2.abs().sqrt()
}

/// Signed Minkowski interval `ds^2`, where the signature convention matters.
///
/// Returns `exp(2x) - (c ln y)^2` under (+---) when `plus_signature` is
/// non-zero, and its negation under (-+++). The sign is the point: positive is
/// timelike under (+---), negative is spacelike, and zero is null.
/// [`els_minkowski_delta`] cannot express that distinction because it returns a
/// magnitude.
#[no_mangle]
pub extern "C" fn els_minkowski_interval_squared(
    x: c_double,
    y: c_double,
    plus_signature: c_int,
    c: c_double,
) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = c * ys.ln();
    debug_assert!(!t.is_nan(), "xv_safe must not produce NaN");
    debug_assert!(
        !s.is_nan() || ys.ln().is_nan(),
        "c and ln(y) must be finite"
    );
    if plus_signature != 0 {
        t * t - s * s
    } else {
        s * s - t * t
    }
}

/// Rapidity phi = atanh(ln(y) / exp(x)). Returns NaN if not timelike.
#[no_mangle]
pub extern "C" fn els_rapidity(x: c_double, y: c_double) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = ys.ln();
    if t.abs() < 1e-300 {
        return f64::NAN;
    }
    let r = s / t;
    if r.abs() >= 1.0 {
        return f64::NAN;
    }
    r.atanh()
}

/// Lorentz boost by rapidity phi with light-speed c.
///
/// # Safety
///
/// `out_x` and `out_y` must each be valid for a write of one `double`, and must
/// not alias each other.
#[no_mangle]
pub unsafe extern "C" fn els_boost(
    x: c_double,
    y: c_double,
    phi: c_double,
    c: c_double,
    out_x: *mut c_double,
    out_y: *mut c_double,
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

/// Batch boost: n independent (xs[i], ys[i], phis[i]) -> (out_xs[i], out_ys[i]).
///
/// # Safety
///
/// `xs`, `ys` and `phis` must each be valid for `n` `double` reads; `out_xs`
/// and `out_ys` must each be valid for `n` `double` writes and must not alias
/// each other or the inputs.
#[no_mangle]
pub unsafe extern "C" fn els_boost_batch(
    xs: *const c_double,
    ys: *const c_double,
    phis: *const c_double,
    c: c_double,
    n: usize,
    out_xs: *mut c_double,
    out_ys: *mut c_double,
) {
    for i in 0..n {
        els_boost(
            *xs.add(i),
            *ys.add(i),
            *phis.add(i),
            c,
            out_xs.add(i),
            out_ys.add(i),
        );
    }
}

// --- Schwarzschild Christoffels ---------------------------------------------

/// Analytic Gamma^lam_{mu nu} for the Schwarzschild metric (radial 2D slice).
/// Returns 0 outside r > rs > 0.
#[no_mangle]
pub extern "C" fn els_schwarzschild_christoffel(
    lam: usize,
    mu: usize,
    nu: usize,
    r: c_double,
    rs: c_double,
) -> c_double {
    if r <= rs || r <= 0.0 {
        return 0.0;
    }
    match (lam, mu, nu) {
        (0, 0, 1) | (0, 1, 0) => rs / (2.0 * r * (r - rs)),
        (1, 0, 0) => rs * (1.0 - rs / r) / (2.0 * r * r),
        (1, 1, 1) => -rs / (2.0 * r * (r - rs)),
        _ => 0.0,
    }
}

// --- Clifford geometric product ---------------------------------------------

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
        if s & 1 == 1 {
            sign *= f64::from(signature[k]);
        }
        s >>= 1;
        k += 1;
    }
    (sign, ia ^ jb)
}

/// Geometric product in Cl(p, q). signature is an array of p+q entries, each
/// +1 or -1. a, b, out are arrays of 2^(p+q) doubles indexed by bitmask
/// blade-id.
///
/// # Safety
///
/// `p + q` must be at least 1 and at most 16; a larger sum overflows the blade
/// shift. `signature` must be valid for `p + q` `int` reads and each entry must
/// be +1 or -1. `a` and `b` must each be valid for `2^(p+q)` `double` reads and
/// `out` for as many `double` writes, with `out` not aliasing either input: it
/// is zeroed before accumulation.
#[no_mangle]
pub unsafe extern "C" fn els_geometric_product(
    p: usize,
    q: usize,
    signature: *const c_int,
    a: *const c_double,
    b: *const c_double,
    out: *mut c_double,
) {
    let n = p + q;
    let total = 1usize << n;
    let sig: Vec<i8> = std::slice::from_raw_parts(signature, n)
        .iter()
        .map(|&v| v as i8)
        .collect();
    let a_slice = std::slice::from_raw_parts(a, total);
    let b_slice = std::slice::from_raw_parts(b, total);
    let out_slice = std::slice::from_raw_parts_mut(out, total);
    for v in out_slice.iter_mut() {
        *v = 0.0;
    }
    for (ia, &av) in a_slice.iter().enumerate() {
        if av == 0.0 {
            continue;
        }
        for (jb, &bv) in b_slice.iter().enumerate() {
            if bv == 0.0 {
                continue;
            }
            let (sign, k) = blade_product(ia, jb, &sig);
            out_slice[k] += sign * av * bv;
        }
    }
}

// --- Lattice utilities ------------------------------------------------------

/// E8: squared norm of an 8-component point.
///
/// # Safety
///
/// `pt` must be valid for 8 `double` reads.
#[no_mangle]
pub unsafe extern "C" fn els_e8_norm_squared(pt: *const c_double) -> c_double {
    let pt = std::slice::from_raw_parts(pt, 8);
    pt.iter().map(|c| c * c).sum()
}

/// Leech: squared norm of a 24-component point.
///
/// # Safety
///
/// `pt` must be valid for 24 `double` reads.
#[no_mangle]
pub unsafe extern "C" fn els_leech_norm_squared(pt: *const c_double) -> c_double {
    let pt = std::slice::from_raw_parts(pt, 24);
    pt.iter().map(|c| c * c).sum()
}
