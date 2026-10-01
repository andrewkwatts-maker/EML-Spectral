//! Tests for the C ABI.
//!
//! The octonion table, the blade product and the two spectral-flow guards are
//! *duplicated* between this crate and `eml_spectral_core`. Every kernel that
//! exists in both is therefore checked against the other copy, because silent
//! drift between them is the failure mode this duplication invites. The Lorentz
//! and Schwarzschild entry points exist only here, so they are checked against
//! their own closed forms.

use eml_spectral::{
    els_boost, els_boost_batch, els_e8_norm_squared, els_geometric_product, els_leech_norm_squared,
    els_minkowski_delta, els_minkowski_interval_squared, els_octonion_mul, els_octonion_mul_batch,
    els_octonion_norm, els_rapidity, els_schwarzschild_christoffel, els_spectral_flow,
    els_spectral_flow_batch, els_spectral_flow_step,
};
use eml_spectral_core::clifford::geometric_product_n;
use eml_spectral_core::lattice::e8_norms_squared_n;
use eml_spectral_core::octonion::{octonion_mul, octonion_norm_n};
use eml_spectral_core::spectral::spectral_flow_n;

const TOL: f64 = 1e-12;

/// The C octonion product, as a safe Rust value.
fn c_octonion_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), 8, "the C ABI reads exactly eight doubles");
    debug_assert_eq!(b.len(), 8, "the C ABI reads exactly eight doubles");
    let mut out = vec![0.0f64; 8];
    // SAFETY: both inputs are 8 doubles, the output is a distinct 8-double
    // buffer, and none of the three alias.
    unsafe { els_octonion_mul(a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
    out
}

/// The C spectral flow, as a safe Rust value.
fn c_trajectory(x0: f64, y0: f64, n_steps: usize) -> Vec<(f64, f64)> {
    debug_assert!(n_steps < 4096, "test trajectories stay small");
    let mut xs = vec![0.0f64; n_steps + 1];
    let mut ys = vec![0.0f64; n_steps + 1];
    // SAFETY: both buffers hold n_steps + 1 doubles and do not overlap.
    unsafe { els_spectral_flow(x0, y0, n_steps, xs.as_mut_ptr(), ys.as_mut_ptr()) };
    let traj: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    debug_assert_eq!(traj.len(), n_steps + 1, "one state per step plus the start");
    traj
}

/// The C boost, as a safe Rust value.
fn c_boost(x: f64, y: f64, phi: f64, c: f64) -> (f64, f64) {
    debug_assert!(c != 0.0, "the boost divides by the light speed");
    debug_assert!(phi.is_finite(), "test rapidities are finite");
    let mut out_x = 0.0f64;
    let mut out_y = 0.0f64;
    // SAFETY: two distinct single-double outputs.
    unsafe { els_boost(x, y, phi, c, &mut out_x, &mut out_y) };
    (out_x, out_y)
}

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = e.abs().max(1.0);
        assert!(
            (a - e).abs() <= TOL * scale,
            "{what}: element {i} was {a}, expected {e}"
        );
    }
}

fn generic_octonions() -> (Vec<f64>, Vec<f64>) {
    (
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![8.0, -7.0, 6.0, -5.0, 4.0, -3.0, 2.0, -1.0],
    )
}

#[test]
fn the_c_octonion_product_matches_the_rust_core_product() {
    let (a, b) = generic_octonions();
    let mut cases = vec![(a.clone(), b.clone()), (b.clone(), a.clone())];
    for i in 0..8 {
        for j in 0..8 {
            let mut ei = vec![0.0; 8];
            let mut ej = vec![0.0; 8];
            ei[i] = 1.0;
            ej[j] = 1.0;
            cases.push((ei, ej));
        }
    }
    for (x, y) in &cases {
        let from_core = octonion_mul(x.clone(), y.clone()).expect("well-formed octonions");
        assert_eq!(
            c_octonion_mul(x, y),
            from_core,
            "the two Fano tables have drifted apart for {x:?} * {y:?}"
        );
    }
}

#[test]
fn the_c_octonion_product_has_a_multiplicative_norm() {
    let (a, b) = generic_octonions();
    // SAFETY: each input is an 8-double buffer.
    let (na, nb) = unsafe { (els_octonion_norm(a.as_ptr()), els_octonion_norm(b.as_ptr())) };
    let product = c_octonion_mul(&a, &b);
    // SAFETY: the product is an 8-double buffer.
    let n_product = unsafe { els_octonion_norm(product.as_ptr()) };
    assert!(
        (n_product - na * nb).abs() <= TOL * (na * nb),
        "|ab| was {n_product}, |a||b| was {}",
        na * nb
    );
}

#[test]
fn the_c_octonion_norm_matches_the_rust_core_norm() {
    let (a, b) = generic_octonions();
    let batch = vec![a.clone(), b.clone(), vec![0.0; 8]];
    let from_core = octonion_norm_n(batch.clone()).expect("well-formed octonions");
    for (i, v) in batch.iter().enumerate() {
        // SAFETY: each element is an 8-double buffer.
        let from_c = unsafe { els_octonion_norm(v.as_ptr()) };
        assert_eq!(from_c, from_core[i], "norm {i} differs between the crates");
    }
}

#[test]
fn the_c_octonion_batch_matches_repeated_single_products() {
    let (a, b) = generic_octonions();
    let n = 3usize;
    let mut a_flat = Vec::with_capacity(n * 8);
    let mut b_flat = Vec::with_capacity(n * 8);
    for k in 0..n {
        a_flat.extend(a.iter().map(|c| c + k as f64));
        b_flat.extend(b.iter().map(|c| c - k as f64));
    }
    let mut out = vec![0.0f64; n * 8];
    // SAFETY: all three buffers hold n * 8 doubles and the output does not
    // alias the inputs.
    unsafe {
        els_octonion_mul_batch(n, a_flat.as_ptr(), b_flat.as_ptr(), out.as_mut_ptr());
    }
    for k in 0..n {
        let expected = c_octonion_mul(&a_flat[k * 8..(k + 1) * 8], &b_flat[k * 8..(k + 1) * 8]);
        assert_eq!(
            &out[k * 8..(k + 1) * 8],
            expected.as_slice(),
            "batch row {k} differs from the single product"
        );
    }
}

#[test]
fn the_c_spectral_flow_matches_the_rust_core_trajectory() {
    let starts = [
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -2.0),
        (1e300, 1e300),
        (709.7827, 1e-300),
    ];
    for (x0, y0) in starts {
        let from_core = spectral_flow_n(x0, y0, 16).expect("a small step count is accepted");
        let from_c = c_trajectory(x0, y0, 16);
        assert_eq!(from_c.len(), from_core.len(), "trajectory length");
        for (i, (state_c, state_core)) in from_c.iter().zip(from_core.iter()).enumerate() {
            assert_eq!(
                state_c.0.to_bits(),
                state_core.0.to_bits(),
                "start ({x0}, {y0}) state {i}: x differs between the crates"
            );
            assert_eq!(
                state_c.1.to_bits(),
                state_core.1.to_bits(),
                "start ({x0}, {y0}) state {i}: y differs between the crates"
            );
        }
    }
}

#[test]
fn the_c_spectral_flow_step_matches_the_first_step_of_the_trajectory() {
    let mut out_x = 0.0f64;
    let mut out_y = 0.0f64;
    // SAFETY: two distinct single-double outputs.
    unsafe { els_spectral_flow_step(2.0, -3.0, &mut out_x, &mut out_y) };
    let traj = c_trajectory(2.0, -3.0, 1);
    assert_eq!(
        (out_x.to_bits(), out_y.to_bits()),
        (traj[1].0.to_bits(), traj[1].1.to_bits()),
        "one step must equal the first step of an iterated trajectory"
    );
}

#[test]
fn the_c_batch_flow_writes_one_row_per_start() {
    let xs = [1.0, 0.0, -4.0];
    let ys = [2.0, 0.0, 0.5];
    let n_steps = 5usize;
    let stride = n_steps + 1;
    let mut out_xs = vec![0.0f64; xs.len() * stride];
    let mut out_ys = vec![0.0f64; xs.len() * stride];
    // SAFETY: inputs hold 3 doubles each, outputs hold 3 * stride doubles each,
    // and no buffer aliases another.
    unsafe {
        els_spectral_flow_batch(
            xs.as_ptr(),
            ys.as_ptr(),
            xs.len(),
            n_steps,
            out_xs.as_mut_ptr(),
            out_ys.as_mut_ptr(),
        );
    }
    for (row, (x0, y0)) in xs.iter().zip(ys.iter()).enumerate() {
        let single = c_trajectory(*x0, *y0, n_steps);
        for (i, state) in single.iter().enumerate() {
            assert_eq!(
                out_xs[row * stride + i].to_bits(),
                state.0.to_bits(),
                "row {row} state {i}: x"
            );
            assert_eq!(
                out_ys[row * stride + i].to_bits(),
                state.1.to_bits(),
                "row {row} state {i}: y"
            );
        }
    }
}

#[test]
fn the_c_geometric_product_matches_the_rust_core_product() {
    // Cl(1,3) and Cl(3,0), with every blade populated so no zero-skip can hide
    // a sign difference between the two blade_product copies.
    let cases: [(usize, usize, Vec<i32>); 2] = [(1, 3, vec![1, -1, -1, -1]), (3, 0, vec![1, 1, 1])];
    for (p, q, signature) in cases {
        let n = p + q;
        let total = 1usize << n;
        let a: Vec<f64> = (0..total).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let b: Vec<f64> = (0..total).map(|i| 2.0 - (i as f64) * 0.25).collect();

        let mut out = vec![0.0f64; total];
        // SAFETY: signature holds p + q ints, a and b hold 2^(p+q) doubles
        // each, and out is a distinct buffer of the same length.
        unsafe {
            els_geometric_product(
                p,
                q,
                signature.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
            );
        }

        let sig_i8: Vec<i8> = signature.iter().map(|v| *v as i8).collect();
        let from_core = geometric_product_n(vec![a.clone()], vec![b.clone()], sig_i8)
            .expect("well-formed multivectors");
        assert_close(
            &out,
            &from_core[0],
            "geometric product across the two crates",
        );
    }
}

#[test]
fn the_c_lattice_norms_match_the_rust_core_batch_norm() {
    let e8_point = vec![0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5];
    let leech_point: Vec<f64> = (0..24).map(|i| ((i % 5) as f64) - 2.0).collect();

    // SAFETY: the first point holds 8 doubles, the second holds 24.
    let (c_e8, c_leech) = unsafe {
        (
            els_e8_norm_squared(e8_point.as_ptr()),
            els_leech_norm_squared(leech_point.as_ptr()),
        )
    };
    let from_core = e8_norms_squared_n(vec![e8_point.clone(), leech_point.clone()])
        .expect("well-formed points");
    assert_eq!(c_e8, from_core[0], "E8 squared norm differs between crates");
    assert_eq!(
        c_leech, from_core[1],
        "Leech squared norm differs between crates"
    );
    assert!(
        (c_e8 - 2.0).abs() <= TOL,
        "this half-integer point is an E8 root, so its squared norm is 2"
    );
}

#[test]
fn a_boost_and_its_inverse_return_the_original_point() {
    let c = 1.0;
    for (x, y, phi) in [(0.5, 2.0, 0.3), (-1.0, 0.75, -0.8), (2.0, 5.0, 1.25)] {
        let (bx, by) = c_boost(x, y, phi, c);
        let (rx, ry) = c_boost(bx, by, -phi, c);
        assert!(
            (rx - x).abs() <= 1e-9 * x.abs().max(1.0),
            "x round-tripped to {rx}, expected {x}"
        );
        assert!(
            (ry - y).abs() <= 1e-9 * y.abs().max(1.0),
            "y round-tripped to {ry}, expected {y}"
        );
    }
}

#[test]
fn a_boost_by_zero_rapidity_is_the_identity() {
    for (x, y) in [(0.5, 2.0), (-1.0, 0.75)] {
        let (bx, by) = c_boost(x, y, 0.0, 1.0);
        assert!(
            (bx - x).abs() <= 1e-12,
            "x moved to {bx} under a null boost"
        );
        assert!(
            (by - y).abs() <= 1e-12,
            "y moved to {by} under a null boost"
        );
    }
}

#[test]
fn the_c_boost_batch_matches_repeated_single_boosts() {
    let xs = [0.5, -1.0, 2.0];
    let ys = [2.0, 0.75, 5.0];
    let phis = [0.3, -0.8, 1.25];
    let c = 1.0;
    let mut out_xs = vec![0.0f64; xs.len()];
    let mut out_ys = vec![0.0f64; xs.len()];
    // SAFETY: three input buffers of 3 doubles, two distinct output buffers of
    // the same length.
    unsafe {
        els_boost_batch(
            xs.as_ptr(),
            ys.as_ptr(),
            phis.as_ptr(),
            c,
            xs.len(),
            out_xs.as_mut_ptr(),
            out_ys.as_mut_ptr(),
        );
    }
    for i in 0..xs.len() {
        let (bx, by) = c_boost(xs[i], ys[i], phis[i], c);
        assert_eq!(out_xs[i].to_bits(), bx.to_bits(), "row {i}: x");
        assert_eq!(out_ys[i].to_bits(), by.to_bits(), "row {i}: y");
    }
}

#[test]
fn the_rapidity_is_the_inverse_hyperbolic_tangent_of_the_velocity_ratio() {
    // With x = 0 the time-like component exp(x) is 1, so ln(y) is the ratio and
    // tanh of the returned rapidity has to reproduce it.
    let y = (0.5f64).exp();
    let phi = els_rapidity(0.0, y);
    assert!(phi.is_finite(), "a time-like point has a real rapidity");
    assert!(
        (phi.tanh() - 0.5).abs() <= 1e-12,
        "tanh of the rapidity was {}, expected 0.5",
        phi.tanh()
    );
}

#[test]
fn the_rapidity_returns_nan_for_a_point_that_is_not_time_like() {
    // Documented sentinel: |ln(y) / exp(x)| >= 1 has no real rapidity.
    let space_like = els_rapidity(0.0, (2.0f64).exp());
    assert!(
        space_like.is_nan(),
        "a ratio of 2 is outside the domain of atanh, got {space_like}"
    );
    let light_like = els_rapidity(0.0, (1.0f64).exp());
    assert!(
        light_like.is_nan(),
        "a ratio of exactly 1 is still outside the domain, got {light_like}"
    );
}

#[test]
fn the_minkowski_delta_ignores_its_signature_flag_because_it_returns_a_magnitude() {
    // Pins current behaviour: the function squares both terms and takes the
    // absolute value before the square root, so (+---) and (-+++) cannot be
    // told apart from the result. Anything that needs the sign of the interval
    // has to compute it itself.
    for (x, y) in [(0.5, 2.0), (-1.0, 0.75), (3.0, 1e-8)] {
        let plus = els_minkowski_delta(x, y, 1, 1.0);
        let minus = els_minkowski_delta(x, y, 0, 1.0);
        assert_eq!(
            plus.to_bits(),
            minus.to_bits(),
            "the two signature conventions gave different magnitudes at ({x}, {y})"
        );
        assert!(plus >= 0.0, "a magnitude is never negative");
    }
}

#[test]
fn the_schwarzschild_christoffels_match_their_closed_forms() {
    let r = 10.0;
    let rs = 2.0;
    let expected_t_tr = rs / (2.0 * r * (r - rs));
    let expected_r_tt = rs * (1.0 - rs / r) / (2.0 * r * r);

    assert!(
        (els_schwarzschild_christoffel(0, 0, 1, r, rs) - expected_t_tr).abs() <= TOL,
        "Gamma^t_{{tr}}"
    );
    assert!(
        (els_schwarzschild_christoffel(1, 0, 0, r, rs) - expected_r_tt).abs() <= TOL,
        "Gamma^r_{{tt}}"
    );
    assert!(
        (els_schwarzschild_christoffel(1, 1, 1, r, rs) + expected_t_tr).abs() <= TOL,
        "Gamma^r_{{rr}} is the negative of Gamma^t_{{tr}} in these coordinates"
    );
}

#[test]
fn the_schwarzschild_christoffels_are_symmetric_in_their_lower_indices() {
    // A Levi-Civita connection is torsion-free, so swapping mu and nu cannot
    // change the value.
    let r = 7.5;
    let rs = 1.5;
    for lam in 0..3usize {
        for mu in 0..3usize {
            for nu in 0..3usize {
                assert_eq!(
                    els_schwarzschild_christoffel(lam, mu, nu, r, rs).to_bits(),
                    els_schwarzschild_christoffel(lam, nu, mu, r, rs).to_bits(),
                    "Gamma^{lam}_{{{mu}{nu}}} is not symmetric in its lower indices"
                );
            }
        }
    }
}

#[test]
fn the_schwarzschild_christoffels_return_zero_at_and_inside_the_horizon() {
    // Documented sentinel: the Schwarzschild chart breaks down at r = rs, so
    // the function reports 0 rather than dividing by zero.
    let rs = 2.0;
    for r in [rs, rs * 0.5, 0.0, -1.0] {
        assert_eq!(
            els_schwarzschild_christoffel(0, 0, 1, r, rs),
            0.0,
            "r = {r} is not outside the horizon"
        );
    }
    assert!(
        els_schwarzschild_christoffel(0, 0, 1, rs * 1.0001, rs).is_finite(),
        "just outside the horizon the connection is large but finite"
    );
}

#[test]
fn the_signed_interval_is_what_the_signature_actually_changes() {
    // `els_minkowski_delta` returns a magnitude, so its `plus_signature`
    // argument provably cannot matter -- `abs()` discards the only thing
    // the convention affects. The signed form is the one that carries it.
    let (x, y, c) = (1.0_f64, 2.0_f64, 1.0_f64);
    let plus = els_minkowski_interval_squared(x, y, 1, c);
    let minus = els_minkowski_interval_squared(x, y, 0, c);
    assert!(plus.is_finite() && minus.is_finite());
    assert_eq!(plus, -minus, "the two conventions must be exact negations");
    assert_ne!(
        plus, minus,
        "if these agreed the parameter would be a no-op here too"
    );
}

#[test]
fn the_magnitude_is_the_root_of_the_absolute_signed_interval() {
    // The two functions must stay consistent: |ds| = sqrt(|ds^2|).
    for (x, y) in [(1.0, 2.0), (0.5, 0.25), (-1.0, 3.0), (2.0, 1.0)] {
        let squared = els_minkowski_interval_squared(x, y, 1, 1.0);
        let magnitude = els_minkowski_delta(x, y, 1, 1.0);
        let expected = squared.abs().sqrt();
        assert!(
            (magnitude - expected).abs() < 1e-12,
            "at ({x}, {y}): magnitude {magnitude} vs sqrt(|{squared}|) = {expected}"
        );
    }
}

#[test]
fn the_sign_classifies_the_separation() {
    // Under (+---): positive is timelike, negative spacelike, zero null.
    // y = 1 gives ln(y) = 0, so the spatial part vanishes and the interval
    // is purely timelike whatever x is.
    assert!(
        els_minkowski_interval_squared(1.0, 1.0, 1, 1.0) > 0.0,
        "a vanishing spatial part must read as timelike"
    );
    // A large spatial term dominates the temporal one.
    assert!(
        els_minkowski_interval_squared(-5.0, 1000.0, 1, 1.0) < 0.0,
        "a dominant spatial part must read as spacelike"
    );
}
