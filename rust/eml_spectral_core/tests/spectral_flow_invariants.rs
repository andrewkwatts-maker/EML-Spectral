//! Invariants of the discrete spectral flow.
//!
//! The flow conserves no quantity -- it is an iterated exp/log map, not a
//! symplectic or measure-preserving one -- so there is no conserved scalar to
//! assert. What it *does* guarantee, and what the two guards in the
//! implementation exist for, is:
//!
//! 1. the frame-shift guard keeps the first coordinate of every evolved state
//!    strictly positive, so no later step can take `ln(0)`;
//! 2. the overflow guard keeps `exp` in range, so a finite state can never
//!    evolve into a non-finite one, however many steps are taken;
//! 3. the trajectory and batch entry points are exactly the step function
//!    iterated -- no drift between the three APIs.
//!
//! Those three are what these tests pin.

use eml_spectral_core::spectral::{
    spectral_flow_batch, spectral_flow_n, spectral_flow_step, MAX_FLOW_STEPS,
};

/// Starting points chosen to hit both guards and the ordinary path: zero,
/// negative and denormal y values, and x values on both sides of the
/// `exp` overflow threshold of 709.78.
fn extreme_starts() -> Vec<(f64, f64)> {
    let starts = vec![
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -1.0),
        (2.5, 0.0),
        (709.0, 1e-300),
        (709.78, 1.0),
        (709.7827, 1.0),
        (1e300, 1e300),
        (-1e300, -1e300),
        (f64::MAX, f64::MIN),
        (1e-320, -1e-320),
    ];
    debug_assert!(
        starts.iter().all(|(x, y)| x.is_finite() && y.is_finite()),
        "the finiteness invariant is only claimed for finite starting points"
    );
    debug_assert!(
        starts.iter().any(|(_, y)| *y <= 0.0),
        "the set must exercise the frame-shift guard"
    );
    starts
}

fn assert_bitwise_eq(actual: (f64, f64), expected: (f64, f64), what: &str) {
    assert_eq!(
        actual.0.to_bits(),
        expected.0.to_bits(),
        "{what}: x was {}, expected {}",
        actual.0,
        expected.0
    );
    assert_eq!(
        actual.1.to_bits(),
        expected.1.to_bits(),
        "{what}: y was {}, expected {}",
        actual.1,
        expected.1
    );
}

#[test]
fn a_trajectory_holds_the_starting_state_plus_one_state_per_step() {
    for steps in [0usize, 1, 2, 17] {
        let traj = spectral_flow_n(1.5, 2.5, steps).expect("a small step count is accepted");
        assert_eq!(traj.len(), steps + 1, "length is n_steps + 1");
        assert_bitwise_eq(traj[0], (1.5, 2.5), "index 0 is the starting state");
    }
}

#[test]
fn a_trajectory_is_exactly_the_step_function_iterated() {
    for (x0, y0) in extreme_starts() {
        let traj = spectral_flow_n(x0, y0, 12).expect("a small step count is accepted");
        for i in 0..(traj.len() - 1) {
            let expected = spectral_flow_step(traj[i].0, traj[i].1);
            assert_bitwise_eq(
                traj[i + 1],
                expected,
                &format!("start ({x0}, {y0}), state {}", i + 1),
            );
        }
    }
}

#[test]
fn the_batch_flow_agrees_with_the_single_trajectory_for_every_start() {
    let starts = extreme_starts();
    let steps = 9;
    let batched =
        spectral_flow_batch(starts.clone(), steps).expect("a small step count is accepted");
    assert_eq!(batched.len(), starts.len(), "one trajectory per start");

    for (row, (x0, y0)) in starts.iter().enumerate() {
        let single = spectral_flow_n(*x0, *y0, steps).expect("a small step count is accepted");
        assert_eq!(batched[row].len(), single.len(), "row {row}: length");
        for (i, state) in single.iter().enumerate() {
            assert_bitwise_eq(batched[row][i], *state, &format!("row {row}, state {i}"));
        }
    }
}

#[test]
fn the_frame_shift_guard_keeps_every_evolved_state_strictly_positive_in_x() {
    for (x0, y0) in extreme_starts() {
        let traj = spectral_flow_n(x0, y0, 32).expect("a small step count is accepted");
        for (i, (x, _)) in traj.iter().enumerate().skip(1) {
            assert!(
                *x > 0.0,
                "start ({x0}, {y0}), state {i}: x was {x}, which would make the \
                 next step take ln of a non-positive number"
            );
        }
    }
}

#[test]
fn the_overflow_guard_keeps_a_finite_start_finite_for_every_step() {
    for (x0, y0) in extreme_starts() {
        let traj = spectral_flow_n(x0, y0, 64).expect("a small step count is accepted");
        for (i, (x, y)) in traj.iter().enumerate() {
            assert!(
                x.is_finite() && y.is_finite(),
                "start ({x0}, {y0}), state {i}: ({x}, {y}) left the finite range"
            );
        }
    }
}

#[test]
fn above_the_overflow_threshold_the_flow_substitutes_the_logarithm_of_x() {
    // Documented behaviour: for x > 709.78, exp(xv_safe(x)) is exp(ln(x)) = x,
    // which is what keeps the result finite where exp(x) would not be.
    let x = 1.0e5;
    let (out_x, out_y) = spectral_flow_step(x, 1.0);
    assert_eq!(out_x, 1.0, "x' is y_safe(y)");
    assert!(
        (out_y - x).abs() <= 1e-9 * x,
        "y' was {out_y}, expected exp(ln(x)) - ln(1) = {x}"
    );

    // Just below the threshold the unguarded exp is still finite, so the
    // result is enormous but not infinite -- the guard is not needed there.
    let (_, below) = spectral_flow_step(709.78, 1.0);
    assert!(below.is_finite(), "exp(709.78) is inside the double range");
    assert!(below > 1e300, "the unguarded branch really does run");
}

#[test]
fn a_non_finite_start_propagates_as_nan_instead_of_panicking() {
    // Documented sentinel: the flow does not refuse non-finite input, it
    // carries it through under IEEE-754 rules.
    let (x_out, y_out) = spectral_flow_step(f64::NAN, 1.0);
    assert_eq!(x_out, 1.0, "a NaN x does not disturb x' = y_safe(y)");
    assert!(y_out.is_nan(), "a NaN x reaches y' through exp");

    let (x_out, y_out) = spectral_flow_step(1.0, f64::NAN);
    assert!(
        x_out.is_nan() && y_out.is_nan(),
        "a NaN y reaches both outputs"
    );

    let traj = spectral_flow_n(f64::INFINITY, f64::NAN, 3).expect("step count is accepted");
    assert_eq!(
        traj.len(),
        4,
        "a non-finite start still produces a trajectory"
    );
}

#[test]
fn a_batch_with_no_starting_points_produces_no_trajectories() {
    // Documented sentinel: an empty batch is empty output, not an error.
    let batched = spectral_flow_batch(Vec::new(), 5).expect("an empty batch is accepted");
    assert!(batched.is_empty(), "no starts means no trajectories");
}

#[test]
fn the_step_bound_is_enforced_before_a_trajectory_is_allocated() {
    // usize::MAX would overflow the n_steps + 1 capacity computation, so the
    // bound has to be checked first; the refusal is the proof that it is.
    assert!(
        spectral_flow_n(1.0, 1.0, usize::MAX).is_err(),
        "an unbounded step count must be refused"
    );
    assert!(
        spectral_flow_n(1.0, 1.0, MAX_FLOW_STEPS + 1).is_err(),
        "one step past the bound must be refused"
    );
}
