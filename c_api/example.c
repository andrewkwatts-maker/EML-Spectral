/* example.c — minimal demo of the eml-spectral C API.
 *
 * Build (Linux/macOS):
 *   gcc example.c -L../target/release -leml_spectral -lm -o example
 *
 * Build (Windows, MSVC):
 *   cl /I . example.c /link ..\target\release\eml_spectral.dll.lib
 */
#include <stdio.h>
#include "eml_spectral.h"

int main(void) {
    /* Spectral flow: one step. */
    double x, y;
    els_spectral_flow_step(1.0, 1.0, &x, &y);
    printf("Phi(1.0, 1.0) = (%g, %g)\n", x, y);

    /* Lorentz boost: round-trip. */
    double bx, by;
    els_boost(1.0, 2.0, 0.5, 1.0, &bx, &by);
    printf("boost((1, 2), phi=0.5) = (%g, %g)\n", bx, by);
    printf("Minkowski delta = %g\n", els_minkowski_delta(1.0, 2.0, 1, 1.0));

    /* Octonion: e1 * e2 = e4 (Fano). */
    double a[8] = {0, 1, 0, 0, 0, 0, 0, 0};
    double b[8] = {0, 0, 1, 0, 0, 0, 0, 0};
    double out[8];
    els_octonion_mul(a, b, out);
    printf("e1 * e2 = ");
    for (int i = 0; i < 8; i++) printf("%g ", out[i]);
    printf("\n");

    return 0;
}
