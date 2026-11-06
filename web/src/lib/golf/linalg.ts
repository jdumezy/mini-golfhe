/**
 * Small dense matrices, enough for the 5×5 augmented flow matrices of the
 * physics (state x, y, vx, vy plus a constant 1).
 */

/** A square matrix, row-major. */
export type Mat = Float64Array;

export function identity(n: number): Mat {
  const m = new Float64Array(n * n);
  for (let i = 0; i < n; i++) m[i * n + i] = 1;
  return m;
}

export function mul(a: Mat, b: Mat, n: number): Mat {
  const c = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++) {
      const aik = a[i * n + k];
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) c[i * n + j] += aik * b[k * n + j];
    }
  return c;
}

/** `a · v` for a vector `v` of length `n`. */
export function apply(a: Mat, v: ArrayLike<number>, n: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += a[i * n + j] * v[j];
    out[i] = s;
  }
  return out;
}

/** Matrix exponential by scaling and squaring of a Taylor series. */
export function expm(a: Mat, n: number): Mat {
  let norm = 0;
  for (let i = 0; i < n; i++) {
    let row = 0;
    for (let j = 0; j < n; j++) row += Math.abs(a[i * n + j]);
    norm = Math.max(norm, row);
  }
  const squarings = Math.max(0, Math.ceil(Math.log2(norm / 0.25)));
  const scale = 2 ** -squarings;
  const x = a.map((v) => v * scale);
  let term = identity(n);
  let sum = identity(n);
  for (let k = 1; k <= 18; k++) {
    term = mul(term, x, n).map((v) => v / k);
    for (let i = 0; i < n * n; i++) sum[i] += term[i];
  }
  for (let s = 0; s < squarings; s++) sum = mul(sum, sum, n);
  return sum;
}
