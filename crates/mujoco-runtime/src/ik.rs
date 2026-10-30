//! Diagnostic DLS IK using the runtime's side-effect-free FK. No policy/target-q access.
use crate::Simulator;
use robotics_core::{Error, Result};
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for k in 0..3 {
        let pivot = (k..3)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        a.swap(k, pivot);
        b.swap(k, pivot);
        if a[k][k].abs() < 1e-15 {
            return [0.; 3];
        }
        let d = a[k][k];
        for value in &mut a[k][k..] {
            *value /= d;
        }
        let pivot_row = a[k];
        b[k] /= d;
        for i in 0..3 {
            if i != k {
                let d = a[i][k];
                for (value, pivot) in a[i][k..].iter_mut().zip(&pivot_row[k..]) {
                    *value -= d * pivot;
                }
                b[i] -= d * b[k];
            }
        }
    }
    b
}
impl Simulator {
    pub fn dls_delta(
        &self,
        position: &[f64],
        target: [f64; 3],
        indices: &[usize],
        damping: f64,
        max_delta: f64,
    ) -> Result<Vec<f64>> {
        if !target.iter().all(|v| v.is_finite())
            || !damping.is_finite()
            || damping <= 0.
            || !max_delta.is_finite()
            || max_delta <= 0.
            || position.iter().any(|v| !v.is_finite())
            || indices.is_empty()
            || indices
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != indices.len()
            || indices.iter().any(|&i| i >= position.len())
        {
            return Err(Error::Input("invalid IK input".into()));
        }
        let ranges = self.action_bounds();
        let mut q = position.to_vec();
        if q.len() != ranges.len() {
            return Err(Error::Input("IK dimension mismatch".into()));
        }
        for (v, [lo, hi]) in q.iter_mut().zip(&ranges) {
            *v = v.clamp(*lo, *hi);
        }
        let site = self.default_end_effector()?;
        let current = self.forward_kinematics(&q, site)?;
        let mut jac = vec![];
        for &i in indices {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[i] = (q[i] + 1e-5).min(ranges[i][1]);
            minus[i] = (q[i] - 1e-5).max(ranges[i][0]);
            let a = self.forward_kinematics(&plus, site)?;
            let b = self.forward_kinematics(&minus, site)?;
            let h = plus[i] - minus[i];
            jac.push(std::array::from_fn::<_, 3, _>(|k| (a[k] - b[k]) / h));
        }
        let mut matrix = [[0.; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                matrix[r][c] = jac.iter().map(|j| j[r] * j[c]).sum::<f64>()
                    + if r == c { damping * damping } else { 0. };
            }
        }
        let y = solve3(matrix, std::array::from_fn(|k| target[k] - current[k]));
        let mut result = vec![0.; q.len()];
        for (&i, j) in indices.iter().zip(jac) {
            result[i] = j
                .iter()
                .zip(y)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-max_delta, max_delta);
        }
        Ok(result)
    }
}
