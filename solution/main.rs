use std::error::Error;
use std::io::Write;

const THRESHOLD: f64 = 80.0;
const DROP: f64 = 100.0 - THRESHOLD;
// Standard normal 90th percentile. Scheduling off the upper end of the pack's own fade-rate
// distribution rather than its middle is what keeps roughly one pack in ten from crossing
// before it is pulled.
const Z90: f64 = 1.281_551_565_544_600_4;

struct Pack {
    pack_id: String,
    cycles: f64,
    chemistry: String,
    avg_temp_c: f64,
    depth_of_discharge: f64,
    fast_charge_pct: f64,
    capacity_pct: f64,
}

fn read(path: &str) -> Result<Vec<Pack>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let at = |name: &str| headers.iter().position(|h| h == name);
    let i_cap = at("capacity_pct"); // teardown cohort only
    let col = |name: &str| at(name).unwrap();
    let (i_id, i_cyc, i_chem) = (col("pack_id"), col("cycles"), col("chemistry"));
    let i_temp = col("avg_temp_c");
    let i_dod = col("depth_of_discharge");
    let i_fast = col("fast_charge_pct");

    let mut out = Vec::new();
    for rec in rdr.records() {
        let r = rec?;
        let f = |i: usize| r[i].parse::<f64>().unwrap();
        out.push(Pack {
            pack_id: r[i_id].to_string(),
            cycles: f(i_cyc),
            chemistry: r[i_chem].to_string(),
            avg_temp_c: f(i_temp),
            depth_of_discharge: f(i_dod),
            fast_charge_pct: f(i_fast),
            capacity_pct: i_cap.map(f).unwrap_or(0.0),
        });
    }
    Ok(out)
}

/// Drivers of the log fade rate. Capacity follows 100 - rate*sqrt(cycles), and the stress
/// terms multiply the rate rather than shifting it, so on a log scale they are linear.
fn design(p: &Pack) -> [f64; 6] {
    [
        1.0,
        p.avg_temp_c - 28.0,
        p.depth_of_discharge - 0.65,
        p.fast_charge_pct,
        if p.chemistry == "lfp" { 1.0 } else { 0.0 },
        if p.chemistry == "nca" { 1.0 } else { 0.0 },
    ]
}

/// Drivers of how much pack-to-pack SPREAD there is around that rate. Hard duty does not
/// only fade a pack faster, it makes the fade less predictable.
fn duty(p: &Pack) -> [f64; 3] {
    [
        1.0,
        p.fast_charge_pct,
        (p.avg_temp_c - 28.0).max(0.0) / 10.0,
    ]
}

/// Gaussian elimination with partial pivoting; the systems here are 6x6 and 3x3.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let piv = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs())).unwrap();
        a.swap(k, piv);
        b.swap(k, piv);
        for i in k + 1..n {
            let m = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= m * a[k][j];
            }
            b[i] -= m * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    x
}

/// Ordinary least squares through the normal equations.
fn ols(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let k = x[0].len();
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for (row, &yi) in x.iter().zip(y) {
        for a in 0..k {
            xty[a] += row[a] * yi;
            for b in 0..k {
                xtx[a][b] += row[a] * row[b];
            }
        }
    }
    solve(xtx, xty)
}

/// Maximum likelihood for log(sigma) = z.g under resid ~ N(0, sigma_i^2), by Newton.
///
/// Regressing log|resid| on duty is the obvious first pass and it biases the gentle end:
/// a teardown rate estimate is (100-capacity)/sqrt(cycles), so bench measurement error is
/// divided by a small number at low cycles and inflates the apparent spread exactly where
/// the real spread is smallest. Using the whole residual instead of its log-magnitude is
/// far better behaved. Objective: sum(2*z.g + resid^2 * exp(-2*z.g)).
fn fit_log_sigma(z: &[[f64; 3]], resid: &[f64]) -> [f64; 3] {
    let n = resid.len() as f64;
    let mean = resid.iter().sum::<f64>() / n;
    let sd = (resid.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / n).sqrt();
    let mut g = [sd.ln(), 0.0, 0.0];

    for _ in 0..200 {
        let mut grad = [0.0; 3];
        let mut hess = vec![vec![0.0; 3]; 3];
        for (zi, &r) in z.iter().zip(resid) {
            let zg: f64 = (0..3).map(|a| zi[a] * g[a]).sum();
            let w = r * r * (-2.0 * zg).exp();
            for a in 0..3 {
                grad[a] += 2.0 * zi[a] * (1.0 - w);
                for b in 0..3 {
                    hess[a][b] += 4.0 * zi[a] * w * zi[b];
                }
            }
        }
        for (a, row) in hess.iter_mut().enumerate() {
            row[a] += 1e-9;
        }
        let step = solve(hess, grad.to_vec());
        for a in 0..3 {
            g[a] -= step[a];
        }
        if step.iter().all(|s| s.abs() < 1e-11) {
            break;
        }
    }
    g
}

fn main() -> Result<(), Box<dyn Error>> {
    let train = read("/app/data/train.csv")?;
    let field = read("/app/data/test_features.csv")?;

    // For each teardown pack the bench capacity gives one observation of its own fade rate.
    let x: Vec<Vec<f64>> = train.iter().map(|p| design(p).to_vec()).collect();
    let y: Vec<f64> = train
        .iter()
        .map(|p| (((100.0 - p.capacity_pct) / p.cycles.sqrt()).max(1e-6)).ln())
        .collect();
    let beta = ols(&x, &y);

    let resid: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(row, &yi)| yi - row.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let z: Vec<[f64; 3]> = train.iter().map(duty).collect();
    let g = fit_log_sigma(&z, &resid);

    let mut out = std::fs::File::create("/app/predictions.csv")?;
    writeln!(out, "pack_id,capacity_pct,replace_cycle")?;
    for p in &field {
        let log_rate: f64 = design(p).iter().zip(&beta).map(|(a, b)| a * b).sum();
        let base = log_rate.exp();
        let capacity = 100.0 - base * p.cycles.sqrt();

        // Schedule off a pessimistic fade rate for this pack's duty, not the median one:
        // the spread is what decides how much margin a pack needs, and it is not the same
        // for a gently used pack as for a hot, fast-charged one.
        let sigma: f64 = duty(p).iter().zip(&g).map(|(a, b)| a * b).sum::<f64>().exp();
        let fast_rate = base * (Z90 * sigma).exp();
        let replace_cycle = (DROP / fast_rate).powi(2);

        writeln!(
            out,
            "{},{:.3},{:.1}",
            p.pack_id,
            capacity.clamp(0.0, 100.0),
            replace_cycle
        )?;
    }
    println!("wrote /app/predictions.csv ({} packs)", field.len());
    Ok(())
}
