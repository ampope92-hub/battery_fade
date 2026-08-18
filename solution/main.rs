use std::error::Error;
use std::io::Write;

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;

const CHEMS: [&str; 3] = ["lfp", "nca", "nmc"];

struct Pack {
    pack_id: String,
    cycles: f64,
    chemistry: String,
    avg_temp_c: f64,
    depth_of_discharge: f64,
    fast_charge_pct: f64,
    cell_count: f64,
    pack_age_days: f64,
    capacity_pct: f64,
}

fn read(path: &str) -> Result<Vec<Pack>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let at = |name: &str| headers.iter().position(|h| h == name);
    // train.csv carries the bench capacity and no pack_id; the field export is the reverse
    let i_id = at("pack_id");
    let i_cap = at("capacity_pct");
    let col = |name: &str| at(name).unwrap();
    let i_cyc = col("cycles");
    let i_chem = col("chemistry");
    let i_temp = col("avg_temp_c");
    let i_dod = col("depth_of_discharge");
    let i_fast = col("fast_charge_pct");
    let i_cells = col("cell_count");
    let i_age = col("pack_age_days");

    let mut out = Vec::new();
    for rec in rdr.records() {
        let r = rec?;
        let f = |i: usize| r[i].parse::<f64>().unwrap();
        out.push(Pack {
            pack_id: i_id.map(|i| r[i].to_string()).unwrap_or_default(),
            cycles: f(i_cyc),
            chemistry: r[i_chem].to_string(),
            avg_temp_c: f(i_temp),
            depth_of_discharge: f(i_dod),
            fast_charge_pct: f(i_fast),
            cell_count: f(i_cells),
            pack_age_days: f(i_age),
            capacity_pct: i_cap.map(f).unwrap_or(0.0),
        });
    }
    Ok(out)
}

fn design(packs: &[Pack]) -> Vec<Vec<f64>> {
    packs
        .iter()
        .map(|p| {
            let s = p.cycles.sqrt();
            let chem: Vec<f64> = CHEMS
                .iter()
                .map(|c| if *c == p.chemistry { 1.0 } else { 0.0 })
                .collect();
            let mut v = vec![
                s,
                p.avg_temp_c,
                p.depth_of_discharge,
                p.fast_charge_pct,
                p.cell_count,
                p.pack_age_days,
            ];
            v.extend(chem.iter().copied());
            // fade rate scales with chemistry and with how hard the pack was worked
            v.extend(chem.iter().map(|c| s * c));
            v.push(s * p.avg_temp_c);
            v.push(s * p.depth_of_discharge);
            v.push(s * p.fast_charge_pct);
            v
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let train = read("/app/data/train.csv")?;
    let field = read("/app/data/test_features.csv")?;

    let xtr: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&design(&train));
    let xte: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&design(&field));
    let y: Vec<f64> = train.iter().map(|p| p.capacity_pct).collect();

    let model = LinearRegression::fit(&xtr, &y, Default::default())?;
    let pred = model.predict(&xte)?;

    let mut out = std::fs::File::create("/app/predictions.csv")?;
    writeln!(out, "pack_id,capacity_pct")?;
    for (p, yhat) in field.iter().zip(&pred) {
        writeln!(out, "{},{:.3}", p.pack_id, yhat.clamp(0.0, 100.0))?;
    }
    println!("wrote /app/predictions.csv ({} packs)", field.len());
    Ok(())
}
