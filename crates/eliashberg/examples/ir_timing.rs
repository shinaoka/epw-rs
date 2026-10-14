//! Timing of the IR setup (SVE, bases) versus the convolutions.
use eliashberg::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for (lambda, eps) in [(1e4, 1e-10), (1e5, 1e-6), (1e5, 1e-10)] {
        let t0 = std::time::Instant::now();
        let set = IrBasisSet::new(lambda, eps)?;
        let t_sve = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut sum = IrSum::new(&set, 0.0013, &SamplingPoints::default(), |nu| {
            Ok(nu.iter().map(|v| 1.0 / (1.0 + v * v)).collect())
        })?;
        let t_basis = t1.elapsed().as_secs_f64();
        let n = sum.indices().len();
        let f = tenferro_tensor::TypedTensor::from_vec_col_major(vec![n, 3], vec![1e-3; 3 * n])?;
        let t2 = std::time::Instant::now();
        for _ in 0..100 {
            sum.convolve(&f, Parity::Even)?;
        }
        let t_conv = t2.elapsed().as_secs_f64() / 100.0;
        println!("Lambda {lambda:.0e} eps {eps:.0e}: nfreq {n}, SVE {t_sve:.2} s, bases+sampling {t_basis:.2} s, convolve {:.3} ms", t_conv * 1e3);
    }
    Ok(())
}
