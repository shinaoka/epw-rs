//! `epw-rs`: phonon linewidths, lambda and alpha^2F from EPW Wannier files.
//!
//! Usage: `epw-rs <wannier-dir> [output-dir] [--ranks N]`
//!
//! `<wannier-dir>` holds `crystal.fmt`, `epwdata.fmt`, `wigner.fmt` and
//! `<prefix>.epmatwp`. The settings are those of QE `test-suite/epw_metal`.
//! `--ranks N` emulates N ranks with threads (results are identical).

use eph_comm::{SelfComm, ThreadComm};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            let mut source = e.source();
            while let Some(cause) = source {
                eprintln!("  caused by: {cause}");
                source = cause.source();
            }
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional = Vec::new();
    let mut ranks = 1_usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--ranks" {
            ranks = args.next().ok_or("--ranks needs a value")?.parse()?;
        } else {
            positional.push(PathBuf::from(arg));
        }
    }
    let dir = positional
        .first()
        .ok_or("usage: epw-rs <wannier-dir> [output-dir] [--ranks N]")?;
    let out_dir = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| PathBuf::from("."));

    let model = eph_io_epw::read_wannier_model(dir)?;
    let epmatwp = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "epmatwp"))
        .ok_or("no *.epmatwp file in the Wannier directory")?;
    let vertex = eph_io_epw::EpmatwpFile::open(&epmatwp, eph_io_epw::eph_dims(&model))?;
    let config = eph_driver::Config::epw_metal_test();

    let start = std::time::Instant::now();
    let results = if ranks == 1 {
        eph_driver::run(&model, &vertex, &config, &SelfComm)?
    } else {
        eph_driver::run(&model, &vertex, &config, &ThreadComm::new(ranks)?)?
    };
    let elapsed = start.elapsed();

    std::fs::create_dir_all(&out_dir)?;
    std::fs::write(
        out_dir.join("linewidths.dat"),
        eph_driver::output::linewidth_table(&results),
    )?;
    std::fs::write(
        out_dir.join("a2f.dat"),
        eph_driver::output::a2f_table(&results),
    )?;
    print!("{}", eph_driver::output::summary(&results));
    println!(
        "Wall time                   : {:12.3} s ({ranks} rank(s))",
        elapsed.as_secs_f64()
    );
    Ok(())
}
