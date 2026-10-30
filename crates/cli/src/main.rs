use std::process::ExitCode;

use cofox_core::Config;

const USAGE: &str = "usage: cofox [--ecm DIGITS] <n>...

  --ecm DIGITS  run ECM up to the level for primes of this many digits
                (15 to 40, default 20); each level takes about four times
                as long as the one before";

fn main() -> ExitCode {
    let mut config = Config::default();
    let mut numbers = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ecm" => match args.next().and_then(|d| d.parse().ok()) {
                Some(digits) => config.ecm_digits = digits,
                None => {
                    eprintln!("cofox: --ecm needs a number of digits\n\n{USAGE}");
                    return ExitCode::FAILURE;
                }
            },
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => numbers.push(arg),
        }
    }
    if numbers.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }

    let mut status = ExitCode::SUCCESS;
    let mut unsplit = false;
    for arg in &numbers {
        match cofox_core::factor_str_with(arg, &config) {
            Ok(factors) => {
                unsplit |= factors.iter().any(|f| !f.is_prime);
                let factors: Vec<String> = factors.iter().map(ToString::to_string).collect();
                println!("{arg}: {}", factors.join(" "));
            }
            Err(e) => {
                eprintln!("cofox: invalid number {arg:?}: {e}");
                status = ExitCode::FAILURE;
            }
        }
    }
    if unsplit {
        eprintln!(
            "cofox: factors in parentheses are composite but could not be split \
             (--ecm {} would search for larger factors)",
            config.ecm_digits + 5
        );
    }
    status
}
