use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: cofox <n>...");
        return ExitCode::FAILURE;
    }

    let mut status = ExitCode::SUCCESS;
    let mut unsplit = false;
    for arg in &args {
        match cofox_core::factor_str(arg) {
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
        eprintln!("cofox: factors in parentheses are composite but could not be split");
    }
    status
}
