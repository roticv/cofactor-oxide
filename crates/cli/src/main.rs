use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: cofox <n>...");
        return ExitCode::FAILURE;
    }

    let mut status = ExitCode::SUCCESS;
    for arg in &args {
        match arg.parse::<u64>() {
            Ok(n) => {
                let factors: Vec<String> =
                    cofox_core::factor(n).iter().map(u64::to_string).collect();
                println!("{n}: {}", factors.join(" "));
            }
            Err(e) => {
                eprintln!("cofox: invalid number {arg:?}: {e}");
                status = ExitCode::FAILURE;
            }
        }
    }
    status
}
