use clap::Parser;

mod config;

/// LLM Orch — single-host LLM orchestrator.
#[derive(Parser, Debug)]
#[command(name = "llm-orch", version, about)]
struct Cli {
    /// Path to the main configuration file.
    #[arg(
        short = 'c',
        long = "config",
        default_value = "config.yaml",
        value_hint = clap::ValueHint::FilePath
    )]
    config: std::path::PathBuf,

    /// Validate the configuration file and exit without starting the server.
    /// Exits 0 on valid config, non-zero on errors.
    #[arg(long = "check-config", value_name = "PATH", value_hint = clap::ValueHint::FilePath)]
    check_config: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // §2 — check-config mode: validate and exit.
    if let Some(check_path) = cli.check_config {
        match config::Config::load(&check_path) {
            Ok(cfg) => {
                // TODO: full validation (duplicate names, alias targets, etc.)
                eprintln!(
                    "OK: {} model(s), {} alias(es)",
                    cfg.models.len(),
                    cfg.aliases.len()
                );
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("ERROR: {}", e);
                std::process::exit(1);
            }
        }
    }

    eprintln!("llm-orch: starting with config: {}", cli.config.display());
    // TODO: load config, start watchers, run HTTP server.
    println!("llm-orch starting...");
}