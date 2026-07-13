use clap::Parser;

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
        eprintln!("llm-orch: checking config: {}", check_path.display());
        // TODO: load & validate config (task §2).
        eprintln!("llm-orch: config validation not yet implemented");
        std::process::exit(1);
    }

    eprintln!("llm-orch: starting with config: {}", cli.config.display());
    // TODO: load config, start watchers, run HTTP server.
    println!("llm-orch starting...");
}