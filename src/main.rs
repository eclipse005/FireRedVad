use anyhow::Result;
use std::process::Command;

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() == 1 {
        eprintln!("FireRedVAD CLI");
        eprintln!("This program should be run from a command line.");
        eprintln!("Example:");
        eprintln!(r#"  fireredvad.exe D:\path\to\audio.wav"#);
        eprintln!();
        #[cfg(target_os = "windows")]
        {
            let _ = Command::new("cmd").args(["/C", "pause"]).status();
        }
        return Ok(());
    }
    fireredvad::cli::run()
}
