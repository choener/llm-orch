// ── apikeys file operator tooling ───────────────────────────────────────────
//
// Backs `llm-orch keygen`, `hash-apikeys`, and `verify-apikey`. The
// orchestrator itself only ever *reads* the apikeys file — mutating it is an
// operator action, so the daemon never rewrites the credentials it
// authenticates against (read-only /etc, symlinked secrets, and bricked-auth
// risks are all operator-side problems, not runtime ones).

use crate::apikeys::{self, ApikeysError, ApikeysStore, HASH_PREFIX};
use std::path::{Path, PathBuf};

/// Errors from the file-writing tooling.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("failed to read apikeys file: {0}")]
    Io(#[from] std::io::Error),

    #[error("apikeys file is invalid: {0}")]
    Invalid(#[from] ApikeysError),

    #[error("{0} is a symlink; refusing to replace it (convert the file it points at)")]
    IsSymlink(PathBuf),

    #[error("{0} is not writable (mode is read-only for you; chmod 0600 to convert)")]
    NotWritable(PathBuf),
}

// ── token generation ────────────────────────────────────────────────────────

/// A freshly minted key: the plaintext is shown once and never stored.
#[derive(Debug, Clone)]
pub struct GeneratedKey {
    pub label: String,
    /// The bearer token the client must send.
    pub token: String,
    /// The `label: sha256:…` line to paste into the key file.
    pub line: String,
}

/// Generate `count` new keys. Tokens are 192 bits of CSPRNG entropy
/// (`uuid` v4 is backed by the OS RNG, which is already a dependency).
pub fn generate_keys(label: &str, count: usize) -> Vec<GeneratedKey> {
    (0..count.max(1))
        .map(|i| {
            // `--count 1` keeps the bare label; batches get a numeric suffix.
            let label = if count > 1 {
                format!("{label}-{}", i + 1)
            } else {
                label.to_owned()
            };
            let token = format!("sk-{}", random_hex(24));
            let line = format!("{label}: {}\n", apikeys::hash_value(&token));
            GeneratedKey { label, token, line }
        })
        .collect()
}

/// Lowercase hex from the OS CSPRNG (via UUIDv4), at least `bytes` worth.
fn random_hex(bytes: usize) -> String {
    let mut hex = String::new();
    while hex.len() < bytes * 2 {
        hex.push_str(&uuid::Uuid::new_v4().simple().to_string());
    }
    hex.truncate(bytes * 2);
    hex
}

// ── plaintext → sha256 conversion ──────────────────────────────────────────

/// What a conversion did, plus the resulting file text.
#[derive(Debug, Clone, Default)]
pub struct HashReport {
    /// Labels whose plaintext value was replaced by a digest.
    pub converted: Vec<String>,
    /// Entries that were already `sha256:` (left untouched).
    pub already_hashed: usize,
    /// The resulting file contents (comments, labels and alignment preserved).
    pub text: String,
}

impl HashReport {
    pub fn is_noop(&self) -> bool {
        self.converted.is_empty()
    }
}

/// Convert every plaintext entry in `contents` to its `sha256:` digest.
///
/// Idempotent (already-hashed lines are left alone) and format-preserving:
/// only the value span of a converted line changes, so comments, labels and
/// column alignment survive. The input is parsed and rejected *before*
/// anything is produced, so a malformed file never yields a partial rewrite.
pub fn hash_file_contents(contents: &str) -> Result<HashReport, ToolError> {
    // Validate first: every line must already be a legal entry, otherwise the
    // rewritten file would fail to load and auth would fail closed at restart.
    let store = ApikeysStore::parse(contents)?;

    let mut out = String::with_capacity(contents.len());
    let mut report = HashReport::default();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        // `parse` above guarantees a `:` separator and a non-empty value.
        let Some(colon) = line.find(':') else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let label = line[..colon].trim().to_owned();
        let value_part = &line[colon + 1..];
        let without_comment = apikeys::strip_inline_comment(value_part);
        let comment = &value_part[without_comment.len()..];
        let lead_len = without_comment.len() - without_comment.trim_start().len();
        let (lead, rest) = without_comment.split_at(lead_len);
        let value = rest.trim_end();
        let trail = &rest[value.len()..];

        if value.starts_with(HASH_PREFIX) {
            report.already_hashed += 1;
            out.push_str(line);
        } else {
            // Keep the original indentation; a bare `label:value` gets a space.
            let sep = if lead.is_empty() { " " } else { lead };
            out.push_str(&format!(
                "{}:{}{}{}{}",
                &line[..colon],
                sep,
                apikeys::hash_value(value),
                trail,
                comment
            ));
            report.converted.push(label);
        }
        out.push('\n');
    }

    report.text = out;
    debug_assert_eq!(
        store.len(),
        report.converted.len() + report.already_hashed,
        "converted + already-hashed must account for every entry"
    );
    Ok(report)
}

/// Rewrite the key file atomically (temp file in the same directory, then
/// rename) with `0600` permissions, refusing symlinks and read-only files.
pub fn write_key_file(path: &Path, text: &str) -> Result<(), ToolError> {
    // `symlink_metadata` does not follow the link: an atomic rename would
    // otherwise *replace* a symlink to a secret (e.g. into /run/secrets).
    let meta = std::fs::symlink_metadata(path)?;
    if meta.file_type().is_symlink() {
        return Err(ToolError::IsSymlink(path.to_path_buf()));
    }
    if meta.permissions().readonly() {
        return Err(ToolError::NotWritable(path.to_path_buf()));
    }

    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "apikeys.txt".to_owned());
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let tmp = parent.join(format!("{file_name}.tmp-{}", uuid::Uuid::new_v4()));

    let write = |tmp: &Path| -> Result<(), ToolError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(tmp)?;
            std::io::Write::write_all(&mut file, text.as_bytes())?;
            file.sync_all()?;
        }
        #[cfg(not(unix))]
        {
            std::fs::write(tmp, text)?;
        }
        Ok(())
    };

    match write(&tmp) {
        Ok(()) => match std::fs::rename(&tmp, path) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(ToolError::Io(e))
            }
        },
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

// ── verification ────────────────────────────────────────────────────────────

/// Read a token from stdin the way the client sends it: exactly the bytes
/// given, minus one trailing newline (so `echo "$TOKEN" | …` works while
/// `printf '%s'` stays byte-exact).
pub fn token_from_stdin(raw: &str) -> &str {
    raw.strip_suffix('\n').unwrap_or(raw)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_distinct_hashed_lines() {
        let keys = generate_keys("runner", 3);
        assert_eq!(keys.len(), 3);
        assert_eq!(
            keys.iter().map(|k| k.label.clone()).collect::<Vec<_>>(),
            vec!["runner-1", "runner-2", "runner-3"]
        );
        for k in &keys {
            assert!(k.token.starts_with("sk-"));
            assert_eq!(k.token.len(), 3 + 48);
            // The line's digest really is the digest of the printed token.
            assert_eq!(
                k.line.trim(),
                format!("{}: {}", k.label, apikeys::hash_value(&k.token))
            );
        }
    }

    #[test]
    fn single_key_keeps_the_bare_label() {
        let keys = generate_keys("ci", 1);
        assert_eq!(keys[0].label, "ci");
    }

    #[test]
    fn conversion_preserves_comments_and_layout() {
        let input = "\
# llm-orch API keys
# =================
personal-laptop:   sk-alice            # alice's laptop
work-desktop:      sk-bob
ci-runner:         sk-carol   # rotated 2026-11
";
        let report = hash_file_contents(input).unwrap();
        assert_eq!(
            report.converted,
            vec!["personal-laptop", "work-desktop", "ci-runner"]
        );
        assert_eq!(report.already_hashed, 0);

        // Round-trip: the converted file still loads, and the *original*
        // plaintext tokens still authenticate (same digests).
        let store = ApikeysStore::parse(&report.text).unwrap();
        assert_eq!(store.plaintext_len(), 0);
        assert_eq!(store.hashed_len(), 3);
        assert_eq!(store.authenticate("sk-alice"), Some("personal-laptop"));
        assert_eq!(store.authenticate("sk-bob"), Some("work-desktop"));
        assert_eq!(store.authenticate("sk-carol"), Some("ci-runner"));

        // Layout: comments and the value column survive.
        assert!(
            report
                .text
                .starts_with("# llm-orch API keys\n# =================\n")
        );
        assert!(report.text.contains("personal-laptop:   sha256:"));
        assert!(report.text.contains("# alice's laptop"));
        assert!(report.text.contains("ci-runner:         sha256:"));
        assert!(report.text.contains("# rotated 2026-11"));
    }

    #[test]
    fn conversion_is_idempotent() {
        let input = "a: sk-alice\nb: sk-bob\n";
        let once = hash_file_contents(input).unwrap();
        let twice = hash_file_contents(&once.text).unwrap();
        assert!(twice.is_noop());
        assert_eq!(twice.already_hashed, 2);
        assert_eq!(twice.text, once.text);
    }

    #[test]
    fn conversion_accepts_mixed_files() {
        let input = format!("a: sk-alice\nb: {}\n", apikeys::hash_value("sk-bob"));
        let report = hash_file_contents(&input).unwrap();
        assert_eq!(report.converted, vec!["a"]);
        assert_eq!(report.already_hashed, 1);
        let store = ApikeysStore::parse(&report.text).unwrap();
        assert_eq!(store.hashed_len(), 2);
        assert_eq!(store.authenticate("sk-bob"), Some("b"));
    }

    #[test]
    fn conversion_rejects_invalid_input_without_output() {
        // A malformed digest would produce a file that fails to load.
        let err = hash_file_contents("a: sha256:deadbeef\n").unwrap_err();
        assert!(matches!(err, ToolError::Invalid(_)), "{err:?}");
        assert!(hash_file_contents("no-colon-here\n").is_err());
    }

    #[test]
    fn bare_label_colon_value_gets_a_space() {
        let report = hash_file_contents("a:sk-x\n").unwrap();
        assert!(report.text.starts_with("a: sha256:"));
    }

    #[test]
    fn token_from_stdin_strips_one_newline() {
        assert_eq!(token_from_stdin("sk-x\n"), "sk-x");
        assert_eq!(token_from_stdin("sk-x"), "sk-x");
        // A token genuinely ending in whitespace keeps everything but the
        // newline — it must match what the client sends.
        assert_eq!(token_from_stdin("sk-x \n"), "sk-x ");
        assert_eq!(token_from_stdin("sk-x\n\n"), "sk-x\n");
    }

    #[test]
    fn write_refuses_symlinks_and_readonly_files() {
        let dir = std::env::temp_dir().join(format!("llm-orch-keytool-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let target = dir.join("keys.txt");
        std::fs::write(&target, "a: sk-x\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // 0400 — the mode the docs mandate at runtime — is owner-read-only.
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o400)).unwrap();
            assert!(matches!(
                write_key_file(&target, "a: sk-y\n"),
                Err(ToolError::NotWritable(_))
            ));
            // Untouched on disk.
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "a: sk-x\n");
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        write_key_file(&target, "b: sk-y\n").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "b: sk-y\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
                0o600,
                "rewritten key file must not be group/world readable"
            );
        }

        let link = dir.join("link.txt");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();
        #[cfg(unix)]
        assert!(matches!(
            write_key_file(&link, "c: sk-z\n"),
            Err(ToolError::IsSymlink(_))
        ));
        // The link still points at the untouched original.
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "b: sk-y\n");

        std::fs::remove_dir_all(&dir).ok();
    }
}
