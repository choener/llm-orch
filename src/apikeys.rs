use sha2::{Digest as _, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// Key forms
// ---------------------------------------------------------------------------

/// Prefix identifying a SHA-256 digest entry: `label: sha256:<64 hex>`.
///
/// The slug names the *algorithm*, not the concept, so a future KDF build can
/// introduce `$argon2id$…` lines without a format break.
pub const HASH_PREFIX: &str = "sha256:";

/// Modular-crypt-format style prefixes that are *reserved*: recognized and
/// rejected with a clear message rather than being silently treated as a
/// plaintext token (which would match nothing and stay invisible).
pub const RESERVED_PREFIXES: &[&str] = &["$argon2id$", "$argon2i$", "$argon2d$"];

/// SHA-256 of the **exact bytes** of a bearer token, lowercase hex.
///
/// No trimming, case folding, or unicode normalization: the digest covers
/// precisely what the client sends after the `Bearer ` prefix. That is the
/// single most common source of never-matching hashed keys (`echo key |
/// sha256sum` hashes a trailing newline), which is why the tooling in
/// [`crate::apikeys`] exists instead of hand-hashing.
pub fn sha256_hex(token: &str) -> String {
    let digest = Sha256::digest(token.as_bytes());
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(char::from_digit((byte >> 4) as u32, 16).unwrap());
        out.push(char::from_digit((byte & 0x0f) as u32, 16).unwrap());
    }
    out
}

/// The value form for a token, ready to write into the key file:
/// `sha256:<hex>`.
pub fn hash_value(token: &str) -> String {
    format!("{HASH_PREFIX}{}", sha256_hex(token))
}

fn sha256_bytes(token: &str) -> [u8; 32] {
    let digest = Sha256::digest(token.as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Decode exactly 64 hex characters (any case) into a digest.
fn decode_hex_32(s: &str) -> Option<[u8; 32]> {
    let bytes = s.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in bytes.chunks(2).enumerate() {
        let hi = (chunk[0] as char).to_digit(16)?;
        let lo = (chunk[1] as char).to_digit(16)?;
        out[i] = (hi << 4 | lo) as u8;
    }
    Some(out)
}

/// Strip a trailing inline comment from a value: the first `#` **preceded by
/// whitespace** ends the value, which is then trimmed.
///
/// Requiring the preceding whitespace keeps tokens that contain `#` (e.g.
/// `sk-foo#bar`) working — and is strictly safe, because a bearer token can
/// never contain whitespace (RFC 6750 `b64token`).
pub(crate) fn strip_inline_comment(value: &str) -> &str {
    let bytes = value.as_bytes();
    for i in 1..bytes.len() {
        if bytes[i] == b'#' && matches!(bytes[i - 1], b' ' | b'\t') {
            return &value[..i - 1];
        }
    }
    value
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when loading or validating the API keys file.
#[derive(Debug, thiserror::Error)]
pub enum ApikeysError {
    #[error("failed to read apikeys file: {0}")]
    Io(#[from] std::io::Error),

    #[error("duplicate label at line {line}: `{label}`")]
    DuplicateLabel { label: String, line: usize },

    #[error("duplicate key at line {line} (label `{label}`)")]
    DuplicateKey { label: String, line: usize },

    #[error("missing `:` separator at line {line}")]
    MissingColon { line: usize },

    #[error("empty label at line {line}")]
    EmptyLabel { line: usize },

    #[error("empty key at line {line} (label `{label}`)")]
    EmptyKey { label: String, line: usize },

    #[error("invalid hashed key at line {line} (label `{label}`): {reason}")]
    BadHash {
        label: String,
        line: usize,
        reason: String,
    },

    #[error(
        "unsupported key format at line {line} (label `{label}`): `{prefix}` keys are reserved \
         for password-KDF digests, which this build does not support"
    )]
    UnsupportedFormat {
        label: String,
        line: usize,
        prefix: String,
    },

    #[error(
        "plaintext key at line {line} (label `{label}`) refused: this instance only accepts \
         hashed (`{HASH_PREFIX}`) keys"
    )]
    PlaintextKey { label: String, line: usize },
}

// ---------------------------------------------------------------------------
// Load options
// ---------------------------------------------------------------------------

/// Options for loading the apikeys file.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApikeysOptions {
    /// Reject plaintext entries; only `sha256:` digests are accepted.
    ///
    /// Lives on the command line (not in `config.yaml`) for the same reason
    /// the key file's *path* does: whoever may edit the config must not be able
    /// to relax an auth-integrity guarantee.
    pub require_hashed: bool,
}

// ---------------------------------------------------------------------------
// ApiKeys store
// ---------------------------------------------------------------------------

/// A parsed, ready-to-query set of API keys.
///
/// Two indexes, both O(1): plaintext tokens as given, and SHA-256 digests.
/// `authenticate` looks the presented token up directly and then by digest,
/// so authenticating never scans the file — which matters because the key file
/// is runtime-managed and expected to hold many entries.
#[derive(Debug, Clone)]
pub struct ApikeysStore {
    /// plaintext token → label
    plain: HashMap<String, String>,
    /// sha256(token) → label
    hashed: HashMap<[u8; 32], String>,
}

impl ApikeysStore {
    /// Load and parse an API keys file.
    pub fn load(path: &Path) -> Result<Self, ApikeysError> {
        Self::load_with(path, ApikeysOptions::default())
    }

    /// Load and parse an API keys file with explicit options.
    pub fn load_with(path: &Path, opts: ApikeysOptions) -> Result<Self, ApikeysError> {
        let contents = std::fs::read_to_string(path)?;
        Self::parse_with(&contents, opts)
    }

    /// Parse API keys from a string (useful for testing and tooling).
    pub fn parse(input: &str) -> Result<Self, ApikeysError> {
        Self::parse_with(input, ApikeysOptions::default())
    }

    /// Parse API keys from a string with explicit options.
    pub fn parse_with(input: &str, opts: ApikeysOptions) -> Result<Self, ApikeysError> {
        let mut plain = HashMap::new();
        let mut hashed = HashMap::new();
        let mut seen_labels = HashSet::new();

        for (idx, line) in input.lines().enumerate() {
            let line_num = idx + 1;
            let trimmed = line.trim();

            // Skip blank lines and whole-line comments.
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Split on the first `:` only.
            let (label, value) = trimmed
                .split_once(':')
                .ok_or(ApikeysError::MissingColon { line: line_num })?;

            let label = label.trim().to_owned();
            // An inline ` # …` suffix is a comment, not part of the key.
            let value = strip_inline_comment(value).trim().to_owned();

            if label.is_empty() {
                return Err(ApikeysError::EmptyLabel { line: line_num });
            }
            if value.is_empty() {
                return Err(ApikeysError::EmptyKey {
                    label,
                    line: line_num,
                });
            }
            if !seen_labels.insert(label.clone()) {
                return Err(ApikeysError::DuplicateLabel {
                    label,
                    line: line_num,
                });
            }

            if let Some(prefix) = RESERVED_PREFIXES.iter().find(|p| value.starts_with(**p)) {
                return Err(ApikeysError::UnsupportedFormat {
                    label,
                    line: line_num,
                    prefix: (*prefix).to_owned(),
                });
            }

            if let Some(hex) = value.strip_prefix(HASH_PREFIX) {
                let digest = decode_hex_32(hex).ok_or_else(|| {
                    let reason = if hex.is_empty() {
                        "empty digest".to_owned()
                    } else if hex.len() != 64 {
                        format!("expected 64 hex characters, got {}", hex.len())
                    } else {
                        "not hexadecimal".to_owned()
                    };
                    ApikeysError::BadHash {
                        label: label.clone(),
                        line: line_num,
                        reason,
                    }
                })?;
                if hashed.contains_key(&digest) || plain.contains_key(&value) {
                    return Err(ApikeysError::DuplicateKey {
                        label,
                        line: line_num,
                    });
                }
                hashed.insert(digest, label);
            } else {
                if opts.require_hashed {
                    return Err(ApikeysError::PlaintextKey {
                        label,
                        line: line_num,
                    });
                }
                if plain.contains_key(&value) {
                    return Err(ApikeysError::DuplicateKey {
                        label,
                        line: line_num,
                    });
                }
                plain.insert(value, label);
            }
        }

        Ok(Self { plain, hashed })
    }

    /// Returns the label for a given key, or `None` if the key is unknown.
    ///
    /// Direct hit on a plaintext entry first, then a digest match; both are
    /// single hash-map lookups.
    pub fn authenticate(&self, key: &str) -> Option<&str> {
        if let Some(label) = self.plain.get(key) {
            return Some(label.as_str());
        }
        self.hashed.get(&sha256_bytes(key)).map(|s| s.as_str())
    }

    /// Returns `true` when no keys are configured (fail-closed auth).
    pub fn is_empty(&self) -> bool {
        self.plain.is_empty() && self.hashed.is_empty()
    }

    /// Number of configured keys (plaintext + hashed).
    pub fn len(&self) -> usize {
        self.plain.len() + self.hashed.len()
    }

    /// Number of `sha256:` digest entries.
    pub fn hashed_len(&self) -> usize {
        self.hashed.len()
    }

    /// Number of plaintext (non-digest) entries.
    pub fn plaintext_len(&self) -> usize {
        self.plain.len()
    }

    /// Create an empty store (fail-closed — all requests denied).
    pub fn empty() -> Self {
        Self {
            plain: HashMap::new(),
            hashed: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid() {
        let input = "\
# comment
personal-laptop: sk-alice
work-desktop: sk-bob
";
        let store = ApikeysStore::parse(input).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.authenticate("sk-alice"), Some("personal-laptop"));
        assert_eq!(store.authenticate("sk-bob"), Some("work-desktop"));
        assert_eq!(store.authenticate("sk-nope"), None);
    }

    #[test]
    fn blank_lines_and_comments() {
        let input = "\n\n# header\n\npersonal-laptop: sk-alice\n";
        let store = ApikeysStore::parse(input).unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn missing_colon_is_error() {
        let input = "personal-laptop sk-alice\n";
        let err = ApikeysStore::parse(input).unwrap_err();
        assert!(matches!(err, ApikeysError::MissingColon { line: 1 }));
    }

    #[test]
    fn empty_label_is_error() {
        let input = ": sk-alice\n";
        let err = ApikeysStore::parse(input).unwrap_err();
        assert!(matches!(err, ApikeysError::EmptyLabel { line: 1 }));
    }

    #[test]
    fn empty_key_is_error() {
        let input = "personal-laptop: \n";
        let err = ApikeysStore::parse(input).unwrap_err();
        assert!(matches!(err, ApikeysError::EmptyKey { .. }));
    }

    #[test]
    fn duplicate_label_is_error() {
        let input = "\
personal-laptop: sk-alice
personal-laptop: sk-bob
";
        let err = ApikeysStore::parse(input).unwrap_err();
        assert!(matches!(err, ApikeysError::DuplicateLabel { line: 2, .. }));
    }

    #[test]
    fn duplicate_key_is_error() {
        let input = "\
personal-laptop: sk-shared
work-desktop: sk-shared
";
        let err = ApikeysStore::parse(input).unwrap_err();
        assert!(matches!(err, ApikeysError::DuplicateKey { line: 2, .. }));
    }

    #[test]
    fn is_empty_when_no_keys() {
        let input = "# just a comment\n\n";
        let store = ApikeysStore::parse(input).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.authenticate("anything"), None);
    }

    // ── hashed keys ─────────────────────────────────────────────────────

    #[test]
    fn sha256_hex_matches_known_vectors() {
        // Guards the hand-rolled hex encoding.
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hash_value("abc"),
            format!(
                "{HASH_PREFIX}ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            )
        );
    }

    #[test]
    fn hashed_key_authenticates() {
        let input = format!("work-desktop: {}\n", hash_value("sk-bob"));
        let store = ApikeysStore::parse(&input).unwrap();
        assert_eq!(store.authenticate("sk-bob"), Some("work-desktop"));
        assert_eq!(store.authenticate("sk-bob "), None);
        assert_eq!(store.authenticate("SK-BOB"), None);
        assert_eq!(store.len(), 1);
        assert_eq!(store.hashed_len(), 1);
        assert_eq!(store.plaintext_len(), 0);
    }

    #[test]
    fn hash_covers_exact_bytes() {
        // A trailing newline or space is part of the token — the classic
        // `echo key | sha256sum` mismatch must fail closed, not match.
        let input = format!("work-desktop: {}\n", hash_value("sk-bob"));
        let store = ApikeysStore::parse(&input).unwrap();
        assert_eq!(store.authenticate("sk-bob\n"), None);
        assert_eq!(store.authenticate("sk-bob"), Some("work-desktop"));
    }

    #[test]
    fn mixed_plaintext_and_hashed_file() {
        let input = format!(
            "\
personal-laptop: sk-alice
work-desktop: {}
",
            hash_value("sk-bob")
        );
        let store = ApikeysStore::parse(&input).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.plaintext_len(), 1);
        assert_eq!(store.hashed_len(), 1);
        assert_eq!(store.authenticate("sk-alice"), Some("personal-laptop"));
        assert_eq!(store.authenticate("sk-bob"), Some("work-desktop"));
        assert_eq!(store.authenticate("sk-carol"), None);
    }

    #[test]
    fn hex_is_case_insensitive_but_canonical_on_lookup() {
        let upper = format!("a: {}{}\n", HASH_PREFIX, sha256_hex("sk-x").to_uppercase());
        let lower = format!("b: {}\n", hash_value("sk-x"));
        // Same digest in two spellings = the same key.
        let err = ApikeysStore::parse(&format!("{upper}{lower}")).unwrap_err();
        assert!(matches!(err, ApikeysError::DuplicateKey { line: 2, .. }));
        // And either spelling authenticates the token.
        assert_eq!(
            ApikeysStore::parse(&upper).unwrap().authenticate("sk-x"),
            Some("a")
        );
        assert_eq!(
            ApikeysStore::parse(&lower).unwrap().authenticate("sk-x"),
            Some("b")
        );
    }

    #[test]
    fn malformed_digests_are_errors_with_line_numbers() {
        let cases = [
            ("a: sha256:\n", "empty digest"),
            ("a: sha256:abc\n", "expected 64 hex characters, got 3"),
            (
                &format!("a: {}\nb: sha256:{}Z\n", hash_value("k"), "0".repeat(63)),
                "not hexadecimal",
            ),
        ];
        for (input, needle) in cases {
            let err = ApikeysStore::parse(input).unwrap_err();
            match &err {
                ApikeysError::BadHash { reason, .. } => {
                    assert!(reason.contains(needle), "{reason} must mention {needle}")
                }
                other => panic!("expected BadHash for {input:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn reserved_kdf_prefixes_are_hard_errors() {
        for prefix in RESERVED_PREFIXES {
            let input = format!("research: {prefix}v=19$m=65536,t=3,p=4$abc\n");
            let err = ApikeysStore::parse(&input).unwrap_err();
            match &err {
                ApikeysError::UnsupportedFormat {
                    prefix: p, line, ..
                } => {
                    assert_eq!(p, prefix);
                    assert_eq!(*line, 1);
                }
                other => panic!("expected UnsupportedFormat for {prefix}, got {other:?}"),
            }
        }
    }

    #[test]
    fn require_hashed_rejects_plaintext_lines() {
        let input = format!("a: {}\nb: sk-plain\n", hash_value("sk-hash"));
        let opts = ApikeysOptions {
            require_hashed: true,
        };
        let err = ApikeysStore::parse_with(&input, opts).unwrap_err();
        assert!(
            matches!(err, ApikeysError::PlaintextKey { line: 2, .. }),
            "{err:?}"
        );
        // A fully hashed file passes.
        assert!(ApikeysStore::parse_with(&format!("a: {}\n", hash_value("sk-hash")), opts).is_ok());
    }

    // ── inline comments ─────────────────────────────────────────────────

    #[test]
    fn inline_comments_are_stripped() {
        let hashed = hash_value("sk-carol");
        let input = format!(
            "\
a:   sk-alice   # trailing note
b:\tsk-bob\t# tab-preceded
c: {hashed}  # rotated 2026-11
d: sk-dave#no-space-stays-part-of-the-token
"
        );
        let store = ApikeysStore::parse(&input).unwrap();
        assert_eq!(store.len(), 4);
        assert_eq!(store.authenticate("sk-alice"), Some("a"));
        assert_eq!(store.authenticate("sk-bob"), Some("b"));
        assert_eq!(store.authenticate("sk-carol"), Some("c"));
        assert_eq!(
            store.authenticate("sk-dave#no-space-stays-part-of-the-token"),
            Some("d")
        );
    }

    #[test]
    fn comment_only_value_is_empty_key_error() {
        let err = ApikeysStore::parse("a: # just a note\n").unwrap_err();
        assert!(matches!(err, ApikeysError::EmptyKey { .. }), "{err:?}");
    }

    #[test]
    fn duplicate_detection_sees_one_digest_through_two_spellings() {
        let hashed = hash_value("sk-shared");
        let input = format!("a: {hashed}\nb: {hashed}\n");
        let err = ApikeysStore::parse(&input).unwrap_err();
        assert!(matches!(err, ApikeysError::DuplicateKey { line: 2, .. }));
    }
}
