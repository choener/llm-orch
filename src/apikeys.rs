use std::collections::{HashMap, HashSet};
use std::path::Path;

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
}

// ---------------------------------------------------------------------------
// ApiKeys store
// ---------------------------------------------------------------------------

/// A parsed, ready-to-query set of API keys.
///
/// Internally maps each key to its label for O(1) lookup during authentication.
#[derive(Debug, Clone)]
pub struct ApikeysStore {
    /// key → label
    keys: HashMap<String, String>,
}

impl ApikeysStore {
    /// Load and parse an API keys file.
    pub fn load(path: &Path) -> Result<Self, ApikeysError> {
        let contents = std::fs::read_to_string(path)?;
        Self::parse(&contents)
    }

    /// Parse API keys from a string (useful for testing).
    fn parse(input: &str) -> Result<Self, ApikeysError> {
        let mut keys = HashMap::new();
        let mut seen_labels = HashSet::new();

        for (idx, line) in input.lines().enumerate() {
            let line_num = idx + 1;
            let trimmed = line.trim();

            // Skip blank lines and comments.
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Split on the first `:` only.
            let (label, key) = trimmed
                .split_once(':')
                .ok_or(ApikeysError::MissingColon { line: line_num })?;

            let label = label.trim().to_owned();
            let key = key.trim().to_owned();

            if label.is_empty() {
                return Err(ApikeysError::EmptyLabel { line: line_num });
            }
            if key.is_empty() {
                return Err(ApikeysError::EmptyKey {
                    label,
                    line: line_num,
                });
            }
            if !seen_labels.insert(label.clone()) {
                return Err(ApikeysError::DuplicateLabel { label, line: line_num });
            }
            if keys.contains_key(&key) {
                return Err(ApikeysError::DuplicateKey { label, line: line_num });
            }

            keys.insert(key, label);
        }

        Ok(Self { keys })
    }

    /// Returns the label for a given key, or `None` if the key is unknown.
    pub fn authenticate(&self, key: &str) -> Option<&str> {
        self.keys.get(key).map(|s| s.as_str())
    }

    /// Returns `true` when no keys are configured (fail-closed auth).
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of configured keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Create an empty store (fail-closed — all requests denied).
    pub fn empty() -> Self {
        Self {
            keys: HashMap::new(),
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
}