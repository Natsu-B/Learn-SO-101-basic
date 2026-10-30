pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("worker error: {0}")]
    Worker(String),
    #[error("invalid input: {0}")]
    Input(String),
    #[error("SO101_MODEL_DIR is not set; enter nix develop or pass a model directory")]
    MissingModelDirectory,
    #[error("command dimension mismatch: expected {expected}, got {actual}")]
    Dimension { expected: usize, actual: usize },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("model error: {0}")]
    Model(String),
    #[error("expected MuJoCo 3.10.0, found {0}")]
    Version(String),
    #[error(
        "invalid command for {joint}: {value}; expected finite target in [{min}, {max}] radians"
    )]
    Command {
        joint: String,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error("simulation became non-finite or MuJoCo reset time; call reset() before continuing")]
    Unstable,
}
