//! The TOML config file: `~/.config/ptop/config.toml`, or `%APPDATA%\ptop\config.toml` on Windows.

use crate::collect::{safe, ClusterCfg, Kind, Password};
use anyhow::{bail, Context, Result};
use etcetera::{choose_base_strategy, BaseStrategy};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub user: Option<String>,
    pub interval: Option<u64>,
    #[serde(default)]
    pub ssh_args: Vec<String>,
    #[serde(default, rename = "cluster")]
    pub clusters: Vec<ClusterEntry>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ClusterEntry {
    pub name: String,
    pub host: String,
    pub user: Option<String>,
    #[serde(default)]
    pub partitions: Vec<String>,
    password: Option<String>,
    password_command: Option<String>,
}

impl ClusterEntry {
    pub fn into_cfg(self) -> Result<ClusterCfg> {
        if !safe(&self.name, "_-") {
            bail!("bad cluster name {:?}", self.name);
        }
        if !safe(&self.host, ".-") {
            bail!("cluster {}: bad host {:?}", self.name, self.host);
        }
        if let Some(u) = self.user.as_deref().filter(|u| !safe(u, "._-")) {
            bail!("cluster {}: bad user {u:?}", self.name);
        }
        if let Some(p) = self.partitions.iter().find(|p| !safe(p, "_-.")) {
            bail!("cluster {}: bad partition {p:?}", self.name);
        }
        let password = match (self.password, self.password_command) {
            (Some(_), Some(_)) => bail!("cluster {}: set password or password_command, not both", self.name),
            (Some(p), None) => Some(Password::Plain(p)),
            (None, Some(c)) => Some(Password::Command(c)),
            (None, None) => None,
        };
        Ok(ClusterCfg {
            name: self.name,
            host: self.host,
            user: self.user,
            kind: if self.partitions.is_empty() { Kind::Auto } else { Kind::Slurm },
            partitions: self.partitions,
            password,
        })
    }
}

/// XDG layout on Linux and macOS (`~/.config`), the roaming AppData folder on Windows.
pub fn default_path() -> Option<PathBuf> {
    Some(choose_base_strategy().ok()?.config_dir().join("ptop").join("config.toml"))
}

/// `~/x` -> `<home>/x`, so key paths in `ssh_args` work the same on every platform.
pub fn expand_home(s: &str) -> String {
    match (s.strip_prefix("~/").or_else(|| s.strip_prefix("~\\")), etcetera::home_dir()) {
        (Some(rest), Ok(home)) => home.join(rest).to_string_lossy().into_owned(),
        _ => s.to_string(),
    }
}

/// The commented starter file written by `ptop init`.
pub fn template(user: Option<&str>) -> String {
    let local = std::env::var("USER").or_else(|_| std::env::var("USERNAME")).ok().filter(|u| safe(u, "._-"));
    let user = user.map(String::from).or(local).unwrap_or_else(|| "your-login".into());
    format!(
        r#"# ptop config. Run `ptop --help` or see the README for details.

# Login name on the clusters.
user = "{user}"

# Seconds between polls.
interval = 60

# Extra arguments for every ssh call, for example a key file.
# ssh_args = ["-i", "~/.ssh/cluster_key"]

[[cluster]]
name = "ice"
host = "login-ice.pace.gatech.edu"

[[cluster]]
name = "phoenix"
host = "login-phoenix.pace.gatech.edu"

# A host that only accepts a password. Connect to it once by hand first so its
# host key is known. Prefer password_command (a command that prints the password,
# such as a keychain lookup) over a plain password.
#
# [[cluster]]
# name = "lab"
# host = "gpu-box.example.edu"
# user = "someone"                    # if it differs from the user above
# partitions = ["gpu-a", "gpu-b"]     # skip partition discovery
# password_command = "pass show lab"
# password = "..."
"#
    )
}

/// Write the template to the default location. Never overwrites an existing file.
pub fn init(user: Option<&str>) -> Result<PathBuf> {
    let path = default_path().context("cannot work out where the config file belongs on this system")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        // private from the start, so adding a password later passes the permission check
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = match opts.open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            bail!("{} already exists; edit it, or delete it to start over", path.display())
        }
        Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
    };
    std::io::Write::write_all(&mut file, template(user).as_bytes())?;
    Ok(path)
}

pub fn parse(text: &str) -> Result<Config> {
    let mut cfg: Config = toml::from_str(text)?;
    cfg.ssh_args = cfg.ssh_args.iter().map(|a| expand_home(a)).collect();
    Ok(cfg)
}

/// A file holding a password must not be readable by other users.
#[cfg(unix)]
fn check_private(path: &Path, cfg: &Config) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if cfg.clusters.iter().any(|c| c.password.is_some()) {
        let mode = std::fs::metadata(path)?.permissions().mode();
        if mode & 0o077 != 0 {
            bail!("{} holds a password but other users can read it; run: chmod 600 {}", path.display(), path.display());
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_private(_: &Path, _: &Config) -> Result<()> {
    Ok(())
}

/// Load the given file, or the default one if it exists.
pub fn load(explicit: Option<PathBuf>) -> Result<Config> {
    let (path, required) = match explicit {
        Some(p) => (p, true),
        None => match default_path() {
            Some(p) => (p, false),
            None => return Ok(Config::default()),
        },
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if !required && e.kind() == std::io::ErrorKind::NotFound => return Ok(Config::default()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let cfg = parse(&text).with_context(|| format!("in {}", path.display()))?;
    check_private(&path, &cfg)?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_file() {
        let cfg = parse(
            r#"
            user = "gburdell3"
            interval = 90
            ssh_args = ["-i", "/keys/id"]

            [[cluster]]
            name = "ice"
            host = "login-ice.pace.gatech.edu"

            [[cluster]]
            name = "lab"
            host = "gpu-box.example.edu"
            user = "someone"
            partitions = ["gpu-a", "gpu-b"]
            password_command = "pass show lab"
            "#,
        )
        .unwrap();
        assert_eq!((cfg.user.as_deref(), cfg.interval), (Some("gburdell3"), Some(90)));
        let mut c: Vec<ClusterCfg> = cfg.clusters.into_iter().map(|c| c.into_cfg().unwrap()).collect();
        let lab = c.pop().unwrap();
        assert_eq!((lab.kind, lab.user.as_deref(), lab.partitions.len()), (Kind::Slurm, Some("someone"), 2));
        assert!(matches!(lab.password, Some(Password::Command(_))));
        assert!(c[0].password.is_none() && c[0].kind == Kind::Auto);
    }

    #[test]
    fn template_is_valid() {
        let cfg = parse(&template(Some("gburdell3"))).unwrap();
        assert_eq!(cfg.user.as_deref(), Some("gburdell3"));
        assert_eq!(cfg.clusters.len(), 2);
        // the commented example must parse too once uncommented
        let example: String = template(None).lines().skip_while(|l| !l.starts_with("# [[cluster]]")).map(|l| l.trim_start_matches("# ").to_string() + "\n").collect();
        let lab = parse(&example.replace("password = \"...\"", "")).unwrap();
        assert_eq!(lab.clusters[0].partitions.len(), 2);
    }

    #[test]
    fn rejects_mistakes() {
        assert!(parse("usr = \"x\"").is_err(), "unknown key");
        let both = parse("[[cluster]]\nname=\"a\"\nhost=\"h\"\npassword=\"x\"\npassword_command=\"y\"").unwrap();
        assert!(both.clusters.into_iter().next().unwrap().into_cfg().is_err());
        let inj = parse("[[cluster]]\nname=\"a\"\nhost=\"-oProxyCommand=evil\"").unwrap();
        assert!(inj.clusters.into_iter().next().unwrap().into_cfg().is_err());
    }
}
