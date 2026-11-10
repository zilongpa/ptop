//! Periodic, read-only collection over ssh. One task per cluster.

use crate::slurm::{self, MyJob, Node, Pending, TypeSummary, SEP};
use crate::smi;
use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::{Notify, RwLock};

const SSH_TIMEOUT: Duration = Duration::from_secs(90);
/// Ignore manual refreshes that come sooner than this after the last poll.
const MIN_GAP: i64 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Not known yet: detected on first contact.
    #[default]
    Auto,
    /// Slurm cluster: sinfo / squeue.
    Slurm,
    /// Standalone GPU server: nvidia-smi.
    Smi,
}

#[derive(Clone, Debug)]
pub struct ClusterCfg {
    pub name: String,
    pub host: String,
    /// Login name for this host; None = the global --user.
    pub user: Option<String>,
    pub kind: Kind,
    /// Empty = discover from the user's groups / accounts / QOS.
    pub partitions: Vec<String>,
    /// None = key or agent authentication only.
    pub password: Option<Password>,
}

/// Name of the variable that carries the password to our own askpass invocation.
pub const ASKPASS_ENV: &str = "PTOP_ASKPASS_SECRET";

#[derive(Clone)]
pub enum Password {
    Plain(String),
    /// A shell command that prints the password, e.g. a keychain lookup.
    Command(String),
}

impl std::fmt::Debug for Password {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Password(..)")
    }
}

impl Password {
    async fn resolve(&self) -> Result<String> {
        match self {
            Password::Plain(p) => Ok(p.clone()),
            Password::Command(c) => {
                let (shell, flag) = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
                let out = Command::new(shell).arg(flag).arg(c).stdin(Stdio::null()).output().await.context("running password_command")?;
                let pw = String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or("").trim().to_string();
                if !out.status.success() || pw.is_empty() {
                    bail!("password_command failed or printed nothing");
                }
                Ok(pw)
            }
        }
    }
}

/// Called first thing in `main`: when ssh runs us as its askpass helper, answer and exit.
/// Only a password prompt gets the password; a host-key question is declined.
pub fn answer_askpass() {
    if let Ok(pw) = std::env::var(ASKPASS_ENV) {
        let prompt = std::env::args().nth(1).unwrap_or_default().to_lowercase();
        if prompt.contains("assword") {
            println!("{pw}");
            std::process::exit(0);
        }
        std::process::exit(1);
    }
}

pub fn safe(s: &str, extra: &str) -> bool {
    !s.is_empty() && !s.starts_with('-') && s.chars().all(|c| c.is_ascii_alphanumeric() || extra.contains(c))
}

/// `NAME=[USER@]HOST[:PART1,PART2]`. Giving partitions implies a Slurm cluster.
pub fn parse_spec(spec: &str) -> Result<ClusterCfg> {
    let Some((name, rest)) = spec.split_once('=') else {
        bail!("bad cluster {spec:?}, want NAME=[USER@]HOST[:PARTS]")
    };
    let (login, parts) = rest.split_once(':').unwrap_or((rest, ""));
    let (user, host) = match login.split_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, login),
    };
    if !safe(name, "_-")
        || !safe(host, ".-")
        || user.is_some_and(|u| !safe(u, "._-"))
        || !(parts.is_empty() || safe(parts, "_-.,"))
    {
        bail!("bad cluster {spec:?}");
    }
    let partitions: Vec<String> = parts.split(',').filter(|p| !p.is_empty()).map(String::from).collect();
    Ok(ClusterCfg {
        name: name.into(),
        host: host.into(),
        user: user.map(String::from),
        kind: if partitions.is_empty() { Kind::Auto } else { Kind::Slurm },
        partitions,
        password: None,
    })
}

#[derive(Clone)]
pub struct SshCfg {
    pub user: String,
    pub extra_args: Vec<String>,
    pub password: Option<Password>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ClusterState {
    pub name: String,
    pub kind: Kind,
    pub partitions: Vec<String>,
    pub ok: bool,
    pub error: Option<String>,
    pub last_attempt: Option<DateTime<Utc>>,
    pub last_success: Option<DateTime<Utc>>,
    /// Whether squeue shows other users' jobs (Phoenix yes, ICE no).
    pub jobs_visible: bool,
    pub summary: Vec<TypeSummary>,
    pub pending: Vec<Pending>,
    pub nodes: Vec<Node>,
    /// The user's own Slurm jobs on this cluster (any partition).
    pub my_jobs: Vec<MyJob>,
}

pub struct Cluster {
    pub state: RwLock<ClusterState>,
    pub wake: Notify,
}

pub type Clusters = Arc<Vec<Arc<Cluster>>>;

/// Create shared state for each cluster and spawn its collector task.
pub fn spawn_all(cfgs: Vec<ClusterCfg>, ssh_cfg: SshCfg, interval: Duration) -> Clusters {
    let clusters: Clusters = Arc::new(
        cfgs.iter()
            .map(|c| {
                Arc::new(Cluster {
                    state: RwLock::new(ClusterState {
                        name: c.name.clone(),
                        kind: c.kind,
                        partitions: c.partitions.clone(),
                        ..Default::default()
                    }),
                    wake: Notify::new(),
                })
            })
            .collect(),
    );
    for (c, cfg) in clusters.iter().zip(cfgs) {
        tokio::spawn(run(c.clone(), cfg, ssh_cfg.clone(), interval));
    }
    clusters
}

/// `bash -s` is not a login shell, so on hosts that put Slurm on the PATH in /etc/profile
/// (or the user's profile) the tools are missing. Load the login environment in that case
/// only. The profiles run with stdin closed so they cannot swallow the rest of the script.
const LOGIN_ENV: &str = r#"if ! command -v sinfo >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
  [ -r /etc/profile ] && . /etc/profile </dev/null >/dev/null 2>&1
  for f in "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.profile"; do
    if [ -r "$f" ]; then . "$f" </dev/null >/dev/null 2>&1; break; fi
  done
fi"#;

/// Run `script` on the host with `bash -s`, fed on stdin, so the user's login shell
/// (tcsh on some College of Computing machines) never has to parse it.
async fn ssh(cfg: &SshCfg, user: &str, host: &str, script: &str) -> Result<String> {
    let mut cmd = Command::new("ssh");
    cmd.args(["-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15"]);
    match &cfg.password {
        // never prompt: a host that wants a password just fails
        None => {
            cmd.args(["-o", "BatchMode=yes"]);
        }
        // ssh asks its askpass helper for the password; the helper is this same binary
        Some(p) => {
            cmd.args(["-o", "NumberOfPasswordPrompts=1"])
                .env("SSH_ASKPASS", std::env::current_exe().context("locating the ptop binary")?)
                .env("SSH_ASKPASS_REQUIRE", "force")
                .env("DISPLAY", std::env::var("DISPLAY").unwrap_or_else(|_| ":0".into()))
                .env(ASKPASS_ENV, p.resolve().await?);
        }
    }
    let mut child = cmd
        .args(&cfg.extra_args)
        .arg(format!("{user}@{host}"))
        .arg("bash -s")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("cannot spawn ssh")?;
    let mut stdin = child.stdin.take().context("no ssh stdin")?;
    let script = format!("{LOGIN_ENV}\n{script}");
    let run = async move {
        // a host that rejects us closes stdin early; the exit status below says why
        let _ = stdin.write_all(script.as_bytes()).await;
        let _ = stdin.write_all(b"\n").await;
        drop(stdin);
        child.wait_with_output().await
    };
    let out = tokio::time::timeout(SSH_TIMEOUT, run)
        .await
        .map_err(|_| anyhow!("ssh timed out after {}s", SSH_TIMEOUT.as_secs()))??;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        // drop the institutional login banner, keep the actual error
        let err: Vec<&str> = err.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('*')).collect();
        let err = err.join("\n");
        bail!("ssh exit {}: {}", out.status, if err.is_empty() { "(no stderr)" } else { &err });
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

async fn detect(ssh_cfg: &SshCfg, user: &str, host: &str) -> Result<Kind> {
    let out = ssh(
        ssh_cfg,
        user,
        host,
        "if command -v sinfo >/dev/null 2>&1; then echo slurm; \
         elif command -v nvidia-smi >/dev/null 2>&1; then echo smi; else echo none; fi",
    )
    .await?;
    match out.lines().last().unwrap_or("").trim() {
        "slurm" => Ok(Kind::Slurm),
        "smi" => Ok(Kind::Smi),
        _ => bail!("{host} has neither sinfo nor nvidia-smi on its PATH"),
    }
}

async fn discover(ssh_cfg: &SshCfg, user: &str, host: &str) -> Result<Vec<String>> {
    let cmd = format!(
        "id -Gn; echo {SEP}; sacctmgr -nP show assoc user={user} format=account,qos; \
         echo {SEP}; scontrol show partition -o"
    );
    let out = ssh(ssh_cfg, user, host, &cmd).await?;
    let parts: Vec<&str> = out.split(SEP).collect();
    if parts.len() < 3 {
        bail!("unexpected discovery output");
    }
    let acc = slurm::parse_access(parts[0], parts[1]);
    let p = slurm::usable_gpu_partitions(parts[2], &acc);
    if p.is_empty() {
        bail!("no GPU partition found that {user} can submit to");
    }
    Ok(p)
}

#[derive(Default)]
struct Poll {
    nodes: Vec<Node>,
    jobs_visible: bool,
    pending: Vec<Pending>,
    my_jobs: Vec<MyJob>,
}

async fn poll_slurm(ssh_cfg: &SshCfg, user: &str, host: &str, partitions: &[String]) -> Result<Poll> {
    let p = partitions.join(",");
    let cmd = format!(
        "date -Iseconds; echo {SEP}; sinfo -h -N -p {p} -O \"{nf}\"; echo {SEP}; \
         squeue -h -t R -p {p} -O \"{rf}\"; echo {SEP}; squeue -h -t PD -p {p} -O \"{pf}\"; \
         echo {SEP}; squeue -h -u {user} -O \"{mf}\"",
        nf = slurm::NODE_FMT,
        rf = slurm::RUN_FMT,
        pf = slurm::PEND_FMT,
        mf = slurm::MINE_FMT,
    );
    let out = ssh(ssh_cfg, user, host, &cmd).await?;
    let parts: Vec<&str> = out.split(SEP).collect();
    if parts.len() < 5 {
        bail!("unexpected poll output");
    }
    let cluster_time = DateTime::parse_from_rfc3339(parts[0].trim()).context("bad `date` output")?;
    let mut nodes = slurm::parse_nodes(parts[1]);
    if nodes.is_empty() {
        bail!("sinfo returned no GPU nodes for {p}");
    }
    let jobs_visible = slurm::attach_running(&mut nodes, parts[2], user, *cluster_time.offset());
    Ok(Poll {
        nodes,
        jobs_visible,
        pending: slurm::parse_pending(parts[3]),
        my_jobs: slurm::parse_mine(parts[4]),
    })
}

async fn poll_smi(ssh_cfg: &SshCfg, user: &str, host: &str) -> Result<Poll> {
    let out = ssh(ssh_cfg, user, host, &format!("date -Iseconds; echo {SEP}; {}", smi::command())).await?;
    let (_, rest) = out.split_once(SEP).context("unexpected nvidia-smi output")?;
    let node = smi::parse(rest).context("could not parse nvidia-smi output")?;
    Ok(Poll {
        // every process on a standalone host is visible
        jobs_visible: true,
        nodes: vec![node],
        ..Default::default()
    })
}

pub async fn run(cluster: Arc<Cluster>, cfg: ClusterCfg, ssh_cfg: SshCfg, interval: Duration) {
    let mut partitions = cfg.partitions.clone();
    let mut kind = cfg.kind;
    let user = cfg.user.clone().unwrap_or_else(|| ssh_cfg.user.clone());
    let ssh_cfg = SshCfg { password: cfg.password.clone(), ..ssh_cfg };
    loop {
        let now = Utc::now();
        let result = async {
            if kind == Kind::Auto {
                kind = detect(&ssh_cfg, &user, &cfg.host).await.context("detecting cluster type")?;
            }
            match kind {
                Kind::Smi => poll_smi(&ssh_cfg, &user, &cfg.host).await,
                _ => {
                    if partitions.is_empty() {
                        partitions = discover(&ssh_cfg, &user, &cfg.host).await.context("partition discovery")?;
                    }
                    poll_slurm(&ssh_cfg, &user, &cfg.host, &partitions).await
                }
            }
        }
        .await;
        {
            let mut s = cluster.state.write().await;
            s.last_attempt = Some(now);
            s.partitions = partitions.clone();
            s.kind = kind;
            match result {
                Ok(p) => {
                    s.ok = true;
                    s.error = None;
                    s.last_success = Some(now);
                    s.jobs_visible = p.jobs_visible;
                    s.summary = slurm::summarize(&p.nodes);
                    s.nodes = p.nodes;
                    s.pending = p.pending;
                    s.my_jobs = p.my_jobs;
                }
                Err(e) => {
                    // keep the last good data on screen, flagged as stale
                    s.ok = false;
                    s.error = Some(format!("{e:#}"));
                }
            }
        }
        // retry sooner while failing (VPN drop etc.)
        let wait = if cluster.state.read().await.ok { interval } else { interval.min(Duration::from_secs(60)) };
        tokio::select! {
            _ = tokio::time::sleep(wait) => {}
            _ = cluster.wake.notified() => {
                let since = Utc::now() - now;
                if since.num_seconds() < MIN_GAP {
                    tokio::time::sleep(Duration::from_secs((MIN_GAP - since.num_seconds()) as u64)).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs() {
        let c = parse_spec("fry=gburdell3@fry.cc.gatech.edu").unwrap();
        assert_eq!((c.user.as_deref(), c.host.as_str(), c.kind), (Some("gburdell3"), "fry.cc.gatech.edu", Kind::Auto));
        let c = parse_spec("phoenix=login-phoenix.pace.gatech.edu:gpu-h200,gpu-l40s").unwrap();
        assert_eq!((c.user, c.kind, c.partitions.len()), (None, Kind::Slurm, 2));
        assert!(parse_spec("x=host;rm -rf").is_err());
        assert!(parse_spec("x=-oProxyCommand=foo").is_err());
    }
}
