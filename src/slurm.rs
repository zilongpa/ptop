//! Pure parsing of sinfo / squeue / scontrol output. No I/O here.

use chrono::{DateTime, FixedOffset, NaiveDateTime, TimeZone, Utc};
use regex::Regex;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

/// Separator echoed between the outputs of the commands in one ssh round trip.
pub const SEP: &str = "@@PTOP@@";

pub const NODE_FMT: &str = "NodeList:80,Partition:60,StateLong:60,Gres:400,GresUsed:400,\
                            CPUsState:40,Memory:24,AllocMem:24";
pub const RUN_FMT: &str = "UserName:40,NumNodes:10,NodeList:600,tres-alloc:600,EndTime:30";
pub const PEND_FMT: &str = "Partition:300,tres-per-job:200,tres-per-node:200,NumNodes:10,Reason:80";
/// Own jobs. Name goes last because it is the only field that may contain spaces.
pub const MINE_FMT: &str = "JobID:24,Partition:40,StateCompact:8,TimeUsed:16,TimeLimit:16,NumNodes:8,\
                            tres-alloc:400,StartTime:24,ReasonList:400,Name:200";

const OFFLINE_STATES: &[&str] = &[
    "drain", "down", "fail", "maint", "reserv", "not_resp", "inval", "power", "unknown", "future",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    /// Free GPUs (if any) can be allocated right now.
    Ok,
    /// Trailing '-' in StateLong (PLANNED): backfill reserved the node for a pending job.
    Reserved,
    CpuFull,
    MemFull,
    /// drain / down / maint / ...: nothing can start here.
    Offline,
}

#[derive(Clone, Debug, Serialize)]
pub struct Release {
    /// Job time-limit end; jobs often finish earlier, so this is an upper bound.
    pub at: DateTime<Utc>,
    pub gpus: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Node {
    pub name: String,
    pub partitions: Vec<String>,
    pub state: String,
    pub gpu_type: String,
    pub total: u32,
    pub used: u32,
    pub free: u32,
    /// Free GPUs that a new job can actually get now (0 unless status == Ok).
    pub usable: u32,
    pub status: NodeStatus,
    pub cpu_idle: u32,
    pub cpu_total: u32,
    pub mem_free_gb: u64,
    pub mem_total_gb: u64,
    pub releases: Vec<Release>,
    /// Per-GPU detail; only filled for standalone (nvidia-smi) hosts.
    pub gpus: Vec<crate::smi::GpuDev>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TypeSummary {
    pub gpu_type: String,
    pub nodes: u32,
    pub total: u32,
    pub used: u32,
    pub usable: u32,
    /// Free but held back by a PLANNED reservation or full CPU/memory.
    pub blocked: u32,
    pub offline: u32,
    pub max_node_free: u32,
    pub full_nodes: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Pending {
    pub partitions: String,
    pub gpu_type: Option<String>,
    pub jobs: u32,
    pub gpus: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct MyJob {
    pub id: String,
    pub partition: String,
    /// R, PD, CG, ...
    pub state: String,
    pub time_used: String,
    pub time_limit: String,
    pub gpus: u32,
    pub gpu_type: Option<String>,
    /// Running: node list. Pending: the reason, e.g. (Priority).
    pub where_or_why: String,
    /// Pending only: Slurm's start estimate (often N/A or pessimistic).
    pub est_start: Option<String>,
    pub name: String,
}

#[derive(Debug)]
pub struct Access {
    pub groups: HashSet<String>,
    pub accounts: HashSet<String>,
    pub qos: HashSet<String>,
}

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).unwrap())
}

/// "gpu:h200:8(S:0-1),gpu:a100:2" -> {"h200": 8, "a100": 2}; untyped "gpu:4" -> {"gpu": 4}.
pub fn gres_counts(s: &str) -> BTreeMap<String, u32> {
    static TYPED: OnceLock<Regex> = OnceLock::new();
    static PLAIN: OnceLock<Regex> = OnceLock::new();
    let mut out = BTreeMap::new();
    for c in re(&TYPED, r"gpu:([A-Za-z][A-Za-z0-9_\-]*):(\d+)").captures_iter(s) {
        *out.entry(c[1].to_string()).or_insert(0) += c[2].parse::<u32>().unwrap_or(0);
    }
    if out.is_empty() {
        for c in re(&PLAIN, r"gpu:(\d+)").captures_iter(s) {
            *out.entry("gpu".to_string()).or_insert(0) += c[1].parse::<u32>().unwrap_or(0);
        }
    }
    out
}

/// "a-[1,3-4]-0,b" -> ["a-1-0", "a-3-0", "a-4-0", "b"] (one bracket group per host).
pub fn expand_hostlist(s: &str) -> Vec<String> {
    static HOST: OnceLock<Regex> = OnceLock::new();
    let mut hosts = Vec::new();
    for c in re(&HOST, r"([^,\[]+)(?:\[([^\]]+)\])?([^,]*)").captures_iter(s) {
        let pre = &c[1];
        let post = c.get(3).map_or("", |m| m.as_str());
        let Some(rng) = c.get(2) else {
            hosts.push(format!("{pre}{post}"));
            continue;
        };
        for part in rng.as_str().split(',') {
            match part.split_once('-') {
                Some((a, b)) => {
                    let (Ok(lo), Ok(hi)) = (a.parse::<u64>(), b.parse::<u64>()) else { continue };
                    for i in lo..=hi {
                        hosts.push(format!("{pre}{i:0w$}{post}", w = a.len()));
                    }
                }
                None => hosts.push(format!("{pre}{part}{post}")),
            }
        }
    }
    hosts
}

pub fn parse_nodes(text: &str) -> Vec<Node> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 8 {
            continue;
        }
        let (name, part, state) = (f[0], f[1], f[2]);
        // a node is listed once per partition it belongs to
        if let Some(&i) = index.get(name) {
            nodes[i].partitions.push(part.to_string());
            continue;
        }
        let total = gres_counts(f[3]);
        let Some((gpu_type, &tot)) = total.iter().max_by_key(|(_, n)| **n) else { continue };
        let used = *gres_counts(f[4]).get(gpu_type).unwrap_or(&0);
        let cpus: Vec<u32> = f[5].split('/').map(|x| x.parse().unwrap_or(0)).collect();
        let (cpu_idle, cpu_total) = (*cpus.get(1).unwrap_or(&0), *cpus.get(3).unwrap_or(&0));
        let mem_mb: u64 = f[6].parse().unwrap_or(0);
        let alloc_mb: u64 = f[7].parse().unwrap_or(0);
        let mem_free_gb = mem_mb.saturating_sub(alloc_mb) / 1024;
        let free = tot.saturating_sub(used);
        let lower = state.to_lowercase();
        let status = if OFFLINE_STATES.iter().any(|b| lower.contains(b)) {
            NodeStatus::Offline
        } else if free > 0 && state.ends_with('-') {
            NodeStatus::Reserved
        } else if free > 0 && cpu_idle == 0 {
            NodeStatus::CpuFull
        } else if free > 0 && mem_free_gb < 8 {
            NodeStatus::MemFull
        } else {
            NodeStatus::Ok
        };
        index.insert(name.to_string(), nodes.len());
        nodes.push(Node {
            name: name.to_string(),
            partitions: vec![part.to_string()],
            state: state.to_string(),
            gpu_type: gpu_type.clone(),
            total: tot,
            used,
            free,
            usable: if status == NodeStatus::Ok { free } else { 0 },
            status,
            cpu_idle,
            cpu_total,
            mem_free_gb,
            mem_total_gb: mem_mb / 1024,
            releases: Vec::new(),
            gpus: Vec::new(),
        });
    }
    nodes.sort_by(|a, b| a.gpu_type.cmp(&b.gpu_type).then(a.name.cmp(&b.name)));
    nodes
}

/// Attach running jobs' end times to nodes. Returns true if any job belongs to
/// someone other than `me` (i.e. the cluster lets us see other users' jobs).
pub fn attach_running(nodes: &mut [Node], text: &str, me: &str, tz: FixedOffset) -> bool {
    static GPU: OnceLock<Regex> = OnceLock::new();
    let gpu_re = re(&GPU, r"gres/gpu(?::[^=,]+)?=(\d+)");
    let index: HashMap<String, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.name.clone(), i)).collect();
    let mut others = false;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        others |= f[0] != me;
        // gres/gpu=N and gres/gpu:type=N both appear; take the max (they are equal)
        let gpus = gpu_re.captures_iter(f[3]).filter_map(|c| c[1].parse::<u32>().ok()).max();
        let Some(gpus) = gpus.filter(|g| *g > 0) else { continue };
        let Ok(naive) = NaiveDateTime::parse_from_str(f[4], "%Y-%m-%dT%H:%M:%S") else { continue };
        let Some(at) = tz.from_local_datetime(&naive).earliest() else { continue };
        let hosts = expand_hostlist(f[2]);
        let per = gpus / hosts.len().max(1) as u32;
        for h in hosts {
            if let (Some(&i), true) = (index.get(&h), per > 0) {
                nodes[i].releases.push(Release { at: at.with_timezone(&Utc), gpus: per });
            }
        }
    }
    for n in nodes.iter_mut() {
        n.releases.sort_by_key(|r| r.at);
    }
    others
}

/// GPU jobs really competing for GPUs (reason Priority/Resources, not held/dependency).
pub fn parse_pending(text: &str) -> Vec<Pending> {
    static GPU: OnceLock<Regex> = OnceLock::new();
    let gpu_re = re(&GPU, r"gpu(?::([A-Za-z][A-Za-z0-9_\-]*))?[:=](\d+)");
    let mut agg: BTreeMap<(String, Option<String>), (u32, u32)> = BTreeMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 || !matches!(f[4], "Priority" | "Resources") {
            continue;
        }
        let (per_job, src) = if f[1] != "N/A" { (true, f[1]) } else { (false, f[2]) };
        let Some(c) = gpu_re.captures(src) else { continue };
        let n: u32 = c[2].parse().unwrap_or(0);
        let nodes: u32 = f[3].parse().unwrap_or(1);
        let gpus = if per_job { n } else { n * nodes.max(1) };
        let e = agg.entry((f[0].to_string(), c.get(1).map(|m| m.as_str().to_string()))).or_default();
        e.0 += 1;
        e.1 += gpus;
    }
    let mut out: Vec<Pending> = agg
        .into_iter()
        .map(|((partitions, gpu_type), (jobs, gpus))| Pending { partitions, gpu_type, jobs, gpus })
        .collect();
    out.sort_by(|a, b| b.gpus.cmp(&a.gpus));
    out
}

/// Parse `squeue -u me -O MINE_FMT`.
pub fn parse_mine(text: &str) -> Vec<MyJob> {
    static GPU: OnceLock<Regex> = OnceLock::new();
    let gpu_re = re(&GPU, r"gres/gpu(?::([A-Za-z][A-Za-z0-9_\-]*))?=(\d+)");
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 9 {
            continue;
        }
        let mut gpus = 0;
        let mut gpu_type = None;
        for c in gpu_re.captures_iter(f[6]) {
            gpus = gpus.max(c[2].parse().unwrap_or(0));
            if let Some(t) = c.get(1) {
                gpu_type = Some(t.as_str().to_string());
            }
        }
        let state = f[2].to_string();
        out.push(MyJob {
            id: f[0].into(),
            partition: f[1].into(),
            time_used: f[3].into(),
            time_limit: f[4].into(),
            gpus,
            gpu_type,
            est_start: (state == "PD" && f[7] != "N/A").then(|| f[7].to_string()),
            state,
            where_or_why: f[8].into(),
            name: f[9..].join(" "),
        });
    }
    out.sort_by(|a, b| (a.state != "R").cmp(&(b.state != "R")).then(a.id.cmp(&b.id)));
    out
}

pub fn summarize(nodes: &[Node]) -> Vec<TypeSummary> {
    let mut by: BTreeMap<&str, TypeSummary> = BTreeMap::new();
    for n in nodes {
        let s = by.entry(&n.gpu_type).or_insert_with(|| TypeSummary {
            gpu_type: n.gpu_type.clone(),
            ..Default::default()
        });
        s.nodes += 1;
        s.total += n.total;
        s.used += n.used;
        s.usable += n.usable;
        match n.status {
            NodeStatus::Offline => s.offline += n.free,
            NodeStatus::Ok => {}
            _ => s.blocked += n.free,
        }
        s.max_node_free = s.max_node_free.max(n.usable);
        s.full_nodes += (n.usable == n.total) as u32;
    }
    let mut v: Vec<TypeSummary> = by.into_values().collect();
    v.sort_by(|a, b| b.usable.cmp(&a.usable).then(b.total.cmp(&a.total)));
    v
}

/// `id -Gn` / `sacctmgr -nP show assoc ... format=account,qos` output.
pub fn parse_access(groups: &str, assoc: &str) -> Access {
    let mut accounts = HashSet::new();
    let mut qos = HashSet::new();
    for line in assoc.lines() {
        if let Some((acct, q)) = line.trim().split_once('|') {
            accounts.insert(acct.to_string());
            qos.extend(q.split(',').filter(|s| !s.is_empty()).map(String::from));
        }
    }
    Access {
        groups: groups.split_whitespace().map(String::from).collect(),
        accounts,
        qos,
    }
}

/// From `scontrol show partition -o`, pick UP, non-hidden GPU partitions this user may submit to.
/// Clusters that do not track GPUs in partition TRES (mixed CPU / GPU / FPGA partitions) list no
/// GPU partition at all; then every partition the user may submit to is returned, and nodes
/// without GPUs are dropped later when sinfo is parsed.
pub fn usable_gpu_partitions(scontrol: &str, acc: &Access) -> Vec<String> {
    let allowed = |val: Option<&&str>, mine: &HashSet<String>| match val {
        None | Some(&"ALL") => true,
        Some(v) => v.split(',').any(|x| mine.contains(x)),
    };
    let not_all_denied = |val: Option<&&str>, mine: &HashSet<String>| match val {
        None | Some(&"(null)") => true,
        Some(v) => {
            let denied: HashSet<&str> = v.split(',').collect();
            mine.is_empty() || mine.iter().any(|m| !denied.contains(m.as_str()))
        }
    };
    let mut out = Vec::new();
    let mut any = Vec::new();
    for line in scontrol.lines() {
        let kv: HashMap<&str, &str> =
            line.split_whitespace().filter_map(|t| t.split_once('=')).collect();
        let Some(name) = kv.get("PartitionName") else { continue };
        let ok = kv.get("State") == Some(&"UP")
            && kv.get("Hidden") != Some(&"YES")
            && allowed(kv.get("AllowGroups"), &acc.groups)
            && allowed(kv.get("AllowAccounts"), &acc.accounts)
            && allowed(kv.get("AllowQos"), &acc.qos)
            && not_all_denied(kv.get("DenyAccounts"), &acc.accounts)
            && not_all_denied(kv.get("DenyQos"), &acc.qos);
        if ok {
            any.push(name.to_string());
            if kv.get("TRES").is_some_and(|t| t.contains("gres/gpu")) && !name.contains("cpu") {
                out.push(name.to_string());
            }
        }
    }
    if out.is_empty() { any } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostlists() {
        assert_eq!(
            expand_hostlist("atl1-1-02-012-[2,9-10]-0,atl1-1-03-019-14-0"),
            ["atl1-1-02-012-2-0", "atl1-1-02-012-9-0", "atl1-1-02-012-10-0", "atl1-1-03-019-14-0"]
        );
        assert_eq!(expand_hostlist("n[08-10]"), ["n08", "n09", "n10"]);
    }

    #[test]
    fn gres() {
        let g = gres_counts("gpu:rtx_pro_6000_blackwell:16(S:0-1)");
        assert_eq!(g["rtx_pro_6000_blackwell"], 16);
        assert_eq!(gres_counts("gpu:a40:2(IDX:0-1)")["a40"], 2);
        assert_eq!(gres_counts("gpu:mi210:0(IDX:N/A)")["mi210"], 0);
    }

    #[test]
    fn node_status() {
        let t = "n1 gpu-h200 mixed- gpu:h200:8(S:0-1) gpu:h200:6(IDX:0-5) 40/24/0/64 2063000 100000\n\
                 n2 gpu-h200 drained gpu:h200:8(S:0-1) gpu:h200:0(IDX:N/A) 0/0/64/64 2063000 0\n\
                 n3 gpu-l40s mixed gpu:l40s:8(S:0-3) gpu:l40s:5(IDX:0-4) 64/0/0/64 500000 1000\n\
                 n3 other mixed gpu:l40s:8(S:0-3) gpu:l40s:5(IDX:0-4) 64/0/0/64 500000 1000\n";
        let n = parse_nodes(t);
        assert_eq!(n.len(), 3);
        let by = |x: &str| n.iter().find(|m| m.name == x).unwrap();
        assert_eq!(by("n1").status, NodeStatus::Reserved);
        assert_eq!(by("n2").status, NodeStatus::Offline);
        assert_eq!(by("n3").status, NodeStatus::CpuFull);
        assert_eq!(by("n3").partitions.len(), 2);
        let s = summarize(&n);
        assert_eq!(s.iter().find(|x| x.gpu_type == "h200").unwrap().offline, 8);
    }

    #[test]
    fn partitions() {
        let acc = parse_access("users", "");
        let gpu = "PartitionName=gpu-a State=UP Hidden=NO AllowGroups=ALL TRES=cpu=64,gres/gpu=8\n\
                   PartitionName=cpu-a State=UP Hidden=NO AllowGroups=ALL TRES=cpu=64\n\
                   PartitionName=gpu-b State=UP Hidden=NO AllowGroups=staff TRES=cpu=64,gres/gpu=8\n";
        assert_eq!(usable_gpu_partitions(gpu, &acc), ["gpu-a"]);
        // no partition advertises GPUs: fall back to everything the user can submit to
        let mixed = "PartitionName=techfee State=UP Hidden=NO AllowGroups=ALL TRES=cpu=900,mem=9T,node=150\n\
                     PartitionName=general State=UP Hidden=NO AllowGroups=ALL TRES=cpu=900,mem=9T,node=150\n\
                     PartitionName=down State=DOWN Hidden=NO AllowGroups=ALL TRES=cpu=1\n";
        assert_eq!(usable_gpu_partitions(mixed, &acc), ["techfee", "general"]);
    }

    #[test]
    fn mine() {
        let t = "123 ice-gpu R 1:02:03 4:00:00 1 cpu=8,mem=64G,node=1,billing=8,gres/gpu=1,gres/gpu:h100=1 2026-11-09T10:00:00 atl1-1-03-012-18-0 train vla\n\
                 124 ice-gpu PD 0:00 8:00:00 1 cpu=8,mem=64G,node=1,billing=8,gres/gpu=2 2026-11-10T03:00:00 (Priority) eval\n";
        let j = parse_mine(t);
        assert_eq!(j.len(), 2);
        assert_eq!(j[0].name, "train vla");
        assert_eq!((j[0].gpus, j[0].gpu_type.as_deref()), (1, Some("h100")));
        assert_eq!(j[1].est_start.as_deref(), Some("2026-11-10T03:00:00"));
        assert_eq!(j[1].where_or_why, "(Priority)");
    }

    #[test]
    fn pending() {
        let t = "gpu-h200 gres/gpu:h200:2 N/A 1 Priority\n\
                 gpu-h200 gres/gpu:h200:2 N/A 1 Dependency\n\
                 gpu-l40s N/A gres/gpu:1 2 Resources\n";
        let p = parse_pending(t);
        assert_eq!(p.iter().map(|x| x.gpus).sum::<u32>(), 4);
    }
}
