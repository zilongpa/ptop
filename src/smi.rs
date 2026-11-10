//! Standalone GPU servers without Slurm: parse nvidia-smi / ps / free output.

use crate::slurm::{Node, NodeStatus, SEP};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// A GPU counts as busy if it has a compute process, uses more memory than this, or is this utilized.
const BUSY_MEM_MB: u64 = 1024;
const BUSY_UTIL: u32 = 10;

#[derive(Clone, Debug, Default, Serialize)]
pub struct GpuDev {
    pub index: u32,
    pub name: String,
    pub mem_used_mb: u64,
    pub mem_total_mb: u64,
    pub util: u32,
    pub users: Vec<String>,
    pub busy: bool,
}

pub fn command() -> String {
    format!(
        "hostname -s; echo {SEP}; \
         nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; \
         echo {SEP}; nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits; \
         echo {SEP}; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); \
         do ps -o pid=,user= -p \"$p\"; done; \
         echo {SEP}; nproc; cat /proc/loadavg; free -m"
    )
}

/// "NVIDIA H100 80GB HBM3" -> "h100", "NVIDIA RTX A6000" -> "a6000", unknown -> slug of the name.
pub fn gpu_type(name: &str) -> String {
    let s = name.to_lowercase();
    const KNOWN: &[(&str, &str)] = &[
        ("rtx pro 6000", "rtx_pro_6000_blackwell"),
        ("rtx 6000 ada", "rtx_6000_ada"),
        ("quadro rtx 6000", "rtx_6000"),
        ("rtx a6000", "a6000"),
        ("rtx a5000", "a5000"),
        ("rtx a4000", "a4000"),
        ("h200", "h200"),
        ("h100", "h100"),
        ("gh200", "gh200"),
        ("b200", "b200"),
        ("a100", "a100"),
        ("a40", "a40"),
        ("a30", "a30"),
        ("a10", "a10"),
        ("l40s", "l40s"),
        ("l40", "l40"),
        ("l4", "l4"),
        ("v100", "v100"),
        ("p100", "p100"),
        ("t4", "t4"),
        ("titan rtx", "titan_rtx"),
        ("titan v", "titan_v"),
        ("titan x", "titan_x"),
    ];
    for (pat, out) in KNOWN {
        if s.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| w == *pat) || (pat.contains(' ') && s.contains(pat)) {
            return out.to_string();
        }
    }
    // consumer cards: keep the model number, e.g. "geforce rtx 4090" -> "rtx_4090"
    let s = s.replace("nvidia", "").replace("geforce", "").replace("tesla", "");
    let slug: String = s.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect::<Vec<_>>().join("_");
    if slug.is_empty() { "gpu".into() } else { slug }
}

/// Turn the output of [`command`] into one node with per-GPU detail.
pub fn parse(out: &str) -> Option<Node> {
    let parts: Vec<&str> = out.split(SEP).collect();
    if parts.len() < 5 {
        return None;
    }
    let host = parts[0].trim().to_string();

    let mut uuid_idx: HashMap<String, usize> = HashMap::new();
    let mut gpus: Vec<GpuDev> = Vec::new();
    for line in parts[1].lines() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 6 {
            continue;
        }
        let Ok(index) = f[0].parse::<u32>() else { continue };
        uuid_idx.insert(f[1].to_string(), gpus.len());
        gpus.push(GpuDev {
            index,
            name: f[2].to_string(),
            mem_used_mb: f[3].parse().unwrap_or(0),
            mem_total_mb: f[4].parse().unwrap_or(0),
            util: f[5].parse().unwrap_or(0),
            ..Default::default()
        });
    }
    if gpus.is_empty() {
        return None;
    }

    let owner: HashMap<&str, &str> = parts[3]
        .lines()
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            Some((w.next()?, w.next()?))
        })
        .collect();
    for line in parts[2].lines() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 2 {
            continue;
        }
        if let Some(&i) = uuid_idx.get(f[0]) {
            let user = owner.get(f[1]).copied().unwrap_or("?").to_string();
            if !gpus[i].users.contains(&user) {
                gpus[i].users.push(user);
            }
        }
    }
    for g in gpus.iter_mut() {
        g.busy = !g.users.is_empty() || g.mem_used_mb > BUSY_MEM_MB || g.util > BUSY_UTIL;
    }

    // nproc / loadavg / free -m
    let sys: Vec<&str> = parts[4].lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let cpu_total: u32 = sys.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let load: f64 = sys.get(1).and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let mem = sys.iter().find(|l| l.starts_with("Mem:")).map(|l| l.split_whitespace().collect::<Vec<_>>());
    let mem_total_mb: u64 = mem.as_ref().and_then(|m| m.get(1)).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mem_avail_mb: u64 = mem.as_ref().and_then(|m| m.get(6)).and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut by_type: BTreeMap<String, u32> = BTreeMap::new();
    for g in &gpus {
        *by_type.entry(gpu_type(&g.name)).or_default() += 1;
    }
    let ty = by_type.into_iter().max_by_key(|(_, n)| *n).map(|(t, _)| t).unwrap_or_default();
    let total = gpus.len() as u32;
    let used = gpus.iter().filter(|g| g.busy).count() as u32;
    let free = total - used;
    Some(Node {
        name: host,
        partitions: vec!["(direct)".into()],
        state: match used {
            0 => "idle".into(),
            u if u == total => "allocated".into(),
            _ => "mixed".into(),
        },
        gpu_type: ty,
        total,
        used,
        free,
        usable: free,
        status: NodeStatus::Ok,
        cpu_idle: (cpu_total as f64 - load).max(0.0).round() as u32,
        cpu_total,
        mem_free_gb: mem_avail_mb / 1024,
        mem_total_gb: mem_total_mb / 1024,
        releases: Vec::new(),
        gpus,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(gpu_type("NVIDIA H100 80GB HBM3"), "h100");
        assert_eq!(gpu_type("NVIDIA RTX A6000"), "a6000");
        assert_eq!(gpu_type("Tesla V100-SXM2-32GB"), "v100");
        assert_eq!(gpu_type("NVIDIA A100-SXM4-80GB"), "a100");
        assert_eq!(gpu_type("NVIDIA L40S"), "l40s");
        assert_eq!(gpu_type("NVIDIA GeForce RTX 4090"), "rtx_4090");
        assert_eq!(gpu_type("NVIDIA RTX PRO 6000 Blackwell Server Edition"), "rtx_pro_6000_blackwell");
    }

    #[test]
    fn parse_box() {
        let out = format!(
            "fry\n{SEP}\n\
             0, GPU-aaa, NVIDIA RTX A6000, 30000, 49140, 97\n\
             1, GPU-bbb, NVIDIA RTX A6000, 3, 49140, 0\n\
             2, GPU-ccc, NVIDIA RTX A6000, 2000, 49140, 0\n\
             {SEP}\nGPU-aaa, 1234\nGPU-aaa, 1235\n{SEP}\n 1234 alice\n 1235 bob\n{SEP}\n\
             64\n12.50 10.0 9.0 3/900 12345\n\
             total used free shared buff/cache available\n\
             Mem: 515000 100000 300000 100 115000 400000\nSwap: 0 0 0\n"
        );
        let n = parse(&out).unwrap();
        assert_eq!(n.name, "fry");
        assert_eq!(n.gpu_type, "a6000");
        assert_eq!((n.total, n.used, n.usable), (3, 2, 1));
        assert_eq!(n.gpus[0].users, ["alice", "bob"]);
        assert!(n.gpus[2].busy, "2 GB of memory in use counts as busy");
        assert_eq!(n.cpu_idle, 52);
        assert_eq!(n.mem_free_gb, 390);
    }
}
