//! ptop: an htop-style terminal view of free GPUs on PACE / Slurm clusters and
//! standalone nvidia-smi servers. Read-only: it only runs sinfo / squeue / nvidia-smi over ssh.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Local, Utc};
use clap::{Parser, Subcommand};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use collect::{ClusterState, Clusters, Kind, SshCfg};
use slurm::{Node, NodeStatus};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, BorderType, Cell, Clear, Padding, Paragraph, Row, Table, TableState, Wrap},
    DefaultTerminal, Frame,
};
use std::{path::PathBuf, time::Duration};

mod collect;
mod config;
mod dump;
mod slurm;
mod smi;
mod webui;

/// Clusters watched when none is configured: PACE ICE and Phoenix.
const DEFAULT_CLUSTERS: &[&str] = &["ice=login-ice.pace.gatech.edu", "phoenix=login-phoenix.pace.gatech.edu"];

/// htop-style monitor for free GPUs. Read-only: runs sinfo / squeue / nvidia-smi over ssh.
#[derive(Parser, Debug)]
#[command(version, after_help = CONFIG_HELP)]
struct Args {
    /// Login name on the clusters (or `user` in the config file, or $PTOP_USER)
    #[arg(short, long, env = "PTOP_USER", global = true)]
    user: Option<String>,

    /// Cluster as NAME=[USER@]HOST[:PART1,PART2]; repeatable; replaces the config file list
    #[arg(short, long = "cluster", value_name = "SPEC", global = true)]
    clusters: Vec<String>,

    /// Seconds between polls
    #[arg(short, long, global = true, value_parser = clap::value_parser!(u64).range(20..))]
    interval: Option<u64>,

    /// Config file (default ~/.config/ptop/config.toml, on Windows %APPDATA%\\ptop\\config.toml)
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Extra ssh argument, e.g. --ssh-arg=-i --ssh-arg=/path/key
    #[arg(long = "ssh-arg", allow_hyphen_values = true, value_name = "ARG", global = true)]
    ssh_args: Vec<String>,

    #[command(subcommand)]
    command: Option<Cmd>,

    /// Poll every cluster once, render one frame at COLSxROWS to stdout and exit
    #[arg(long, value_name = "COLSxROWS", hide = true)]
    dump: Option<String>,

    /// With --dump: press these keys first, e.g. "v" or "n4<Enter>"
    #[arg(long, hide = true, default_value = "")]
    keys: String,

    /// With --dump: also write a colored HTML render to this file
    #[arg(long, hide = true)]
    html: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Create a starter config file and print where it is
    Init {
        /// Print the template instead of writing the file
        #[arg(long)]
        print: bool,
    },
    /// Serve the same data as a web page instead of the terminal UI
    Webui {
        /// Listen address; use 0.0.0.0:8787 to share it on your network
        #[arg(short, long, default_value = "127.0.0.1:8787")]
        bind: std::net::SocketAddr,
    },
}

const CONFIG_HELP: &str = "\
Config file (TOML): ~/.config/ptop/config.toml, on Windows %APPDATA%\\ptop\\config.toml

  user = \"gburdell3\"
  interval = 60

  [[cluster]]
  name = \"ice\"
  host = \"login-ice.pace.gatech.edu\"

  [[cluster]]
  name = \"lab\"
  host = \"gpu-box.example.edu\"
  password_command = \"pass show lab\"   # or: password = \"...\"

Run `ptop init` to create it. Without any cluster the default is PACE ICE + Phoenix.";

// ---------------------------------------------------------------- UI state

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Nodes,
    Jobs,
    Queue,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortKey {
    Free,
    Ready,
    Type,
    Name,
    Cpu,
    Mem,
}

impl SortKey {
    const ALL: [SortKey; 6] = [SortKey::Free, SortKey::Ready, SortKey::Type, SortKey::Name, SortKey::Cpu, SortKey::Mem];
    fn label(self) -> &'static str {
        match self {
            SortKey::Free => "FREE",
            SortKey::Ready => "READY",
            SortKey::Type => "TYPE",
            SortKey::Name => "NODE",
            SortKey::Cpu => "CPU",
            SortKey::Mem => "MEM",
        }
    }
}

enum Input {
    None,
    Search(String),
    Need(String),
}

enum Popup {
    None,
    Help,
    Detail(String, String), // cluster, node
    Errors(u16),            // scroll offset
}

struct Ui {
    tab: usize, // 0 = all clusters
    view: View,
    sort: SortKey,
    reverse: bool,
    type_filter: Option<String>,
    search: String,
    need: u32,
    only_free: bool,
    input: Input,
    popup: Popup,
    table: TableState,
    interval: u64,
    user: String,
    flash: Option<(String, std::time::Instant)>,
}

/// One row of the nodes table, flattened across clusters.
struct NodeRow<'a> {
    cluster: &'a str,
    node: &'a Node,
    jobs_visible: bool,
    /// When the node will have `need` (at least 1) GPUs a new job can take.
    ready: Option<DateTime<Utc>>,
}

fn ready_at(n: &Node, need: u32, jobs_visible: bool, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let need = need.max(1);
    if n.usable >= need {
        return Some(now);
    }
    if n.status == NodeStatus::Offline || n.total < need || !jobs_visible {
        return None;
    }
    let mut avail = n.usable;
    for r in &n.releases {
        avail += r.gpus;
        if avail >= need {
            return Some(r.at);
        }
    }
    None
}

fn human_dur(d: chrono::Duration) -> String {
    let m = d.num_minutes().max(0);
    match m {
        0 => "<1m".into(),
        m if m < 60 => format!("{m}m"),
        m if m < 48 * 60 => format!("{}h{:02}m", m / 60, m % 60),
        m => format!("{}d{}h", m / 1440, (m % 1440) / 60),
    }
}

fn ago(t: Option<DateTime<Utc>>, now: DateTime<Utc>) -> String {
    t.map(|t| human_dur(now - t)).unwrap_or_else(|| "-".into())
}

impl Ui {
    fn scope<'a>(&self, states: &'a [ClusterState]) -> Vec<&'a ClusterState> {
        match self.tab {
            0 => states.iter().collect(),
            i => states.get(i - 1).into_iter().collect(),
        }
    }

    fn node_rows<'a>(&self, states: &'a [ClusterState], now: DateTime<Utc>) -> Vec<NodeRow<'a>> {
        let q = self.search.to_lowercase();
        let mut rows: Vec<NodeRow> = Vec::new();
        for s in self.scope(states) {
            for n in &s.nodes {
                if self.type_filter.as_ref().is_some_and(|t| &n.gpu_type != t) {
                    continue;
                }
                if self.only_free && n.usable == 0 {
                    continue;
                }
                if self.need > 0 && (n.total < self.need || n.status == NodeStatus::Offline) {
                    continue;
                }
                if !q.is_empty() {
                    let hay = format!("{} {} {} {}", s.name, n.name, n.gpu_type, n.partitions.join(" ")).to_lowercase();
                    if !hay.contains(&q) {
                        continue;
                    }
                }
                rows.push(NodeRow {
                    cluster: &s.name,
                    node: n,
                    jobs_visible: s.jobs_visible,
                    ready: ready_at(n, self.need, s.jobs_visible, now),
                });
            }
        }
        let far = now + chrono::Duration::days(3650);
        rows.sort_by(|a, b| {
            let o = match self.sort {
                SortKey::Free => {
                    let off = |r: &NodeRow| r.node.status == NodeStatus::Offline;
                    b.node.usable.cmp(&a.node.usable).then(off(a).cmp(&off(b))).then(b.node.free.cmp(&a.node.free))
                }
                SortKey::Ready => a.ready.unwrap_or(far).cmp(&b.ready.unwrap_or(far)),
                SortKey::Type => a.node.gpu_type.cmp(&b.node.gpu_type),
                SortKey::Name => a.node.name.cmp(&b.node.name),
                SortKey::Cpu => b.node.cpu_idle.cmp(&a.node.cpu_idle),
                SortKey::Mem => b.node.mem_free_gb.cmp(&a.node.mem_free_gb),
            };
            let o = o.then(a.cluster.cmp(b.cluster)).then(a.node.gpu_type.cmp(&b.node.gpu_type)).then(a.node.name.cmp(&b.node.name));
            if self.reverse { o.reverse() } else { o }
        });
        rows
    }

    fn row_count(&self, states: &[ClusterState], now: DateTime<Utc>) -> usize {
        match self.view {
            View::Nodes => self.node_rows(states, now).len(),
            View::Jobs => self.scope(states).iter().map(|s| s.my_jobs.len()).sum(),
            View::Queue => self.scope(states).iter().map(|s| s.pending.len()).sum(),
        }
    }

    fn all_types(&self, states: &[ClusterState]) -> Vec<String> {
        let mut t: Vec<String> = self.scope(states).iter().flat_map(|s| s.nodes.iter().map(|n| n.gpu_type.clone())).collect();
        t.sort();
        t.dedup();
        t
    }

    fn flash(&mut self, msg: impl Into<String>) {
        self.flash = Some((msg.into(), std::time::Instant::now()));
    }
}

// ---------------------------------------------------------------- drawing

// xterm-256 palette indices: render the same in Terminal.app, iTerm2, tmux and over ssh.
const ACCENT: Color = Color::Indexed(75);
const C_FREE: Color = Color::Indexed(114);
const C_USED: Color = Color::Indexed(131);
const C_HELD: Color = Color::Indexed(179);
const C_OFF: Color = Color::Indexed(240);
const C_DIM: Color = Color::Indexed(245);
const C_TEXT: Color = Color::Indexed(252);
const C_BORDER: Color = Color::Indexed(239);
const C_SEL: Color = Color::Indexed(236);
const C_ERR: Color = Color::Indexed(203);
const C_WARN: Color = Color::Indexed(221);

fn dim() -> Style {
    Style::new().fg(C_DIM)
}

fn pill(text: &str, active: bool) -> Span<'static> {
    let st = if active { Style::new().fg(Color::Black).bg(ACCENT).bold() } else { Style::new().fg(C_TEXT) };
    Span::styled(format!(" {text} "), st)
}

fn truncate(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_string()
    } else {
        s.chars().take(w.saturating_sub(1)).collect::<String>() + "~"
    }
}

/// Split `w` cells between [used, held, offline, free] in proportion, keeping every
/// non-zero category visible so a single free GPU never rounds away.
fn bar_cells(counts: [u32; 4], w: usize) -> [usize; 4] {
    let total: u32 = counts.iter().sum();
    let mut cells = [0usize; 4];
    if total == 0 || w == 0 {
        return cells;
    }
    for i in 0..4 {
        cells[i] = counts[i] as usize * w / total as usize;
        if counts[i] > 0 && cells[i] == 0 {
            cells[i] = 1;
        }
    }
    loop {
        let sum: usize = cells.iter().sum();
        let big = (0..4).max_by_key(|&i| cells[i]).unwrap();
        match sum.cmp(&w) {
            std::cmp::Ordering::Equal => break,
            std::cmp::Ordering::Greater if cells[big] > 1 => cells[big] -= 1,
            std::cmp::Ordering::Greater => break,
            std::cmp::Ordering::Less => cells[big] += 1,
        }
    }
    cells
}

struct Meter {
    label: String,
    usable: u32,
    used: u32,
    held: u32,
    offline: u32,
    total: u32,
}

fn meter_line(m: &Meter, label_w: usize, width: usize) -> Line<'static> {
    let nums_w = 8;
    let bar_w = width.saturating_sub(label_w + nums_w + 3).max(4);
    let c = bar_cells([m.used, m.held, m.offline, m.usable], bar_w);
    let free_st = if m.usable > 0 { Style::new().fg(C_FREE).bold() } else { dim() };
    Line::from(vec![
        Span::styled(format!(" {:<label_w$} ", truncate(&m.label, label_w)), Style::new().fg(C_TEXT)),
        Span::styled("━".repeat(c[0]), Style::new().fg(C_USED)),
        Span::styled("━".repeat(c[1]), Style::new().fg(C_HELD)),
        Span::styled("━".repeat(c[2]), Style::new().fg(C_OFF)),
        Span::styled("━".repeat(c[3]), Style::new().fg(C_FREE)),
        Span::styled(format!(" {:>3}", m.usable), free_st),
        Span::styled(format!("/{:<3}", m.total), dim()),
    ])
}

struct Panel {
    name: String,
    usable: u32,
    total: u32,
    right: String,
    border: Color,
    error: Option<String>,
    placeholder: Option<&'static str>,
    meters: Vec<Meter>,
}

fn cluster_panel(ui: &Ui, s: &ClusterState, now: DateTime<Utc>) -> Panel {
    let meters: Vec<Meter> = s
        .summary
        .iter()
        .filter(|t| ui.type_filter.as_ref().is_none_or(|x| x == &t.gpu_type))
        .map(|t| Meter { label: t.gpu_type.clone(), usable: t.usable, used: t.used, held: t.blocked, offline: t.offline, total: t.total })
        .collect();
    let (right, border) = if s.last_attempt.is_none() {
        ("connecting".to_string(), C_WARN)
    } else if s.ok {
        (format!("{} ago", ago(s.last_success, now)), C_BORDER)
    } else if s.last_success.is_some() {
        (format!("stale, last ok {} ago", ago(s.last_success, now)), C_ERR)
    } else {
        ("unreachable".to_string(), C_ERR)
    };
    Panel {
        name: s.name.clone(),
        usable: meters.iter().map(|m| m.usable).sum(),
        total: meters.iter().map(|m| m.total).sum(),
        right,
        border,
        error: s.error.clone(),
        placeholder: if !meters.is_empty() {
            None
        } else if s.last_attempt.is_none() {
            Some("waiting for the first ssh poll...")
        } else {
            Some("no data")
        },
        meters,
    }
}

fn panel_cols(p: &Panel, width: u16) -> usize {
    if width >= 110 && p.meters.len() > 3 { 2 } else { 1 }
}

fn draw_panel(f: &mut Frame, p: &Panel, area: Rect) {
    let free_st = if p.usable > 0 { Style::new().fg(C_FREE).bold() } else { dim() };
    let title = Line::from(vec![
        Span::styled(format!(" {} ", p.name), Style::new().fg(ACCENT).bold()),
        Span::styled(format!("{} free", p.usable), free_st),
        Span::styled(format!(" of {} ", p.total), dim()),
    ]);
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(p.border))
        .title_top(title)
        .title_top(Line::from(Span::styled(format!(" {} ", p.right), dim())).right_aligned());
    if let Some(e) = &p.error {
        let one_line = e.split_whitespace().collect::<Vec<_>>().join(" ");
        let e = truncate(&one_line, area.width.saturating_sub(16) as usize);
        block = block
            .title_bottom(Line::from(Span::styled(format!(" {e} "), Style::new().fg(C_ERR))))
            .title_bottom(Line::from(Span::styled(" e: full ", dim())).right_aligned());
    }
    let inner = block.inner(area);
    f.render_widget(block, area);
    if let Some(msg) = p.placeholder {
        f.render_widget(Paragraph::new(format!(" {msg}")).style(Style::new().fg(C_WARN)), inner);
        return;
    }
    let cols = panel_cols(p, area.width);
    let rows = inner.height as usize;
    let label_w = p.meters.iter().map(|m| m.label.chars().count()).max().unwrap_or(4).min(24);
    let areas = Layout::horizontal(vec![Constraint::Ratio(1, cols as u32); cols]).spacing(1).split(inner);
    for (ci, a) in areas.iter().enumerate() {
        let lines: Vec<Line> = p.meters.iter().skip(ci * rows).take(rows).map(|m| meter_line(m, label_w, a.width as usize)).collect();
        f.render_widget(Paragraph::new(lines), *a);
    }
}

fn draw(f: &mut Frame, ui: &mut Ui, states: &[ClusterState], now: DateTime<Utc>) {
    let area = f.area();
    let scope = ui.scope(states);

    // one panel per cluster while they fit side by side, otherwise a single merged panel
    let mut panels: Vec<Panel> = scope.iter().map(|s| cluster_panel(ui, s, now)).collect();
    if panels.len() > (area.width as usize / 46).max(1) {
        let mut merged = Panel {
            name: "all clusters".into(),
            usable: panels.iter().map(|p| p.usable).sum(),
            total: panels.iter().map(|p| p.total).sum(),
            right: String::new(),
            border: C_BORDER,
            error: None,
            placeholder: None,
            meters: Vec::new(),
        };
        for p in panels.drain(..) {
            for mut m in p.meters {
                m.label = format!("{} {}", p.name, m.label);
                merged.meters.push(m);
            }
        }
        merged.meters.sort_by(|a, b| b.usable.cmp(&a.usable).then(b.total.cmp(&a.total)));
        if merged.meters.is_empty() {
            merged.placeholder = Some("waiting for the first ssh poll...");
        }
        panels.push(merged);
    }
    let each_w = area.width / panels.len().max(1) as u16;
    let want = panels.iter().map(|p| p.meters.len().div_ceil(panel_cols(p, each_w)).max(1)).max().unwrap_or(1) as u16 + 2;
    let panel_h = want.min(area.height.saturating_sub(12).max(3));

    let [header, panel_area, info, main, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(panel_h),
        Constraint::Length(1),
        Constraint::Min(4),
        Constraint::Length(1),
    ])
    .areas(area);

    // header: cluster tabs on the left, session info on the right
    let right = format!("{}  every {}s  {} ", ui.user, ui.interval, Local::now().format("%H:%M:%S"));
    let [hl, hr] = Layout::horizontal([Constraint::Min(10), Constraint::Length(right.len() as u16)]).areas(header);
    let mut h = vec![Span::styled(" ptop ", Style::new().fg(Color::Black).bg(C_FREE).bold()), Span::raw(" ")];
    h.push(pill("0 all", ui.tab == 0));
    for (i, s) in states.iter().enumerate() {
        let dot = if s.last_attempt.is_none() { C_WARN } else if s.ok { C_FREE } else { C_ERR };
        let active = ui.tab == i + 1;
        if active {
            h.push(pill(&format!("{} {}", i + 1, s.name), true));
        } else {
            h.push(Span::styled(format!(" {} ", i + 1), dim()));
            h.push(Span::styled(format!("{} ", s.name), Style::new().fg(C_TEXT)));
        }
        h.push(Span::styled("● ", Style::new().fg(dot)));
    }
    f.render_widget(Paragraph::new(Line::from(h)), hl);
    f.render_widget(Paragraph::new(Span::styled(right, dim())), hr);

    // panels
    let pa = Layout::horizontal(vec![Constraint::Ratio(1, panels.len() as u32); panels.len()]).split(panel_area);
    for (p, a) in panels.iter().zip(pa.iter()) {
        draw_panel(f, p, *a);
    }

    // info line: totals and active filters on the left, legend (or a message) on the right
    let usable: u32 = panels.iter().map(|p| p.usable).sum();
    let total: u32 = panels.iter().map(|p| p.total).sum();
    let mut sl = vec![
        Span::styled(format!(" {usable}"), if usable > 0 { Style::new().fg(C_FREE).bold() } else { Style::new().fg(C_TEXT).bold() }),
        Span::styled(format!(" free now of {total} GPUs  "), dim()),
    ];
    let chip = |s: String| Span::styled(format!(" {s} "), Style::new().fg(Color::Black).bg(C_HELD));
    let mut chips: Vec<String> = Vec::new();
    if let Some(t) = &ui.type_filter {
        chips.push(format!("type: {t}"));
    }
    if !ui.search.is_empty() {
        chips.push(format!("search: {}", ui.search));
    }
    if ui.need > 0 {
        chips.push(format!("need: {} per node", ui.need));
    }
    if ui.only_free {
        chips.push("free only".into());
    }
    for c in chips {
        sl.push(chip(c));
        sl.push(Span::raw(" "));
    }
    let legend: Line = match &ui.flash {
        Some((m, t)) if t.elapsed() < Duration::from_secs(4) => Line::from(Span::styled(format!("{m} "), Style::new().fg(C_WARN))),
        _ => Line::from(vec![
            Span::styled("━ ", Style::new().fg(C_FREE)),
            Span::styled("free  ", dim()),
            Span::styled("━ ", Style::new().fg(C_USED)),
            Span::styled("in use  ", dim()),
            Span::styled("━ ", Style::new().fg(C_HELD)),
            Span::styled("held  ", dim()),
            Span::styled("━ ", Style::new().fg(C_OFF)),
            Span::styled("offline ", dim()),
        ]),
    };
    let [il, ir] = Layout::horizontal([Constraint::Min(10), Constraint::Length(legend.width() as u16)]).areas(info);
    f.render_widget(Paragraph::new(Line::from(sl)), il);
    f.render_widget(Paragraph::new(legend), ir);

    match ui.view {
        View::Nodes => draw_nodes(f, ui, states, now, main),
        View::Jobs => draw_jobs(f, ui, states, main),
        View::Queue => draw_queue(f, ui, states, main),
    }

    // footer: a prompt while typing, otherwise the key hints
    let line = match &ui.input {
        Input::Search(s) => Line::from(vec![
            Span::styled(" search ", Style::new().fg(Color::Black).bg(C_HELD).bold()),
            Span::raw(format!(" {s}_")),
            Span::styled("   Enter keep   Esc clear", dim()),
        ]),
        Input::Need(s) => Line::from(vec![
            Span::styled(" GPUs needed on one node ", Style::new().fg(Color::Black).bg(C_HELD).bold()),
            Span::raw(format!(" {s}_")),
            Span::styled("   Enter apply   0 or empty to clear   Esc cancel", dim()),
        ]),
        Input::None => {
            let keys = [
                ("?", "help"),
                ("Tab", "cluster"),
                ("v", "view"),
                ("/", "search"),
                ("t", "type"),
                ("s", "sort"),
                ("f", "free only"),
                ("n", "need N"),
                ("Enter", "detail"),
                ("r", "refresh"),
                ("q", "quit"),
            ];
            let mut k = vec![Span::raw(" ")];
            if states.iter().any(|s| s.error.is_some()) {
                k.push(Span::styled("e", Style::new().fg(C_ERR).bold()));
                k.push(Span::styled(" errors  ", Style::new().fg(C_ERR)));
            }
            for (key, label) in keys {
                k.push(Span::styled(key, Style::new().fg(ACCENT).bold()));
                k.push(Span::styled(format!(" {label}  "), dim()));
            }
            Line::from(k)
        }
    };
    f.render_widget(Paragraph::new(line), footer);

    match &ui.popup {
        Popup::None => {}
        Popup::Help => draw_help(f, area),
        Popup::Errors(scroll) => draw_errors(f, area, states, *scroll, now),
        Popup::Detail(c, n) => {
            if let Some(s) = states.iter().find(|s| &s.name == c) {
                if let Some(node) = s.nodes.iter().find(|x| &x.name == n) {
                    draw_detail(f, area, s, node, ui.need, now);
                }
            }
        }
    }
}

/// The frame around the main table; its title doubles as the view switcher.
fn table_block(ui: &Ui, count: usize, right: String) -> Block<'static> {
    let mut t = vec![Span::raw(" ")];
    for (v, name) in [(View::Nodes, "Nodes"), (View::Jobs, "My jobs"), (View::Queue, "Queue")] {
        if v == ui.view {
            t.push(pill(&format!("{name} {count}"), true));
        } else {
            t.push(Span::styled(format!(" {name} "), dim()));
        }
        t.push(Span::raw(" "));
    }
    let mut b = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(C_BORDER))
        .padding(Padding::horizontal(1))
        .title_top(Line::from(t));
    if !right.is_empty() {
        b = b.title_top(Line::from(Span::styled(format!(" {right} "), dim())).right_aligned());
    }
    b
}

fn header_row(cells: &[&str], sort_label: Option<&str>, reverse: bool) -> Row<'static> {
    Row::new(cells.iter().map(|c| {
        if sort_label.is_some_and(|l| l == *c) {
            Cell::from(format!("{c} {}", if reverse { "▲" } else { "▼" })).style(Style::new().fg(ACCENT).bold())
        } else {
            Cell::from(c.to_string()).style(dim().bold())
        }
    }))
}

fn sel_style() -> Style {
    Style::new().bg(C_SEL).add_modifier(Modifier::BOLD)
}

fn draw_nodes(f: &mut Frame, ui: &mut Ui, states: &[ClusterState], now: DateTime<Utc>, area: Rect) {
    let rows = ui.node_rows(states, now);
    let show_cluster = ui.tab == 0;
    let ready_label = if ui.need > 1 { format!("{}+ FREE IN", ui.need) } else { "READY".to_string() };
    let mut head = vec![];
    if show_cluster {
        head.push("CLUSTER");
    }
    head.extend(["NODE", "TYPE", "FREE", "GPUS", "STATE", "CPU", "MEM GB", ready_label.as_str(), "PARTITIONS"]);
    let sort_label = match ui.sort {
        SortKey::Ready => ready_label.clone(),
        SortKey::Cpu => "CPU".into(),
        SortKey::Mem => "MEM GB".into(),
        k => k.label().to_string(),
    };
    let body: Vec<Row> = rows
        .iter()
        .map(|r| {
            let n = r.node;
            let off = n.status == NodeStatus::Offline;
            let (status, sst) = match n.status {
                NodeStatus::Offline => (n.state.clone(), Style::new().fg(C_OFF)),
                NodeStatus::Reserved => ("reserved".into(), Style::new().fg(C_HELD)),
                NodeStatus::CpuFull => ("cpu full".into(), Style::new().fg(C_HELD)),
                NodeStatus::MemFull => ("mem full".into(), Style::new().fg(C_HELD)),
                NodeStatus::Ok => {
                    let s = n.state.trim_end_matches('-').to_string();
                    let st = match s.as_str() {
                        "idle" => Style::new().fg(C_FREE),
                        "allocated" => dim(),
                        _ => Style::new().fg(C_TEXT),
                    };
                    (s, st)
                }
            };
            let gpus: Line = Line::from(vec![
                Span::styled("■".repeat(n.used as usize), Style::new().fg(C_USED)),
                Span::styled("■".repeat((n.free - n.usable) as usize), Style::new().fg(if off { C_OFF } else { C_HELD })),
                Span::styled("■".repeat(n.usable as usize), Style::new().fg(C_FREE)),
            ]);
            let ready = match r.ready {
                Some(t) if t <= now => Span::styled("now", Style::new().fg(C_FREE).bold()),
                Some(t) => Span::styled(human_dur(t - now), Style::new().fg(C_TEXT)),
                None if !r.jobs_visible && !off => Span::styled("?", dim()),
                None => Span::styled("-", dim()),
            };
            let free: Line = Line::from(vec![
                Span::styled(format!("{:>2}", n.usable), if n.usable > 0 { Style::new().fg(C_FREE).bold() } else { dim() }),
                Span::styled(format!("/{:<2}", n.total), Style::new().fg(if off { C_OFF } else { C_DIM })),
            ]);
            let frac = |a: u64, b: u64, w: usize| -> Line<'static> {
                Line::from(vec![
                    Span::styled(
                        format!("{a:>w$}"),
                        Style::new().fg(if off { C_OFF } else if a == 0 { C_HELD } else { C_TEXT }),
                    ),
                    Span::styled(format!("/{b}"), Style::new().fg(if off { C_OFF } else { C_DIM })),
                ])
            };
            let mut cells: Vec<Cell> = Vec::new();
            if show_cluster {
                cells.push(Cell::from(r.cluster.to_string()).style(Style::new().fg(ACCENT)));
            }
            cells.extend([
                Cell::from(n.name.clone()).style(Style::new().fg(if off { C_OFF } else { C_TEXT })),
                Cell::from(n.gpu_type.clone()).style(Style::new().fg(if off { C_OFF } else { C_TEXT })),
                Cell::from(free),
                Cell::from(gpus),
                Cell::from(status).style(sst),
                Cell::from(frac(n.cpu_idle as u64, n.cpu_total as u64, 3)),
                Cell::from(frac(n.mem_free_gb, n.mem_total_gb, 4)),
                Cell::from(ready),
                Cell::from(n.partitions.join(" ")).style(dim()),
            ]);
            let row = Row::new(cells);
            if off { row.style(Style::new().fg(C_OFF)) } else { row }
        })
        .collect();
    let max_gpus = rows.iter().map(|r| r.node.total).max().unwrap_or(8).clamp(4, 16) as u16;
    let mut widths = vec![];
    if show_cluster {
        widths.push(Constraint::Length(rows.iter().map(|r| r.cluster.len()).max().unwrap_or(7).max(7) as u16));
    }
    widths.extend([
        Constraint::Length(rows.iter().map(|r| r.node.name.len()).max().unwrap_or(10).max(4) as u16),
        Constraint::Length(rows.iter().map(|r| r.node.gpu_type.len()).max().unwrap_or(6).clamp(4, 24) as u16),
        Constraint::Length(6),
        Constraint::Length(max_gpus),
        Constraint::Length(9),
        Constraint::Length(7),
        Constraint::Length(9),
        Constraint::Length(ready_label.len().max(8) as u16 + 2),
        Constraint::Min(10),
    ]);
    let right = format!("sorted by {}{}", sort_label, if ui.reverse { ", reversed" } else { "" });
    let table = Table::new(body, widths)
        .column_spacing(2)
        .header(header_row(&head, Some(&sort_label), ui.reverse))
        .row_highlight_style(sel_style())
        .block(table_block(ui, rows.len(), right));
    f.render_stateful_widget(table, area, &mut ui.table);
}

fn draw_jobs(f: &mut Frame, ui: &mut Ui, states: &[ClusterState], area: Rect) {
    let scope = ui.scope(states);
    let show_cluster = ui.tab == 0;
    let text = Style::new().fg(C_TEXT);
    let mut body = Vec::new();
    for s in &scope {
        for j in &s.my_jobs {
            let st = match j.state.as_str() {
                "R" => Style::new().fg(C_FREE).bold(),
                "PD" => Style::new().fg(C_HELD),
                _ => dim(),
            };
            let gpus = match (&j.gpu_type, j.gpus) {
                (_, 0) => "-".to_string(),
                (Some(t), n) => format!("{n} x {t}"),
                (None, n) => n.to_string(),
            };
            // "2026-11-10T03:00:00" -> "11-10 03:00"
            let mut place = j.where_or_why.clone();
            if let Some(est) = j.est_start.as_deref().and_then(|e| e.get(5..16)) {
                place += &format!("  est. start {}", est.replace('T', " "));
            }
            let mut cells = Vec::new();
            if show_cluster {
                cells.push(Cell::from(s.name.clone()).style(Style::new().fg(ACCENT)));
            }
            cells.extend([
                Cell::from(j.id.clone()).style(text),
                Cell::from(j.state.clone()).style(st),
                Cell::from(j.name.clone()).style(text),
                Cell::from(j.partition.clone()).style(text),
                Cell::from(gpus).style(text),
                Cell::from(j.time_used.clone()).style(text),
                Cell::from(j.time_limit.clone()).style(dim()),
                Cell::from(place).style(text),
            ]);
            body.push(Row::new(cells));
        }
    }
    let smi_only = !scope.is_empty() && scope.iter().all(|s| s.kind == Kind::Smi);
    let right = if smi_only { "standalone hosts have no Slurm jobs".to_string() } else { format!("squeue -u {}", ui.user) };
    let n = body.len();
    let mut head = vec![];
    let mut widths = vec![];
    if show_cluster {
        head.push("CLUSTER");
        widths.push(Constraint::Length(8));
    }
    head.extend(["JOBID", "ST", "NAME", "PARTITION", "GPUS", "ELAPSED", "LIMIT", "NODE / REASON"]);
    widths.extend([
        Constraint::Length(9),
        Constraint::Length(2),
        Constraint::Fill(2),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Fill(3),
    ]);
    let table = Table::new(body, widths)
        .column_spacing(2)
        .header(header_row(&head, None, false))
        .row_highlight_style(sel_style())
        .block(table_block(ui, n, right));
    f.render_stateful_widget(table, area, &mut ui.table);
}

fn draw_queue(f: &mut Frame, ui: &mut Ui, states: &[ClusterState], area: Rect) {
    let scope = ui.scope(states);
    let mut body = Vec::new();
    for s in &scope {
        for p in &s.pending {
            body.push(Row::new(vec![
                Cell::from(s.name.clone()).style(Style::new().fg(ACCENT)),
                Cell::from(p.partitions.clone()).style(Style::new().fg(C_TEXT)),
                Cell::from(p.gpu_type.clone().unwrap_or_else(|| "any".into())).style(Style::new().fg(C_TEXT)),
                Cell::from(format!("{:>5}", p.jobs)).style(Style::new().fg(C_TEXT)),
                Cell::from(format!("{:>5}", p.gpus)).style(Style::new().fg(C_HELD).bold()),
            ]));
        }
    }
    let hidden: Vec<&str> = scope.iter().filter(|s| s.kind == Kind::Slurm && !s.jobs_visible).map(|s| s.name.as_str()).collect();
    let mut right = "pending jobs waiting on Priority or Resources".to_string();
    if !hidden.is_empty() {
        right += &format!("; {} only shows your own jobs", hidden.join(", "));
    }
    let n = body.len();
    let table = Table::new(body, [Constraint::Length(8), Constraint::Length(44), Constraint::Length(24), Constraint::Length(6), Constraint::Min(6)])
        .column_spacing(2)
        .header(header_row(&["CLUSTER", "PARTITIONS", "TYPE", " JOBS", " GPUS"], None, false))
        .row_highlight_style(sel_style())
        .block(table_block(ui, n, right));
    f.render_stateful_widget(table, area, &mut ui.table);
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width.saturating_sub(2));
    let h = h.min(area.height.saturating_sub(2));
    Rect { x: area.x + (area.width - w) / 2, y: area.y + (area.height - h) / 2, width: w, height: h }
}

fn popup_block(title: &str) -> Block<'static> {
    popup_block_hint(title, "any key to close")
}

fn popup_block_hint(title: &str, hint: &str) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(ACCENT))
        .title_top(Line::from(Span::styled(format!(" {title} "), Style::new().fg(ACCENT).bold())))
        .title_bottom(Line::from(Span::styled(format!(" {hint} "), dim())).right_aligned())
}

/// Full, wrapped error text of every failing cluster, in a scrollable window.
fn draw_errors(f: &mut Frame, area: Rect, states: &[ClusterState], scroll: u16, now: DateTime<Utc>) {
    let mut lines: Vec<Line> = Vec::new();
    for s in states.iter().filter(|s| s.error.is_some()) {
        if !lines.is_empty() {
            lines.push(Line::default());
        }
        let since = match s.last_success {
            Some(_) => format!("last ok {} ago", ago(s.last_success, now)),
            None => "never reached".to_string(),
        };
        lines.push(Line::from(vec![Span::styled(s.name.clone(), Style::new().fg(C_ERR).bold()), Span::styled(format!("  {since}"), dim())]));
        for l in s.error.as_deref().unwrap_or("").lines() {
            lines.push(Line::from(Span::styled(l.to_string(), Style::new().fg(C_TEXT))));
        }
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled("No errors. Every cluster answered its last poll.", dim())));
    }
    let r = centered(area, (area.width * 4 / 5).max(40), (area.height * 3 / 5).max(8));
    let block = popup_block_hint("errors", "Up/Down scroll, any other key closes").padding(Padding::horizontal(1));
    // long lines wrap, so allow scrolling well past the unwrapped line count
    let max = (lines.len() as u16 * 4).saturating_sub(r.height.saturating_sub(2));
    f.render_widget(Clear, r);
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((scroll.min(max), 0)).block(block), r);
}

fn draw_help(f: &mut Frame, area: Rect) {
    let rows: [(&str, &str); 18] = [
        ("Tab / Shift-Tab / 0-9", "switch cluster (0 = all)"),
        ("v", "switch view: nodes, my jobs, queue"),
        ("/", "search node, type or partition; Esc clears"),
        ("t / T", "cycle the GPU type filter / clear it"),
        ("s / I", "next sort column / reverse order"),
        ("f", "only nodes with a GPU you can get now"),
        ("n", "GPUs needed on one node: hides smaller nodes, sorts by wait"),
        ("Enter", "node detail: release schedule, per-GPU use on standalone hosts"),
        ("r", "poll now (delayed if the last poll was under 20s ago)"),
        ("Up Down PgUp PgDn j k g G", "move"),
        ("e", "full ssh error text of failing clusters, scrollable"),
        ("q / Ctrl-C", "quit"),
        ("", ""),
        ("READY  now", "a new job can get a GPU here right now"),
        ("READY  2h31m", "wait until enough GPUs free up, from job time limits (upper bound)"),
        ("READY  ?", "this cluster hides other users' jobs, so no forecast"),
        ("held", "free GPUs you cannot get: node reserved by backfill, or CPU / memory full"),
        ("F1-F10", "htop keys also work: help cluster search type refresh sort view free need quit"),
    ];
    let lines: Vec<Line> = rows
        .iter()
        .map(|(k, v)| Line::from(vec![Span::styled(format!(" {k:>26}  "), Style::new().fg(ACCENT).bold()), Span::styled(*v, Style::new().fg(C_TEXT))]))
        .collect();
    let r = centered(area, 104, rows.len() as u16 + 2);
    f.render_widget(Clear, r);
    f.render_widget(Paragraph::new(lines).block(popup_block("ptop keys")), r);
}

fn draw_detail(f: &mut Frame, area: Rect, s: &ClusterState, n: &Node, need: u32, now: DateTime<Utc>) {
    let kv = |k: &str, v: String| Line::from(vec![Span::styled(format!(" {k:<11}"), dim()), Span::styled(v, Style::new().fg(C_TEXT))]);
    let mut lines = vec![
        kv("gpu", format!("{} x {}    {} free now, {} in use, {} held or offline", n.total, n.gpu_type, n.usable, n.used, n.free - n.usable)),
        kv("state", n.state.clone()),
        kv("cpu", format!("{} of {} cores idle", n.cpu_idle, n.cpu_total)),
        kv("memory", format!("{} of {} GB free", n.mem_free_gb, n.mem_total_gb)),
        kv("partitions", n.partitions.join(", ")),
        Line::default(),
    ];
    let head = |t: &str| Line::from(Span::styled(format!(" {t}"), Style::new().fg(ACCENT).bold()));
    if !n.gpus.is_empty() {
        lines.push(head("Per-GPU use (nvidia-smi)"));
        for g in &n.gpus {
            let st = if g.busy { Style::new().fg(C_USED) } else { Style::new().fg(C_FREE) };
            lines.push(Line::from(vec![
                Span::styled(format!("  gpu{:<2} ", g.index), st.bold()),
                Span::styled(
                    format!(
                        "{:>3}%  {:>6} / {:<6} MiB  {}",
                        g.util,
                        g.mem_used_mb,
                        g.mem_total_mb,
                        if g.users.is_empty() { "-".to_string() } else { g.users.join(", ") }
                    ),
                    Style::new().fg(C_TEXT),
                ),
            ]));
        }
    } else if s.kind == Kind::Slurm && !s.jobs_visible {
        lines.push(Line::from(Span::styled(" This cluster only shows your own jobs, so releases cannot be forecast.", dim())));
    } else if n.releases.is_empty() {
        lines.push(Line::from(Span::styled(" No running GPU jobs on this node.", dim())));
    } else {
        lines.push(head("Releases by job time limit (upper bound)"));
        let mut avail = n.usable;
        for r in n.releases.iter().take(16) {
            avail += r.gpus;
            let hit = need > 0 && avail >= need && avail - r.gpus < need;
            let mut l = vec![
                Span::styled(format!("  {}  ", r.at.with_timezone(&Local).format("%a %H:%M")), Style::new().fg(C_TEXT)),
                Span::styled(format!("in {:<7}", human_dur(r.at - now)), dim()),
                Span::styled(format!(" +{}", r.gpus), Style::new().fg(C_FREE)),
                Span::styled(format!("  {avail} free"), Style::new().fg(C_TEXT)),
            ];
            if hit {
                l.push(Span::styled(format!("   <- {need} available"), Style::new().fg(C_FREE).bold()));
            }
            lines.push(Line::from(l));
        }
    }
    let r = centered(area, 80, lines.len() as u16 + 2);
    f.render_widget(Clear, r);
    f.render_widget(Paragraph::new(lines).block(popup_block(&format!("{} / {}", s.name, n.name))), r);
}

// ---------------------------------------------------------------- events

enum Action {
    Continue,
    Quit,
}

fn handle_key(k: KeyEvent, ui: &mut Ui, states: &[ClusterState], clusters: &Clusters, now: DateTime<Utc>) -> Action {
    if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
        return Action::Quit;
    }
    // text input modes
    match &mut ui.input {
        Input::Search(s) => {
            match k.code {
                KeyCode::Esc => {
                    ui.search.clear();
                    ui.input = Input::None;
                }
                KeyCode::Enter => ui.input = Input::None,
                KeyCode::Backspace => {
                    s.pop();
                    ui.search = s.clone();
                }
                KeyCode::Char(c) => {
                    s.push(c);
                    ui.search = s.clone();
                }
                _ => {}
            }
            ui.table.select(Some(0));
            return Action::Continue;
        }
        Input::Need(s) => {
            match k.code {
                KeyCode::Esc => ui.input = Input::None,
                KeyCode::Enter => {
                    ui.need = s.parse().unwrap_or(0).min(64);
                    if ui.need > 0 {
                        ui.sort = SortKey::Ready;
                        ui.reverse = false;
                        ui.view = View::Nodes;
                    }
                    ui.input = Input::None;
                    ui.table.select(Some(0));
                }
                KeyCode::Backspace => {
                    s.pop();
                }
                KeyCode::Char(c) if c.is_ascii_digit() && s.len() < 2 => s.push(c),
                _ => {}
            }
            return Action::Continue;
        }
        Input::None => {}
    }
    if let Popup::Errors(scroll) = &mut ui.popup {
        match k.code {
            KeyCode::Down | KeyCode::Char('j') => *scroll = scroll.saturating_add(1),
            KeyCode::Up | KeyCode::Char('k') => *scroll = scroll.saturating_sub(1),
            KeyCode::PageDown => *scroll = scroll.saturating_add(10),
            KeyCode::PageUp => *scroll = scroll.saturating_sub(10),
            KeyCode::Home | KeyCode::Char('g') => *scroll = 0,
            _ => ui.popup = Popup::None,
        }
        return Action::Continue;
    }
    if !matches!(ui.popup, Popup::None) {
        ui.popup = Popup::None;
        return Action::Continue;
    }

    let n_tabs = states.len() + 1;
    let rows = ui.row_count(states, now);
    let sel = ui.table.selected().unwrap_or(0);
    let page = 15;
    match k.code {
        KeyCode::F(10) | KeyCode::Char('q') => return Action::Quit,
        KeyCode::F(1) | KeyCode::Char('?') | KeyCode::Char('h') => ui.popup = Popup::Help,
        KeyCode::Char('e') => ui.popup = Popup::Errors(0),
        KeyCode::F(2) | KeyCode::Tab => {
            ui.tab = (ui.tab + 1) % n_tabs;
            ui.table.select(Some(0));
        }
        KeyCode::BackTab => {
            ui.tab = (ui.tab + n_tabs - 1) % n_tabs;
            ui.table.select(Some(0));
        }
        KeyCode::Char(c @ '0'..='9') => {
            let i = c as usize - '0' as usize;
            if i < n_tabs {
                ui.tab = i;
                ui.table.select(Some(0));
            }
        }
        KeyCode::F(3) | KeyCode::Char('/') => ui.input = Input::Search(ui.search.clone()),
        KeyCode::Esc => {
            ui.search.clear();
            ui.type_filter = None;
        }
        KeyCode::F(4) | KeyCode::Char('t') => {
            let types = ui.all_types(states);
            ui.type_filter = match &ui.type_filter {
                None => types.first().cloned(),
                Some(cur) => types.iter().position(|t| t == cur).and_then(|i| types.get(i + 1)).cloned(),
            };
            ui.table.select(Some(0));
        }
        KeyCode::Char('T') => ui.type_filter = None,
        KeyCode::F(5) | KeyCode::Char('r') => {
            for c in clusters.iter() {
                c.wake.notify_one();
            }
            ui.flash("polling now");
        }
        KeyCode::F(6) | KeyCode::Char('s') | KeyCode::Char('>') | KeyCode::Char('<') => {
            let i = SortKey::ALL.iter().position(|x| *x == ui.sort).unwrap_or(0);
            ui.sort = SortKey::ALL[(i + 1) % SortKey::ALL.len()];
            ui.view = View::Nodes;
        }
        KeyCode::Char('I') => ui.reverse = !ui.reverse,
        KeyCode::F(7) | KeyCode::Char('v') => {
            ui.view = match ui.view {
                View::Nodes => View::Jobs,
                View::Jobs => View::Queue,
                View::Queue => View::Nodes,
            };
            ui.table.select(Some(0));
        }
        KeyCode::F(8) | KeyCode::Char('f') => {
            ui.only_free = !ui.only_free;
            ui.table.select(Some(0));
        }
        KeyCode::F(9) | KeyCode::Char('n') => ui.input = Input::Need(if ui.need > 0 { ui.need.to_string() } else { String::new() }),
        KeyCode::Down | KeyCode::Char('j') => ui.table.select(Some((sel + 1).min(rows.saturating_sub(1)))),
        KeyCode::Up | KeyCode::Char('k') => ui.table.select(Some(sel.saturating_sub(1))),
        KeyCode::PageDown => ui.table.select(Some((sel + page).min(rows.saturating_sub(1)))),
        KeyCode::PageUp => ui.table.select(Some(sel.saturating_sub(page))),
        KeyCode::Home | KeyCode::Char('g') => ui.table.select(Some(0)),
        KeyCode::End | KeyCode::Char('G') => ui.table.select(Some(rows.saturating_sub(1))),
        KeyCode::Enter if ui.view == View::Nodes => {
            if let Some(r) = ui.node_rows(states, now).get(sel) {
                ui.popup = Popup::Detail(r.cluster.to_string(), r.node.name.clone());
            }
        }
        _ => {}
    }
    Action::Continue
}

async fn snapshot(clusters: &Clusters) -> Vec<ClusterState> {
    let mut out = Vec::with_capacity(clusters.len());
    for c in clusters.iter() {
        let s = c.state.read().await;
        out.push(s.clone());
    }
    out
}

async fn run(mut term: DefaultTerminal, clusters: Clusters, mut ui: Ui) -> Result<()> {
    ui.table.select(Some(0));
    loop {
        let states = snapshot(&clusters).await;
        let now = Utc::now();
        let rows = ui.row_count(&states, now);
        if ui.table.selected().is_some_and(|s| s >= rows) {
            ui.table.select(Some(rows.saturating_sub(1)));
        }
        term.draw(|f| draw(f, &mut ui, &states, now))?;
        // crossterm polling blocks this worker only; collectors run on the others
        if tokio::task::block_in_place(|| event::poll(Duration::from_millis(500)))? {
            match tokio::task::block_in_place(event::read)? {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    if let Action::Quit = handle_key(k, &mut ui, &states, &clusters, now) {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
    }
}

fn parse_keys(s: &str) -> Vec<KeyEvent> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(c) = rest.chars().next() {
        if let Some(end) = rest.strip_prefix('<').and_then(|r| r.find('>')) {
            let name = &rest[1..end + 1];
            let code = match name {
                "Enter" => Some(KeyCode::Enter),
                "Esc" => Some(KeyCode::Esc),
                "Tab" => Some(KeyCode::Tab),
                "Down" => Some(KeyCode::Down),
                n if n.starts_with('F') => n[1..].parse().ok().map(KeyCode::F),
                _ => None,
            };
            if let Some(code) = code {
                out.push(KeyEvent::new(code, KeyModifiers::NONE));
                rest = &rest[end + 2..];
                continue;
            }
        }
        out.push(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        rest = &rest[c.len_utf8()..];
    }
    out
}

async fn dump_once(size: &str, keys: &str, html: Option<PathBuf>, clusters: Clusters, mut ui: Ui) -> Result<()> {
    let (w, h) = size.split_once('x').context("--dump wants COLSxROWS")?;
    let (w, h): (u16, u16) = (w.parse()?, h.parse()?);
    let deadline = std::time::Instant::now() + Duration::from_secs(150);
    loop {
        let states = snapshot(&clusters).await;
        if states.iter().all(|s| s.last_attempt.is_some()) || std::time::Instant::now() > deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let states = snapshot(&clusters).await;
    let now = Utc::now();
    ui.table.select(Some(0));
    for k in parse_keys(keys) {
        if let Action::Quit = handle_key(k, &mut ui, &states, &clusters, now) {
            return Ok(());
        }
    }
    let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h))?;
    term.draw(|f| draw(f, &mut ui, &states, now))?;
    let buf = term.backend().buffer();
    print!("{}", dump::text(buf));
    if let Some(p) = html {
        std::fs::write(&p, dump::html(buf))?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    collect::answer_askpass();
    let args = Args::parse();
    if let Some(Cmd::Init { print }) = args.command {
        let user = args.user.as_deref();
        if user.is_some_and(|u| !collect::safe(u, "._-")) {
            bail!("bad user {:?}", user.unwrap_or_default());
        }
        if print {
            print!("{}", config::template(user));
        } else {
            let path = config::init(user)?;
            let next = if user.is_some() { "Run `ptop`, or edit the file to change clusters." } else { "Edit it to set your login and clusters, then run `ptop`." };
            println!("Wrote {}\n{next}", path.display());
        }
        return Ok(());
    }
    let cfg = config::load(args.config.clone())?;
    let Some(user) = args.user.clone().or(cfg.user.clone()) else {
        bail!("no cluster user: run `ptop init` to create a config file, or pass `-u <name>`");
    };
    if !collect::safe(&user, "._-") {
        bail!("bad user {user:?}");
    }
    let cfgs = if !args.clusters.is_empty() {
        args.clusters.iter().map(|s| collect::parse_spec(s)).collect::<Result<Vec<_>>>()?
    } else if !cfg.clusters.is_empty() {
        cfg.clusters.into_iter().map(|c| c.into_cfg()).collect::<Result<Vec<_>>>()?
    } else {
        DEFAULT_CLUSTERS.iter().map(|s| collect::parse_spec(s)).collect::<Result<Vec<_>>>()?
    };
    let mut names = std::collections::HashSet::new();
    if let Some(dup) = cfgs.iter().find(|c| !names.insert(c.name.clone())) {
        bail!("cluster name {:?} is used twice", dup.name);
    }
    let interval = args.interval.or(cfg.interval).unwrap_or(60).max(20);
    let mut ssh_args = cfg.ssh_args;
    ssh_args.extend(args.ssh_args.iter().map(|a| config::expand_home(a)));

    let clusters = collect::spawn_all(cfgs, SshCfg { user: user.clone(), extra_args: ssh_args, password: None }, Duration::from_secs(interval));
    let ui = Ui {
        tab: 0,
        view: View::Nodes,
        sort: SortKey::Free,
        reverse: false,
        type_filter: None,
        search: String::new(),
        need: 0,
        only_free: false,
        input: Input::None,
        popup: Popup::None,
        table: TableState::default(),
        interval,
        user,
        flash: None,
    };
    if let Some(Cmd::Webui { bind }) = args.command {
        return webui::serve(bind, clusters).await;
    }
    if let Some(size) = args.dump.clone() {
        return dump_once(&size, &args.keys, args.html.clone(), clusters, ui).await;
    }
    let term = ratatui::init();
    let res = run(term, clusters, ui).await;
    ratatui::restore();
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use slurm::Release;

    fn node(usable: u32, total: u32, releases: &[(i64, u32)], now: DateTime<Utc>) -> Node {
        Node {
            name: "n".into(),
            partitions: vec![],
            state: "mixed".into(),
            gpu_type: "h200".into(),
            total,
            used: total - usable,
            free: usable,
            usable,
            status: NodeStatus::Ok,
            cpu_idle: 1,
            cpu_total: 1,
            mem_free_gb: 100,
            mem_total_gb: 100,
            releases: releases.iter().map(|&(h, g)| Release { at: now + chrono::Duration::hours(h), gpus: g }).collect(),
            gpus: vec![],
        }
    }

    #[test]
    fn readiness() {
        let now = Utc::now();
        let n = node(1, 8, &[(1, 2), (3, 4)], now);
        assert_eq!(ready_at(&n, 1, true, now), Some(now));
        assert_eq!(ready_at(&n, 3, true, now), Some(now + chrono::Duration::hours(1)));
        assert_eq!(ready_at(&n, 4, true, now), Some(now + chrono::Duration::hours(3)));
        assert_eq!(ready_at(&n, 8, true, now), None);
        assert_eq!(ready_at(&n, 3, false, now), None);
        assert_eq!(ready_at(&n, 9, true, now), None);
    }

    #[test]
    fn durations() {
        assert_eq!(human_dur(chrono::Duration::minutes(135)), "2h15m");
        assert_eq!(human_dur(chrono::Duration::hours(80)), "3d8h");
    }

    #[test]
    fn bars() {
        assert_eq!(bar_cells([47, 0, 0, 1], 20).iter().sum::<usize>(), 20);
        assert_eq!(bar_cells([47, 0, 0, 1], 20)[3], 1, "one free GPU stays visible");
        assert_eq!(bar_cells([0, 0, 0, 4], 10), [0, 0, 0, 10]);
        assert_eq!(bar_cells([0, 0, 0, 0], 10), [0; 4]);
        let m = Meter { label: "rtx_pro_6000_blackwell".into(), usable: 24, used: 40, held: 0, offline: 0, total: 64 };
        for w in [40usize, 60, 120] {
            assert!(meter_line(&m, 22, w).width() <= w, "width {w}");
        }
    }
}
