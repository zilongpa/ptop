//! `ptop webui`: the same data as the terminal UI, served as a single web page.

use crate::collect::{ClusterState, Clusters};
use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::net::SocketAddr;

const PAGE: &str = include_str!("webui.html");

#[derive(Serialize)]
struct StateOut {
    clusters: Vec<ClusterState>,
}

async fn state(State(clusters): State<Clusters>) -> Json<StateOut> {
    let mut out = Vec::with_capacity(clusters.len());
    for c in clusters.iter() {
        out.push(c.state.read().await.clone());
    }
    Json(StateOut { clusters: out })
}

async fn refresh(State(clusters): State<Clusters>) -> StatusCode {
    for c in clusters.iter() {
        c.wake.notify_one();
    }
    StatusCode::ACCEPTED
}

async fn page() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-cache")], Html(PAGE))
}

pub async fn serve(bind: SocketAddr, clusters: Clusters) -> Result<()> {
    let router = Router::new()
        .route("/", get(page))
        .route("/api/state", get(state))
        .route("/api/refresh", post(refresh))
        .with_state(clusters);
    let listener = tokio::net::TcpListener::bind(bind).await.with_context(|| format!("cannot listen on {bind}"))?;
    eprintln!("ptop webui: http://{bind}");
    axum::serve(listener, router).await?;
    Ok(())
}
