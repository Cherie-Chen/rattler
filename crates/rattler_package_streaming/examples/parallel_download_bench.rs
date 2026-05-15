//! Lockfile-driven side-by-side benchmark for single-GET vs parallel range-GET
//! package download.
//!
//! Workflow per run of this binary:
//!
//! 1. Build a list of real conda package URLs.  By default we use a small,
//!    representative subset of conda-forge packages spanning sub-threshold,
//!    medium, and large sizes.  Set `PIXI_LOCK=/path/to/pixi.lock` to instead
//!    parse a real Pixi lockfile and use the conda packages it references
//!    (capped via `BENCH_MAX_PACKAGES`, default 12, to keep run time bounded).
//!
//! 2. Pre-fetch every package once from `conda.anaconda.org` into an
//!    in-memory cache.  This warms Cloudflare and lets us re-serve the exact
//!    same bytes through a local throttled HTTP server later.
//!
//! 3. **Phase A** — measure single-GET vs parallel range-GET against the real
//!    CDN, end-to-end (download + extract).  This is the "where parallel is
//!    on a saturated CDN path" datapoint and is honest about the cases where
//!    parallel is *not* a win.
//!
//! 4. **Phase B** — stand up a local axum server with **per-connection
//!    bandwidth throttling** that re-serves the cached bytes under their
//!    original conda URL paths.  Sweep bandwidth caps (50, 100, 250, 500,
//!    1000 Mbps) and run both modes for every package at every cap.
//!
//! 5. Print a summary aggregating Phase B medians by bandwidth, including the
//!    parallel/single speedup.
//!
//! Per-connection throttling models real-world AWS per-flow bandwidth caps and
//! Cloudflare per-connection edge throughput — the regimes where parallel
//! range GETs can recover aggregate throughput by opening multiple flows.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release \
//!     --features reqwest \
//!     --example parallel_download_bench
//!
//! # Or, against a real lockfile:
//! PIXI_LOCK=path/to/pixi.lock BENCH_MAX_PACKAGES=20 cargo run ...
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axum::{
    Router,
    body::Body,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::any,
};
use bytes::Bytes;
use futures::stream;
use rattler_package_streaming::reqwest::parallel_download::ParallelDownloadConfig;
use rattler_package_streaming::reqwest::tokio::{extract, extract_with_config};
use reqwest_middleware::{ClientWithMiddleware, Middleware, Next};
use tempfile::tempdir;
use tokio::time::sleep;
use url::Url;

// =============================================================================
// Default package set: a small, varied mix from conda-forge spanning sub-
// threshold (parallel falls back to single-GET) up through large.  This gives
// us a meaningful comparison without taking forever to run.
// =============================================================================

const DEFAULT_PACKAGES: &[&str] = &[
    // Sub-threshold (~3 MB) — parallel will fall back to single-GET.
    "https://conda.anaconda.org/conda-forge/linux-64/openssl-3.3.2-hb9d3cd8_0.conda",
    // Just over the 8 MiB threshold (~12 MB) — only ~2 chunks fit.
    "https://conda.anaconda.org/conda-forge/linux-64/icu-75.1-he02047a_0.conda",
    // Medium (~30 MB) — ~4 chunks.
    "https://conda.anaconda.org/conda-forge/linux-64/python-3.12.7-hc5c86c4_0_cpython.conda",
    // Mid-large (~38 MB) — ~5 chunks.
    "https://conda.anaconda.org/conda-forge/linux-64/libllvm19-19.1.5-ha7bfdaf_0.conda",
    // Large (~119 MB) — exercises full 8-way concurrency.
    "https://conda.anaconda.org/conda-forge/linux-64/mkl-2024.2.2-ha957f24_16.conda",
];

// =============================================================================
// Counting middleware: counts HTTP requests by method and bytes received via
// successful GET responses.  Only attributes Content-Length to GETs so HEADs
// don't double the byte total.
// =============================================================================

#[derive(Default, Clone)]
struct CountingMiddleware {
    get_count: Arc<AtomicU64>,
    head_count: Arc<AtomicU64>,
    bytes_received: Arc<AtomicU64>,
}

impl CountingMiddleware {
    fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.get_count.load(Ordering::Relaxed),
            self.head_count.load(Ordering::Relaxed),
            self.bytes_received.load(Ordering::Relaxed),
        )
    }

    fn reset(&self) {
        self.get_count.store(0, Ordering::Relaxed);
        self.head_count.store(0, Ordering::Relaxed);
        self.bytes_received.store(0, Ordering::Relaxed);
    }
}

#[async_trait]
impl Middleware for CountingMiddleware {
    async fn handle(
        &self,
        req: reqwest::Request,
        extensions: &mut http::Extensions,
        next: Next<'_>,
    ) -> reqwest_middleware::Result<reqwest::Response> {
        let is_get = req.method().as_str() == "GET";
        match req.method().as_str() {
            "GET" => {
                self.get_count.fetch_add(1, Ordering::Relaxed);
            }
            "HEAD" => {
                self.head_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        let resp = next.run(req, extensions).await?;
        if is_get && let Some(len) = resp.content_length() {
            self.bytes_received.fetch_add(len, Ordering::Relaxed);
        }
        Ok(resp)
    }
}

// =============================================================================
// Throttled mock server.
//
// Serves the *real* package bytes that we cached during the prefetch step.
// The route uses a wildcard so packages are reachable under their original
// conda URL path (e.g.  /conda-forge/linux-64/python-...conda).  Range and
// HEAD are supported; throttling is per response (= per connection).
// =============================================================================

#[derive(Clone)]
struct MockState {
    /// Keyed by *path* (the URL path component, including leading slash).
    payloads: Arc<HashMap<String, Bytes>>,
    /// Per-connection bandwidth cap in bytes/sec.  0 = unthrottled.
    bandwidth_bps: Arc<AtomicU64>,
}

async fn handle_request(
    State(state): State<MockState>,
    AxumPath(rest): AxumPath<String>,
    method: Method,
    headers: HeaderMap,
) -> Response {
    let key = format!("/{rest}");
    let payload = match state.payloads.get(&key) {
        Some(p) => p.clone(),
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    let total = payload.len();

    let (start, end, status) = match headers.get(header::RANGE) {
        Some(v) => {
            let s = v.to_str().unwrap_or("");
            match parse_range(s, total) {
                Some((a, b)) => (a, b, StatusCode::PARTIAL_CONTENT),
                None => return StatusCode::RANGE_NOT_SATISFIABLE.into_response(),
            }
        }
        None => (0usize, total - 1, StatusCode::OK),
    };

    let length = end - start + 1;
    let mut resp_headers = HeaderMap::new();
    resp_headers.insert(header::CONTENT_LENGTH, HeaderValue::from(length));
    resp_headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    if status == StatusCode::PARTIAL_CONTENT {
        resp_headers.insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes {start}-{end}/{total}")).unwrap(),
        );
    }

    if method == Method::HEAD {
        return (status, resp_headers).into_response();
    }

    let bps = state.bandwidth_bps.load(Ordering::Relaxed);
    let body = throttled_body(payload.slice(start..=end), bps);
    (status, resp_headers, body).into_response()
}

fn parse_range(s: &str, total: usize) -> Option<(usize, usize)> {
    let s = s.strip_prefix("bytes=")?;
    let mut parts = s.split('-');
    let start: usize = parts.next()?.parse().ok()?;
    let end: usize = match parts.next() {
        Some("") | None => total.saturating_sub(1),
        Some(s) => s.parse().ok()?,
    };
    if start > end || end >= total {
        return None;
    }
    Some((start, end))
}

/// Build a throttled body Stream.  Uses a "target elapsed" model so accuracy
/// holds even when individual `tokio::time::sleep` calls round up to ms.
fn throttled_body(bytes: Bytes, bandwidth_bps: u64) -> Body {
    if bandwidth_bps == 0 {
        return Body::from(bytes);
    }
    let chunk_size: usize = 256 * 1024;
    let stream = stream::unfold(
        (bytes, 0usize, Instant::now()),
        move |(bytes, offset, start_time)| async move {
            if offset >= bytes.len() {
                return None;
            }
            let end = std::cmp::min(offset + chunk_size, bytes.len());
            let chunk = bytes.slice(offset..end);
            let bytes_sent_so_far = (offset + chunk.len()) as u64;
            let target_elapsed =
                Duration::from_secs_f64(bytes_sent_so_far as f64 / bandwidth_bps as f64);
            let elapsed = start_time.elapsed();
            if target_elapsed > elapsed {
                sleep(target_elapsed - elapsed).await;
            }
            Some((Ok::<_, std::io::Error>(chunk), (bytes, end, start_time)))
        },
    );
    Body::from_stream(stream)
}

struct MockServer {
    base_url: String,
    bandwidth_bps: Arc<AtomicU64>,
}

impl MockServer {
    /// Start the mock server, binding to an OS-assigned port on 127.0.0.1.
    /// Returns once the listener is up.
    async fn start(payloads: HashMap<String, Bytes>) -> Self {
        let bandwidth_bps = Arc::new(AtomicU64::new(0));
        let state = MockState {
            payloads: Arc::new(payloads),
            bandwidth_bps: bandwidth_bps.clone(),
        };

        // axum 0.8 uses `{*var}` for catch-all path captures.
        let app = Router::new()
            .route("/{*rest}", any(handle_request))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });

        MockServer {
            base_url: format!("http://{addr}"),
            bandwidth_bps,
        }
    }

    fn set_bandwidth_mbps(&self, mbps: u64) {
        // Mbps -> bytes/sec: (Mbps * 1_000_000) / 8.
        let bps = if mbps == 0 { 0 } else { (mbps * 1_000_000) / 8 };
        self.bandwidth_bps.store(bps, Ordering::Relaxed);
    }

    /// Build a URL on the mock server for the original conda URL's *path*.
    /// We re-use the path so `extract` correctly identifies the archive type
    /// from the file extension.
    fn local_url(&self, original: &Url) -> String {
        format!("{}{}", self.base_url, original.path())
    }
}

// =============================================================================
// Lockfile parsing + URL list construction.
// =============================================================================

/// Read the configured lockfile (`PIXI_LOCK` env var) or fall back to the
/// hard-coded default URL list.  Caps the result at `BENCH_MAX_PACKAGES`
/// (default 12) so a giant lockfile doesn't make the bench take all day.
fn load_urls() -> Vec<String> {
    let max = std::env::var("BENCH_MAX_PACKAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12usize);

    if let Ok(path) = std::env::var("PIXI_LOCK") {
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("could not read PIXI_LOCK={path}: {e}"));
        let mut urls = parse_pixi_lock(&body);
        urls.truncate(max);
        eprintln!(
            "Using {} URLs from {} (capped at BENCH_MAX_PACKAGES={})",
            urls.len(),
            path,
            max
        );
        return urls;
    }

    DEFAULT_PACKAGES.iter().map(|s| (*s).to_string()).collect()
}

/// Minimal pixi.lock parser: scan for lines of the form
/// `- conda: https://...` or `- pypi: ...`.  We only keep `conda:` URLs that
/// look like real archive names (`.conda` or `.tar.bz2`).  De-duplicates.
///
/// Pixi's lockfile is YAML; we deliberately avoid pulling in `serde_yaml`
/// just for an example.  This stays robust to comments and indentation.
fn parse_pixi_lock(body: &str) -> Vec<String> {
    let mut urls = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for line in body.lines() {
        let trimmed = line.trim();
        // Look for "conda:" style entries (with or without leading dash).
        let candidate = trimmed
            .strip_prefix("- conda:")
            .or_else(|| trimmed.strip_prefix("conda:"))
            .map(str::trim);
        let url = match candidate {
            Some(s) if s.starts_with("http") => s,
            _ => continue,
        };
        // Drop any trailing whitespace/comment.
        let url = url.split_whitespace().next().unwrap_or(url);
        if !(url.ends_with(".conda") || url.ends_with(".tar.bz2")) {
            continue;
        }
        if seen.insert(url.to_string()) {
            urls.push(url.to_string());
        }
    }
    urls
}

// =============================================================================
// Pre-fetch step.  Downloads every URL once into memory so Phase B can
// re-serve the same bytes through the throttled mock server.
// =============================================================================

async fn prefetch_packages(urls: &[String]) -> HashMap<String, Bytes> {
    let mut out = HashMap::new();
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client");
    for u in urls {
        let parsed = Url::parse(u).unwrap_or_else(|e| panic!("bad URL {u}: {e}"));
        eprintln!("  prefetch: {}", parsed.path());
        let resp = client
            .get(parsed.clone())
            .send()
            .await
            .unwrap_or_else(|e| panic!("prefetch {u}: {e}"))
            .error_for_status()
            .unwrap_or_else(|e| panic!("prefetch {u}: {e}"));
        let bytes = resp
            .bytes()
            .await
            .unwrap_or_else(|e| panic!("prefetch body {u}: {e}"));
        out.insert(parsed.path().to_string(), bytes);
    }
    out
}

// =============================================================================
// Bench drivers.  Each measurement does the FULL extract end-to-end (download
// + decompression + tar/zip extraction) into a fresh tempdir.  This is what
// baszalmstra asked for in conda/rattler#2434.
// =============================================================================

#[derive(Debug, Clone, Copy)]
struct RunStats {
    elapsed_ms: u128,
    gets: u64,
    heads: u64,
    bytes: u64,
}

async fn run_once(
    client: &ClientWithMiddleware,
    url_str: &str,
    parallel: bool,
    counter: &CountingMiddleware,
) -> RunStats {
    let url = Url::parse(url_str).expect("url");
    let dir = tempdir().expect("tempdir");
    counter.reset();
    let start = Instant::now();
    if parallel {
        let cfg = ParallelDownloadConfig::default();
        extract_with_config(client.clone(), url, dir.path(), None, None, &cfg)
            .await
            .expect("extract_with_config");
    } else {
        extract(client.clone(), url, dir.path(), None, None)
            .await
            .expect("extract");
    }
    let elapsed_ms = start.elapsed().as_millis();
    let (gets, heads, bytes) = counter.snapshot();
    RunStats {
        elapsed_ms,
        gets,
        heads,
        bytes,
    }
}

// =============================================================================
// Entry point.  Phase A first (real CDN), then Phase B (throttled curve).
// =============================================================================

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let urls = load_urls();
    println!("Lockfile workload: {} packages", urls.len());

    eprintln!("Prefetching packages from conda.anaconda.org…");
    let cache = prefetch_packages(&urls).await;

    // Build the instrumented HTTP client used for both phases.
    let counter = CountingMiddleware::default();
    let raw_client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client");
    let client = reqwest_middleware::ClientBuilder::new(raw_client)
        .with(counter.clone())
        .build();

    let runs_per_mode: usize = 2;

    // -------------------------------------------------------------------------
    // Phase A: real CDN, end-to-end.
    // -------------------------------------------------------------------------
    println!();
    println!("=== Phase A: real conda.anaconda.org (Cloudflare-fronted) ===");
    println!(
        "{:<60} {:<8} {:>3} {:>7} {:>5} {:>5} {:>8} {:>10}",
        "package", "mode", "run", "ms", "GETs", "HEADs", "MB", "MB/s"
    );

    // (size_mb, single_mbps_samples, parallel_mbps_samples)
    let mut phase_a_samples: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();

    for u in &urls {
        // Warm-up (not recorded).
        let _ = run_once(&client, u, false, &counter).await;
        for run in 0..runs_per_mode {
            let s_single = run_once(&client, u, false, &counter).await;
            let s_parallel = run_once(&client, u, true, &counter).await;
            let name = pkg_name(u);
            print_row(&name, "single", run, &s_single);
            print_row(&name, "parallel", run, &s_parallel);
            phase_a_samples
                .entry(name.clone())
                .or_default()
                .0
                .push(mbps_of(&s_single));
            phase_a_samples
                .entry(name)
                .or_default()
                .1
                .push(mbps_of(&s_parallel));
        }
    }

    // -------------------------------------------------------------------------
    // Phase B: throttled mock server with cached real package bytes.
    // -------------------------------------------------------------------------
    let server = MockServer::start(cache).await;
    let bandwidths_mbps: &[u64] = &[50, 100, 250, 500, 1000];

    println!();
    println!("=== Phase B: local throttled server, real package bytes ===");
    println!(
        "{:>7} {:<60} {:<8} {:>3} {:>7} {:>5} {:>5} {:>8} {:>10}",
        "bw_mbps", "package", "mode", "run", "ms", "GETs", "HEADs", "MB", "MB/s"
    );

    // Keyed by (bw_mbps, package_name, parallel?) -> Vec<MB/s>.
    let mut phase_b_samples: HashMap<(u64, String, bool), Vec<f64>> = HashMap::new();

    for &bw in bandwidths_mbps {
        server.set_bandwidth_mbps(bw);
        for u in &urls {
            let local = server.local_url(&Url::parse(u).unwrap());
            // Warm-up against the local server.
            let _ = run_once(&client, &local, false, &counter).await;
            for run in 0..runs_per_mode {
                let s_single = run_once(&client, &local, false, &counter).await;
                let s_parallel = run_once(&client, &local, true, &counter).await;
                let name = pkg_name(u);
                print_row_bw(bw, &name, "single", run, &s_single);
                print_row_bw(bw, &name, "parallel", run, &s_parallel);
                phase_b_samples
                    .entry((bw, name.clone(), false))
                    .or_default()
                    .push(mbps_of(&s_single));
                phase_b_samples
                    .entry((bw, name, true))
                    .or_default()
                    .push(mbps_of(&s_parallel));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Summaries.
    // -------------------------------------------------------------------------
    println!();
    println!("=== Phase A summary (real CDN; median MB/s, {runs_per_mode}-run) ===");
    println!(
        "{:<60} {:>10} {:>10} {:>9}",
        "package", "single", "parallel", "speedup"
    );
    for u in &urls {
        let name = pkg_name(u);
        if let Some((s, p)) = phase_a_samples.get(&name) {
            let s = median(s.clone());
            let p = median(p.clone());
            let sp = if s > 0.0 { p / s } else { 0.0 };
            println!("{name:<60} {s:>10.2} {p:>10.2} {sp:>8.2}x");
        }
    }

    println!();
    println!("=== Phase B summary (throttled; median MB/s aggregated across packages) ===");
    println!(
        "{:>7} {:>10} {:>10} {:>9}",
        "bw_mbps", "single", "parallel", "speedup"
    );
    for &bw in bandwidths_mbps {
        let mut s_all: Vec<f64> = Vec::new();
        let mut p_all: Vec<f64> = Vec::new();
        for u in &urls {
            let name = pkg_name(u);
            if let Some(v) = phase_b_samples.get(&(bw, name.clone(), false)) {
                s_all.extend(v);
            }
            if let Some(v) = phase_b_samples.get(&(bw, name, true)) {
                p_all.extend(v);
            }
        }
        let s = median(s_all);
        let p = median(p_all);
        let sp = if s > 0.0 { p / s } else { 0.0 };
        println!("{bw:>7} {s:>10.2} {p:>10.2} {sp:>8.2}x");
    }

    println!();
    println!("=== Phase B detail (per package × bandwidth; median MB/s, {runs_per_mode}-run) ===",);
    println!(
        "{:>7} {:<60} {:>10} {:>10} {:>9}",
        "bw_mbps", "package", "single", "parallel", "speedup"
    );
    for &bw in bandwidths_mbps {
        for u in &urls {
            let name = pkg_name(u);
            let s = median(
                phase_b_samples
                    .get(&(bw, name.clone(), false))
                    .cloned()
                    .unwrap_or_default(),
            );
            let p = median(
                phase_b_samples
                    .get(&(bw, name.clone(), true))
                    .cloned()
                    .unwrap_or_default(),
            );
            let sp = if s > 0.0 { p / s } else { 0.0 };
            println!("{bw:>7} {name:<60} {s:>10.2} {p:>10.2} {sp:>8.2}x");
        }
    }
}

fn pkg_name(url: &str) -> String {
    Url::parse(url)
        .ok()
        .and_then(|u| {
            Path::new(u.path())
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| url.to_string())
}

fn mbps_of(s: &RunStats) -> f64 {
    let mb = s.bytes as f64 / 1024.0 / 1024.0;
    let secs = (s.elapsed_ms as f64 / 1000.0).max(1e-6);
    mb / secs
}

fn print_row(name: &str, mode: &str, run: usize, s: &RunStats) {
    let mb = s.bytes as f64 / 1024.0 / 1024.0;
    println!(
        "{:<60} {:<8} {:>3} {:>5}ms {:>5} {:>5} {:>8.2} {:>10.2}",
        name,
        mode,
        run,
        s.elapsed_ms,
        s.gets,
        s.heads,
        mb,
        mbps_of(s)
    );
}

fn print_row_bw(bw: u64, name: &str, mode: &str, run: usize, s: &RunStats) {
    let mb = s.bytes as f64 / 1024.0 / 1024.0;
    println!(
        "{:>7} {:<60} {:<8} {:>3} {:>5}ms {:>5} {:>5} {:>8.2} {:>10.2}",
        bw,
        name,
        mode,
        run,
        s.elapsed_ms,
        s.gets,
        s.heads,
        mb,
        mbps_of(s)
    );
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}
