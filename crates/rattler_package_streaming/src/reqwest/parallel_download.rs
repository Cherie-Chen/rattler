//! Optional parallel range-GET package download.
//!
//! This module provides an opt-in alternative to the single full-object GET
//! used by [`super::tokio::extract`] et al. When enabled, it splits a remote
//! Conda package into `N` byte ranges and fetches them concurrently using
//! HTTP `Range` requests, writing the chunks into a temporary file in offset
//! order.  The file is then handed to the existing extractor.
//!
//! ## Why this exists
//!
//! On bandwidth-constrained network paths (cross-region object stores,
//! per-flow-rate-limited links, certain virtualised NICs) a single TLS stream
//! can leave aggregate NIC bandwidth on the table.  Splitting one large object
//! into multiple concurrent connections is the same trick boto3's
//! `TransferManager`, `aws-cli`, and `HuggingFace`'s `hf_transfer` use for the
//! same reason.
//!
//! ## When this is *not* a win
//!
//! - For small packages (< `min_size_for_parallel`) the extra HTTP round-trips
//!   strictly hurt.  The implementation falls back to a single GET in that case.
//! - For same-region S3↔EC2 traffic on a single NIC, single-connection
//!   throughput is usually already saturating.  Parallel range GETs increase
//!   per-download object-store request count without measurable wall-clock
//!   improvement.
//!
//! Default behavior is unchanged; callers must explicitly opt in by calling the
//! `*_with_config` entry points and supplying a [`ParallelDownloadConfig`].
//!
//! ## Compatibility caveats
//!
//! - **OCI / `X-Expected-Sha256`**: rattler's OCI registry middleware uses an
//!   `X-Expected-Sha256` request header to verify the response body in one
//!   shot.  That cannot work against per-chunk range GETs.  The caller in
//!   [`super::tokio::get_reader_with_config`] therefore skips the parallel
//!   path entirely when an `expected_sha256` is supplied; we keep the safer
//!   single-GET path so the integrity check still fires.
//! - **Servers that ignore `Range`**: some servers respond `200 OK` with the
//!   full body even when `Range:` is sent.  We surface that as a per-chunk
//!   error and let the caller fall back to a plain single GET.
//!
//! ## Known POC limitation
//!
//! Each chunk task currently does its own `open + seek + write_all` against a
//! pre-sized temp file.  This works correctly (the offsets do not overlap) but
//! a production-quality implementation should funnel writes through a single
//! writer task using `pwrite`-style positioned writes (`tokio::task::spawn_blocking`
//! with `std::os::unix::fs::FileExt::write_all_at`, or platform equivalent).
//! Doing so will reduce inode/lock contention on the shared file under high
//! concurrency.  Left out of this revision deliberately to keep the diff
//! reviewable; flagged here so a follow-up can address it.

use fs_err::tokio as tokio_fs;
use futures::stream::{self, StreamExt};
use reqwest::header::{ACCEPT_RANGES, CONTENT_LENGTH, RANGE};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use url::Url;

use crate::ExtractError;

/// Hard upper bound on `chunk_size`.  An attacker-controlled or accidental
/// configuration with a huge `chunk_size` could otherwise consume gigabytes of
/// memory per concurrent task (`reqwest::Response::bytes()` buffers the entire
/// response body).  We refuse to engage the parallel path above this and fall
/// back to single-GET so that misconfiguration is at worst a perf miss.
pub const MAX_CHUNK_SIZE: u64 = 256 * 1024 * 1024; // 256 MiB

/// Configuration for opt-in parallel range-GET downloads.
///
/// Created via [`ParallelDownloadConfig::default`] (which mirrors boto3's
/// `TransferConfig` defaults: 8 MiB threshold, 8 MiB chunks, 8-way concurrency)
/// or via the explicit constructor.
#[derive(Debug, Clone)]
pub struct ParallelDownloadConfig {
    /// Object size threshold (bytes) above which to use parallel range GETs.
    /// Below this, fall back to a single full-object GET.
    pub min_size_for_parallel: u64,

    /// Bytes per range GET.  Must be > 0 and <= [`MAX_CHUNK_SIZE`].
    pub chunk_size: u64,

    /// Maximum concurrent range GETs per object.  Must be > 0.
    pub concurrency: usize,
}

impl Default for ParallelDownloadConfig {
    fn default() -> Self {
        // Match boto3 TransferConfig defaults so behavior is familiar to anyone
        // who has used aws-cli / boto3 multipart downloads.
        Self {
            min_size_for_parallel: 8 * 1024 * 1024, // 8 MiB
            chunk_size: 8 * 1024 * 1024,            // 8 MiB
            concurrency: 8,
        }
    }
}

impl ParallelDownloadConfig {
    /// Returns `true` when the config is internally valid and would actually
    /// split downloads (not silently disabled).
    fn is_valid(&self) -> bool {
        self.chunk_size > 0
            && self.chunk_size <= MAX_CHUNK_SIZE
            && self.concurrency > 0
            && self.min_size_for_parallel > 0
    }
}

/// Probe the server for total size and range support.  Returns `Ok(None)` if
/// the server does not advertise byte-range support or omits a Content-Length.
async fn probe_object(
    client: &reqwest_middleware::ClientWithMiddleware,
    url: &Url,
) -> Result<Option<u64>, ExtractError> {
    let resp = client
        .head(url.clone())
        .send()
        .await
        .map_err(ExtractError::ReqwestError)?;
    let resp = resp
        .error_for_status()
        .map_err(|e| ExtractError::ReqwestError(reqwest_middleware::Error::Reqwest(e)))?;

    let supports_range = resp
        .headers()
        .get(ACCEPT_RANGES)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("bytes"));

    let total = resp
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    Ok(if supports_range { total } else { None })
}

/// Attempt to download `url` into a temp file using `config.concurrency`
/// parallel HTTP `Range: bytes=…` GETs.
///
/// Returns `Ok(Some(temp_file))` on success.  Returns `Ok(None)` if the
/// caller should fall back to the single-GET path (object too small, server
/// does not support ranges, or the server doesn't advertise Content-Length).
///
/// On success the caller is responsible for opening / streaming the returned
/// temp file.  We use [`tempfile::NamedTempFile`] so the file is automatically
/// cleaned up when the wrapper is dropped.
pub async fn parallel_download_to_temp(
    client: &reqwest_middleware::ClientWithMiddleware,
    url: &Url,
    config: &ParallelDownloadConfig,
) -> Result<Option<tempfile::NamedTempFile>, ExtractError> {
    // Validate config up-front so a misconfiguration costs us nothing on the
    // network (no HEAD round-trip).  Silently disable rather than error so
    // callers fall back transparently.
    if !config.is_valid() {
        return Ok(None);
    }

    let total = match probe_object(client, url).await? {
        Some(t) if t >= config.min_size_for_parallel => t,
        _ => return Ok(None),
    };

    // Build the list of (start, end_inclusive) byte ranges.
    let chunk_size = config.chunk_size;
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let mut start: u64 = 0;
    while start < total {
        let end = std::cmp::min(start + chunk_size - 1, total - 1);
        ranges.push((start, end));
        start = end + 1;
    }

    let tmp = tempfile::NamedTempFile::new().map_err(ExtractError::IoError)?;
    let path = tmp.path().to_path_buf();

    // Pre-size the file so each chunk task can seek+write to its own offset.
    // Sparse on Linux ext4/tmpfs (no physical block allocation until write).
    {
        let std_file = fs_err::File::create(&path).map_err(ExtractError::IoError)?;
        std_file.set_len(total).map_err(ExtractError::IoError)?;
    }

    // Fan out: N concurrent range GETs.  Each writes its chunk into the
    // pre-sized file at the correct offset.  See the module-level "Known POC
    // limitation" note about funnelling writes through a single writer task in
    // a follow-up revision.
    let path_for_tasks = path.clone();
    let results: Vec<Result<u64, ExtractError>> = stream::iter(ranges)
        .map(|(rng_start, rng_end)| {
            let client = client.clone();
            let url = url.clone();
            let path = path_for_tasks.clone();
            async move {
                let resp = client
                    .get(url.clone())
                    .header(RANGE, format!("bytes={rng_start}-{rng_end}"))
                    .send()
                    .await
                    .map_err(ExtractError::ReqwestError)?;
                // S3-compatible servers return 206 Partial Content for range
                // requests.  Some servers ignore the Range header and return
                // the full body with 200 OK; we detect that and surface a
                // generic IO error so the caller can fall back to single-GET.
                let status = resp.status();
                if status.as_u16() != 206 {
                    return Err(ExtractError::IoError(std::io::Error::other(format!(
                        "expected 206 Partial Content for range request, got {status}"
                    ))));
                }
                let bytes = resp.bytes().await.map_err(|e| {
                    ExtractError::ReqwestError(reqwest_middleware::Error::Reqwest(e))
                })?;

                // Open the shared pre-sized file and write at the correct offset.
                // tokio's File::flush is a no-op; the kernel flushes on close.
                let mut f = tokio_fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .await
                    .map_err(ExtractError::IoError)?;
                f.seek(std::io::SeekFrom::Start(rng_start))
                    .await
                    .map_err(ExtractError::IoError)?;
                f.write_all(&bytes).await.map_err(ExtractError::IoError)?;
                Ok::<u64, ExtractError>(bytes.len() as u64)
            }
        })
        .buffer_unordered(config.concurrency)
        .collect()
        .await;

    // Surface the first error if any.
    for r in results {
        r?;
    }

    Ok(Some(tmp))
}
