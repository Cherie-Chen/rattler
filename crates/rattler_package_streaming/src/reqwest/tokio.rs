//! Functionality to stream and extract packages directly from a
//! [`reqwest::Url`] within a [`tokio`] async context.

use std::{path::Path, sync::Arc};

use fs_err::tokio as tokio_fs;
use futures_util::stream::TryStreamExt;
use rattler_conda_types::package::CondaArchiveType;
use rattler_digest::Sha256Hash;
use reqwest::Response;
use tokio::io::BufReader;
use tokio_util::{either::Either, io::StreamReader};
use tracing;
use url::Url;
use zip::result::ZipError;

use crate::{DownloadReporter, ExtractError, ExtractResult};

/// zip files may use data descriptors to signal that the decompressor needs to
/// seek ahead in the buffer to find the compressed data length.
/// Since we stream the package over a non seek-able HTTP connection, this
/// condition will cause an error during decompression. In this case, we
/// fallback to reading the whole data to a buffer before attempting
/// decompression. Read more in <https://github.com/conda/rattler/issues/794>
const DATA_DESCRIPTOR_ERROR_MESSAGE: &str = "The file length is not available in the local header";

fn error_for_status(response: reqwest::Response) -> reqwest_middleware::Result<Response> {
    response
        .error_for_status()
        .map_err(reqwest_middleware::Error::Reqwest)
}

async fn get_reader(
    url: Url,
    client: reqwest_middleware::ClientWithMiddleware,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
) -> Result<impl tokio::io::AsyncRead, ExtractError> {
    if let Some(reporter) = &reporter {
        reporter.on_download_start();
    }

    if url.scheme() == "file" {
        let file =
            tokio_fs::File::open(url.to_file_path().expect("Could not convert to file path"))
                .await
                .map_err(ExtractError::IoError)?;

        Ok(Either::Left(BufReader::new(file)))
    } else {
        // Send the request for the file
        let mut request = client.get(url.clone());

        if let Some(sha256) = expected_sha256 {
            // This is used by the OCI registry middleware to verify the sha256 of the
            // response
            request = request.header("X-Expected-Sha256", format!("{sha256:x}"));
        }

        let response = request
            .send()
            .await
            .and_then(error_for_status)
            .map_err(ExtractError::ReqwestError)?;

        let total_bytes = response.content_length();
        let mut bytes_received = Box::new(0);
        let byte_stream = response.bytes_stream().inspect_ok(move |frame| {
            *bytes_received += frame.len() as u64;
            if let Some(reporter) = &reporter {
                reporter.on_download_progress(*bytes_received, total_bytes);
            }
        });

        // Get the response as a stream
        Ok(Either::Right(StreamReader::new(byte_stream.map_err(
            |err| {
                if err.is_body() {
                    std::io::Error::new(std::io::ErrorKind::Interrupted, err)
                } else if err.is_decode() {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, err)
                } else {
                    std::io::Error::other(err)
                }
            },
        ))))
    }
}

/// Extracts the contents a `.tar.bz2` package archive from the specified remote
/// location.
///
/// ```rust,no_run
/// # #[tokio::main]
/// # async fn main() {
/// # use std::path::Path;
/// use url::Url;
/// use reqwest::Client;
/// use reqwest_middleware::ClientWithMiddleware;
/// use rattler_package_streaming::reqwest::tokio::extract_tar_bz2;
/// let _ = extract_tar_bz2(
///     ClientWithMiddleware::from(Client::new()),
///     Url::parse("https://conda.anaconda.org/conda-forge/win-64/python-3.11.0-hcf16a7b_0_cpython.tar.bz2").unwrap(),
///     Path::new("/tmp"),
///     None,
///     None)
///     .await
///     .unwrap();
/// # }
/// ```
pub async fn extract_tar_bz2(
    client: reqwest_middleware::ClientWithMiddleware,
    url: Url,
    destination: &Path,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
) -> Result<ExtractResult, ExtractError> {
    let reader = get_reader(url.clone(), client, expected_sha256, reporter.clone()).await?;
    // The `response` is used to stream in the package data
    let result = crate::tokio::async_read::extract_tar_bz2(reader, destination).await?;
    if let Some(reporter) = &reporter {
        reporter.on_download_complete();
    }
    Ok(result)
}

/// Extracts the contents a `.conda` package archive from the specified remote
/// location.
///
/// ```rust,no_run
/// # #[tokio::main]
/// # async fn main() {
/// # use std::path::Path;
/// use rattler_package_streaming::reqwest::tokio::extract_conda;
/// use reqwest::Client;
/// use reqwest_middleware::ClientWithMiddleware;
/// use url::Url;
/// let _ = extract_conda(
///     ClientWithMiddleware::from(Client::new()),
///     Url::parse("https://conda.anaconda.org/conda-forge/linux-64/python-3.10.8-h4a9ceb5_0_cpython.conda").unwrap(),
///     Path::new("/tmp"),
///     None,
///     None)
///     .await
///     .unwrap();
/// # }
/// ```
pub async fn extract_conda(
    client: reqwest_middleware::ClientWithMiddleware,
    url: Url,
    destination: &Path,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
) -> Result<ExtractResult, ExtractError> {
    // The `response` is used to stream in the package data
    let reader = get_reader(
        url.clone(),
        client.clone(),
        expected_sha256,
        reporter.clone(),
    )
    .await?;
    match crate::tokio::async_read::extract_conda(reader, destination).await {
        Ok(result) => {
            if let Some(reporter) = &reporter {
                reporter.on_download_complete();
            }
            Ok(result)
        }
        // https://github.com/conda/rattler/issues/794
        Err(ExtractError::ZipError(ZipError::UnsupportedArchive(zip_error)))
            if (zip_error.contains(DATA_DESCRIPTOR_ERROR_MESSAGE)) =>
        {
            tracing::warn!(
                "Failed to stream decompress conda package from '{}' due to the presence of zip data descriptors. Falling back to non streaming decompression",
                url
            );
            if let Some(reporter) = &reporter {
                reporter.on_download_complete();
            }
            let new_reader =
                get_reader(url.clone(), client, expected_sha256, reporter.clone()).await?;

            match crate::tokio::async_read::extract_conda_via_buffering(new_reader, destination)
                .await
            {
                Ok(result) => {
                    if let Some(reporter) = &reporter {
                        reporter.on_download_complete();
                    }
                    Ok(result)
                }
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    }
}

/// Extracts the contents a package archive from the specified remote location.
/// The type of package is determined based on the path of the url.
///
/// ```rust,no_run
/// # #[tokio::main]
/// # async fn main() {
/// # use std::path::Path;
/// use url::Url;
/// use rattler_package_streaming::reqwest::tokio::extract;
/// use reqwest::Client;
/// use reqwest_middleware::ClientWithMiddleware;
/// let _ = extract(
///     ClientWithMiddleware::from(Client::new()),
///     Url::parse("https://conda.anaconda.org/conda-forge/linux-64/python-3.10.8-h4a9ceb5_0_cpython.conda").unwrap(),
///     Path::new("/tmp"),
///     None,
///     None)
///     .await
///     .unwrap();
/// # }
/// ```
pub async fn extract(
    client: reqwest_middleware::ClientWithMiddleware,
    url: Url,
    destination: &Path,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
) -> Result<ExtractResult, ExtractError> {
    match CondaArchiveType::try_from(Path::new(url.path()))
        .ok_or(ExtractError::UnsupportedArchiveType)?
    {
        CondaArchiveType::TarBz2 => {
            extract_tar_bz2(client, url, destination, expected_sha256, reporter).await
        }
        CondaArchiveType::Conda => {
            extract_conda(client, url, destination, expected_sha256, reporter).await
        }
    }
}

// -----------------------------------------------------------------------------
// Opt-in parallel range-GET download path.
//
// These helpers parallel `extract` / `extract_tar_bz2` / `extract_conda` above
// but accept an additional `ParallelDownloadConfig`.  When the configured
// threshold is met *and* the server advertises `Accept-Ranges: bytes`, the
// package is fetched as N concurrent HTTP `Range` GETs into a temp file and
// then handed to the existing extractor.  Otherwise the implementation falls
// back to the single full-object GET path used by [`get_reader`].
//
// Default behavior is unchanged; existing callers of [`extract`] etc. are not
// affected.
// -----------------------------------------------------------------------------

use super::parallel_download::{ParallelDownloadConfig, parallel_download_to_temp};

/// Returns a reader over the package body, possibly via parallel range GETs.
///
/// On any failure of the parallel path (small object, no range support on the
/// server, partial chunk failures), the function transparently falls back to
/// the single-GET path used by [`get_reader`].
async fn get_reader_with_config(
    url: Url,
    client: reqwest_middleware::ClientWithMiddleware,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
    config: &ParallelDownloadConfig,
) -> Result<
    Either<BufReader<tokio_fs::File>, Box<dyn tokio::io::AsyncRead + Unpin + Send>>,
    ExtractError,
> {
    if let Some(reporter) = &reporter {
        reporter.on_download_start();
    }

    if url.scheme() == "file" {
        let file =
            tokio_fs::File::open(url.to_file_path().expect("Could not convert to file path"))
                .await
                .map_err(ExtractError::IoError)?;
        return Ok(Either::Left(BufReader::new(file)));
    }

    // Try the parallel path first.  We deliberately retain the temp file in
    // scope inside this function until the reader is consumed: dropping
    // `NamedTempFile` removes the on-disk file.  We forward both the file
    // handle and the temp guard out to the caller via a small wrapper that
    // owns both.
    //
    // Skip the parallel path when an `expected_sha256` is provided.  The OCI
    // registry middleware verifies the response body using the
    // `X-Expected-Sha256` request header, which only works for a single
    // full-body GET — it cannot validate against a stream of per-chunk range
    // GETs.  Falling back to single-GET preserves the integrity check.
    if expected_sha256.is_none()
        && let Some(temp) = parallel_download_to_temp(&client, &url, config)
            .await
            .unwrap_or(None)
    {
        let path = temp.path().to_path_buf();
        let f = tokio_fs::File::open(&path)
            .await
            .map_err(ExtractError::IoError)?;
        let reader = TempFileReader {
            inner: BufReader::new(f),
            _guard: temp,
        };
        return Ok(Either::Right(
            Box::new(reader) as Box<dyn tokio::io::AsyncRead + Unpin + Send>
        ));
    }

    // Fallback: single full-object GET (original code path).
    let mut request = client.get(url.clone());
    if let Some(sha256) = expected_sha256 {
        request = request.header("X-Expected-Sha256", format!("{sha256:x}"));
    }
    let response = request
        .send()
        .await
        .and_then(error_for_status)
        .map_err(ExtractError::ReqwestError)?;

    let total_bytes = response.content_length();
    let mut bytes_received = Box::new(0u64);
    let byte_stream = response.bytes_stream().inspect_ok(move |frame| {
        *bytes_received += frame.len() as u64;
        if let Some(reporter) = &reporter {
            reporter.on_download_progress(*bytes_received, total_bytes);
        }
    });
    Ok(Either::Right(
        Box::new(StreamReader::new(byte_stream.map_err(|err| {
            if err.is_body() {
                std::io::Error::new(std::io::ErrorKind::Interrupted, err)
            } else if err.is_decode() {
                std::io::Error::new(std::io::ErrorKind::InvalidData, err)
            } else {
                std::io::Error::other(err)
            }
        }))) as Box<dyn tokio::io::AsyncRead + Unpin + Send>,
    ))
}

/// `AsyncRead` wrapper that owns the temp-file handle and its `NamedTempFile`
/// guard, ensuring the on-disk temp file is removed when the reader is
/// dropped.
struct TempFileReader {
    inner: BufReader<tokio_fs::File>,
    // `_guard` is held only for its `Drop` side effect (deleting the temp file).
    _guard: tempfile::NamedTempFile,
}

impl tokio::io::AsyncRead for TempFileReader {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

/// Same as [`extract_tar_bz2`], but with an opt-in parallel range-GET fast path
/// controlled by `config`.
pub async fn extract_tar_bz2_with_config(
    client: reqwest_middleware::ClientWithMiddleware,
    url: Url,
    destination: &Path,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
    config: &ParallelDownloadConfig,
) -> Result<ExtractResult, ExtractError> {
    let reader =
        get_reader_with_config(url, client, expected_sha256, reporter.clone(), config).await?;
    let result = crate::tokio::async_read::extract_tar_bz2(reader, destination).await?;
    if let Some(reporter) = &reporter {
        reporter.on_download_complete();
    }
    Ok(result)
}

/// Same as [`extract_conda`], but with an opt-in parallel range-GET fast path
/// controlled by `config`.
pub async fn extract_conda_with_config(
    client: reqwest_middleware::ClientWithMiddleware,
    url: Url,
    destination: &Path,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
    config: &ParallelDownloadConfig,
) -> Result<ExtractResult, ExtractError> {
    let reader = get_reader_with_config(
        url.clone(),
        client.clone(),
        expected_sha256,
        reporter.clone(),
        config,
    )
    .await?;
    match crate::tokio::async_read::extract_conda(reader, destination).await {
        Ok(result) => {
            if let Some(reporter) = &reporter {
                reporter.on_download_complete();
            }
            Ok(result)
        }
        Err(ExtractError::ZipError(ZipError::UnsupportedArchive(zip_error)))
            if (zip_error.contains(DATA_DESCRIPTOR_ERROR_MESSAGE)) =>
        {
            tracing::warn!(
                "Failed to stream decompress conda package from '{}' due to the presence of zip data descriptors. Falling back to non streaming decompression",
                url
            );
            if let Some(reporter) = &reporter {
                reporter.on_download_complete();
            }
            let new_reader =
                get_reader_with_config(url, client, expected_sha256, reporter.clone(), config)
                    .await?;
            let result =
                crate::tokio::async_read::extract_conda_via_buffering(new_reader, destination)
                    .await?;
            if let Some(reporter) = &reporter {
                reporter.on_download_complete();
            }
            Ok(result)
        }
        Err(e) => Err(e),
    }
}

/// Same as [`extract`], but with an opt-in parallel range-GET fast path
/// controlled by `config`.
pub async fn extract_with_config(
    client: reqwest_middleware::ClientWithMiddleware,
    url: Url,
    destination: &Path,
    expected_sha256: Option<Sha256Hash>,
    reporter: Option<Arc<dyn DownloadReporter>>,
    config: &ParallelDownloadConfig,
) -> Result<ExtractResult, ExtractError> {
    match CondaArchiveType::try_from(Path::new(url.path()))
        .ok_or(ExtractError::UnsupportedArchiveType)?
    {
        CondaArchiveType::TarBz2 => {
            extract_tar_bz2_with_config(client, url, destination, expected_sha256, reporter, config)
                .await
        }
        CondaArchiveType::Conda => {
            extract_conda_with_config(client, url, destination, expected_sha256, reporter, config)
                .await
        }
    }
}
