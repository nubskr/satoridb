use anyhow::{anyhow, Context, Result};
use flate2::read::GzDecoder;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Simple converter that rewrites .fvecs/.bvecs (optionally .gz) into a flat
/// little-endian f32 blob that can be mmap'd or sequentially read quickly.
///
/// Output format: [u32 dim][u64 count][f32 data ...] where data is contiguous
/// row-major (vec0, vec1, ...), no per-vector length prefixes.
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: cargo run --release --bin prepare_dataset -- <input.(fvecs|bvecs)[.gz]> [output.bin]"
        );
        return Ok(());
    }

    let input = PathBuf::from(&args[1]);
    let output = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        input.with_extension("f32bin")
    };

    // Optional batch size for buffered parsing.
    let batch_size: usize = env::var("CONVERT_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    let mut writer = BufWriter::new(File::create(&output)?);
    // Reserve space for header; fill in at the end.
    writer.write_all(&[0u8; 12])?;

    let mut total: u64 = 0;
    let mut dim: Option<u32> = None;

    if is_bvecs(&input) {
        convert_bvecs(&input, &mut writer, &mut total, &mut dim, batch_size)?;
    } else if is_fvecs(&input) {
        convert_fvecs(&input, &mut writer, &mut total, &mut dim, batch_size)?;
    } else {
        return Err(anyhow!(
            "unrecognized input extension (expected .fvecs/.bvecs with optional .gz): {}",
            input.display()
        ));
    }

    writer.flush()?;

    // Backfill header.
    let final_dim = dim.ok_or_else(|| anyhow!("no vectors read"))?;
    let mut f = writer.into_inner()?;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&final_dim.to_le_bytes())?;
    f.write_all(&total.to_le_bytes())?;
    f.flush()?;

    println!(
        "Wrote {} vectors (dim={}) to {}",
        total,
        final_dim,
        output.display()
    );
    Ok(())
}

fn is_gz(path: &Path) -> bool {
    path.extension()
        .map(|e| e.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
}

fn is_fvecs(path: &Path) -> bool {
    path.extension()
        .map(|e| {
            e.eq_ignore_ascii_case("fvecs")
                || (is_gz(path)
                    && path
                        .file_stem()
                        .map_or(false, |s| s.to_string_lossy().ends_with("fvecs")))
        })
        .unwrap_or(false)
}

fn is_bvecs(path: &Path) -> bool {
    path.extension()
        .map(|e| {
            e.eq_ignore_ascii_case("bvecs")
                || (is_gz(path)
                    && path
                        .file_stem()
                        .map_or(false, |s| s.to_string_lossy().ends_with("bvecs")))
        })
        .unwrap_or(false)
}

fn open_reader(path: &Path) -> Result<Box<dyn Read>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    if is_gz(path) {
        Ok(Box::new(GzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

fn convert_fvecs(
    path: &Path,
    writer: &mut BufWriter<File>,
    total: &mut u64,
    dim: &mut Option<u32>,
    batch_size: usize,
) -> Result<()> {
    let reader = open_reader(path)?;
    let mut reader = BufReader::new(reader);
    let mut dim_buf = [0u8; 4];
    let mut data_buf: Vec<u8> = Vec::new();

    loop {
        for _ in 0..batch_size {
            if let Err(e) = reader.read_exact(&mut dim_buf) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    return Ok(());
                }
                return Err(e.into());
            }
            let d = u32::from_le_bytes(dim_buf);
            if let Some(existing) = dim {
                if *existing != d {
                    return Err(anyhow!("dimension changed: {} -> {}", existing, d));
                }
            } else {
                *dim = Some(d);
            }
            let needed = d as usize * 4;
            data_buf.resize(needed, 0);
            reader.read_exact(&mut data_buf)?;
            writer.write_all(&data_buf)?;
            *total += 1;
        }
    }
}

fn convert_bvecs(
    path: &Path,
    writer: &mut BufWriter<File>,
    total: &mut u64,
    dim: &mut Option<u32>,
    batch_size: usize,
) -> Result<()> {
    let reader = open_reader(path)?;
    let mut reader = BufReader::new(reader);
    let mut dim_buf = [0u8; 4];
    let mut data_buf: Vec<u8> = Vec::new();
    let mut f32_buf: Vec<u8> = Vec::new();

    loop {
        for _ in 0..batch_size {
            if let Err(e) = reader.read_exact(&mut dim_buf) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    return Ok(());
                }
                return Err(e.into());
            }
            let d = u32::from_le_bytes(dim_buf);
            if let Some(existing) = dim {
                if *existing != d {
                    return Err(anyhow!("dimension changed: {} -> {}", existing, d));
                }
            } else {
                *dim = Some(d);
            }
            let needed = d as usize;
            data_buf.resize(needed, 0);
            reader.read_exact(&mut data_buf)?;

            // Upcast bytes -> f32
            f32_buf.clear();
            f32_buf.reserve(needed * 4);
            for &b in &data_buf {
                f32_buf.extend_from_slice(&(b as f32).to_le_bytes());
            }
            writer.write_all(&f32_buf)?;
            *total += 1;
        }
    }
}
