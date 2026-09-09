//! Offline Qdrant shard builder.
//!
//! Builds restore-ready shard artifacts from a partitioned dataset, so that a
//! network-filesystem-backed cluster sees one large sequential copy instead of a
//! continuous stream of small writes and fsyncs.
//!
//! **Port status — complete (stages 1-5).** The pipeline is here: `scatter` (phase 1, through the
//! `InputStore` seam), `plan` (phase 2), `build` (phase 3, bulk and edge routes — the bulk
//! route runs on this fork's own `SegmentBuilder::update_from_points`), plus the pre-flight
//! and debugging commands (`validate`, `inspect-parquet`, `scatter-verify`, `dump-part`,
//! `shard-of`), phase 4 (`assemble`, `assemble-verify`), the live-cluster commands
//! (`verify-config`, `verify-placement`, `install-plan`), and `retarget` (rewrite built
//! segments' recorded configs to new memory placements — see `retarget.rs`). Still to come:
//! the S3 `InputStore` backend (see `store.rs`).

mod assemble;
mod build;
#[cfg(test)]
mod build_e2e_tests;
mod config;
mod dense_codec;
mod document;
mod parquet_source;
mod partfile;
mod payload_index;
mod placement;
mod plan;
mod progress;
mod retarget;
mod ring;
mod scatter;
mod source;
mod store;
mod verify;

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use segment::types::ExtendedPointId;

use crate::config::LoadedConfig;
use crate::ring::ShardRouter;
use crate::store::LocalStore;

#[derive(Parser)]
#[command(
    name = "qdrant-shard-builder",
    about = "Build Qdrant shards offline, then restore them into a cluster",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// CLI spelling of [`build::BuildRoute`].
///
/// A separate type so `clap` derives the flag values here rather than putting a CLI concern into
/// the build module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BuildRouteArg {
    /// Fill the segment's storages directly, then build its indexes once.
    Bulk,
    /// Upsert into a staging shard and let its optimizers produce the segment.
    Edge,
}

impl From<BuildRouteArg> for build::BuildRoute {
    fn from(arg: BuildRouteArg) -> Self {
        match arg {
            BuildRouteArg::Bulk => build::BuildRoute::Bulk,
            BuildRouteArg::Edge => build::BuildRoute::Edge,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Check a config document, print the resolved plan parameters and fingerprint.
    Validate {
        /// The input document: a resolved collection config (Qdrant's `config.json`, or
        /// `result.config` from `GET /collections/{name}`) under a `collection` key, with
        /// optional `mapping` and `payload_index` sections.
        #[arg(long)]
        config: PathBuf,

        /// Total number of points in the dataset, to project the segment plan.
        ///
        /// Points rather than bytes: segment sizing is measured against *vector storage*
        /// bytes, which is `points x bytes_per_point` from the config. Passing a dataset
        /// size would invite comparing compressed parquet bytes (which include payload
        /// text) against a vector-storage threshold, over-estimating segment count several
        /// fold.
        #[arg(long)]
        points: Option<u64>,
    },

    /// Phase 1: partition input files by shard. Resumable; safe to re-run.
    Scatter {
        #[arg(long)]
        config: PathBuf,

        /// Input root: a directory tree (or single file) of .jsonl/.ndjson/.parquet files,
        /// searched recursively. (An s3:// URL becomes valid here when the S3 InputStore
        /// backend lands.)
        #[arg(long)]
        input: PathBuf,

        /// Working directory for scatter output. Reused across runs to resume.
        #[arg(long)]
        work: PathBuf,

        /// Reader threads. Bounded by workers * shards open output files.
        #[arg(long, default_value_t = 4)]
        workers: usize,

        /// Read only files of this format, ignoring anything else in the tree.
        ///
        /// A corpus directory often holds more than the data — `.jsonl` sidecars, manifests, notes.
        /// Every recognised extension is otherwise a candidate, so those are picked up and fail on
        /// read. Naming the format excludes them at discovery instead.
        #[arg(long)]
        input_format: Option<crate::source::InputFormat>,

        /// Stop after this many unreadable files.
        ///
        /// Individual bad files are skipped and reported rather than ending the run, since one file
        /// among thousands should not discard hours of completed work. But a mapping that does not
        /// match the corpus fails *every* file, and that should be caught in the first minute
        /// rather than after reading everything.
        #[arg(long, default_value_t = 10)]
        max_failures: usize,

        /// Process only this process's share of the input, as `I/N` (e.g. `--slice 3/10`).
        ///
        /// For splitting one corpus across several machines. Every process is given the *same*
        /// `--input` and `--work`, and differs only in `I`. Inputs are discovered in sorted
        /// order and taken by stride, so the shares are disjoint without any coordination, and
        /// one machine's share can be re-run alone after a failure.
        ///
        /// Point every process at the same `--input` root: part files are named after a hash of
        /// the store-relative path, so identical roots make an overlapping assignment harmless —
        /// the second write lands on the same name.
        #[arg(long)]
        slice: Option<String>,
    },

    /// Print a Parquet file's schema and row-group footprint, to build a mapping from.
    InspectParquet {
        /// Parquet file to inspect.
        file: PathBuf,
    },

    /// Phase 2: group scattered parts into segments. Writes one plan per shard.
    ///
    /// One small metadata read per part file, of which there are as many as inputs times shards, so
    /// it is parallel, splittable by shard with `--slice`, and resumable — a shard whose plan
    /// already exists is skipped without reading any of its parts.
    Plan {
        #[arg(long)]
        config: PathBuf,

        /// Scatter working directory, which also receives the plan.
        #[arg(long)]
        work: PathBuf,

        /// Threads reading part metadata.
        #[arg(long, default_value_t = 8)]
        workers: usize,

        /// Plan only this machine's share of the shards, as `I/N`.
        #[arg(long)]
        slice: Option<String>,

        /// Re-plan shards that already have a plan.
        ///
        /// A plan is a pure function of the parts and the config, so replanning unchanged input
        /// reproduces it exactly — including the segment UUIDs phase 3 is keyed on.
        #[arg(long)]
        replan: bool,
    },

    /// Phase 3: build indexed segments from the plan. Resumable; segments build in parallel.
    Build {
        #[arg(long)]
        config: PathBuf,

        /// Scatter working directory containing `plan/`.
        #[arg(long)]
        work: PathBuf,

        /// Output root: `<out>/shard_{id}/segments/<uuid>`.
        #[arg(long)]
        out: PathBuf,

        /// Staging root, where each segment is built before being published to --out.
        ///
        /// May be a different filesystem from --out. When it is, a finished segment is copied
        /// rather than renamed into place; when it is not, the publish is a free rename.
        ///
        /// Pointing this at RAM (`/dev/shm`) is the reason the cross-filesystem case is supported.
        /// The index phase re-reads the dense and quantized vectors in graph order, so on a shared
        /// filesystem every cache miss is a network round trip; staging in RAM makes those memory
        /// accesses and pays one sequential write at the end. Budget for it: the segment occupies
        /// RAM until it is published, and tmpfs pages cannot be swapped.
        #[arg(long)]
        staging: PathBuf,

        /// Only build this shard. Omit to build every planned shard.
        #[arg(long)]
        shard: Option<u32>,

        /// Concurrent segment builds. Defaults to cores / 2 (measured cores-per-build).
        #[arg(long)]
        concurrency: Option<usize>,

        /// Threads one segment's index build may use.
        ///
        /// This is the build box's resource decision, made here rather than in the config
        /// document: the document's `hnsw_config.max_indexing_threads` belongs to the serving
        /// cluster and is what gets stamped into the artifact regardless of this flag. Defaults
        /// to the document's resolved value.
        #[arg(long)]
        indexing_threads: Option<usize>,

        /// Points per update batch while streaming parts into the segment.
        #[arg(long, default_value_t = 1024)]
        batch_points: usize,

        /// Build only this machine's share of the segments, as `I/N` (e.g. `--slice 3/10`).
        ///
        /// For splitting the build across machines. Every process is given the same `--work` and
        /// `--out` and differs only in `I`; `--staging` should be node-local. Segments are taken by
        /// stride from a deterministic flat list, so the shares are disjoint without coordination
        /// and one machine's share can be re-run alone after a failure.
        ///
        /// Striding at segment granularity rather than splitting by shard, because shards differ
        /// in size by roughly the hash ring's skew, and because it lets the node count differ
        /// from the shard count.
        ///
        /// Composes with `--shard`: that narrows to one shard, then the slice divides its segments.
        #[arg(long)]
        slice: Option<String>,

        /// How each segment is constructed.
        ///
        /// `bulk` (the default) fills the segment's storages directly and then builds its indexes
        /// once. `edge` upserts into a staging shard and lets its optimizers produce the segment,
        /// which is what this tool did originally.
        ///
        /// `edge` is retained because it is the reference implementation — its output is known to
        /// satisfy the optimizers the serving cluster runs — so it is worth building a sample both
        /// ways and comparing. It is also markedly slower on any collection with sparse vectors:
        /// the staging segment builds a mutable sparse index through `PostingList::upsert` that the
        /// optimizer then discards and rebuilds in bulk. Measured at ~78% of total build time on
        /// the FineWeb corpus.
        #[arg(long, value_enum, default_value_t = BuildRouteArg::Bulk)]
        route: BuildRouteArg,
    },

    /// Re-read scatter output and check every point is in the shard the ring chose.
    ///
    /// A second full pass over the intermediate, so it takes the same shape as `scatter`: parallel
    /// over part files, splittable across machines with `--slice`, and resumable — a part already
    /// verified is skipped unless its length changed.
    ScatterVerify {
        #[arg(long)]
        config: PathBuf,

        /// Scatter working directory to inspect.
        #[arg(long)]
        work: PathBuf,

        /// Reader threads.
        #[arg(long, default_value_t = 8)]
        workers: usize,

        /// Verify only this machine's share of the part files, as `I/N`.
        #[arg(long)]
        slice: Option<String>,

        /// Re-check parts an earlier run already verified.
        #[arg(long)]
        recheck: bool,

        /// Read every record instead of checking each part's header and recorded length.
        ///
        /// The default pass catches what can actually go wrong with a file — a truncated part, a
        /// part written under a different config, a part in the wrong shard's directory — from two
        /// small reads each. Re-deriving every point's shard adds nothing to that: it is the same
        /// function of the id in the same binary, so it cannot disagree.
        ///
        /// `--deep` reads the records anyway. It is the only thing that catches damage in the middle
        /// of a part that left its length unchanged, and it checks the sidecar's record count, which
        /// the plan phase sizes segments from. Worth one run over a corpus; not worth every run.
        #[arg(long)]
        deep: bool,
    },

    /// Print the first records of a part file as JSON. Debugging aid.
    DumpPart {
        #[arg(long)]
        config: PathBuf,

        /// Part file to read.
        file: PathBuf,

        /// Records to print.
        #[arg(long, default_value_t = 1)]
        limit: usize,
    },

    /// Compare a config document against a live collection.
    VerifyConfig {
        #[arg(long)]
        config: PathBuf,

        /// Qdrant base URL, e.g. http://localhost:6333
        #[arg(long)]
        url: String,

        /// Collection name.
        #[arg(long)]
        collection: String,
    },

    /// Phase 4: add WAL and shard metadata so each shard_N is restorable.
    Assemble {
        #[arg(long)]
        config: PathBuf,

        /// Output root produced by the build phase.
        #[arg(long)]
        out: PathBuf,
    },

    /// Check assembled shards without starting Qdrant.
    AssembleVerify {
        #[arg(long)]
        out: PathBuf,
    },

    /// Rewrite built segments' recorded configs to the document's memory placements.
    ///
    /// The sanctioned way to change cold/cached placements *after* a build: the document may
    /// differ from what the artifact was built under only in placement (dense/HNSW/payload
    /// memory, sparse cold-vs-cached), and every segment's recorded config is rewritten to
    /// exactly what a fresh build under this document would record. No data file is touched.
    /// Anything structural — HNSW m, quantization, dims, a sparse pinned flip — is refused by
    /// name; those need a re-plan and re-build. Runs before or after `assemble`; idempotent.
    Retarget {
        #[arg(long)]
        config: PathBuf,

        /// Output root produced by the build phase.
        #[arg(long)]
        out: PathBuf,

        /// Report what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
    },

    /// Verify each shard's artifact is serving as that shard on a live cluster.
    ///
    /// Misplacement is silent: reads fan out so they still succeed, and only writes reveal it
    /// (as duplicated ids). This asks each shard directly instead.
    VerifyPlacement {
        #[arg(long)]
        config: PathBuf,

        /// Scatter working directory the artifacts were built from.
        #[arg(long)]
        work: PathBuf,

        /// REST URL of a peer, repeated once per peer (`--url a --url b`).
        ///
        /// Every peer holding a shard must be listed. The shard-scoped read this check relies on
        /// is local-only and is not forwarded, so a peer can only answer for the shards it holds;
        /// with one shard per node, one URL verifies one shard.
        #[arg(long = "url", required = true, num_args = 1..)]
        urls: Vec<String>,

        #[arg(long)]
        collection: String,

        /// Point ids sampled per shard.
        #[arg(long, default_value_t = 50)]
        per_shard: usize,
    },

    /// Print which peer each shard's artifact must be installed on.
    InstallPlan {
        #[arg(long)]
        url: String,

        #[arg(long)]
        collection: String,
    },

    /// Print the shard a point id routes to. Debugging aid for routing questions.
    ShardOf {
        #[arg(long)]
        config: PathBuf,

        /// Point ids: integers or UUIDs.
        ids: Vec<String>,
    },
    // NOTE(post-stage-5): `--input` currently accepts local paths only; the S3 `InputStore`
    // backend (over the `object_store` crate — see `store.rs`) makes `s3://bucket/prefix`
    // valid here, selected by scheme.
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // `collection` and `segment` read global feature flags during config resolution and
    // segment construction. Without this they log "Feature flags not initialized!" and fall
    // back to defaults. Set defaults explicitly so the builder's behaviour is stated rather
    // than inherited.
    //
    // These must match the serving cluster: `appendable_quantization` and
    // `single_file_mmap_vector_storage` both change how segments are written, so a builder
    // running with different flags than the nodes would produce segments the cluster
    // disagrees with. A later stage should read them from the config document rather than
    // defaulting.
    common::flags::init_feature_flags(common::flags::FeatureFlags::default());

    // Segment construction consults this when choosing multi-mmap layouts, and warns loudly if
    // it was never set. The server sets it from a filesystem probe; the builder writes to a
    // local staging filesystem it controls, so declare support rather than probing. If a
    // staging filesystem turns out not to support multi-mmap, this becomes a probe like the
    // server's.
    let _ = common::mmap::MULTI_MMAP_SUPPORT_CHECK_RESULT.set(true);

    let cli = Cli::parse();

    match cli.command {
        Command::Validate { config, points } => cmd_validate(&config, points),
        Command::Scatter {
            config,
            input,
            work,
            workers,
            input_format,
            max_failures,
            slice,
        } => cmd_scatter(ScatterArgs {
            config_path: &config,
            input: &input,
            work: &work,
            workers,
            slice: slice.as_deref(),
            input_format,
            max_failures,
        }),
        Command::InspectParquet { file } => {
            // A single-file root: the store lists (and resolves) exactly this file.
            let store = LocalStore::new(&file);
            print!("{}", parquet_source::describe(&store, &file)?);
            Ok(())
        }
        Command::Plan {
            config,
            work,
            workers,
            slice,
            replan,
        } => cmd_plan(&config, &work, workers, slice.as_deref(), replan),
        Command::Build {
            config,
            work,
            out,
            staging,
            shard,
            concurrency,
            indexing_threads,
            batch_points,
            slice,
            route,
        } => cmd_build(BuildArgs {
            config_path: &config,
            work: &work,
            out: &out,
            staging: &staging,
            shard,
            concurrency,
            indexing_threads,
            batch_points,
            slice: slice.as_deref(),
            route,
        }),
        Command::ScatterVerify {
            config,
            work,
            workers,
            slice,
            recheck,
            deep,
        } => cmd_scatter_verify(&config, &work, workers, slice.as_deref(), recheck, deep),
        Command::DumpPart {
            config,
            file,
            limit,
        } => cmd_dump_part(&config, &file, limit),
        Command::VerifyConfig {
            config,
            url,
            collection,
        } => cmd_verify_config(&config, &url, &collection),
        Command::Assemble { config, out } => cmd_assemble(&config, &out),
        Command::AssembleVerify { out } => cmd_assemble_verify(&out),
        Command::Retarget {
            config,
            out,
            dry_run,
        } => cmd_retarget(&config, &out, dry_run),
        Command::VerifyPlacement {
            config,
            work,
            urls,
            collection,
            per_shard,
        } => cmd_verify_placement(&config, &work, &urls, &collection, per_shard),
        Command::InstallPlan { url, collection } => cmd_install_plan(&url, &collection),
        Command::ShardOf { config, ids } => cmd_shard_of(&config, &ids),
    }
}

fn cmd_verify_config(config_path: &Path, url: &str, collection: &str) -> Result<()> {
    let document = document::load(config_path)?;
    let loaded = &document.config;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start async runtime")?;

    let report = runtime.block_on(verify::verify(
        url,
        collection,
        loaded,
        document.payload_index.as_ref(),
    ))?;

    println!("collection: {}", report.collection);
    println!("local fingerprint:  {}", report.local_fingerprint);
    println!("remote fingerprint: {}", report.remote_fingerprint);

    if report.matches() && report.differences.is_empty() {
        println!(
            "
OK - config matches the live collection"
        );
        return Ok(());
    }

    println!(
        "
Differences over the rebuild-triggering surface:"
    );
    for difference in &report.differences {
        println!("  {difference}");
    }

    if report.differences.is_empty() {
        println!(
            "  (none named, but fingerprints differ - a hashed field is not covered by the \
             field-level diff; this is a bug in the builder, please report it)"
        );
    }

    bail!(
        "config does not match the live collection. Building against this config would \
         produce segments the cluster rewrites on restore."
    );
}

fn cmd_assemble(config_path: &Path, out: &Path) -> Result<()> {
    let document = document::load(config_path)?;
    let loaded = &document.config;
    let payload_index = document.payload_index.as_ref();
    if let Some(schema) = payload_index {
        println!("payload indexes recorded: {}", schema.len());
    }
    let shards = assemble::discover_shards(out)?;

    println!("shards: {}", shards.len());

    for (shard_id, shard_dir) in &shards {
        let report = assemble::assemble_shard(loaded, shard_dir, *shard_id, payload_index)?;
        println!(
            "  shard {}: {} segments, max version {}, wal base {}",
            report.shard_id, report.segments, report.max_segment_version, report.wal_base,
        );
    }

    for (_, shard_dir) in &shards {
        assemble::verify_assembled(shard_dir)?;
    }

    println!(
        "\nEach shard_N is now a restorable shard directory. To install with no copy, stage \
         each shard_N under its serving node's `storage.shard_adoption_path` (same \
         filesystem as the storage, `storage.temp_path` unset) and either create the \
         collection with `shard_placement` + `adopt_shards_from` (one call), or adopt per \
         shard via the snapshot recover endpoint with an `adopt://` location. The tar \
         snapshot recover API remains the fallback where staging is not possible."
    );
    // Conditional, because the answer is no longer always "none": the build phase creates the
    // declared indexes inside each segment. Printing the warning unconditionally told an operator
    // who *had* declared indexes that none existed, which is the opposite of the truth and would
    // send them looking for a problem that is not there.
    match payload_index {
        Some(schema) => println!(
            "\nPayload field indexes: {} field(s), built into each segment by the build phase, so \
             LocalShard::load has none left to build.",
            schema.len(),
        ),
        None => println!(
            "\nNote: no payload field indexes were created. If the collection declares any, \
             LocalShard::load builds them at load time on the target filesystem, which is slow \
             on a large shard."
        ),
    }
    println!("\nOK");
    Ok(())
}

fn cmd_retarget(config_path: &Path, out: &Path, dry_run: bool) -> Result<()> {
    let document = document::load(config_path)?;
    let loaded = &document.config;

    let report = retarget::run(loaded, out, &retarget::RetargetOptions { dry_run })?;

    for change in &report.changes {
        println!("  {change}");
    }
    println!(
        "{}: {} shards, {} segments examined, {} {}",
        if dry_run { "dry run" } else { "retargeted" },
        report.shards,
        report.segments_examined,
        report.segments_rewritten,
        if dry_run {
            "would be rewritten (nothing was written)"
        } else {
            "rewritten"
        },
    );
    if !dry_run && report.segments_rewritten > 0 {
        println!(
            "\nThe recorded configs now match this document's placements. Keep serving and \
             verifying (verify-config, verify-placement) with this document from here on."
        );
    }
    Ok(())
}

fn cmd_assemble_verify(out: &Path) -> Result<()> {
    let shards = assemble::discover_shards(out)?;

    for (shard_id, shard_dir) in &shards {
        assemble::verify_assembled(shard_dir)
            .with_context(|| format!("shard {shard_id} is not restorable"))?;
        println!("  shard {shard_id}: OK");
    }

    println!("\n{} shard(s) verified", shards.len());
    Ok(())
}

fn cmd_verify_placement(
    config_path: &Path,
    work: &Path,
    urls: &[String],
    collection: &str,
    per_shard: usize,
) -> Result<()> {
    let loaded = &document::load(config_path)?.config;
    let router = ShardRouter::new(loaded)?;
    let layout = scatter::ScatterLayout::new(work);

    let report = placement::verify(loaded, &router, &layout, urls, collection, per_shard)?;

    println!(
        "peers queried: {} (from {} url(s))",
        report.peers_queried,
        urls.len()
    );
    println!("shards checked: {}", report.shards_checked);
    println!("ids checked: {}", report.ids_checked);

    if report.is_correct() {
        println!(
            "\nOK - every sampled point is in the shard the hash ring assigned it to, \
             and absent from the shard next to it"
        );
        return Ok(());
    }

    if !report.misplaced.is_empty() {
        println!("\nMISPLACED - the ring assigns these to a shard that does not hold them:");
        for (shard_id, id) in report.misplaced.iter().take(10) {
            println!("  shard {shard_id} should hold {id:?}");
        }
        if report.misplaced.len() > 10 {
            println!("  ... and {} more", report.misplaced.len() - 10);
        }
    }

    if !report.leaked.is_empty() {
        println!("\nLEAKED - found in a shard the ring did not assign them to:");
        for (shard_id, id) in report.leaked.iter().take(10) {
            println!("  shard {shard_id} unexpectedly holds {id:?}");
        }
    }

    bail!(
        "shard placement is wrong. Reads will still appear to work, but any upsert of an \
         existing id will create a duplicate that is never deduplicated. Re-install the \
         artifacts using `install-plan` to get the shard-to-peer mapping."
    );
}

fn cmd_install_plan(url: &str, collection: &str) -> Result<()> {
    let plan = placement::install_plan(url, collection)?;

    println!("queried peer: {}", plan.queried_peer);
    println!("\nshard -> peer (install shard_N's artifact on this peer):");
    for (shard_id, (peer_id, uri)) in &plan.placement {
        println!("  shard {shard_id}  ->  peer {peer_id}  {uri}");
    }

    println!(
        "\nConsensus owns this mapping, so read it from the cluster rather than assuming \
         directory order. After installing, run `verify-placement`."
    );
    Ok(())
}

/// Parse a `--slice I/N` argument.
fn parse_slice(text: &str) -> Result<(usize, usize)> {
    let (index, total) = text
        .split_once('/')
        .with_context(|| format!("--slice must look like I/N, got '{text}'"))?;
    let index: usize = index
        .trim()
        .parse()
        .with_context(|| format!("--slice index '{index}' is not a number"))?;
    let total: usize = total
        .trim()
        .parse()
        .with_context(|| format!("--slice total '{total}' is not a number"))?;
    if total == 0 || index >= total {
        bail!("--slice must be I/N with N >= 1 and I < N; got {index}/{total}");
    }
    Ok((index, total))
}

/// The `scatter` subcommand's arguments, grouped so the function stays readable as options accrue.
struct ScatterArgs<'a> {
    config_path: &'a Path,
    input: &'a Path,
    work: &'a Path,
    workers: usize,
    slice: Option<&'a str>,
    input_format: Option<source::InputFormat>,
    max_failures: usize,
}

fn cmd_scatter(args: ScatterArgs<'_>) -> Result<()> {
    let ScatterArgs {
        config_path,
        input,
        work,
        workers,
        slice,
        input_format,
        max_failures,
    } = args;
    let document = document::load(config_path)?;
    let loaded = &document.config;
    let router = ShardRouter::new(loaded)?;
    let mapping = document.mapping.as_ref();

    let store = LocalStore::new(input);
    let discovered = scatter::discover_inputs(&store, input_format)
        .with_context(|| format!("under {}", input.display()))?;
    let slice = slice.map(parse_slice).transpose()?;

    let inputs = match slice {
        Some((index, total)) => {
            let mine = scatter::take_slice(discovered.clone(), index, total)?;
            println!(
                "input files: {} of {} (slice {index}/{total})",
                mine.len(),
                discovered.len(),
            );
            mine
        }
        None => {
            println!("input files: {}", discovered.len());
            discovered
        }
    };
    println!("shards: {}", router.shard_count());
    // The part fingerprint is what binds this work directory. Print it rather than the full
    // config one, so a later resume can be checked against what actually gates it.
    println!("part fingerprint: {}", loaded.part_fingerprint);

    let layout = scatter::ScatterLayout::new(work);
    let stats = scatter::run(
        loaded,
        &router,
        &store,
        &inputs,
        &layout,
        &scatter::ScatterOptions {
            workers,
            mapping,
            slice,
            max_failures,
        },
    )?;

    println!("\nfiles_processed: {}", stats.files_processed);
    println!("files_skipped: {}", stats.files_skipped);
    println!("points: {}", stats.points);
    println!("parts_written: {}", stats.parts_written);
    if stats.temp_files_swept > 0 {
        println!("temp_files_swept: {}", stats.temp_files_swept);
    }

    if !stats.files_failed.is_empty() {
        println!("files_failed: {}", stats.files_failed.len());
        for failure in stats.files_failed.iter().take(10) {
            println!("  {}: {}", failure.path, failure.error);
        }
        if stats.files_failed.len() > 10 {
            println!("  ... and {} more", stats.files_failed.len() - 10);
        }

        // Non-zero, but only after scattering everything that could be scattered: a re-run has
        // little left to do. Silence here would let a short collection look like a clean one,
        // which is the failure this whole tool is built to avoid.
        bail!(
            "{} input file(s) could not be read and were skipped; their points are NOT in the \
             output. Fix or remove them and re-run this slice — finished files are skipped.\n\n\
             If they are not meant to be input at all, pass --input-format to exclude them.",
            stats.files_failed.len(),
        );
    }

    println!("\nOK");
    Ok(())
}

/// The `build` subcommand's arguments, grouped for the same reason as [`ScatterArgs`].
struct BuildArgs<'a> {
    config_path: &'a Path,
    work: &'a Path,
    out: &'a Path,
    staging: &'a Path,
    shard: Option<u32>,
    concurrency: Option<usize>,
    indexing_threads: Option<usize>,
    batch_points: usize,
    slice: Option<&'a str>,
    route: BuildRouteArg,
}

fn cmd_build(args: BuildArgs<'_>) -> Result<()> {
    let BuildArgs {
        config_path,
        work,
        out,
        staging,
        shard,
        concurrency,
        indexing_threads,
        batch_points,
        slice,
        route,
    } = args;
    let document = document::load(config_path)?;
    let loaded = &document.config;
    let payload_index = document.payload_index.as_ref();
    let router = ShardRouter::new(loaded)?;
    let scatter_layout = scatter::ScatterLayout::new(work);
    let build_layout = build::BuildLayout::new(out, staging);

    let concurrency = match concurrency {
        Some(0) => bail!("--concurrency must be at least 1"),
        Some(value) => value,
        None => build::default_concurrency(loaded),
    };

    let shards: Vec<u32> = match shard {
        Some(shard) => vec![shard],
        None => (0..router.shard_count()).collect(),
    };

    let mut plans = Vec::new();
    for shard_id in shards {
        let plan_path = plan::ShardPlan::path_in(work, shard_id);
        if !plan_path.exists() {
            log::debug!("no plan for shard {shard_id}, skipping");
            continue;
        }
        plans.push(
            serde_json::from_slice::<plan::ShardPlan>(&fs_err::read(&plan_path)?)
                .with_context(|| format!("cannot parse {}", plan_path.display()))?,
        );
    }

    if plans.is_empty() {
        bail!(
            "no plans found under {}; run the plan phase first",
            work.display()
        );
    }

    let slice = slice.map(parse_slice).transpose()?;

    let segments: usize = plans.iter().map(|plan| plan.segments.len()).sum();
    println!("concurrency: {concurrency} segment builds");
    if let Some(threads) = indexing_threads {
        println!(
            "indexing_threads: {threads} per segment (CLI override; not recorded in artifacts)"
        );
    }
    if let Some((index, total)) = slice {
        println!("slice: {index}/{total} of {segments} planned segments");
    }
    if let Some(schema) = &payload_index {
        println!(
            "payload indexes: {} field(s) built into each segment",
            schema.len()
        );
    }
    println!("batch_points: {batch_points}");
    println!(
        "route: {}",
        match route {
            BuildRouteArg::Bulk => "bulk (storages filled directly, indexes built once)",
            BuildRouteArg::Edge => "edge (staging shard + optimizers)",
        }
    );
    println!("staging: {}", staging.display());
    if let Some(mapping) = &document.mapping {
        println!(
            "vectors: {} dense, {} sparse",
            loaded.config.params.vectors.params_iter().count(),
            loaded
                .config
                .params
                .sparse_vectors
                .as_ref()
                .map_or(0, |map| map.len()),
        );
        println!(
            "payload: {} column(s) kept — {:?}",
            mapping.payload_columns.len(),
            mapping.payload_columns,
        );
    }
    println!("work: {segments} segments across {} shards", plans.len());

    let started = std::time::Instant::now();
    let total = build::run_all(
        loaded,
        &plans,
        &scatter_layout,
        &build_layout,
        &build::BuildOptions {
            concurrency,
            batch_points,
            payload_index,
            slice,
            route: route.into(),
            indexing_threads,
            mapping: document.mapping.as_ref(),
        },
    )?;
    let elapsed = started.elapsed();

    let rate = if elapsed.as_secs_f64() > 0.0 {
        total.points as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!("\nelapsed: {elapsed:.1?}  ({rate:.0} points/s)");

    println!("\nsegments_built: {}", total.segments_built);
    println!("segments_skipped: {}", total.segments_skipped);
    println!("points: {}", total.points);
    println!("\nOK");
    Ok(())
}

fn cmd_plan(
    config_path: &Path,
    work: &Path,
    workers: usize,
    slice: Option<&str>,
    replan: bool,
) -> Result<()> {
    let loaded = &document::load(config_path)?.config;
    let router = ShardRouter::new(loaded)?;
    let layout = scatter::ScatterLayout::new(work);

    let bytes_per_point = plan::bytes_per_point(loaded)?;
    let (min_bytes, max_bytes) = loaded
        .target_segment_band_bytes()
        .context("max_segment_size must be set")?;

    println!("bytes_per_point: {bytes_per_point}");
    println!(
        "segment band: {} .. {}",
        human_bytes(min_bytes),
        human_bytes(max_bytes),
    );

    let slice = slice.map(parse_slice).transpose()?;
    let report = plan::build(
        loaded,
        &router,
        &layout,
        work,
        &plan::PlanOptions {
            workers,
            slice,
            replan,
        },
    )?;
    let plans = &report.plans;

    let total_points: u64 = plans.iter().map(|p| p.points).sum();
    let total_segments: usize = plans.iter().map(|p| p.segments.len()).sum();

    if report.shards_skipped > 0 {
        println!(
            "\nshards_skipped: {} (already planned; pass --replan to redo them)",
            report.shards_skipped,
        );
    }
    println!("\nshards planned: {}", plans.len());
    println!("segments: {total_segments}");
    println!("points: {total_points}");

    let mut warnings = 0;
    for shard_plan in plans {
        for note in plan::review(shard_plan) {
            log::warn!("shard {}: {note}", shard_plan.shard_id);
            warnings += 1;
        }
    }

    println!("\nper shard:");
    for shard_plan in plans {
        println!(
            "  shard {}: {} segments, {} points, largest {}",
            shard_plan.shard_id,
            shard_plan.segments.len(),
            shard_plan.points,
            human_bytes(
                shard_plan
                    .segments
                    .iter()
                    .map(|s| s.vector_bytes)
                    .max()
                    .unwrap_or(0)
            ),
        );
    }

    println!("\nplans written to {}", work.join("plan").display());
    if warnings > 0 {
        println!("{warnings} warning(s) above - review before building");
    }
    println!("\nOK");
    Ok(())
}

fn cmd_scatter_verify(
    config_path: &Path,
    work: &Path,
    workers: usize,
    slice: Option<&str>,
    recheck: bool,
    deep: bool,
) -> Result<()> {
    let loaded = &document::load(config_path)?.config;
    let router = ShardRouter::new(loaded)?;
    let layout = scatter::ScatterLayout::new(work);
    let slice = slice.map(parse_slice).transpose()?;

    println!(
        "mode: {}",
        if deep {
            "deep (every record read)"
        } else {
            "metadata (header + recorded length per part)"
        },
    );

    let report = scatter::verify(
        loaded,
        &router,
        &layout,
        &scatter::VerifyOptions {
            workers,
            slice,
            recheck,
            deep,
        },
    )?;

    println!("parts: {}", report.parts);
    if report.parts_skipped > 0 {
        println!("parts_skipped: {} (already verified)", report.parts_skipped);
    }
    println!("points: {}", report.points);

    // A resumed run that had nothing left to do is a success, not an empty verify.
    if report.points == 0 && report.parts_skipped > 0 {
        println!("\nOK - every part was already verified by an earlier run");
        return Ok(());
    }

    if report.points == 0 {
        bail!(
            "no points found under {}; nothing to verify",
            work.display()
        );
    }

    // Skew matters for phase 2: a shard holding far more than its share produces oversized
    // segments and a long tail in both index build and query latency.
    let counts: Vec<u64> = report.per_shard.values().copied().collect();
    let min = counts.iter().copied().min().unwrap_or(0);
    let max = counts.iter().copied().max().unwrap_or(0);
    let mean = report.points as f64 / f64::from(router.shard_count());

    println!(
        "shards with data: {} of {}",
        counts.len(),
        router.shard_count()
    );
    println!("points/shard: min {min}, max {max}, mean {mean:.0}");

    if min > 0 && max as f64 / min as f64 > 2.0 {
        log::warn!(
            "shard skew is {:.1}x (min {min}, max {max}); the largest shard sets segment \
             sizes and query tail latency",
            max as f64 / min as f64,
        );
    }

    println!("\nOK - every point is in the shard the hash ring chose");
    Ok(())
}

fn cmd_validate(config_path: &Path, points: Option<u64>) -> Result<()> {
    let document = document::load(config_path)?;
    let loaded = &document.config;
    let router = ShardRouter::new(loaded)?;

    println!("config: {}", config_path.display());
    if let Some(schema) = &document.payload_index {
        println!("payload indexes: {} field(s) declared", schema.len());
    }
    println!("fingerprint: {}", loaded.fingerprint);
    // Narrower surface: routing alone. Two configs sharing this may reuse one scatter, even
    // if they differ on HNSW, quantization or segment size.
    println!("part fingerprint: {}", loaded.part_fingerprint);
    println!("shards: {}", router.shard_count());
    println!("sharding_method: {:?}", loaded.sharding_method());
    println!(
        "hash_ring_shard_scale: {} virtual nodes/shard",
        loaded.hash_ring_shard_scale(),
    );

    let (min_bytes, max_bytes) = loaded
        .target_segment_band_bytes()
        .context("max_segment_size must be set")?;

    println!("max_segment_size: {}", human_bytes(max_bytes));
    println!(
        "merge-safe segment band: {} .. {}",
        human_bytes(min_bytes),
        human_bytes(max_bytes),
    );

    let bytes_per_point = plan::bytes_per_point(loaded)?;
    println!("bytes_per_point: {bytes_per_point}");

    if let Some(points) = points {
        let vector_bytes = points
            .checked_mul(bytes_per_point)
            .context("points x bytes_per_point overflows")?;
        plan_segments(loaded, points, vector_bytes, min_bytes, max_bytes);
        report_shard_skew(&router, points, bytes_per_point)?;
    } else {
        println!(
            "\nPass --points to project the segment plan. Segments below {} would be merged \
             by the serving cluster on load.",
            human_bytes(min_bytes),
        );
    }

    println!("\nOK");
    Ok(())
}

/// Project how unevenly the hash ring will fill the shards, and what that costs per node.
///
/// Worth printing because the answer is counter-intuitive and it drives hardware sizing. The
/// skew comes from the fixed lengths of the ring's segments — roughly `O(1/sqrt(vnodes))`,
/// about 10% at the default `hash_ring_shard_scale` of 100 — so it does **not** shrink as the
/// corpus grows. Measured at 10 shards and scale 100: 2M synthetic ids give +13.7%/-19.1%, and
/// the real FineWeb corpus gives +13.8%/-18.9% through the same ring.
///
/// The consequence is that per-node RAM has to be sized for the largest shard. Sizing for
/// `points / shards` under-provisions the busiest node by the skew, and the HNSW graph and
/// quantized vectors are the parts held in memory. Raising `hash_ring_shard_scale` at
/// collection creation is the knob that trades ring memory for a flatter distribution.
fn report_shard_skew(router: &ShardRouter, points: u64, bytes_per_point: u64) -> Result<()> {
    // Enough samples for the fractions to converge to well under a percent, and fast enough to
    // sit in a pre-flight check.
    const SAMPLES: u64 = 400_000;
    let fractions = ring::measure_distribution(router, SAMPLES)?;

    let shards = fractions.len() as f64;
    let even = 1.0 / shards;
    let largest = fractions.iter().copied().fold(f64::MIN, f64::max);
    let smallest = fractions.iter().copied().fold(f64::MAX, f64::min);

    let largest_points = (points as f64 * largest) as u64;
    let even_points = (points as f64 * even) as u64;

    println!(
        "\nprojected hash ring distribution (a property of shard_number and \
         hash_ring_shard_scale, not of the data):"
    );
    println!(
        "  evenly:  {even_points} points/shard, {}",
        human_bytes(even_points.saturating_mul(bytes_per_point)),
    );
    println!(
        "  largest: {largest_points} points ({:+.1}%), {}",
        (largest / even - 1.0) * 100.0,
        human_bytes(largest_points.saturating_mul(bytes_per_point)),
    );
    println!(
        "  smallest: {} points ({:+.1}%)",
        (points as f64 * smallest) as u64,
        (smallest / even - 1.0) * 100.0,
    );

    if largest / even > 1.05 {
        println!(
            "  -> size per-node memory for the largest shard, not the mean: it holds {:.1}% more \
             than an even split. This skew is fixed by the ring and does not shrink with more \
             data; a higher hash_ring_shard_scale at creation would flatten it.",
            (largest / even - 1.0) * 100.0,
        );
    }

    Ok(())
}

/// Report the segment plan and check it against the merge optimizer's one real constraint.
///
/// There is no per-shard size limit. `MergeOptimizer::plan_optimizations` only ever merges
/// segments whose *combined* size stays under `max_segment_size`, and bails when fewer than
/// two of them fit. So the only thing that matters is a per-segment floor of
/// `max_segment_size / 2`: hold that and no merge is ever scheduled, whatever the shard total.
fn plan_segments(
    loaded: &LoadedConfig,
    points: u64,
    vector_bytes: u64,
    min_bytes: u64,
    max_bytes: u64,
) {
    let shards = u64::from(loaded.shard_number());
    let per_shard = vector_bytes.div_ceil(shards);

    println!("\npoints: {points}");
    println!("vector storage: {}", human_bytes(vector_bytes));
    println!("shards: {shards}");
    println!("per_shard vectors (even split): {}", human_bytes(per_shard));
    println!("points/shard: {}", points.div_ceil(shards));

    // Aim at the top of the band: fewest segments, and the most headroom above the floor.
    let segments_per_shard = per_shard.div_ceil(max_bytes).max(1);
    let segment_size = per_shard.div_ceil(segments_per_shard);

    println!("segments_per_shard: {segments_per_shard}");
    println!("segment_size: {}", human_bytes(segment_size));

    if segment_size < min_bytes {
        // Only reachable when a shard holds less than half of one max-size segment, i.e. the
        // dataset is small relative to shard_number. Splitting it into >= 2 segments would
        // invite a merge; one segment is fine, so this is a warning, not an error.
        log::warn!(
            "a shard holds {} which is below the {} merge-safe floor; \
             build a single segment per shard so there is no pair for the optimizer to merge",
            human_bytes(per_shard),
            human_bytes(min_bytes),
        );
        return;
    }

    let headroom = (segment_size - min_bytes) as f64 / min_bytes as f64 * 100.0;
    println!("floor headroom: {headroom:.1}%");

    if headroom < 20.0 {
        log::warn!(
            "segments sit only {headroom:.1}% above the merge-safe floor ({}); \
             uneven partitions could drop a segment below it and trigger a merge. \
             Consider fewer, larger segments.",
            human_bytes(min_bytes),
        );
    }

    // Informational: exceeding default_segment_number is normal and provokes nothing on its
    // own, but it does tell you how far from Qdrant's own shape these artifacts are.
    let default_segments = loaded.config.optimizer_config.default_segment_number;
    if default_segments > 0 && segments_per_shard > default_segments as u64 {
        println!(
            "note: {segments_per_shard} segments/shard exceeds default_segment_number \
             ({default_segments}). That is allowed — the merge optimizer still cannot pair \
             segments this large — but search fans out across all of them."
        );
    }
}

fn cmd_dump_part(config_path: &Path, file: &Path, limit: usize) -> Result<()> {
    let loaded = &document::load(config_path)?.config;
    let mut reader = partfile::PartReader::open(file, &loaded.part_fingerprint)?;

    println!("{}", serde_json::to_string_pretty(reader.header())?);

    for index in 0..limit {
        match reader.next_record()? {
            Some(record) => {
                println!("\n--- record {index}");
                println!("{}", serde_json::to_string_pretty(&record)?);
            }
            None => break,
        }
    }

    Ok(())
}

fn cmd_shard_of(config_path: &Path, ids: &[String]) -> Result<()> {
    let loaded = &document::load(config_path)?.config;
    let router = ShardRouter::new(loaded)?;

    if ids.is_empty() {
        bail!("no point ids given");
    }

    for id in ids {
        let point_id = parse_point_id(id)?;
        println!("{id} -> shard {}", router.shard_of(point_id)?);
    }

    Ok(())
}

fn parse_point_id(raw: &str) -> Result<ExtendedPointId> {
    if let Ok(num) = raw.parse::<u64>() {
        return Ok(ExtendedPointId::NumId(num));
    }
    let uuid = raw
        .parse()
        .with_context(|| format!("point id {raw} is neither an integer nor a UUID"))?;
    Ok(ExtendedPointId::Uuid(uuid))
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numeric_and_uuid_point_ids() {
        assert_eq!(parse_point_id("42").unwrap(), ExtendedPointId::NumId(42),);
        assert!(matches!(
            parse_point_id("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            ExtendedPointId::Uuid(_),
        ));
        assert!(parse_point_id("not-an-id").is_err());
    }

    #[test]
    fn human_bytes_is_readable() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.00 KiB");
        assert_eq!(human_bytes(50 * 1024_u64.pow(4)), "50.00 TiB");
    }

    /// A large dataset is planned into many merge-safe segments, not rejected.
    ///
    /// There is no per-shard size limit: the merge optimizer can only combine segments whose
    /// *pair* fits under `max_segment_size`, so segments at the ceiling are never touched
    /// however many a shard holds.
    #[test]
    fn large_dataset_plans_many_segments_without_error() {
        let mut value = config::tests::valid_config_json();
        value["params"]["shard_number"] = serde_json::json!(64);
        value["optimizer_config"]["max_segment_size"] = serde_json::json!(20 * 1024 * 1024); // 20 GiB
        let loaded = config::from_str(&value.to_string()).unwrap();

        let (min_bytes, max_bytes) = loaded.target_segment_band_bytes().unwrap();
        let bytes_per_point = plan::bytes_per_point(&loaded).unwrap();
        let fifty_tib = 50 * 1024_u64.pow(4);
        let points = fifty_tib / bytes_per_point;

        // Reports rather than rejects: 50 TiB across 64 shards is a valid plan.
        plan_segments(&loaded, points, fifty_tib, min_bytes, max_bytes);

        // 50 TiB / 64 = 800 GiB per shard, at 20 GiB per segment => 40 segments. Far more
        // than default_segment_number (8), which is fine and must not fail.
        let per_shard = fifty_tib / 64;
        let segments = per_shard.div_ceil(max_bytes);
        assert_eq!(segments, 40);
        assert!(
            segments > loaded.config.optimizer_config.default_segment_number as u64,
            "this case must exceed default_segment_number to be a meaningful test",
        );
    }

    /// A shard smaller than the merge-safe floor warns rather than failing.
    #[test]
    fn tiny_shard_is_a_warning_not_an_error() {
        let mut value = config::tests::valid_config_json();
        value["params"]["shard_number"] = serde_json::json!(64);
        value["optimizer_config"]["max_segment_size"] = serde_json::json!(20 * 1024 * 1024);
        let loaded = config::from_str(&value.to_string()).unwrap();

        let (min_bytes, max_bytes) = loaded.target_segment_band_bytes().unwrap();

        // 64 MiB total across 64 shards: 1 MiB each, well under the 10 GiB floor. A single
        // segment per shard has nothing to pair with, so this is allowed.
        // An under-floor shard warns; it is not an error, so there is nothing to unwrap.
        plan_segments(&loaded, 1000, 64 * 1024 * 1024, min_bytes, max_bytes);
    }
}
