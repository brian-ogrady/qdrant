//! Progress reporting for the two phases that take hours.
//!
//! # Why this is not just a progress bar
//!
//! These commands run under a batch scheduler, where stdout is a file. A redrawing progress bar
//! writes a carriage return and a fresh copy of the line on every update, so a run that would show
//! one tidy bar on a terminal leaves tens of thousands of overwritten fragments in a `slurm-*.out`
//! — the exact log you have to read when the job fails.
//!
//! So the output shape follows the destination:
//!
//! * **Terminal** — an [`indicatif`] bar with position, rate and ETA, redrawn in place.
//! * **Anything else** — one line at a fixed interval, and one at the end. Append-only, greppable,
//!   and safe to `tail -f`. A 4-hour build produces a few dozen lines instead of a megabyte of
//!   control characters.
//!
//! Detection is [`IsTerminal`] on stdout, so a pipe or a redirect picks the log form
//! automatically and nothing has to be configured.
//!
//! # Why there is a heartbeat
//!
//! Progress is recorded when a unit *completes*, which for the build phase is a whole segment —
//! minutes of work at production sizes. Reporting only on completion means the first several
//! minutes of a multi-hour job produce nothing at all, which is indistinguishable from a hang. So
//! the log form also ticks on a timer, whether or not anything finished.
//!
//! # Concurrency
//!
//! Both phases report from worker threads, so every method takes `&self` and the counters are
//! atomic, held behind an `Arc` the heartbeat thread shares. `indicatif` handles its own draw
//! locking; the log path holds a short mutex only to decide which caller prints this interval.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

/// How often the non-terminal form prints. Chosen so an hours-long phase yields a readable log
/// rather than either silence or a flood.
const LOG_INTERVAL: Duration = Duration::from_secs(30);

/// How often the heartbeat wakes to check whether a line is due. Well under `LOG_INTERVAL`, so the
/// interval is honoured closely without the thread costing anything measurable.
const HEARTBEAT_TICK: Duration = Duration::from_secs(2);

/// Counters shared between the reporting callers and the heartbeat thread.
struct State {
    label: String,
    unit: String,
    total: u64,
    done: AtomicU64,
    /// Secondary tally shown alongside the count, e.g. points scattered.
    items: AtomicU64,
    started: Instant,
    last_logged: Mutex<Instant>,
}

impl State {
    /// One human-readable progress line.
    fn line(&self) -> String {
        let done = self.done.load(Ordering::Relaxed);
        let items = self.items.load(Ordering::Relaxed);
        let elapsed = self.started.elapsed();

        let percent = if self.total > 0 {
            done as f64 / self.total as f64 * 100.0
        } else {
            100.0
        };
        let rate = if elapsed.as_secs_f64() > 0.0 {
            items as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let eta = if done > 0 && done < self.total {
            let per_unit = elapsed.as_secs_f64() / done as f64;
            let remaining = Duration::from_secs_f64(per_unit * (self.total - done) as f64);
            format!(", eta {}", short_duration(remaining))
        } else {
            String::new()
        };

        format!(
            "{}: {done}/{} {}s ({percent:.1}%), {items} points, {rate:.0} points/s, elapsed {}{eta}",
            self.label,
            self.total,
            self.unit,
            short_duration(elapsed),
        )
    }

    /// Print a line if the interval has elapsed. Returns whether it printed.
    fn log_if_due(&self, force: bool) -> bool {
        let mut last = match self.last_logged.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if force || last.elapsed() >= LOG_INTERVAL {
            *last = Instant::now();
            drop(last);
            log::info!("{}", self.line());
            return true;
        }
        false
    }
}

pub struct Progress {
    /// `Some` only when stdout is a terminal.
    bar: Option<ProgressBar>,
    state: Arc<State>,
    stop: Arc<AtomicBool>,
    heartbeat: Mutex<Option<JoinHandle<()>>>,
}

impl Progress {
    /// `label` names the phase, `unit` names what is being counted ("file", "segment").
    pub fn new(label: impl Into<String>, unit: impl Into<String>, total: u64) -> Self {
        let label = label.into();
        let unit = unit.into();

        let bar = if std::io::stdout().is_terminal() {
            let bar = ProgressBar::new(total);
            // `{wide_bar}` rather than a fixed width so a narrow terminal does not wrap, which
            // would defeat the in-place redraw.
            let style = ProgressStyle::with_template(
                "{msg}\n  [{elapsed_precise}] {wide_bar} {pos}/{len} ({percent}%) eta {eta}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar());
            bar.set_style(style);
            bar.set_message(label.clone());
            // Tick on a timer so the elapsed clock moves even while a long unit is in flight, and
            // the bar visibly is not hung.
            bar.enable_steady_tick(Duration::from_millis(500));
            Some(bar)
        } else {
            log::info!("{label}: starting, {total} {unit}(s) to process");
            None
        };

        let state = Arc::new(State {
            label,
            unit,
            total,
            done: AtomicU64::new(0),
            items: AtomicU64::new(0),
            started: Instant::now(),
            last_logged: Mutex::new(Instant::now()),
        });
        let stop = Arc::new(AtomicBool::new(false));

        // The terminal form already animates itself; only the log form needs a ticker.
        let heartbeat = if bar.is_none() {
            let state = Arc::clone(&state);
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("progress-heartbeat".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(HEARTBEAT_TICK);
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        state.log_if_due(false);
                    }
                })
                .ok()
        } else {
            None
        };

        Self {
            bar,
            state,
            stop,
            heartbeat: Mutex::new(heartbeat),
        }
    }

    /// Record one completed unit, plus `items` of whatever it contained.
    pub fn advance(&self, items: u64) {
        let done = self.state.done.fetch_add(1, Ordering::Relaxed) + 1;
        let total_items = self.state.items.fetch_add(items, Ordering::Relaxed) + items;

        if let Some(bar) = &self.bar {
            bar.set_position(done);
            bar.set_message(format!("{}  ({total_items} points)", self.state.label));
            return;
        }

        // Force a line on the final unit, so a completed phase always ends with a full one.
        self.state.log_if_due(done >= self.state.total);
    }

    /// Called once when the phase ends. Stops the heartbeat.
    pub fn finish(&self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(mut handle) = self.heartbeat.lock()
            && let Some(handle) = handle.take()
        {
            let _ = handle.join();
        }

        match &self.bar {
            Some(bar) => bar.finish_with_message(format!("{} complete", self.state.label)),
            // The log form printed a final line on the last unit; only add one if the phase ended
            // early, so a failed or fully-skipped run still reports where it got to.
            None if self.state.done.load(Ordering::Relaxed) < self.state.total => {
                self.state.log_if_due(true);
            }
            None => {}
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        // A phase that returns early — an error out of a worker — must not leave the thread running.
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Compact duration: `4h12m`, `7m3s`, `12s`. `Debug` on `Duration` prints excessive precision and
/// `humantime` is not a dependency here.
fn short_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    match (secs / 3600, (secs % 3600) / 60, secs % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m{s}s"),
        (h, m, _) => format!("{h}h{m}m"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_are_compact() {
        assert_eq!(short_duration(Duration::from_secs(12)), "12s");
        assert_eq!(short_duration(Duration::from_secs(423)), "7m3s");
        assert_eq!(short_duration(Duration::from_secs(15120)), "4h12m");
    }

    /// Under a test harness stdout is not a terminal, so the log form must be selected. That is
    /// also what a batch scheduler gives us, which is the case that matters.
    #[test]
    fn chooses_the_log_form_when_stdout_is_not_a_terminal() {
        let progress = Progress::new("scatter", "file", 10);
        assert!(
            progress.bar.is_none(),
            "a non-terminal stdout must not get a redrawing bar",
        );
        assert!(
            progress.heartbeat.lock().unwrap().is_some(),
            "the log form needs a ticker, or a long unit looks like a hang",
        );
        progress.finish();
    }

    #[test]
    fn counts_units_and_items_separately() {
        let progress = Progress::new("build", "segment", 3);
        progress.advance(100);
        progress.advance(250);

        assert_eq!(progress.state.done.load(Ordering::Relaxed), 2);
        assert_eq!(progress.state.items.load(Ordering::Relaxed), 350);

        let line = progress.state.line();
        assert!(line.contains("2/3 segments"), "{line}");
        assert!(line.contains("66.7%"), "{line}");
        assert!(line.contains("350 points"), "{line}");
        assert!(
            line.contains("eta"),
            "an unfinished phase should project an eta: {line}"
        );
        progress.finish();
    }

    #[test]
    fn omits_an_eta_once_complete() {
        let progress = Progress::new("scatter", "file", 2);
        progress.advance(5);
        progress.advance(5);
        let line = progress.state.line();
        assert!(
            !line.contains("eta"),
            "a finished phase has nothing to project: {line}"
        );
        progress.finish();
    }

    /// A zero-total phase must not divide by zero.
    #[test]
    fn handles_an_empty_phase() {
        let progress = Progress::new("scatter", "file", 0);
        assert!(progress.state.line().contains("100.0%"));
        progress.finish();
    }

    /// The heartbeat must report before anything has completed — that is its whole purpose.
    #[test]
    fn heartbeat_reports_while_a_unit_is_still_in_flight() {
        let progress = Progress::new("build", "segment", 5);

        // Nothing has completed, yet a line is available and describes the state honestly.
        let line = progress.state.line();
        assert!(line.contains("0/5 segments"), "{line}");
        assert!(line.contains("0.0%"), "{line}");
        assert!(
            !line.contains("eta"),
            "no eta is possible before the first unit finishes: {line}"
        );

        progress.finish();
        assert!(
            progress.heartbeat.lock().unwrap().is_none(),
            "finish must join the heartbeat thread",
        );
    }

    /// Dropping without `finish` must still stop the thread.
    #[test]
    fn drop_stops_the_heartbeat() {
        let stop = {
            let progress = Progress::new("scatter", "file", 3);
            Arc::clone(&progress.stop)
        };
        assert!(
            stop.load(Ordering::Relaxed),
            "drop must signal the heartbeat to stop",
        );
    }
}
