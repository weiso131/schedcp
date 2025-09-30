use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::interval;

/// CPU statistics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    pub timestamp: u64,
    pub user: u64,
    pub nice: u64,
    pub system: u64,
    pub idle: u64,
    pub iowait: u64,
    pub irq: u64,
    pub softirq: u64,
    pub steal: u64,
}

/// Memory statistics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub timestamp: u64,
    pub total_kb: u64,
    pub free_kb: u64,
    pub available_kb: u64,
    pub buffers_kb: u64,
    pub cached_kb: u64,
    pub used_kb: u64,
    pub used_percent: f64,
}

/// Scheduler statistics from /proc/schedstat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedStats {
    pub timestamp: u64,
    pub cpu_count: usize,
    pub total_run_time: u64,
    pub total_wait_time: u64,
    pub total_timeslices: u64,
}

/// A monitoring session
#[derive(Debug, Clone)]
pub struct MonitoringSession {
    pub session_id: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub cpu_samples: Vec<CpuStats>,
    pub memory_samples: Vec<MemoryStats>,
    pub sched_samples: Vec<SchedStats>,
}

/// Summary statistics for a monitoring session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSummary {
    pub session_id: String,
    pub duration_secs: u64,
    pub sample_count: usize,
    pub cpu_avg_percent: f64,
    pub cpu_max_percent: f64,
    pub memory_avg_percent: f64,
    pub memory_max_percent: f64,
    pub memory_avg_used_mb: f64,
    pub sched_total_timeslices: u64,
    pub sched_avg_run_time_ns: u64,
}

/// System monitor manages monitoring sessions
pub struct SystemMonitor {
    sessions: Arc<Mutex<HashMap<String, MonitoringSession>>>,
    active_session: Arc<Mutex<Option<String>>>,
}

impl SystemMonitor {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            active_session: Arc::new(Mutex::new(None)),
        }
    }

    /// Start a new monitoring session
    pub async fn start_monitoring(&self) -> Result<String> {
        // Check if there's already an active session
        let mut active = self.active_session.lock().await;
        if active.is_some() {
            anyhow::bail!("A monitoring session is already active. Stop it first.");
        }

        // Generate session ID
        let session_id = format!("mon_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create session
        let session = MonitoringSession {
            session_id: session_id.clone(),
            start_time,
            end_time: None,
            cpu_samples: Vec::new(),
            memory_samples: Vec::new(),
            sched_samples: Vec::new(),
        };

        // Store session
        let mut sessions = self.sessions.lock().await;
        sessions.insert(session_id.clone(), session);

        // Mark as active
        *active = Some(session_id.clone());

        // Start collection task
        let sessions_clone = self.sessions.clone();
        let active_clone = self.active_session.clone();
        let session_id_clone = session_id.clone();

        tokio::spawn(async move {
            Self::collection_task(sessions_clone, active_clone, session_id_clone).await;
        });

        log::info!("Started monitoring session: {}", session_id);
        Ok(session_id)
    }

    /// Stop the active monitoring session and return summary
    pub async fn stop_monitoring(&self) -> Result<MonitoringSummary> {
        let mut active = self.active_session.lock().await;
        let session_id = active
            .take()
            .ok_or_else(|| anyhow::anyhow!("No active monitoring session"))?;

        // Update end time
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?;

        session.end_time = Some(end_time);

        log::info!("Stopped monitoring session: {}", session_id);

        // Generate summary
        let summary = Self::generate_summary(session)?;
        Ok(summary)
    }

    /// Background task to collect metrics
    async fn collection_task(
        sessions: Arc<Mutex<HashMap<String, MonitoringSession>>>,
        active: Arc<Mutex<Option<String>>>,
        session_id: String,
    ) {
        let mut ticker = interval(Duration::from_secs(1));

        loop {
            ticker.tick().await;

            // Check if session is still active
            {
                let active_guard = active.lock().await;
                if active_guard.as_ref() != Some(&session_id) {
                    log::info!("Collection task stopping for session: {}", session_id);
                    break;
                }
            }

            // Collect metrics
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let cpu_stats = match Self::read_cpu_stats(timestamp).await {
                Ok(stats) => stats,
                Err(e) => {
                    log::warn!("Failed to read CPU stats: {}", e);
                    continue;
                }
            };

            let memory_stats = match Self::read_memory_stats(timestamp).await {
                Ok(stats) => stats,
                Err(e) => {
                    log::warn!("Failed to read memory stats: {}", e);
                    continue;
                }
            };

            let sched_stats = match Self::read_sched_stats(timestamp).await {
                Ok(stats) => stats,
                Err(e) => {
                    log::warn!("Failed to read scheduler stats: {}", e);
                    continue;
                }
            };

            // Store samples
            let mut sessions_guard = sessions.lock().await;
            if let Some(session) = sessions_guard.get_mut(&session_id) {
                session.cpu_samples.push(cpu_stats);
                session.memory_samples.push(memory_stats);
                session.sched_samples.push(sched_stats);
            }
        }
    }

    /// Read CPU statistics from /proc/stat
    async fn read_cpu_stats(timestamp: u64) -> Result<CpuStats> {
        let stat_content = tokio::fs::read_to_string("/proc/stat")
            .await
            .context("Failed to read /proc/stat")?;

        let cpu_line = stat_content
            .lines()
            .find(|line| line.starts_with("cpu "))
            .ok_or_else(|| anyhow::anyhow!("CPU line not found in /proc/stat"))?;

        let fields: Vec<&str> = cpu_line.split_whitespace().collect();
        if fields.len() < 9 {
            anyhow::bail!("Invalid CPU line format");
        }

        Ok(CpuStats {
            timestamp,
            user: fields[1].parse().unwrap_or(0),
            nice: fields[2].parse().unwrap_or(0),
            system: fields[3].parse().unwrap_or(0),
            idle: fields[4].parse().unwrap_or(0),
            iowait: fields[5].parse().unwrap_or(0),
            irq: fields[6].parse().unwrap_or(0),
            softirq: fields[7].parse().unwrap_or(0),
            steal: fields[8].parse().unwrap_or(0),
        })
    }

    /// Read memory statistics from /proc/meminfo
    async fn read_memory_stats(timestamp: u64) -> Result<MemoryStats> {
        let meminfo_content = tokio::fs::read_to_string("/proc/meminfo")
            .await
            .context("Failed to read /proc/meminfo")?;

        let mut stats = HashMap::new();
        for line in meminfo_content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let key = parts[0].trim_end_matches(':');
                if let Ok(value) = parts[1].parse::<u64>() {
                    stats.insert(key.to_string(), value);
                }
            }
        }

        let total = stats.get("MemTotal").copied().unwrap_or(0);
        let free = stats.get("MemFree").copied().unwrap_or(0);
        let available = stats.get("MemAvailable").copied().unwrap_or(0);
        let buffers = stats.get("Buffers").copied().unwrap_or(0);
        let cached = stats.get("Cached").copied().unwrap_or(0);
        let used = total.saturating_sub(free).saturating_sub(buffers).saturating_sub(cached);
        let used_percent = if total > 0 {
            (used as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Ok(MemoryStats {
            timestamp,
            total_kb: total,
            free_kb: free,
            available_kb: available,
            buffers_kb: buffers,
            cached_kb: cached,
            used_kb: used,
            used_percent,
        })
    }

    /// Read scheduler statistics from /proc/schedstat
    async fn read_sched_stats(timestamp: u64) -> Result<SchedStats> {
        let schedstat_content = tokio::fs::read_to_string("/proc/schedstat")
            .await
            .context("Failed to read /proc/schedstat")?;

        let mut cpu_count = 0;
        let mut total_run_time = 0u64;
        let mut total_wait_time = 0u64;
        let mut total_timeslices = 0u64;

        for line in schedstat_content.lines() {
            if line.starts_with("cpu") {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() >= 9 {
                    cpu_count += 1;
                    total_run_time += fields[7].parse().unwrap_or(0);
                    total_wait_time += fields[8].parse().unwrap_or(0);
                }
            } else if line.starts_with("domain") {
                // Skip domain lines for now
                continue;
            } else if !line.starts_with("version") && !line.starts_with("timestamp") {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() >= 1 {
                    total_timeslices += fields[0].parse().unwrap_or(0);
                }
            }
        }

        Ok(SchedStats {
            timestamp,
            cpu_count,
            total_run_time,
            total_wait_time,
            total_timeslices,
        })
    }

    /// Generate summary statistics from a session
    fn generate_summary(session: &MonitoringSession) -> Result<MonitoringSummary> {
        let duration_secs = session
            .end_time
            .unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            })
            .saturating_sub(session.start_time);

        let sample_count = session.cpu_samples.len();

        // Calculate CPU usage percentages
        let mut cpu_percentages = Vec::new();
        for i in 1..session.cpu_samples.len() {
            let prev = &session.cpu_samples[i - 1];
            let curr = &session.cpu_samples[i];

            let prev_total = prev.user + prev.nice + prev.system + prev.idle
                + prev.iowait + prev.irq + prev.softirq + prev.steal;
            let curr_total = curr.user + curr.nice + curr.system + curr.idle
                + curr.iowait + curr.irq + curr.softirq + curr.steal;

            let total_diff = curr_total.saturating_sub(prev_total);
            let idle_diff = curr.idle.saturating_sub(prev.idle);

            if total_diff > 0 {
                let usage_percent = ((total_diff - idle_diff) as f64 / total_diff as f64) * 100.0;
                cpu_percentages.push(usage_percent);
            }
        }

        let cpu_avg_percent = if !cpu_percentages.is_empty() {
            cpu_percentages.iter().sum::<f64>() / cpu_percentages.len() as f64
        } else {
            0.0
        };

        let cpu_max_percent = cpu_percentages
            .iter()
            .copied()
            .fold(0.0f64, f64::max);

        // Memory statistics
        let memory_avg_percent = if !session.memory_samples.is_empty() {
            session.memory_samples.iter()
                .map(|s| s.used_percent)
                .sum::<f64>() / session.memory_samples.len() as f64
        } else {
            0.0
        };

        let memory_max_percent = session.memory_samples.iter()
            .map(|s| s.used_percent)
            .fold(0.0f64, f64::max);

        let memory_avg_used_mb = if !session.memory_samples.is_empty() {
            session.memory_samples.iter()
                .map(|s| s.used_kb as f64 / 1024.0)
                .sum::<f64>() / session.memory_samples.len() as f64
        } else {
            0.0
        };

        // Scheduler statistics
        let sched_total_timeslices = session.sched_samples.last()
            .map(|s| s.total_timeslices)
            .unwrap_or(0);

        let sched_avg_run_time_ns = if !session.sched_samples.is_empty() {
            session.sched_samples.iter()
                .map(|s| s.total_run_time)
                .sum::<u64>() / session.sched_samples.len() as u64
        } else {
            0
        };

        Ok(MonitoringSummary {
            session_id: session.session_id.clone(),
            duration_secs,
            sample_count,
            cpu_avg_percent,
            cpu_max_percent,
            memory_avg_percent,
            memory_max_percent,
            memory_avg_used_mb,
            sched_total_timeslices,
            sched_avg_run_time_ns,
        })
    }

    /// Get the current active session ID
    pub async fn get_active_session(&self) -> Option<String> {
        self.active_session.lock().await.clone()
    }

    /// Get a session by ID
    pub async fn get_session(&self, session_id: &str) -> Option<MonitoringSession> {
        self.sessions.lock().await.get(session_id).cloned()
    }
}