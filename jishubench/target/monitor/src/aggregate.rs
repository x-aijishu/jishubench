//! Aggregate computation over a completed timeseries + mark list.

use crate::api_types::{
    Aggregates, MemAvailableAggregate, SampleRecord, StatAggregate, SwapAggregate,
};

/// Compute aggregates from a completed trace.
///
/// `duration_ns` is `stopped_at - started_at` in nanoseconds.
pub fn compute(duration_ns: u64, samples: &[SampleRecord]) -> Aggregates {
    let duration_ms = duration_ns as f64 / 1_000_000.0;

    // mem_available_mb aggregate – mean + min
    let mem_available_mb = {
        let vals: Vec<f64> = samples.iter().filter_map(|s| s.mem_available_mb).collect();
        if vals.is_empty() {
            None
        } else {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            Some(MemAvailableAggregate { mean, min })
        }
    };

    // swap aggregate – initial / final / delta
    let swap = {
        let non_null: Vec<f64> = samples.iter().filter_map(|s| s.swap_used_mb).collect();
        if non_null.is_empty() {
            None
        } else {
            Some(SwapAggregate {
                initial_mb: *non_null.first().unwrap_or(&0.0),
                final_mb: *non_null.last().unwrap_or(&0.0),
                delta_mb: non_null.last().unwrap_or(&0.0) - non_null.first().unwrap_or(&0.0),
            })
        }
    };

    Aggregates {
        duration_ms,
        system_cpu_util_pct: stat_aggregate(samples.iter().filter_map(|s| s.system_cpu_util_pct)),
        mem_available_mb,
        swap,
    }
}

fn stat_aggregate<I>(values: I) -> Option<StatAggregate>
where
    I: Iterator<Item = f64>,
{
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        return None;
    }
    Some(StatAggregate {
        mean: values.iter().sum::<f64>() / values.len() as f64,
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_system_cpu_utilization() {
        let aggregate = stat_aggregate([20.0, 40.0, 30.0].into_iter()).unwrap();
        assert_eq!(aggregate.mean, 30.0);
        assert_eq!(aggregate.max, 40.0);
    }
}
