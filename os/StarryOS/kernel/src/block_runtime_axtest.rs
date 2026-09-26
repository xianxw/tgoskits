//! Starry axtests for the asynchronous block runtime boundary.

use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};
use core::{
    future::{Future, poll_fn},
    pin::{Pin, pin},
    task::Poll,
};

use ax_fs_ng::{BlockDeviceHandle, BlockError, BlockRegion, block_batch_stats};
use ax_runtime::hal::time::monotonic_time_nanos;
use rdif_block::CompletedRequest;

const SCRATCH_DISK_MARKER: &[u8] = b"TGOS_BLOCK_SCRATCH_V1\n";
const WRITE_TEST_BLOCKS: u32 = 8;
const MIXED_TEST_BLOCKS: u32 = 8;
const MULTI_DESCRIPTOR_TEST_BLOCKS: u32 = 16;
/// The functional sub-regions ([0, 24)) and the multi-descriptor test
/// ([24, 40)) must all fit the runtime-resolved scratch region.
const SCRATCH_MIN_REGION_BLOCKS: u32 = 40;

#[derive(Clone, Copy, Debug, Default)]
struct BlockBatchSnapshot {
    submitted_requests: u64,
    completed_requests: u64,
    failed_requests: u64,
    submission_batches: u64,
    commit_calls: u64,
    commit_failures: u64,
    largest_batch: usize,
    peak_inflight: usize,
}

fn block_batch_snapshot() -> BlockBatchSnapshot {
    let stats = block_batch_stats();
    BlockBatchSnapshot {
        submitted_requests: stats.submitted_requests,
        completed_requests: stats.completed_requests,
        failed_requests: stats.failed_requests,
        submission_batches: stats.submission_batches,
        commit_calls: stats.commit_calls,
        commit_failures: stats.commit_failures,
        largest_batch: stats.largest_batch,
        peak_inflight: stats.peak_inflight,
    }
}

fn block_batch_delta(before: BlockBatchSnapshot, after: BlockBatchSnapshot) -> BlockBatchSnapshot {
    BlockBatchSnapshot {
        submitted_requests: after
            .submitted_requests
            .saturating_sub(before.submitted_requests),
        completed_requests: after
            .completed_requests
            .saturating_sub(before.completed_requests),
        failed_requests: after.failed_requests.saturating_sub(before.failed_requests),
        submission_batches: after
            .submission_batches
            .saturating_sub(before.submission_batches),
        commit_calls: after.commit_calls.saturating_sub(before.commit_calls),
        commit_failures: after.commit_failures.saturating_sub(before.commit_failures),
        largest_batch: after.largest_batch,
        peak_inflight: after.peak_inflight,
    }
}

fn rate_per_second(count: u64, elapsed_ns: u64) -> u64 {
    count.saturating_mul(1_000_000_000) / elapsed_ns.max(1)
}
fn assert_successful_read_bytes(request: &CompletedRequest, byte_len: usize) {
    assert_eq!(request.result, Ok(()));
    assert_eq!(
        request.data.as_ref().map(|data| data.len().get()),
        Some(byte_len)
    );
}

fn assert_successful_read(request: &CompletedRequest, block_size: usize) {
    assert_successful_read_bytes(request, block_size);
}

/// Resolves the destructive-write scratch target. Returns `None` when the
/// machine offers neither path, in which case the write tests skip instead of
/// failing. Two discovery paths exist:
///
/// 1. An operator-declared reserved region on the root card, requested on the
///    kernel command line as `axtest.block_scratch=<start_lba>:<blocks>`. The
///    region must hold at least [`SCRATCH_MIN_REGION_BLOCKS`] blocks and must
///    not overlap any partition of the root disk. A request that fails its
///    eligibility checks is a hard error: the operator explicitly asked for
///    the destructive tests to run.
/// 2. A dedicated non-root disk whose first sector starts with the scratch
///    marker, usable without any command-line configuration.
fn writable_test_device() -> Option<(Arc<BlockDeviceHandle>, BlockRegion)> {
    let devices = BlockDeviceHandle::axtest_devices()
        .expect("block runtime must be installed for block axtest");

    if let Some(scratch) = ax_fs_ng::root::axtest_scratch_region_request() {
        let candidates = devices
            .iter()
            .filter(|device| !device.device_info().read_only)
            .filter(|device| {
                let info = device.device_info();
                // The region must be large enough for every test layout
                // (the functional sub-regions and the multi-descriptor test
                // together need 40 blocks).
                info.logical_block_size == 512
                    && scratch.num_blocks() >= u64::from(SCRATCH_MIN_REGION_BLOCKS)
                    && scratch.end_lba <= info.num_blocks
            })
            .filter(|device| {
                // The reserved extent lives on the root SD card; do not fall
                // back to an unrelated writable disk whose fixed LBA range has
                // not been explicitly identified as scratch space.
                ax_fs_ng::root::axtest_is_root_device(device)
            })
            .filter(|device| {
                let Some(root) = ax_fs_ng::root::axtest_root_region(device) else {
                    return false;
                };
                scratch.end_lba <= root.start_lba || root.end_lba <= scratch.start_lba
            })
            .filter(|device| {
                // init_root mounts every other recognized partition on this
                // disk too (a /boot partition, for example), so the scratch
                // extent must avoid all of them, not just the root one.
                ax_fs_ng::root::axtest_disk_partition_regions(device)
                    .iter()
                    .all(|partition| {
                        scratch.end_lba <= partition.start_lba
                            || partition.end_lba <= scratch.start_lba
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            candidates.len(),
            1,
            "requested scratch region [{}, {}) matches no eligible device",
            scratch.start_lba,
            scratch.end_lba,
        );
        let device = Arc::clone(candidates[0]);
        log_write_device(&device, &scratch);
        return Some((device, scratch));
    }

    let scratch = BlockRegion::new(1, u64::from(SCRATCH_MIN_REGION_BLOCKS));
    let candidates = devices
        .iter()
        .filter(|device| {
            !ax_fs_ng::root::axtest_is_root_device(device) && !device.device_info().read_only
        })
        .filter(|device| {
            let info = device.device_info();
            if info.num_blocks < scratch.end_lba
                || info.logical_block_size < SCRATCH_DISK_MARKER.len()
            {
                return false;
            }
            // init_root mounts every recognized partition on this disk too,
            // so the fixed scratch extent must not overlap any of them
            // before raw writes are allowed.
            if !ax_fs_ng::root::axtest_disk_partition_regions(device)
                .iter()
                .all(|partition| {
                    scratch.end_lba <= partition.start_lba || partition.end_lba <= scratch.start_lba
                })
            {
                return false;
            }
            let Ok(marker) = device.axtest_read_sync(0) else {
                return false;
            };
            if marker.result.is_err()
                || marker.data.as_ref().map(|data| data.len().get())
                    != Some(info.logical_block_size)
            {
                return false;
            }
            let Some(data) = marker.data.as_ref() else {
                return false;
            };
            let mut bytes = vec![0; info.logical_block_size];
            data.copy_to_slice_cpu(&mut bytes);
            bytes.starts_with(SCRATCH_DISK_MARKER)
        })
        .collect::<Vec<_>>();
    let device = Arc::clone(*candidates.first()?);
    log_write_device(&device, &scratch);
    Some((device, scratch))
}

fn log_write_device(device: &BlockDeviceHandle, scratch: &BlockRegion) {
    let info = device.device_info();
    // The root region is logged for the board record; eligibility was already
    // enforced by the candidate filters above.
    if let Some(root) = ax_fs_ng::root::axtest_root_region(device) {
        axtest::axtest_println!(
            "BLOCK_ROOT_REGION start_lba={} end_lba={}",
            root.start_lba,
            root.end_lba,
        );
    }
    axtest::axtest_println!(
        "BLOCK_WRITE_DEVICE name={} blocks={} logical_block_size={} scratch_start_lba={} \
         scratch_blocks={} scratch_end_lba={} scratch_last_lba={}",
        device.name(),
        info.num_blocks,
        info.logical_block_size,
        scratch.start_lba,
        scratch.num_blocks(),
        scratch.end_lba,
        scratch.end_lba - 1,
    );
}

/// Marks a destructive case as skipped because the machine offers neither a
/// bootargs-declared scratch region nor a dedicated marker scratch disk.
fn log_write_tests_skip() {
    axtest::axtest_println!(
        "BLOCK_WRITE_TESTS_SKIPPED reason=no eligible scratch device or region on this machine"
    );
}

fn pattern_bytes(block_size: usize, block_count: u32, seed: u8) -> Vec<u8> {
    let byte_len = block_size
        .checked_mul(block_count as usize)
        .expect("block axtest pattern length must fit in usize");
    (0..byte_len)
        .map(|index| {
            seed.wrapping_add(index as u8)
                .rotate_left((index % 8) as u32)
        })
        .collect()
}

fn read_region_sync(device: &Arc<BlockDeviceHandle>, lba: u64, block_count: u32) -> Vec<u8> {
    let block_size = device.device_info().logical_block_size;
    let byte_len = block_size
        .checked_mul(block_count as usize)
        .expect("block axtest read length must fit in usize");
    let request = device
        .axtest_read_blocks_sync(lba, block_count)
        .unwrap_or_else(|error| panic!("block axtest read failed: {error:?}"));
    assert_successful_read_bytes(&request, byte_len);
    let data = request
        .data
        .as_ref()
        .expect("successful block axtest read must return DMA data");
    let mut bytes = vec![0; byte_len];
    data.copy_to_slice_cpu(&mut bytes);
    bytes
}

fn assert_successful_write(request: &CompletedRequest) {
    assert_eq!(request.result, Ok(()));
}

struct ScratchRegionGuard {
    device: Arc<BlockDeviceHandle>,
    lba: u64,
    block_count: u32,
    original: Vec<u8>,
    armed: bool,
}

impl ScratchRegionGuard {
    fn new(device: Arc<BlockDeviceHandle>, lba: u64, block_count: u32, original: Vec<u8>) -> Self {
        Self {
            device,
            lba,
            block_count,
            original,
            armed: true,
        }
    }

    fn restore(&mut self) {
        if !self.armed {
            return;
        }
        let restore = self
            .device
            .axtest_write_blocks_sync(self.lba, self.block_count, &self.original)
            .unwrap_or_else(|error| panic!("scratch-region restore failed: {error:?}"));
        assert_successful_write(&restore);
        assert_eq!(
            read_region_sync(&self.device, self.lba, self.block_count),
            self.original,
            "scratch-region restore did not reproduce the original bytes"
        );
        self.armed = false;
    }
}

impl Drop for ScratchRegionGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        match self
            .device
            .axtest_write_blocks_sync(self.lba, self.block_count, &self.original)
        {
            Ok(request) if request.result.is_ok() => {
                axtest::axtest_println!(
                    "BLOCK_SCRATCH_RECOVERED_ON_DROP lba={} blocks={}",
                    self.lba,
                    self.block_count,
                );
            }
            Ok(request) => {
                axtest::axtest_println!(
                    "BLOCK_SCRATCH_RECOVERY_FAILED_ON_DROP lba={} blocks={} result={:?}",
                    self.lba,
                    self.block_count,
                    request.result,
                );
            }
            Err(error) => {
                axtest::axtest_println!(
                    "BLOCK_SCRATCH_RECOVERY_SUBMIT_FAILED_ON_DROP lba={} blocks={} error={:?}",
                    self.lba,
                    self.block_count,
                    error,
                );
            }
        }
    }
}

enum MixedExpectation {
    Write,
    Read(Vec<u8>),
}

async fn read_one(
    device: Arc<BlockDeviceHandle>,
    lba: u64,
) -> Result<CompletedRequest, BlockError> {
    device.axtest_read(lba).await
}

/// Confirms registration readiness and consistency between completion paths.
#[axtest::axtest]
fn block_runtime_registration_and_read_consistency() {
    let batch_before = block_batch_snapshot();
    let devices = BlockDeviceHandle::axtest_devices()
        .expect("block runtime must be installed for block axtest");
    assert!(!devices.is_empty(), "block runtime must register a device");
    let device = devices
        .first()
        .cloned()
        .expect("block runtime device list must contain the registered device");
    let info = device.device_info();
    assert!(
        info.num_blocks != 0,
        "registered block device must have blocks"
    );
    assert!(
        info.logical_block_size != 0,
        "registered block device must expose a logical block size"
    );
    axtest::axtest_println!(
        "BLOCK_DEVICE name={} blocks={} logical_block_size={}",
        device.name(),
        info.num_blocks,
        info.logical_block_size,
    );

    let synchronous = device
        .axtest_read_sync(0)
        .unwrap_or_else(|error| panic!("registered device synchronous read failed: {error:?}"));
    assert_successful_read(&synchronous, info.logical_block_size);
    let synchronous_data = synchronous
        .data
        .expect("successful synchronous read must return DMA data")
        .into_cpu_buffer();

    let asynchronous = crate::task::future::block_on(device.axtest_read(0))
        .unwrap_or_else(|error| panic!("registered device asynchronous read failed: {error:?}"));
    assert_successful_read(&asynchronous, info.logical_block_size);
    let asynchronous_data = asynchronous
        .data
        .expect("successful asynchronous read must return DMA data")
        .into_cpu_buffer();
    assert_eq!(
        synchronous_data.as_slice_cpu(),
        asynchronous_data.as_slice_cpu(),
        "synchronous and asynchronous reads of one LBA must agree"
    );

    let batch = block_batch_delta(batch_before, block_batch_snapshot());
    assert_eq!(batch.submitted_requests, 2);
    assert_eq!(batch.completed_requests, 2);
    assert_eq!(batch.failed_requests, 0);
    assert_eq!(batch.commit_calls, 2);
    assert_eq!(batch.commit_failures, 0);
}

/// Drives two independent block requests from one task. This exercises
/// submission, IRQ completion and per-request waker delivery through the real
/// Starry runtime without relying on a synthetic pending poll.
#[axtest::axtest]
fn block_runtime_async_double_read() {
    let batch_before = block_batch_snapshot();
    let devices = BlockDeviceHandle::axtest_devices()
        .expect("block runtime must be installed for block axtest");
    let device = devices
        .first()
        .cloned()
        .expect("block axtest requires an installed block device");
    let info = device.device_info();
    assert!(
        info.num_blocks >= 2,
        "block axtest requires at least two blocks"
    );
    let block_size = info.logical_block_size;
    let mut first = pin!(read_one(Arc::clone(&device), 0));
    let mut second = pin!(read_one(device, 1));
    let mut first_result = None;
    let mut second_result = None;
    let (first_result, second_result) = crate::task::future::block_on(poll_fn(|cx| {
        if first_result.is_none() {
            match first.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    first_result = Some(result);
                }
                Poll::Pending => {}
            }
        }
        if second_result.is_none() {
            match second.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    second_result = Some(result);
                }
                Poll::Pending => {}
            }
        }
        match (first_result.take(), second_result.take()) {
            (Some(first), Some(second)) => Poll::Ready((first, second)),
            (first, second) => {
                first_result = first;
                second_result = second;
                Poll::Pending
            }
        }
    }));

    let first = first_result.expect("first block request result");
    let second = second_result.expect("second block request result");
    assert_successful_read(&first, block_size);
    assert_successful_read(&second, block_size);
    let mut first_data = first.data.expect("first completed DMA").into_cpu_buffer();
    let second_data = second.data.expect("second completed DMA").into_cpu_buffer();
    let second_bytes = second_data.as_slice_cpu().to_vec();
    let mut replacement = second_bytes.clone();
    replacement[0] ^= u8::MAX;
    // Completion transfers each buffer back to the CPU. Independent requests
    // must not alias these independently retained ownership objects.
    first_data.copy_from_slice_cpu(&replacement);
    assert_eq!(first_data.as_slice_cpu(), replacement);
    assert_eq!(second_data.as_slice_cpu(), second_bytes);

    let batch = block_batch_delta(batch_before, block_batch_snapshot());
    assert_eq!(batch.submitted_requests, 2);
    assert_eq!(batch.completed_requests, 2);
    assert_eq!(batch.failed_requests, 0);
    // `commit_calls` counts runtime-to-driver batches, not requests. The two
    // submissions overlap in flight, so a queue with multi-request submit
    // batches may legitimately coalesce them into one commit; only require
    // that every commit succeeded.
    assert_eq!(batch.commit_failures, 0);
}

/// Validates a consecutive sequence of single-block writes and reads. The
/// original bytes are restored before the test completes so the scratch disk
/// remains reusable for the next case.
#[axtest::axtest]
fn block_runtime_single_block_contiguous_read_write() -> axtest::AxTestResult {
    let Some((device, scratch_region)) = writable_test_device() else {
        log_write_tests_skip();
        return axtest::AxTestResult::Ok;
    };
    let info = device.device_info();
    let lba = scratch_region.start_lba;
    let original = read_region_sync(&device, lba, WRITE_TEST_BLOCKS);
    let mut scratch =
        ScratchRegionGuard::new(Arc::clone(&device), lba, WRITE_TEST_BLOCKS, original);
    let pattern = pattern_bytes(info.logical_block_size, WRITE_TEST_BLOCKS, 0x31);
    let batch_before = block_batch_snapshot();
    let start = monotonic_time_nanos();

    for index in 0..WRITE_TEST_BLOCKS {
        let offset = index as usize * info.logical_block_size;
        let request = device
            .axtest_write_sync(
                lba + u64::from(index),
                &pattern[offset..offset + info.logical_block_size],
            )
            .unwrap_or_else(|error| panic!("single-block synchronous write failed: {error:?}"));
        assert_successful_write(&request);
    }
    for index in 0..WRITE_TEST_BLOCKS {
        let offset = index as usize * info.logical_block_size;
        let request = crate::task::future::block_on(device.axtest_read(lba + u64::from(index)))
            .unwrap_or_else(|error| panic!("single-block asynchronous read failed: {error:?}"));
        assert_successful_read(&request, info.logical_block_size);
        let data = request
            .data
            .as_ref()
            .expect("single-block read must return DMA data");
        let mut bytes = vec![0; info.logical_block_size];
        data.copy_to_slice_cpu(&mut bytes);
        assert_eq!(bytes, &pattern[offset..offset + info.logical_block_size]);
    }

    let elapsed_ns = monotonic_time_nanos().saturating_sub(start).max(1);
    let batch = block_batch_delta(batch_before, block_batch_snapshot());
    assert_eq!(batch.submitted_requests, u64::from(WRITE_TEST_BLOCKS * 2));
    assert_eq!(batch.completed_requests, batch.submitted_requests);
    assert_eq!(batch.failed_requests, 0);
    assert_eq!(batch.commit_failures, 0);

    scratch.restore();

    axtest::axtest_println!(
        "BLOCK_SINGLE_RW_FUNCTIONAL lba={} blocks={} bytes={} elapsed_ns={} bytes_per_sec={} \
         submitted={} completed={}",
        lba,
        WRITE_TEST_BLOCKS,
        pattern.len(),
        elapsed_ns,
        rate_per_second((pattern.len() as u64).saturating_mul(2), elapsed_ns),
        batch.submitted_requests,
        batch.completed_requests,
    );
    axtest::AxTestResult::Ok
}

fn run_multi_block_functional_test(
    device: Arc<BlockDeviceHandle>,
    lba: u64,
    block_count: u32,
    seed: u8,
    label: &str,
) {
    let info = device.device_info();
    let original = read_region_sync(&device, lba, block_count);
    let mut scratch = ScratchRegionGuard::new(Arc::clone(&device), lba, block_count, original);
    let pattern = pattern_bytes(info.logical_block_size, block_count, seed);
    let batch_before = block_batch_snapshot();
    let start = monotonic_time_nanos();

    let write =
        crate::task::future::block_on(device.axtest_write_blocks(lba, block_count, &pattern))
            .unwrap_or_else(|error| panic!("multi-block asynchronous write failed: {error:?}"));
    assert_successful_write(&write);
    let read = crate::task::future::block_on(device.axtest_read_blocks(lba, block_count))
        .unwrap_or_else(|error| panic!("multi-block asynchronous read failed: {error:?}"));
    assert_successful_read_bytes(&read, pattern.len());
    let data = read
        .data
        .as_ref()
        .expect("multi-block read must return DMA data");
    let mut read_back = vec![0; pattern.len()];
    data.copy_to_slice_cpu(&mut read_back);
    assert_eq!(read_back, pattern);

    let elapsed_ns = monotonic_time_nanos().saturating_sub(start).max(1);
    let batch = block_batch_delta(batch_before, block_batch_snapshot());
    assert_eq!(batch.submitted_requests, 2);
    assert_eq!(batch.completed_requests, 2);
    assert_eq!(batch.failed_requests, 0);
    assert_eq!(batch.commit_failures, 0);

    scratch.restore();

    axtest::axtest_println!(
        "{label} lba={} blocks={} bytes={} elapsed_ns={} bytes_per_sec={} submitted={} \
         completed={}",
        lba,
        block_count,
        pattern.len(),
        elapsed_ns,
        rate_per_second((pattern.len() as u64).saturating_mul(2), elapsed_ns),
        batch.submitted_requests,
        batch.completed_requests,
    );
}

/// Validates a true multi-block request (`block_count > 1`) through both the
/// asynchronous write and asynchronous read completion paths, including
/// complete buffer contents and block boundaries.
#[axtest::axtest]
fn block_runtime_multi_block_contiguous_read_write() -> axtest::AxTestResult {
    let Some((device, scratch_region)) = writable_test_device() else {
        log_write_tests_skip();
        return axtest::AxTestResult::Ok;
    };
    run_multi_block_functional_test(
        device,
        scratch_region.start_lba + u64::from(WRITE_TEST_BLOCKS),
        WRITE_TEST_BLOCKS,
        0x72,
        "BLOCK_MULTI_RW_FUNCTIONAL",
    );
    axtest::AxTestResult::Ok
}

/// Exercises a real chained IDMAC transfer on VisionFive 2. The kernel DMA
/// allocator returns 4 KiB-aligned buffers and each DW-MMC descriptor carries
/// at most 4 KiB, so this 8 KiB request requires two hardware descriptors.
#[axtest::axtest]
fn block_runtime_multi_descriptor_contiguous_read_write() -> axtest::AxTestResult {
    let Some((device, scratch_region)) = writable_test_device() else {
        log_write_tests_skip();
        return axtest::AxTestResult::Ok;
    };
    run_multi_block_functional_test(
        device,
        scratch_region.start_lba + u64::from(WRITE_TEST_BLOCKS * 2 + MIXED_TEST_BLOCKS),
        MULTI_DESCRIPTOR_TEST_BLOCKS,
        0xC7,
        "BLOCK_MULTI_DESCRIPTOR_RW_FUNCTIONAL",
    );
    axtest::AxTestResult::Ok
}

/// Interleaves asynchronous writes and reads in one task. Reads target the
/// untouched odd blocks while writes target even blocks, so each completion
/// has an independent, deterministic expected result.
#[axtest::axtest]
fn block_runtime_async_mixed_read_write() -> axtest::AxTestResult {
    type MixedFuture = Pin<Box<dyn Future<Output = Result<CompletedRequest, BlockError>>>>;

    let Some((device, scratch_region)) = writable_test_device() else {
        log_write_tests_skip();
        return axtest::AxTestResult::Ok;
    };
    let info = device.device_info();
    let lba = scratch_region.start_lba + u64::from(WRITE_TEST_BLOCKS * 2);
    let block_count = MIXED_TEST_BLOCKS;
    let original = read_region_sync(&device, lba, block_count);
    let mut scratch = ScratchRegionGuard::new(Arc::clone(&device), lba, block_count, original);
    let pattern = pattern_bytes(info.logical_block_size, block_count / 2, 0xA5);
    let batch_before = block_batch_snapshot();
    let start = monotonic_time_nanos();
    let mut futures: Vec<Option<(MixedExpectation, MixedFuture)>> = Vec::new();

    for index in 0..block_count {
        let target_lba = lba + u64::from(index);
        if index % 2 == 0 {
            let offset = (index as usize / 2) * info.logical_block_size;
            let block = pattern[offset..offset + info.logical_block_size].to_vec();
            let future_device = Arc::clone(&device);
            futures.push(Some((
                MixedExpectation::Write,
                Box::pin(async move { future_device.axtest_write(target_lba, &block).await })
                    as MixedFuture,
            )));
        } else {
            let offset = index as usize * info.logical_block_size;
            let expected = scratch.original[offset..offset + info.logical_block_size].to_vec();
            let future_device = Arc::clone(&device);
            futures.push(Some((
                MixedExpectation::Read(expected),
                Box::pin(async move { future_device.axtest_read(target_lba).await }) as MixedFuture,
            )));
        }
    }

    let mut mixed_ok = true;
    crate::task::future::block_on(poll_fn(|cx| {
        let mut pending = false;
        for slot in &mut futures {
            let poll = slot.as_mut().map(|(_, future)| future.as_mut().poll(cx));
            match poll {
                Some(Poll::Ready(Ok(request))) => {
                    let (expectation, _) = slot.take().expect("ready mixed request ownership");
                    match expectation {
                        MixedExpectation::Write => {
                            if request.result.is_err() {
                                mixed_ok = false;
                            }
                        }
                        MixedExpectation::Read(expected) => {
                            if request.result.is_err() {
                                mixed_ok = false;
                            } else if request.data.as_ref().map(|data| data.len().get())
                                != Some(info.logical_block_size)
                            {
                                mixed_ok = false;
                            } else {
                                let data = request.data.as_ref().expect("checked read data");
                                let mut bytes = vec![0; info.logical_block_size];
                                data.copy_to_slice_cpu(&mut bytes);
                                if bytes != expected {
                                    mixed_ok = false;
                                }
                            }
                        }
                    }
                }
                Some(Poll::Ready(Err(_))) => {
                    let _ = slot.take();
                    mixed_ok = false;
                }
                Some(Poll::Pending) => pending = true,
                None => {}
            }
        }
        if pending {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }));

    let elapsed_ns = monotonic_time_nanos().saturating_sub(start).max(1);
    let batch = block_batch_delta(batch_before, block_batch_snapshot());
    assert_eq!(batch.submitted_requests, u64::from(block_count));
    assert_eq!(batch.completed_requests, batch.submitted_requests);
    assert_eq!(batch.failed_requests, 0);
    assert_eq!(batch.commit_failures, 0);

    let mixed_read_back = read_region_sync(&device, lba, block_count);
    for index in 0..block_count as usize {
        let offset = index * info.logical_block_size;
        let expected = if index % 2 == 0 {
            let pattern_offset = (index / 2) * info.logical_block_size;
            &pattern[pattern_offset..pattern_offset + info.logical_block_size]
        } else {
            &scratch.original[offset..offset + info.logical_block_size]
        };
        if mixed_read_back[offset..offset + info.logical_block_size] != expected[..] {
            mixed_ok = false;
        }
    }

    scratch.restore();
    assert!(
        mixed_ok,
        "mixed read/write completion or data validation failed"
    );

    axtest::axtest_println!(
        "BLOCK_MIXED_RW_FUNCTIONAL lba={} blocks={} bytes={} elapsed_ns={} bytes_per_sec={} \
         submitted={} completed={}",
        lba,
        block_count,
        mixed_read_back.len(),
        elapsed_ns,
        rate_per_second(mixed_read_back.len() as u64, elapsed_ns),
        batch.submitted_requests,
        batch.completed_requests,
    );
    axtest::AxTestResult::Ok
}
