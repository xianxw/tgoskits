//! Pure eligibility policy for the destructive block axtest scratch target.
//!
//! The destructive write tests need a region that no mounted filesystem
//! uses. Whether a device qualifies is the safety boundary that keeps raw
//! writes away from mounted data, so the decision is kept free of device
//! I/O here and covered by deterministic host unit tests; the device scan
//! and marker read live in [`crate::block_runtime_axtest`].

use ax_fs_ng::BlockRegion;
use rdif_block::DeviceInfo;

/// First-sector marker identifying a dedicated scratch disk.
pub(crate) const SCRATCH_DISK_MARKER: &[u8] = b"TGOS_BLOCK_SCRATCH_V1\n";

/// Blocks the destructive tests may touch: the functional sub-regions
/// ([0, 24)) plus the multi-descriptor test ([24, 40)) must all fit the
/// resolved scratch region.
pub(crate) const SCRATCH_MIN_REGION_BLOCKS: u32 = 40;

/// The fixed scratch extent a dedicated marker disk must reserve: `[1, 41)`.
/// LBA 0 stays untouched so the marker itself remains readable.
pub(crate) fn marker_scratch_region() -> BlockRegion {
    BlockRegion::new(1, u64::from(SCRATCH_MIN_REGION_BLOCKS))
}

/// Half-open extent disjointness: the regions share no LBA. Extents that
/// merely touch (one ends where the other starts) are disjoint.
pub(crate) fn regions_disjoint(left: BlockRegion, right: BlockRegion) -> bool {
    left.end_lba <= right.start_lba || right.end_lba <= left.start_lba
}

/// Decides whether the root disk may host the operator-declared scratch
/// region (`axtest.block_scratch=<start_lba>:<blocks>`).
///
/// `root_filesystem_region` is `None` for disks that are not the root
/// device, so only the root disk can qualify. The extent must be writable,
/// address 512-byte sectors (the functional layouts' addressing unit), hold
/// at least [`SCRATCH_MIN_REGION_BLOCKS`] blocks, fit inside the device, and
/// avoid the mounted root filesystem plus every other identified partition
/// (init_root mounts the non-root partitions alongside the root filesystem).
pub(crate) fn declared_scratch_device_eligible(
    info: DeviceInfo,
    scratch: BlockRegion,
    root_filesystem_region: Option<BlockRegion>,
    partition_regions: &[BlockRegion],
) -> bool {
    !info.read_only
        && info.logical_block_size == 512
        && scratch.num_blocks() >= u64::from(SCRATCH_MIN_REGION_BLOCKS)
        && scratch.end_lba <= info.num_blocks
        && root_filesystem_region.is_some_and(|root| regions_disjoint(scratch, root))
        && partition_regions
            .iter()
            .all(|&partition| regions_disjoint(scratch, partition))
}

/// Decides whether a disk qualifies as a dedicated marker scratch disk.
///
/// The fixed [`marker_scratch_region`] extent must fit the device, the
/// sector size must hold the first-sector marker, and no identified
/// partition may reach into the extent. The root disk and read-only devices
/// never qualify; the caller must additionally verify the marker bytes.
pub(crate) fn marker_scratch_device_eligible(
    info: DeviceInfo,
    is_root: bool,
    partition_regions: &[BlockRegion],
) -> bool {
    let scratch = marker_scratch_region();
    !is_root
        && !info.read_only
        && info.num_blocks >= scratch.end_lba
        && info.logical_block_size >= SCRATCH_DISK_MARKER.len()
        && partition_regions
            .iter()
            .all(|&partition| regions_disjoint(scratch, partition))
}

#[cfg(all(test, not(axtest)))]
mod tests {
    use super::*;

    fn writable_device(num_blocks: u64) -> DeviceInfo {
        DeviceInfo::new(num_blocks, 512)
    }

    // VisionFive 2 card layout: the rootfs occupies [2099456, 124735488) and
    // the region declared by `axtest.block_scratch=2099200:256` sits just
    // below it, sharing no LBA with the filesystem.
    const CARD_BLOCKS: u64 = 124_735_488;
    const ROOT_FILESYSTEM: BlockRegion = BlockRegion::new(2_099_456, CARD_BLOCKS - 2_099_456);
    const DECLARED_SCRATCH: BlockRegion = BlockRegion::new(2_099_200, 256);

    #[test]
    fn declared_scratch_accepts_the_reserved_region_below_the_root_filesystem() {
        assert!(declared_scratch_device_eligible(
            writable_device(CARD_BLOCKS),
            DECLARED_SCRATCH,
            Some(ROOT_FILESYSTEM),
            &[ROOT_FILESYSTEM],
        ));
    }

    #[test]
    fn declared_scratch_rejects_extents_shorter_than_the_test_layout() {
        // 16 blocks was accepted before the minimum-length check existed and
        // let the multi-descriptor test write past the declared reservation.
        let short = BlockRegion::new(2_099_200, 16);
        assert!(!declared_scratch_device_eligible(
            writable_device(CARD_BLOCKS),
            short,
            Some(ROOT_FILESYSTEM),
            &[ROOT_FILESYSTEM],
        ));
        let missing_one = BlockRegion::new(
            2_099_200,
            u64::from(SCRATCH_MIN_REGION_BLOCKS) - 1,
        );
        assert!(!declared_scratch_device_eligible(
            writable_device(CARD_BLOCKS),
            missing_one,
            Some(ROOT_FILESYSTEM),
            &[ROOT_FILESYSTEM],
        ));
        let exact = BlockRegion::new(2_099_200, u64::from(SCRATCH_MIN_REGION_BLOCKS));
        assert!(declared_scratch_device_eligible(
            writable_device(CARD_BLOCKS),
            exact,
            Some(ROOT_FILESYSTEM),
            &[ROOT_FILESYSTEM],
        ));
    }

    #[test]
    fn declared_scratch_rejects_extents_passing_the_device_end() {
        // The root filesystem ends at LBA 100 and the device ends exactly
        // with the declared extent.
        let root = BlockRegion::new(0, 100);
        let fitting = BlockRegion::new(100, 256);
        assert!(declared_scratch_device_eligible(
            writable_device(356),
            fitting,
            Some(root),
            &[root],
        ));
        // One block beyond the device must disqualify the extent.
        let overflowing = BlockRegion::new(101, 256);
        assert!(!declared_scratch_device_eligible(
            writable_device(356),
            overflowing,
            Some(root),
            &[root],
        ));
    }

    #[test]
    fn declared_scratch_must_not_overlap_the_root_filesystem_region() {
        let root = BlockRegion::new(1_000, 1_000);
        let overlapping = BlockRegion::new(900, 256);
        assert!(!declared_scratch_device_eligible(
            writable_device(4_096),
            overlapping,
            Some(root),
            &[root],
        ));
        // Extents adjacent to the filesystem share no LBA and stay allowed.
        let before = BlockRegion::new(744, 256);
        assert!(declared_scratch_device_eligible(
            writable_device(4_096),
            before,
            Some(root),
            &[root],
        ));
        let after = BlockRegion::new(2_000, 256);
        assert!(declared_scratch_device_eligible(
            writable_device(4_096),
            after,
            Some(root),
            &[root],
        ));
    }

    #[test]
    fn declared_scratch_must_not_overlap_other_partitions_on_the_root_disk() {
        // init_root mounts a /boot partition alongside the root filesystem,
        // so a scratch region inside it must be rejected even though it does
        // not overlap the root filesystem itself.
        let boot = BlockRegion::new(2_048, 2_048);
        let root = BlockRegion::new(1_048_576, CARD_BLOCKS - 1_048_576);
        let overlapping_boot = BlockRegion::new(1_024, 2_048);
        assert!(!declared_scratch_device_eligible(
            writable_device(CARD_BLOCKS),
            overlapping_boot,
            Some(root),
            &[boot, root],
        ));
        let adjacent = BlockRegion::new(0, 2_048);
        assert!(declared_scratch_device_eligible(
            writable_device(CARD_BLOCKS),
            adjacent,
            Some(root),
            &[boot, root],
        ));
    }

    #[test]
    fn declared_scratch_only_applies_to_the_writable_root_disk() {
        // A missing root filesystem region means the disk is not the root
        // device, so an operator declaration must never select it.
        assert!(!declared_scratch_device_eligible(
            writable_device(CARD_BLOCKS),
            DECLARED_SCRATCH,
            None,
            &[],
        ));
        let mut read_only = writable_device(CARD_BLOCKS);
        read_only.read_only = true;
        assert!(!declared_scratch_device_eligible(
            read_only,
            DECLARED_SCRATCH,
            Some(ROOT_FILESYSTEM),
            &[ROOT_FILESYSTEM],
        ));
        let mut large_sector = writable_device(CARD_BLOCKS);
        large_sector.logical_block_size = 4_096;
        assert!(!declared_scratch_device_eligible(
            large_sector,
            DECLARED_SCRATCH,
            Some(ROOT_FILESYSTEM),
            &[ROOT_FILESYSTEM],
        ));
    }

    #[test]
    fn marker_scratch_rejects_partitions_reaching_into_the_fixed_extent() {
        // The extent is [1, 41): a partition starting at LBA 40 shares LBA
        // 40 with the multi-descriptor test region and must disqualify the
        // disk. Comparing against the block count instead of the half-open
        // end LBA accepted this layout and corrupted the first partition
        // sector.
        let reaching_in = BlockRegion::new(40, 60);
        assert!(!marker_scratch_device_eligible(
            writable_device(1_024),
            false,
            &[reaching_in],
        ));
        // A partition starting exactly at the extent end shares no LBA.
        let after = BlockRegion::new(41, 59);
        assert!(marker_scratch_device_eligible(
            writable_device(1_024),
            false,
            &[after],
        ));
        // LBA 0 stays outside the extent, so a partition covering only the
        // boot sector is fine while one reaching LBA 1 is not.
        let boot_sector_only = BlockRegion::new(0, 1);
        assert!(marker_scratch_device_eligible(
            writable_device(1_024),
            false,
            &[boot_sector_only],
        ));
        let covering_start = BlockRegion::new(0, 2);
        assert!(!marker_scratch_device_eligible(
            writable_device(1_024),
            false,
            &[covering_start],
        ));
    }

    #[test]
    fn marker_scratch_requires_the_extent_to_fit_the_disk() {
        assert!(!marker_scratch_device_eligible(
            writable_device(40),
            false,
            &[],
        ));
        assert!(marker_scratch_device_eligible(
            writable_device(41),
            false,
            &[],
        ));
    }

    #[test]
    fn marker_scratch_rejects_root_readonly_and_small_sector_disks() {
        assert!(!marker_scratch_device_eligible(
            writable_device(1_024),
            true,
            &[],
        ));
        let mut read_only = writable_device(1_024);
        read_only.read_only = true;
        assert!(!marker_scratch_device_eligible(
            read_only,
            false,
            &[],
        ));
        // The marker must fit one logical sector.
        let tiny_sector = DeviceInfo::new(1_024, 16);
        assert!(!marker_scratch_device_eligible(
            tiny_sector,
            false,
            &[],
        ));
    }
}
