// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title YangGo
/// @notice On-chain registry for AI training runs: datasets, model checkpoints, and gradient attestations.
/// @dev Training coordinators submit runs; validators attest; rewards accrue by run and by validator.
///
/// Designed for decentralized ML pipelines with verifiable training metadata and checkpoint hashes.
///
/// Version 2 introduces run phases, config presets, and optional gradient snapshot extensions.
/// Compatible with EIP-1559 and standard access patterns; governor and fee collector are immutable.

contract YangGo {

    // -------------------------------------------------------------------------
    // EVENTS
    // -------------------------------------------------------------------------

    event RunRegistered(
        uint256 indexed runId,
        bytes32 datasetHash,
        bytes32 configHash,
        uint8 modelTier,
        uint256 epochCount,
        address indexed coordinator,
        uint256 registeredAt
    );
    event CheckpointAttached(uint256 indexed runId, uint256 checkpointIndex, bytes32 checkpointHash, uint256 atBlock);
    event ValidatorAttestation(uint256 indexed runId, address indexed validator, bool approved, uint256 atBlock);
    event RewardDistributed(address indexed recipient, uint256 amount, uint256 atBlock);
    event CoordinatorWhitelisted(address indexed coordinator, address indexed setBy, uint256 atBlock);
    event CoordinatorRemoved(address indexed coordinator, address indexed setBy, uint256 atBlock);
    event ValidatorRegistered(address indexed validator, uint256 stakeAmount, uint256 atBlock);
    event ValidatorUnregistered(address indexed validator, uint256 atBlock);
    event TrainingPaused(uint256 atBlock, address indexed caller);
    event TrainingResumed(uint256 atBlock, address indexed caller);
    event EpochLimitUpdated(uint256 previousMax, uint256 newMax, uint256 atBlock);
    event FeeCollected(address indexed from, uint256 amount, uint256 atBlock);
    event FallbackDeposit(address indexed from, uint256 amount);

    // -------------------------------------------------------------------------
    // ERRORS
    // -------------------------------------------------------------------------

    error YangGo_NotGovernor();
    error YangGo_NotCoordinator();
    error YangGo_NotValidator();
    error YangGo_TrainingPaused();
    error YangGo_InvalidRunId();
    error YangGo_RunAlreadyFinalized();
    error YangGo_InvalidEpochCount();
    error YangGo_InvalidModelTier();
    error YangGo_ZeroHash();
    error YangGo_ZeroAddress();
    error YangGo_Reentrancy();
    error YangGo_TransferFailed();
    error YangGo_AlreadyAttested();
    error YangGo_MinStakeRequired();
    error YangGo_ValidatorNotRegistered();
    error YangGo_CheckpointIndexOutOfRange();
    error YangGo_ArrayLengthMismatch();
    error YangGo_QuorumNotReached();
    error YangGo_RunNotFinalized();
    error YangGo_InvalidFee();
    error YangGo_InsufficientPayment();
    error YangGo_CoordinatorNotWhitelisted();
    error YangGo_InvalidBatchSize();

    // -------------------------------------------------------------------------
    // CONSTANTS
    // -------------------------------------------------------------------------

    uint256 public constant YANGGO_VERSION = 2;
    uint8 public constant MODEL_TIER_BASE = 1;
    uint8 public constant MODEL_TIER_MID = 2;
    uint8 public constant MODEL_TIER_LARGE = 3;
    uint8 public constant MODEL_TIER_XL = 4;
    uint8 public constant MAX_MODEL_TIER = 4;
    uint256 public constant MIN_EPOCHS = 1;
    uint256 public constant MAX_EPOCHS_DEFAULT = 10000;
    uint256 public constant MIN_VALIDATOR_STAKE = 0.1 ether;
    uint256 public constant QUORUM_BPS = 6600;
