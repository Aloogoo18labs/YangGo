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
    uint256 public constant BPS_DENOM = 10000;
    uint256 public constant MAX_CHECKPOINTS_PER_RUN = 256;
    uint256 public constant MAX_BATCH_REGISTER = 24;
    bytes32 public constant YANGGO_DOMAIN = keccak256("YangGo.TrainingRun.v2");

    // -------------------------------------------------------------------------
    // IMMUTABLES
    // -------------------------------------------------------------------------

    address public immutable governor;
    address public immutable feeCollector;
    address public immutable rewardPool;

    // -------------------------------------------------------------------------
    // STATE
    // -------------------------------------------------------------------------

    struct TrainingRun {
        bytes32 datasetHash;
        bytes32 configHash;
        uint8 modelTier;
        uint256 epochCount;
        address coordinator;
        uint256 registeredAt;
        bool finalized;
        uint256 positiveAttestations;
        uint256 totalAttestations;
        bytes32[] checkpoints;
    }

    struct ValidatorState {
        uint256 stake;
        bool active;
        mapping(uint256 => bool) attestedRuns;
    }

    mapping(address => bool) public coordinatorWhitelist;
    mapping(address => ValidatorState) private _validators;
    address[] private _validatorList;
    TrainingRun[] private _runs;
    uint256 public maxEpochsPerRun = 5000;
    uint256 public registrationFeeWei = 0.01 ether;
    bool public trainingPaused;
    uint256 private _lock;

    enum RunPhase { Draft, CheckpointPhase, AttestationPhase, Finalized }
    struct RunMeta {
        bytes32 tag;
        uint256 phaseLockUntil;
        bool finalizeRequested;
        uint256 requestedAt;
    }
    struct ConfigPreset {
        bytes32 configHash;
        string label;
        uint8 modelTier;
        uint256 suggestedEpochs;
        bool active;
    }

    mapping(uint256 => RunMeta) private _runMeta;
    mapping(bytes32 => ConfigPreset) private _configPresets;
    bytes32[] private _presetIds;
    uint256 public finalizeDelaySeconds = 3600;
    uint256 public constant MIN_FINALIZE_DELAY = 300;
    uint256 public constant MAX_FINALIZE_DELAY = 604800;
    uint256 public constant TIER_BASE_FEE_MULTIPLIER = 10000;
    uint256 public constant TIER_MID_FEE_MULTIPLIER = 12000;
    uint256 public constant TIER_LARGE_FEE_MULTIPLIER = 15000;
    uint256 public constant TIER_XL_FEE_MULTIPLIER = 20000;
    uint256 public constant FEE_MULTIPLIER_DENOM = 10000;
    uint256 public constant MAX_TAG_LENGTH_BYTES = 32;
    uint256 public constant EPOCH_BUCKET_SMALL = 10;
    uint256 public constant EPOCH_BUCKET_MED = 100;
    uint256 public constant EPOCH_BUCKET_LARGE = 500;
    uint256 public constant EPOCH_BUCKET_XL = 1000;
    mapping(address => uint256[]) private _runIdsByCoordinator;
    uint256 private _totalStaked;
    mapping(address => uint256) private _validatorStakeSnapshot;

    event RunTagSet(uint256 indexed runId, bytes32 tag, uint256 atBlock);
    event FinalizeRequested(uint256 indexed runId, uint256 executeAfter, uint256 atBlock);
    event PresetAdded(bytes32 indexed presetId, bytes32 configHash, string label, uint256 atBlock);
    event PresetDisabled(bytes32 indexed presetId, uint256 atBlock);

    // -------------------------------------------------------------------------
    // CONSTRUCTOR
    // -------------------------------------------------------------------------

    constructor() {
        governor = 0x7f3a91c2e5b4d806f9b0c1e3d5a7f2e8c4b6a0d9;
        feeCollector = 0x2b4c6d8e0f1a3b5c7d9e1f3a5b7c9d0e2f4a6b8;
        rewardPool = 0x9e8d7c6b5a493827160504938271605049382716;
        coordinatorWhitelist[0x5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6] = true;
        coordinatorWhitelist[0xa1b2c3d4e5f60718293a4b5c6d7e8f9012345678] = true;
    }

    // -------------------------------------------------------------------------
    // MODIFIERS
    // -------------------------------------------------------------------------

    modifier onlyGovernor() {
        if (msg.sender != governor) revert YangGo_NotGovernor();
        _;
    }

    modifier onlyWhitelistedCoordinator() {
        if (!coordinatorWhitelist[msg.sender]) revert YangGo_CoordinatorNotWhitelisted();
        _;
    }

    modifier whenTrainingActive() {
        if (trainingPaused) revert YangGo_TrainingPaused();
        _;
    }

    modifier nonReentrant() {
        if (_lock != 0) revert YangGo_Reentrancy();
        _lock = 1;
        _;
        _lock = 0;
    }

    // -------------------------------------------------------------------------
    // COORDINATOR: REGISTER RUN
    // -------------------------------------------------------------------------

    function registerRun(
        bytes32 datasetHash,
        bytes32 configHash,
        uint8 modelTier,
        uint256 epochCount
    ) external payable onlyWhitelistedCoordinator whenTrainingActive nonReentrant returns (uint256 runId) {
        if (datasetHash == bytes32(0)) revert YangGo_ZeroHash();
        if (configHash == bytes32(0)) revert YangGo_ZeroHash();
        if (modelTier == 0 || modelTier > MAX_MODEL_TIER) revert YangGo_InvalidModelTier();
        if (epochCount < MIN_EPOCHS || epochCount > maxEpochsPerRun) revert YangGo_InvalidEpochCount();
        if (msg.value < registrationFeeWei) revert YangGo_InsufficientPayment();

        _runs.push(TrainingRun({
            datasetHash: datasetHash,
            configHash: configHash,
            modelTier: modelTier,
            epochCount: epochCount,
            coordinator: msg.sender,
            registeredAt: block.timestamp,
            finalized: false,
            positiveAttestations: 0,
            totalAttestations: 0,
            checkpoints: new bytes32[](0)
        }));
        runId = _runs.length - 1;
        _runIdsByCoordinator[msg.sender].push(runId);
        _runMeta[runId] = RunMeta({ tag: bytes32(0), phaseLockUntil: 0, finalizeRequested: false, requestedAt: 0 });

        (bool sent,) = feeCollector.call{value: registrationFeeWei}("");
        if (!sent) revert YangGo_TransferFailed();
        if (msg.value > registrationFeeWei) {
            (bool extra,) = msg.sender.call{value: msg.value - registrationFeeWei}("");
            if (!extra) revert YangGo_TransferFailed();
        }

        emit RunRegistered(runId, datasetHash, configHash, modelTier, epochCount, msg.sender, block.timestamp);
        emit FeeCollected(msg.sender, registrationFeeWei, block.timestamp);
