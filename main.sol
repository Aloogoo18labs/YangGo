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
    }

    function registerRunBatch(
        bytes32[] calldata datasetHashes,
        bytes32[] calldata configHashes,
        uint8[] calldata modelTiers,
        uint256[] calldata epochCounts
    ) external payable onlyWhitelistedCoordinator whenTrainingActive nonReentrant returns (uint256[] memory runIds) {
        uint256 n = datasetHashes.length;
        if (n != configHashes.length || n != modelTiers.length || n != epochCounts.length) revert YangGo_ArrayLengthMismatch();
        if (n > MAX_BATCH_REGISTER) revert YangGo_InvalidBatchSize();

        uint256 totalFee = registrationFeeWei * n;
        if (msg.value < totalFee) revert YangGo_InsufficientPayment();

        runIds = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            if (datasetHashes[i] == bytes32(0) || configHashes[i] == bytes32(0)) revert YangGo_ZeroHash();
            if (modelTiers[i] == 0 || modelTiers[i] > MAX_MODEL_TIER) revert YangGo_InvalidModelTier();
            if (epochCounts[i] < MIN_EPOCHS || epochCounts[i] > maxEpochsPerRun) revert YangGo_InvalidEpochCount();

            _runs.push(TrainingRun({
                datasetHash: datasetHashes[i],
                configHash: configHashes[i],
                modelTier: modelTiers[i],
                epochCount: epochCounts[i],
                coordinator: msg.sender,
                registeredAt: block.timestamp,
                finalized: false,
                positiveAttestations: 0,
                totalAttestations: 0,
                checkpoints: new bytes32[](0)
            }));
            runIds[i] = _runs.length - 1;
            _runIdsByCoordinator[msg.sender].push(runIds[i]);
            _runMeta[runIds[i]] = RunMeta({ tag: bytes32(0), phaseLockUntil: 0, finalizeRequested: false, requestedAt: 0 });
            emit RunRegistered(runIds[i], datasetHashes[i], configHashes[i], modelTiers[i], epochCounts[i], msg.sender, block.timestamp);
        }

        (bool sent,) = feeCollector.call{value: totalFee}("");
        if (!sent) revert YangGo_TransferFailed();
        if (msg.value > totalFee) {
            (bool extra,) = msg.sender.call{value: msg.value - totalFee}("");
            if (!extra) revert YangGo_TransferFailed();
        }
        emit FeeCollected(msg.sender, totalFee, block.timestamp);
    }

    // -------------------------------------------------------------------------
    // COORDINATOR: ATTACH CHECKPOINTS
    // -------------------------------------------------------------------------

    function attachCheckpoint(uint256 runId, bytes32 checkpointHash) external whenTrainingActive nonReentrant {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        if (r.coordinator != msg.sender) revert YangGo_NotCoordinator();
        if (r.finalized) revert YangGo_RunAlreadyFinalized();
        if (checkpointHash == bytes32(0)) revert YangGo_ZeroHash();
        if (r.checkpoints.length >= MAX_CHECKPOINTS_PER_RUN) revert YangGo_CheckpointIndexOutOfRange();

        r.checkpoints.push(checkpointHash);
        emit CheckpointAttached(runId, r.checkpoints.length - 1, checkpointHash, block.number);
    }

    function attachCheckpointBatch(uint256 runId, bytes32[] calldata checkpointHashes) external whenTrainingActive nonReentrant {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        if (r.coordinator != msg.sender) revert YangGo_NotCoordinator();
        if (r.finalized) revert YangGo_RunAlreadyFinalized();
        uint256 addCount = checkpointHashes.length;
        if (r.checkpoints.length + addCount > MAX_CHECKPOINTS_PER_RUN) revert YangGo_CheckpointIndexOutOfRange();

        for (uint256 i = 0; i < addCount; i++) {
            if (checkpointHashes[i] == bytes32(0)) revert YangGo_ZeroHash();
            r.checkpoints.push(checkpointHashes[i]);
            emit CheckpointAttached(runId, r.checkpoints.length - 1, checkpointHashes[i], block.number);
        }
    }

    // -------------------------------------------------------------------------
    // COORDINATOR: FINALIZE RUN
    // -------------------------------------------------------------------------

    function finalizeRun(uint256 runId) external whenTrainingActive nonReentrant {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        if (r.coordinator != msg.sender) revert YangGo_NotCoordinator();
        if (r.finalized) revert YangGo_RunAlreadyFinalized();

        r.finalized = true;
    }

    // -------------------------------------------------------------------------
    // VALIDATOR: REGISTER / UNREGISTER
    // -------------------------------------------------------------------------

    function registerValidator() external payable nonReentrant {
        if (msg.value < MIN_VALIDATOR_STAKE) revert YangGo_MinStakeRequired();
        ValidatorState storage vs = _validators[msg.sender];
        if (vs.active) revert YangGo_ValidatorNotRegistered();

        vs.stake = msg.value;
        vs.active = true;
        _validatorStakeSnapshot[msg.sender] = msg.value;
        _totalStaked += msg.value;
        _validatorList.push(msg.sender);
        emit ValidatorRegistered(msg.sender, msg.value, block.timestamp);
    }

    function unregisterValidator() external nonReentrant {
        ValidatorState storage vs = _validators[msg.sender];
        if (!vs.active) revert YangGo_ValidatorNotRegistered();

        vs.active = false;
        uint256 amount = vs.stake;
        vs.stake = 0;
        _totalStaked -= amount;
        _validatorStakeSnapshot[msg.sender] = 0;
        for (uint256 i = 0; i < _validatorList.length; i++) {
            if (_validatorList[i] == msg.sender) {
                _validatorList[i] = _validatorList[_validatorList.length - 1];
                _validatorList.pop();
                break;
            }
        }
        (bool sent,) = msg.sender.call{value: amount}("");
        if (!sent) revert YangGo_TransferFailed();
        emit ValidatorUnregistered(msg.sender, block.timestamp);
    }

    // -------------------------------------------------------------------------
    // VALIDATOR: ATTEST
    // -------------------------------------------------------------------------

    function attestRun(uint256 runId, bool approved) external whenTrainingActive nonReentrant {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        ValidatorState storage vs = _validators[msg.sender];
        if (!vs.active) revert YangGo_ValidatorNotRegistered();
        if (vs.attestedRuns[runId]) revert YangGo_AlreadyAttested();

        TrainingRun storage r = _runs[runId];
        if (!r.finalized) revert YangGo_RunNotFinalized();

        vs.attestedRuns[runId] = true;
        r.totalAttestations++;
        if (approved) r.positiveAttestations++;

        emit ValidatorAttestation(runId, msg.sender, approved, block.timestamp);
    }

    // -------------------------------------------------------------------------
    // VIEWS
    // -------------------------------------------------------------------------

    function getRun(uint256 runId) external view returns (
        bytes32 datasetHash,
        bytes32 configHash,
        uint8 modelTier,
        uint256 epochCount,
        address coordinator,
        uint256 registeredAt,
        bool finalized,
        uint256 positiveAttestations,
        uint256 totalAttestations,
        uint256 checkpointCount
    ) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        return (
            r.datasetHash,
            r.configHash,
            r.modelTier,
            r.epochCount,
            r.coordinator,
            r.registeredAt,
            r.finalized,
            r.positiveAttestations,
            r.totalAttestations,
            r.checkpoints.length
        );
    }

    function getCheckpoint(uint256 runId, uint256 index) external view returns (bytes32) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        if (index >= r.checkpoints.length) revert YangGo_CheckpointIndexOutOfRange();
        return r.checkpoints[index];
    }

    function runCount() external view returns (uint256) {
        return _runs.length;
    }

    function validatorCount() external view returns (uint256) {
        return _validatorList.length;
    }

    function getValidatorAt(uint256 index) external view returns (address) {
        if (index >= _validatorList.length) revert YangGo_CheckpointIndexOutOfRange();
        return _validatorList[index];
    }

    function hasAttested(uint256 runId, address validator) external view returns (bool) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _validators[validator].attestedRuns[runId];
    }

    function quorumReached(uint256 runId) external view returns (bool) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        if (r.totalAttestations == 0) return false;
        uint256 activeCount = _validatorList.length;
        if (activeCount == 0) return false;
        return (r.totalAttestations * BPS_DENOM) >= (activeCount * QUORUM_BPS);
    }

    function positiveQuorumReached(uint256 runId) external view returns (bool) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        uint256 activeCount = _validatorList.length;
        if (activeCount == 0) return false;
        if (r.totalAttestations == 0) return false;
        if ((r.totalAttestations * BPS_DENOM) < (activeCount * QUORUM_BPS)) return false;
        return (r.positiveAttestations * BPS_DENOM) >= (r.totalAttestations * QUORUM_BPS);
    }

    function getRunMeta(uint256 runId) external view returns (bytes32 tag, uint256 phaseLockUntil, bool finalizeRequested, uint256 requestedAt) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        RunMeta storage m = _runMeta[runId];
        return (m.tag, m.phaseLockUntil, m.finalizeRequested, m.requestedAt);
    }

    function getRunFull(uint256 runId) external view returns (
        bytes32 datasetHash,
        bytes32 configHash,
        uint8 modelTier,
        uint256 epochCount,
        address coordinator,
        uint256 registeredAt,
        bool finalized,
        uint256 positiveAttestations,
        uint256 totalAttestations,
        uint256 checkpointCount,
        bytes32 tag,
        bool finalizeRequested,
        uint256 requestedAt
    ) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        RunMeta storage m = _runMeta[runId];
        return (
            r.datasetHash,
            r.configHash,
            r.modelTier,
            r.epochCount,
            r.coordinator,
            r.registeredAt,
            r.finalized,
            r.positiveAttestations,
            r.totalAttestations,
            r.checkpoints.length,
            m.tag,
            m.finalizeRequested,
            m.requestedAt
        );
    }

    function getRunsByCoordinator(address coordinator) external view returns (uint256[] memory runIds) {
        runIds = _runIdsByCoordinator[coordinator];
    }

    function getRunIdsInRange(uint256 fromId, uint256 toId) external view returns (uint256[] memory runIds) {
        if (toId >= _runs.length) toId = _runs.length - 1;
        if (fromId > toId) return new uint256[](0);
        uint256 len = toId - fromId + 1;
        runIds = new uint256[](len);
        for (uint256 i = 0; i < len; i++) runIds[i] = fromId + i;
    }

    function getValidatorStake(address validator) external view returns (uint256) {
        return _validatorStakeSnapshot[validator];
    }

    function getTotalStaked() external view returns (uint256) {
        return _totalStaked;
    }

    function getGlobalStats() external view returns (
        uint256 totalRuns,
        uint256 totalValidators,
        uint256 totalStaked,
        uint256 pausedFlag
    ) {
        return (_runs.length, _validatorList.length, _totalStaked, trainingPaused ? 1 : 0);
    }

    function setRunTag(uint256 runId, bytes32 tag) external whenTrainingActive {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        if (r.coordinator != msg.sender) revert YangGo_NotCoordinator();
        if (r.finalized) revert YangGo_RunAlreadyFinalized();
        _runMeta[runId].tag = tag;
        emit RunTagSet(runId, tag, block.number);
    }

    function requestFinalize(uint256 runId) external whenTrainingActive {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        RunMeta storage m = _runMeta[runId];
        if (r.coordinator != msg.sender) revert YangGo_NotCoordinator();
        if (r.finalized) revert YangGo_RunAlreadyFinalized();
        if (m.finalizeRequested) revert YangGo_RunAlreadyFinalized();
        m.finalizeRequested = true;
        m.requestedAt = block.timestamp;
        m.phaseLockUntil = block.timestamp + finalizeDelaySeconds;
        emit FinalizeRequested(runId, m.phaseLockUntil, block.timestamp);
    }

    function executeFinalize(uint256 runId) external whenTrainingActive nonReentrant {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        RunMeta storage m = _runMeta[runId];
        if (r.coordinator != msg.sender) revert YangGo_NotCoordinator();
        if (r.finalized) revert YangGo_RunAlreadyFinalized();
        if (!m.finalizeRequested) revert YangGo_RunNotFinalized();
        if (block.timestamp < m.phaseLockUntil) revert YangGo_RunNotFinalized();
        r.finalized = true;
    }

    function addConfigPreset(bytes32 presetId, bytes32 configHash, string calldata label, uint8 modelTier, uint256 suggestedEpochs) external onlyGovernor {
        _configPresets[presetId] = ConfigPreset({
            configHash: configHash,
            label: label,
            modelTier: modelTier,
            suggestedEpochs: suggestedEpochs,
            active: true
        });
        _presetIds.push(presetId);
        emit PresetAdded(presetId, configHash, label, block.timestamp);
    }

    function disableConfigPreset(bytes32 presetId) external onlyGovernor {
        if (!_configPresets[presetId].active) revert YangGo_InvalidRunId();
        _configPresets[presetId].active = false;
        emit PresetDisabled(presetId, block.timestamp);
    }

    function getPreset(bytes32 presetId) external view returns (bytes32 configHash, string memory label, uint8 modelTier, uint256 suggestedEpochs, bool active) {
        ConfigPreset storage p = _configPresets[presetId];
        return (p.configHash, p.label, p.modelTier, p.suggestedEpochs, p.active);
    }

    function getPresetIds() external view returns (bytes32[] memory) {
        return _presetIds;
    }

    function presetCount() external view returns (uint256) {
        return _presetIds.length;
    }

    function attestRunBatch(uint256[] calldata runIds, bool[] calldata approved) external whenTrainingActive nonReentrant {
        if (runIds.length != approved.length) revert YangGo_ArrayLengthMismatch();
        ValidatorState storage vs = _validators[msg.sender];
        if (!vs.active) revert YangGo_ValidatorNotRegistered();
        for (uint256 i = 0; i < runIds.length; i++) {
            uint256 runId = runIds[i];
            if (runId >= _runs.length) revert YangGo_InvalidRunId();
            if (vs.attestedRuns[runId]) revert YangGo_AlreadyAttested();
            TrainingRun storage r = _runs[runId];
            if (!r.finalized) revert YangGo_RunNotFinalized();
            vs.attestedRuns[runId] = true;
            r.totalAttestations++;
            if (approved[i]) r.positiveAttestations++;
            emit ValidatorAttestation(runId, msg.sender, approved[i], block.timestamp);
        }
    }

    function getPhase(uint256 runId) external view returns (RunPhase) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        if (r.finalized) return RunPhase.Finalized;
        RunMeta storage m = _runMeta[runId];
        if (m.finalizeRequested && block.timestamp >= m.phaseLockUntil) return RunPhase.AttestationPhase;
        if (m.finalizeRequested) return RunPhase.AttestationPhase;
        if (r.checkpoints.length > 0) return RunPhase.CheckpointPhase;
        return RunPhase.Draft;
    }

    // -------------------------------------------------------------------------
    // GOVERNOR
    // -------------------------------------------------------------------------

    function setCoordinatorWhitelist(address account, bool allowed) external onlyGovernor {
        if (account == address(0)) revert YangGo_ZeroAddress();
        coordinatorWhitelist[account] = allowed;
        if (allowed) emit CoordinatorWhitelisted(account, msg.sender, block.timestamp);
        else emit CoordinatorRemoved(account, msg.sender, block.timestamp);
    }

    function setMaxEpochsPerRun(uint256 newMax) external onlyGovernor {
        if (newMax < MIN_EPOCHS) revert YangGo_InvalidEpochCount();
        uint256 prev = maxEpochsPerRun;
        maxEpochsPerRun = newMax;
        emit EpochLimitUpdated(prev, newMax, block.timestamp);
    }

    function setRegistrationFee(uint256 newFeeWei) external onlyGovernor {
        registrationFeeWei = newFeeWei;
    }

    function setTrainingPaused(bool paused_) external onlyGovernor {
        trainingPaused = paused_;
        if (paused_) emit TrainingPaused(block.timestamp, msg.sender);
        else emit TrainingResumed(block.timestamp, msg.sender);
    }

    function setFinalizeDelay(uint256 newDelaySeconds) external onlyGovernor {
        if (newDelaySeconds < MIN_FINALIZE_DELAY || newDelaySeconds > MAX_FINALIZE_DELAY) revert YangGo_InvalidEpochCount();
        finalizeDelaySeconds = newDelaySeconds;
    }

    function distributeReward(address recipient, uint256 amount) external onlyGovernor nonReentrant {
        if (recipient == address(0)) revert YangGo_ZeroAddress();
        (bool sent,) = recipient.call{value: amount}("");
        if (!sent) revert YangGo_TransferFailed();
        emit RewardDistributed(recipient, amount, block.timestamp);
    }

    // -------------------------------------------------------------------------
    // FALLBACK
    // -------------------------------------------------------------------------

    receive() external payable {
        emit FallbackDeposit(msg.sender, msg.value);
    }

    // -------------------------------------------------------------------------
    // ADDITIONAL VIEW HELPERS (batch and stats)
    // -------------------------------------------------------------------------

    function getRunBatch(uint256 fromId, uint256 toId) external view returns (
        uint256[] memory runIds,
        bytes32[] memory datasetHashes,
        bytes32[] memory configHashes,
        uint8[] memory modelTiers,
        uint256[] memory epochCounts,
        address[] memory coordinators,
        uint256[] memory registeredAts,
        bool[] memory finalizes,
        uint256[] memory positiveAttestations,
        uint256[] memory totalAttestations,
        uint256[] memory checkpointCounts
    ) {
        uint256 total = _runs.length;
        if (fromId >= total) {
            runIds = new uint256[](0);
            datasetHashes = new bytes32[](0);
            configHashes = new bytes32[](0);
            modelTiers = new uint8[](0);
            epochCounts = new uint256[](0);
            coordinators = new address[](0);
            registeredAts = new uint256[](0);
            finalizes = new bool[](0);
            positiveAttestations = new uint256[](0);
            totalAttestations = new uint256[](0);
            checkpointCounts = new uint256[](0);
            return (runIds, datasetHashes, configHashes, modelTiers, epochCounts, coordinators, registeredAts, finalizes, positiveAttestations, totalAttestations, checkpointCounts);
        }
        if (toId >= total) toId = total - 1;
        if (fromId > toId) {
            runIds = new uint256[](0);
            datasetHashes = new bytes32[](0);
            configHashes = new bytes32[](0);
            modelTiers = new uint8[](0);
            epochCounts = new uint256[](0);
            coordinators = new address[](0);
            registeredAts = new uint256[](0);
            finalizes = new bool[](0);
            positiveAttestations = new uint256[](0);
            totalAttestations = new uint256[](0);
            checkpointCounts = new uint256[](0);
            return (runIds, datasetHashes, configHashes, modelTiers, epochCounts, coordinators, registeredAts, finalizes, positiveAttestations, totalAttestations, checkpointCounts);
        }
        uint256 n = toId - fromId + 1;
        runIds = new uint256[](n);
        datasetHashes = new bytes32[](n);
        configHashes = new bytes32[](n);
        modelTiers = new uint8[](n);
        epochCounts = new uint256[](n);
        coordinators = new address[](n);
        registeredAts = new uint256[](n);
        finalizes = new bool[](n);
        positiveAttestations = new uint256[](n);
        totalAttestations = new uint256[](n);
        checkpointCounts = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            uint256 runId = fromId + i;
            TrainingRun storage r = _runs[runId];
            runIds[i] = runId;
            datasetHashes[i] = r.datasetHash;
            configHashes[i] = r.configHash;
            modelTiers[i] = r.modelTier;
            epochCounts[i] = r.epochCount;
            coordinators[i] = r.coordinator;
            registeredAts[i] = r.registeredAt;
            finalizes[i] = r.finalized;
            positiveAttestations[i] = r.positiveAttestations;
            totalAttestations[i] = r.totalAttestations;
            checkpointCounts[i] = r.checkpoints.length;
        }
    }

    function getCheckpointsBatch(uint256 runId, uint256 fromIndex, uint256 toIndex) external view returns (bytes32[] memory hashes) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        uint256 total = r.checkpoints.length;
        if (fromIndex >= total) return new bytes32[](0);
        if (toIndex >= total) toIndex = total - 1;
        if (fromIndex > toIndex) return new bytes32[](0);
        uint256 n = toIndex - fromIndex + 1;
        hashes = new bytes32[](n);
        for (uint256 i = 0; i < n; i++) hashes[i] = r.checkpoints[fromIndex + i];
    }

    function countRunsByModelTier(uint8 tier) external view returns (uint256 count) {
        for (uint256 i = 0; i < _runs.length; i++) {
            if (_runs[i].modelTier == tier) count++;
        }
    }

    function countFinalizedRuns() external view returns (uint256 count) {
        for (uint256 i = 0; i < _runs.length; i++) {
            if (_runs[i].finalized) count++;
        }
    }

    function countRunsWithPositiveQuorum() external view returns (uint256 count) {
        uint256 activeCount = _validatorList.length;
        for (uint256 i = 0; i < _runs.length; i++) {
            TrainingRun storage r = _runs[i];
            if (r.totalAttestations == 0) continue;
            if ((r.totalAttestations * BPS_DENOM) < (activeCount * QUORUM_BPS)) continue;
            if ((r.positiveAttestations * BPS_DENOM) >= (r.totalAttestations * QUORUM_BPS)) count++;
        }
    }

    function getLatestRunId() external view returns (uint256) {
        if (_runs.length == 0) revert YangGo_InvalidRunId();
        return _runs.length - 1;
    }

    function isCoordinatorWhitelisted(address account) external view returns (bool) {
        return coordinatorWhitelist[account];
    }

    function getConfigHash(uint256 runId) external view returns (bytes32) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _runs[runId].configHash;
    }

    function getDatasetHash(uint256 runId) external view returns (bytes32) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _runs[runId].datasetHash;
    }

    function getCoordinator(uint256 runId) external view returns (address) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _runs[runId].coordinator;
    }

    function isRunFinalized(uint256 runId) external view returns (bool) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _runs[runId].finalized;
    }

    function getAttestationStats(uint256 runId) external view returns (uint256 positive, uint256 total) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        return (r.positiveAttestations, r.totalAttestations);
    }

    function computeQuorumBps(uint256 runId) external view returns (uint256 attestationBps, uint256 positiveBps) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        TrainingRun storage r = _runs[runId];
        uint256 activeCount = _validatorList.length;
        if (activeCount == 0) return (0, 0);
        attestationBps = (r.totalAttestations * BPS_DENOM) / activeCount;
        if (r.totalAttestations == 0) positiveBps = 0;
        else positiveBps = (r.positiveAttestations * BPS_DENOM) / r.totalAttestations;
    }

    function getRunRegisteredAt(uint256 runId) external view returns (uint256) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _runs[runId].registeredAt;
    }

    function getRunEpochCount(uint256 runId) external view returns (uint256) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _runs[runId].epochCount;
    }

    function getRunModelTier(uint256 runId) external view returns (uint8) {
        if (runId >= _runs.length) revert YangGo_InvalidRunId();
        return _runs[runId].modelTier;
    }
}

// -----------------------------------------------------------------------------
// YangGo Gradient Snapshot Extension
// Stores per-run gradient norm attestations for verifiable training metrics.
// -----------------------------------------------------------------------------

contract YangGoGradientSnapshot {

    event SnapshotRecorded(uint256 indexed runId, uint256 epochIndex, bytes32 gradientNormHash, address indexed recorder, uint256 atBlock);
    event SnapshotBatchRecorded(uint256 indexed runId, uint256 startEpoch, uint256 count, address indexed recorder, uint256 atBlock);

    error YGS_NotCoordinator();
    error YGS_InvalidRunId();
    error YGS_RunFinalized();
    error YGS_ZeroHash();
    error YGS_EpochOutOfRange();
    error YGS_ArrayLengthMismatch();
    error YGS_Reentrancy();
    error YGS_Unauthorized();

    address public immutable yangGoCore;
    uint256 private _lock;

    struct EpochSnapshot {
        bytes32 gradientNormHash;
        uint256 recordedAt;
        address recorder;
    }

    mapping(uint256 => mapping(uint256 => EpochSnapshot)) private _snapshots;
    mapping(uint256 => uint256) private _snapshotCountByRun;

    constructor(address core_) {
        yangGoCore = core_;
    }

    modifier nonReentrant() {
        if (_lock != 0) revert YGS_Reentrancy();
        _lock = 1;
        _;
        _lock = 0;
    }

    function recordSnapshot(uint256 runId, uint256 epochIndex, bytes32 gradientNormHash) external nonReentrant {
        if (gradientNormHash == bytes32(0)) revert YGS_ZeroHash();
        (,,,, address coordinator,, bool finalized,,,,) = IYangGoView(yangGoCore).getRun(runId);
        if (msg.sender != coordinator) revert YGS_NotCoordinator();
        if (finalized) revert YGS_RunFinalized();

        _snapshots[runId][epochIndex] = EpochSnapshot({
            gradientNormHash: gradientNormHash,
            recordedAt: block.timestamp,
            recorder: msg.sender
        });
        uint256 count = _snapshotCountByRun[runId];
        if (epochIndex >= count) _snapshotCountByRun[runId] = epochIndex + 1;
        emit SnapshotRecorded(runId, epochIndex, gradientNormHash, msg.sender, block.number);
    }

    function recordSnapshotBatch(uint256 runId, uint256[] calldata epochIndices, bytes32[] calldata gradientNormHashes) external nonReentrant {
        if (epochIndices.length != gradientNormHashes.length) revert YGS_ArrayLengthMismatch();
        (,,,, address coordinator,, bool finalized,,,,) = IYangGoView(yangGoCore).getRun(runId);
        if (msg.sender != coordinator) revert YGS_NotCoordinator();
        if (finalized) revert YGS_RunFinalized();

        for (uint256 i = 0; i < epochIndices.length; i++) {
            if (gradientNormHashes[i] == bytes32(0)) revert YGS_ZeroHash();
            _snapshots[runId][epochIndices[i]] = EpochSnapshot({
                gradientNormHash: gradientNormHashes[i],
                recordedAt: block.timestamp,
                recorder: msg.sender
            });
            uint256 c = _snapshotCountByRun[runId];
            if (epochIndices[i] >= c) _snapshotCountByRun[runId] = epochIndices[i] + 1;
        }
        emit SnapshotBatchRecorded(runId, epochIndices.length > 0 ? epochIndices[0] : 0, epochIndices.length, msg.sender, block.number);
    }

    function getSnapshot(uint256 runId, uint256 epochIndex) external view returns (bytes32 gradientNormHash, uint256 recordedAt, address recorder) {
        EpochSnapshot storage s = _snapshots[runId][epochIndex];
        return (s.gradientNormHash, s.recordedAt, s.recorder);
    }

    function snapshotCountForRun(uint256 runId) external view returns (uint256) {
        return _snapshotCountByRun[runId];
    }
}

interface IYangGoView {
    function getRun(uint256 runId) external view returns (
        bytes32 datasetHash,
        bytes32 configHash,
        uint8 modelTier,
        uint256 epochCount,
        address coordinator,
        uint256 registeredAt,
        bool finalized,
        uint256 positiveAttestations,
        uint256 totalAttestations,
        uint256 checkpointCount
    );
    function runCount() external view returns (uint256);
    function validatorCount() external view returns (uint256);
    function getValidatorAt(uint256 index) external view returns (address);
    function getRunsByCoordinator(address coordinator) external view returns (uint256[] memory);
    function quorumReached(uint256 runId) external view returns (bool);
    function positiveQuorumReached(uint256 runId) external view returns (bool);
}

// -----------------------------------------------------------------------------
// YangGo Run Query - Off-chain and front-end helper for batch run inspection.
// -----------------------------------------------------------------------------

contract YangGoRunQuery {

    IYangGoView public immutable core;

    struct RunSummary {
        uint256 runId;
        bytes32 datasetHash;
        uint8 modelTier;
        uint256 epochCount;
        address coordinator;
        uint256 registeredAt;
        bool finalized;
        uint256 positiveAttestations;
        uint256 totalAttestations;
        uint256 checkpointCount;
    }

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function getRunSummaries(uint256 fromId, uint256 limit) external view returns (RunSummary[] memory out) {
        uint256 total = core.runCount();
        if (fromId >= total) return new RunSummary[](0);
        uint256 end = fromId + limit;
        if (end > total) end = total;
        uint256 n = end - fromId;
        out = new RunSummary[](n);
        for (uint256 i = 0; i < n; i++) {
            uint256 runId = fromId + i;
            (
                bytes32 datasetHash,
                bytes32 configHash,
                uint8 modelTier,
                uint256 epochCount,
                address coordinator,
                uint256 registeredAt,
                bool finalized,
                uint256 positiveAttestations,
                uint256 totalAttestations,
                uint256 checkpointCount
            ) = core.getRun(runId);
            out[i] = RunSummary({
                runId: runId,
                datasetHash: datasetHash,
                modelTier: modelTier,
                epochCount: epochCount,
                coordinator: coordinator,
                registeredAt: registeredAt,
                finalized: finalized,
                positiveAttestations: positiveAttestations,
                totalAttestations: totalAttestations,
                checkpointCount: checkpointCount
            });
        }
    }

    function getRunSummariesForCoordinator(address coordinator, uint256 maxResults) external view returns (RunSummary[] memory out) {
        uint256[] memory runIds = core.getRunsByCoordinator(coordinator);
        uint256 n = runIds.length;
        if (n > maxResults) n = maxResults;
        out = new RunSummary[](n);
        for (uint256 i = 0; i < n; i++) {
            uint256 runId = runIds[i];
            (
                bytes32 datasetHash,
                bytes32 configHash,
                uint8 modelTier,
                uint256 epochCount,
                address coord,
                uint256 registeredAt,
                bool finalized,
                uint256 positiveAttestations,
                uint256 totalAttestations,
                uint256 checkpointCount
            ) = core.getRun(runId);
            out[i] = RunSummary({
                runId: runId,
                datasetHash: datasetHash,
                modelTier: modelTier,
                epochCount: epochCount,
                coordinator: coord,
                registeredAt: registeredAt,
                finalized: finalized,
                positiveAttestations: positiveAttestations,
                totalAttestations: totalAttestations,
                checkpointCount: checkpointCount
            });
        }
    }

    function getRunsWithQuorum(uint256 fromId, uint256 limit) external view returns (uint256[] memory runIds, bool[] memory positiveQuorum) {
        uint256 total = core.runCount();
        if (fromId >= total) {
            runIds = new uint256[](0);
            positiveQuorum = new bool[](0);
            return (runIds, positiveQuorum);
        }
        uint256 end = fromId + limit;
        if (end > total) end = total;
        uint256 n = end - fromId;
        runIds = new uint256[](n);
        positiveQuorum = new bool[](n);
        for (uint256 i = 0; i < n; i++) {
            uint256 runId = fromId + i;
            runIds[i] = runId;
            positiveQuorum[i] = core.positiveQuorumReached(runId);
        }
    }

    function getAllValidatorAddresses() external view returns (address[] memory addrs) {
        uint256 count = core.validatorCount();
        addrs = new address[](count);
        for (uint256 i = 0; i < count; i++) addrs[i] = core.getValidatorAt(i);
    }
}

// -----------------------------------------------------------------------------
// YangGo Reward Calculator - Pure view logic for potential reward distribution.
// -----------------------------------------------------------------------------

contract YangGoRewardCalculator {

    uint256 public constant BPS_DENOM = 10000;
    uint256 public constant QUORUM_BPS = 6600;

    struct RunRewardInfo {
        uint256 runId;
        bool quorumReached;
        bool positiveQuorum;
        uint256 positiveAttestations;
        uint256 totalAttestations;
        uint256 validatorCount;
    }

    function computeRunRewardInfo(
        uint256 runId,
        uint256 positiveAttestations,
        uint256 totalAttestations,
        uint256 validatorCount
    ) external pure returns (RunRewardInfo memory info) {
        info.runId = runId;
        info.positiveAttestations = positiveAttestations;
        info.totalAttestations = totalAttestations;
        info.validatorCount = validatorCount;
        info.quorumReached = validatorCount > 0 && (totalAttestations * BPS_DENOM) >= (validatorCount * QUORUM_BPS);
        info.positiveQuorum = info.quorumReached && (positiveAttestations * BPS_DENOM) >= (totalAttestations * QUORUM_BPS);
    }

    function computeTierMultiplier(uint8 modelTier) external pure returns (uint256 multiplierBps) {
        if (modelTier == 1) return 10000;
        if (modelTier == 2) return 12000;
        if (modelTier == 3) return 15000;
        if (modelTier == 4) return 20000;
        return 10000;
    }

    function computeEpochScore(uint256 epochCount) external pure returns (uint256 score) {
        if (epochCount <= 10) return 1000;
        if (epochCount <= 100) return 2000;
        if (epochCount <= 500) return 5000;
        if (epochCount <= 1000) return 8000;
        return 10000;
    }
}

// -----------------------------------------------------------------------------
// YangGo Epoch Tracker - Optional per-run epoch completion flags (gas-heavy, use off-chain when possible).
// -----------------------------------------------------------------------------

contract YangGoEpochTracker {

    event EpochMarkedComplete(uint256 indexed runId, uint256 epochIndex, address indexed marker, uint256 atBlock);

    error YGET_NotCoordinator();
    error YGET_InvalidRunId();
    error YGET_RunFinalized();
    error YGET_EpochOutOfRange();
    error YGET_Reentrancy();

    IYangGoView public immutable core;
    uint256 private _lock;
    mapping(uint256 => mapping(uint256 => bool)) private _epochComplete;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    modifier nonReentrant() {
        if (_lock != 0) revert YGET_Reentrancy();
        _lock = 1;
        _;
        _lock = 0;
    }

    function markEpochComplete(uint256 runId, uint256 epochIndex) external nonReentrant {
        (,,,, address coordinator,, bool finalized,,, uint256 checkpointCount) = core.getRun(runId);
        if (msg.sender != coordinator) revert YGET_NotCoordinator();
        if (finalized) revert YGET_RunFinalized();
        if (epochIndex >= 10000) revert YGET_EpochOutOfRange();
        _epochComplete[runId][epochIndex] = true;
        emit EpochMarkedComplete(runId, epochIndex, msg.sender, block.number);
    }

    function isEpochComplete(uint256 runId, uint256 epochIndex) external view returns (bool) {
        return _epochComplete[runId][epochIndex];
    }

    function countCompleteEpochs(uint256 runId, uint256 maxEpoch) external view returns (uint256 count) {
        for (uint256 i = 0; i < maxEpoch; i++) {
            if (_epochComplete[runId][i]) count++;
        }
    }
}

// -----------------------------------------------------------------------------
// YangGo Dataset Registry - Optional registry of dataset hashes with metadata.
// -----------------------------------------------------------------------------

contract YangGoDatasetRegistry {

    event DatasetRegistered(bytes32 indexed datasetHash, string label, address indexed registrant, uint256 atBlock);
    event DatasetLabelUpdated(bytes32 indexed datasetHash, string newLabel, uint256 atBlock);

    error YGDR_ZeroHash();
    error YGDR_NotRegistrant();
    error YGDR_Reentrancy();

    struct DatasetMeta {
        string label;
        address registrant;
        uint256 registeredAt;
    }

    mapping(bytes32 => DatasetMeta) private _meta;
    bytes32[] private _datasetHashes;
    uint256 private _lock;

    modifier nonReentrant() {
        if (_lock != 0) revert YGDR_Reentrancy();
        _lock = 1;
        _;
        _lock = 0;
    }

    function registerDataset(bytes32 datasetHash, string calldata label) external nonReentrant {
        if (datasetHash == bytes32(0)) revert YGDR_ZeroHash();
        if (_meta[datasetHash].registrant != address(0)) revert YGDR_NotRegistrant();
        _meta[datasetHash] = DatasetMeta({ label: label, registrant: msg.sender, registeredAt: block.timestamp });
        _datasetHashes.push(datasetHash);
        emit DatasetRegistered(datasetHash, label, msg.sender, block.timestamp);
    }

    function updateDatasetLabel(bytes32 datasetHash, string calldata newLabel) external nonReentrant {
        if (_meta[datasetHash].registrant != msg.sender) revert YGDR_NotRegistrant();
        _meta[datasetHash].label = newLabel;
        emit DatasetLabelUpdated(datasetHash, newLabel, block.timestamp);
    }

    function getDatasetMeta(bytes32 datasetHash) external view returns (string memory label, address registrant, uint256 registeredAt) {
        DatasetMeta storage m = _meta[datasetHash];
        return (m.label, m.registrant, m.registeredAt);
    }

    function getAllDatasetHashes() external view returns (bytes32[] memory) {
        return _datasetHashes;
    }

    function datasetCount() external view returns (uint256) {
        return _datasetHashes.length;
    }
}

// -----------------------------------------------------------------------------
// YangGo Checkpoint Verifier - Optional stub for future checkpoint verification.
// -----------------------------------------------------------------------------

contract YangGoCheckpointVerifier {

    event CheckpointVerified(uint256 indexed runId, uint256 checkpointIndex, bool valid, uint256 atBlock);

    error YGCV_InvalidRunId();
    error YGCV_IndexOutOfRange();

    address public immutable core;

    constructor(address core_) {
        core = core_;
    }

    function verifyCheckpointExists(uint256 runId, uint256 checkpointIndex) external view returns (bool exists, bytes32 checkpointHash) {
        (,,,,, uint256 registeredAt, bool finalized, uint256 positiveAttestations, uint256 totalAttestations, uint256 checkpointCount) = IYangGoFull(core).getRun(runId);
        if (runId >= IYangGoFull(core).runCount()) revert YGCV_InvalidRunId();
        if (checkpointIndex >= checkpointCount) revert YGCV_IndexOutOfRange();
        checkpointHash = IYangGoFull(core).getCheckpoint(runId, checkpointIndex);
        exists = checkpointHash != bytes32(0);
    }
}

interface IYangGoFull is IYangGoView {
    function getCheckpoint(uint256 runId, uint256 index) external view returns (bytes32);
}

// -----------------------------------------------------------------------------
// YangGo Run Filter - Filter runs by tier, finalized status, time range.
// -----------------------------------------------------------------------------

contract YangGoRunFilter {

    IYangGoView public immutable core;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function filterByModelTier(uint8 tier, uint256 maxResults) external view returns (uint256[] memory runIds) {
        uint256 total = core.runCount();
        uint256[] memory temp = new uint256[](total);
        uint256 count = 0;
        for (uint256 i = 0; i < total && count < maxResults; i++) {
            (, , uint8 modelTier, , , , bool finalized, , , ) = core.getRun(i);
            if (modelTier == tier) {
                temp[count] = i;
                count++;
            }
        }
        runIds = new uint256[](count);
        for (uint256 j = 0; j < count; j++) runIds[j] = temp[j];
    }

    function filterFinalized(uint256 fromId, uint256 limit) external view returns (uint256[] memory runIds) {
        uint256 total = core.runCount();
        if (fromId >= total) return new uint256[](0);
        uint256[] memory temp = new uint256[](limit);
        uint256 count = 0;
        for (uint256 i = fromId; i < total && count < limit; i++) {
            (, , , , , , bool finalized, , , ) = core.getRun(i);
            if (finalized) {
                temp[count] = i;
                count++;
            }
        }
        runIds = new uint256[](count);
        for (uint256 j = 0; j < count; j++) runIds[j] = temp[j];
    }

    function filterByPositiveQuorum(uint256 maxResults) external view returns (uint256[] memory runIds) {
        uint256 total = core.runCount();
        uint256[] memory temp = new uint256[](maxResults);
        uint256 count = 0;
        for (uint256 i = 0; i < total && count < maxResults; i++) {
            if (core.positiveQuorumReached(i)) {
                temp[count] = i;
                count++;
            }
        }
        runIds = new uint256[](count);
        for (uint256 j = 0; j < count; j++) runIds[j] = temp[j];
    }
}

// -----------------------------------------------------------------------------
// YangGo Stats Aggregator - Aggregate stats across runs and validators.
// -----------------------------------------------------------------------------

contract YangGoStatsAggregator {

    IYangGoView public immutable core;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function totalRuns() external view returns (uint256) {
        return core.runCount();
    }

    function totalValidators() external view returns (uint256) {
        return core.validatorCount();
    }

    function getRunStatsSummary() external view returns (
        uint256 total,
        uint256 byTier1,
        uint256 byTier2,
        uint256 byTier3,
        uint256 byTier4
    ) {
        total = core.runCount();
        for (uint256 i = 0; i < total; i++) {
            (, , uint8 modelTier, , , , , , , ) = core.getRun(i);
            if (modelTier == 1) byTier1++;
            else if (modelTier == 2) byTier2++;
            else if (modelTier == 3) byTier3++;
            else if (modelTier == 4) byTier4++;
        }
    }

    function getValidatorListPaginated(uint256 offset, uint256 limit) external view returns (address[] memory addrs) {
        uint256 total = core.validatorCount();
        if (offset >= total) return new address[](0);
        uint256 end = offset + limit;
        if (end > total) end = total;
        uint256 n = end - offset;
        addrs = new address[](n);
        for (uint256 i = 0; i < n; i++) addrs[i] = core.getValidatorAt(offset + i);
    }
}

// -----------------------------------------------------------------------------
// YangGo Fee Tier Helper - Compute fee by model tier (view only).
// -----------------------------------------------------------------------------

contract YangGoFeeTierHelper {

    uint256 public constant BASE_FEE = 0.01 ether;
    uint256 public constant TIER_1_MUL = 10000;
    uint256 public constant TIER_2_MUL = 12000;
    uint256 public constant TIER_3_MUL = 15000;
    uint256 public constant TIER_4_MUL = 20000;
    uint256 public constant MUL_DENOM = 10000;

    function computeFeeForTier(uint8 modelTier) external pure returns (uint256 feeWei) {
        if (modelTier == 1) return (BASE_FEE * TIER_1_MUL) / MUL_DENOM;
        if (modelTier == 2) return (BASE_FEE * TIER_2_MUL) / MUL_DENOM;
        if (modelTier == 3) return (BASE_FEE * TIER_3_MUL) / MUL_DENOM;
        if (modelTier == 4) return (BASE_FEE * TIER_4_MUL) / MUL_DENOM;
        return BASE_FEE;
    }

    function computeTierMultiplier(uint8 modelTier) external pure returns (uint256 multiplierBps) {
        if (modelTier == 1) return TIER_1_MUL;
        if (modelTier == 2) return TIER_2_MUL;
        if (modelTier == 3) return TIER_3_MUL;
        if (modelTier == 4) return TIER_4_MUL;
        return TIER_1_MUL;
    }
}

// -----------------------------------------------------------------------------
// YangGo Epoch Bucket - Map epoch count to bucket index (view only).
// -----------------------------------------------------------------------------

contract YangGoEpochBucket {

    function getBucket(uint256 epochCount) external pure returns (uint8 bucket) {
        if (epochCount <= 10) return 1;
        if (epochCount <= 100) return 2;
        if (epochCount <= 500) return 3;
        if (epochCount <= 1000) return 4;
        return 5;
    }

    function getBucketLabel(uint8 bucket) external pure returns (string memory) {
        if (bucket == 1) return "small";
        if (bucket == 2) return "medium";
        if (bucket == 3) return "large";
        if (bucket == 4) return "xlarge";
        if (bucket == 5) return "mega";
        return "unknown";
    }
}

// -----------------------------------------------------------------------------
// YangGo Domain Separator - EIP-712 style domain for off-chain signing.
// -----------------------------------------------------------------------------

contract YangGoDomainSeparator {

    bytes32 public immutable DOMAIN_SEPARATOR;
    bytes32 public constant ATTESTATION_TYPEHASH = keccak256("Attest(uint256 runId,bool approved,uint256 nonce)");

    constructor(bytes32 domainTypeHash, bytes32 nameHash, bytes32 versionHash, uint256 chainId, address verifyingContract) {
        DOMAIN_SEPARATOR = keccak256(abi.encode(domainTypeHash, nameHash, versionHash, chainId, verifyingContract));
    }

    function getDomainSeparator() external view returns (bytes32) {
        return DOMAIN_SEPARATOR;
    }
}

// -----------------------------------------------------------------------------
// YangGo Run Nonce - Per-run nonce for replay protection (optional).
// -----------------------------------------------------------------------------

contract YangGoRunNonce {

    mapping(uint256 => uint256) private _nonce;

    function getNonce(uint256 runId) external view returns (uint256) {
        return _nonce[runId];
    }

    function incrementNonce(uint256 runId) external {
        _nonce[runId]++;
    }
}

// -----------------------------------------------------------------------------
// YangGo Hash Utils - Pure helpers for hashing (no state).
// -----------------------------------------------------------------------------

contract YangGoHashUtils {

    function hashDataset(bytes32 datasetRoot, bytes32 configHash) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(datasetRoot, configHash));
    }

    function hashRunMeta(bytes32 datasetHash, bytes32 configHash, uint8 modelTier, uint256 epochCount) external pure returns (bytes32) {
        return keccak256(abi.encode(datasetHash, configHash, modelTier, epochCount));
    }

    function hashCheckpointBatch(bytes32[] calldata checkpointHashes) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(checkpointHashes));
    }
}

// -----------------------------------------------------------------------------
// YangGo Run Index - Index runs by tag and by config hash (read-only after init).
// -----------------------------------------------------------------------------

contract YangGoRunIndex {

    IYangGoView public immutable core;
    mapping(bytes32 => uint256[]) private _runIdsByTag;
    mapping(bytes32 => uint256[]) private _runIdsByConfigHash;
    bytes32[] private _knownTags;
    bytes32[] private _knownConfigHashes;

    event TagIndexed(bytes32 indexed tag, uint256 runId, uint256 atBlock);
    event ConfigIndexed(bytes32 indexed configHash, uint256 runId, uint256 atBlock);

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function indexRunByTag(uint256 runId, bytes32 tag) external {
        (,,,,,, bool finalized,,,) = core.getRun(runId);
        if (_runIdsByTag[tag].length == 0) _knownTags.push(tag);
        _runIdsByTag[tag].push(runId);
        emit TagIndexed(tag, runId, block.number);
    }

    function indexRunByConfig(bytes32 configHash, uint256 runId) external {
        (,,,,,, bool finalized,,,) = core.getRun(runId);
        if (_runIdsByConfigHash[configHash].length == 0) _knownConfigHashes.push(configHash);
        _runIdsByConfigHash[configHash].push(runId);
        emit ConfigIndexed(configHash, runId, block.number);
    }

    function getRunIdsByTag(bytes32 tag) external view returns (uint256[] memory) {
        return _runIdsByTag[tag];
    }

    function getRunIdsByConfigHash(bytes32 configHash) external view returns (uint256[] memory) {
        return _runIdsByConfigHash[configHash];
    }

    function getKnownTags() external view returns (bytes32[] memory) {
        return _knownTags;
    }

    function getKnownConfigHashes() external view returns (bytes32[] memory) {
        return _knownConfigHashes;
    }

    function tagCount() external view returns (uint256) {
        return _knownTags.length;
    }

    function configHashCount() external view returns (uint256) {
        return _knownConfigHashes.length;
    }
}

// -----------------------------------------------------------------------------
// YangGo Time Locks - Optional time-lock for governor actions (stub).
// -----------------------------------------------------------------------------

contract YangGoTimeLocks {

    uint256 public constant MIN_DELAY = 1 days;
    uint256 public constant MAX_DELAY = 30 days;

    mapping(bytes32 => uint256) private _lockedUntil;
    mapping(bytes32 => bytes) private _lockedPayload;

    event ActionScheduled(bytes32 indexed actionId, uint256 executeAfter, uint256 atBlock);
    event ActionExecuted(bytes32 indexed actionId, uint256 atBlock);

    function schedule(bytes32 actionId, uint256 delaySeconds, bytes calldata payload) external {
        if (delaySeconds < MIN_DELAY || delaySeconds > MAX_DELAY) revert();
        uint256 executeAfter = block.timestamp + delaySeconds;
        _lockedUntil[actionId] = executeAfter;
        _lockedPayload[actionId] = payload;
        emit ActionScheduled(actionId, executeAfter, block.timestamp);
    }

    function canExecute(bytes32 actionId) external view returns (bool) {
        return block.timestamp >= _lockedUntil[actionId] && _lockedUntil[actionId] != 0;
    }

    function getExecuteAfter(bytes32 actionId) external view returns (uint256) {
        return _lockedUntil[actionId];
    }
}

// -----------------------------------------------------------------------------
// YangGo Validator Weights - Optional weighting for validators (stub).
// -----------------------------------------------------------------------------

contract YangGoValidatorWeights {

    IYangGoView public immutable core;
    mapping(address => uint256) private _weight;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function setWeight(address validator, uint256 weight) external {
        _weight[validator] = weight;
    }

    function getWeight(address validator) external view returns (uint256) {
        return _weight[validator];
    }

    function getTotalWeight() external view returns (uint256 total) {
        uint256 n = core.validatorCount();
        for (uint256 i = 0; i < n; i++) {
            address v = core.getValidatorAt(i);
            total += _weight[v];
        }
    }
}

// -----------------------------------------------------------------------------
// YangGo Run Metadata Extension - Optional string metadata per run (off-chain pointer).
// -----------------------------------------------------------------------------

contract YangGoRunMetadata {

    event MetadataSet(uint256 indexed runId, string uri, uint256 atBlock);

    IYangGoView public immutable core;
    mapping(uint256 => string) private _metadataUri;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function setMetadata(uint256 runId, string calldata uri) external {
        (,,,, address coordinator,, bool finalized,,,) = core.getRun(runId);
        if (msg.sender != coordinator) revert();
        if (finalized) revert();
        _metadataUri[runId] = uri;
        emit MetadataSet(runId, uri, block.timestamp);
    }

    function getMetadata(uint256 runId) external view returns (string memory) {
        return _metadataUri[runId];
    }
}

// -----------------------------------------------------------------------------
// YangGo Slashing Stub - Placeholder for future slashing logic.
// -----------------------------------------------------------------------------

contract YangGoSlashingStub {

    event SlashRequested(address indexed validator, uint256 amount, bytes32 reason, uint256 atBlock);

    function requestSlash(address validator, uint256 amount, bytes32 reason) external {
        emit SlashRequested(validator, amount, reason, block.timestamp);
    }
}

// -----------------------------------------------------------------------------
// YangGo Treasury Stub - Optional treasury for protocol fees.
// -----------------------------------------------------------------------------

contract YangGoTreasuryStub {

    address public immutable governor;
    uint256 private _balance;

    event Deposited(uint256 amount, address indexed from, uint256 atBlock);
    event Withdrawn(uint256 amount, address indexed to, uint256 atBlock);

    constructor(address governor_) {
        governor = governor_;
    }

    receive() external payable {
        _balance += msg.value;
        emit Deposited(msg.value, msg.sender, block.timestamp);
    }

    function withdraw(address to, uint256 amount) external {
        if (msg.sender != governor) revert();
        if (amount > _balance) revert();
        _balance -= amount;
        (bool ok,) = to.call{value: amount}("");
        if (!ok) revert();
        emit Withdrawn(amount, to, block.timestamp);
    }

    function balance() external view returns (uint256) {
        return _balance;
    }
}

// -----------------------------------------------------------------------------
// YangGo Run Paginator - Paginate runs with optional filters.
// -----------------------------------------------------------------------------

contract YangGoRunPaginator {

    IYangGoView public immutable core;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function getPage(uint256 pageSize, uint256 pageIndex) external view returns (uint256[] memory runIds) {
        uint256 total = core.runCount();
        if (pageSize == 0 || pageIndex * pageSize >= total) return new uint256[](0);
        uint256 start = pageIndex * pageSize;
        uint256 end = start + pageSize;
        if (end > total) end = total;
        uint256 n = end - start;
        runIds = new uint256[](n);
        for (uint256 i = 0; i < n; i++) runIds[i] = start + i;
    }

    function getPageCount(uint256 pageSize) external view returns (uint256) {
        uint256 total = core.runCount();
        if (pageSize == 0) return 0;
        return (total + pageSize - 1) / pageSize;
    }

    function getRunsForPage(uint256 pageSize, uint256 pageIndex) external view returns (
        uint256[] memory runIds,
        address[] memory coordinators,
        bool[] memory finalizes,
        uint256[] memory totalAttestations
    ) {
        uint256 total = core.runCount();
        if (pageSize == 0 || pageIndex * pageSize >= total) {
            runIds = new uint256[](0);
            coordinators = new address[](0);
            finalizes = new bool[](0);
            totalAttestations = new uint256[](0);
            return (runIds, coordinators, finalizes, totalAttestations);
        }
        uint256 start = pageIndex * pageSize;
        uint256 end = start + pageSize;
        if (end > total) end = total;
        uint256 n = end - start;
        runIds = new uint256[](n);
        coordinators = new address[](n);
        finalizes = new bool[](n);
        totalAttestations = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            uint256 runId = start + i;
            (, , , , address coord, , bool fin, , uint256 totAtt, ) = core.getRun(runId);
            runIds[i] = runId;
            coordinators[i] = coord;
            finalizes[i] = fin;
            totalAttestations[i] = totAtt;
        }
    }
}

// -----------------------------------------------------------------------------
// YangGo Quorum Calculator - Pure view quorum math.
// -----------------------------------------------------------------------------

contract YangGoQuorumCalculator {

    uint256 public constant BPS_DENOM = 10000;
    uint256 public constant QUORUM_BPS = 6600;

    function quorumReached(uint256 totalAttestations, uint256 validatorCount) external pure returns (bool) {
        if (validatorCount == 0) return false;
        return (totalAttestations * BPS_DENOM) >= (validatorCount * QUORUM_BPS);
    }

    function positiveQuorumReached(uint256 positiveAttestations, uint256 totalAttestations) external pure returns (bool) {
        if (totalAttestations == 0) return false;
        return (positiveAttestations * BPS_DENOM) >= (totalAttestations * QUORUM_BPS);
    }

    function attestationsNeededForQuorum(uint256 validatorCount) external pure returns (uint256) {
        return (validatorCount * QUORUM_BPS + BPS_DENOM - 1) / BPS_DENOM;
    }

    function positiveVotesNeededForPositiveQuorum(uint256 totalAttestations) external pure returns (uint256) {
        return (totalAttestations * QUORUM_BPS + BPS_DENOM - 1) / BPS_DENOM;
    }
}

// -----------------------------------------------------------------------------
// YangGo Model Tier Labels - Human-readable tier names.
// -----------------------------------------------------------------------------

contract YangGoModelTierLabels {

    function getTierName(uint8 tier) external pure returns (string memory) {
        if (tier == 1) return "base";
        if (tier == 2) return "mid";
        if (tier == 3) return "large";
        if (tier == 4) return "xl";
        return "unknown";
    }

    function getTierIndex(string calldata name) external pure returns (uint8) {
        if (keccak256(bytes(name)) == keccak256("base")) return 1;
        if (keccak256(bytes(name)) == keccak256("mid")) return 2;
        if (keccak256(bytes(name)) == keccak256("large")) return 3;
        if (keccak256(bytes(name)) == keccak256("xl")) return 4;
        return 0;
    }
}

// -----------------------------------------------------------------------------
// YangGo Run Age - Compute run age in seconds.
// -----------------------------------------------------------------------------

contract YangGoRunAge {

    IYangGoView public immutable core;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function getRunAge(uint256 runId, uint256 asOfTimestamp) external view returns (uint256 ageSeconds) {
        (, , , , , uint256 registeredAt, , , , ) = core.getRun(runId);
        if (asOfTimestamp <= registeredAt) return 0;
        return asOfTimestamp - registeredAt;
    }

    function getRunAgeNow(uint256 runId) external view returns (uint256 ageSeconds) {
        (, , , , , uint256 registeredAt, , , , ) = core.getRun(runId);
        if (block.timestamp <= registeredAt) return 0;
        return block.timestamp - registeredAt;
    }
}

// -----------------------------------------------------------------------------
// YangGo Run Sorter Stub - Off-chain sort keys; on-chain returns run IDs in range.
// -----------------------------------------------------------------------------

contract YangGoRunSorter {

    IYangGoView public immutable core;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function getRunIdsInRange(uint256 fromId, uint256 toId) external view returns (uint256[] memory runIds) {
        uint256 total = core.runCount();
        if (fromId >= total) return new uint256[](0);
        if (toId >= total) toId = total - 1;
        if (fromId > toId) return new uint256[](0);
        uint256 n = toId - fromId + 1;
        runIds = new uint256[](n);
        for (uint256 i = 0; i < n; i++) runIds[i] = fromId + i;
    }

    function getRunIdsForCoordinatorPaginated(address coordinator, uint256 offset, uint256 limit) external view returns (uint256[] memory runIds) {
        uint256[] memory all = core.getRunsByCoordinator(coordinator);
        if (offset >= all.length) return new uint256[](0);
        uint256 end = offset + limit;
        if (end > all.length) end = all.length;
        uint256 n = end - offset;
        runIds = new uint256[](n);
        for (uint256 i = 0; i < n; i++) runIds[i] = all[offset + i];
    }
}

// -----------------------------------------------------------------------------
// YangGo Config Hash Registry - Optional registry of config hashes with labels.
// -----------------------------------------------------------------------------

contract YangGoConfigHashRegistry {

    event ConfigRegistered(bytes32 indexed configHash, string label, address indexed registrant, uint256 atBlock);

    struct ConfigMeta {
        string label;
        address registrant;
        uint256 registeredAt;
    }

    mapping(bytes32 => ConfigMeta) private _meta;
    bytes32[] private _configHashes;

    function registerConfig(bytes32 configHash, string calldata label) external {
        if (_meta[configHash].registrant != address(0)) revert();
        _meta[configHash] = ConfigMeta({ label: label, registrant: msg.sender, registeredAt: block.timestamp });
        _configHashes.push(configHash);
        emit ConfigRegistered(configHash, label, msg.sender, block.timestamp);
    }

    function getConfigMeta(bytes32 configHash) external view returns (string memory label, address registrant, uint256 registeredAt) {
        ConfigMeta storage m = _meta[configHash];
        return (m.label, m.registrant, m.registeredAt);
    }

    function getAllConfigHashes() external view returns (bytes32[] memory) {
        return _configHashes;
    }

    function configCount() external view returns (uint256) {
        return _configHashes.length;
    }
}

// -----------------------------------------------------------------------------
// YangGo Version Info - Static version and domain info.
// -----------------------------------------------------------------------------

contract YangGoVersionInfo {

    uint256 public constant VERSION = 2;
    bytes32 public constant DOMAIN = keccak256("YangGo.TrainingRun.v2");
    string public constant LABEL = "YangGo AI Training Registry v2";

    function getVersion() external pure returns (uint256) {
        return VERSION;
    }

    function getDomain() external pure returns (bytes32) {
        return DOMAIN;
    }

    function getLabel() external pure returns (string memory) {
        return LABEL;
    }
}

// -----------------------------------------------------------------------------
// YangGo Run Existence Checker - Lightweight run existence and bounds.
// -----------------------------------------------------------------------------

contract YangGoRunExistenceChecker {

    IYangGoView public immutable core;

    constructor(address core_) {
        core = IYangGoView(core_);
    }

    function exists(uint256 runId) external view returns (bool) {
        return runId < core.runCount();
    }

    function totalRuns() external view returns (uint256) {
        return core.runCount();
    }

    function isValidRunId(uint256 runId) external view returns (bool) {
        return runId < core.runCount();
    }

    function getBounds() external view returns (uint256 minRunId, uint256 maxRunId, uint256 count) {
        count = core.runCount();
        minRunId = 0;
        maxRunId = count == 0 ? 0 : count - 1;
    }

    function getFirstRunId() external view returns (uint256) {
        return 0;
    }

    function getLastRunId() external view returns (uint256) {
        uint256 c = core.runCount();
        return c == 0 ? 0 : c - 1;
    }

    function runCountView() external view returns (uint256) {
        return core.runCount();
    }

    /// @dev Returns whether any runs exist
    function hasRuns() external view returns (bool) {
        return core.runCount() > 0;
    }
}
