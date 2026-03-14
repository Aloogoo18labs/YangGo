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

