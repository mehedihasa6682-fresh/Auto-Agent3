package com.example.agent

import android.content.Context
import android.util.Log
import com.example.ai.ApiKeyFailoverManager
import com.example.ai.GeminiAiPlanner
import com.example.game.BoardPieceCounts
import com.example.game.CarromGeometrySolver
import com.example.game.CarromMemoryData
import com.example.game.CarromPowerStyle
import com.example.game.CarromShotMemoryStore
import com.example.game.CarromTurnMemory
import com.example.game.CarromVisionDetector
import com.example.game.CarromVisionOutcome
import com.example.game.CoinColor
import com.example.game.PendingCarromShotRecord
import com.example.service.AgentKeepAliveService
import com.example.service.LiveMediaProjectionCapturer
import com.example.service.TapPilotAccessibilityService
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.lang.ref.WeakReference

private const val TAG = "AgentCoordinator"
private const val MAX_TRACE_ENTRIES = 80
private const val MAX_NOTIF_BUFFER = 10

object AgentCoordinator {

    private data class PendingSliderVerify(
        val targetStrikerX: Float,
        val preStrikerX: Float,
        val commandedDeltaX: Float,
        val correctionAttempts: Int
    )

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private var appContext: Context? = null
    private var settingsStore: AgentSettingsStore? = null
    private var knowledgeStore: AgentKnowledgeStore? = null
    private var actionCacheStore: ActionCacheStore? = null
    private var contactProfileStore: ContactProfileStore? = null
    private var carromShotMemoryStore: CarromShotMemoryStore? = null
    private var settingsCollectJob: Job? = null
    private var playbooksCollectJob: Job? = null
    private var cacheCollectJob: Job? = null
    private var contactsCollectJob: Job? = null
    private var carromMemoryCollectJob: Job? = null
    private val aiPlanner = GeminiAiPlanner()

    private var serviceRef: WeakReference<TapPilotAccessibilityService>? = null
    private var activeRunJob: Job? = null
    private var pauseGate: CompletableDeferred<Unit>? = null
    private var confirmationDeferred: CompletableDeferred<Boolean>? = null
    private var notificationWakeSignal: CompletableDeferred<Unit>? = null

    private val recentNotifications = ArrayDeque<NotificationSignal>()

    private val _isServiceConnected = MutableStateFlow(false)
    val isServiceConnected: StateFlow<Boolean> = _isServiceConnected.asStateFlow()

    private val _settings = MutableStateFlow(AgentSettings())
    val settings: StateFlow<AgentSettings> = _settings.asStateFlow()

    private val _playbooks = MutableStateFlow<List<TaskPlaybook>>(emptyList())
    val playbooks: StateFlow<List<TaskPlaybook>> = _playbooks.asStateFlow()

    private val _cachedActions = MutableStateFlow<List<CachedActionEntry>>(emptyList())
    val cachedActions: StateFlow<List<CachedActionEntry>> = _cachedActions.asStateFlow()

    private val _contacts = MutableStateFlow<List<ContactProfile>>(emptyList())
    val contacts: StateFlow<List<ContactProfile>> = _contacts.asStateFlow()

    private val _carromMemory = MutableStateFlow(CarromMemoryData())
    val carromMemory: StateFlow<CarromMemoryData> = _carromMemory.asStateFlow()

    private val _uiState = MutableStateFlow<AgentUiState>(AgentUiState.Idle)
    val uiState: StateFlow<AgentUiState> = _uiState.asStateFlow()

    fun initialize(context: Context) {
        if (settingsStore != null) return
        val app = context.applicationContext
        appContext = app
        val store = AgentSettingsStore(app)
        settingsStore = store
        settingsCollectJob = scope.launch {
            store.settingsFlow.collect { latest ->
                _settings.value = latest
            }
        }
        val kStore = AgentKnowledgeStore(app)
        knowledgeStore = kStore
        playbooksCollectJob = scope.launch {
            kStore.playbooks.collect { list ->
                _playbooks.value = list
            }
        }
        val cStore = ActionCacheStore(app)
        actionCacheStore = cStore
        cacheCollectJob = scope.launch {
            cStore.cachedActions.collect { list ->
                _cachedActions.value = list
            }
        }
        val contactStore = ContactProfileStore(app)
        contactProfileStore = contactStore
        contactsCollectJob = scope.launch {
            contactStore.contacts.collect { list ->
                _contacts.value = list
            }
        }
        val carromStore = CarromShotMemoryStore(app)
        carromShotMemoryStore = carromStore
        carromMemoryCollectJob = scope.launch {
            carromStore.memory.collect { mem ->
                _carromMemory.value = mem
            }
        }
        scope.launch {
            kStore.loadInitial()
            cStore.loadInitial()
            contactStore.loadInitial()
            carromStore.loadInitial()
        }
    }

    fun attachService(service: TapPilotAccessibilityService) {
        serviceRef = WeakReference(service)
        _isServiceConnected.value = true
        Log.d(TAG, "AccessibilityService attached.")
    }

    fun detachService(service: TapPilotAccessibilityService) {
        if (serviceRef?.get() === service) {
            serviceRef = null
            _isServiceConnected.value = false
            stopTask("Accessibility Service disconnected.")
        }
    }

    fun onNotificationReceived(signal: NotificationSignal) {
        synchronized(recentNotifications) {
            if (recentNotifications.size >= MAX_NOTIF_BUFFER) {
                recentNotifications.removeFirst()
            }
            recentNotifications.addLast(signal)
        }
        // Wake up background monitor sleep early so it can inspect new message/comment!
        notificationWakeSignal?.complete(Unit)
    }

    fun updateSettings(transform: (AgentSettings) -> AgentSettings) {
        val store = settingsStore ?: return
        scope.launch {
            store.updateSettings(transform)
        }
    }

    fun deletePlaybook(topicKey: String) {
        val store = knowledgeStore ?: return
        scope.launch {
            store.deletePlaybook(topicKey)
        }
    }

    fun deleteCachedAction(cacheKey: String) {
        val store = actionCacheStore ?: return
        scope.launch {
            store.removeEntry(cacheKey)
        }
    }

    fun clearActionCache() {
        val store = actionCacheStore ?: return
        scope.launch {
            store.clearAll()
        }
    }

    fun upsertContact(profile: ContactProfile) {
        val store = contactProfileStore ?: return
        scope.launch {
            store.upsertContact(profile)
        }
    }

    fun deleteContact(id: String) {
        val store = contactProfileStore ?: return
        scope.launch {
            store.deleteContact(id)
        }
    }

    fun resetKeyFailoverHealth() {
        ApiKeyFailoverManager.resetHealth()
    }

    fun startTask(
        goalInput: String,
        mode: AgentExecutionMode = AgentExecutionMode.SINGLE_SESSION
    ) {
        val goal = goalInput.trim()
        if (goal.isEmpty()) {
            _uiState.value = AgentUiState.Error(
                goal = "",
                message = "Please enter an automation goal first.",
                workingMemory = "",
                trace = emptyList()
            )
            return
        }

        val service = serviceRef?.get()
        if (service == null) {
            _uiState.value = AgentUiState.Error(
                goal = goal,
                message = "Accessibility Service is not enabled. Enable TapPilot in Android Accessibility Settings first.",
                workingMemory = "",
                trace = emptyList()
            )
            return
        }

        val initialSettings = _settings.value
        val firebaseReady = aiPlanner.isFirebaseConfigured()
        if (!initialSettings.hasAnyRunnableKey(firebaseReady)) {
            _uiState.value = AgentUiState.Error(
                goal = goal,
                message = "No active API key configured. Open Settings and paste 1–30 free keys for Google AI Studio, Groq Cloud, or OpenRouter.",
                workingMemory = "",
                trace = emptyList()
            )
            return
        }

        activeRunJob?.cancel()
        pauseGate = null
        confirmationDeferred = null
        notificationWakeSignal = null
        synchronized(recentNotifications) {
            recentNotifications.clear()
        }

        val trace = mutableListOf<AgentStepLog>()
        var workingMemory = ""
        val maxStepsLimit: Int? = if (mode == AgentExecutionMode.SINGLE_SESSION) {
            initialSettings.maxSteps
        } else {
            null
        }

        if (mode == AgentExecutionMode.CONTINUOUS_MONITOR) {
            appContext?.let { AgentKeepAliveService.start(it) }
        }

        activeRunJob = scope.launch {
            try {
                var activePlaybook = knowledgeStore?.findMatchingPlaybook(goal)
                var lastOutcomeHint = ""

                if (activePlaybook == null && initialSettings.autoResearchBeforeTask) {
                    _uiState.value = AgentUiState.Running(
                        goal = goal,
                        mode = mode,
                        currentStep = 0,
                        maxSteps = maxStepsLimit,
                        phase = AgentPhase.RESEARCHING,
                        lastThought = "Unfamiliar task — researching rules, controls, winning shots & fouls first...",
                        workingMemory = workingMemory,
                        trace = trace.toList()
                    )
                    val researched = withContext(Dispatchers.IO) {
                        aiPlanner.researchTaskPlaybook(goal, initialSettings)
                    }.getOrNull()

                    if (researched != null) {
                        activePlaybook = knowledgeStore?.saveOrMergePlaybook(researched) ?: researched
                        workingMemory = "Researched [${activePlaybook.topicTitle}]: ${activePlaybook.rulesSummary.take(160)}"
                        appendTrace(
                            trace,
                            AgentStepLog(
                                stepNumber = 0,
                                timestampMs = System.currentTimeMillis(),
                                actionType = AgentActionType.WAIT,
                                targetSummary = "Playbook: ${activePlaybook.topicTitle}",
                                reason = "Researched ${activePlaybook.winningStrategies.size} strategies & ${activePlaybook.foulLessons.size} foul guards.",
                                status = StepStatus.COMPLETED,
                                requiresConfirmation = false
                            )
                        )
                    }
                } else if (activePlaybook != null) {
                    knowledgeStore?.incrementTrainingSession(activePlaybook.topicKey)
                    activePlaybook = knowledgeStore?.findMatchingPlaybook(goal) ?: activePlaybook
                    workingMemory = "Loaded trained playbook [${activePlaybook.topicTitle}] (${activePlaybook.winningStrategies.size} tactics, ${activePlaybook.foulLessons.size} foul guards)."
                }

                if (initialSettings.goHomeOnLaunch) {
                    _uiState.value = AgentUiState.Running(
                        goal = goal,
                        mode = mode,
                        currentStep = 0,
                        maxSteps = maxStepsLimit,
                        phase = AgentPhase.ACTING,
                        lastThought = "Navigating to Home screen...",
                        workingMemory = workingMemory,
                        trace = trace.toList()
                    )
                    service.navigateHome()
                    delay(900)
                }

                var step = 1
                var pendingCacheCandidate: PendingCacheCandidate? = null
                var lastReplayEntry: Pair<String, String>? = null
                var carromOfflineWaits = 0
                var didSlideCarromThisStep = false
                var movedStrikerThisTurn = false
                var pendingSliderVerify: PendingSliderVerify? = null
                var pendingCarromShot: PendingCarromShotRecord? = null
                val carromTurnMemory = CarromTurnMemory()
                val awaitingRepliesByPeer = mutableMapOf<String, ChatTurnGuard.AwaitingReplyState>()
                var pendingSentTextForPeer: Pair<String, String>? = null
                var prevCompositeScreen: String? = null
                var prevNonWaitActionType: AgentActionType? = null
                var consecutiveNoChangeCount = 0
                CarromVisionDetector.resetMotionHistory()

                while (maxStepsLimit == null || step <= maxStepsLimit) {
                    val currentSettings = _settings.value
                    awaitIfPaused(goal, mode, step, maxStepsLimit, workingMemory, trace)

                    // Wait cleanly if device screen is currently locked during continuous monitoring
                    while (service.isDeviceLocked()) {
                        _uiState.value = AgentUiState.Running(
                            goal = goal,
                            mode = mode,
                            currentStep = step,
                            maxSteps = maxStepsLimit,
                            phase = AgentPhase.MONITOR_SLEEPING,
                            lastThought = "Screen locked — waiting for device unlock...",
                            workingMemory = workingMemory,
                            trace = trace.toList()
                        )
                        delay(4_000L)
                        awaitIfPaused(goal, mode, step, maxStepsLimit, workingMemory, trace)
                    }

                    _uiState.value = AgentUiState.Running(
                        goal = goal,
                        mode = mode,
                        currentStep = step,
                        maxSteps = maxStepsLimit,
                        phase = AgentPhase.OBSERVING,
                        lastThought = trace.lastOrNull()?.reason.orEmpty(),
                        workingMemory = workingMemory,
                        trace = trace.toList()
                    )

                    val snapshot = service.captureSnapshot(currentSettings.useVisionScreenshot)
                    val notificationsSnapshot = synchronized(recentNotifications) {
                        recentNotifications.toList()
                    }

                    val currentSig = ActionCacheStore.computeScreenSignature(snapshot.nodes)
                    val currentCompositeScreen = "${snapshot.packageName}::$currentSig"
                    var justEvictedStaleCache = false

                    val prevReplay = lastReplayEntry
                    if (prevReplay != null) {
                        if (prevReplay.second == currentCompositeScreen) {
                            // Cached tap did not advance the screen — evict stale entry and use live AI
                            actionCacheStore?.removeEntry(prevReplay.first)
                            justEvictedStaleCache = true
                        } else {
                            actionCacheStore?.incrementHit(prevReplay.first)
                        }
                        lastReplayEntry = null
                    } else {
                        val prevCandidate = pendingCacheCandidate
                        if (prevCandidate != null) {
                            actionCacheStore?.recordVerifiedSuccess(prevCandidate, snapshot)
                            pendingCacheCandidate = null
                        }
                    }

                    val interactiveNodeCount = snapshot.nodes.count { it.isClickable || it.isEditable }
                    val isCarromContext = CarromContextHelper.isCarromGoalOrApp(
                        goal = goal,
                        packageName = snapshot.packageName,
                        playbookTopicKey = activePlaybook?.topicKey
                    )
                    val isLikelyCarromLobby = isCarromContext && CarromContextHelper.isLikelyGameLobbyMenu(snapshot.nodes)
                    val isActiveCarromMatchScreen = isCarromContext && !isLikelyCarromLobby

                    val cacheMatch = if (
                        currentSettings.useZeroApiActionCache &&
                        !justEvictedStaleCache &&
                        !isActiveCarromMatchScreen
                    ) {
                        actionCacheStore?.findReplayMatch(goal, snapshot)
                    } else {
                        null
                    }
                    var usedLocalCarromSolver = false

                    val chatObs = ChatTurnGuard.extractChatObservation(snapshot)
                    val currentContacts = _contacts.value
                    val matchedContact = ContactProfileStore.matchContactFromList(
                        contacts = currentContacts,
                        peerName = chatObs.peerName,
                        packageName = snapshot.packageName,
                        goal = goal,
                        recentNotifications = notificationsSnapshot
                    )
                    val savedContactsSummary = ContactProfileStore.formatRosterSummary(currentContacts)

                    val activePeerKey = if (chatObs.isMessagingApp) {
                        ChatTurnGuard.peerKey(snapshot.packageName, chatObs.peerName)
                    } else {
                        ""
                    }

                    // Check if any awaited chat peer replied (via notification or live screen bubble)
                    val repliedKeys = awaitingRepliesByPeer.entries
                        .filter { (_, awaiting) ->
                            ChatTurnGuard.didIncomingReplyArrive(chatObs, notificationsSnapshot, awaiting)
                        }
                        .map { it.key to it.value }
                    for ((repliedKey, repliedState) in repliedKeys) {
                        awaitingRepliesByPeer.remove(repliedKey)
                        val who = repliedState.peerName.ifBlank { "contact" }
                        lastOutcomeHint = "[NEW REPLY DETECTED from $who! Read their new message and respond naturally.] $lastOutcomeHint"
                    }

                    var activeAwaitingReply = if (activePeerKey.isNotBlank()) {
                        awaitingRepliesByPeer[activePeerKey]
                            ?: if (chatObs.peerName.isNotBlank()) {
                                awaitingRepliesByPeer.values.firstOrNull {
                                    it.peerName.equals(chatObs.peerName, ignoreCase = true)
                                }
                            } else {
                                null
                            }
                    } else {
                        null
                    }

                    // Latch the updated transcript signature once our own just-sent bubble renders on screen
                    if (activeAwaitingReply != null &&
                        pendingSentTextForPeer != null &&
                        pendingSentTextForPeer.first == activePeerKey &&
                        chatObs.inChatThread
                    ) {
                        val latched = activeAwaitingReply.copy(transcriptSignatureAtSend = chatObs.transcriptSignature)
                        awaitingRepliesByPeer[activePeerKey] = latched
                        activeAwaitingReply = latched
                        pendingSentTextForPeer = null
                    }

                    // Track consecutive actions that left the UI tree completely unchanged
                    if (!isCarromContext &&
                        prevCompositeScreen != null &&
                        prevNonWaitActionType != null &&
                        prevNonWaitActionType != AgentActionType.WAIT
                    ) {
                        if (prevCompositeScreen == currentCompositeScreen) {
                            consecutiveNoChangeCount++
                        } else {
                            consecutiveNoChangeCount = 0
                        }
                    } else if (prevCompositeScreen != currentCompositeScreen) {
                        consecutiveNoChangeCount = 0
                    }

                    if (consecutiveNoChangeCount > 0) {
                        lastOutcomeHint = "[NO SCREEN CHANGE x$consecutiveNoChangeCount after ${prevNonWaitActionType?.name} — try another node, scroll, visual coordinates, or BACK!] $lastOutcomeHint"
                    }

                    val chatGuardHint = buildString {
                        if (chatObs.isMessagingApp) {
                            append("App=${snapshot.packageName}, InChatThread=${chatObs.inChatThread}")
                            if (chatObs.peerName.isNotBlank()) {
                                append(", ActiveChatPeer='${chatObs.peerName}'")
                            }
                            if (chatObs.lastMessageText.isNotBlank()) {
                                append(", LatestVisibleBubble='${chatObs.lastMessageText.take(80)}'")
                            }
                            if (activeAwaitingReply != null && currentSettings.preventDoubleText) {
                                append(" | ANTI-DOUBLE-TEXT LOCKED: You already sent '${activeAwaitingReply.lastSentText.take(60)}' and NO reply has arrived yet. You MUST return WAIT.")
                            }
                            if (currentSettings.autoReplySavedContactsOnly && chatObs.inChatThread && matchedContact == null) {
                                append(" | SAVED-CONTACTS-ONLY: '${chatObs.peerName}' is NOT saved; do NOT message them, return WAIT or BACK.")
                            }
                        }
                    }

                    val shouldZeroApiWaitForReply = cacheMatch == null &&
                        currentSettings.preventDoubleText &&
                        activeAwaitingReply != null &&
                        chatObs.isMessagingApp &&
                        chatObs.inChatThread &&
                        (chatObs.transcriptSignature == activeAwaitingReply.transcriptSignatureAtSend ||
                            ChatTurnGuard.matchesOwnSentMessage(chatObs.lastMessageText, activeAwaitingReply.lastSentText))

                    var nextPendingCarromShotCandidate: PendingCarromShotRecord? = null
                    var aiPreCarromCounts: BoardPieceCounts? = null
                    val myCarromCoinColor = if (
                        workingMemory.contains("black", ignoreCase = true) ||
                        goal.contains("black", ignoreCase = true) ||
                        goal.contains("কালো", ignoreCase = true)
                    ) {
                        CoinColor.BLACK
                    } else {
                        CoinColor.WHITE
                    }

                    val rawPlannedAction: PlannedAction = if (shouldZeroApiWaitForReply) {
                        snapshot.screenshot?.recycle()
                        val waitTarget = chatObs.peerName.ifBlank { activeAwaitingReply?.peerName.orEmpty() }.ifBlank { "recipient" }
                        PlannedAction(
                            actionType = AgentActionType.WAIT,
                            targetNodeId = null,
                            x = null,
                            y = null,
                            textToType = null,
                            waitSeconds = currentSettings.monitorIntervalSec,
                            memoryNote = null,
                            reason = "Anti-Double-Text Guard: waiting for $waitTarget to reply before sending another message.",
                            requiresConfirmation = false
                        )
                    } else if (cacheMatch != null) {
                        carromOfflineWaits = 0
                        didSlideCarromThisStep = false
                        carromTurnMemory.resetTurn()
                        snapshot.screenshot?.recycle()
                        lastReplayEntry = cacheMatch.entry.cacheKey to currentCompositeScreen
                        cacheMatch.action
                    } else {
                        didSlideCarromThisStep = false
                        val localCarromAction: PlannedAction? = if (
                            currentSettings.carromPrecisionSolver &&
                            isActiveCarromMatchScreen &&
                            interactiveNodeCount < 18 &&
                            snapshot.screenshot != null
                        ) {
                            val nowCarromMs = System.currentTimeMillis()
                            val isLiveStream = LiveMediaProjectionCapturer.isLiveActive.value
                            val currentCarromMem = _carromMemory.value
                            val allowRepositionByBudget = !carromTurnMemory.isTurnActive ||
                                carromTurnMemory.canAffordAnotherSliderMove(nowCarromMs, isLiveStream)

                            val outcome = CarromVisionDetector.analyzeAndSolve(
                                screenshot = snapshot.screenshot,
                                screenWidth = snapshot.screenWidth,
                                screenHeight = snapshot.screenHeight,
                                myCoinColor = myCarromCoinColor,
                                consecutiveOfflineWaits = carromOfflineWaits,
                                shotMemory = currentCarromMem,
                                allowStrikerRepositioning = allowRepositionByBudget,
                                usePhysicsRollout = currentSettings.usePhysicsRollout,
                                useSubPixelRefinement = currentSettings.useSubPixelCircleRefinement
                            )

                            // Evaluate the previous Carrom shot once coins have stopped moving
                            val isMovingWait = outcome is CarromVisionOutcome.WaitStillnessOrTurn &&
                                outcome.reason.contains("motion", ignoreCase = true)
                            val prevShot = pendingCarromShot
                            if (prevShot != null && !isMovingWait) {
                                val settledPieces = when (outcome) {
                                    is CarromVisionOutcome.SolvedShot -> outcome.boardPieces
                                    is CarromVisionOutcome.WaitStillnessOrTurn -> outcome.boardPieces
                                    is CarromVisionOutcome.NeedsAiVision -> outcome.boardPieces
                                }
                                val postCounts = if (settledPieces.isNotEmpty()) {
                                    CarromShotMemoryStore.countBoardPieces(settledPieces, prevShot.myCoinColor)
                                } else {
                                    CarromVisionDetector.inspectBoardCounts(
                                        screenshot = snapshot.screenshot,
                                        screenWidth = snapshot.screenWidth,
                                        screenHeight = snapshot.screenHeight,
                                        myCoinColor = prevShot.myCoinColor
                                    )
                                }
                                if (postCounts != null) {
                                    val eval = carromShotMemoryStore?.recordShotOutcome(prevShot, postCounts)
                                    pendingCarromShot = null
                                    if (eval != null) {
                                        lastOutcomeHint = "${eval.summaryMessage} $lastOutcomeHint"
                                        workingMemory = eval.summaryMessage
                                        if (eval.learnedTacticLesson != null || eval.learnedFoulLesson != null) {
                                            val pbKey = activePlaybook?.topicKey ?: "carrom_board_game"
                                            val updatedPb = knowledgeStore?.recordStepLearning(
                                                topicKey = pbKey,
                                                fallbackTopicTitle = activePlaybook?.topicTitle ?: "Carrom Board Game",
                                                newStrategy = eval.learnedTacticLesson,
                                                newFoulLesson = eval.learnedFoulLesson
                                            )
                                            if (updatedPb != null) {
                                                activePlaybook = updatedPb
                                            }
                                        }
                                    }
                                }
                            }

                            when (outcome) {
                                is CarromVisionOutcome.WaitStillnessOrTurn -> {
                                    carromOfflineWaits++
                                    pendingSliderVerify = null
                                    carromTurnMemory.resetTurn()
                                    PlannedAction(
                                        actionType = AgentActionType.WAIT,
                                        targetNodeId = null,
                                        x = null,
                                        y = null,
                                        textToType = null,
                                        waitSeconds = if (isLiveStream) 1 else 2,
                                        memoryNote = null,
                                        reason = outcome.reason,
                                        requiresConfirmation = false
                                    )
                                }
                                is CarromVisionOutcome.SolvedShot -> {
                                    carromOfflineWaits = 0
                                    usedLocalCarromSolver = true
                                    carromTurnMemory.ensureTurnStarted(nowCarromMs)

                                    val striker = outcome.detectedStriker
                                    val sliderKnob = outcome.detectedSliderKnob

                                    // Measure slider gain feedback from previous striker slide (every move refines calibration)
                                    var observedGainThisStep: Float? = null
                                    val verifyState = pendingSliderVerify
                                    if (verifyState != null && striker != null) {
                                        val actualDelta = striker.center.x - verifyState.preStrikerX
                                        if (kotlin.math.abs(verifyState.commandedDeltaX) > 18f &&
                                            kotlin.math.abs(actualDelta) > 8f
                                        ) {
                                            val observedGain = (actualDelta / verifyState.commandedDeltaX).coerceIn(0.65f, 1.45f)
                                            observedGainThisStep = observedGain
                                            carromShotMemoryStore?.updateSliderGain(observedGain)
                                        }
                                    }

                                    // Record pre-slide / post-slide scan into Turn Memory BEFORE choosing slide or fire
                                    carromTurnMemory.recordObservation(
                                        nowMs = nowCarromMs,
                                        pieces = outcome.boardPieces,
                                        striker = striker,
                                        sliderKnob = sliderKnob,
                                        evaluatedSolution = outcome.solution,
                                        observedSliderGain = observedGainThisStep
                                    )

                                    val consensusPieces = carromTurnMemory.buildConsensusPieces(
                                        latestPieces = outcome.boardPieces,
                                        coinRadiusPx = outcome.board.coinRadiusPx
                                    )
                                    val fusedPiecesWithStriker = if (striker != null) {
                                        consensusPieces + striker
                                    } else {
                                        consensusPieces
                                    }

                                    val turnGain = carromTurnMemory.effectiveTurnSliderGain(
                                        currentCarromMem.matchCalibration.sliderGain
                                    )
                                    val canAffordSlide = sliderKnob != null &&
                                        striker != null &&
                                        carromTurnMemory.canAffordAnotherSliderMove(nowCarromMs, isLiveStream)

                                    // Evaluate optimal shot across baseline vs in-place Sure Shot using fused turn memory
                                    val inPlaceSol = if (striker != null) {
                                        CarromGeometrySolver.solveBestShot(
                                            board = outcome.board,
                                            pieces = fusedPiecesWithStriker,
                                            myCoinColor = myCarromCoinColor,
                                            screenWidth = snapshot.screenWidth,
                                            screenHeight = snapshot.screenHeight,
                                            existingStriker = striker,
                                            canRepositionStriker = false,
                                            shotMemory = currentCarromMem,
                                            usePhysicsRollout = currentSettings.usePhysicsRollout
                                        )
                                    } else {
                                        null
                                    }

                                    val optimalSlideSol = if (canAffordSlide) {
                                        CarromGeometrySolver.solveBestShot(
                                            board = outcome.board,
                                            pieces = fusedPiecesWithStriker,
                                            myCoinColor = myCarromCoinColor,
                                            screenWidth = snapshot.screenWidth,
                                            screenHeight = snapshot.screenHeight,
                                            existingStriker = striker,
                                            canRepositionStriker = true,
                                            shotMemory = currentCarromMem,
                                            usePhysicsRollout = currentSettings.usePhysicsRollout
                                        ) ?: outcome.solution
                                    } else {
                                        inPlaceSol ?: outcome.solution
                                    }

                                    // Unlimited slider adjustments within the 9-second turn timer until we lock onto a Sure Shot
                                    val desiredTargetX = if (verifyState != null &&
                                        currentSettings.useClosedLoopSliderVerify &&
                                        kotlin.math.abs(verifyState.targetStrikerX - (striker?.center?.x ?: 0f)) > 22f &&
                                        !optimalSlideSol.isSureShot
                                    ) {
                                        verifyState.targetStrikerX
                                    } else {
                                        optimalSlideSol.strikerPos.x
                                    }

                                    val alreadyHasSureInPlace = inPlaceSol != null &&
                                        inPlaceSol.isSureShot &&
                                        (!optimalSlideSol.isSureShot || inPlaceSol.confidenceScore >= optimalSlideSol.confidenceScore - 0.06f)

                                    val needsSliderAdjustment = canAffordSlide &&
                                        striker != null &&
                                        sliderKnob != null &&
                                        !alreadyHasSureInPlace &&
                                        kotlin.math.abs(desiredTargetX - striker.center.x) > 22f &&
                                        !carromTurnMemory.isSliderStalledOrOscillating(desiredTargetX, striker.center.x)

                                    if (needsSliderAdjustment) {
                                        val rawDelta = desiredTargetX - striker!!.center.x
                                        val calibratedDelta = (rawDelta / turnGain).coerceIn(-380f, 380f)
                                        val targetKnobEndX = (sliderKnob!!.x + calibratedDelta)
                                            .toInt()
                                            .coerceIn(16, (snapshot.screenWidth - 16).coerceAtLeast(17))

                                        didSlideCarromThisStep = true
                                        movedStrikerThisTurn = true
                                        carromTurnMemory.recordSliderMoveDispatched(striker.center.x)
                                        pendingSliderVerify = PendingSliderVerify(
                                            targetStrikerX = desiredTargetX,
                                            preStrikerX = striker.center.x,
                                            commandedDeltaX = (targetKnobEndX - sliderKnob.x.toInt()).toFloat(),
                                            correctionAttempts = (verifyState?.correctionAttempts ?: 0) + 1
                                        )
                                        val remSec = carromTurnMemory.remainingMs(nowCarromMs) / 1000f
                                        PlannedAction(
                                            actionType = AgentActionType.DRAG,
                                            targetNodeId = null,
                                            x = sliderKnob.x.toInt(),
                                            y = sliderKnob.y.toInt(),
                                            endX = targetKnobEndX,
                                            endY = sliderKnob.y.toInt(),
                                            durationMs = if (kotlin.math.abs(calibratedDelta) < 120f) 160L else 210L,
                                            textToType = null,
                                            waitSeconds = null,
                                            memoryNote = "Turn Memory (${carromTurnMemory.scanCount} scans, rem=${String.format(java.util.Locale.US, "%.1f", remSec)}s): sliding #${carromTurnMemory.sliderMovesThisTurn} (gain=${String.format(java.util.Locale.US, "%.2f", turnGain)})",
                                            reason = "Carrom Turn Scan #${carromTurnMemory.scanCount}: slide striker #${carromTurnMemory.sliderMovesThisTurn} (${String.format(java.util.Locale.US, "%.1f", remSec)}s left)",
                                            requiresConfirmation = false
                                        )
                                    } else {
                                        // Combine ALL turn scan memory + live striker + physics rollout & FIRE immediately within 9s!
                                        pendingSliderVerify = null
                                        val didSlideBeforeThisShot = movedStrikerThisTurn
                                        movedStrikerThisTurn = false

                                        val finalSol = inPlaceSol ?: carromTurnMemory.solveCombinedFinalShot(
                                            board = outcome.board,
                                            latestPieces = outcome.boardPieces,
                                            striker = striker,
                                            myCoinColor = myCarromCoinColor,
                                            screenWidth = snapshot.screenWidth,
                                            screenHeight = snapshot.screenHeight,
                                            shotMemory = currentCarromMem,
                                            usePhysicsRollout = currentSettings.usePhysicsRollout,
                                            fallbackSolution = outcome.solution
                                        )

                                        val telemetry = carromTurnMemory.formatTurnTelemetry(nowCarromMs, finalSol)
                                        carromTurnMemory.resetTurn()

                                        val shootStartX = striker?.center?.x ?: finalSol.strikerPos.x
                                        val shootStartY = striker?.center?.y ?: finalSol.strikerPos.y
                                        val strikerNormX = ((shootStartX - outcome.board.left) / outcome.board.width).coerceIn(0f, 1f)

                                        if (consensusPieces.isNotEmpty() || outcome.boardPieces.isNotEmpty()) {
                                            val countSource = consensusPieces.ifEmpty { outcome.boardPieces }
                                            nextPendingCarromShotCandidate = PendingCarromShotRecord(
                                                myCoinColor = myCarromCoinColor,
                                                targetCoinColor = finalSol.targetCoin.color,
                                                pocketIndex = finalSol.pocketIndex,
                                                cutAngleDeg = finalSol.cutAngleDeg,
                                                strikerNormalizedX = strikerNormX,
                                                powerStyle = finalSol.powerStyle,
                                                movedStrikerWithSlider = didSlideBeforeThisShot,
                                                isBankShot = finalSol.isBankShot,
                                                isComboShot = finalSol.isComboShot,
                                                isBreakShot = finalSol.isBreakShot,
                                                preCounts = CarromShotMemoryStore.countBoardPieces(countSource, myCarromCoinColor)
                                            )
                                        }

                                        val dx = finalSol.dragEnd.x - finalSol.strikerPos.x
                                        val dy = finalSol.dragEnd.y - finalSol.strikerPos.y
                                        val shootEndX = (shootStartX + dx).toInt().coerceIn(16, (snapshot.screenWidth - 16).coerceAtLeast(17))
                                        val shootEndY = (shootStartY + dy).toInt().coerceIn(16, (snapshot.screenHeight - 16).coerceAtLeast(17))
                                        PlannedAction(
                                            actionType = AgentActionType.DRAG,
                                            targetNodeId = null,
                                            x = shootStartX.toInt(),
                                            y = shootStartY.toInt(),
                                            endX = shootEndX,
                                            endY = shootEndY,
                                            durationMs = 520L,
                                            textToType = null,
                                            waitSeconds = null,
                                            memoryNote = "Carrom (${myCarromCoinColor.name}) $telemetry: ${finalSol.reason}",
                                            reason = "Zero-API Carrom $telemetry: ${finalSol.reason} (conf=${(finalSol.confidenceScore * 100).toInt()}%)",
                                            requiresConfirmation = false
                                        )
                                    }
                                }
                                is CarromVisionOutcome.NeedsAiVision -> {
                                    carromOfflineWaits = 0
                                    pendingSliderVerify = null
                                    carromTurnMemory.resetTurn()
                                    lastOutcomeHint = "[Carrom Local Vision: ${outcome.diagnostic}] $lastOutcomeHint"
                                    if (outcome.boardPieces.isNotEmpty()) {
                                        aiPreCarromCounts = CarromShotMemoryStore.countBoardPieces(outcome.boardPieces, myCarromCoinColor)
                                    }
                                    null
                                }
                            }
                        } else {
                            carromTurnMemory.resetTurn()
                            if (isCarromContext && snapshot.screenshot == null) {
                                lastOutcomeHint = "[CARROM SCREENSHOT MISSING (${snapshot.captureNote}) — Turn ON Multimodal Vision Screenshot in Settings and ensure game allows screen capture.] $lastOutcomeHint"
                            }
                            null
                        }

                        if (localCarromAction != null) {
                            snapshot.screenshot?.recycle()
                            localCarromAction
                        } else {
                            try {
                                val thinkingNote = if (isCarromContext && snapshot.screenshot == null) {
                                    "Carrom Screenshot Missing: ${snapshot.captureNote}"
                                } else {
                                    snapshot.captureNote
                                }
                                _uiState.value = AgentUiState.Running(
                                    goal = goal,
                                    mode = mode,
                                    currentStep = step,
                                    maxSteps = maxStepsLimit,
                                    phase = AgentPhase.THINKING,
                                    lastThought = thinkingNote,
                                    workingMemory = workingMemory,
                                    trace = trace.toList()
                                )

                                val carromPromptMem = if (isCarromContext) {
                                    _carromMemory.value.formatForPrompt()
                                } else {
                                    ""
                                }
                                val planResult = withContext(Dispatchers.IO) {
                                    aiPlanner.planNextAction(
                                        goal = goal,
                                        mode = mode,
                                        settings = currentSettings,
                                        snapshot = snapshot,
                                        recentHistory = trace,
                                        recentNotifications = notificationsSnapshot,
                                        workingMemory = workingMemory,
                                        playbook = activePlaybook,
                                        lastOutcomeHint = lastOutcomeHint,
                                        stepNumber = step,
                                        maxSteps = maxStepsLimit,
                                        matchedContact = matchedContact,
                                        savedContactsSummary = savedContactsSummary,
                                        chatGuardHint = chatGuardHint,
                                        carromMemorySummary = carromPromptMem,
                                        forceVision = isCarromContext
                                    )
                                }

                                planResult.getOrElse { err ->
                                    val msg = err.localizedMessage ?: "AI planning failed."
                                    val isConfigOrAuthError = msg.contains("Firebase is not", ignoreCase = true) ||
                                        msg.contains("Missing API key", ignoreCase = true) ||
                                        msg.contains("HTTP 401") ||
                                        msg.contains("HTTP 403")
                                    if (mode == AgentExecutionMode.CONTINUOUS_MONITOR && !isConfigOrAuthError) {
                                        // Recoverable network/transient hiccup in all-day monitor: wait and retry
                                        appendTrace(
                                            trace,
                                            AgentStepLog(
                                                stepNumber = step,
                                                timestampMs = System.currentTimeMillis(),
                                                actionType = AgentActionType.WAIT,
                                                targetSummary = "Transient AI error",
                                                reason = msg,
                                                status = StepStatus.WAITING,
                                                requiresConfirmation = false
                                            )
                                        )
                                        sleepMonitorInterval(
                                            goal = goal,
                                            mode = mode,
                                            step = step,
                                            maxSteps = maxStepsLimit,
                                            waitSec = currentSettings.monitorIntervalSec,
                                            reason = "Retrying after transient error: ${err.localizedMessage ?: "network"}",
                                            workingMemory = workingMemory,
                                            trace = trace
                                        )
                                        step++
                                        continue
                                    } else {
                                        _uiState.value = AgentUiState.Error(
                                            goal = goal,
                                            message = err.localizedMessage ?: "AI planning failed.",
                                            workingMemory = workingMemory,
                                            trace = trace.toList()
                                        )
                                        return@launch
                                    }
                                }
                            } finally {
                                snapshot.screenshot?.recycle()
                            }
                        }
                    }

                    val rawTargetNode = rawPlannedAction.targetNodeId?.let { id ->
                        snapshot.nodes.firstOrNull { it.id == id }
                    }
                    val plannedAction: PlannedAction = when {
                        currentSettings.autoReplySavedContactsOnly &&
                            chatObs.isMessagingApp &&
                            chatObs.inChatThread &&
                            chatObs.peerName.isNotBlank() &&
                            matchedContact == null &&
                            ChatTurnGuard.isChatSendAction(rawPlannedAction, rawTargetNode, snapshot) -> {
                            PlannedAction(
                                actionType = AgentActionType.WAIT,
                                targetNodeId = null,
                                x = null,
                                y = null,
                                textToType = null,
                                waitSeconds = currentSettings.monitorIntervalSec,
                                memoryNote = rawPlannedAction.memoryNote,
                                reason = "Saved-Contacts-Only mode: skipped unsaved contact '${chatObs.peerName}'.",
                                requiresConfirmation = false
                            )
                        }
                        currentSettings.preventDoubleText &&
                            ChatTurnGuard.shouldBlockDuplicateSend(
                                action = rawPlannedAction,
                                targetNode = rawTargetNode,
                                obs = chatObs,
                                recentNotifications = notificationsSnapshot,
                                awaiting = activeAwaitingReply,
                                snapshot = snapshot
                            ) -> {
                            val waitTarget = chatObs.peerName.ifBlank { "contact" }
                            PlannedAction(
                                actionType = AgentActionType.WAIT,
                                targetNodeId = null,
                                x = null,
                                y = null,
                                textToType = null,
                                waitSeconds = currentSettings.monitorIntervalSec,
                                memoryNote = rawPlannedAction.memoryNote,
                                reason = "Anti-Double-Text Guard: blocked repeat message to $waitTarget until they reply.",
                                requiresConfirmation = false
                            )
                        }
                        else -> rawPlannedAction
                    }

                    if (!plannedAction.memoryNote.isNullOrBlank()) {
                        workingMemory = plannedAction.memoryNote
                    }

                    if (!plannedAction.learnedRule.isNullOrBlank() || !plannedAction.foulOrMistakeLesson.isNullOrBlank()) {
                        val topicKey = activePlaybook?.topicKey
                            ?: AgentKnowledgeStore.normalizeKey(goal.take(40))
                        val updatedPlaybook = knowledgeStore?.recordStepLearning(
                            topicKey = topicKey,
                            fallbackTopicTitle = activePlaybook?.topicTitle ?: goal.take(40),
                            newStrategy = plannedAction.learnedRule,
                            newFoulLesson = plannedAction.foulOrMistakeLesson
                        )
                        if (updatedPlaybook != null) {
                            activePlaybook = updatedPlaybook
                        }
                    }

                    val targetNode = plannedAction.targetNodeId?.let { id ->
                        snapshot.nodes.firstOrNull { it.id == id }
                    }
                    val targetSummary = buildTargetSummary(plannedAction, targetNode)

                    if (plannedAction.actionType == AgentActionType.COMPLETE) {
                        appendTrace(
                            trace,
                            AgentStepLog(
                                stepNumber = step,
                                timestampMs = System.currentTimeMillis(),
                                actionType = AgentActionType.COMPLETE,
                                targetSummary = "Goal satisfied",
                                reason = plannedAction.reason,
                                status = StepStatus.COMPLETED,
                                requiresConfirmation = false
                            )
                        )
                        _uiState.value = AgentUiState.Completed(
                            goal = goal,
                            summary = plannedAction.reason,
                            totalSteps = step,
                            workingMemory = workingMemory,
                            trace = trace.toList()
                        )
                        return@launch
                    }

                    if (plannedAction.actionType == AgentActionType.FAIL) {
                        appendTrace(
                            trace,
                            AgentStepLog(
                                stepNumber = step,
                                timestampMs = System.currentTimeMillis(),
                                actionType = AgentActionType.FAIL,
                                targetSummary = targetSummary,
                                reason = plannedAction.reason,
                                status = StepStatus.FAILED,
                                requiresConfirmation = false
                            )
                        )
                        _uiState.value = AgentUiState.Error(
                            goal = goal,
                            message = plannedAction.reason,
                            workingMemory = workingMemory,
                            trace = trace.toList()
                        )
                        return@launch
                    }

                    if (plannedAction.actionType == AgentActionType.WAIT) {
                        val waitSec = plannedAction.waitSeconds
                            ?: if (mode == AgentExecutionMode.CONTINUOUS_MONITOR) {
                                currentSettings.monitorIntervalSec
                            } else {
                                3
                            }
                        appendTrace(
                            trace,
                            AgentStepLog(
                                stepNumber = step,
                                timestampMs = System.currentTimeMillis(),
                                actionType = AgentActionType.WAIT,
                                targetSummary = "Wait ${waitSec}s (wakes on notification)",
                                reason = plannedAction.reason,
                                status = StepStatus.WAITING,
                                requiresConfirmation = false
                            )
                        )
                        sleepMonitorInterval(
                            goal = goal,
                            mode = mode,
                            step = step,
                            maxSteps = maxStepsLimit,
                            waitSec = waitSec,
                            reason = plannedAction.reason,
                            workingMemory = workingMemory,
                            trace = trace
                        )
                        step++
                        continue
                    }

                    if (!usedLocalCarromSolver && SafetyGuard.isStuckLoop(trace, plannedAction, targetSummary)) {
                        _uiState.value = AgentUiState.Error(
                            goal = goal,
                            message = "Stopped to prevent an infinite loop repeating: $targetSummary",
                            workingMemory = workingMemory,
                            trace = trace.toList()
                        )
                        return@launch
                    }

                    val safetyCheck = SafetyGuard.evaluate(
                        action = plannedAction,
                        targetNode = targetNode,
                        policyMode = currentSettings.safetyPolicyMode
                    )

                    var finalStatus = StepStatus.EXECUTED
                    if (safetyCheck.needsConfirmation) {
                        val deferred = CompletableDeferred<Boolean>()
                        confirmationDeferred = deferred
                        _uiState.value = AgentUiState.AwaitingConfirmation(
                            goal = goal,
                            mode = mode,
                            currentStep = step,
                            maxSteps = maxStepsLimit,
                            pendingAction = plannedAction,
                            targetDescription = targetSummary,
                            safetyReason = safetyCheck.reason,
                            workingMemory = workingMemory,
                            trace = trace.toList()
                        )
                        val approved = deferred.await()
                        confirmationDeferred = null
                        if (!approved) {
                            appendTrace(
                                trace,
                                AgentStepLog(
                                    stepNumber = step,
                                    timestampMs = System.currentTimeMillis(),
                                    actionType = plannedAction.actionType,
                                    targetSummary = targetSummary,
                                    reason = plannedAction.reason,
                                    status = StepStatus.REJECTED,
                                    requiresConfirmation = true
                                )
                            )
                            _uiState.value = AgentUiState.Completed(
                                goal = goal,
                                summary = "Stopped safely: payment/sensitive action declined by user.",
                                totalSteps = step,
                                workingMemory = workingMemory,
                                trace = trace.toList()
                            )
                            return@launch
                        }
                        finalStatus = StepStatus.APPROVED
                    }

                    awaitIfPaused(goal, mode, step, maxStepsLimit, workingMemory, trace)

                    _uiState.value = AgentUiState.Running(
                        goal = goal,
                        mode = mode,
                        currentStep = step,
                        maxSteps = maxStepsLimit,
                        phase = AgentPhase.ACTING,
                        lastThought = plannedAction.reason,
                        workingMemory = workingMemory,
                        trace = trace.toList()
                    )

                    val execResult = service.executeAction(plannedAction, snapshot)
                    val stepResultSummary = execResult.summary.ifBlank { targetSummary }
                    prevCompositeScreen = currentCompositeScreen
                    prevNonWaitActionType = plannedAction.actionType

                    if (execResult.success) {
                        if (nextPendingCarromShotCandidate != null) {
                            pendingCarromShot = nextPendingCarromShotCandidate
                        } else if (isCarromContext &&
                            aiPreCarromCounts != null &&
                            plannedAction.actionType == AgentActionType.DRAG &&
                            plannedAction.x != null && plannedAction.y != null &&
                            plannedAction.endX != null && plannedAction.endY != null
                        ) {
                            val dragDx = plannedAction.endX - plannedAction.x
                            val dragDy = plannedAction.endY - plannedAction.y
                            val isHorizontalSlider = kotlin.math.abs(dragDy) < 28 && plannedAction.y > snapshot.screenHeight * 0.78f
                            if (isHorizontalSlider) {
                                movedStrikerThisTurn = true
                            } else {
                                val dragDist = kotlin.math.hypot(dragDx.toDouble(), dragDy.toDouble()).toFloat()
                                val power = when {
                                    dragDist < 205f -> CarromPowerStyle.SOFT
                                    dragDist > 315f -> CarromPowerStyle.HARD
                                    else -> CarromPowerStyle.MEDIUM
                                }
                                // Shot direction is opposite slingshot pull
                                val aimLeft = -dragDx < 0
                                val aimTop = -dragDy <= 0
                                val estPocketIdx = when {
                                    aimTop && aimLeft -> 0
                                    aimTop && !aimLeft -> 1
                                    !aimTop && aimLeft -> 2
                                    else -> 3
                                }
                                pendingCarromShot = PendingCarromShotRecord(
                                    myCoinColor = myCarromCoinColor,
                                    targetCoinColor = myCarromCoinColor,
                                    pocketIndex = estPocketIdx,
                                    cutAngleDeg = 18f,
                                    powerStyle = power,
                                    movedStrikerWithSlider = movedStrikerThisTurn,
                                    isBankShot = false,
                                    isComboShot = false,
                                    isBreakShot = false,
                                    preCounts = aiPreCarromCounts
                                )
                                movedStrikerThisTurn = false
                            }
                        }
                    }

                    if (execResult.success && chatObs.isMessagingApp) {
                        val peerKeyForAction = ChatTurnGuard.peerKey(snapshot.packageName, chatObs.peerName)
                        if (plannedAction.actionType == AgentActionType.INPUT_TEXT && !plannedAction.textToType.isNullOrBlank()) {
                            pendingSentTextForPeer = peerKeyForAction to plannedAction.textToType.trim()
                        } else if (ChatTurnGuard.isChatSendAction(plannedAction, targetNode, snapshot)) {
                            val sentText = pendingSentTextForPeer
                                ?.takeIf { it.first == peerKeyForAction }
                                ?.second
                                ?: chatObs.lastMessageText
                            awaitingRepliesByPeer[peerKeyForAction] = ChatTurnGuard.AwaitingReplyState(
                                packageName = snapshot.packageName,
                                peerName = chatObs.peerName,
                                lastSentText = sentText,
                                transcriptSignatureAtSend = chatObs.transcriptSignature,
                                sentAtMs = System.currentTimeMillis()
                            )
                        }
                    }

                    if (!usedLocalCarromSolver && cacheMatch == null && execResult.success && currentSettings.useZeroApiActionCache) {
                        pendingCacheCandidate = PendingCacheCandidate(
                            goal = goal,
                            packageName = snapshot.packageName,
                            screenSignature = currentSig,
                            interactiveNodeCount = interactiveNodeCount,
                            action = plannedAction,
                            targetNode = targetNode,
                            hadConfirmedSafety = safetyCheck.needsConfirmation
                        )
                    } else if (!execResult.success && cacheMatch != null) {
                        actionCacheStore?.removeEntry(cacheMatch.entry.cacheKey)
                        lastReplayEntry = null
                    }
                    lastOutcomeHint = "Previous step executed ${plannedAction.actionType.name}: $stepResultSummary " +
                        "[status=${if (execResult.success) finalStatus.name else "FAILED"}]. " +
                        "Reason was: ${plannedAction.reason}. Compare the new screenshot carefully: if it succeeded/pocketed, save learnedRule; if foul/missed/wrong, save foulOrMistakeLesson."
                    appendTrace(
                        trace,
                        AgentStepLog(
                            stepNumber = step,
                            timestampMs = System.currentTimeMillis(),
                            actionType = plannedAction.actionType,
                            targetSummary = stepResultSummary,
                            reason = if (cacheMatch != null) {
                                "Zero-API Cache Replay (0 API calls) — ${cacheMatch.entry.targetLabel}"
                            } else {
                                plannedAction.reason
                            },
                            status = if (execResult.success) finalStatus else StepStatus.FAILED,
                            requiresConfirmation = safetyCheck.needsConfirmation
                        )
                    )

                    _uiState.value = AgentUiState.Running(
                        goal = goal,
                        mode = mode,
                        currentStep = step,
                        maxSteps = maxStepsLimit,
                        phase = AgentPhase.WAITING_DELAY,
                        lastThought = plannedAction.reason,
                        workingMemory = workingMemory,
                        trace = trace.toList()
                    )
                    val effectiveDelayMs = when {
                        usedLocalCarromSolver && didSlideCarromThisStep && LiveMediaProjectionCapturer.isLiveActive.value -> 110L
                        usedLocalCarromSolver && didSlideCarromThisStep -> 140L
                        LiveMediaProjectionCapturer.isLiveActive.value && usedLocalCarromSolver -> 460L
                        else -> currentSettings.stepDelayMs
                    }
                    delay(effectiveDelayMs)
                    step++
                }

                _uiState.value = AgentUiState.Completed(
                    goal = goal,
                    summary = "Reached single-task step limit (${maxStepsLimit ?: 0}) without explicit completion.",
                    totalSteps = maxStepsLimit ?: trace.size,
                    workingMemory = workingMemory,
                    trace = trace.toList()
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected automation error", e)
                _uiState.value = AgentUiState.Error(
                    goal = goal,
                    message = e.localizedMessage ?: "Unexpected automation failure.",
                    workingMemory = workingMemory,
                    trace = trace.toList()
                )
            } finally {
                notificationWakeSignal = null
                appContext?.let { AgentKeepAliveService.stop(it) }
            }
        }
    }

    fun pauseTask(reason: String = "Automation paused. Tap Resume or Stop.") {
        val current = _uiState.value
        if (current is AgentUiState.Running && pauseGate == null) {
            pauseGate = CompletableDeferred()
            _uiState.value = AgentUiState.Paused(
                goal = current.goal,
                mode = current.mode,
                currentStep = current.currentStep,
                maxSteps = current.maxSteps,
                pauseReason = reason,
                workingMemory = current.workingMemory,
                trace = current.trace
            )
        }
    }

    fun resumeTask() {
        pauseGate?.complete(Unit)
        pauseGate = null
    }

    fun confirmPendingAction(approved: Boolean) {
        confirmationDeferred?.complete(approved)
    }

    fun stopTask(reason: String? = null) {
        activeRunJob?.cancel()
        activeRunJob = null
        pauseGate?.cancel()
        pauseGate = null
        confirmationDeferred?.cancel()
        confirmationDeferred = null
        notificationWakeSignal?.cancel()
        notificationWakeSignal = null
        appContext?.let { AgentKeepAliveService.stop(it) }

        val currentTrace = extractTrace(_uiState.value)
        val currentGoal = extractGoal(_uiState.value)
        val currentMemory = extractMemory(_uiState.value)
        _uiState.value = if (reason != null) {
            AgentUiState.Error(
                goal = currentGoal,
                message = reason,
                workingMemory = currentMemory,
                trace = currentTrace
            )
        } else if (currentTrace.isNotEmpty()) {
            AgentUiState.Completed(
                goal = currentGoal,
                summary = "Automation stopped by user.",
                totalSteps = currentTrace.size,
                workingMemory = currentMemory,
                trace = currentTrace
            )
        } else {
            AgentUiState.Idle
        }
    }

    fun dismissFinishedState() {
        val current = _uiState.value
        if (current is AgentUiState.Completed || current is AgentUiState.Error) {
            _uiState.value = AgentUiState.Idle
        }
    }

    private suspend fun sleepMonitorInterval(
        goal: String,
        mode: AgentExecutionMode,
        step: Int,
        maxSteps: Int?,
        waitSec: Int,
        reason: String,
        workingMemory: String,
        trace: List<AgentStepLog>
    ) {
        val wakeSignal = CompletableDeferred<Unit>()
        notificationWakeSignal = wakeSignal
        _uiState.value = AgentUiState.Running(
            goal = goal,
            mode = mode,
            currentStep = step,
            maxSteps = maxSteps,
            phase = AgentPhase.MONITOR_SLEEPING,
            lastThought = "Waiting up to ${waitSec}s • $reason",
            workingMemory = workingMemory,
            trace = trace.toList()
        )
        try {
            withTimeoutOrNull(waitSec * 1000L) {
                wakeSignal.await()
            }
        } finally {
            if (notificationWakeSignal === wakeSignal) {
                notificationWakeSignal = null
            }
        }
    }

    private suspend fun awaitIfPaused(
        goal: String,
        mode: AgentExecutionMode,
        step: Int,
        maxSteps: Int?,
        workingMemory: String,
        trace: List<AgentStepLog>
    ) {
        val gate = pauseGate ?: return
        _uiState.value = AgentUiState.Paused(
            goal = goal,
            mode = mode,
            currentStep = step,
            maxSteps = maxSteps,
            pauseReason = "Automation paused. Tap Resume or Stop.",
            workingMemory = workingMemory,
            trace = trace.toList()
        )
        gate.await()
    }

    private fun appendTrace(trace: MutableList<AgentStepLog>, entry: AgentStepLog) {
        if (trace.size >= MAX_TRACE_ENTRIES) {
            trace.removeAt(0)
        }
        trace.add(entry)
    }

    private fun buildTargetSummary(action: PlannedAction, node: ScreenNode?): String {
        return when {
            action.actionType == AgentActionType.DRAG &&
                action.x != null && action.y != null &&
                action.endX != null && action.endY != null ->
                "Drag (${action.x},${action.y}) -> (${action.endX},${action.endY})"
            node != null -> "#${node.id} ${node.displayLabel()} (${node.bounds.centerX},${node.bounds.centerY})"
            action.x != null && action.y != null -> "(${action.x}, ${action.y})"
            !action.textToType.isNullOrBlank() -> "\"${action.textToType}\""
            else -> action.actionType.name
        }
    }

    private fun extractTrace(state: AgentUiState): List<AgentStepLog> = when (state) {
        is AgentUiState.Idle -> emptyList()
        is AgentUiState.Running -> state.trace
        is AgentUiState.Paused -> state.trace
        is AgentUiState.AwaitingConfirmation -> state.trace
        is AgentUiState.Completed -> state.trace
        is AgentUiState.Error -> state.trace
    }

    private fun extractGoal(state: AgentUiState): String = when (state) {
        is AgentUiState.Idle -> ""
        is AgentUiState.Running -> state.goal
        is AgentUiState.Paused -> state.goal
        is AgentUiState.AwaitingConfirmation -> state.goal
        is AgentUiState.Completed -> state.goal
        is AgentUiState.Error -> state.goal
    }

    private fun isFirebaseConfigured(): Boolean {
        return runCatching { com.google.firebase.FirebaseApp.getInstance() }.isSuccess
    }

    private fun extractMemory(state: AgentUiState): String = when (state) {
        is AgentUiState.Idle -> ""
        is AgentUiState.Running -> state.workingMemory
        is AgentUiState.Paused -> state.workingMemory
        is AgentUiState.AwaitingConfirmation -> state.workingMemory
        is AgentUiState.Completed -> state.workingMemory
        is AgentUiState.Error -> state.workingMemory
    }
}
