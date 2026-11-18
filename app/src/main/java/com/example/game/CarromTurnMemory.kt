package com.example.game

import kotlin.math.abs
import kotlin.math.hypot

/**
 * Per-Turn Working Memory & 9-Second Turn Shot Clock for Carrom.
 *
 * Accumulates every board observation captured BEFORE and AFTER each striker slider move
 * in the current turn, fuses non-striker pieces across scans into a stabilized consensus map,
 * allows unlimited slider moves as long as the 9-second turn shot clock permits, and finally
 * fuses all turn observations + physics rollout + match calibration to fire a verified Sure Shot
 * strictly within 9 seconds.
 */
class CarromTurnMemory {

    companion object {
        const val MAX_TURN_BUDGET_MS = 9_000L
        private const val LIVE_FIRE_RESERVE_MS = 1_250L
        private const val A11Y_FIRE_RESERVE_MS = 2_350L
        private const val MAX_STORED_SCANS = 18
    }

    data class TurnScanFrame(
        val timestampMs: Long,
        val strikerX: Float?,
        val sliderKnobX: Float?,
        val pieces: List<CarromPiece>,
        val bestConfidence: Float,
        val isSureShot: Boolean
    )

    private var turnStartMs: Long = 0L
    private val scanFrames = mutableListOf<TurnScanFrame>()
    private val visitedStrikerXs = mutableListOf<Float>()
    private val localSliderGains = mutableListOf<Float>()
    private var _sliderMovesThisTurn: Int = 0
    private var bestEvaluatedThisTurn: CarromShotSolution? = null

    val sliderMovesThisTurn: Int get() = _sliderMovesThisTurn
    val scanCount: Int get() = scanFrames.size
    val isTurnActive: Boolean get() = turnStartMs > 0L

    fun ensureTurnStarted(nowMs: Long = System.currentTimeMillis()) {
        if (turnStartMs == 0L) {
            turnStartMs = nowMs
        }
    }

    fun elapsedMs(nowMs: Long = System.currentTimeMillis()): Long {
        if (turnStartMs == 0L) return 0L
        return (nowMs - turnStartMs).coerceAtLeast(0L)
    }

    fun remainingMs(nowMs: Long = System.currentTimeMillis()): Long {
        if (turnStartMs == 0L) return MAX_TURN_BUDGET_MS
        return (MAX_TURN_BUDGET_MS - elapsedMs(nowMs)).coerceAtLeast(0L)
    }

    /**
     * Returns true if there is enough time left in the 9-second turn window to perform
     * another striker slider drag, settle/read the next frame, and still execute the shot drag
     * cleanly before 9,000ms elapses.
     */
    fun canAffordAnotherSliderMove(
        nowMs: Long = System.currentTimeMillis(),
        isLiveStreamActive: Boolean
    ): Boolean {
        ensureTurnStarted(nowMs)
        val reserve = if (isLiveStreamActive) LIVE_FIRE_RESERVE_MS else A11Y_FIRE_RESERVE_MS
        return remainingMs(nowMs) > reserve
    }

    fun recordSliderMoveDispatched(preStrikerX: Float?) {
        _sliderMovesThisTurn++
        if (preStrikerX != null) {
            visitedStrikerXs.add(preStrikerX)
        }
    }

    fun recordObservation(
        nowMs: Long = System.currentTimeMillis(),
        pieces: List<CarromPiece>,
        striker: CarromPiece?,
        sliderKnob: Point2D?,
        evaluatedSolution: CarromShotSolution?,
        observedSliderGain: Float? = null
    ) {
        ensureTurnStarted(nowMs)
        if (observedSliderGain != null) {
            localSliderGains.add(observedSliderGain.coerceIn(0.65f, 1.45f))
        }
        val nonStriker = pieces.filter { it.color != CoinColor.STRIKER }
        scanFrames.add(
            TurnScanFrame(
                timestampMs = nowMs,
                strikerX = striker?.center?.x,
                sliderKnobX = sliderKnob?.x,
                pieces = nonStriker,
                bestConfidence = evaluatedSolution?.confidenceScore ?: 0f,
                isSureShot = evaluatedSolution?.isSureShot == true
            )
        )
        if (scanFrames.size > MAX_STORED_SCANS) {
            scanFrames.removeAt(0)
        }
        if (striker != null) {
            visitedStrikerXs.add(striker.center.x)
        }
        if (evaluatedSolution != null) {
            val prev = bestEvaluatedThisTurn
            if (prev == null ||
                (evaluatedSolution.isSureShot && !prev.isSureShot) ||
                (evaluatedSolution.isSureShot == prev.isSureShot && evaluatedSolution.confidenceScore > prev.confidenceScore)
            ) {
                bestEvaluatedThisTurn = evaluatedSolution
            }
        }
    }

    /**
     * Detects when the slider has hit the baseline stop or started oscillating between the
     * same X positions so the agent stops sliding and fires right away instead of burning the clock.
     */
    fun isSliderStalledOrOscillating(targetStrikerX: Float, currentStrikerX: Float): Boolean {
        if (visitedStrikerXs.size < 2) return false
        val lastPrev = visitedStrikerXs[visitedStrikerXs.size - 2]
        val movedFromPrev = abs(currentStrikerX - lastPrev)
        val stillFarFromTarget = abs(targetStrikerX - currentStrikerX) > 24f
        if (_sliderMovesThisTurn >= 2 && movedFromPrev < 6f && stillFarFromTarget) {
            return true
        }
        // Check 3-point X oscillation (e.g. X1 -> X2 -> X1)
        if (visitedStrikerXs.size >= 4) {
            val xA = visitedStrikerXs[visitedStrikerXs.size - 1]
            val xC = visitedStrikerXs[visitedStrikerXs.size - 3]
            if (abs(xA - xC) < 12f && abs(xA - targetStrikerX) > 24f) {
                return true
            }
        }
        return false
    }

    fun effectiveTurnSliderGain(persistedGain: Float): Float {
        if (localSliderGains.isEmpty()) return persistedGain
        val localAvg = localSliderGains.average().toFloat()
        return ((persistedGain + localAvg * localSliderGains.size) / (1 + localSliderGains.size))
            .coerceIn(0.65f, 1.45f)
    }

    /**
     * Fuses all pre-slide and post-slide coin/Queen observations in the current turn into a
     * jitter-free consensus board piece list. Always anchors on [latestPieces] while averaging
     * stabilized coordinates from matching pieces across earlier turn scans and preserving
     * any real coin briefly missed by single-frame glare/occlusion.
     */
    fun buildConsensusPieces(
        latestPieces: List<CarromPiece>,
        coinRadiusPx: Float = 28f
    ): List<CarromPiece> {
        val latestNonStriker = latestPieces.filter { it.color != CoinColor.STRIKER }
        if (scanFrames.size <= 1) return latestNonStriker

        val matchThreshold = (coinRadiusPx * 1.2f).coerceAtLeast(20f)
        val fused = ArrayList<CarromPiece>(latestNonStriker.size + 4)

        // 1. Stabilize each latest detected coin with matching detections across recent turn frames
        for (piece in latestNonStriker) {
            var sumX = piece.center.x * 1.4f
            var sumY = piece.center.y * 1.4f
            var weight = 1.4f

            for (frame in scanFrames.dropLast(1)) {
                val match = frame.pieces.minByOrNull { prev ->
                    if (prev.color != piece.color) Float.MAX_VALUE
                    else hypot(prev.center.x - piece.center.x, prev.center.y - piece.center.y)
                }
                if (match != null &&
                    match.color == piece.color &&
                    hypot(match.center.x - piece.center.x, match.center.y - piece.center.y) <= matchThreshold
                ) {
                    sumX += match.center.x
                    sumY += match.center.y
                    weight += 1f
                }
            }
            fused.add(
                piece.copy(
                    center = Point2D(sumX / weight, sumY / weight)
                )
            )
        }

        // 2. Recover any persistent coin spotted in >= 2 prior scans this turn that dropped out in latest frame
        if (scanFrames.size >= 3) {
            val candidatePool = scanFrames.dropLast(1).flatMap { it.pieces }
            val minConsensusFrames = 2
            val visitedExtra = mutableListOf<Point2D>()

            for (cand in candidatePool) {
                val alreadyInFused = fused.any {
                    hypot(it.center.x - cand.center.x, it.center.y - cand.center.y) <= matchThreshold
                }
                val alreadyRecovered = visitedExtra.any {
                    hypot(it.x - cand.center.x, it.y - cand.center.y) <= matchThreshold
                }
                if (alreadyInFused || alreadyRecovered) continue

                var occurrences = 0
                var sumX = 0f
                var sumY = 0f
                for (frame in scanFrames.dropLast(1)) {
                    val m = frame.pieces.firstOrNull { prev ->
                        prev.color == cand.color &&
                            hypot(prev.center.x - cand.center.x, prev.center.y - cand.center.y) <= matchThreshold
                    }
                    if (m != null) {
                        occurrences++
                        sumX += m.center.x
                        sumY += m.center.y
                    }
                }
                if (occurrences >= minConsensusFrames) {
                    val center = Point2D(sumX / occurrences, sumY / occurrences)
                    visitedExtra.add(center)
                    fused.add(CarromPiece(cand.color, center, cand.radiusPx))
                }
            }
        }

        return fused
    }

    /**
     * Combines all accumulated turn scan memory with the current striker position,
     * match calibration, scratch heatmap, and 2D physics rollout to choose the final shot.
     */
    fun solveCombinedFinalShot(
        board: CarromBoardLayout,
        latestPieces: List<CarromPiece>,
        striker: CarromPiece?,
        myCoinColor: CoinColor,
        screenWidth: Int,
        screenHeight: Int,
        shotMemory: CarromMemoryData?,
        usePhysicsRollout: Boolean,
        fallbackSolution: CarromShotSolution
    ): CarromShotSolution {
        val consensusPieces = buildConsensusPieces(latestPieces, board.coinRadiusPx)
        val combinedPieces = if (striker != null) consensusPieces + striker else consensusPieces

        return if (striker != null) {
            CarromGeometrySolver.solveBestShot(
                board = board,
                pieces = combinedPieces,
                myCoinColor = myCoinColor,
                screenWidth = screenWidth,
                screenHeight = screenHeight,
                existingStriker = striker,
                canRepositionStriker = false,
                shotMemory = shotMemory,
                usePhysicsRollout = usePhysicsRollout
            ) ?: fallbackSolution
        } else {
            fallbackSolution
        }
    }

    fun formatTurnTelemetry(nowMs: Long, sol: CarromShotSolution): String {
        val sec = elapsedMs(nowMs) / 1000f
        val status = if (sol.isSureShot) "SURE" else "BEST"
        return String.format(
            java.util.Locale.US,
            "[9s Clock %.1fs/9.0s | %d scans, %d slides | %s]",
            sec,
            scanCount.coerceAtLeast(1),
            sliderMovesThisTurn,
            status
        )
    }

    fun resetTurn() {
        turnStartMs = 0L
        scanFrames.clear()
        visitedStrikerXs.clear()
        localSliderGains.clear()
        _sliderMovesThisTurn = 0
        bestEvaluatedThisTurn = null
    }
}
