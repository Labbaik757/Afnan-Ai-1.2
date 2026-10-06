package com.afnan.ai.data

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.min
import kotlin.random.Random
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay

/**
 * Retry a streaming block with exponential backoff.
 *
 * Mirrors `afnan_ai/control/client/reconnect.py` (ReconnectManager):
 * [ConnectionException] triggers backoff and retry;
 * [AuthenticationException] is re-raised immediately because bad
 * credentials never heal with retries. Coroutine cancellation is
 * never swallowed.
 */
class ReconnectManager(
    private val baseDelayMs: Long = 1_000,
    private val factor: Double = 2.0,
    private val maxDelayMs: Long = 60_000,
    private val jitter: Boolean = true,
    private val random: Random = Random.Default,
) {
    private val stopped = AtomicBoolean(false)
    private val attemptCount = AtomicInteger(0)

    /** Number of consecutive failed attempts in the current streak. */
    val attempts: Int get() = attemptCount.get()

    /**
     * Backoff delay for the current attempt streak: 1s, 2s, 4s, ...
     * capped at 60s, plus up to 25% jitter.
     */
    fun nextDelayMs(): Long {
        var delay = baseDelayMs
        repeat(attemptCount.get()) {
            delay = min((delay * factor).toLong(), maxDelayMs)
        }
        delay = min(delay, maxDelayMs)
        if (jitter && delay > 0) {
            delay += (random.nextDouble() * delay * 0.25).toLong()
        }
        return delay
    }

    /** Reset the attempt streak after a successful connection. */
    fun reset() {
        attemptCount.set(0)
    }

    /** Ask an in-progress [runStreaming] loop to stop. */
    fun stop() {
        stopped.set(true)
    }

    /**
     * Run [block] until it returns normally or [stop] is called.
     *
     * A [ConnectionException] from [block] increments the attempt
     * streak and retries after [nextDelayMs]. An
     * [AuthenticationException] ends the loop immediately. Any other
     * throwable (including [CancellationException]) propagates
     * untouched.
     */
    suspend fun runStreaming(block: suspend () -> Unit) {
        stopped.set(false)
        while (!stopped.get()) {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (e: AuthenticationException) {
                throw e
            } catch (e: ConnectionException) {
                attemptCount.incrementAndGet()
                val waitMs = nextDelayMs()
                try {
                    delay(waitMs)
                } catch (ce: CancellationException) {
                    throw ce
                }
                continue
            }
            // Clean return (or a stop() from inside block) ends the loop.
            reset()
            return
        }
    }
}
