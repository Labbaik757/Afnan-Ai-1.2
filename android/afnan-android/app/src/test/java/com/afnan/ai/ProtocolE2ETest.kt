package com.afnan.ai

import com.afnan.ai.data.CommandRepository
import com.afnan.ai.data.ControlEvent
import com.afnan.ai.data.ControlJson
import com.afnan.ai.data.ControlPlaneClient
import com.afnan.ai.data.EventStream
import com.afnan.ai.data.PairingRepository
import com.afnan.ai.data.RemoteCommandResult
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * End-to-end test of the Kotlin client against a REAL Python control
 * plane (`~/android-build/e2e_server.py`, real SecurityCenter /
 * TaskManager / GoalManager / ActivityCenter, auto-approving pairer).
 *
 * Runs only when the live server is explicitly requested:
 *   AFNAN_ANDROID_E2E=1 python3 ~/android-build/e2e_server.py
 *   AFNAN_ANDROID_E2E=1 ./gradlew :app:testDebugUnitTest
 *
 * Without the flag the test is skipped so a plain
 * `testDebugUnitTest` passes on a clean machine.
 */
class ProtocolE2ETest {

    private val baseUrl = "http://127.0.0.1:8765"

    private fun completed(result: RemoteCommandResult): JsonObject {
        assertEquals(
            "command ${result.commandId}/${result.status}: ${result.error}",
            "completed",
            result.status,
        )
        return result.result
    }

    @Test
    fun `pair approve redeem then drive real tasks`() = runBlocking {
        assumeTrue(
            "Set AFNAN_ANDROID_E2E=1 with ~/android-build/e2e_server.py running",
            System.getenv("AFNAN_ANDROID_E2E") == "1",
        )
        val client = ControlPlaneClient(baseUrl)
        val pairing = PairingRepository(client)

        // 1. Pairing request -> server, auto-approved by the test server.
        val (pairingId, code) = pairing.requestPairing(
            "e2e-android-1", "E2E Test Phone"
        )
        assertTrue(pairingId.isNotEmpty())
        assertTrue(code.length >= 4)

        // 2. Redeem (the test server auto-approves within ~1s; retry gently
        // to stay under the pairing rate limiter).
        var token = ""
        var sessionId = ""
        var lastErr: Exception? = null
        Thread.sleep(2000)
        repeat(5) {
            try {
                val r = pairing.redeem(pairingId, code)
                token = r.token
                sessionId = r.sessionId
                lastErr = null
                return@repeat
            } catch (e: Exception) {
                lastErr = e
                Thread.sleep(3000)
            }
        }
        assertTrue("redeem failed: $lastErr", token.isNotEmpty())
        assertTrue(sessionId.isNotEmpty())

        val commands = CommandRepository(client)

        // 3. list_tasks reflects the real server state.
        var tasks = completed(
            commands.execute(token, "list_tasks")
        )
        val initialCount = Regex("\"task_id\"").findAll(
            tasks["tasks"].toString()
        ).count()

        // 4. Start the SSE stream FIRST, then trigger a mutating command
        // so the stream has a real event to deliver.
        val stream = EventStream(client)
        val latch = CountDownLatch(1)
        val seen = mutableListOf<ControlEvent>()
        val job = thread(isDaemon = true) {
            runBlocking {
                try {
                    stream.streamEvents(token, 0) { ev ->
                        synchronized(seen) { seen.add(ev) }
                        latch.countDown()
                    }
                } catch (_: Exception) {
                }
            }
        }
        Thread.sleep(1500) // let the stream connect

        // 5. start_task goes through the real TaskManager.
        val goalText = "e2e verify android client ${System.currentTimeMillis()}"
        val created = completed(
            commands.execute(
                token, "start_task",
                buildJsonObject { put("goal_text", goalText) },
            )
        )
        val taskObj = created["task"]?.toString() ?: ""
        val taskId = Regex("\"task_id\"\\s*:\\s*\"([^\"]+)\"")
            .find(taskObj)?.groupValues?.get(1) ?: ""
        assertTrue("no task_id in $created", taskId.isNotEmpty())

        // 6. The SSE stream must deliver the task event.
        val gotEvent = latch.await(20, TimeUnit.SECONDS)
        stream.stop()
        job.join(2000)
        assertTrue(
            "no SSE events in 20s (saw ${seen.size})",
            gotEvent || seen.isNotEmpty(),
        )

        // 7. list_tasks now shows the real task with our goal text.
        tasks = completed(commands.execute(token, "list_tasks"))
        val listed = tasks["tasks"].toString()
        val finalCount = Regex("\"task_id\"").findAll(listed).count()
        assertEquals(initialCount + 1, finalCount)
        assertTrue("task $taskId missing from $listed", listed.contains(taskId))
        assertTrue(
            "goal text missing from $listed", listed.contains("e2e verify")
        )

        // 8. Agent status reflects a live runtime.
        val status = completed(commands.execute(token, "get_agent_status"))
        assertTrue("status missing: $status", status.toString().isNotEmpty())

        // 9. Heartbeat keeps the session alive.
        val hb = client.post(
            "/v1/session/heartbeat",
            buildJsonObject { put("session_id", sessionId) },
            token,
        )
        assertEquals(
            sessionId, hb["session_id"].toString().trim('"')
        )

        client.close()
    }
}
