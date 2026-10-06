package com.afnan.ai.data

import java.io.EOFException
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.Socket
import java.net.URL
import java.nio.charset.CodingErrorAction
import java.security.SecureRandom
import java.util.Base64
import java.util.concurrent.atomic.AtomicBoolean
import javax.net.ssl.SSLSocketFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

// ---------------------------------------------------------------------------
// RFC 6455 framing. Client-to-server frames MUST be masked; server frames
// are never masked (a masked server frame is a protocol error).
// ---------------------------------------------------------------------------

object WsOpcode {
    const val TEXT = 0x1
    const val BINARY = 0x2
    const val CLOSE = 0x8
    const val PING = 0x9
    const val PONG = 0xA
}

/** Maximum accepted WebSocket payload (1 MiB), mirroring the server. */
const val WS_MAX_PAYLOAD = 1_048_576

/** Close codes used by this client. */
object WsCloseCode {
    const val NORMAL = 1000
    const val PROTOCOL_ERROR = 1002
    const val BAD_UTF8 = 1007
    const val MESSAGE_TOO_BIG = 1009
}

/** One decoded WebSocket frame. */
data class WsFrame(val opcode: Int, val payload: ByteArray) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is WsFrame) return false
        return opcode == other.opcode && payload.contentEquals(other.payload)
    }

    override fun hashCode(): Int = 31 * opcode + payload.contentHashCode()
}

private val secureRandom = SecureRandom()

/**
 * Encode one client-to-server frame with a fresh random mask
 * (RFC 6455 section 5.3 — clients MUST mask).
 */
fun encodeClientFrame(opcode: Int, payload: ByteArray): ByteArray {
    val mask = ByteArray(4).also { secureRandom.nextBytes(it) }
    val masked = ByteArray(payload.size) { i ->
        (payload[i].toInt() xor mask[i % 4].toInt()).toByte()
    }
    val header = mutableListOf<Byte>()
    header.add((0x80 or (opcode and 0x0F)).toByte())
    val length = masked.size
    when {
        length < 126 -> header.add((0x80 or length).toByte())
        length < 65536 -> {
            header.add((0x80 or 126).toByte())
            header.add((length ushr 8).toByte())
            header.add((length and 0xFF).toByte())
        }
        else -> {
            header.add((0x80 or 127).toByte())
            val longLength = length.toLong()
            for (shift in 56 downTo 0 step 8) {
                header.add(
                    ((longLength ushr shift) and 0xFF).toByte()
                )
            }
        }
    }
    return header.toByteArray() + mask + masked
}

private fun readExactly(input: InputStream, count: Int): ByteArray? {
    val out = ByteArray(count)
    var offset = 0
    while (offset < count) {
        val read = try {
            input.read(out, offset, count - offset)
        } catch (e: IOException) {
            return null
        }
        if (read < 0) return null
        offset += read
    }
    return out
}

private val RESERVED_OPCODES = setOf(
    0x3, 0x4, 0x5, 0x6, 0x7, 0xB, 0xC, 0xD, 0xE, 0xF
)

private fun isValidUtf8(bytes: ByteArray): Boolean {
    return try {
        Charsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(java.nio.ByteBuffer.wrap(bytes))
        true
    } catch (e: Exception) {
        false
    }
}

/**
 * Decode one server-to-client frame from [input].
 *
 * Returns null on clean EOF. Throws [ProtocolException] on protocol
 * violations: masked server frame (`ws_masked_server`), reserved
 * opcode (`ws_bad_opcode`), oversized declared length
 * (`ws_frame_too_big`, rejected before allocation), fragmented data
 * message (`ws_fragmented`), invalid UTF-8 text (`ws_bad_utf8`).
 */
@Throws(ProtocolException::class)
fun decodeServerFrame(input: InputStream): WsFrame? {
    val head = readExactly(input, 2) ?: return null
    val byte1 = head[0].toInt() and 0xFF
    val byte2 = head[1].toInt() and 0xFF
    val fin = (byte1 and 0x80) != 0
    val opcode = byte1 and 0x0F
    val masked = (byte2 and 0x80) != 0
    var length = (byte2 and 0x7F).toLong()

    if (masked) {
        throw ProtocolException(
            "server sent a masked frame", "ws_masked_server"
        )
    }
    if (opcode in RESERVED_OPCODES) {
        throw ProtocolException("reserved opcode", "ws_bad_opcode")
    }
    if (opcode >= 0x8 && length > 125) {
        throw ProtocolException(
            "control frame too large", "ws_bad_opcode"
        )
    }
    length = when (length) {
        126L -> {
            val ext = readExactly(input, 2)
                ?: throw EOFException("truncated frame length")
            (((ext[0].toInt() and 0xFF).toLong() shl 8) or
                (ext[1].toInt() and 0xFF).toLong())
        }
        127L -> {
            val ext = readExactly(input, 8)
                ?: throw EOFException("truncated frame length")
            var value = 0L
            for (b in ext) {
                value = (value shl 8) or (b.toInt() and 0xFF).toLong()
            }
            if (value < 0) {
                throw ProtocolException(
                    "frame too big", "ws_frame_too_big"
                )
            }
            value
        }
        else -> length
    }
    if (length > WS_MAX_PAYLOAD) {
        throw ProtocolException("frame too big", "ws_frame_too_big")
    }
    if (!fin && opcode < 0x8) {
        throw ProtocolException(
            "fragmented messages not supported", "ws_fragmented"
        )
    }
    val payload = if (length == 0L) {
        ByteArray(0)
    } else {
        readExactly(input, length.toInt())
            ?: throw EOFException("truncated frame payload")
    }
    if (opcode == WsOpcode.TEXT && !isValidUtf8(payload)) {
        throw ProtocolException("invalid UTF-8", "ws_bad_utf8")
    }
    return WsFrame(opcode, payload)
}

private fun encodeCloseFrame(code: Int = WsCloseCode.NORMAL): ByteArray {
    val payload = byteArrayOf(
        (code ushr 8).toByte(), (code and 0xFF).toByte()
    )
    return encodeClientFrame(WsOpcode.CLOSE, payload)
}

// ---------------------------------------------------------------------------
// SSE — mirrors EventClient.stream_sse.
// ---------------------------------------------------------------------------

/**
 * Server-Sent Events subscription for `GET /v1/events`.
 *
 * Mirrors `afnan_ai/control/client/events.py` (`EventClient.stream_sse`).
 */
class EventStream(
    private val client: ControlPlaneClient,
) {
    private val stopped = AtomicBoolean(false)

    /** Ask an in-progress [streamEvents] call to stop. */
    fun stop() {
        stopped.set(true)
    }

    /**
     * Stream events until the server closes the stream, [stop] is
     * called, or the calling coroutine is cancelled.
     *
     * Parses `data:` lines into [ControlEvent] and invokes [onEvent]
     * for each; `: ` comment lines (keep-alives) are skipped.
     */
    suspend fun streamEvents(
        token: String,
        cursor: Long = 0,
        onEvent: (ControlEvent) -> Unit,
    ): Unit = withContext(Dispatchers.IO) {
        stopped.set(false)
        val url = URL("${client.baseUrl}/v1/events?cursor=$cursor")
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = client.timeoutMs
            // The server holds the stream open; bound each read.
            readTimeout = 65_000
            setRequestProperty("Accept", "text/event-stream")
            setRequestProperty("User-Agent", "AfnanAndroidClient/1.0")
            setRequestProperty("Authorization", "Bearer $token")
        }
        try {
            val status = connection.responseCode
            if (status == 401) {
                throw AuthenticationException("SSE authentication failed")
            }
            if (status !in 200..299) {
                throw ProtocolException(
                    "SSE stream rejected", "http_$status", status
                )
            }
            val reader = connection.inputStream.bufferedReader(
                Charsets.UTF_8
            )
            val dataLines = mutableListOf<String>()
            while (!stopped.get()) {
                ensureActive()
                val line = try {
                    reader.readLine()
                } catch (e: IOException) {
                    break
                } ?: break
                when {
                    line.isEmpty() -> {
                        if (dataLines.isNotEmpty()) {
                            val raw = dataLines.joinToString("\n")
                            dataLines.clear()
                            parseEvent(raw)?.let { event ->
                                ensureActive()
                                onEvent(event)
                            }
                        }
                    }
                    line.startsWith(":") -> {
                        // keep-alive comment — skip
                    }
                    line.startsWith("data:") -> {
                        dataLines.add(
                            line.removePrefix("data:").trimStart()
                        )
                    }
                    // Other SSE fields (event:, id:, retry:) are
                    // not used by this protocol — ignore.
                }
            }
        } catch (e: ProtocolException) {
            throw e
        } catch (e: IOException) {
            if (!stopped.get()) {
                throw ConnectionException(
                    "SSE stream failed: ${e.javaClass.simpleName}",
                    "sse_failed",
                )
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun parseEvent(raw: String): ControlEvent? {
        return try {
            val element = ControlJson.parseToJsonElement(raw)
            ControlJson.decodeFromJsonElement(
                ControlEvent.serializer(), element.jsonObject
            )
        } catch (e: Exception) {
            null
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocket — mirrors WebSocketConnection in events.py.
// ---------------------------------------------------------------------------

/**
 * One authenticated WebSocket session to the control plane.
 *
 * The RFC 6455 upgrade carries the token in the `Authorization`
 * header (never in the URL). Client frames are masked; incoming
 * frames go through [decodeServerFrame]. Ping/pong and the close
 * handshake are handled; the receive loop runs on a daemon thread
 * and dispatches to [onEvent]/[onResult]/[onError].
 */
class WsConnection(
    baseUrl: String,
    private val timeoutMs: Int = 30_000,
) {
    private val wsUrl: URL = URL(
        baseUrl.trimEnd('/')
            .replaceFirst("https://", "wss://")
            .replaceFirst("http://", "ws://")
    )

    /** Called for `{"type":"event","event":{...}}` frames. */
    var onEvent: ((ControlEvent) -> Unit)? = null

    /** Called for `{"type":"command_result","result":{...}}` frames. */
    var onResult: ((RemoteCommandResult) -> Unit)? = null

    /** Called for `{"type":"error",...}` frames. */
    var onError: ((String) -> Unit)? = null

    /**
     * Callbacks fire on the internal reader thread, not the main
     * thread — hop dispatchers in the UI layer as needed.
     */

    val isConnected = AtomicBoolean(false)

    private val sendLock = Any()
    private var socket: Socket? = null
    private val stopFlag = AtomicBoolean(false)
    private var readerThread: Thread? = null

    /**
     * Perform the TCP/TLS connect and the WebSocket upgrade.
     *
     * @throws AuthenticationException on HTTP 401
     * @throws ProtocolException when the upgrade is rejected
     * @throws ConnectionException on transport failures
     */
    suspend fun connect(token: String): Unit =
        withContext(Dispatchers.IO) {
            stopFlag.set(false)
            val secure = wsUrl.protocol == "wss"
            val host = wsUrl.host.ifEmpty { "127.0.0.1" }
            val port = wsUrl.port.takeIf { it > 0 }
                ?: if (secure) 443 else 80
            val raw = try {
                Socket(host, port).also {
                    it.soTimeout = timeoutMs
                }
            } catch (e: IOException) {
                throw ConnectionException(
                    "WebSocket TCP connect to $host:$port failed",
                    "ws_connect_failed",
                )
            }
            val sock: Socket = if (secure) {
                try {
                    val factory =
                        SSLSocketFactory.getDefault() as SSLSocketFactory
                    factory.createSocket(raw, host, port, true)
                        .also { it.soTimeout = timeoutMs }
                } catch (e: IOException) {
                    raw.closeQuietly()
                    throw ConnectionException(
                        "WebSocket TLS failed", "ws_tls_failed"
                    )
                }
            } else {
                raw
            }
            val key = ByteArray(16).also { secureRandom.nextBytes(it) }
                .let { Base64.getEncoder().encodeToString(it) }
            // Token travels in the Authorization header only.
            val request = buildString {
                append("GET /v1/stream HTTP/1.1\r\n")
                append("Host: $host:$port\r\n")
                append("Upgrade: websocket\r\n")
                append("Connection: Upgrade\r\n")
                append("Authorization: Bearer $token\r\n")
                append("Sec-WebSocket-Key: $key\r\n")
                append("Sec-WebSocket-Version: 13\r\n")
                append("\r\n")
            }
            try {
                sock.getOutputStream().write(
                    request.toByteArray(Charsets.UTF_8)
                )
                sock.getOutputStream().flush()
                val response = readHttpResponse(sock.getInputStream())
                val statusLine = response.lineSequence().firstOrNull()
                    .orEmpty()
                when {
                    statusLine.contains(" 101 ") -> Unit
                    statusLine.contains(" 401 ") -> {
                        sock.closeQuietly()
                        throw AuthenticationException(
                            "WebSocket authentication failed"
                        )
                    }
                    else -> {
                        sock.closeQuietly()
                        throw ProtocolException(
                            "WebSocket upgrade rejected",
                            "ws_upgrade_rejected",
                        )
                    }
                }
            } catch (e: ProtocolException) {
                sock.closeQuietly()
                throw e
            } catch (e: IOException) {
                sock.closeQuietly()
                throw ConnectionException(
                    "WebSocket handshake failed", "ws_handshake_failed"
                )
            }
            socket = sock
            isConnected.set(true)
            readerThread = Thread(::receiveLoop, "afnan-ws-receive").apply {
                isDaemon = true
                start()
            }
        }

    private fun readHttpResponse(input: InputStream): String {
        val buffer = StringBuilder()
        val one = ByteArray(1)
        while (true) {
            val read = try {
                input.read(one)
            } catch (e: IOException) {
                throw ConnectionException(
                    "WebSocket handshake failed", "ws_handshake_failed"
                )
            }
            if (read < 0) break
            buffer.append(one[0].toInt().toChar())
            if (buffer.endsWith("\r\n\r\n")) break
            if (buffer.length > 65_536) {
                throw ConnectionException(
                    "WebSocket handshake headers too large",
                    "ws_handshake_failed",
                )
            }
        }
        return buffer.toString()
    }

    // -- sending ----------------------------------------------------------

    private fun sendFrame(opcode: Int, payload: ByteArray) {
        val sock = synchronized(sendLock) { socket }
            ?: throw ConnectionException(
                "WebSocket is not connected", "ws_not_connected"
            )
        if (stopFlag.get()) {
            throw ConnectionException(
                "WebSocket is not connected", "ws_not_connected"
            )
        }
        try {
            synchronized(sendLock) {
                val out = sock.getOutputStream()
                out.write(encodeClientFrame(opcode, payload))
                out.flush()
            }
        } catch (e: IOException) {
            throw ConnectionException(
                "WebSocket send failed", "ws_send_failed"
            )
        }
    }

    /** Send a RemoteCommand body; the server answers `command_result`. */
    fun sendCommand(command: RemoteCommand) {
        val text = ControlJson.encodeToString(
            RemoteCommand.serializer(), command
        )
        sendFrame(WsOpcode.TEXT, text.toByteArray(Charsets.UTF_8))
    }

    /** Send the protocol heartbeat shortcut. */
    fun sendHeartbeat() {
        sendFrame(
            WsOpcode.TEXT, """{"type":"heartbeat"}""".toByteArray(
                Charsets.UTF_8
            )
        )
    }

    /** Send a WebSocket ping. */
    fun ping(payload: ByteArray = ByteArray(0)) {
        sendFrame(WsOpcode.PING, payload)
    }

    // -- receiving --------------------------------------------------------

    private fun receiveLoop() {
        val sock = socket ?: return
        try {
            while (!stopFlag.get()) {
                val frame = try {
                    decodeServerFrame(sock.getInputStream())
                } catch (e: ProtocolException) {
                    try {
                        synchronized(sendLock) {
                            sock.getOutputStream().write(
                                encodeCloseFrame(WsCloseCode.PROTOCOL_ERROR)
                            )
                        }
                    } catch (_: IOException) {
                    }
                    break
                } ?: break
                when (frame.opcode) {
                    WsOpcode.CLOSE -> {
                        try {
                            synchronized(sendLock) {
                                sock.getOutputStream().write(
                                    encodeCloseFrame(WsCloseCode.NORMAL)
                                )
                            }
                        } catch (_: IOException) {
                        }
                        break
                    }
                    WsOpcode.PING -> {
                        try {
                            sendFrame(WsOpcode.PONG, frame.payload)
                        } catch (_: ConnectionException) {
                            break
                        }
                    }
                    WsOpcode.PONG -> Unit
                    WsOpcode.TEXT -> dispatchText(frame.payload)
                    else -> Unit // binary frames are not part of the protocol
                }
            }
        } catch (e: IOException) {
            // Transport drop — teardown below.
        } finally {
            teardown()
        }
    }

    private fun dispatchText(payload: ByteArray) {
        val text = try {
            payload.toString(Charsets.UTF_8)
        } catch (e: Exception) {
            return
        }
        val message = try {
            ControlJson.parseToJsonElement(text).jsonObject
        } catch (e: Exception) {
            return
        }
        when (message["type"]?.jsonPrimitive?.contentOrNull) {
            "event" -> {
                val eventObj = message["event"]?.jsonObject ?: return
                val event = try {
                    ControlJson.decodeFromJsonElement(
                        ControlEvent.serializer(), eventObj
                    )
                } catch (e: Exception) {
                    return
                }
                onEvent?.invoke(event)
            }
            "command_result" -> {
                val resultObj = message["result"]?.jsonObject ?: return
                val result = try {
                    ControlJson.decodeFromJsonElement(
                        RemoteCommandResult.serializer(), resultObj
                    )
                } catch (e: Exception) {
                    return
                }
                onResult?.invoke(result)
            }
            "heartbeat_ack" -> Unit
            "error" -> {
                onError?.invoke(
                    message["error"]?.jsonPrimitive?.contentOrNull
                        .orEmpty()
                )
            }
        }
    }

    // -- teardown ----------------------------------------------------------

    private fun teardown() {
        isConnected.set(false)
        val sock = synchronized(sendLock) { socket }
        synchronized(sendLock) { socket = null }
        sock?.closeQuietly()
    }

    /** Send the close handshake and close the socket. */
    fun disconnect() {
        stopFlag.set(true)
        try {
            synchronized(sendLock) {
                socket?.getOutputStream()?.write(
                    encodeCloseFrame(WsCloseCode.NORMAL)
                )
            }
        } catch (_: IOException) {
        }
        val thread = readerThread
        readerThread = null
        teardown()
        if (thread != null && thread != Thread.currentThread()) {
            thread.join(5_000)
        }
    }
}

private fun Socket.closeQuietly() {
    try {
        close()
    } catch (_: IOException) {
    }
}
