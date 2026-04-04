package com.ezequiel.djimini4pro

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream

class DashboardServer(
    port: Int = 8080
) : NanoHTTPD(port) {

    // Shared state — updated from MainActivity
    var telemetry = JSONObject()
    var flightId: Long = -1
    var totalUniqueVehicles = 0
    var activeTrackCount = 0
    var modelName = ""
    var isRecording = false
    var frameNumber = 0

    // Ring buffer of recent detections for the 9-slot grid
    data class RecentDetection(
        val trackId: Int,
        val className: String,
        val confidence: Float,
        val colorName: String,
        val timestamp: Long,
        val cropBase64: String?
    )

    private val recentDetections = ArrayDeque<RecentDetection>(60)
    private val sseClients = mutableListOf<SSEClient>()

    fun pushDetection(detection: RecentDetection) {
        synchronized(recentDetections) {
            recentDetections.addLast(detection)
            while (recentDetections.size > 60) recentDetections.removeFirst()
        }
        // Push SSE event
        val json = JSONObject().apply {
            put("type", "detection")
            put("trackId", detection.trackId)
            put("className", detection.className)
            put("confidence", detection.confidence)
            put("colorName", detection.colorName)
            put("timestamp", detection.timestamp)
            put("cropBase64", detection.cropBase64 ?: "")
        }
        pushSSE(json.toString())
    }

    fun pushTelemetryUpdate() {
        val json = JSONObject().apply {
            put("type", "telemetry")
            put("data", telemetry)
            put("flightId", flightId)
            put("uniqueVehicles", totalUniqueVehicles)
            put("activeTracks", activeTrackCount)
            put("modelName", modelName)
            put("isRecording", isRecording)
            put("frameNumber", frameNumber)
        }
        pushSSE(json.toString())
    }

    private fun pushSSE(data: String) {
        synchronized(sseClients) {
            val dead = mutableListOf<SSEClient>()
            for (client in sseClients) {
                try {
                    client.send(data)
                } catch (e: Exception) {
                    dead.add(client)
                }
            }
            sseClients.removeAll(dead)
        }
    }

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri ?: "/"
        return when {
            uri == "/" -> serveHTML()
            uri == "/events" -> serveSSE(session)
            uri == "/api/state" -> serveState()
            uri.startsWith("/crops/") -> serveCropFile(uri)
            else -> newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Not found")
        }
    }

    private fun serveState(): Response {
        val state = JSONObject().apply {
            put("telemetry", telemetry)
            put("flightId", flightId)
            put("uniqueVehicles", totalUniqueVehicles)
            put("activeTracks", activeTrackCount)
            put("modelName", modelName)
            put("isRecording", isRecording)
            put("frameNumber", frameNumber)
            val arr = JSONArray()
            synchronized(recentDetections) {
                for (d in recentDetections) {
                    arr.put(JSONObject().apply {
                        put("trackId", d.trackId)
                        put("className", d.className)
                        put("confidence", d.confidence)
                        put("colorName", d.colorName)
                        put("timestamp", d.timestamp)
                        put("cropBase64", d.cropBase64 ?: "")
                    })
                }
            }
            put("recentDetections", arr)
        }
        return newFixedLengthResponse(Response.Status.OK, "application/json", state.toString())
    }

    private fun serveCropFile(uri: String): Response {
        val path = uri.removePrefix("/crops/")
        val file = File(path)
        return if (file.exists()) {
            val fis = FileInputStream(file)
            newFixedLengthResponse(Response.Status.OK, "image/jpeg", fis, file.length())
        } else {
            newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Not found")
        }
    }

    // ── SSE ──

    class SSEClient(private val session: IHTTPSession) {
        private val outputStream = session.javaClass.getDeclaredField("outputStream").apply {
            isAccessible = true
        }.get(session) as java.io.OutputStream

        fun send(data: String) {
            val msg = "data: $data\n\n"
            outputStream.write(msg.toByteArray())
            outputStream.flush()
        }

        fun sendHeader() {
            val header = "HTTP/1.1 200 OK\r\n" +
                "Content-Type: text/event-stream\r\n" +
                "Cache-Control: no-cache\r\n" +
                "Connection: keep-alive\r\n" +
                "Access-Control-Allow-Origin: *\r\n\r\n"
            outputStream.write(header.toByteArray())
            outputStream.flush()
        }
    }

    private fun serveSSE(session: IHTTPSession): Response {
        val client = SSEClient(session)
        client.sendHeader()
        synchronized(sseClients) { sseClients.add(client) }
        // Block this thread to keep connection alive
        try {
            while (true) Thread.sleep(30000)
        } catch (_: Exception) {}
        synchronized(sseClients) { sseClients.remove(client) }
        return newFixedLengthResponse("")
    }

    // ── Dashboard HTML ──

    private fun serveHTML(): Response {
        val html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DJI Mini 4 Pro - Detection Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; height: 100vh; }
.header { background: #16213e; padding: 8px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0f3460; height: 42px; }
.header h1 { font-size: 16px; color: #e94560; }
.status-badge { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; }
.recording { background: #e94560; animation: pulse 1s infinite; }
.idle { background: #555; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

.dashboard { display: grid; grid-template-columns: 200px 1fr; gap: 10px; padding: 10px; height: calc(100vh - 42px); }

.sidebar { display: flex; flex-direction: column; gap: 10px; overflow: hidden; }
.telemetry { background: #16213e; border-radius: 6px; padding: 10px; }
.telemetry h2 { font-size: 11px; color: #53d8fb; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
.telem-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 12px; }
.telem-label { color: #666; }
.telem-value { font-family: monospace; color: #53d8fb; font-size: 12px; }
.telem-value.warn { color: #e94560; }

.stats { background: #16213e; border-radius: 6px; padding: 10px; text-align: center; }
.stat-big { font-size: 32px; font-weight: bold; color: #53d8fb; }
.stat-label { font-size: 10px; color: #888; }
.stat-row { margin-top: 6px; font-size: 12px; }

.grid-container { display: flex; flex-direction: column; overflow: hidden; }
.grid-title { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.detection-grid { display: grid; grid-template-columns: repeat(7, 1fr); grid-template-rows: repeat(7, 1fr); gap: 3px; flex: 1; min-height: 0; }
.slot { background: #16213e; border-radius: 4px; overflow: hidden; display: flex; flex-direction: column; border: 1px solid transparent; min-height: 0; }
.slot.active { border-color: #53d8fb; }
.slot.fresh { border-color: #e94560; animation: fadeIn 0.3s; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.slot .img-wrap { flex: 1; min-height: 0; overflow: hidden; background: #0a0a1a; display: flex; align-items: center; justify-content: center; }
.slot img { max-width: 100%; max-height: 100%; object-fit: contain; }
.slot .label { padding: 2px 4px; font-size: 9px; background: #0f3460; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
.slot .label .cls { font-weight: bold; color: #53d8fb; }
.slot .label .cdot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.slot .label .conf { color: #888; }
.slot.empty { display: flex; align-items: center; justify-content: center; color: #222; font-size: 14px; }
.conn { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-left: 8px; }
.conn.ok { background: #2ecc71; }
.conn.err { background: #e74c3c; }
</style>
</head>
<body>
<div class="header">
    <h1>DJI Mini 4 Pro — Dashboard</h1>
    <div style="display:flex;align-items:center;">
        <span id="model-name" style="color:#888;font-size:11px;margin-right:10px;">—</span>
        <span id="rec-badge" class="status-badge idle">IDLE</span>
        <span id="conn-dot" class="conn err" title="SSE connection"></span>
    </div>
</div>
<div class="dashboard">
    <div class="sidebar">
        <div class="telemetry">
            <h2>Telemetry</h2>
            <div class="telem-row"><span class="telem-label">BAT</span><span class="telem-value" id="t-bat">—</span></div>
            <div class="telem-row"><span class="telem-label">ALT</span><span class="telem-value" id="t-alt">—</span></div>
            <div class="telem-row"><span class="telem-label">DIST</span><span class="telem-value" id="t-dist">—</span></div>
            <div class="telem-row"><span class="telem-label">H.SPD</span><span class="telem-value" id="t-hsp">—</span></div>
            <div class="telem-row"><span class="telem-label">GMB</span><span class="telem-value" id="t-gmb">—</span></div>
            <div class="telem-row"><span class="telem-label">SAT</span><span class="telem-value" id="t-sat">—</span></div>
            <div class="telem-row"><span class="telem-label">GPS</span><span class="telem-value" id="t-gps" style="font-size:10px">—</span></div>
            <div class="telem-row"><span class="telem-label">FRM</span><span class="telem-value" id="t-frame">—</span></div>
        </div>
        <div class="stats">
            <h2 style="font-size:11px;color:#53d8fb;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Detections</h2>
            <div class="stat-big" id="s-total">0</div>
            <div class="stat-label">unique vehicles</div>
            <div class="stat-row">
                <span class="telem-value" id="s-active">0</span>
                <span style="color:#888;font-size:10px;"> active tracks</span>
            </div>
        </div>
    </div>
    <div class="grid-container">
        <div class="grid-title">Recent Detections</div>
        <div class="detection-grid" id="grid"></div>
    </div>
</div>

<script>
const grid = document.getElementById('grid');
const SLOTS = 49;
const slots = [];
let slotIdx = 0;

for (let i = 0; i < SLOTS; i++) {
    const div = document.createElement('div');
    div.className = 'slot empty';
    div.innerHTML = '—';
    grid.appendChild(div);
    slots.push({ el: div, timer: null });
}

const colorMap = {
    red:'#e74c3c', orange:'#e67e22', yellow:'#f1c40f', lime:'#a0d911',
    green:'#2ecc71', cyan:'#00bcd4', blue:'#3498db', purple:'#9b59b6',
    pink:'#e91e63', white:'#ecf0f1', silver:'#bdc3c7', gray:'#7f8c8d',
    black:'#2c3e50', unknown:'#555'
};

function updateSlot(det) {
    const s = slots[slotIdx % SLOTS];
    if (s.timer) clearTimeout(s.timer);
    const colorHex = colorMap[det.colorName] || '#555';
    const imgSrc = det.cropBase64 ? 'data:image/jpeg;base64,' + det.cropBase64 : '';
    s.el.className = 'slot fresh';
    s.el.innerHTML =
        '<div class="img-wrap">' +
        (imgSrc ? '<img src="' + imgSrc + '">' : '') +
        '</div>' +
        '<div class="label">' +
            '<span class="cls">' + det.className + '</span>' +
            '<span class="cdot" style="background:' + colorHex + '"></span>' +
            '<span class="conf">' + Math.round(det.confidence * 100) + '%</span>' +
        '</div>';
    setTimeout(() => { s.el.className = 'slot active'; }, 400);
    s.timer = setTimeout(() => { s.el.className = 'slot'; }, 10000);
    slotIdx++;
}

function updateTelemetry(data) {
    const t = data.telemetry || data.data || {};
    const el = (id) => document.getElementById(id);
    const bat = t.battery != null ? t.battery : 0;
    el('t-bat').textContent = bat + '%';
    el('t-bat').className = 'telem-value' + (bat <= 20 ? ' warn' : '');
    el('t-alt').textContent = (t.altitude != null ? Number(t.altitude).toFixed(1) : '0.0') + 'm';
    el('t-dist').textContent = (t.distance != null ? Number(t.distance).toFixed(0) : '0') + 'm';
    el('t-hsp').textContent = (t.hSpeed != null ? Number(t.hSpeed).toFixed(1) : '0.0') + 'm/s';
    el('t-gmb').textContent = (t.gimbalPitch != null ? Number(t.gimbalPitch).toFixed(0) : '0') + '\u00B0';
    el('t-sat').textContent = t.satellites != null ? t.satellites : '0';
    el('t-gps').textContent = (t.lat != null ? Number(t.lat).toFixed(5) : '0') + ', ' + (t.lng != null ? Number(t.lng).toFixed(5) : '0');
    el('t-frame').textContent = data.frameNumber != null ? data.frameNumber : '0';
    el('s-total').textContent = data.uniqueVehicles != null ? data.uniqueVehicles : '0';
    el('s-active').textContent = data.activeTracks != null ? data.activeTracks : '0';
    el('model-name').textContent = data.modelName || '—';
    const badge = el('rec-badge');
    if (data.isRecording) {
        badge.textContent = 'REC #' + data.flightId;
        badge.className = 'status-badge recording';
    } else {
        badge.textContent = 'IDLE';
        badge.className = 'status-badge idle';
    }
}

// SSE
let sseOk = false;
function connectSSE() {
    const es = new EventSource('/events');
    es.onopen = () => { sseOk = true; document.getElementById('conn-dot').className = 'conn ok'; };
    es.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            if (msg.type === 'detection') updateSlot(msg);
            else if (msg.type === 'telemetry') updateTelemetry(msg);
        } catch(err) { console.error('SSE parse error:', err); }
    };
    es.onerror = () => {
        sseOk = false;
        document.getElementById('conn-dot').className = 'conn err';
        es.close();
        setTimeout(connectSSE, 3000);
    };
}

// Polling fallback (runs always, catches what SSE misses)
function poll() {
    fetch('/api/state').then(r => r.json()).then(state => {
        updateTelemetry(state);
        if (!sseOk) {
            (state.recentDetections || []).slice(-SLOTS).forEach(updateSlot);
        }
    }).catch(() => {});
}

connectSSE();
poll();
setInterval(poll, 2000);
</script>
</body>
</html>
""".trimIndent()
        return newFixedLengthResponse(Response.Status.OK, "text/html", html)
    }

    companion object {
        fun bitmapToBase64(bitmap: Bitmap?, maxSize: Int = 150): String? {
            bitmap ?: return null
            val scaled = Bitmap.createScaledBitmap(bitmap, maxSize,
                (maxSize * bitmap.height.toFloat() / bitmap.width).toInt(), true)
            val baos = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, 70, baos)
            scaled.recycle()
            return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
        }
    }
}
