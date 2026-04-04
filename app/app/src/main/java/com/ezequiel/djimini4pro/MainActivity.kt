package com.ezequiel.djimini4pro

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.graphics.YuvImage
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import dji.sdk.keyvalue.key.BatteryKey
import dji.sdk.keyvalue.key.CameraKey
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.key.RemoteControllerKey
import dji.sdk.keyvalue.value.camera.CameraMode
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.common.register.DJISDKInitEvent
import dji.v5.et.create
import dji.v5.et.listen
import dji.v5.manager.KeyManager
import dji.v5.manager.SDKManager
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import dji.v5.manager.interfaces.SDKManagerCallback
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity(), TextureView.SurfaceTextureListener {

    private lateinit var textureView: TextureView
    private lateinit var statusText: TextView
    private lateinit var telemetryText: TextView
    private lateinit var photoButton: Button
    private lateinit var yoloButton: Button
    private lateinit var modelButton: Button
    private lateinit var roiButton: Button
    private lateinit var toggle4kButton: Button
    private lateinit var recordButton: Button
    private lateinit var rthButton: Button
    private lateinit var detectionOverlay: DetectionOverlay
    private var surface: Surface? = null
    private var surfaceWidth = 0
    private var surfaceHeight = 0
    private val cameraIndex = ComponentIndexType.LEFT_OR_MAIN
    private val mainHandler = Handler(Looper.getMainLooper())

    // ROI
    private var roiEnabled = false
    private var roiTop = 0.15f
    private var roiBottom = 0.80f

    // YOLO
    private var yoloDetector: YoloDetector? = null
    private var yoloEnabled = false
    private var yoloProcessing = false
    private var inferenceThread: HandlerThread? = null
    private var inferenceHandler: Handler? = null

    // Tracker
    private val vehicleTracker = VehicleTracker()

    // Recording
    private var record4kEnabled = false
    private var is4kRecording = false
    private var detectionDb: DetectionDatabase? = null
    private var isRecording = false
    private var currentFlightId: Long = -1
    private var frameNumber = 0
    private var uniqueVehicleCount = 0
    private lateinit var cropsDir: File
    private lateinit var flightCropsDir: File

    // Dashboard
    private var dashboardServer: DashboardServer? = null

    // Telemetry state
    private var batteryPercent = 0
    private var altitude = 0.0
    private var horizontalSpeed = 0.0
    private var verticalSpeed = 0.0
    private var satelliteCount = 0
    private var aircraftLat = 0.0
    private var aircraftLng = 0.0
    private var homeLat = 0.0
    private var homeLng = 0.0
    private var distanceToHome = 0f
    private var isFlying = false
    private var gimbalPitch = 0f
    private var gimbalYaw = 0f
    private var gimbalRoll = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textureView = findViewById(R.id.video_surface)
        statusText = findViewById(R.id.status_text)
        telemetryText = findViewById(R.id.telemetry_text)
        photoButton = findViewById(R.id.photo_button)
        yoloButton = findViewById(R.id.yolo_button)
        modelButton = findViewById(R.id.model_button)
        roiButton = findViewById(R.id.roi_button)
        toggle4kButton = findViewById(R.id.toggle_4k_button)
        recordButton = findViewById(R.id.record_button)
        rthButton = findViewById(R.id.rth_button)
        detectionOverlay = findViewById(R.id.detection_overlay)

        textureView.surfaceTextureListener = this

        photoButton.setOnClickListener { takePhoto() }
        photoButton.isEnabled = false
        yoloButton.setOnClickListener { toggleYolo() }
        modelButton.setOnClickListener { switchModel() }
        roiButton.setOnClickListener { showRoiDialog() }
        toggle4kButton.setOnClickListener { toggle4k() }
        recordButton.setOnClickListener { toggleRecording() }
        rthButton.setOnClickListener { returnToHome() }
        rthButton.isEnabled = false

        cropsDir = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "crops")
        cropsDir.mkdirs()
        detectionDb = DetectionDatabase(this)

        inferenceThread = HandlerThread("YoloInference").also { it.start() }
        inferenceHandler = Handler(inferenceThread!!.looper)

        // Start dashboard server
        try {
            dashboardServer = DashboardServer(8080)
            dashboardServer?.start()
            Log.i(DJIApp.TAG, "Dashboard server started on port 8080")
        } catch (e: Exception) {
            Log.e(DJIApp.TAG, "Dashboard server failed: ${e.message}")
        }

        updateModelButtonText()
        initSDK()
    }

    // ── YOLO toggle ──

    private fun toggleYolo() {
        yoloEnabled = !yoloEnabled
        if (yoloEnabled) {
            yoloButton.text = "YOLO: ON"
            if (yoloDetector == null) {
                updateStatus("Loading YOLO model...")
                inferenceHandler?.post {
                    yoloDetector = YoloDetector(this@MainActivity)
                    mainHandler.post {
                        updateStatus("YOLO ready: ${yoloDetector?.currentModel?.displayName}")
                        updateModelButtonText()
                    }
                }
            }
            if (DJIApp.isProductConnected) registerFrameListener()
        } else {
            yoloButton.text = "YOLO: OFF"
            unregisterFrameListener()
            detectionOverlay.clear()
        }
    }

    private fun switchModel() {
        val detector = yoloDetector ?: return
        val newModel = if (detector.currentModel == YoloModel.VISDRONE) YoloModel.COCO else YoloModel.VISDRONE
        updateStatus("Switching to ${newModel.displayName}...")
        inferenceHandler?.post {
            detector.loadModel(newModel)
            mainHandler.post {
                updateStatus("Model: ${newModel.displayName}")
                updateModelButtonText()
            }
        }
    }

    private fun updateModelButtonText() {
        val name = yoloDetector?.currentModel?.displayName ?: YoloModel.VISDRONE.displayName
        modelButton.text = if (name.contains("VisDrone")) "VisDrone" else "COCO"
    }

    // ── ROI ──

    private fun showRoiDialog() {
        val options = arrayOf(
            "OFF (full frame)",
            "15%-80% (recommended)",
            "10%-85% (wide)",
            "20%-75% (narrow)",
            "25%-70% (very narrow)"
        )
        val values = arrayOf(
            floatArrayOf(0f, 1f),
            floatArrayOf(0.15f, 0.80f),
            floatArrayOf(0.10f, 0.85f),
            floatArrayOf(0.20f, 0.75f),
            floatArrayOf(0.25f, 0.70f)
        )

        android.app.AlertDialog.Builder(this)
            .setTitle("Detection ROI")
            .setItems(options) { _, which ->
                val v = values[which]
                roiTop = v[0]
                roiBottom = v[1]
                roiEnabled = which > 0

                detectionOverlay.roiEnabled = roiEnabled
                detectionOverlay.roiTop = roiTop
                detectionOverlay.roiBottom = roiBottom
                detectionOverlay.postInvalidate()

                roiButton.text = if (roiEnabled) "ROI ●" else "ROI"
                val pct = if (roiEnabled) "${(roiTop * 100).toInt()}%-${(roiBottom * 100).toInt()}%" else "OFF"
                updateStatus("ROI: $pct")
                Log.i(DJIApp.TAG, "ROI set: $pct")
            }
            .show()
    }

    // ── 4K Toggle ──

    private fun toggle4k() {
        record4kEnabled = !record4kEnabled
        toggle4kButton.text = if (record4kEnabled) "4K: ON" else "4K: OFF"
        Log.i(DJIApp.TAG, "4K recording ${if (record4kEnabled) "enabled" else "disabled"}")
    }

    private fun start4kRecording() {
        val cameraModeKey = KeyTools.createKey(CameraKey.KeyCameraMode, cameraIndex)
        KeyManager.getInstance().setValue(cameraModeKey, CameraMode.VIDEO_NORMAL, object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                val startRecKey = KeyTools.createKey(CameraKey.KeyStartRecord, cameraIndex)
                KeyManager.getInstance().performAction(startRecKey, null,
                    object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                        override fun onSuccess(result: EmptyMsg?) {
                            is4kRecording = true
                            Log.i(DJIApp.TAG, "4K recording started")
                            updateStatus("4K recording to SD card")
                        }
                        override fun onFailure(error: IDJIError) {
                            is4kRecording = false
                            val desc = error.description()
                            Log.e(DJIApp.TAG, "4K record failed: $desc")
                            if (desc.contains("SD", ignoreCase = true) || desc.contains("card", ignoreCase = true) || desc.contains("storage", ignoreCase = true)) {
                                updateStatus("4K failed: no SD card inserted!")
                            } else {
                                updateStatus("4K failed: $desc")
                            }
                        }
                    })
            }
            override fun onFailure(error: IDJIError) {
                Log.e(DJIApp.TAG, "Set video mode failed: ${error.description()}")
                updateStatus("Video mode failed: ${error.description()}")
            }
        })
    }

    private fun stop4kRecording() {
        if (!is4kRecording) return
        val stopRecKey = KeyTools.createKey(CameraKey.KeyStopRecord, cameraIndex)
        KeyManager.getInstance().performAction(stopRecKey, null,
            object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(result: EmptyMsg?) {
                    is4kRecording = false
                    Log.i(DJIApp.TAG, "4K recording stopped")
                }
                override fun onFailure(error: IDJIError) {
                    Log.e(DJIApp.TAG, "4K record stop failed: ${error.description()}")
                }
            })
    }

    // ── Recording ──

    private fun toggleRecording() {
        if (!isRecording) startRecording() else stopRecording()
    }

    private fun startRecording() {
        if (!yoloEnabled) toggleYolo()
        val modelName = yoloDetector?.currentModel?.displayName ?: "unknown"
        currentFlightId = detectionDb?.startFlight(modelName) ?: -1
        frameNumber = 0
        uniqueVehicleCount = 0
        vehicleTracker.reset()
        isRecording = true

        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        flightCropsDir = File(cropsDir, "flight_${currentFlightId}_$ts")
        flightCropsDir.mkdirs()

        // Start 4K recording if enabled
        if (record4kEnabled) start4kRecording()

        dashboardServer?.isRecording = true
        dashboardServer?.flightId = currentFlightId

        recordButton.text = if (record4kEnabled) "REC ● 4K" else "REC ●"
        updateStatus("Recording flight #$currentFlightId${if (record4kEnabled) " + 4K" else ""}")
    }

    private fun stopRecording() {
        if (record4kEnabled) stop4kRecording()
        isRecording = false
        detectionDb?.endFlight(currentFlightId)
        dashboardServer?.isRecording = false
        recordButton.text = "REC"
        updateStatus("Flight #$currentFlightId ended. $uniqueVehicleCount unique vehicles.")
    }

    // ── Frame processing ──

    private fun registerFrameListener() {
        try {
            MediaDataCenter.getInstance().cameraStreamManager.addFrameListener(
                cameraIndex, ICameraStreamManager.FrameFormat.NV21, yoloFrameListener
            )
        } catch (e: Exception) {
            Log.e(DJIApp.TAG, "Frame listener error: ${e.message}")
        }
    }

    private fun unregisterFrameListener() {
        try {
            MediaDataCenter.getInstance().cameraStreamManager.removeFrameListener(yoloFrameListener)
        } catch (_: Exception) {}
    }

    private val yoloFrameListener = ICameraStreamManager.CameraFrameListener { data, offset, length, width, height, format ->
        if (!yoloEnabled || yoloProcessing || yoloDetector == null) return@CameraFrameListener
        yoloProcessing = true

        val frameData = data.copyOfRange(offset, offset + length)
        val frameW = width
        val frameH = height
        val currentFrameNum = frameNumber++

        // Snapshot telemetry
        val snapLat = aircraftLat
        val snapLng = aircraftLng
        val snapAlt = altitude
        val snapSpeedH = horizontalSpeed
        val snapSpeedV = verticalSpeed
        val snapPitch = gimbalPitch
        val snapYaw = gimbalYaw
        val snapRoll = gimbalRoll
        val snapSat = satelliteCount

        // Snapshot ROI settings for this frame
        val snapRoiEnabled = roiEnabled
        val snapRoiTop = roiTop
        val snapRoiBottom = roiBottom

        inferenceHandler?.post {
            try {
                val yuvImage = YuvImage(frameData, ImageFormat.NV21, frameW, frameH, null)
                val out = ByteArrayOutputStream()
                yuvImage.compressToJpeg(Rect(0, 0, frameW, frameH), 80, out)
                val fullBitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
                    ?: run { yoloProcessing = false; return@post }

                // Apply ROI: crop bitmap for inference, keep full for crops
                val roiYStart: Int
                val inferBitmap: Bitmap
                if (snapRoiEnabled) {
                    roiYStart = (fullBitmap.height * snapRoiTop).toInt()
                    val roiYEnd = (fullBitmap.height * snapRoiBottom).toInt()
                    val roiH = roiYEnd - roiYStart
                    inferBitmap = Bitmap.createBitmap(fullBitmap, 0, roiYStart, fullBitmap.width, roiH)
                } else {
                    roiYStart = 0
                    inferBitmap = fullBitmap
                }

                val rawDetections = yoloDetector?.detect(inferBitmap) ?: emptyList()

                // Remap detections back to full frame coordinates
                val detections = if (snapRoiEnabled && rawDetections.isNotEmpty()) {
                    val roiH = snapRoiBottom - snapRoiTop
                    rawDetections.map { det ->
                        Detection(
                            bbox = android.graphics.RectF(
                                det.bbox.left,
                                det.bbox.top * roiH + snapRoiTop,
                                det.bbox.right,
                                det.bbox.bottom * roiH + snapRoiTop
                            ),
                            confidence = det.confidence,
                            classId = det.classId,
                            className = det.className
                        )
                    }
                } else rawDetections

                if (inferBitmap !== fullBitmap) inferBitmap.recycle()
                val bitmap = fullBitmap  // use full frame for cropping

                // Update tracker
                val trackResult = vehicleTracker.update(detections, bitmap)

                // Persist exited tracks (best crop)
                if (isRecording) {
                    for (track in trackResult.exitedTracks) {
                        persistTrack(track, currentFrameNum,
                            snapLat, snapLng, snapAlt, snapSpeedH, snapSpeedV,
                            snapPitch, snapYaw, snapRoll, snapSat)
                    }
                }

                // Push new tracks to dashboard
                for (track in trackResult.newTracks) {
                    uniqueVehicleCount++
                    val base64 = DashboardServer.bitmapToBase64(track.bestBitmap)
                    dashboardServer?.pushDetection(DashboardServer.RecentDetection(
                        trackId = track.trackId,
                        className = track.className,
                        confidence = track.bestConfidence,
                        colorName = track.color?.name ?: "unknown",
                        timestamp = System.currentTimeMillis(),
                        cropBase64 = base64
                    ))
                }

                bitmap.recycle()

                // Update UI
                mainHandler.post {
                    detectionOverlay.updateDetections(detections)
                    val (total, active) = vehicleTracker.getStats()
                    dashboardServer?.totalUniqueVehicles = uniqueVehicleCount
                    dashboardServer?.activeTrackCount = active
                    dashboardServer?.frameNumber = currentFrameNum
                    if (isRecording) {
                        updateStatus("REC #$currentFlightId | $uniqueVehicleCount vehicles | $active active | f$currentFrameNum")
                    }
                }
            } catch (e: Exception) {
                Log.e(DJIApp.TAG, "YOLO error: ${e.message}")
            } finally {
                yoloProcessing = false
            }
        }
    }

    private fun persistTrack(
        track: Track, frameNum: Int,
        lat: Double, lng: Double, alt: Double,
        speedH: Double, speedV: Double,
        gPitch: Float, gYaw: Float, gRoll: Float, satCount: Int
    ) {
        // Save best crop
        var cropPath: String? = null
        track.bestBitmap?.let { crop ->
            val fileName = "t${track.trackId}_${track.className}_${track.color?.name ?: "unk"}.jpg"
            val cropFile = File(flightCropsDir, fileName)
            try {
                FileOutputStream(cropFile).use { fos ->
                    crop.compress(Bitmap.CompressFormat.JPEG, 92, fos)
                }
                cropPath = cropFile.absolutePath
            } catch (e: Exception) {
                Log.e(DJIApp.TAG, "Crop save error: ${e.message}")
            }
        }

        // Also push to dashboard as final detection with best crop
        val base64 = DashboardServer.bitmapToBase64(track.bestBitmap)
        dashboardServer?.pushDetection(DashboardServer.RecentDetection(
            trackId = track.trackId,
            className = track.className,
            confidence = track.bestConfidence,
            colorName = track.color?.name ?: "unknown",
            timestamp = System.currentTimeMillis(),
            cropBase64 = base64
        ))

        detectionDb?.insertDetection(
            flightId = currentFlightId,
            frameNumber = frameNum,
            classId = track.classId,
            className = track.className,
            confidence = track.bestConfidence,
            bboxX = track.bestBbox.left,
            bboxY = track.bestBbox.top,
            bboxW = track.bestBbox.width(),
            bboxH = track.bestBbox.height(),
            colorName = track.color?.name,
            colorH = track.color?.h,
            colorS = track.color?.s,
            colorV = track.color?.v,
            droneLat = lat, droneLng = lng, droneAlt = alt,
            droneSpeedH = speedH, droneSpeedV = speedV,
            gimbalPitch = gPitch, gimbalYaw = gYaw, gimbalRoll = gRoll,
            satelliteCount = satCount,
            cropPath = cropPath
        )

        track.bestBitmap?.recycle()
        track.bestBitmap = null
        track.persisted = true
    }

    // ── SDK init ──

    private fun initSDK() {
        updateStatus("Initializing SDK...")
        SDKManager.getInstance().init(this, object : SDKManagerCallback {
            override fun onRegisterSuccess() {
                DJIApp.isRegistered = true
                updateStatus("SDK registered. Connect drone via RC.")
            }
            override fun onRegisterFailure(error: IDJIError) {
                updateStatus("SDK registration failed: ${error.description()}")
            }
            override fun onProductConnect(productId: Int) {
                DJIApp.isProductConnected = true
                updateStatus("Drone connected.")
                runOnUiThread { photoButton.isEnabled = true; rthButton.isEnabled = true }
                listenForCamera()
                startTelemetry()
                listenRCButtons()
                if (yoloEnabled) registerFrameListener()
            }
            override fun onProductDisconnect(productId: Int) {
                DJIApp.isProductConnected = false
                stopVideoStream()
                if (isRecording) stopRecording()
                runOnUiThread { photoButton.isEnabled = false; rthButton.isEnabled = false; telemetryText.text = "" }
                detectionOverlay.clear()
                updateStatus("Drone disconnected.")
            }
            override fun onProductChanged(productId: Int) {}
            override fun onInitProcess(event: DJISDKInitEvent, totalProcess: Int) {
                if (event == DJISDKInitEvent.INITIALIZE_COMPLETE) SDKManager.getInstance().registerApp()
            }
            override fun onDatabaseDownloadProgress(current: Long, total: Long) {}
        })
    }

    // ── RC Trigger ──

    private fun listenRCButtons() {
        try {
            RemoteControllerKey.KeyShutterButtonDown.create().listen(this) { pressed ->
                if (pressed == true) mainHandler.post { takePhoto() }
            }
        } catch (e: Exception) {
            Log.w(DJIApp.TAG, "RC button listen failed: ${e.message}")
        }
    }

    // ── Telemetry ──

    private fun startTelemetry() {
        BatteryKey.KeyChargeRemainingInPercent.create().listen(this) { v ->
            v?.let { batteryPercent = it; updateTelemetryUI() }
        }
        FlightControllerKey.KeyAltitude.create().listen(this) { v ->
            v?.let { altitude = it; updateTelemetryUI() }
        }
        FlightControllerKey.KeyAircraftVelocity.create().listen(this) { v ->
            v?.let { horizontalSpeed = Math.sqrt(it.x * it.x + it.y * it.y); verticalSpeed = -it.z; updateTelemetryUI() }
        }
        FlightControllerKey.KeyGPSSatelliteCount.create().listen(this) { v ->
            v?.let { satelliteCount = it; updateTelemetryUI() }
        }
        FlightControllerKey.KeyAircraftLocation.create().listen(this) { v ->
            v?.let { aircraftLat = it.latitude; aircraftLng = it.longitude; computeDistance(); updateTelemetryUI() }
        }
        FlightControllerKey.KeyHomeLocation.create().listen(this) { v ->
            v?.let { homeLat = it.latitude; homeLng = it.longitude; computeDistance() }
        }
        FlightControllerKey.KeyIsFlying.create().listen(this) { v ->
            v?.let { isFlying = it; updateTelemetryUI() }
        }
        try {
            GimbalKey.KeyGimbalAttitude.create().listen(this) { v ->
                v?.let { gimbalPitch = it.pitch.toFloat(); gimbalYaw = it.yaw.toFloat(); gimbalRoll = it.roll.toFloat(); updateTelemetryUI() }
            }
        } catch (_: Exception) {}
    }

    private fun computeDistance() {
        if (homeLat != 0.0 && aircraftLat != 0.0) {
            val results = FloatArray(1)
            android.location.Location.distanceBetween(homeLat, homeLng, aircraftLat, aircraftLng, results)
            distanceToHome = results[0]
        }
    }

    private var lastTelemetryPush = 0L

    private fun updateTelemetryUI() {
        val batteryWarn = if (batteryPercent <= 20) "!" else ""
        val flyStatus = if (isFlying) "FLYING" else "GROUND"
        val text = """
            |BAT$batteryWarn ${batteryPercent}%
            |ALT  ${"%.1f".format(altitude)}m
            |DIST ${"%.0f".format(distanceToHome)}m
            |H.SP ${"%.1f".format(horizontalSpeed)}m/s
            |V.SP ${"%.1f".format(verticalSpeed)}m/s
            |SAT  $satelliteCount
            |GMB  ${"%.0f".format(gimbalPitch)}°
            |$flyStatus
        """.trimMargin()
        runOnUiThread { telemetryText.text = text }

        // Push to dashboard (throttled to ~2Hz)
        val now = System.currentTimeMillis()
        if (now - lastTelemetryPush > 500) {
            lastTelemetryPush = now
            dashboardServer?.let { srv ->
                srv.telemetry = org.json.JSONObject().apply {
                    put("battery", batteryPercent)
                    put("altitude", altitude)
                    put("distance", distanceToHome)
                    put("hSpeed", horizontalSpeed)
                    put("vSpeed", verticalSpeed)
                    put("gimbalPitch", gimbalPitch)
                    put("satellites", satelliteCount)
                    put("lat", aircraftLat)
                    put("lng", aircraftLng)
                }
                srv.modelName = yoloDetector?.currentModel?.displayName ?: ""
                srv.pushTelemetryUpdate()
            }
        }
    }

    // ── RTH / Photo ──

    private fun returnToHome() {
        updateStatus("Returning to home...")
        KeyManager.getInstance().performAction(
            KeyTools.createKey(FlightControllerKey.KeyStartGoHome), null,
            object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(t: EmptyMsg?) { updateStatus("RTH in progress...") }
                override fun onFailure(error: IDJIError) { updateStatus("RTH failed: ${error.description()}") }
            })
    }

    private fun takePhoto() {
        updateStatus("Taking photo...")
        val cameraModeKey = KeyTools.createKey(CameraKey.KeyCameraMode, cameraIndex)
        KeyManager.getInstance().setValue(cameraModeKey, CameraMode.PHOTO_NORMAL, object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                KeyManager.getInstance().performAction(
                    KeyTools.createKey(CameraKey.KeyStartShootPhoto, cameraIndex), null,
                    object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                        override fun onSuccess(result: EmptyMsg?) { updateStatus("Photo taken!") }
                        override fun onFailure(error: IDJIError) { updateStatus("Photo failed: ${error.description()}") }
                    })
            }
            override fun onFailure(error: IDJIError) { updateStatus("Photo mode failed: ${error.description()}") }
        })
    }

    // ── Video Stream ──

    private val cameraListener = object : ICameraStreamManager.AvailableCameraUpdatedListener {
        override fun onAvailableCameraUpdated(cameras: MutableList<ComponentIndexType>) {
            if (cameras.contains(ComponentIndexType.LEFT_OR_MAIN)) startVideoStream()
        }
        override fun onCameraStreamEnableUpdate(map: MutableMap<ComponentIndexType, Boolean>) {}
    }

    private fun listenForCamera() {
        try { MediaDataCenter.getInstance().cameraStreamManager.addAvailableCameraUpdatedListener(cameraListener) }
        catch (e: Exception) { Log.e(DJIApp.TAG, "Camera listener error: ${e.message}") }
    }

    private fun startVideoStream() {
        val s = surface ?: run { mainHandler.postDelayed({ startVideoStream() }, 1000); return }
        if (surfaceWidth <= 0 || surfaceHeight <= 0) return
        try {
            MediaDataCenter.getInstance().cameraStreamManager.putCameraStreamSurface(
                cameraIndex, s, surfaceWidth, surfaceHeight, ICameraStreamManager.ScaleType.CENTER_INSIDE)
        } catch (e: Exception) { Log.e(DJIApp.TAG, "Video stream error: ${e.message}") }
    }

    private fun stopVideoStream() {
        surface?.let { try { MediaDataCenter.getInstance().cameraStreamManager.removeCameraStreamSurface(it) } catch (_: Exception) {} }
    }

    private fun updateStatus(msg: String) { runOnUiThread { statusText.text = msg } }

    // ── TextureView ──

    override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
        surface = Surface(st); surfaceWidth = w; surfaceHeight = h
        if (DJIApp.isProductConnected) startVideoStream()
    }
    override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) { surfaceWidth = w; surfaceHeight = h }
    override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean { stopVideoStream(); surface?.release(); surface = null; return true }
    override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        if (isRecording) stopRecording()
        unregisterFrameListener()
        yoloDetector?.close()
        inferenceThread?.quitSafely()
        detectionDb?.close()
        dashboardServer?.stop()
        vehicleTracker.reset()
        try { KeyManager.getInstance().cancelListen(this) } catch (_: Exception) {}
        try { MediaDataCenter.getInstance().cameraStreamManager.removeAvailableCameraUpdatedListener(cameraListener) } catch (_: Exception) {}
        stopVideoStream()
        surface?.release()
    }
}
