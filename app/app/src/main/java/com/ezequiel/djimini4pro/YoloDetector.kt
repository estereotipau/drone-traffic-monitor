package com.ezequiel.djimini4pro

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import java.nio.FloatBuffer

data class Detection(
    val bbox: RectF,       // normalized 0..1
    val confidence: Float,
    val classId: Int,
    val className: String
)

enum class YoloModel(
    val fileName: String,
    val displayName: String,
    val classes: Array<String>
) {
    COCO(
        "yolov8n.onnx",
        "YOLOv8n (COCO)",
        arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        )
    ),
    VISDRONE(
        "yolov8n_visdrone.onnx",
        "YOLOv8n (VisDrone)",
        arrayOf(
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor"
        )
    );
}

class YoloDetector(private val context: Context) {

    companion object {
        private const val INPUT_SIZE = 640
        var confThreshold = 0.35f
        var iouThreshold = 0.45f
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    var currentModel: YoloModel = YoloModel.VISDRONE
        private set

    init {
        ortEnv = OrtEnvironment.getEnvironment()
        loadModel(YoloModel.VISDRONE)
    }

    fun loadModel(model: YoloModel) {
        try {
            ortSession?.close()
            val modelBytes = context.assets.open(model.fileName).readBytes()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
            }
            ortSession = ortEnv!!.createSession(modelBytes, sessionOptions)
            currentModel = model
            Log.i(DJIApp.TAG, "YOLO: Loaded ${model.displayName}")
        } catch (e: Exception) {
            Log.e(DJIApp.TAG, "YOLO: Failed to load ${model.displayName}: ${e.message}", e)
        }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val session = ortSession ?: return emptyList()
        val env = ortEnv ?: return emptyList()

        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // [1, 3, 640, 640] NCHW
        val buffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) buffer.put(((pixel shr 16) and 0xFF) / 255f)
        for (pixel in pixels) buffer.put(((pixel shr 8) and 0xFF) / 255f)
        for (pixel in pixels) buffer.put((pixel and 0xFF) / 255f)
        buffer.rewind()

        val inputTensor = OnnxTensor.createTensor(
            env, buffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        val results = session.run(mapOf("images" to inputTensor))
        val outputTensor = results[0] as OnnxTensor

        val numClasses = currentModel.classes.size
        // Output: [1, 4+numClasses, numCandidates]
        val output = (outputTensor.value as Array<*>)[0] as Array<FloatArray>
        val numCandidates = output[0].size

        inputTensor.close()
        results.close()

        val detections = mutableListOf<Detection>()
        for (j in 0 until numCandidates) {
            var maxConf = 0f
            var maxIdx = 0
            for (c in 0 until numClasses) {
                val conf = output[4 + c][j]
                if (conf > maxConf) {
                    maxConf = conf
                    maxIdx = c
                }
            }

            if (maxConf < confThreshold) continue

            val cx = output[0][j] / INPUT_SIZE
            val cy = output[1][j] / INPUT_SIZE
            val w = output[2][j] / INPUT_SIZE
            val h = output[3][j] / INPUT_SIZE

            detections.add(Detection(
                bbox = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                confidence = maxConf,
                classId = maxIdx,
                className = currentModel.classes.getOrElse(maxIdx) { "unknown" }
            ))
        }

        return nms(detections)
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val result = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best.bbox, it.bbox) > iouThreshold && best.classId == it.classId }
        }
        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)
        val interArea = maxOf(0f, interRight - interLeft) * maxOf(0f, interBottom - interTop)
        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        return interArea / (aArea + bArea - interArea + 1e-5f)
    }

    fun close() {
        ortSession?.close()
        ortEnv?.close()
    }
}
