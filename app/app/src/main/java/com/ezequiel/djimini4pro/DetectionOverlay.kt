package com.ezequiel.djimini4pro

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class DetectionOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()

    // ROI boundaries (0..1 normalized)
    var roiTop: Float = 0f
    var roiBottom: Float = 1f
    var roiEnabled: Boolean = false

    private val roiPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(100, 0, 0, 0)  // semi-transparent black
    }

    private val roiLinePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = Color.rgb(255, 255, 0)  // yellow
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(10f, 10f), 0f)
    }

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
        isFakeBoldText = true
    }

    // Color palette for different classes
    private val colors = intArrayOf(
        Color.rgb(255, 56, 56),   // red
        Color.rgb(255, 157, 151), // salmon
        Color.rgb(255, 112, 31),  // orange
        Color.rgb(255, 178, 29),  // yellow
        Color.rgb(207, 210, 49),  // lime
        Color.rgb(72, 249, 10),   // green
        Color.rgb(146, 204, 23),  // olive
        Color.rgb(61, 219, 134),  // teal
        Color.rgb(26, 147, 52),   // dark green
        Color.rgb(0, 212, 187),   // cyan
    )

    fun updateDetections(newDetections: List<Detection>) {
        detections = newDetections
        postInvalidate()
    }

    fun clear() {
        detections = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        // Draw ROI overlay (dim excluded zones)
        if (roiEnabled) {
            val topY = roiTop * h
            val bottomY = roiBottom * h
            canvas.drawRect(0f, 0f, w, topY, roiPaint)
            canvas.drawRect(0f, bottomY, w, h, roiPaint)
            canvas.drawLine(0f, topY, w, topY, roiLinePaint)
            canvas.drawLine(0f, bottomY, w, bottomY, roiLinePaint)
        }

        for (det in detections) {
            val color = colors[det.classId % colors.size]
            boxPaint.color = color
            textBgPaint.color = color

            // Scale normalized bbox to view size
            val rect = RectF(
                det.bbox.left * w,
                det.bbox.top * h,
                det.bbox.right * w,
                det.bbox.bottom * h
            )

            // Draw bounding box
            canvas.drawRect(rect, boxPaint)

            // Draw label background + text
            val label = "${det.className} ${"%.0f".format(det.confidence * 100)}%"
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize

            canvas.drawRect(
                rect.left, rect.top - textHeight - 8,
                rect.left + textWidth + 16, rect.top,
                textBgPaint
            )
            canvas.drawText(label, rect.left + 8, rect.top - 6, textPaint)
        }
    }
}
