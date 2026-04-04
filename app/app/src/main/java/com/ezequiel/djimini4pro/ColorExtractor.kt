package com.ezequiel.djimini4pro

import android.graphics.Bitmap
import android.graphics.Color

data class VehicleColor(
    val name: String,
    val h: Float,
    val s: Float,
    val v: Float
)

object ColorExtractor {

    fun extractDominantColor(crop: Bitmap): VehicleColor {
        // Sample the center 60% of the crop to avoid edges/background
        val marginX = (crop.width * 0.2f).toInt()
        val marginY = (crop.height * 0.2f).toInt()
        val startX = marginX.coerceAtLeast(0)
        val startY = marginY.coerceAtLeast(0)
        val endX = (crop.width - marginX).coerceAtLeast(startX + 1)
        val endY = (crop.height - marginY).coerceAtLeast(startY + 1)

        var totalH = 0f
        var totalS = 0f
        var totalV = 0f
        var count = 0

        val hsv = FloatArray(3)
        val step = 2 // sample every 2 pixels for speed

        for (y in startY until endY step step) {
            for (x in startX until endX step step) {
                val pixel = crop.getPixel(x, y)
                Color.colorToHSV(pixel, hsv)
                totalH += hsv[0]
                totalS += hsv[1]
                totalV += hsv[2]
                count++
            }
        }

        if (count == 0) return VehicleColor("unknown", 0f, 0f, 0f)

        val avgH = totalH / count
        val avgS = totalS / count
        val avgV = totalV / count

        val name = classifyColor(avgH, avgS, avgV)
        return VehicleColor(name, avgH, avgS, avgV)
    }

    private fun classifyColor(h: Float, s: Float, v: Float): String {
        // Low value = black
        if (v < 0.2f) return "black"
        // Low saturation + high value = white
        if (s < 0.15f && v > 0.7f) return "white"
        // Low saturation + medium value = gray/silver
        if (s < 0.15f) return if (v > 0.45f) "silver" else "gray"

        // Chromatic colors by hue
        return when {
            h < 15 || h >= 345 -> "red"
            h < 30 -> "orange"
            h < 55 -> "yellow"
            h < 85 -> "lime"
            h < 165 -> "green"
            h < 195 -> "cyan"
            h < 260 -> "blue"
            h < 290 -> "purple"
            h < 345 -> "pink"
            else -> "unknown"
        }
    }
}
