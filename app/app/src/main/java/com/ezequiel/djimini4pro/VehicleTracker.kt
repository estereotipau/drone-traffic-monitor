package com.ezequiel.djimini4pro

import android.graphics.Bitmap
import android.graphics.RectF
import kotlin.math.sqrt

data class Track(
    val trackId: Int,
    val className: String,
    val classId: Int,
    var centroidX: Float,
    var centroidY: Float,
    var bbox: RectF,
    var bestConfidence: Float,
    var bestBbox: RectF,
    var bestBitmap: Bitmap? = null,
    var color: VehicleColor? = null,
    var framesAlive: Int = 0,
    var framesMissed: Int = 0,
    var persisted: Boolean = false
)

class VehicleTracker(
    private val maxMissedFrames: Int = 8,
    private val matchDistanceThreshold: Float = 0.12f  // normalized coords
) {
    private var nextTrackId = 1
    private val activeTracks = mutableListOf<Track>()
    private val completedTracks = mutableListOf<Track>()

    data class TrackResult(
        val newTracks: List<Track>,          // just appeared
        val exitedTracks: List<Track>,       // just exited (with best crop)
        val activeTracks: List<Track>        // all currently active
    )

    fun update(detections: List<Detection>, frameBitmap: Bitmap?): TrackResult {
        val newTracks = mutableListOf<Track>()
        val matched = BooleanArray(activeTracks.size)
        val detMatched = BooleanArray(detections.size)

        // Match detections to existing tracks by centroid distance + same class
        for ((di, det) in detections.withIndex()) {
            val detCx = (det.bbox.left + det.bbox.right) / 2
            val detCy = (det.bbox.top + det.bbox.bottom) / 2

            var bestTrackIdx = -1
            var bestDist = Float.MAX_VALUE

            for ((ti, track) in activeTracks.withIndex()) {
                if (matched[ti]) continue
                if (track.className != det.className) continue

                val dx = detCx - track.centroidX
                val dy = detCy - track.centroidY
                val dist = sqrt(dx * dx + dy * dy)

                if (dist < bestDist && dist < matchDistanceThreshold) {
                    bestDist = dist
                    bestTrackIdx = ti
                }
            }

            if (bestTrackIdx >= 0) {
                // Update existing track
                val track = activeTracks[bestTrackIdx]
                matched[bestTrackIdx] = true
                detMatched[di] = true
                track.centroidX = detCx
                track.centroidY = detCy
                track.bbox = det.bbox
                track.framesAlive++
                track.framesMissed = 0

                // Keep best crop (highest confidence)
                if (det.confidence > track.bestConfidence) {
                    track.bestConfidence = det.confidence
                    track.bestBbox = RectF(det.bbox)
                    track.bestBitmap?.recycle()
                    track.bestBitmap = cropDetection(frameBitmap, det.bbox)
                    track.color = track.bestBitmap?.let { ColorExtractor.extractDominantColor(it) }
                }
            }
        }

        // Create new tracks for unmatched detections
        for ((di, det) in detections.withIndex()) {
            if (detMatched[di]) continue

            val detCx = (det.bbox.left + det.bbox.right) / 2
            val detCy = (det.bbox.top + det.bbox.bottom) / 2
            val crop = cropDetection(frameBitmap, det.bbox)
            val color = crop?.let { ColorExtractor.extractDominantColor(it) }

            val track = Track(
                trackId = nextTrackId++,
                className = det.className,
                classId = det.classId,
                centroidX = detCx,
                centroidY = detCy,
                bbox = det.bbox,
                bestConfidence = det.confidence,
                bestBbox = RectF(det.bbox),
                bestBitmap = crop,
                color = color,
                framesAlive = 1,
                framesMissed = 0
            )
            activeTracks.add(track)
            newTracks.add(track)
        }

        // Increment missed frames for unmatched tracks
        val exitedTracks = mutableListOf<Track>()
        val iterator = activeTracks.iterator()
        while (iterator.hasNext()) {
            val track = iterator.next()
            if (!matched[activeTracks.indexOf(track).coerceIn(0, matched.size - 1)]) {
                // Check if this track was matched (handle index carefully)
                val idx = activeTracks.indexOf(track)
                if (idx >= 0 && idx < matched.size && !matched[idx]) {
                    track.framesMissed++
                }
            }
            if (track.framesMissed >= maxMissedFrames) {
                exitedTracks.add(track)
                completedTracks.add(track)
                iterator.remove()
            }
        }

        return TrackResult(
            newTracks = newTracks,
            exitedTracks = exitedTracks,
            activeTracks = activeTracks.toList()
        )
    }

    private fun cropDetection(bitmap: Bitmap?, bbox: RectF): Bitmap? {
        bitmap ?: return null
        val left = (bbox.left * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
        val top = (bbox.top * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
        val right = (bbox.right * bitmap.width).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = (bbox.bottom * bitmap.height).toInt().coerceIn(top + 1, bitmap.height)
        val w = right - left
        val h = bottom - top
        if (w < 10 || h < 10) return null
        return Bitmap.createBitmap(bitmap, left, top, w, h)
    }

    fun reset() {
        activeTracks.forEach { it.bestBitmap?.recycle() }
        completedTracks.forEach { it.bestBitmap?.recycle() }
        activeTracks.clear()
        completedTracks.clear()
        nextTrackId = 1
    }

    fun getStats(): Pair<Int, Int> = Pair(nextTrackId - 1, activeTracks.size)
}
