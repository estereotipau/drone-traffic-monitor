package com.ezequiel.djimini4pro

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class DetectionDatabase(context: Context) : SQLiteOpenHelper(context, DB_NAME, null, DB_VERSION) {

    companion object {
        private const val DB_NAME = "detections.db"
        private const val DB_VERSION = 1
        const val TABLE_FLIGHTS = "flights"
        const val TABLE_DETECTIONS = "detections"
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE $TABLE_FLIGHTS (
                flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                model_name TEXT NOT NULL,
                notes TEXT
            )
        """)

        db.execSQL("""
            CREATE TABLE $TABLE_DETECTIONS (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_w REAL NOT NULL,
                bbox_h REAL NOT NULL,
                color_name TEXT,
                color_h REAL,
                color_s REAL,
                color_v REAL,
                drone_lat REAL,
                drone_lng REAL,
                drone_alt REAL,
                drone_speed_h REAL,
                drone_speed_v REAL,
                gimbal_pitch REAL,
                gimbal_yaw REAL,
                gimbal_roll REAL,
                satellite_count INTEGER,
                crop_path TEXT,
                FOREIGN KEY (flight_id) REFERENCES $TABLE_FLIGHTS(flight_id)
            )
        """)

        db.execSQL("CREATE INDEX idx_det_flight ON $TABLE_DETECTIONS(flight_id)")
        db.execSQL("CREATE INDEX idx_det_class ON $TABLE_DETECTIONS(class_name)")
    }

    override fun onUpgrade(db: SQLiteDatabase, old: Int, new: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_DETECTIONS")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_FLIGHTS")
        onCreate(db)
    }

    fun startFlight(modelName: String): Long {
        val values = ContentValues().apply {
            put("start_time", System.currentTimeMillis().toString())
            put("model_name", modelName)
        }
        return writableDatabase.insert(TABLE_FLIGHTS, null, values)
    }

    fun endFlight(flightId: Long) {
        val values = ContentValues().apply {
            put("end_time", System.currentTimeMillis().toString())
        }
        writableDatabase.update(TABLE_FLIGHTS, values, "flight_id = ?", arrayOf(flightId.toString()))
    }

    fun insertDetection(
        flightId: Long,
        frameNumber: Int,
        classId: Int,
        className: String,
        confidence: Float,
        bboxX: Float, bboxY: Float, bboxW: Float, bboxH: Float,
        colorName: String?, colorH: Float?, colorS: Float?, colorV: Float?,
        droneLat: Double, droneLng: Double, droneAlt: Double,
        droneSpeedH: Double, droneSpeedV: Double,
        gimbalPitch: Float, gimbalYaw: Float, gimbalRoll: Float,
        satelliteCount: Int,
        cropPath: String?
    ): Long {
        val values = ContentValues().apply {
            put("flight_id", flightId)
            put("timestamp", System.currentTimeMillis().toString())
            put("frame_number", frameNumber)
            put("class_id", classId)
            put("class_name", className)
            put("confidence", confidence)
            put("bbox_x", bboxX)
            put("bbox_y", bboxY)
            put("bbox_w", bboxW)
            put("bbox_h", bboxH)
            put("color_name", colorName)
            put("color_h", colorH)
            put("color_s", colorS)
            put("color_v", colorV)
            put("drone_lat", droneLat)
            put("drone_lng", droneLng)
            put("drone_alt", droneAlt)
            put("drone_speed_h", droneSpeedH)
            put("drone_speed_v", droneSpeedV)
            put("gimbal_pitch", gimbalPitch)
            put("gimbal_yaw", gimbalYaw)
            put("gimbal_roll", gimbalRoll)
            put("satellite_count", satelliteCount)
            put("crop_path", cropPath)
        }
        return writableDatabase.insert(TABLE_DETECTIONS, null, values)
    }

    fun getDetectionCount(flightId: Long): Int {
        val cursor = readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM $TABLE_DETECTIONS WHERE flight_id = ?",
            arrayOf(flightId.toString())
        )
        cursor.moveToFirst()
        val count = cursor.getInt(0)
        cursor.close()
        return count
    }
}
