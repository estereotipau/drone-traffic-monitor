package com.ezequiel.djimini4pro

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class UsbAttachActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val intent = Intent(this, MainActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
        startActivity(intent)
        finish()
    }
}
