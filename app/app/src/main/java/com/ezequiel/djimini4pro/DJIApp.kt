package com.ezequiel.djimini4pro

import android.app.Application
import android.content.Context

class DJIApp : Application() {

    companion object {
        const val TAG = "DJIMini4Pro"
        var isRegistered = false
        var isProductConnected = false
    }

    override fun attachBaseContext(base: Context?) {
        super.attachBaseContext(base)
        com.cySdkyc.clx.Helper.install(this)
    }
}
