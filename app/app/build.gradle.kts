plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.ezequiel.djimini4pro"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.ezequiel.djimini4pro"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += "arm64-v8a"
        }

        manifestPlaceholders["API_KEY"] = project.property("AIRCRAFT_API_KEY") as String
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
        freeCompilerArgs += "-Xjvm-default=all"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    androidResources {
        noCompress += "onnx"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += "lib/arm64-v8a/libc++_shared.so"
            pickFirsts += "lib/armeabi-v7a/libc++_shared.so"
        }
        resources {
            pickFirsts += "lib/arm64-v8a/libc++_shared.so"
            pickFirsts += "lib/armeabi-v7a/libc++_shared.so"
        }
        jniLibs.keepDebugSymbols += listOf(
            "*/*/libdjisdk_jni.so",
            "*/*/libDJIRegister.so",
            "*/*/libdjibase.so",
            "*/*/libDJICSDKCommon.so",
            "*/*/libDJIFlySafeCore-CSDK.so",
            "*/*/libdjifs_jni-CSDK.so",
            "*/*/libDJIUpgradeCore.so",
            "*/*/libDJIUpgradeJNI.so",
            "*/*/libDJIWaypointV2Core-CSDK.so",
            "*/*/libdjiwpv2-CSDK.so",
            "*/*/libFlightRecordEngine.so",
            "*/*/libvideo-framing.so",
            "*/*/libwaes.so",
            "*/*/libagora-rtsa-sdk.so",
            "*/*/libc++.so",
            "*/*/libc++_shared.so",
            "*/*/libmrtc_28181.so",
            "*/*/libmrtc_agora.so",
            "*/*/libmrtc_core.so",
            "*/*/libmrtc_core_jni.so",
            "*/*/libmrtc_data.so",
            "*/*/libmrtc_log.so",
            "*/*/libmrtc_onvif.so",
            "*/*/libmrtc_rtmp.so",
            "*/*/libmrtc_rtsp.so",
            "*/*/libconstants.so",
            "*/*/libdji_innertools.so"
        )
    }
}

dependencies {
    implementation("com.dji:dji-sdk-v5-aircraft:5.17.0")
    compileOnly("com.dji:dji-sdk-v5-aircraft-provided:5.17.0")
    runtimeOnly("com.dji:dji-sdk-v5-networkImp:5.17.0")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // ONNX Runtime for on-device YOLO inference
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")

    // Embedded web server for dashboard
    implementation("org.nanohttpd:nanohttpd:2.3.1")
}
