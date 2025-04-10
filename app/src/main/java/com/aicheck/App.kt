package com.aicheck

import android.app.Application
import android.content.Context

class App : Application() {
    lateinit var phishingDetector: VoicePhishingDetector

    companion object {
        private lateinit var instance: App

        fun getContext(): Context = instance.applicationContext
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        phishingDetector = VoicePhishingDetector(this)
    }
}