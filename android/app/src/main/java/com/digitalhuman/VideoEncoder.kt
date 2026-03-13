package com.digitalhuman

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

/**
 * 将 Bitmap 序列编码为 H.264 视频。
 */
class VideoEncoder(
    private val width: Int,
    private val height: Int,
    private val fps: Int = 20
) {
    private var mediaCodec: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var trackIndex: Int = -1
    private var muxerStarted: Boolean = false

    fun start(outputPath: String) {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, 2_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        mediaCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            start()
        }
        muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    }

    fun encodeFrame(bitmap: Bitmap) {
        val codec = mediaCodec ?: return
        val bufInfo = MediaCodec.BufferInfo()
        var inputIndex = codec.dequeueInputBuffer(10_000)
        if (inputIndex >= 0) {
            val yuv = bitmapToYuv420(bitmap)
            val buffer = codec.getInputBuffer(inputIndex) ?: return
            buffer.clear()
            buffer.put(yuv)
            codec.queueInputBuffer(inputIndex, 0, yuv.size, System.nanoTime() / 1000, 0)
        }

        var outputIndex = codec.dequeueOutputBuffer(bufInfo, 10_000)
        while (outputIndex != MediaCodec.INFO_TRY_AGAIN_LATER) {
            if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED && !muxerStarted) {
                val newFormat = codec.outputFormat
                trackIndex = muxer?.addTrack(newFormat) ?: -1
                muxer?.start()
                muxerStarted = true
            } else if (outputIndex >= 0) {
                if (bufInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && bufInfo.size > 0 && muxerStarted) {
                    val outBuffer = codec.getOutputBuffer(outputIndex) ?: return
                    outBuffer.position(bufInfo.offset)
                    outBuffer.limit(bufInfo.offset + bufInfo.size)
                    muxer?.writeSampleData(trackIndex, outBuffer, bufInfo)
                }
                codec.releaseOutputBuffer(outputIndex, false)
            }
            outputIndex = codec.dequeueOutputBuffer(bufInfo, 0)
        }
    }

    private fun bitmapToYuv420(bitmap: Bitmap): ByteArray {
        val width = bitmap.width
        val height = bitmap.height
        val ySize = width * height
        val uvSize = width * height / 2
        val yuv = ByteArray(ySize + uvSize)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (j in 0 until height) {
            for (i in 0 until width) {
                val p = pixels[j * width + i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                yuv[j * width + i] = ((66 * r + 129 * g + 25 * b + 128) shr 8 + 16).toByte()
            }
        }

        var uvIndex = ySize
        for (j in 0 until height step 2) {
            for (i in 0 until width step 2) {
                val p = pixels[j * width + i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                val y = (66 * r + 129 * g + 25 * b + 128) shr 8 + 16
                val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8 + 128).coerceIn(0, 255)
                val v = ((112 * r - 94 * g - 18 * b + 128) shr 8 + 128).coerceIn(0, 255)
                yuv[uvIndex++] = u.toByte()
                yuv[uvIndex++] = v.toByte()
            }
        }
        return yuv
    }

    fun stop() {
        val codec = mediaCodec ?: return
        var inputIndex = codec.dequeueInputBuffer(10_000)
        if (inputIndex >= 0) {
            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
        var outputIndex = codec.dequeueOutputBuffer(MediaCodec.BufferInfo(), 10_000)
        while (outputIndex != MediaCodec.INFO_TRY_AGAIN_LATER) {
            if (outputIndex >= 0) {
                codec.releaseOutputBuffer(outputIndex, false)
            }
            outputIndex = codec.dequeueOutputBuffer(MediaCodec.BufferInfo(), 10_000)
        }
        codec.stop()
        codec.release()
        muxer?.stop()
        muxer?.release()
        mediaCodec = null
        muxer = null
    }
}
