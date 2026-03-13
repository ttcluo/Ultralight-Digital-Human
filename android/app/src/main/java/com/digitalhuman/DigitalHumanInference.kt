package com.digitalhuman

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.InputStream
import java.nio.FloatBuffer

/**
 * 数字人 ONNX 推理，与 inference_onnx.py 逻辑一致。
 */
class DigitalHumanInference(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var numFrames: Int = 0
    private var featDim1: Int = 16
    private var featDim2: Int = 512
    private var audioFeats: FloatArray? = null
    private var images: MutableList<Bitmap> = mutableListOf()
    private var landmarks: MutableList<Array<FloatArray>> = mutableListOf()
    private var lenImg: Int = 0

    val outputWidth: Int get() = images.firstOrNull()?.width ?: 0
    val outputHeight: Int get() = images.firstOrNull()?.height ?: 0

    fun loadModels(onnxStream: InputStream) {
        ortEnv = OrtEnvironment.getEnvironment()
        val bytes = onnxStream.readBytes()
        session = ortEnv?.createSession(bytes)
    }

    fun loadAudioFeaturesFromBin(input: InputStream) {
        java.io.DataInputStream(input).use { dis ->
            val d0 = dis.readInt()
            val d1 = dis.readInt()
            val d2 = dis.readInt()
            val d3 = dis.readInt()
            val size = d0 * d1 * d2 * d3
            val bytes = ByteArray(size * 4)
            dis.readFully(bytes)
            audioFeats = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let { buf ->
                FloatArray(size).also { buf.get(it) }
            }
            numFrames = d0
            featDim1 = d1
            featDim2 = d2
        }
    }

    fun loadAvatarAssets(imgDir: String, lmsDir: String) {
        val imgList = context.assets.list(imgDir)?.filter { it.endsWith(".jpg") }?.sortedBy { it.replace(".jpg", "").toIntOrNull() ?: 0 } ?: emptyList()
        lenImg = imgList.size - 1
        for (f in imgList) {
            context.assets.open("$imgDir/$f").use { images.add(BitmapFactory.decodeStream(it)) }
        }
        for (i in 0..lenImg) {
            context.assets.open("$lmsDir/$i.lms").use {
                landmarks.add(ImageProcessor.loadLandmarks(it))
            }
        }
    }

    private fun getAudioFeatures(index: Int): FloatArray {
        val feats = audioFeats ?: throw IllegalStateException("audio features not loaded")
        val left = (index - 4).coerceAtLeast(0)
        val right = (index + 4).coerceAtMost(numFrames)
        val padLeft = index - 4 - left
        val padRight = right - (index + 4)

        val frameSize = featDim1 * featDim2
        val result = FloatArray(8 * frameSize)

        if (padLeft > 0) {
            for (i in 0 until padLeft * frameSize) result[i] = 0f
        }
        for (i in left until right) {
            val srcStart = i * frameSize
            val dstStart = (padLeft + (i - left)) * frameSize
            feats.copyInto(result, dstStart, srcStart, srcStart + frameSize)
        }
        if (padRight > 0) {
            val dstStart = (8 - padRight) * frameSize
            for (i in dstStart until 8 * frameSize) result[i] = 0f
        }
        return result
    }

    fun run(callback: (Int, Int, Bitmap) -> Unit) {
        val sess = session ?: throw IllegalStateException("model not loaded")
        val feats = audioFeats ?: throw IllegalStateException("audio features not loaded")

        var stepStride = 0
        var imgIdx = 0

        for (i in 0 until numFrames) {
            if (imgIdx > lenImg - 1) stepStride = -1
            if (imgIdx < 1) stepStride = 1
            imgIdx += stepStride

            val img = images[imgIdx].copy(Bitmap.Config.ARGB_8888, true)
            val lms = landmarks[imgIdx]
            val bbox = ImageProcessor.getBbox(lms)

            val imgInput = ImageProcessor.prepareModelInput(img, bbox)
            val audioInput = getAudioFeatures(i)

            val imgTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(imgInput), longArrayOf(1, 6, 160, 160))
            val audioTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(audioInput), longArrayOf(1, 128, 16, 32))

            val inputNames = sess.inputNames.toList()
            val result = sess.run(mapOf(
                inputNames[0] to imgTensor,
                inputNames[1] to audioTensor
            ))

            val outputTensor = result[0] as OnnxTensor
            val pred = FloatArray(3 * 160 * 160)
            outputTensor.floatBuffer.get(pred)
            outputTensor.close()
            imgTensor.close()
            audioTensor.close()
            result.close()

            val outBitmap = ImageProcessor.overlayPrediction(img, bbox, pred)
            img.recycle()
            callback(i, numFrames, outBitmap)
        }
    }

    fun release() {
        session?.close()
        ortEnv?.close()
        images.forEach { it.recycle() }
    }
}
