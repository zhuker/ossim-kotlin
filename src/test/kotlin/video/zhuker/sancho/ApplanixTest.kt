package video.zhuker.sancho

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import video.zhuker.sancho.SimpleOssim.findTarget
import video.zhuker.sancho.SimpleOssim.makeParams
import java.io.File

class ApplanixTest {
    data class WithArray(val data: DoubleArray) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as WithArray

            if (!data.contentEquals(other.data)) return false

            return true
        }

        override fun hashCode(): Int {
            return data.contentHashCode()
        }

        fun deepCopy(): WithArray {
            return this.copy(data = this.data.copyOf())
        }
    }

    @Test
    fun howDataClassWithArrayWorks() {
        val d1 = WithArray(doubleArrayOf(1.0, 2.0, 3.0))
        val copy = d1.copy()
        assertTrue(d1.data === copy.data)
        val clone = d1.deepCopy()
        assertFalse(d1.data === clone.data)
    }

    @Test
    fun metersPerDegree() {
        val radius = ossimGpt().theDatum.ellipsoid().geodeticRadius(0.0)
        assertEquals(6378137.0, radius)
        val metersPerDegree = ossimGpt().metersPerDegree()
        assertEquals(111319.49079327357, metersPerDegree.x)
        assertEquals(111319.49079327357, metersPerDegree.y)
    }

    @Test
    fun testResizeList() {
        val empty = emptyList<String>()
        val grown = empty.resize(4) { "x" }
        assertIterableEquals(listOf("x", "x", "x", "x"), grown)
        val shrunk = grown.resize(3) { "y" }
        assertIterableEquals(listOf("x", "x", "x"), shrunk)
        val grownBack = shrunk.resize(5) { "z" }
        assertIterableEquals(listOf("x", "x", "x", "z", "z"), grownBack)
    }

    @Test
    fun testResizeArray() {
        val empty = emptyArray<String>()
        val grown = empty.resize(4) { "x" }
        assertIterableEquals(listOf("x", "x", "x", "x"), grown.asIterable())
        val shrunk = grown.resize(3) { "y" }
        assertIterableEquals(listOf("x", "x", "x"), shrunk.asIterable())
        val grownBack = shrunk.resize(5) { "z" }
        assertIterableEquals(listOf("x", "x", "x", "z", "z"), grownBack.asIterable())
    }

    @Test
    fun testUpdateModel() {
        val KWL = ossimKeywordlist.hardcodedConfig(File("testdata").absolutePath)
        ossimElevManager.instance().loadState(KWL, "elevation_manager.")

        val res = findTarget(makeParams(37.2490921080559, -121.95113440773974))
        println(res)
        assertEquals(37.251320410557689, res.lat, 0.0000000000001)
        assertEquals(-121.94733956444934, res.lon, 0.0000000000001)
        assertEquals(-0.0000444104488377848, res.hgt, 0.0000000000001)
    }


    @Test
    fun testUpdateModel2() {
        val KWL = ossimKeywordlist.hardcodedConfig(File("testdata").absolutePath)
        ossimElevManager.instance().loadState(KWL, "elevation_manager.")

        val res = findTarget(makeParams(37.2490921080559, -122.087915))
        println(res)
        assertEquals(37.233936863641851, res.lat, 0.0000000000001)
        assertEquals(-122.113713241769318, res.lon, 0.0000000000001)
        assertEquals(849.44254878849074, res.hgt, 0.00000001)
    }

    @Test
    fun testUpdateModel2CustomDb() {
        var customDbCalled = false
        val dir = File("testdata")
        val customDb = object : ossimSrtmElevationDatabase() {
            override fun createCell(gpt: ossimGpt): ossimElevCellHandler? {
                val h = object : ossimSrtmHandler() {
                    override fun open(f: File, mmap: Boolean): Boolean {
                        println("custom open $f")
                        customDbCalled = true
                        srtmFile = SrtmElevationFile.loadHgt(f)
                        return true
                    }
                }
                val filename = createRelativePath(gpt)
                h.open(File(dir, filename), false)
                return h
            }
        }
        val mgr = ossimElevManager(listOf(customDb))

        val res = findTarget(makeParams(37.2490921080559, -122.087915), mgr)
        assertTrue(customDbCalled)
        println(res)
        assertEquals(37.233936863641851, res.lat, 0.0000000000001)
        assertEquals(-122.113713241769318, res.lon, 0.0000000000001)
        assertEquals(849.44254878849074, res.hgt, 0.00000001)
    }

    @Test
    fun testDatabase2() {
        val hgt = SrtmElevationFile.loadHgt(File("testdata/N37W123.hgt"))
        assertEquals(-1.0, hgt.get_elevation(38.0, -122.34166666665604))
        assertEquals(-0.9999999999986227, hgt.get_precise_elevation(37.99999999, -122.34166666665604))
    }

    @Test
    fun testDatabase() {
        val db = ossimSrtmElevationDatabase(File("testdata").absolutePath)
        //negative
        val negative = db.getHeightAboveMSL(ossimGpt(37.99999999, -122.34166666665604))
        assertEquals(-0.9999999999986227, negative)

        val heightAboveMSL = db.getHeightAboveMSL(ossimGpt(37.10967697992352, -122.1895308657649))
        assertEquals(454.3607390550334, heightAboveMSL)

        // we dont have hgt file for the lat/lon
        val noData = db.getHeightAboveMSL(ossimGpt(37.10967697992352, -121.1895308657649))
        assertTrue(noData.isNaN())


    }
}