package video.zhuker.sancho

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
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

        //37.2490921080559 -121.95113440773974 108.8 0 75.4 53.7 0.0 0.0 24.0 0.05693451728060701 0.05693451728060701 1920 1080 960 540
        val params = Params(
            latitude = 37.2490921080559,
            longitude = -121.95113440773974,
            elevation = 108.8,
            roll = 0.0,
            pitch = 75.4,
            heading = 53.7,
            px = 0.0,
            py = 0.0,
            focal_length = 24.0,
            pix_size_x = 0.05693451728060701,
            pix_size_y = 0.05693451728060701,
            img_width = 1920,
            img_height = 1080,
            u = 960.0,
            v = 540.0
        )
        val model = ossimApplanixEcefModel(
            ossimDrect(ossimDpt(0.0, 0.0), ossimDpt(params.img_width.toDouble(), params.img_height.toDouble())),
            ossimGpt(params.latitude, params.longitude, params.elevation),
            params.roll,
            params.pitch,
            params.heading,
            ossimDpt(params.px, params.py),
            params.focal_length,
            ossimDpt(params.pix_size_x, params.pix_size_y)
        )
        model.setPrincipalPoint(ossimDpt(params.px, params.py))
        val res = ossimGpt()
        model.lineSampleToWorld(ossimDpt(params.u, params.v), res);
        println(res)
//        lat = {ossim_float64} 37.251320410557689
//        lon = {ossim_float64} -121.94733956444934
//        hgt = {ossim_float64} -0.0000444104488377848
        assertEquals(37.251320410557689, res.lat, 0.0000000000001)
        assertEquals(-121.94733956444934, res.lon, 0.0000000000001)
        assertEquals(-0.0000444104488377848, res.hgt, 0.0000000000001)
    }

    @Test
    fun testUpdateModel2() {
        val KWL = ossimKeywordlist.hardcodedConfig(File("testdata").absolutePath)
        ossimElevManager.instance().loadState(KWL, "elevation_manager.")

        //37.2490921080559 -121.95113440773974 108.8 0 75.4 53.7 0.0 0.0 24.0 0.05693451728060701 0.05693451728060701 1920 1080 960 540
        val params = Params(
            latitude = 37.2490921080559,
            longitude = -122.087915,
            elevation = 108.8,
            roll = 0.0,
            pitch = 75.4,
            heading = 53.7,
            px = 0.0,
            py = 0.0,
            focal_length = 24.0,
            pix_size_x = 0.05693451728060701,
            pix_size_y = 0.05693451728060701,
            img_width = 1920,
            img_height = 1080,
            u = 960.0,
            v = 540.0
        )
        val model = ossimApplanixEcefModel(
            ossimDrect(ossimDpt(0.0, 0.0), ossimDpt(params.img_width.toDouble(), params.img_height.toDouble())),
            ossimGpt(params.latitude, params.longitude, params.elevation),
            params.roll,
            params.pitch,
            params.heading,
            ossimDpt(params.px, params.py),
            params.focal_length,
            ossimDpt(params.pix_size_x, params.pix_size_y)
        )
        model.setPrincipalPoint(ossimDpt(params.px, params.py))
        val res = ossimGpt()
        model.lineSampleToWorld(ossimDpt(params.u, params.v), res);
        println(res)
//        lat = {ossim_float64} 37.251320410557689
//        lon = {ossim_float64} -121.94733956444934
//        hgt = {ossim_float64} -0.0000444104488377848
        assertEquals(37.233936863641851, res.lat, 0.0000000000001)
        assertEquals(-122.113713241769318, res.lon, 0.0000000000001)
        assertEquals(849.44254878849074, res.hgt, 0.00000001)
    }
}