package video.zhuker.sancho

import org.ejml.simple.SimpleMatrix
import video.zhuker.sancho.Constants.Companion.DEG_PER_RAD
import video.zhuker.sancho.Constants.Companion.FLT_EPSILON
import video.zhuker.sancho.Constants.Companion.OSSIM_INT_NAN
import video.zhuker.sancho.Constants.Companion.RAD_PER_DEG
import video.zhuker.sancho.Constants.Companion.fabs
import video.zhuker.sancho.ossim.Companion.cosd
import video.zhuker.sancho.ossim.Companion.sind
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.*


private fun SimpleMatrix.Nrows(): Int = this.numRows()
private fun SimpleMatrix.Ncols(): Int = this.numCols()
fun SimpleMatrix.i(): SimpleMatrix = this.invert()
operator fun SimpleMatrix.times(other: SimpleMatrix): SimpleMatrix {
    return this.mult(other)
}

private val ByteBuffer.size: Int get() = this.remaining()
fun <T> List<T>.resize(newsize: Int, ctor: () -> T): List<T> {
    return when {
        this.size == newsize -> this
        this.size > newsize -> this.subList(0, newsize)
        this.size < newsize -> {
            this + (0 until (newsize - this.size)).map { ctor() }
        }
        else -> throw Exception("BUG")
    }
}

fun <T> Array<T>.resize(newsize: Int, ctor: () -> T): Array<T> {
    return when {
        this.size == newsize -> this
        this.size > newsize -> this.copyOf(newsize) as Array<T>
        this.size < newsize -> {
            val resized = this.copyOf(newsize)
            for (i in (this.size until newsize)) {
                resized[i] = ctor()
            }
            return resized as Array<T>
        }
        else -> throw Exception("BUG")
    }
}

class SrtmElevationFile(
    val latitude: Double,
    val longitude: Double,
    val data: ByteBuffer,
    val square_side: Int,
    val resolution: Double
) {
    fun get_elevation(latitude: Double, longitude: Double): Double {
        val self = this
        if (!(self.latitude - self.resolution <= latitude || latitude < self.latitude + 1)) {
            throw IllegalArgumentException()
        } else if (!(self.longitude <= longitude || longitude < self.longitude + 1 + self.resolution)) {
            throw IllegalArgumentException()
        }
        val (row, column) = self.get_row_and_column(latitude, longitude)
        return self.get_elevation_from_row_and_column(row, column)
    }

    val theNullHeightValue = -32768.0
    fun get_precise_elevation(latitude: Double, longitude: Double): Double {
        val self = this
        if (!(self.latitude - self.resolution <= latitude || latitude < self.latitude + 1)) {
            throw IllegalArgumentException()
        } else if (!(self.longitude <= longitude || longitude < self.longitude + 1 + self.resolution)) {
            throw IllegalArgumentException()
        }

        val xi = ((longitude - self.longitude) * (self.square_side - 1))
        val yi = ((self.latitude + 1 - latitude) * (self.square_side - 1))
        var x0 = (xi.toInt());
        var y0 = (yi.toInt());

        if (x0 == (square_side - 1)) {
            --x0; // Move over one post.
        }

        // Check for top edge.
        //    if (gpt.lat == theGroundRect.ul().lat)
        if (y0 == (square_side - 1)) {
            --y0; // Move down one post.
        }
        if (xi < 0 || yi < 0 ||
            x0 > (square_side - 2) ||
            y0 > (square_side - 2)
        ) {
            println("not enough srtm data to interpolate $latitude $longitude")
            return Double.NaN
        }
        val (row, column) = get_row_and_column(latitude, longitude)
        val p00 = self.get_elevation_from_row_and_column(row, column)
        val p01 = self.get_elevation_from_row_and_column(row, column + 1)
        val p10 = self.get_elevation_from_row_and_column(row + 1, column)
        val p11 = self.get_elevation_from_row_and_column(row + 1, column + 1)

        val xt0 = xi - x0
        val yt0 = yi - y0
        val xt1 = 1 - xt0
        val yt1 = 1 - yt0

        var w00 = xt1 * yt1
        var w01 = xt0 * yt1
        var w10 = xt1 * yt0
        var w11 = xt0 * yt0
        if (p00 == theNullHeightValue)
            w00 = 0.0;
        if (p01 == theNullHeightValue)
            w01 = 0.0;
        if (p10 == theNullHeightValue)
            w10 = 0.0;
        if (p11 == theNullHeightValue)
            w11 = 0.0;
        val sum_weights = w00 + w01 + w10 + w11
        if (abs(sum_weights) > 0.0) {
            return (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11) / sum_weights;
        }

        return Double.NaN
    }


    private fun get_elevation_from_row_and_column(row: Int, column: Int): Double {
        val self = this
        val i = row * (self.square_side) + column
        val hi = self.data[i * 2].toInt() and 0xff
        val lo = self.data[i * 2 + 1].toInt() and 0xff
        val x = (hi shl 8) or lo
        return x.toShort().toDouble()
    }

    private fun get_row_and_column(latitude: Double, longitude: Double): Pair<Int, Int> {
        val self = this
        return floor((self.latitude + 1 - latitude) * (self.square_side - 1)).toInt() to
                floor((longitude - self.longitude) * (self.square_side - 1)).toInt()
    }

    companion object {
        private val regex = "([NS])(\\d+)([EW])(\\d+)\\.hgt".toRegex()

        fun loadHgt(file: File): SrtmElevationFile {
            val bytes = file.inputStream().channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
//            val bytes = file.readBytes()
            val matchResult = regex.matchEntire(file.name)!!
            val ns = matchResult.groupValues[1]
            val latstr = matchResult.groupValues[2]
            val ew = matchResult.groupValues[3]
            val lonstr = matchResult.groupValues[4]
            val lat = latstr.toDouble() * (if (ns == "S") -1f else 1f)
            val lon = lonstr.toDouble() * (if (ew == "W") -1f else 1f)

            val square_side = sqrt(bytes.size / 2.0).toInt()
            val resolution = 1.0 / (square_side - 1.0)

            return SrtmElevationFile(lat, lon, bytes, square_side, resolution)
        }
    }
}

class ossim {
    companion object {
        fun sind(x: Double): Double {
            return sin(x * RAD_PER_DEG)
        }

        fun cosd(x: Double): Double {
            return cos(x * RAD_PER_DEG)
        }

        fun wrap(x: Double, a: Double, b: Double): Double {
            if (a == b && !x.isNaN()) return a
            val z = if (x < a) b else a
            return std.fmod(x - z, b - a) + z
        }
    }
}

class std {
    companion object {
        fun abs(x: Double): Double {
            return kotlin.math.abs(x)
        }

        fun fmod(a: Double, b: Double): Double {
            val result = floor(a / b).toInt()
            return a - result * b
        }
    }
}

class Constants {
    companion object {
        const val M_PI = 3.141592653589793238462643
        const val RAD_PER_DEG = M_PI / 180.0
        const val DEG_PER_RAD = 180.0 / M_PI
        const val OSSIM_INT_NAN: Int = 0x80000000.toInt()
        val FLT_EPSILON = Math.ulp(1.0f)        //1.1920929E-7


        fun fabs(x: Double): Double {
            return abs(x)
        }

    }
}

data class Params(
    val latitude: Double,
    val longitude: Double,
    val elevation: Double,
    val roll: Double,
    val pitch: Double,
    val heading: Double,
    val px: Double,
    val py: Double,
    val focal_length: Double,
    val pix_size_x: Double,
    val pix_size_y: Double,
    val img_width: Int,
    val img_height: Int,
    val u: Double,
    val v: Double
)

data class ossimDpt(var x: Double = 0.0, var y: Double = 0.0) {
    constructor(pt: ossimGpt) : this(pt.lon, pt.lat)

    fun hasNans(): Boolean = x.isNaN() || y.isNaN()
    operator fun minus(p: ossimDpt): ossimDpt {
        return ossimDpt(x - p.x, y - p.y)
    }

    fun assign(other: ossimDpt) {
        x = other.x
        y = other.y
    }
}

enum class ossimCoordSysOrientMode {
    OSSIM_LEFT_HANDED, OSSIM_RIGHT_HANDED
}

class ossimDrect(
    ul_corner: ossimDpt,
    lr_corner: ossimDpt,
    val mode: ossimCoordSysOrientMode = ossimCoordSysOrientMode.OSSIM_LEFT_HANDED
) {
    private val theUlCorner: ossimDpt = ul_corner.copy()
    private val theUrCorner: ossimDpt = ossimDpt(lr_corner.x, ul_corner.y)
    private val theLrCorner: ossimDpt = lr_corner.copy()
    private val theLlCorner: ossimDpt = ossimDpt(ul_corner.x, lr_corner.y)

    fun ul(): ossimDpt = theUlCorner
    fun ur(): ossimDpt = theUrCorner
    fun lr(): ossimDpt = theLrCorner
    fun ll(): ossimDpt = theLlCorner

    fun midPoint(): ossimDpt {
        return ossimDpt(
            (ul().x + ur().x + ll().x + lr().x) * .25, (ul().y + ur().y + ll().y + lr().y) * .25
        );
    }

    fun hasNans(): Boolean {
        return theUlCorner.hasNans() ||
                theLlCorner.hasNans() ||
                theLrCorner.hasNans() ||
                theUrCorner.hasNans()
    }

    fun width(): Double {
        return fabs(theLrCorner.x - theLlCorner.x) + 1.0;
    }

    fun height(): Double {
        return fabs(theLlCorner.y - theUlCorner.y) + 1.0;
    }
}

class ossimEllipsoid(
    val theName: String,
    val theCode: String,
    val theEpsgCode: UInt,
    val theA: Double,
    val theB: Double,
    val theFlattening: Double,
    val theA_squared: Double,
    val theB_squared: Double,
    val theEccentricitySquared: Double
) {
    fun geodeticRadius(lat: Double): Double {
        val cos_lat = cosd(lat)
        val sin_lat = sind(lat)
        val cos2_lat = cos_lat * cos_lat
        val sin2_lat = sin_lat * sin_lat
        val a2_cos = theA_squared * cos_lat
        val b2_sin = theB_squared * sin_lat
        return sqrt(((a2_cos * a2_cos) + (b2_sin * b2_sin)) / (theA_squared * cos2_lat + theB_squared * sin2_lat))
    }

    fun latLonHeightToXYZ(lat: Double, lon: Double, height: Double, xyz: ossimColumnVector3d) {
        val sin_latitude = sind(lat);
        val cos_latitude = cosd(lat);
        val N = theA / sqrt(1.0 - theEccentricitySquared * sin_latitude * sin_latitude);
        xyz.x = (N + height) * cos_latitude * cosd(lon);
        xyz.y = (N + height) * cos_latitude * sind(lon);
        xyz.z = (N * (1 - theEccentricitySquared) + height) * sin_latitude;
    }

    fun XYZToLatLonHeight(x: Double, y: Double, z: Double, gpt: ossimGpt) {
        val tol = 1e-15;
        val d = sqrt(x * x + y * y);
        val MAX_ITER = 10;

        val a2 = theA * theA;
        val b2 = theB * theB;
        val pa2 = d * d * a2;
        val qb2 = z * z * b2;
        val ae2 = a2 * eccentricitySquared();
        val ae4 = ae2 * ae2;

        val c3 = -(ae4 / 2 + pa2 + qb2);          // s^2
        val c4 = ae2 * (pa2 - qb2);               // s^1
        val c5 = ae4 / 4 * (ae4 / 4 - pa2 - qb2);   // s^0

        var s0 = 0.5 * (a2 + b2) * hypot(d / theA, z / theB)
        for (iterIdx in 0 until MAX_ITER) {
            val pol = c5 + s0 * (c4 + s0 * (c3 + s0 * s0));
            val der = c4 + s0 * (2 * c3 + 4 * s0 * s0);
            val ds = -pol / der;
            s0 += ds;
            if (fabs(ds) < tol * s0) {
                val t = s0 - 0.5 * (a2 + b2);
                val x_ell = d / (1.0 + t / a2);
                val y_ell = z / (1.0 + t / b2);

                gpt.hgt = (d - x_ell) * x_ell / a2 + (z - y_ell) * y_ell / b2;
                gpt.hgt /= hypot(x_ell / a2, y_ell / b2);

                gpt.lat = atan2(y_ell / b2, x_ell / a2) * DEG_PER_RAD;
                gpt.lon = atan2(y, x) * DEG_PER_RAD;

                return;
            }
        }
    }

    private fun eccentricitySquared(): Double {
        return theEccentricitySquared
    }

    fun nearestIntersection(ray: ossimEcefRay, offset: Double, rtnPt: ossimEcefPoint): Boolean {
        var success = false;

        val eccSqComp = 1.0 - theEccentricitySquared;
        val CONVERGENCE_THRESHOLD = 0.0001;
        val MAX_NUM_ITERATIONS = 10;
        val start = ray.origin()
        val direction = ray.direction()

        val start_x = start.x()
        val start_y = start.y()
        val start_z = start.z()

        val direction_x: Double = direction.x()
        val direction_y: Double = direction.y()
        val direction_z: Double = direction.z()

        var rayGpt = ossimGpt.fromEcefPoint(start);
        var iterations = 0
        var done = false
        do {
            val rayLat = rayGpt.latd();
            val raySinLat = ossim.sind(rayLat);
            val rayScale = 1.0 / sqrt(1.0 - theEccentricitySquared * raySinLat * raySinLat);
            val rayN = theA * rayScale;
            val A_offset: Double = rayN + offset
            val B_offset: Double = rayN * eccSqComp + offset
            val A_offset_inv = 1.0 / A_offset
            val B_offset_inv = 1.0 / B_offset

            val scaled_x = start_x * A_offset_inv
            val scaled_y = start_y * A_offset_inv
            val scaled_z = start_z * B_offset_inv

            val A_over_B = A_offset * B_offset_inv

            //***
            // Solve the coefficients of the quadratic formula
            //***

            //***
            // Solve the coefficients of the quadratic formula
            //***
            var a = (direction_x * direction_x + direction_y * direction_y) * A_offset_inv
            a += direction_z * direction_z * B_offset_inv * A_over_B

            val b = 2.0 * (scaled_x * direction_x + scaled_y * direction_y + scaled_z * direction_z * A_over_B)

            var c = scaled_x * start_x + scaled_y * start_y
            c += scaled_z * start_z * A_over_B
            c -= A_offset

            //***
            // solve the quadratic
            //***
            //***
            // solve the quadratic
            //***
            val root = b * b - 4.0 * a * c
            if (root < 0.0) {
                return false
            }
            val squareRoot = sqrt(root)
            val oneOver2a = 1.0 / (2.0 * a)

            val t1 = (-b + squareRoot) * oneOver2a
            val t2 = (-b - squareRoot) * oneOver2a


            // OLK: Alternate means to find closest intersection, not necessarily "in front of" origin:

            // OLK: Alternate means to find closest intersection, not necessarily "in front of" origin:
            var tEstimate = t1
            if (fabs(t2) < fabs(t1)) tEstimate = t2
            // Get estimated intersection point.
            // Get estimated intersection point.
            val rayEcef = ray.extend(tEstimate)
            rayGpt = ossimGpt.fromEcefPoint(rayEcef);
            val offsetError = fabs(rayGpt.height() - offset)
            if (offsetError < CONVERGENCE_THRESHOLD) {
                done = true;
                success = true;
                rtnPt.assign(rayEcef);
            }
        } while ((!done) && (iterations++ < MAX_NUM_ITERATIONS));
        return success
    }
}

fun wgs84Ellipsoid(): ossimEllipsoid {
    return ossimEllipsoid(
        "WGS 84",
        "WE",
        theEpsgCode = 7030U,
        theA = 6378137.0,
        theB = 6356752.3141999999,
        theFlattening = 0.0033528106718309896,
        theA_squared = 40680631590769.0,
        theB_squared = 40408299984087.055,
        theEccentricitySquared = 0.0066943800042608354
    )
}

interface ossimDatum {
    fun ellipsoid(): ossimEllipsoid
}

class ossimWgs84Datum : ossimDatum {
    private val theEllipsoid: ossimEllipsoid = wgs84Ellipsoid()
    override fun ellipsoid(): ossimEllipsoid {
        return theEllipsoid
    }
}


data class ossimGpt(
    var lat: Double = 0.0, var lon: Double = 0.0, var hgt: Double = 0.0, var theDatum: ossimDatum = ossimWgs84Datum()
) {

    companion object {
        fun fromEcefPoint(ecef_point: ossimEcefPoint, datum: ossimDatum = ossimWgs84Datum()): ossimGpt {
            if (ecef_point.isNan()) {
                val gpt = ossimGpt()
                gpt.makeNan()
                return gpt
            } else {
                val gpt = ossimGpt(theDatum = datum)
                ossimWgs84Datum().ellipsoid().XYZToLatLonHeight(ecef_point.x(), ecef_point.y(), ecef_point.z(), gpt)
                return gpt;
            }
        }
    }

    fun metersPerDegree(): ossimDpt {
        val radius = theDatum.ellipsoid().geodeticRadius(lat)
        val y = RAD_PER_DEG * radius
        val x = y * cosd(lat)
        return ossimDpt(x, y)
    }

    fun isHgtNan(): Boolean = hgt.isNaN()
    fun datum(): ossimDatum = theDatum
    fun latd(): Double = lat
    fun lond(): Double = lon
    fun height(): Double = hgt
    fun height(h: Double) {
        hgt = h
    }

    fun latd(d: Double) {
        lat = d
    }

    fun lond(d: Double) {
        lon = d
    }

    fun makeNan() {
        lat = Double.NaN
        lon = Double.NaN
        hgt = Double.NaN
    }

    fun assign(other: ossimGpt) {
        lat = other.lat
        lon = other.lon
        hgt = other.hgt
        theDatum = other.theDatum
    }

    fun hasNans(): Boolean {
        return lat.isNaN() || lon.isNaN() || hgt.isNaN()
    }

    fun distanceTo(arg_pt: ossimGpt): Double {
        val p1 = ossimEcefPoint(this)
        val p2 = ossimEcefPoint(arg_pt)
        return (p1 - p2).magnitude()
    }
}

operator fun Double.times(rhs: ossimColumnVector3d): ossimColumnVector3d {
    val data = rhs;
    val scalar = this;
    return ossimColumnVector3d(
        data[0] * scalar,
        data[1] * scalar,
        data[2] * scalar
    )
}

data class ossimColumnVector3d(var x: Double = 0.0, var y: Double = 0.0, var z: Double = 0.0) {
    constructor(other: ossimColumnVector3d) : this(other.x, other.y, other.z)

    operator fun minus(rhs: ossimColumnVector3d): ossimColumnVector3d {
        val data = this
        return ossimColumnVector3d(
            data[0] - rhs[0],
            data[1] - rhs[1],
            data[2] - rhs[2]
        );
    }

    operator fun plus(rhs: ossimColumnVector3d): ossimColumnVector3d {
        val data = this
        return ossimColumnVector3d(
            data[0] + rhs[0],
            data[1] + rhs[1],
            data[2] + rhs[2]
        );
    }

    operator fun get(idx: Int): Double {
        return when (idx) {
            0 -> x
            1 -> y
            2 -> z
            else -> throw IndexOutOfBoundsException()
        }
    }

    operator fun set(idx: Int, value: Double) {
        when (idx) {
            0 -> x = value
            1 -> y = value
            2 -> z = value
        }
    }

    fun cross(rhs: ossimColumnVector3d): ossimColumnVector3d {
        val data = this
        return ossimColumnVector3d(
            data[1] * rhs[2] - data[2] * rhs[1],
            data[2] * rhs[0] - data[0] * rhs[2],
            data[0] * rhs[1] - data[1] * rhs[0]
        ); }

    fun magnitude(): Double {
        val data = this
        return sqrt(
            data[0] * data[0] +
                    data[1] * data[1] +
                    data[2] * data[2]
        );
    }

    operator fun times(scalar: Double): ossimColumnVector3d {
        val data = this
        return ossimColumnVector3d(
            data[0] * scalar,
            data[1] * scalar,
            data[2] * scalar
        ); }

    operator fun div(scalar: Double): ossimColumnVector3d {
        val data = this
        return ossimColumnVector3d(
            data[0] / scalar,
            data[1] / scalar,
            data[2] / scalar
        );
    }
}

data class ossimEcefPoint(var theData: ossimColumnVector3d = ossimColumnVector3d(0.0, 0.0, 0.0)) {
    fun isNan(): Boolean = theData[0].isNaN() && theData[1].isNaN() && theData[2].isNaN()
    fun x(): Double = theData.x
    fun y(): Double = theData.y
    fun z(): Double = theData.z
    operator fun minus(p: ossimEcefPoint): ossimEcefVector {
        return ossimEcefVector(theData - p.theData)
    }

    operator fun plus(v: ossimEcefVector): ossimEcefPoint {
        return ossimEcefPoint(theData + v.theData)
    }

    fun assign(rayEcef: ossimEcefPoint) {
        theData = rayEcef.theData.copy()
    }

    fun makeNan() {
        theData.x = Double.NaN
        theData.y = Double.NaN
        theData.z = Double.NaN
    }

    constructor(gpt: ossimGpt) : this() {
        if (!gpt.isHgtNan()) {
            gpt.datum().ellipsoid().latLonHeightToXYZ(
                gpt.latd(), gpt.lond(), gpt.height(), theData
            );

        } else {
            TODO("not implemented")
        }
    }

    constructor(other: ossimEcefPoint) : this(other.theData.copy())
}

class ossimMeanRadialLensDistortion() {
    fun valid(): Boolean {
        return false
    }
}

data class AdjParam(
    var description: String,
    var unit: String,
    var theParameter: Double,
    var theSigma: Double,
    var theCenter: Double = 0.0
) {
    fun computeOffset(): Double {
        return theCenter + theSigma * theParameter;
    }
}

class ossimLsrSpace(origin: ossimGpt, y_azimuth: Double) {
    var theLsrToEcefRotMatrix: SimpleMatrix = SimpleMatrix.diag(0.0)
    val theOrigin = ossimEcefPoint(origin)

    init {
        val sin_lat = ossim.sind(origin.lat);
        val cos_lat = ossim.cosd(origin.lat);
        val sin_lon = ossim.sind(origin.lon);
        val cos_lon = ossim.cosd(origin.lon);
        val E = ossimColumnVector3d(
            -sin_lon,
            cos_lon,
            0.0
        );
        val N = ossimColumnVector3d(
            -sin_lat * cos_lon,
            -sin_lat * sin_lon,
            cos_lat
        );
        val U = ossimColumnVector3d(E.cross(N));
        if (std.abs(y_azimuth) > FLT_EPSILON) {
            val cos_azim = ossim.cosd(y_azimuth);
            val sin_azim = ossim.sind(y_azimuth);
            val X = cos_azim * E - sin_azim * N
            val Y = sin_azim * E + cos_azim * N
            val Z = X.cross(Y)
            theLsrToEcefRotMatrix = ossimMatrix3x3.create(
                X[0], Y[0], Z[0],
                X[1], Y[1], Z[1],
                X[2], Y[2], Z[2]
            )
        } else {
            theLsrToEcefRotMatrix = ossimMatrix3x3.create(
                E[0], N[0], U[0],
                E[1], N[1], U[1],
                E[2], N[2], U[2]
            );
        }
    }

    fun lsrToEcefRotMatrix(): SimpleMatrix {
        return theLsrToEcefRotMatrix
    }
}

class ossimMatrix3x3 {
    companion object {
        fun create(
            v00: Double, v01: Double, v02: Double,
            v10: Double, v11: Double, v12: Double,
            v20: Double, v21: Double, v22: Double
        ): SimpleMatrix {
            return SimpleMatrix(
                3, 3, true, doubleArrayOf(
                    v00, v01, v02, //
                    v10, v11, v12, //
                    v20, v21, v22
                )
            )
        }
    }
}

class ossimMatrix4x4(m: SimpleMatrix) {
    private var theData: SimpleMatrix = SimpleMatrix.diag(0.0, 0.0, 0.0, 0.0)

    fun getData(): SimpleMatrix {
        return theData
    }

    init {
        if (m.numRows() == 4 && m.numCols() == 4) {
            theData = m
        } else if (m.numCols() == 3 && m.numCols() == 3) {
            theData[0, 0] = m[0, 0];
            theData[0, 1] = m[0, 1];
            theData[0, 2] = m[0, 2];
            theData[0, 3] = 0.0;
            theData[1, 0] = m[1, 0];
            theData[1, 1] = m[1, 1];
            theData[1, 2] = m[1, 2];
            theData[1, 3] = 0.0;
            theData[2, 0] = m[2, 0];
            theData[2, 1] = m[2, 1];
            theData[2, 2] = m[2, 2];
            theData[2, 3] = 0.0;
            theData[3, 0] = 0.0;
            theData[3, 1] = 0.0;
            theData[3, 2] = 0.0;
            theData[3, 3] = 1.0;
        } else {
            theData = SimpleMatrix.diag(1.0, 1.0, 1.0, 1.0)
        }
    }

    companion object {
        fun createRotationXMatrix(angle: Double, orientationMode: ossimCoordSysOrientMode): SimpleMatrix {
            val Cosine = cos(angle * RAD_PER_DEG)
            val Sine = sin(angle * RAD_PER_DEG)

            val mmm = if (orientationMode == ossimCoordSysOrientMode.OSSIM_RIGHT_HANDED) {
                doubleArrayOf(
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, Cosine, Sine, 0.0, //
                    0.0, -Sine, Cosine, 0.0, //
                    0.0, 0.0, 0.0, 1.0
                )
            } else {
                doubleArrayOf(
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, Cosine, -Sine, 0.0, //
                    0.0, Sine, Cosine, 0.0, //
                    0.0, 0.0, 0.0, 1.0
                )
            }
            val m = SimpleMatrix(4, 4, true, mmm)
            return m
        }

        fun createRotationYMatrix(angle: Double, orientationMode: ossimCoordSysOrientMode): SimpleMatrix {
            val Cosine = cos(angle * RAD_PER_DEG)
            val Sine = sin(angle * RAD_PER_DEG)

            val mmm = if (orientationMode == ossimCoordSysOrientMode.OSSIM_RIGHT_HANDED) {
                doubleArrayOf(
                    Cosine, 0.0, -Sine, 0.0  //
                    , 0.0, 1.0, 0.0, 0.0 //
                    , Sine, 0.0, Cosine, 0.0 //
                    , 0.0, 0.0, 0.0, 1.0 //
                )
            } else {
                doubleArrayOf(
                    Cosine, 0.0, Sine, 0.0 //
                    , 0.0, 1.0, 0.0, 0.0 //
                    , -Sine, 0.0, Cosine, 0.0 //
                    , 0.0, 0.0, 0.0, 1.0 //
                )
            }
            val m = SimpleMatrix(4, 4, true, mmm)
            return m
        }
    }
}

operator fun SimpleMatrix.times(rhs: ossimColumnVector3d): ossimColumnVector3d {
    val lhs = this
    if ((lhs.Ncols() == 3) && (lhs.Nrows() == 3)) {
        return ossimColumnVector3d(
            (lhs[0, 0] * rhs[0] + lhs[0, 1] * rhs[1] + lhs[0, 2] * rhs[2]),
            (lhs[1, 0] * rhs[0] + lhs[1, 1] * rhs[1] + lhs[1, 2] * rhs[2]),
            (lhs[2, 0] * rhs[0] + lhs[2, 1] * rhs[1] + lhs[2, 2] * rhs[2])
        );
    } else if ((lhs.Ncols() == 4) && (lhs.Nrows() == 4)) {
        return ossimColumnVector3d(
            (lhs[0, 0] * rhs[0] + lhs[0, 1] * rhs[1] + lhs[0, 2] * rhs[2] + lhs[0, 3]),
            (lhs[1, 0] * rhs[0] + lhs[1, 1] * rhs[1] + lhs[1, 2] * rhs[2] + lhs[1, 3]),
            (lhs[2, 0] * rhs[0] + lhs[2, 1] * rhs[1] + lhs[2, 2] * rhs[2] + lhs[2, 3])
        );
    }
    TODO("Multiplying a 3 row column vector by an invalid matrix")
}

enum class ossimVertexOrdering {
    OSSIM_VERTEX_ORDER_UNKNOWN,
    OSSIM_CLOCKWISE_ORDER,
    OSSIM_COUNTERCLOCKWISE_ORDER
};

data class ossimGeoid(val dummy: Double = 0.0) {
    fun offsetFromEllipsoid(gpt: ossimGpt): Double {
        return 0.0
    }
}

class ossimGeoidManager {
    private val theIdentityGeoid = ossimGeoid()
    fun findGeoidByShortName(value: String): ossimGeoid {
        if (value == "identity") {
            return theIdentityGeoid.copy()
        }
        TODO("Not yet implemented")
    }

    companion object {
        private val m_instance = ossimGeoidManager()
        fun instance(): ossimGeoidManager {
            return m_instance;
        }
    }
}

abstract class ossimElevCellHandler {
    abstract fun getHeightAboveMSL(gpt: ossimGpt): Double
}

abstract class ossimElevationDatabase : ossimElevSource() {
    private var m_geoid = ossimGeoid()
    protected var m_connectionString: String? = null

    override fun loadState(kwl: ossimKeywordlist, prefix: String): Boolean {
        m_connectionString = kwl.find(prefix, "connection_string");
        val value = kwl.find(prefix, "geoid.type")
        if (value != null) {
            m_geoid = ossimGeoidManager.instance().findGeoidByShortName(value);
        }
        return super.loadState(kwl, prefix)
    }

    private fun getOffsetFromEllipsoid(gpt: ossimGpt): Double {
        return m_geoid.offsetFromEllipsoid(gpt)
    }

    fun getHeightAboveEllipsoid(gpt: ossimGpt): Double {
        var h = getHeightAboveMSL(gpt)
        if (!h.isNaN()) {
            h += getOffsetFromEllipsoid(gpt);
        }
        return h
    }

    abstract fun getHeightAboveMSL(gpt: ossimGpt): Double
}

class ossimSrtmHandler : ossimElevCellHandler() {
    var srtmFile: SrtmElevationFile? = null
    fun open(f: File, mmap: Boolean): Boolean {
        println("open $f")
        srtmFile = SrtmElevationFile.loadHgt(f)
        return true
    }

    override fun getHeightAboveMSL(gpt: ossimGpt): Double {
        return srtmFile?.get_precise_elevation(gpt.lat, gpt.lon) ?: Double.NaN
    }
}

abstract class ossimElevationCellDatabase : ossimElevationDatabase() {
    private val m_memoryMapCellsFlag = false

    private val cache = mutableMapOf<ULong, ossimElevCellHandler>()
    private val maxCacheEntries = 2
    private fun getOrCreateCellHandler(gpt: ossimGpt): ossimElevCellHandler? {
        val id = createId(gpt)
        //check if in cache by id
        if (cache.containsKey(id)) {
            return cache[id]
        }
        val result = createCell(gpt)
        if (result != null) {
            cache[id] = result
        }
        if (cache.size > maxCacheEntries) {
            val idToRemove = cache.keys.firstOrNull { it != id }
            if (idToRemove != null)
                cache.remove(idToRemove)
        }
        return result
    }

    private fun createCell(gpt: ossimGpt): ossimElevCellHandler? {
        val f: File = createFullPath(gpt);
        if (f.exists()) {
            val h = ossimSrtmHandler()
            if (h.open(f, m_memoryMapCellsFlag)) {
                return h
            }
        } else {
            println("file not found $f")
        }
        return null
    }

    private fun createFullPath(gpt: ossimGpt): File {
        val relativeFile = createRelativePath(gpt);
        return File(m_connectionString, relativeFile)
    }

    private fun createRelativePath(gpt: ossimGpt): String {
        var ilat = floor(gpt.latd()).toInt()
        var file = if (ilat < 0) "S" else "N"
        ilat = abs(ilat)
        file += "%02d".format(ilat)

        var ilon = floor(gpt.lond()).toInt()
        file += if (ilon < 0) "W" else "E"
        ilon = abs(ilon)
        file += "%03d".format(ilon)
        file += ".hgt"
        return file
    }

    override fun getHeightAboveMSL(gpt: ossimGpt): Double {
        if (isSourceEnabled()) {
            val handler = getOrCreateCellHandler(gpt)
            return handler?.getHeightAboveMSL(gpt) ?: Double.NaN // still need to shift
        }
        TODO("Not yet implemented")
    }

    abstract fun createId(pt: ossimGpt): ULong
}

class ossimSrtmElevationDatabase() : ossimElevationCellDatabase() {
    constructor(srtmDirPath: String) : this() {
        m_connectionString = srtmDirPath
    }

    override fun createId(pt: ossimGpt): ULong {
        var y: ULong = (ossim.wrap(pt.latd(), -90.0, 90.0) + 90.0).toULong();
        var x: ULong = (ossim.wrap(pt.lond(), -180.0, 180.0) + 180.0).toULong();
        x = if (x == 360UL) 359UL else x
        y = if (y == 180UL) 179UL else y
        return (y * 360UL + x);
    }
}

class ossimElevationDatabaseFactory {
    fun createDatabase(kwl: ossimKeywordlist, prefix: String): ossimElevationDatabase {
        val type: String? = kwl.find(prefix, ossimKeywordNames.TYPE_KW)
        if (type != null) {
            val result = createDatabase(type)
            result.loadState(kwl, prefix)
            return result
        }

        TODO("Not yet implemented")
    }

    private fun createDatabase(typeName: String): ossimElevationDatabase {
        if (typeName == "srtm" || typeName == "srtm_directory") {
            return ossimSrtmElevationDatabase()
        }
        TODO("Not yet implemented")
    }

    companion object {
        private val m_instance = ossimElevationDatabaseFactory()
        fun instance(): ossimElevationDatabaseFactory {
            return m_instance
        }
    }
}

class ossimElevationDatabaseRegistry {
    companion object {
        private val m_instance = ossimElevationDatabaseRegistry()

        init {
            m_instance.registerFactory(ossimElevationDatabaseFactory.instance())
        }

        fun instance(): ossimElevationDatabaseRegistry {
            return m_instance
        }
    }

    private val m_factoryList: LinkedList<ossimElevationDatabaseFactory> = LinkedList()

    private fun registerFactory(databaseFactory: ossimElevationDatabaseFactory, pushToFrontFlag: Boolean = false) {
        if (m_factoryList.contains(databaseFactory)) return
        if (pushToFrontFlag)
            m_factoryList.addFirst(databaseFactory)
        else
            m_factoryList.addLast(databaseFactory)
    }

    fun createDatabase(kwl: ossimKeywordlist, prefix: String): ossimElevationDatabase {
        return m_factoryList.first().createDatabase(kwl, prefix);
    }
}

open class ossimConnectableObject {
    open fun loadState(kwl: ossimKeywordlist, prefix: String): Boolean {
        return true
    }

}

open class ossimSource : ossimConnectableObject() {
    var theEnabledFlag: Boolean = true
    fun isSourceEnabled(): Boolean {
        return theEnabledFlag;
    }

    override fun loadState(kwl: ossimKeywordlist, prefix: String): Boolean {
        val lookup: String? = kwl.find(prefix, ossimKeywordNames.ENABLED_KW)
        if (lookup != null) {
            theEnabledFlag = lookup.toBoolean()
        }
        return super.loadState(kwl, prefix)
    }
}

open class ossimElevSource : ossimSource()

object ossimKeywordNames {
    val TYPE_KW = "type"
    val ENABLED_KW = "enabled"
}

class ossimElevManager : ossimElevSource() {
    private var m_useStandardPaths: Boolean = false
    private var m_useGeoidIfNullFlag: Boolean = false
    private val m_maxRoundRobinSize = 1
    private val m_defaultHeightAboveEllipsoid = Double.NaN
    private val m_elevationOffset = Double.NaN

    override fun loadState(kwl: ossimKeywordlist, prefix: String): Boolean {
        if (!super.loadState(kwl, prefix)) {
            return false
        }
        val copyPrefix = prefix;
        val elevationOffset = kwl.find(copyPrefix, "elevation_offset");
        val defaultHeightAboveEllipsoid = kwl.find(copyPrefix, "default_height_above_ellipsoid");
        val elevRndRbnSize = kwl.find(copyPrefix, "threads");

        m_useGeoidIfNullFlag = kwl.getBoolKeywordValue("use_geoid_if_null", copyPrefix);
        m_useStandardPaths = kwl.getBoolKeywordValue("use_standard_elev_paths", copyPrefix);

        val regExpression = "^(" + copyPrefix + "elevation_source[0-9]+.)"
        val keys: List<String> = kwl.getSubstringKeyList(regExpression)
        keys.forEach { key ->
            val enabled = kwl.findKey(key + ossimKeywordNames.ENABLED_KW).toBooleanStrictOrNull() ?: false
            if (enabled) {
                val database = ossimElevationDatabaseRegistry.instance().createDatabase(kwl, key)
                addDatabase(database)
            }
        }
        return true
    }

    private val m_dbRoundRobin: MutableList<ossimElevationDatabase> = mutableListOf()
    private fun addDatabase(database: ossimElevationDatabase, set_as_first: Boolean = false) {
        m_dbRoundRobin.add(database)
    }

    private fun getNextElevDbList(): List<ossimElevationDatabase> {
        return m_dbRoundRobin
    }

    private fun getHeightAboveEllipsoid(gpt: ossimGpt): Double {
        var result = Double.NaN
        if (!isSourceEnabled())
            return result;
        val elevDbList = getNextElevDbList()
        for (elevDb in elevDbList) {
            result = elevDb.getHeightAboveEllipsoid(gpt)
            if (!result.isNaN())
                break
        }
        if (result.isNaN()) {
            // No elevation value was returned from the database, so try next best alternatives depending
            // on ossim_preferences settings. Priority goes to default ellipsoid height if available:
            if (!m_defaultHeightAboveEllipsoid.isNaN()) {
                TODO()
            } else if (m_useGeoidIfNullFlag) {
                TODO()
            }
        }
        // Next, ossim_preferences may have indicated an elevation offset to use (top of trees, error
        // bias, etc):
        if (!m_elevationOffset.isNaN() && !result.isNaN()) {
            return result + m_elevationOffset
        }
        return result
    }

    fun intersectRay(ray: ossimEcefRay, gpt: ossimGpt, defaultElevValue: Double = 0.0): Boolean {

        val CONVERGENCE_THRESHOLD = 0.001; // meters
        val MAX_NUM_ITERATIONS = 50;

        var h_ellips = 0.0; // height above ellipsoid
        var intersected = false;
        val prev_intersect_pt = ossimEcefPoint(ray.origin());
        val new_intersect_pt = ossimEcefPoint();
        var distance = 0.0;
        var done = false;
        var iteration_count = 0;
        if (ray.hasNans()) {
            gpt.makeNan();
            return false;
        }
        val datum = gpt.datum();
        val ellipsoid = datum.ellipsoid();
        gpt.assign(ossimGpt.fromEcefPoint(prev_intersect_pt, datum));
        do {
            h_ellips = getHeightAboveEllipsoid(gpt);
            if (h_ellips.isNaN()) h_ellips = defaultElevValue
            intersected = ellipsoid.nearestIntersection(ray, h_ellips, new_intersect_pt)
            if (!intersected) {
                //
                // No intersection (looking over the horizon), so set ground point
                // to NaNs:
                //
                gpt.makeNan()
                done = true
            } else {
                gpt.assign(ossimGpt.fromEcefPoint(new_intersect_pt, datum));
                distance = (new_intersect_pt - prev_intersect_pt).magnitude();
                if (distance < CONVERGENCE_THRESHOLD)
                    done = true;
                else {
                    prev_intersect_pt.assign(new_intersect_pt);
                }
            }
            iteration_count++;

        } while ((!done) && (iteration_count < MAX_NUM_ITERATIONS))

        return intersected
    }

    companion object {
        private val m_instance = ossimElevManager()
        fun instance(): ossimElevManager {
            return m_instance
        }

    }
}

class ossimPolygon(
    var theOrderingType: ossimVertexOrdering = ossimVertexOrdering.OSSIM_VERTEX_ORDER_UNKNOWN,
    var theVertexList: Array<ossimDpt> = emptyArray(),
    var theCurrentVertex: Int = 0
) {
    fun resize(newsize: Int) {
        theVertexList = theVertexList.resize(newsize) { ossimDpt() }
        theOrderingType = ossimVertexOrdering.OSSIM_VERTEX_ORDER_UNKNOWN;
        theCurrentVertex = 0;
    }

    operator fun set(i: Int, value: ossimDpt) {
        theVertexList[i] = value
    }

    operator fun set(i: Int, value: ossimGpt) {
        theVertexList[i] = ossimDpt(value)
    }
}

data class ossimEcefVector(val theData: ossimColumnVector3d = ossimColumnVector3d()) {
    fun magnitude(): Double {
        return theData.magnitude()
    }

    operator fun times(scalar: Double): ossimEcefVector {
        return ossimEcefVector(theData * scalar);
    }

    fun unitVector(): ossimEcefVector {
        return ossimEcefVector(theData / theData.magnitude());

    }

    fun isNan(): Boolean {
        return theData[0].isNaN() && theData[1].isNaN() && theData[2].isNaN()
    }

    fun x(): Double = theData.x
    fun y(): Double = theData.y
    fun z(): Double = theData.z
}

class ossimEcefRay(
    var theOrigin: ossimEcefPoint = ossimEcefPoint(),
    var theDirection: ossimEcefVector = ossimEcefVector()
) {
    fun setOrigin(orig: ossimEcefPoint) {
        theOrigin = orig.copy()
    }

    fun setDirection(d: ossimEcefVector) {
        theDirection = d.unitVector()
    }

    fun origin(): ossimEcefPoint {
        return theOrigin.copy()
    }

    fun hasNans(): Boolean {
        return theOrigin.isNan() || theDirection.isNan()
    }

    fun direction(): ossimEcefVector {
        return theDirection
    }

    fun extend(t: Double): ossimEcefPoint {
        return (theOrigin + theDirection * t);


    }

    fun intersectAboveEarthEllipsoid(argHeight: Double, argDatum: ossimDatum = ossimWgs84Datum()): ossimEcefPoint {
        val datum = argDatum;
        val solution = ossimEcefPoint()
        val intersected = datum.ellipsoid().nearestIntersection(this, argHeight, solution)
        if (!intersected)
            solution.makeNan()
        return solution;
    }
}

class ossimKeywordlist(val config: Map<String, String>) {
    fun find(prefix: String, kw: String): String? = config[prefix + kw]
    fun getBoolKeywordValue(s: String, prefix: String): Boolean {
        return false
    }

    fun getSubstringKeyList(regExpression: String): List<String> {
        val regex = regExpression.toRegex()
        val filter = config.keys.filter { it.contains(regex) }.mapNotNull {
            regex.find(it)?.value
        }.distinct()
        return filter
    }

    fun findKey(key: String): String {
        return config.getOrDefault(key, "")
    }

    companion object {
        fun hardcodedConfig(srtmDataPath: String = "/Users/azhukov/personal/uav_pixel_to_coord/data/elevation/srtm_nasa"): ossimKeywordlist {
            val str = """
                elevation_manager.elevation_source0.connection_string: $srtmDataPath
                elevation_manager.elevation_source0.type: srtm_directory
                elevation_manager.elevation_source0.enabled: true
                elevation_manager.elevation_source0.geoid.type: identity

                elevation.enabled:  true
                elevation.auto_sort.enabled:  true
            """.trimIndent()
            val config = str.split("\n").asSequence().filter { it.isNotBlank() }
                .map { it.trim().split(": ", limit = 2) }
                .filter { it.size == 2 }
                .map { it[0].trim() to it[1].trim() }.toMap()
            return ossimKeywordlist(config)
        }
    }
}

data class ossimIpt(val x: Int = 0, val y: Int = 0) {
    fun hasNans(): Boolean {
        return x == OSSIM_INT_NAN || y == OSSIM_INT_NAN
    }
}

class ossimApplanixEcefModel(
    imageRect: ossimDrect,
    platformPosition: ossimGpt,
    private val theRoll: Double,
    private val thePitch: Double,
    private val theHeading: Double,
    principalPoint: ossimDpt,
    private val theFocalLength: Double,
    pixelSize: ossimDpt
) {
    private val theImageSize = ossimIpt()
    private val thePixelSize = pixelSize.copy()
    private val theImageClipRect = imageRect
    private val theRefImgPt: ossimDpt = theImageClipRect.midPoint()
    private val thePrincipalPoint = principalPoint.copy()
    private val theLensDistortion = ossimMeanRadialLensDistortion()
    private var theEcefPlatformPosition: ossimEcefPoint = ossimEcefPoint(platformPosition)
    private var theAdjEcefPlatformPosition: ossimEcefPoint = ossimEcefPoint(platformPosition);

    private var theCompositeMatrix = SimpleMatrix.diag(0.0, 0.0, 0.0, 0.0)!!
    private var theCompositeMatrixInverse = SimpleMatrix.diag(0.0)!!

    private var theBoundGndPolygon: ossimPolygon = ossimPolygon()
    private var theExtrapolateImageFlag: Boolean = false
    private var theExtrapolateGroundFlag: Boolean = false
    private var theGSD = ossimDpt()// meters
    private var theMeanGSD = 0.0 // meters

    init {
        initAdjustableParameters()
        updateModel()
        computeGsd();
    }

    private fun computeGsd() {
        if (theImageSize.hasNans()) {
            TODO("Not yet implemented")
        }
        var centerImagePoint: ossimDpt = theRefImgPt
        val quarterSize = if (!theImageSize.hasNans()) {
            ossimDpt(theImageSize.x / 4.0, theImageSize.y / 4.0)
        } else if (!theImageClipRect.hasNans()) {
            val w: Float = theImageClipRect.width().toFloat()
            val h: Float = theImageClipRect.height().toFloat()
            ossimDpt(w / 4.0, h / 4.0)
        } else {
            ossimDpt(1.0, 1.0)
        }

        if (centerImagePoint.hasNans() && (!theImageSize.hasNans())) {
            centerImagePoint.x = (theImageSize.x) / 2.0;
            centerImagePoint.y = (theImageSize.y - 1.0) / 2.0;
        } else if (centerImagePoint.hasNans() && !theImageClipRect.hasNans()) {
            centerImagePoint = theImageClipRect.midPoint();
        }
        if (!centerImagePoint.hasNans()) {
            val leftDpt = ossimDpt(centerImagePoint.x - quarterSize.x, centerImagePoint.y);//  (0.0,     midLine);
            val rightDpt = ossimDpt(centerImagePoint.x + quarterSize.y, centerImagePoint.y);// (endSamp, midLine);
            val topDpt = ossimDpt(centerImagePoint.x, centerImagePoint.y - quarterSize.y);//   (midSamp, 0.0);
            val bottomDpt = ossimDpt(centerImagePoint.x, centerImagePoint.y + quarterSize.y);//(midSamp, endLine);
            val leftGpt = ossimGpt()
            val rightGpt = ossimGpt()
            val topGpt = ossimGpt()
            val bottomGpt = ossimGpt()

            //---
            // Left point.
            // For the first point use lineSampleToWorld to get the height.
            //---
            lineSampleToWorld(leftDpt, leftGpt);
            if (leftGpt.hasNans()) {
                TODO()
            }
            //---
            // Right point:
            // Use lineSampleHeightToWorld using the left height since we want the horizontal distance.
            //---
            lineSampleHeightToWorld(rightDpt, leftGpt.hgt, rightGpt);
            if (rightGpt.hasNans()) {
                TODO()
            }
            //---
            // Top point:
            // Use lineSampleHeightToWorld using the left height since we want the horizontal distance.
            //---
            lineSampleHeightToWorld(topDpt, leftGpt.hgt, topGpt);
            if (topGpt.hasNans()) {
                TODO()
            }
            //---
            // Bottom point:
            // Use lineSampleHeightToWorld using the left height since we want the horizontal distance.
            //---
            lineSampleHeightToWorld(bottomDpt, leftGpt.hgt, bottomGpt);
            if (bottomGpt.hasNans()) {
                TODO()
            }
            theGSD.x = leftGpt.distanceTo(rightGpt) / (rightDpt.x - leftDpt.x);
            theGSD.y = topGpt.distanceTo(bottomGpt) / (bottomDpt.y - topDpt.y);
            theMeanGSD = (theGSD.x + theGSD.y) / 2.0;
        } else {
            TODO()
        }
    }

    private fun lineSampleHeightToWorld(image_point: ossimDpt, heightEllipsoid: Double, worldPoint: ossimGpt) {
        val ray = ossimEcefRay()
        imagingRay(image_point, ray);
        val Pecf = ossimEcefPoint(ray.intersectAboveEarthEllipsoid(heightEllipsoid));
        worldPoint.assign(ossimGpt.fromEcefPoint(Pecf));

    }

    private fun updateModel() {
        var gpt = ossimGpt()
        val wgs84Pt = ossimGpt()
        val metersPerDegree = wgs84Pt.metersPerDegree().x
        val degreePerMeter = 1.0 / metersPerDegree
        val latShift: Double = -computeParameterOffset(1) * theMeanGSD * degreePerMeter
        val lonShift: Double = computeParameterOffset(0) * theMeanGSD * degreePerMeter
        gpt = ossimGpt.fromEcefPoint(theEcefPlatformPosition);
        val height = gpt.height()
        gpt.height(height + computeParameterOffset(5));
        gpt.latd(gpt.latd() + latShift);
        gpt.lond(gpt.lond() + lonShift);
        theAdjEcefPlatformPosition = ossimEcefPoint(gpt);
        val lsrSpace =
            ossimLsrSpace(ossimGpt.fromEcefPoint(theAdjEcefPlatformPosition), theHeading + computeParameterOffset(4));

        // make a left handed roational matrix;
        val lsrMatrix = ossimMatrix4x4(lsrSpace.lsrToEcefRotMatrix());
        val rotx = ossimMatrix4x4.createRotationXMatrix(
            thePitch + computeParameterOffset(3),
            ossimCoordSysOrientMode.OSSIM_LEFT_HANDED
        )
        val roty = ossimMatrix4x4.createRotationYMatrix(
            theRoll + computeParameterOffset(2), ossimCoordSysOrientMode.OSSIM_LEFT_HANDED
        )
        val orientation = rotx * roty

        theCompositeMatrix = lsrMatrix.getData() * orientation
        theCompositeMatrixInverse = theCompositeMatrix.i();
        theBoundGndPolygon.resize(4);

        theExtrapolateImageFlag = false;
        theExtrapolateGroundFlag = false;

        lineSampleToWorld(theImageClipRect.ul(), gpt);//+ossimDpt(-w, -h), gpt);
        theBoundGndPolygon[0] = gpt;
        lineSampleToWorld(theImageClipRect.ur(), gpt);//+ossimDpt(w, -h), gpt);
        theBoundGndPolygon[1] = gpt;
        lineSampleToWorld(theImageClipRect.lr(), gpt);//+ossimDpt(w, h), gpt);
        theBoundGndPolygon[2] = gpt;
        lineSampleToWorld(theImageClipRect.ll(), gpt);//+ossimDpt(-w, h), gpt);
        theBoundGndPolygon[3] = gpt;
    }

    fun lineSampleToWorld(image_point: ossimDpt, gpt: ossimGpt) {
        if (image_point.hasNans()) {
            gpt.makeNan();
        } else {
            val ray = ossimEcefRay()
            imagingRay(image_point, ray);
            ossimElevManager.instance().intersectRay(ray, gpt)
        }
    }

    private fun imagingRay(image_point: ossimDpt, image_ray: ossimEcefRay) {
        val f1 = image_point - theRefImgPt
        f1.x *= thePixelSize.x;
        f1.y *= -thePixelSize.y;
        val film = f1 - thePrincipalPoint
        if (theLensDistortion.valid()) {
//            var filmOut: ossimDpt
//            theLensDistortion.undistort(film, filmOut)
//            film = filmOut
        }
        val cam_ray_dir = ossimColumnVector3d(
            film.x,
            film.y,
            -theFocalLength
        );
        var ecf_ray_dir = ossimEcefVector(theCompositeMatrix * cam_ray_dir);
        ecf_ray_dir = ecf_ray_dir * (1.0 / ecf_ray_dir.magnitude());
        image_ray.setOrigin(theAdjEcefPlatformPosition);
        image_ray.setDirection(ecf_ray_dir);
    }

    private fun computeParameterOffset(i: Int): Double {
        return params[i].computeOffset()
    }

    private fun initAdjustableParameters() {
        resizeAdjustableParameterArray(6);

        setAdjustableParameter(0, 0.0);
        setParameterDescription(0, "x_offset");
        setParameterUnit(0, "pixels");

        setAdjustableParameter(1, 0.0);
        setParameterDescription(1, "y_offset");
        setParameterUnit(1, "pixels");

        setAdjustableParameter(2, 0.0);
        setParameterDescription(2, "roll");
        setParameterUnit(2, "degrees");

        setAdjustableParameter(3, 0.0);
        setParameterDescription(3, "pitch");
        setParameterUnit(3, "degrees");

        setAdjustableParameter(4, 0.0);
        setParameterDescription(4, "heading");
        setParameterUnit(4, "degrees");

        setAdjustableParameter(5, 0.0);
        setParameterDescription(5, "altitude");
        setParameterUnit(5, "meters");


        setParameterSigma(0, 20.0);
        setParameterSigma(1, 20.0);
        setParameterSigma(2, .1);
        setParameterSigma(3, .1);
        setParameterSigma(4, .1);
        setParameterSigma(5, 50.0);
    }

    private fun setParameterSigma(i: Int, d: Double) {
        params[i].theSigma = d
    }

    private fun setParameterUnit(i: Int, s: String) {
        params[i].unit = s
    }

    private fun setParameterDescription(i: Int, s: String) {
        params[i].description = s
    }

    private var params: Array<AdjParam> = emptyArray()

    private fun resizeAdjustableParameterArray(sz: Int) {
        params = Array(sz) { AdjParam("", "", 0.0, 0.0) }
    }

    private fun setAdjustableParameter(i: Int, d: Double) {
        params[i].theParameter = d
    }

    fun setPrincipalPoint(ossimDpt: ossimDpt) {
        thePrincipalPoint.assign(ossimDpt)
    }
}
