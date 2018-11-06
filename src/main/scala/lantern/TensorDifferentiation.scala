package lantern

import scala.util.continuations._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import scala.virtualization.lms.common._
import scala.collection.mutable.ArrayBuffer
import scala.math._

trait TensorDsl extends DslOps with Diff {

  /**
    Memory Management:
      finally we used a temperate solution called "memory arena". The base code will claim a large piece of code for the whole program.
      internally, every malloc will borrow memory from this arena.
      By using getAllocMem and setAllocMem, we can selectively return a big trunk of memory after one iteration of training.
   **/

  class Timer (val index: Int){
    unchecked[Unit](s"clock_t begin_$index, end_$index; double time_spent_$index")
    def startTimer = { unchecked[Unit](s"begin_$index = clock()") }
    def stopTimer = { unchecked[Unit](s"end_$index = clock()") }
    def printElapsedTime = {
      unchecked[Unit](
        s"end_$index = clock(); printf(",
        "\"Time elapsed: %f\\n\", ",
        s"(double)(end_$index - begin_$index) / CLOCKS_PER_SEC)")
    }
  }

  object Timer {
    var index: Int = 0
    def apply(): Timer = {
      val timer = new Timer(index)
      index += 1
      timer
    }
  }

  def get_time() = unchecked[Double]("((double)clock() / CLOCKS_PER_SEC)")

  class Timer2 (index: Int) {
    unchecked[Unit](s"struct timeval begin_$index, end_$index, diff_$index")
    def startTimer = { unchecked[Unit](s"gettimeofday(&begin_$index, NULL)") }
    def getElapsedTime: Rep[Long] = {
      unchecked[Unit](s"gettimeofday(&end_$index, NULL)")
      unchecked[Unit](s"timeval_subtract(&diff_$index, &end_$index, &begin_$index);")
      unchecked[Long](s"((diff_$index.tv_sec * 1000000L) + (diff_$index.tv_usec))")
    }
  }

  object Timer2 {
    var index: Int = 0
    def apply(): Timer2 = {
      val timer = new Timer2(index)
      index += 1
      timer
    }
  }

  object Dataset {
    class DataLoader(name: String, train: Boolean, mean: Float, std: Float, dims: Seq[Int]) {

      val fd = open(s"../data/bin/${name}_${if (train) "train" else "test"}.bin")
      val len = filelen(fd)
      val data = mmap[Float](fd, len)
      val dLength = (len/4L).toInt

      val tfd = open(s"../data/bin/${name}_${if (train) "train" else "test"}_target.bin")
      val tlen = filelen(tfd)
      val target = mmap[Int](tfd, tlen)
      val length = tlen/4

      def dataset = new Tensor(data, Seq(60000, dims(1), dims(2)))

      @virtualize
      def normalize() = {
        this.foreach { (i, t, d) =>
          t.normalize(mean, std, inPlace = true)
        }
      }

      @virtualize
      def foreach(f: (Rep[Int], Tensor, Rep[Int]) => Unit) = {
        var off = var_new(0)
        for (index <- 0 until length: Rep[Range]) {
          val dataPtr = slice(data, off)
          val t = Tensor(dataPtr, dims : _*)
          f(index, t, target(index))
          off += t.scalarCount
        }
        assertC(off == dLength, "Data length doesn't match\\n")
      }

      @virtualize
      def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
        var off = var_new(0)
        for (batchIndex <- 0 until (length / batchSize): Rep[Range]) {
          val dataPtr = slice(data, off)
          val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
          val targets = slice(target, batchIndex * batchSize)
          f(batchIndex, t, targets)
          off += t.scalarCount
        }
      }
    }

    class Cifar10DataLoader(name: String, train: Boolean, dims: Seq[Int]) {

      val fd = open(name)
      val len = filelen(fd)
      val data = mmap[Char](fd, len)
      // each entry is target + image
      val entrySize = (dims.product + 1)
      val dLength = (len/entrySize.toLong).toInt
      val length = dLength

      val x = NewArray[Float](dLength * dims.product)
      val y = NewArray[Int](dLength)

      for (i <- (0 until dLength): Rep[Range]) {
        y(i) = unchecked[Int]("(int32_t)(unsigned char)", data(i * entrySize))
        for (j <- (0 until dims.product): Rep[Range]) {
          x(i * dims.product + j) = uncheckedPure[Float]("(float)(unsigned char)", data(i * entrySize + 1 + j)) / 255.0f
        }
      }

      @virtualize
      def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
        for (batchIndex <- 0 until (dLength / batchSize): Rep[Range]) {
          val dataPtr = slice(x, batchIndex * batchSize * dims.product)
          val targets = slice(y, batchIndex * batchSize)
          val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
          f(batchIndex, t, targets)
        }
      }
    }

    class DataLoaderTest(name_x: String, name_y: String, dims: Seq[Int]) {

      val fd = open(name_x)
      val len = filelen(fd)
      val data = mmap[Float](fd, len)
      val dLength = (len/4L).toInt

      val tfd = open(name_y)
      val tlen = filelen(tfd)
      val target = mmap[Int](tfd, tlen)
      val length = tlen/4

      @virtualize
      def foreachBatch(batchSize: Int)(f: (Rep[Int], Tensor, Rep[Array[Int]]) => Unit) = {
        var off = var_new(0)
        for (batchIndex <- 0 until (length / batchSize): Rep[Range]) {
          val dataPtr = slice(data, off)
          val t = Tensor(dataPtr, (batchSize +: dims.toSeq): _*)
          val targets = slice(target, batchIndex * batchSize)
          f(batchIndex, t, targets)
          off += t.scalarCount
        }
      }
    }
  }

  def convSize(size: Int, kernelSize: Int, strideSize: Int) = (size - kernelSize)/strideSize + 1
  def convSize(size: Int, kernelSize: Int, strideSize: Int, pad: Int) = (size + 2 * pad - kernelSize) / strideSize + 1
  def mmax(a: Int, b: Int) = if (a >= b) a else b

  @virtualize
  def assertC(cond: Rep[Boolean], msg: String, args: Rep[Any]*): Unit = {
    if(!cond) { printf(msg, args : _*); exit() }
  }

  def slice[T: Manifest](arr: Rep[Array[T]], off: Rep[Int]) = uncheckedPure[Array[T]](arr, "+", off)

  object Encoding {
    val ix_a = 96  // index starts from 1

    def char_to_ix(ch: Rep[Char]): Rep[Int] = ch.AsInstanceOf[Int] - ix_a
    def ix_to_char(ix: Rep[Int]): Rep[Char] = (ix + ix_a).AsInstanceOf[Char]
  }

  case class Dimensions(val dims: Seq[Int]) {
    def apply(idx: Int) = {
      if (!dims.indices.contains(idx)) throw new IndexOutOfBoundsException(s"Index $idx out of bounds")
      else dims(idx)
    }
    // `head` and `last` have default value 1 for scalar tensors.
    def head = dims.headOption.getOrElse(1)
    def last = dims.lastOption.getOrElse(1)
    def get(idx: Int) = if (!dims.indices.contains(idx)) 1 else dims(idx)
    def reverse = Dimensions(dims.reverse)

    val (scalarCount +: strides) = (dims :\ Seq[Int](1)) {
      case (dim, seq@(t +: q)) => (dim * t) +: seq
    }

    override def toString = dims mkString " x "
  }

  implicit def Dimensions2Seq(x: Dimensions) = x.dims

  object Random {
    def rand() = unchecked[Float]("(float)rand()/RAND_MAX")
    def srand(seed: Option[Int] = None) = unchecked[Unit]("srand(",seed.map(_.toString).getOrElse("time(NULL)"),")")
  }

  def exit() = unchecked[Unit]("exit(0)")

  abstract class DataLoop {
    def foreach(f: Rep[Int] => Unit): Unit
  }

  @virtualize
  object DataLoop {
    def apply(size: Int) = if (size <= 3) {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Range) f(unit(i))
        }
      }
    } else {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Rep[Range]) f(i)
        }
      }
    }
  }

  /**
    * A code generation backend for tensor operations.
    *
    * Note: Eventually, a tensor operation IR may be introduced to enable analyses and
    * transformations such as operator fusion and matrix chain multiplication optimization.
    */
  trait Backend extends Serializable {
    // Setup the backend. This is the first function that will be called in DSL programs.
    def setup(): Unit

    // Cleanup the backend. This is the last function that will be called in DSL programs.
    def cleanup(): Unit

    // Allocate an array with the specified length.
    def mallocArray[T: Manifest](length: Rep[Int]): Rep[Array[T]]

    // Copy data from one array to another.
    // NOTE: This function is intentionally not defined generically to simplify the codegen implementation.
    // The only user of this function is currently `copyTensorData`.
    def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Int): Unit

    // Copy data from one tensor to another.
    def copyTensorData(dest: Tensor, src: Tensor): Unit = {
      assert(dest.scalarCount == src.scalarCount,
        s"Tensors do not have same scalar count: ${dest.scalarCount}, ${src.scalarCount}")
      copyFloatArray(dest.data, src.data, dest.scalarCount)
    }

    // wrap array to tensor
    def arrayToTensor(array: Rep[Array[Float]], dims: Int*): Tensor

    // Initialize a tensor with the specified dimensions and scalar values.
    def makeTensor(dims: Seq[Int], scalars: Float*): Tensor

    // Initialize a tensor with the specified dimensions and repeated value.
    def fill(dims: Seq[Int], value: Rep[Float]): Tensor

    // Initialize a tensor with the specified bias tensor at the specified dimension.
    def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int): Tensor

    // Fill a tensor in-place with the specified value.
    def fillInPlace(x: Tensor, value: Rep[Float])

    // Initialize a tensor with scalars sampled from a zero-centered uniform distribution.
    // By default, the uniform distribution is over [-0.5, 0.5].
    def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor

    // Compute vector-vector dot product, i.e. inner product.
    // [V] dot [V] => [1] (scalar)
    def vectorVectorDot(x: Tensor, y: Tensor): Tensor

    // Compute matrix-vector dot product.
    // [M1 x M2] dot [M2] => [M1]
    def matrixVectorDot(x: Tensor, y: Tensor): Tensor

    // Compute matrix-matrix dot product.
    // [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def matrixMatrixDot(x: Tensor, y: Tensor): Tensor

    def dot(x: Tensor, y: Tensor): Tensor =
      (x.rank, y.rank) match {
        case (1, 1) => vectorVectorDot(x, y)
        case (2, 1) => matrixVectorDot(x, y)
        case (2, 2) => matrixMatrixDot(x, y)
        case _ => throw new IllegalArgumentException(s"Incompatible shapes: ${x.shape}, ${y.shape}")
      }

    def dot_grad(x: TensorR, y: TensorR, output: TensorR): Unit

    // Elementwise addition.
    def +(x: Tensor, y: Rep[Float]): Tensor
    def +(x: Tensor, y: Tensor): Tensor

    // In-place elementwise addition.
    def +=(x: Tensor, y: Rep[Float]): Unit
    def +=(x: Tensor, y: Tensor): Unit

    // Elementwise subtraction.
    def -(x: Tensor, y: Rep[Float]): Tensor
    def -(x: Tensor, y: Tensor): Tensor

    // In-place elementwise subtraction.
    def -=(x: Tensor, y: Rep[Float]): Unit
    def -=(x: Tensor, y: Tensor): Unit

    // Elementwise multiplication.
    def *(x: Tensor, y: Rep[Float]): Tensor
    def *(x: Tensor, y: Tensor): Tensor

    // In-place elementwise multiplication.
    def *=(x: Tensor, y: Rep[Float]): Unit
    def *=(x: Tensor, y: Tensor): Unit

    // Elementwise division.
    def /(x: Tensor, y: Rep[Float]): Tensor
    def /(x: Tensor, y: Tensor): Tensor

    // In-place elementwise division.
    def /=(x: Tensor, y: Rep[Float]): Unit
    def /=(x: Tensor, y: Tensor): Unit

    // Why do we have plusBias and what is the difference of plusBias with elementWise + with broadcasting
    // Ans: plusBias is less general than elementwise + with broadcasting, since it is assume that
    // the bias may be broadcasted, while the other tensor (call it main tensor) doesn't need to.
    // That resulted in easier implementation in cuDNN API calls.
    // It also carries the assumption that the main tensor is not used by other ops until it was added to the bias,
    // so an optimization can be done, such that plusBias is in-place (directly changing the main tensor).
    def plusBias(main: Tensor, bias: Tensor): Tensor
    def plusBias_grad(main: TensorR, bias: TensorR): Unit

    // output = x * alpha + y * beta (can be in-place if output is x or y)
    def geam(x: Tensor, transX: Boolean, alpha: Float, y: Tensor, transY: Boolean, beta: Float, output: Tensor): Unit
    def trans(x: Tensor): Tensor
    def trans_grad(x: TensorR, y: TensorR): Unit

    def gemm(x: Tensor, transX: Boolean, y: Tensor, transY: Boolean, alpha: Float): Tensor
    def gemm_grad(x: TensorR, transX: Boolean, y: TensorR, transY: Boolean, alpha: Float, output: TensorR): Unit

    /**
      * 2D convolution.
      * @param input Input with shape [batchSize, inChannels, iW, iH].
      * @param kernel Kernel with shape [outChannels, inChannels, kW, kH].
      * @param bias Optional bias with shape [outChannels].
      * @param strides Kernel strides of length two: [strideWidth, strideHeight].
      * @param pads Padding of length four: [padTop, padBottom, padLeft, padRight].
      * @return Result of 2D convolution.
      */
    // NOTE: cuDNN accepts only two padding arguments: [padVertical, padHorizontal].
    def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor])

    @virtualize
    def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                          padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit

    def maxPool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]])
    def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]], kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit

    def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor
    def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit

    def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor
    def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor])
    def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor]): Unit

    def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int])
    def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit

    // Activation functions.
    def relu(x: Tensor): Tensor
    def tanh(x: Tensor): Tensor
    def sigmoid(x: Tensor): Tensor
    def exp(x: Tensor): Tensor
    def log(x: Tensor): Tensor
    def sqrt(x: Tensor): Tensor
    def square(x: Tensor): Tensor

    def relu_grad(input: TensorR, res: TensorR): Unit
    def tanh_grad(input: TensorR, res: TensorR): Unit
    def sigmoid_grad(input: TensorR, res: TensorR): Unit
    def exp_grad(input: TensorR, res: TensorR): Unit
    def log_grad(input: TensorR, res: TensorR): Unit
    def sqrt_grad(input: TensorR, res: TensorR): Unit
    def square_grad(input: TensorR, res: TensorR): Unit

    // Softmax functions.
    def softmax(x: Tensor): Tensor
    def logSoftmax(x: Tensor): Tensor

    def softmax_grad(input: TensorR, res: TensorR): Unit
    def logSoftmax_grad(input: TensorR, res: TensorR): Unit

    // Loss functions.
    def nllLoss(x: Tensor, target: Rep[Array[Int]]): Tensor
    def nllLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Int]]): Unit

    // Reduction operations.
    def sum(x: Tensor): Tensor
    def sum_grad(input: TensorR, res: TensorR): Unit
    def mean(x: Tensor): Tensor
    def mean_grad(input: TensorR, res: TensorR): Unit

    // concatenate and split
    def concat(dim: Int, tensors: Seq[Tensor]): Tensor
    def concat_grad(dim: Int, tensorRs: Seq[TensorR], output: TensorR): Unit

    // TODO: Add more ops:
    // - Reduction operators (e.g. sum).
    //   - Reduction op GPU implementations are non-trivial.
    //   - Roll out own reduction op kernels? There may be significant boilerplate.
    //   - Use thrust library reduction ops? Need to consider device_vector initialization overhead.
    // - Fused multiply add operations?
  }

  /**
    * CPU tensor operation backend. WIP.
    * Tensor ops are defined in terms of primitive operations.
    */
  class BackendCPU protected() extends Backend {
    override def setup() {}
    override def cleanup() {}
    override def mallocArray[T: Manifest](length: Rep[Int]): Rep[Array[T]] = NewArray[T](length)

    override def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Int): Unit = {
      for (i <- DataLoop(length)) dest(i) = src(i)
    }

    override def arrayToTensor(array: Rep[Array[Float]], dims: Int*): Tensor = new Tensor(array, dims)

    override def makeTensor(dims: Seq[Int], scalars: Float*): Tensor = {
      Tensor(Array(scalars.map(unit(_)): _*), dims: _*)
    }

    override def fill(dims: Seq[Int], value: Rep[Float]): Tensor = {
      val scalarCount = dims.product
      val array = mallocArray[Float](scalarCount)
      for (i <- DataLoop(scalarCount)) array(i) = value
      Tensor(array, dims: _*)
    }

    // TODO (Optimizations) (Fei Wang): bias has the feature that the values before bias is never used otherwise
    // The consequence is that add bias can be done in-place with broadcasting
    // and backprop to bias can be done by += with reduction
    // In that sense, this function should be removed, and we should use plusBias/plusBias_grad instead
    override def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int): Tensor = {
      assert(dim >= 0 && dim < dims.size, s"Target dimension $dim is out of range $dims")
      assert(bias.rank == 1 && bias.scalarCount == dims(dim),
        "Bias must be 1D and have length equal to the target dimension")
      val scalarCount = dims.product
      val res = mallocArray[Float](scalarCount)

      // iterate for higherDims
      val offset = var_new(0)
      for (hd <- DataLoop(dims.take(dim).product)) {
        // iterate for current dim
        for (cd <- DataLoop(dims.drop(dim).head)) {
          // iterate for lowerDims
          for (ld <- DataLoop(dims.drop(dim+1).product)) {
            res(offset) = bias.data(cd)
            offset += 1
          }
        }
      }
      Tensor(res, dims: _*)
    }

    // TODO (Optimization) (Fei Wang): It is advisable that all mapping like functions (fillInPlace, map, mapInplace)
    // should take a function/closure that starts from index (i => compute_value_at_pos_i)
    override def fillInPlace(x: Tensor, value: Rep[Float]): Unit = {
      for (i <- DataLoop(x.scalarCount)) x.data(i) = value
    }

    override def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor = {
      seed match {
        case None => ()
        case Some(seed) => Random.srand(Some(seed))
      }
      val scalarCount = dims.product
      val res = mallocArray[Float](scalarCount)
      for (i <- DataLoop(scalarCount)) res(i) = (Random.rand() - 0.5f) * scale
      new Tensor(res, dims)
    }

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(0) == y.shape(0))
      val value = var_new(0.0f)
      for (i <- DataLoop(x.shape.last)) {
        value += x.data(i) * y.data(i)
      }
      val res = mallocArray[Float](1)
      res(0) = readVar(value)
      Tensor(res, 1)
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(1) == y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val res = mallocArray[Float](dim1)
      unchecked[Unit] (
        "cblas_sgemv(CblasRowMajor, CblasNoTrans, ",
        dim1, ",", dim2, ",", 1, ",",
        x.data, ",", dim2, ",", y.data, ",", 1, ",", 0, ",", res, ",", 1, ")")
      Tensor(res, dim1)
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      assert(x.shape(1) == y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val dim3 = y.shape(1)
      val res = mallocArray[Float](dim1 * dim3)
      unchecked[Unit](
        "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
        dim1, ",", dim3, ",", dim2, ",", 1, ",",
        x.data, ",", dim2, ",", y.data, ",", dim3, ",", 0, ",", res, ",", dim3, ")")
      Tensor(res, dim1, dim3)
    }

    override def dot_grad(x: TensorR, y: TensorR, output: TensorR): Unit = {
      (x.x.rank, y.x.rank) match {
        case (1, 1) => x.d.addMul(output.d.data(0), y.x); y.d.addMul(output.d.data(0), x.x)
        case (2, 1) => x.d.add_cartesian(y.x, output.d); // that.d.add_composion(this.x, y.d)
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1)
          unchecked[Unit](
            "cblas_sgemv(CblasRowMajor, CblasTrans, ",
            dim1, ",", dim2, ",", 1, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", 1, ",", 1, ",", y.d.data, ",", 1, ")")
        case (2, 2) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          generateRawComment("backprop of matrix-matrix-dot")
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim1, ",", dim2, ",", dim3, ",", 1, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim2, ")")
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim2, ",", dim3, ",", dim1, ",", 1, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", 1, ",", y.d.data, ",", dim3, ")")
      }
    }

    // TODO (Optimize) (Fei Wang): adapt it so that it can be done with a kernel function (index => f(index))
    // nested loop is smart, but not elegant
    def elementWiseOpWithBroadCast(x: Tensor, y: Tensor, op: ((Rep[Float], Rep[Float]) => Rep[Float])) = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case None => throw new IllegalArgumentException(s"dimensions of vector do not match! ${x.shape.seq} != ${y.shape.seq}")
        case Some((xShape, yShape, resShape)) => {
          val resData = mallocArray[Float](resShape.scalarCount)
          val res = new Tensor(resData, resShape)

          def inplace(offX: Rep[Int], offY: Rep[Int], offRes: Rep[Int], dim: Int): Unit = {
            val offres = var_new[Int](offRes)
            val offx = var_new[Int](offX)
            val offy = var_new[Int](offY)

            // Handle scalar tensor case.
            // TODO: Unify with logic below.
            if (resShape.isEmpty) {
              resData(offres) = op(x.data(offx), y.data(offy))
              return
            }

            for (i <- DataLoop(resShape(dim))) {
              if (dim == resShape.size - 1) {
                resData(offres) = op(x.data(offx), y.data(offy))
              } else {
                inplace(offx, offy, offres, dim + 1)
              }
              offres += resShape.strides(dim)
              if (xShape(dim) > 1) offx += xShape.strides(dim)
              if (yShape(dim) > 1) offy += yShape.strides(dim)
            }
          }
          inplace(0, 0, 0, 0)
          res
        }
      }
    }

    // TODO (Optimize) (Fei Wang): adapt it so that it can be done with a kernel function (index => f(index))
    // nested loop is smart, but not elegant
    def backpropElementWiseOpWithBroadCast(in1: TensorR, that: TensorR, y: TensorR, opThis: ((Rep[Float], Rep[Float], Rep[Float]) => Rep[Float]), opThat: ((Rep[Float], Rep[Float], Rep[Float]) => Rep[Float])): Unit = {
      // assume y.x = elementWiseOpWithBroadCast(this.x, that.x, someOp)
      // assume that opThis returns the increment of this.d; opThat returns the increment of that.d
      // both opThis and opThat takes this.x, that.x, and y.d as parameters
      // TODO (Fei Wang): in some cases (if this, or that are inputs (x, y), there gradients are not initialized/useless)
      Tensor.dimBroadcast(in1.x.shape, that.x.shape) match {
        case None => throw new IllegalArgumentException(s"dimensions of tensors do not match! ${in1.x.shape.seq} != ${that.x.shape.seq}")
        case Some((thisShape, thatShape, yShape)) => {
          def inplace(offThis: Rep[Int], offThat: Rep[Int], offY: Rep[Int], dim: Int): Unit = {
            val offthis = var_new[Int](offThis)
            val offthat = var_new[Int](offThat)
            val offy = var_new[Int](offY)

            // Handle scalar tensor case.
            // TODO: Unify with logic below.
            if (yShape.isEmpty) {
              in1.d.data(offthis) += opThis(in1.x.data(offthis), that.x.data(offthat), y.d.data(offy))
              that.d.data(offthat) += opThat(in1.x.data(offthis), that.x.data(offthat), y.d.data(offy))
              return
            }

            for (i <- DataLoop(yShape(dim))) {
              if (dim == yShape.size - 1) {
                in1.d.data(offthis) += opThis(in1.x.data(offthis), that.x.data(offthat), y.d.data(offy))
                that.d.data(offthat) += opThat(in1.x.data(offthis), that.x.data(offthat), y.d.data(offy))
              } else {
                inplace(offthis, offthat, offy, dim + 1)
              }
              offy += yShape.strides(dim)
              if (thisShape(dim) > 1) offthis += thisShape.strides(dim)
              if (thatShape(dim) > 1) offthat += thatShape.strides(dim)
            }
          }
          inplace(0, 0, 0, 0)
        }
      }
    }

    // TODO (Optimize) (Fei Wang): adapt it so that it can be done with a kernel function (index => f(index))
    // nested loop is smart, but not elegant
    // x += y (with potentially broadcasting y, or reducing y (reverse of broadcasting x))
    def inplaceElementWiseOpWithBroadCastOrReduce(x: Tensor, y: Tensor, op: ((Rep[Float], Rep[Float]) => Rep[Float])): Unit = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case None => throw new IllegalArgumentException(s"dimensions of vector do not match! ${x.shape.seq} != ${y.shape.seq}")
        case Some((xShape, yShape, resShape)) => {

          def inplace(offX: Rep[Int], offY: Rep[Int], offRes: Rep[Int], dim: Int): Unit = {
            val offres = var_new[Int](offRes)
            val offx = var_new[Int](offX)
            val offy = var_new[Int](offY)
            for (i <- DataLoop(resShape(dim))) {
              if (dim == resShape.size - 1) {
                x.data(offx) = op(x.data(offx), y.data(offy))
              } else {
                inplace(offx, offy, offres, dim + 1)
              }
              offres += resShape.strides(dim)
              if (xShape(dim) > 1) offx += xShape.strides(dim)
              if (yShape(dim) > 1) offy += yShape.strides(dim)
            }
          }
          inplace(0, 0, 0, 0)
        }
      }
    }

    override def plusBias(main: Tensor, bias: Tensor): Tensor = {
      this.inplaceElementWiseOpWithBroadCastOrReduce(main, bias, (_ + _))
      main
    }

    override def plusBias_grad(main: TensorR, bias: TensorR): Unit = {
      this.inplaceElementWiseOpWithBroadCastOrReduce(bias.d, main.d, (_ + _))
    }

    override def +(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s + y)
    override def +(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ + _)

    override def +=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s + y)
    override def +=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ + _))

    override def -(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s - y)
    override def -(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ - _)

    override def -=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s - y)
    override def -=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ - _))

    override def *(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s * y)
    override def *(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ * _)

    override def *=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s * y)
    override def *=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ * _))

    override def /(x: Tensor, y: Rep[Float]): Tensor = x.map(s => s / y)
    override def /(x: Tensor, y: Tensor): Tensor = elementWiseOpWithBroadCast(x, y, _ / _)

    override def /=(x: Tensor, y: Rep[Float]): Unit = x.mapInPlace(s => s / y)
    override def /=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ / _))

    override def geam(x: Tensor, transX: Boolean, alpha: Float, y: Tensor, transY: Boolean,  beta: Float, output: Tensor): Unit = {
      (transX, transY) match {
        case (false, false) => output.changeTo { i => x.data(i) * alpha + y.data(i) * beta }
        case _ => ???
      }
    }

    override def trans(x: Tensor): Tensor = {
      assert(x.rank == 2, "transpose is only for matrix. Tensor transpose is not supported here")
      val res = backend.mallocArray[Float](x.scalarCount)
      val offT = var_new(0)
      for (i <- DataLoop(x.shape(1))) {
        val off = var_new(0)
        for (j <- DataLoop(x.shape(0))) {
          res(offT + j) = x.data(off + i)
          off += x.shape(1)
        }
        offT += x.shape(0)
      }
      new Tensor(res, x.shape.reverse)
    }

    override def trans_grad(x: TensorR, y: TensorR): Unit = {
      val offT = var_new(0)
      for (i <- DataLoop(x.x.shape(1))) {
        val off = var_new(0)
        for (j <- DataLoop(x.x.shape(0))) {
          x.d.data(off + i) += y.d.data(offT + j)
          off += x.x.shape(1)
        }
        offT += x.x.shape(0)
      }
    }

    override def gemm(x: Tensor, transX: Boolean, y: Tensor, transY: Boolean, alpha: Float): Tensor = {
      (transX, transY) match {
        case (false, false) =>
          assert(x.shape(1) == y.shape(0))
          val dim1 = x.shape(0)
          val dim2 = x.shape(1)
          val dim3 = y.shape(1)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim2, ",", y.data, ",", dim3, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
        case (false, true) =>
          assert(x.shape(1) == y.shape(1))
          val dim1 = x.shape(0)
          val dim2 = x.shape(1)
          val dim3 = y.shape(0)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim2, ",", y.data, ",", dim2, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
        case (true, false) =>
          assert(x.shape(0) == y.shape(0), s"gemm dims don't match, got ${x.shape.seq}, ${y.shape.seq}")
          val dim1 = x.shape(1)
          val dim2 = x.shape(0)
          val dim3 = y.shape(1)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim1, ",", y.data, ",", dim3, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
        case (true, true) =>
          assert(x.shape(0) == y.shape(1))
          val dim1 = x.shape(1)
          val dim2 = x.shape(0)
          val dim3 = y.shape(0)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim1, ",", y.data, ",", dim2, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
      }
    }

    override def gemm_grad(x: TensorR, transX: Boolean, y: TensorR, transY: Boolean, alpha: Float, output: TensorR): Unit = {
      generateRawComment(s"backprop of gemm ${x.x.shape.seq}, ${transX}, ${y.x.shape.seq}, ${transY}")
      (transX, transY) match {
        case (false, false) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim1, ",", dim2, ",", dim3, ",", alpha, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim2, ")")
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim2, ",", dim3, ",", dim1, ",", alpha, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", 1, ",", y.d.data, ",", dim3, ")")
        case (false, true) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(0)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
            dim1, ",", dim2, ",", dim3, ",", alpha, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim2, ",", 1, ",", x.d.data, ",", dim2, ")")
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim3, ",", dim2, ",", dim1, ",", alpha, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim2, ",", 1, ",", y.d.data, ",", dim2, ")")
        case (true, false) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(1)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim2, ",", dim1, ",", dim3, ",", alpha, ",",
            y.x.data, ",", dim3, ",", output.d.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim1, ")")
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
            dim2, ",", dim3, ",", dim1, ",", alpha, ",",
            x.x.data, ",", dim1, ",", output.d.data, ",", dim3, ",", 1, ",", y.d.data, ",", dim3, ")")
        case (true, true) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(0)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ",
            dim2, ",", dim1, ",", dim3, ",", alpha, ",",
            y.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim1, ")")
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ",
            dim3, ",", dim2, ",", dim1, ",", alpha, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim1, ",", 1, ",", y.d.data, ",", dim2, ")")
      }
    }

    // implementation of Conv2D following Pytorch's idea (transform conv2d into matrix-matrix-dot, and use OpenBLAS)
    // https://github.com/pytorch/pytorch/blob/0a8c8c1dbead2f845e524ae32c19167d80363148/aten/src/THNN/generic/SpatialConvolutionMM.c
    type RAF = Rep[Array[Float]]
    def memsetFloatZero(where: RAF, howmany: Rep[Int]) = {
      unchecked[Unit]("memset(", where, ", 0, 4 * ", howmany, ");")
    }
    def memcpyFloat(dst: RAF, src: RAF, howmany: Rep[Int]) = {
      unchecked[Unit]("memcpy(", dst, ", ", src, ", 4 * ", howmany, ");")
    }

    def unfoldedCopy(finput: RAF, input: RAF, kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
    nInputPlane: Int, inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) {
      for (k <- (0 until nInputPlane * kH * kW): Rep[Range]) {
        val nip = k / (kH * kW)
        val rest = k % (kH * kW)
        val kh = rest / kW
        val kw = rest % kW
        val dst = slice(finput, nip*kH*kW*outputHeight*outputWidth + kh*kW*outputHeight*outputWidth + kw*outputWidth*outputWidth)
        val src = slice(input,  nip*inputHeight*inputWidth)
        if (padW > 0 || padH > 0) {
          for (y <- (0 until outputHeight): Rep[Range]) {
            val iy = y * dH - padH + kh
            __ifThenElse ((iy < 0 || iy >= inputHeight), {
              memsetFloatZero(slice(dst, y*outputWidth), outputWidth); ()
            }, {
              if (dW == 1) {
                val ix = 0 - padW + kw;
                val lpad = __ifThenElse ((padW-kw > 0), padW-kw, 0)
                val rpad = __ifThenElse ((padW-(kW-kw-1) > 0), padW-(kW-kw-1), 0)
                __ifThenElse ((outputWidth-rpad-lpad <= 0), {
                  memsetFloatZero(slice(dst, y*outputWidth), outputWidth)
                }, {
                  __ifThenElse ((lpad > 0), memsetFloatZero(slice(dst, y*outputWidth), lpad), ())
                  generateRawComment("may have segfault here")
                  memcpyFloat(slice(dst, y*outputWidth+lpad), slice(src, iy*inputWidth+ix+lpad), outputWidth-rpad-lpad)
                  __ifThenElse ((rpad > 0), memsetFloatZero(slice(dst, y*outputWidth+outputWidth-rpad), rpad), ())
                })
              } else {
                for (x <- (0 until outputWidth): Rep[Range]) {
                  val ix = x * dW - padW + kw
                  __ifThenElse ((ix < 0 || ix >= inputWidth), memsetFloatZero(slice(dst, y*outputWidth+x), 1),
                    memcpyFloat(slice(dst, y*outputWidth+x), slice(src, iy*inputWidth+ix), 1))
                }
              }
            })
          }
        } else {
          for (y <- (0 until outputHeight): Rep[Range]) {
            val iy = y * dH + kh
            val ix = kw
            if (dW == 1) memcpyFloat(slice(dst, y*outputWidth), slice(src, iy*inputWidth+ix), outputWidth)
            else for (x <- (0 until outputWidth): Rep[Range])
              memcpyFloat(slice(dst, y*outputWidth+x), slice(src, iy*inputWidth+ix+x*dW), 1)
          }
        }
      }
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) = {
      val ((dH:Int) :: (dW:Int) :: Nil) = strides.take(2).toList
      val ((padH:Int) :: (_:Int) :: (padW:Int) :: (_:Int) :: Nil) = pads.take(4).toList
      val nOutputPlane = kernel.shape(0)
      val kH = kernel.shape(2)
      val kW = kernel.shape(3)
      val batchSize = input.shape(0)
      val nInputPlane = input.shape(1)
      val inputHeight = input.shape(2)
      val inputWidth = input.shape(3)
      val outputHeight = (inputHeight + 2*padH - kH) / dH + 1
      val outputWidth  = (inputWidth + 2*padW - kW) / dW + 1
      val output = bias match {
          case Some(bias) => Tensor.fillWithBias(Seq(input.shape(0), kernel.shape(0), outputHeight, outputWidth), bias, 1)
          case None => Tensor.zeros(input.shape(0), kernel.shape(0), outputHeight, outputWidth)
        }
      val finput = Tensor.zeros(batchSize, kW * kH * nInputPlane, outputHeight * outputWidth)
      for (t <- (0 until batchSize): Rep[Range]) {
        val input_t = input(t).data
        val output_t = output(t).data
        val finput_t = finput(t).data
        ConvOutputFrame(input_t, output_t, kernel.data, finput_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight)
      }
      (output, Some(finput))
    }

    def ConvOutputFrame(input: RAF, output: RAF, weight: RAF, finput: RAF, kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int,
      nInputPlane: Int, inputWidth: Int, inputHeight: Int, nOutputPlane: Int, outputWidth: Int, outputHeight: Int) {

      unfoldedCopy(finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
      // finput viewed as: kW*kH*nInputPlane, outputHeight * outputWidth
      // input  viewed as: nInputPlane, inputWidth, inputHeight
      val dim1 = nOutputPlane
      val dim2 = kW * kH *nInputPlane
      val dim3 = outputHeight * outputWidth
      unchecked[Unit](
        "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
        dim1, ",", dim3, ",", dim2, ",", 1, ",",
        weight, ",", dim2, ",", finput, ",", dim3, ",", 1, ",", output, ",", dim3, ")")
    }

    // Gradient of `conv2D_batch`.
    @virtualize
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      // NOTE: Strides/paddings may be in the wrong order.
      assert(dilations._1 == 1 && dilations._2 == 1, "Currently, only dilations of 1 are supported")
      val finputR: TensorR = finput match {
        case None => assert(false, "BackendCPU needs finput to be Some[TensorR], found None"); TensorR(Tensor.zeros(1))
        case Some(finputr) => finputr
      }

      // back-propagate to inputs
      if (!input.isInput) ConvGradInput(res.d, input.d, finputR.d, filter.x, strides._1, strides._2, padding._1, padding._2)
      // back-propagate to weights
      bias match {
        case None => ConvGradParam(finputR.x, res.d, filter.d, None, strides._1, strides._2, padding._1, padding._2)
        case Some(bias) => ConvGradParam(finputR.x, res.d, filter.d, Some(bias.d), strides._1, strides._2, padding._1, padding._2)
      }
    }

    def ConvGradParam(finput: Tensor, gradOutput: Tensor, gradWeight: Tensor, gradBias: Option[Tensor], dH: Int, dW: Int, padH: Int, padW: Int, scale: Float = 1.0f) = {
      val nInputPlane = gradWeight.shape(1)
      val kH = gradWeight.shape(2)
      val kW = gradWeight.shape(3)
      val batchSize = gradOutput.shape(0)
      val nOutputPlane = gradOutput.shape(1)
      val outputHeight = gradOutput.shape(2)
      val outputWidth = gradOutput.shape(3)
      for (t <- (0 until batchSize): Rep[Range]) {
        val gradOutput_t = gradOutput(t).data
        val finput_t = finput(t).data
        val dim1 = nOutputPlane
        val dim2 = outputWidth * outputHeight
        val dim3 = kW * kH * nInputPlane
        unchecked[Unit](
          "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
          dim1, ",", dim3, ",", dim2, ",", scale, ",",
          gradOutput_t, ",", dim2, ",", finput_t, ",", dim2, ",", 1, ",", gradWeight.data, ",", dim3, ")")
        gradBias match {
          case None => ()
          case Some(gradBias) =>
            for (i <- (0 until nOutputPlane): Rep[Range]) {
              val sum = var_new(0.0f)
              val data = slice(gradOutput_t, i * outputWidth * outputHeight)
              for (k <- (0 until outputWidth * outputHeight): Rep[Range]) {
                sum += data(k)
              }
              gradBias.data(i) += scale * sum
            }
        }
      }
    }

    def ConvGradInput(gradOutput: Tensor, gradInput: Tensor, fgradInput: Tensor, weight: Tensor, dH: Int, dW: Int, padH: Int, padW: Int) = {
      val batchSize = gradInput.shape(0)
      val inputHeight = gradInput.shape(2)
      val inputWidth = gradInput.shape(3)
      val nOutputPlane = weight.shape(0)
      val nInputPlane = weight.shape(1)
      val kH = weight.shape(2)
      val kW = weight.shape(3)
      val outputHeight = gradOutput.shape(2)
      val outputWidth = gradOutput.shape(3)
      for (t <- DataLoop(batchSize)) {
        val gradInput_t = gradInput(t).data
        val gradOutput_t = gradOutput(t).data
        val fgradInput_t = fgradInput(t).data
        val dim1 = kW * kH * nInputPlane
        val dim2 = nOutputPlane
        val dim3 = outputHeight * outputWidth
        unchecked[Unit](
          "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
          dim1, ",", dim3, ",", dim2, ",", 1, ",",
          weight.data, ",", dim1, ",", gradOutput_t, ",", dim3, ",", 0, ",", fgradInput_t, ",", dim3, ")")
        unfoldedAcc(fgradInput_t, gradInput_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
      }
    }

    def unfoldedAcc(finput: RAF, input: RAF, kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int, nInputPlane: Int, inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) {
      for (nip <- (0 until nInputPlane): Rep[Range]) {
        for (kh <- (0 until kH): Rep[Range]) {
          for (kw <- (0 until kW): Rep[Range]) {
            val src = slice(finput, nip*kH*kW*outputHeight*outputWidth + kh*kW*outputHeight*outputWidth + kw*outputHeight*outputWidth)
            val dst = slice(input, nip*inputHeight*inputWidth)
            if (padW > 0 || padH > 0) {
              for (y <- (0 until outputHeight): Rep[Range]) {
                val iy: Rep[Int] = y * dH - padH + kh
                __ifThenElse ((iy < 0 || iy >= inputHeight), (), {
                  if (dW == 1) {
                    val ix: Rep[Int] = 0 - padW + kw
                    val lpad: Rep[Int] = __ifThenElse((padW-kw > 0), padW-kw, 0)
                    val rpad: Rep[Int] = __ifThenElse((padW-(kW-kw-1) > 0), padW-(kW-kw-1), 0)
                    val dst_slice = slice(dst, iy*inputWidth+ix+lpad)
                    val src_slice = slice(src, y*outputWidth+lpad)
                    for (i <- 0 until (outputWidth - lpad - rpad)) dst_slice(i) += src_slice(i)
                  } else {
                    for (x <- (0 until outputWidth): Rep[Range]) {
                      val ix = x*dW - padW + kw
                      __ifThenElse ((ix < 0 || ix >= inputWidth), (), dst(iy*inputWidth+ix) += src(y*outputWidth+x))
                    }
                  }
                  ()
                })
              }
            } else {
              for (y <- (0 until outputHeight): Rep[Range]) {
                val iy = y*dH + kh
                val ix = kw
                if (dW == 1) {
                  val dst_slice = slice(dst, iy*inputWidth+ix)
                  val src_slice = slice(src, y*outputWidth)
                  for (i <- (0 until outputWidth): Rep[Range]) dst_slice(i) += src_slice(i)
                } else {
                  for (x <- (0 until outputWidth): Rep[Range]) {
                    dst(iy*inputWidth+ix+x*dW) += src(y*outputWidth+x)
                  }
                }
              }
            }
          }
        }
      }
    }

    @virtualize
    override def relu(x: Tensor): Tensor = {
      val res = mallocArray[Float](x.scalarCount)
      for (i <- 0 until x.scalarCount: Rep[Range]) {
        if (x.data(i) < 0.0f)
          res(i) = 0.0f
        else
          res(i) = x.data(i)
      }
      Tensor(res, x.shape.seq : _*)
    }

    @virtualize
    override def relu_grad(input: TensorR, res: TensorR): Unit = {
      for (i <- 0 until input.x.scalarCount: Rep[Range]) {
        input.d.data(i) = if (input.x.data(i) < 0.0f) 0.0f else res.d.data(i)
      }
    }

    override def tanh(x: Tensor) = x.map(s => Math.tanh(s).toFloat)
    override def tanh_grad(input: TensorR, res: TensorR): Unit = {
      input.d.add_oneMinusSquare_mult(res.x, res.d)
    }

    override def sigmoid(x: Tensor) = x.map(s => 1.0f / (Math.exp(-1.0f * s).toFloat + 1.0f))
    override def sigmoid_grad(input: TensorR, res: TensorR): Unit = {
      input.d.add_oneMinusThenMult_mult(res.x, res.d)
    }

    def buildTensor(dims: Seq[Int], byIndex: Rep[Int] => Rep[Float]): Tensor = {
      val res = this.mallocArray[Float](dims.product)
      for (i <- DataLoop(dims.product)) res(i) = byIndex(i)
      Tensor(res, dims: _*)
    }

    override def exp(x: Tensor) = buildTensor(x.shape, i => Math.exp(x.data(i)).toFloat)
    override def exp_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) * y.x.data(i) }

    override def log(x: Tensor) = buildTensor(x.shape, i => Math.log(x.data(i)).toFloat)
    override def log_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) / x.x.data(i) }

    override def sqrt(x: Tensor) = buildTensor(x.shape, i => Math.sqrt(x.data(i)).toFloat)
    override def sqrt_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) / y.x.data(i) / 2.0f }

    override def square(x: Tensor) = buildTensor(x.shape, {i => val t = x.data(i); t * t})
    override def square_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) * x.x.data(i) * 2.0f }

    @virtualize
    override def softmax(x: Tensor): Tensor = {
      assert(x.rank == 2, "Softmax input must be 2-D: [batchSize, logits]")
      val max = x.max2D(dim = 1)
      val res = Tensor.zeros_like(x)
      val offset = var_new(0)
      for (batch <- DataLoop(x.shape(0))) {
        for (i <- DataLoop(x.shape(1))) {
          res.data(offset) = Math.exp(x.data(offset) - max.data(batch)).toFloat
          offset += 1
        }
      }
      val sum = res.sum(dim = 1)
      offset = 0
      for (batch <- DataLoop(res.shape(0))) {
        for (i <- DataLoop(res.shape(1))) {
          res.data(offset) = res.data(offset) / sum.data(batch)
          offset += 1
        }
      }
      res
    }

    @virtualize
    override def logSoftmax(x: Tensor): Tensor = {
      assert(x.rank == 2, "Log softmax input must be 2-D: [batchSize, logits]")

      val max = x.max2D(dim = 1)
      val res = Tensor.zeros_like(x)
      // fill res with exp(x_i - max)
      val offset = var_new(0)
      for (batch <- DataLoop(x.shape(0))) {
        for (i <- DataLoop(x.shape(1))) {
          res.data(offset) = Math.exp(x.data(offset) - max.data(batch)).toFloat
          offset += 1
        }
      }
      val sum = res.sum(dim = 1)
      offset = 0
      for (batch <- DataLoop(res.shape(0))) {
        val logsum = max.data(batch) + Math.log(sum.data(batch)).toFloat
        for (i <- DataLoop(res.shape(1))) {
          res.data(offset) = x.data(offset) - logsum
          offset += 1
        }
      }
      res
    }

    // TODO: Implement `softmax_grad` for CPU.
    // Low-priority if softmax is not used in models.
    override def softmax_grad(input: TensorR, res: TensorR): Unit = ???

    override def logSoftmax_grad(input: TensorR, res: TensorR): Unit = {
      val sum = res.d.sum(dim = 1)
      val offset = var_new(0)
      for (batch <- DataLoop(input.x.shape(0))) {
        for (i <- DataLoop(input.x.shape(1))) {
          input.d.data(offset) += res.d.data(offset) - Math.exp(res.x.data(offset)).toFloat * sum.data(batch)
          offset += 1
        }
      }
    }

    override def maxPool2D_batch(input: Tensor, kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]] = None): (Tensor, Option[Rep[Array[Int]]]) = {
      assert(input.rank == 4, "the input for maxPool (with batch) should have 4 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2")
      pads match {
        case None => ()
        case Some(paddings) => assert(paddings.size == 4, "paddings should be size 4 for maxPool_k_batch")
      }
      val (strideRow :: strideCol :: _) = strides.toList
      val (kernelRow :: kernelCol :: _) = kernels.toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = pads match {
        case None => List(0, 0, 0, 0)
        case Some(paddings) => paddings.toList
      }
      assert(strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert(strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert(input.shape(2) + 2 * padUp >= kernelRow && input.shape(3) + 2 * padUp >= kernelCol, "Image too small for maxPool_k: " + input.shape + "|" + (kernelRow, kernelCol))
      assert(padUp == padDown && padUp == padLeft && padUp == padRight && padUp >= 0, "pad should be the same")

      val resWidth = convSize(input.shape(2) + padUp + padDown, kernelRow, strideRow)
      val resHeight = convSize(input.shape(3) + padLeft + padRight, kernelCol, strideCol)
      val res = Tensor.fill(Seq(input.shape(0), input.shape(1), resWidth, resHeight), scala.Float.MinValue)
      val savedIdx = NewArray[Int](res.scalarCount)

      for (i <- DataLoop(input.shape(0))) {
        val ptrInput  = slice(input.data, i * input.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        val ptrIdx    = slice(savedIdx, i * res.shape.strides(0))
        val saveIdxBase = i * input.shape.strides(0)
        maxPool_k_inplace(Tensor(ptrInput, input.shape.drop(1): _*),
          kernelRow, kernelCol, strideRow, strideCol, padUp, padDown, padLeft, padRight,
          Tensor(ptrOutput, res.shape.drop(1): _*), ptrIdx, saveIdxBase)
      }
      (res, Some(savedIdx))
    }

    def maxPool_k_inplace(input: Tensor, kernelRow: Int, kernelCol: Int, strideRow: Int, strideCol: Int,
                          padUp: Int, padDown: Int, padLeft: Int, padRight: Int,
                          res: Tensor, savedIdx: Rep[Array[Int]], saveIdxBase: Rep[Int]): Unit = {
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)

      if (padUp == 0) {
        // looping for the output
        val offout = var_new[Int](0)                              // offset of res, by channel
        val offin  = var_new[Int](0)                              // offset of input, by channel
        for (outPane <- DataLoop(res.shape(0))) {
          val offout_1 = var_new[Int](offout)                     // offset of res, built on offout, by row
          val offin_1  = var_new[Int](offin)                      // offset of input, built on offin, by row
          for (outRow <- DataLoop(res.shape(1))) {
            val offout_2 = var_new[Int](offout_1)                 // offset of res, built on offout_1, by col
            val offin_2  = var_new[Int](offin_1)                  // offset of input, built on offin_1, by col
            for (outCol <- DataLoop(res.shape(2))) {

              // looping for the kernel
              val this_index_1 = var_new[Int](offin_2)            // offset of this (input) by row of kernel size
              for (dummy1 <- DataLoop(kernelRow)) {
                val this_index_2 = var_new[Int](this_index_1)     // offset of this (input), built on this_index_1, by col of kernel size
                for (dummy <- DataLoop(kernelCol)) {
                  __ifThenElse ((input.data(this_index_2) > res.data(offout_2)), {
                    res.data(offout_2) = input.data(this_index_2)
                    savedIdx(offout_2) = this_index_2 + saveIdxBase
                  }, ())
                  this_index_2 += 1
                }
                this_index_1 += input.shape.strides(1)
              }

              offout_2 += 1
              offin_2  += strideCol
            }
            offout_1 += res.shape.strides(1)
            offin_1  += strideRow * input.shape.strides(1)
          }
          offout += res.shape.strides(0)
          offin  += input.shape.strides(0)
        }
      } else {
        // looping for the output
        for (resPane <- DataLoop(res.shape(0))) {
          for (resRow <- DataLoop(res.shape(1))) {
            for (resCol <- DataLoop(res.shape(2))) {
              val resOff = resPane * res.shape.strides(0) + resRow * res.shape.strides(1) + resCol
              // looping for the kernel
              for (kRow <- DataLoop(kernelRow)) {
                for (kCol <- DataLoop(kernelCol)) {
                  val inRow = resRow * strideRow - padUp + kRow
                  val inCol = resCol * strideCol - padUp + kCol
                  __ifThenElse ((inRow < 0 || inRow >= input.shape(1) || inCol < 0 || inCol >= input.shape(2)), (), {
                    val inOff = resPane * input.shape.strides(0) + inRow * input.shape.strides(1) + inCol
                    __ifThenElse ((input.data(inOff) > res.data(resOff)), {
                      res.data(resOff) = input.data(inOff)
                      savedIdx(resOff) = inOff
                    }, ())
                  })
                }
              }
            }
          }
        }
      }
    }

    override def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]],
                                      kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      sidx match {
        case None => ???
        case Some(sidx) =>
          for (i <- DataLoop(output.d.scalarCount)) {
            input.d.data(sidx(i)) += output.d.data(i)
          }
      }
    }

    override def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor = {
      val (strideRow :: strideCol :: Nil) = strides.toList
      val (kernelRow :: kernelCol :: Nil) = kernel.toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = pads.toList

      val resWidth = convSize(input.shape(2) + padUp + padDown, kernelRow, strideRow)
      val resHeight = convSize(input.shape(3) + padLeft + padRight, kernelCol, strideCol)
      val res = Tensor.zeros(input.shape(0), input.shape(1), resWidth, resHeight)

      for (i <- DataLoop(input.shape(0))) {
        val ptrInput = slice(input.data, i * input.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        this.averagePool_inplace(Tensor(ptrInput, input.shape.drop(1): _*),
          kernelRow, kernelCol, strideRow, strideCol, padUp, padDown, padLeft, padRight, Tensor(ptrOutput, res.shape.drop(1): _*))
      }
      res
    }

    @virtualize
    def averagePool_inplace(input: Tensor, kernelRow: Int, kernelCol: Int, strideRow: Int, strideCol: Int, padUp: Int, padDown: Int, padLeft: Int, padRight: Int, res: Tensor): Unit = {
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)
      val kernelSize = kernelRow * kernelCol * 1.0f

      if (padUp == 0) {
        // looping for the output
        for (resPane <- DataLoop(res.shape(0))) {
          for (resRow <- DataLoop(res.shape(1))) {
            for (resCol <- DataLoop(res.shape(2))) {
              val resOff = resPane * res.shape.strides(0) + resRow * res.shape.strides(1) + resCol
              val inOff = resPane * input.shape.strides(0) + resRow * strideRow * input.shape.strides(1) + resCol * strideCol
              // looping for the kernel
              val sum = var_new[Float](0.0f)
              for (kRow <- DataLoop(kernelRow)) {
                for (kCol <- DataLoop(kernelCol)) {
                  sum += input.data(inOff + kRow * input.shape.strides(1) + kCol)
                }
              }
              res.data(resOff) = sum / kernelSize
            }
          }
        }
      } else {
        ???
      }
    }

    override def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      val strideRow = strides.head
      val strideCol = strides.last
      val kernelRow = kernel.head
      val kernelCol = kernel.last
      val kernelSize = kernelRow * kernelCol
      val pad = pads(0)

      if (pad == 0) {
        for (batch <- DataLoop(input.x.shape(0))) {
          // looping for the output
          for (yPane <- DataLoop(output.x.shape(1))) {
            for (yRow <- DataLoop(output.x.shape(2))) {
              for (yCol <- DataLoop(output.x.shape(3))) {
                val indexCurr = batch * output.x.shape.strides(0) + yPane * output.x.shape.strides(1) + yRow * output.x.shape.strides(2) + yCol
                val dCurr = output.d.data(indexCurr) / kernelSize
                val indexThis = batch * input.x.shape.strides(0) + yPane * input.x.shape.strides(1) + yRow * strideRow * input.x.shape.strides(2) + yCol * strideCol
                // looping for the kernel
                for (kRow <- DataLoop(kernelRow)) {
                  for (kCol <- DataLoop(kernelCol)) {
                    input.d.data(indexThis + kRow * input.x.shape.strides(2) + kCol) += dCurr
                  }
                }
              }
            }
          }
        }
      } else {
        ???
      }
    }

    override def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = {
      val epsilon: Float = 0.00001f
      val out1 = (x - runningMean.resize(-1,1,1)) / (runningVar + epsilon).sqrt().resize(-1, 1, 1)
      out1 * scale.resize(-1,1,1) + bias.resize(-1,1,1)
    }
    override def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor]) = {
      val saveMean = x.batchNormAv()
      val diff = x - saveMean
      val saveInvVariance = diff.square().batchNormAv()
      val epsilon = 0.00001f
      val xhat = diff / (saveInvVariance + epsilon).sqrt()
      val outy = xhat * scale.resize(-1, 1, 1) + bias.resize(-1, 1, 1)
      // runningMean and runningVariance should also be updated???
      (outy, Some(saveMean), Some(saveInvVariance))
    }
    override def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor]): Unit = {
      ???
    }

    @virtualize
    override def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int]) = {
      assert(0.0f <= prob && prob < 1.0f, s"dropout rate should be [0.0, 1), got $prob")

      val res = backend.mallocArray[Float](input.scalarCount)
      val mask = backend.mallocArray[Float](input.scalarCount)
      val scale = 1.0f / (1.0f - prob)

      for (i <- DataLoop(input.scalarCount)) {
        if (Random.rand() > prob) {
          res(i) = input.data(i) * scale
          mask(i) = scale
        } else {
          res(i) = 0.0f
          mask(i) = 0.0f
        }
      }
      (Tensor(res, input.shape.seq : _*), mask, 0)
    }

    override def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit = {
      input.d += Tensor(helper, input.x.shape: _*) * output.d  // TODO (Fei Wang): should optimized by fusing loops
    }

    override def nllLoss(x: Tensor, target: Rep[Array[Int]]): Tensor = {
      assert(x.rank == 2, "Input must be a 2-D tensor")
      generateRawComment("nllLoss forward in CPU")
      val batchSize = x.shape(0)
      val res = mallocArray[Float](batchSize)
      val offset = var_new(0)
      for (batch <- DataLoop(batchSize)) {
        res(batch) = -1.0f * x.data(offset + target(batch))
        offset += x.shape.strides(0)
      }
      Tensor(res, batchSize)
    }

    override def nllLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Int]]): Unit = {
      generateRawComment("nllLoss_grad implementation in CPU")
      val offset = var_new(0)
      for (batch <- DataLoop(input.d.shape(0))) {
        input.d.data(offset + target(batch)) += -1.0f * res.d.data(batch)
        offset += input.d.shape.strides(0)
      }
    }

    override def sum(x: Tensor): Tensor = {
      Tensor.scalar(x.fold(0.0f)(_ + _))
    }
    override def sum_grad(input: TensorR, res: TensorR): Unit = { +=(input.d, res.d) }
    override def mean(x: Tensor): Tensor = {
      this.sum(x) / x.scalarCount
    }
    override def mean_grad(input: TensorR, res: TensorR): Unit = {
      += (input.d, res.d / input.x.scalarCount)  // TODO (Fei Wang): optimize
    }

    override def concat(dim: Int, tensors: Seq[Tensor]): Tensor = {
      // prepare result tensor
      val higherDims = tensors(0).shape.take(dim)
      val higherDimsSquashed = higherDims.product
      val resDims    = (0 until tensors(0).rank: Range).map { i =>
        if (i != dim) tensors(0).shape(i)
        else tensors.map(x => x.shape(dim)).sum
      }
      val totalnbElem = resDims.product

      val res = this.mallocArray[Float](totalnbElem)
      val targetId = var_new(0)             // this is the index of res to write to
      // looping over dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the concatenation dim
        for (whichTensor <- tensors) {
          // looping over the dimensions lower than or equal to dim, in the current tensor
          val stride = if (dim == 0) whichTensor.shape.scalarCount else whichTensor.shape.strides(dim-1)
          val ptrIntput = slice(whichTensor.data, high * stride)
          for (lowOrEqual <- DataLoop(stride)) {
            res(targetId) = ptrIntput(lowOrEqual)
            targetId += 1
          }
        }
      }
      Tensor(res, resDims: _*)
    }

    override def concat_grad(dim: Int, tensorRs: Seq[TensorR], output: TensorR): Unit = {
      val higherDims = tensorRs(0).x.shape.take(dim)
      val higherDimsSquashed = higherDims.product

      val targetId = var_new(0)        // this is the index of res to read gradient from
      // looping over dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the concatenation dim
        for (whichTensorR <- tensorRs) {
          // looping over the dimensions lower than or equal to dim (but within an input tensor)
          val stride = if (dim == 0) whichTensorR.x.shape.scalarCount else whichTensorR.x.shape.strides(dim-1)
          val ptrInput = slice(whichTensorR.d.data, high * stride)
          for (lowOrEqual <- DataLoop(stride)) {
            ptrInput(lowOrEqual) += output.d.data(targetId)
            targetId += 1
          }
        }
      }
    }

  }

  object BackendCPU {
    def apply() = new BackendCPU
  }

  // The current backend for code generation.
  // To switch code generation to a different backend, simply change this value
  // in your DSL program.
  var backend: Backend = BackendCPU()

  class Tensor(val data: Rep[Array[Float]], val dimensions: Seq[Int]) extends Serializable {

    def shape = Dimensions(dimensions)
    val rank = dimensions.length
    assert (rank > 0, "Tensors need to have nonEmpty dimensions")
    val scalarCount = shape.scalarCount
    val isScalar = scalarCount == 1

    assert(scalarCount != 0, "Tensor cannot have scalar count 0")

    def apply(i: Rep[Int]): Tensor = new Tensor(slice(data, i * shape.strides(0)), shape.tail)
    // def apply(i: Rep[Int], j: Rep[Int]): Tensor = new Tensor(slice(data, i * shape.strides(0)), (j - i + 1) +: shape.tail)

    // TODO (Fei Wang): This function needs to be backend dependent (GPU version needs to use user defined kernel)
    // OR, should we depend this function on other function such as map?
    @virtualize
    def clipAt(bound: Float) = {
      for (i <- DataLoop(scalarCount)) {
        val temp = data(i)
        if (temp > bound) data(i) = bound
        if (temp < -1.0f * bound) data(i) = -1.0f * bound
      }
    }

    // OR, do we really need this function here?
    // TODO (Fei Wang): This function needs to be backend dependent (GPU version needs to use user defined kernel)
    def mutate(delta: Rep[Int] => Rep[Float]) = {
      for (i <- DataLoop(scalarCount)) this.data(i) += delta(i)
    }

    // OR, do we really need this function here?
    // TODO (Fei Wang): This function needs to be backend dependent (GPU version needs to use user defined kernel)
    def mapInPlace(op: Rep[Float] => Rep[Float]) = {
      for (i <- DataLoop(scalarCount)) this.data(i) = op(this.data(i))
    }

    // TODO (Fei Wang): This function needs to be backend dependent (GPU version needs to use user defined kernel)
    def changeTo(gen: Rep[Int] => Rep[Float]) = {
      for (i <- DataLoop(scalarCount)) this.data(i) = gen(i)
    }

    // OR, do we really need this function here?
    def map(op: Rep[Float] => Rep[Float]) = {
      val res = backend.mallocArray[Float](scalarCount)
      // TODO (Fei Wang): This assignment needs to be backend dependent (GPU version needs to use user defined kernel)
      for (i <- DataLoop(scalarCount)) res(i) = op(this.data(i))
      new Tensor(res, shape)
    }

    // TODO (Fei Wang): either remove it or make it backend dependent
    def fold(init: Rep[Float])(op: (Rep[Float], Rep[Float]) => Rep[Float]) = {
      val res = var_new[Float](init)
      for (i <- DataLoop(scalarCount)) var_assign(res, op(res, this.data(i)))
      res
    }

    def plusBias(that: Tensor): Tensor = backend.plusBias(this, that)

    // Elementwise addition.
    def +(that: Rep[Float]): Tensor = backend.+(this, that)
    def +(that: Tensor): Tensor = backend.+(this, that)

    // In-place elementwise addition.
    def +=(that: Rep[Float]): Unit = backend.+=(this, that)
    def += (that: Tensor): Unit = backend.+=(this, that)

    // Elementwise subtraction.
    def -(that: Rep[Float]): Tensor = backend.-(this, that)
    def -(that: Tensor): Tensor = backend.-(this, that)

    // In-place elementwise subtraction.
    def -=(that: Rep[Float]): Unit = backend.-=(this, that)
    def -= (that: Tensor): Unit = backend.-=(this, that)

    // Elementwise multiplication.
    def *(that: Rep[Float]): Tensor = backend.*(this, that)
    def *(that: Tensor): Tensor = backend.*(this, that)

    // In-place elementwise multiplication.
    def *=(that: Rep[Float]): Unit = backend.*=(this, that)
    def *= (that: Tensor): Unit = backend.*=(this, that)

    // Elementwise division.
    def /(that: Rep[Float]): Tensor = backend./(this, that)
    def /(that: Tensor): Tensor = backend./(this, that)

    // In-place elementwise division.
    def /=(that: Rep[Float]): Unit = backend./=(this, that)
    def /= (that: Tensor): Unit = backend./=(this, that)

    def fillInPlace(value: Rep[Float]): Unit = backend.fillInPlace(this, value)
    def setAsOne() = fillInPlace(1)
    def clear() = fillInPlace(0)

    // Copy data from another tensor, in-place.
    def copy_data(src: Tensor) = backend.copyTensorData(this, src)

    // `dot` represents the following:
    // - vector-vector dot product.
    //   [V] dot [V] => [1] (scalar)
    // - matrix-vector multiplication.
    //   [M1 x M2] dot [M2] => [M1]
    // - matrix-matrix multiplication.
    //   [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def dot(that: Tensor) = {
      generateRawComment(s"dot: ${this.shape.seq}, ${that.shape.seq}")
      (this.rank, that.rank) match {
        case (1, 1) => assert(this.shape(0) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case (2, 1) | (2, 2) => assert(this.shape(1) == that.shape(0), s"Incompatible shapes: ${this.shape}, ${that.shape}")
        case _ => throw new IllegalArgumentException(
          s"Only vector-vector, matrix-vector, and matrix-matrix multiplication are allowed (actual shapes: ${this.shape}, ${that.shape})")
      }
      backend.dot(this, that)
    }

    def gemm(that: Tensor, transX: Boolean, transY: Boolean, alpha: Float): Tensor = {
      generateRawComment(s"gemm: ${this.shape.seq}, ${that.shape.seq}")
      backend.gemm(this, transX, that, transY, alpha)
    }

    // NOTE: only handles (Vector Cartesian Vector)
    // limited support for GPU backend. Do not recommend using this function
    @deprecated
    def cart(that: Tensor) = {
      assert(this.rank == 1 && that.rank == 1, "cartesian product is only for 1d vectors")
      val res = backend.mallocArray[Float](this.shape(0) * that.shape(0))
      val off = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        for (j <- DataLoop(that.shape(0))) {
          res(off) = data(i) * that.data(j)
          off += 1
        }
      }
      Tensor(res, this.shape(0), that.shape(0))
    }

    def trans() = backend.trans(this)

    def exp() = backend.exp(this)
    def log() = backend.log(this)
    def sqrt() = backend.sqrt(this)
    def square() = backend.square(this)

    def relu() = backend.relu(this)
    def tanh() = backend.tanh(this)
    def sigmoid() = backend.sigmoid(this)

    // NOTE: sum all elements
    // TODO (Fei Wang): prefer to have a general reduce function, and depend sum() to that function
    def sum() = backend.sum(this)

    // sum over one dimension
    // TODO (Fei Wang): prefer to have a general reduce over given dimension function, and depend sum(dim) to that function
    def sum(dim: Int) = {
      assert(dim >= 0 && dim < this.rank, "dim should be within range of this.nbDims")
      val higherDims = this.shape.take(dim)
      val higherDimsSquashed = higherDims.product
      val resDims = higherDims ++ this.shape.drop(dim + 1)
      val res = Tensor.zeros(resDims: _*)

      // looping over the dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the dimension to be summed
        val offres = var_new(high * (if (dim == 0) res.scalarCount else res.shape.strides(dim - 1)))
        val offthis = var_new(high * (if (dim == 0) this.scalarCount else this.shape.strides(dim - 1)))
        for (sum <- DataLoop(this.shape(dim))) {
          // looping over the dims lower than dim
          for (low <- DataLoop(this.shape.strides(dim))) {
            res.data(offres + low) += this.data(offthis + low)
          }
          offthis += this.shape.strides(dim)
        }
      }
      res
    }

    def mean() = backend.mean(this)

    def batchNormInference(scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor =
      backend.batchNormInference(this, scale, bias, runningMean, runningVar)

    @virtualize
    def batchNormTraining(scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor]) =
      backend.batchNormTraining(this, scale, bias, runningMean, runningVar)

    // NOTE: this function has nothing to do with Batch Normalization. Sorry for the confusing name
    @virtualize
    def batchNormAv() = {
      assert(this.rank == 4, "tensor for batch normal averaging should have 4 dimensions")
      val base: Rep[Float] = this.shape(0) * this.shape(2) * this.shape(3) * 1.0f
      val res = Tensor.zeros(this.shape(1), 1, 1)

      for (batch <- DataLoop(this.shape(0))) {
        val offsetBatch = batch * this.shape.strides(0)
        for (channel <- DataLoop(this.shape(1))) {
          val offset = offsetBatch + channel * this.shape.strides(1)
          for (lower <- DataLoop(this.shape.strides(1))) {
            res.data(channel) = res.data(channel) + this.data(offset + lower)
          }
        }
      }
      res / base
    }

    // mark: (Fei Wang) HERE: more gardening below
    // only for debugging, maybe remove?
    @virtualize
    def check(limit: Float) = {
      val idx = var_new(0)
      while (idx < this.scalarCount && -limit < this.data(idx) && this.data(idx) < limit) {
        idx += 1
      }
      idx != this.scalarCount
    }

    @virtualize
    def max() = this.fold(scala.Float.MinValue)((agg, x) => if (x > agg) x else agg)

    @virtualize
    def max2D(dim: Int) = {
      assert (this.rank == 2, "Only deal with 2D tensor")
      assert (dim == 0 || dim == 1, "dim must be in range of this.nbDims")

      if (dim == 0) ???
      else {
        val res = backend.mallocArray[Float](this.shape(0))
        val offset = var_new(0)
        for (i <- DataLoop(this.shape(0))) {
          val max = var_new(scala.Float.MinValue)
          for (j <- DataLoop(this.shape(1))) {
            if (this.data(offset) > max) max = this.data(offset)
            offset += 1
          }
          res(i) = max
        }
        Tensor(res, this.shape(0))
      }
    }

    // FIXME: Proper tensor
    @virtualize
    def maxIndex() = {
      assert(this.rank == 1)
      val vMax = var_new(this.data(0))
      val iMax = var_new(0)
      for (idx <- 1 until this.scalarCount: Rep[Range]) {
        if (this.data(idx) > vMax) {
          iMax = idx
          vMax = this.data(idx)
        }
      }
      iMax
    }

    @virtualize
    def softmax_batch() = backend.softmax(this)

    @virtualize
    def softmax() = {
      assert(this.rank == 1, "TODO: softmax only handles 1d vectors so far: " + this.rank)

      val m = this.max
      val normalized = this.map(x => x - m)
      val nor_exp = normalized.exp()
      nor_exp / nor_exp.sum()
    }

    @virtualize  // batched log softmax
    def logSoftmaxB() = backend.logSoftmax(this)

    @virtualize
    def logSoftmax() = {
      assert(this.rank == 1, "TODO: logSoftmax only handles 1d vectors so far")

      val m = this.max
      val logsum = m + Math.log(this.fold(0.0f)((agg, x) => agg + Math.exp(x - m).toFloat)).toFloat
      this.map(x => x - logsum)
    }

    def nllLossB(target: Rep[Array[Int]]) = backend.nllLoss(this, target)

    @virtualize
    def nllLoss(target: Rep[Int]) = {
      assert(this.rank == 1, "Input must be a 1-D tensor")

      // assertC(0 <= target && target < this.nbElem, "Incorrect target")
      Tensor.scalar(-1.0f * this.data(target))
    }

    def reshape(dims: Int*) = {
      assert(scalarCount == dims.product, s"Invalid shape, scalar count mismatch: $shape, $dims")
      Tensor(data, dims: _*)
    }

    def resize(dims: Int*) = {
      generateRawComment(s"resize to $dims")
      val new_dims = if (dims.forall(_ > 0)) dims else {
        assert(dims.filter(_ < 0) == Seq(-1), s"there should be at most one -1 in the resize dims, got $dims")
        dims.updated(dims.indexOf(-1, 0), this.scalarCount / dims.filter(_ > 0).product)
      }
      assert(new_dims.product == this.scalarCount, s"dims: $new_dims != scalarCount: $scalarCount")

      Tensor(this.data, new_dims : _*)
    }

    @virtualize
    def amax() = this.fold(0.0f)((agg, x) => if (Math.abs(x) > Math.abs(agg)) x else agg)

    def printHead(count: Int = 10, msg: String = ""): Unit = {
      if (msg != "")
        printf(s"$msg (size ${this.shape.seq mkString " x "})\\n")
      printf(s"Max Abs: ${format}|| ", this.amax())
      for (i <- 0 until count: Rep[Range]) {
        printf(format, this.data(i))
      }
      printf("\\n")
    }

    def printPane(msg: String = ""): Unit = {
      assert(this.rank == 4, "printPane only applies to Tensors of rank 4")
      printf(s"${msg} --> Pane # - ${this.shape(2)} x ${this.shape(3)}\\n")
      for (k <- 0 until this.shape(2): Rep[Range]) {
        for (l <- 0 until this.shape(3): Rep[Range]) {
          printf(format, this.data(k * this.shape.strides(2) + l))
        }
        printf("\\n")
      }
      printf("\\n\\n")
    }

    def print(msg: String = ""): Unit = {
      if (msg != "")
        printf(s"$msg (size ${this.shape.seq mkString " x "})\\n")
      if (this.rank == 4) this.print4D
      else if (this.rank == 3) this.print3D
      else this.printRaw(this.shape.lastOption.getOrElse(1))
    }

    val format = "%.5f "
    def print4D() = {
      val idx = var_new(1)
      for (i <- 0 until this.shape(0): Rep[Range]) {
        val idx1 = var_new(1)
        for (j <- 0 until this.shape(1): Rep[Range]) {
          printf(s"Pane #(%d, %d) - ${this.shape(2)} x ${this.shape(3)}\\n", idx, idx1)
          for (k <- 0 until this.shape(2): Rep[Range]) {
            for (l <- 0 until this.shape(3): Rep[Range]) {
              printf(format, this.data(i * this.shape.strides(0) + j * this.shape.strides(1) + k * this.shape.strides(2) + l))
            }
            printf("\\n")
          }
          printf("\\n\\n")
          idx1 += 1
        }
        idx += 1
      }
    }

    def print3D() = {
      val idx = var_new(1)
      for (i <- 0 until this.shape(0): Rep[Range]) {
        printf(s"Pane #%d - ${this.shape(1)} x ${this.shape(2)}\\n", idx)
        for (k <- 0 until this.shape(1): Rep[Range]) {
          for (l <- 0 until this.shape(2): Rep[Range]) {
            printf(format, this.data(i * this.shape.strides(0) + k * this.shape.strides(1) + l))
          }
          printf("\\n")
        }
        printf("\\n\\n")
        idx += 1
      }
    }

    @virtualize
    def printRaw(row: Int = 10) = {
      for (i <- 0 until this.scalarCount: Rep[Range]) {
        printf(format, data(i))
        val imod = i % row
        if (imod == row - 1)
          printf("\\n")
      }
      printf("\\n")
    }

    // setting: this is matrix, that is dims(0)-sized vector, y is dims(1)-sized vector
    // the result is to update this so that this += that * y, where * is Cartesian product
    def add_cartesian(that: Tensor, y: Tensor) = {
      generateRawComment("add_cartesian")
      assert(this.rank == 2 && that.shape == Dimensions(Seq(this.shape(1))) && y.shape == Dimensions(Seq(this.shape(0))) ||
        this.rank == 1 && that.shape == this.shape && y.isScalar, s"${shape} - ${that.shape} - ${y.shape}")
      val off = var_new(0)
      val up = this.shape.head
      for (i <- DataLoop(up)) {
        for (j <- DataLoop(shape(1))) {
          this.data(off + j) = this.data(off + j) + that.data(j) * y.data(i)
        }
        off += this.shape(1)
      }
    }

    // setting: this is dims(0)-sized vector, that is matrix (dims(0) * dims(1)), y is dims(1)-sized vector
    // the result is to update this so that this accumulate every matrix col * y
    def add_composion(that: Tensor, y: Tensor) = {
      assert(that.rank == 2 && this.shape.seq == Seq(that.shape(1)) && y.shape.seq == Seq(that.shape(0))
        || that.rank == 1 && this.shape == that.shape && y.isScalar, s"${shape} - ${that.shape} - ${y.shape}")
      val off = var_new(0)
      val up = that.shape.head
      for (i <- DataLoop(up)) {
        for (j <- DataLoop(that.shape(1))) {
          data(j) += that.data(off + j) * y.data(i)
        }
        off += that.shape(1)
      }
    }

    @virtualize
    def addMul(that: Tensor, y: Tensor) = {
      assert(this.rank == 2 && that.rank == 2 && y.rank == 2, s"Dimensions: ${this.shape.seq} - ${that.shape.seq} - ${y.shape.seq}")
      assert(this.shape(0) == that.shape(0) && this.shape(1) == y.shape(1) && that.shape(1) == y.shape(0), s"Dimensions: ${this.shape.seq} + ${that.shape.seq} * ${y.shape.seq}")
      var offThis = var_new(0)
      var offThatR = var_new(0)
      var offYC = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val offYR = var_new(offYC)
        for (j <- DataLoop(this.shape(1))) {
          val offY = var_new(offYR)
          val offThat = var_new(offThatR)
          for (k <- DataLoop(that.shape(1))) {
            this.data(offThis) = this.data(offThis) + that.data(offThat) * y.data(offY)
            offThat += 1
            offY += y.shape.strides(0)
          }
          offThis += 1
          offYR += 1
        }
        offThatR += that.shape.strides(0)
        offYC *= 0
      }
    }

    // private function to get data with default to the only element
    def getAt(i: Rep[Int]) = {
      if (this.isScalar) data(0)
      else data(i)
    }

    def square(t: Rep[Float]) = t * t
    def add_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_mult")

      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + a.getAt(i) * b.getAt(i) }
        else { data(i) = data(i) + a.getAt(i) * b.getAt(i) }
      }
    }

    def addMul(a: Rep[Float], b: Tensor) = {
      assert(this.shape == b.shape)

      generateRawComment("Generate code for addMul")
      for (i <- DataLoop(this.scalarCount)) {
        this.data(i) = this.data(i) + a * b.data(i)
      }
    }

    def cmulAdd(a: Float, b: Tensor) = {
      assert(this.shape == b.shape)
      for (i <- DataLoop(this.scalarCount))
        this.data(i) = a * this.data(i) + b.data(i)

      this // FIXME ??
    }

    def add_div(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_div")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + a.getAt(i) / b.getAt(i) }
        else { data(i) = data(i) + a.getAt(i) / b.getAt(i) }
      }
    }

    def minus_mult_div_square(a: Tensor, b: Tensor, c: Tensor) = {
      assert(Tensor.dimCompatible(a, b)    && Tensor.dimCompatible(a, c)    && Tensor.dimCompatible(c, b)    &&
        Tensor.dimCompatible(this, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, c),
        "dim not competible in minus_mult_div_square")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
        else { data(i) = data(i) - a.getAt(i) * b.getAt(i) / square(c.getAt(i)) }
      }
    }

    def add_oneMinusSquare_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusSquare_mult")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
        else { data(i) = data(i) + (1.0f - square(a.getAt(i))) * b.getAt(i) }
      }
    }

    def oneMinusThenMult(t: Rep[Float]) = (1.0f - t) * t

    def add_oneMinusThenMult_mult(a: Tensor, b: Tensor) = {
      assert(Tensor.dimCompatible(a, b) && Tensor.dimCompatible(a, this) && Tensor.dimCompatible(this, b), "dim not Compatible in add_oneMinusThenMult_mult")
      val dims0M = mmax(shape.head, mmax(a.shape.head, b.shape.head))
      val dims1M = mmax(shape.get(1), mmax(a.shape.get(1), b.shape.get(1)))
      for (i <- DataLoop(dims0M * dims1M)) {
        if (this.isScalar) { data(0) = data(0) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
        else { data(i) = data(i) + oneMinusThenMult(a.getAt(i)) * b.getAt(i) }
      }
    }

    @virtualize
    def conv2D_batch(kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) =
      backend.conv2D_batch(this, kernel, bias, strides, pads)

    @virtualize
    def averagePool_batch(kernels: Seq[Int], strides: Seq[Int], paddings: Option[Seq[Int]]): Tensor = {
      val (strideRow :: strideCol :: Nil) = strides.take(2).toList
      val (kernelRow :: kernelCol :: Nil) = kernels.take(2).toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = paddings match {
        case None => List(0, 0, 0, 0)
        case Some(pads) => pads.take(4).toList
      }
      assert(this.rank == 4, "the input for averagePool_batch should have 4 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2")
      assert(strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert(strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert(this.shape(2) + 2 * padUp >= kernelRow && this.shape(3) + 2 * padUp >= kernelCol, "Image too small for averagePool_batch: " + this.shape + "|" + (kernelRow, kernelCol))
      assert(padUp == padDown && padUp == padLeft && padUp == padRight && padUp >= 0, "pad should be the same")

      backend.averagePool2D_batch(this, kernels, strides, paddings match {case None => Seq(0, 0, 0, 0); case Some(paddings) => paddings})
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int) = {
      assert(this.rank == 3)

      val resHeight = this.shape(1) / strideRow
      val resWidth = this.shape(2) / strideCol
      val res = Tensor.fill(Seq(this.shape(0), resHeight, resWidth), scala.Float.MinValue)

      // FIXME adhoc transform tensor to be using generic type!
      val savedIdx = NewArray[Int](res.scalarCount)

      val oidxW = var_new(0)  // walks by channel in res
      val iidx = var_new(0)   // walks by 1 in input (this.data)
      for (ichan <- DataLoop(this.shape(0))) {
        val oidx = var_new(oidxW)  // walks by row in res
        for (ox <- DataLoop(res.shape(1))) {
          for (sx <- DataLoop(strideRow)) {
            val oidx2 = var_new(oidx)  // walks by 1 in res
            for (oy <- DataLoop(res.shape(2))) {
              for (sy <- DataLoop(strideCol)) {
                if (this.data(iidx) > res.data(oidx2)) {
                  res.data(oidx2) = this.data(iidx)
                  savedIdx(oidx2) = iidx
                }
                iidx += 1
              }
              oidx2 += 1
            }
          }
          oidx += res.shape.strides(1)
        }
        oidxW += res.shape.strides(0)
      }

      (res, savedIdx)
    }

    def maxPool2D_batch(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]]) =
      backend.maxPool2D_batch(this, kernels, strides, pads)

    def dropout(prob: Float = 0.5f) =
      backend.dropout(this, prob)

    // TODO make concat backend-dependent, and implement backendGPU concat and concat_grad (split)
    @virtualize
    def concat(dim: Int, others: Tensor*) = {
      assert(others.size >= 1, "there should be at least one tensor in others")
      assert(dim >= 0 && dim < this.rank, "dim should be within range of this.nbDims")
      assert(others.forall(x => x.rank == this.rank), "all tensors should have the same number of dimensions")
      assert(others.forall(x => (0 until this.rank: Range).forall(i =>  i == dim || x.shape(i) == this.shape(i))),
        "all dimensions except the concatenation dimension should be the same")

      // call backend-specific function for this.
      backend.concat(dim: Int, this +: others)
    }

    @virtualize
    def global_ave_batch() = {
      assert(this.rank == 4, "assume this is Tensor with 4D (batch * channel * width * height")
      val resTensor = Tensor.zeros(this.shape.take(2): _*)

      val scale = this.shape.strides(1)
      val offset = var_new(0)                      // offset of this, should step by this.strides(2)
      val res_offset = var_new(0)                  // offset of res, should step by 1
      // looping over each batch
      for (batch <- DataLoop(this.shape(0))) {
        // looping over each channel
        for (channel <- DataLoop(this.shape(1))) {
          // get average of a channel
          val sum = var_new(0.0f)
          val offset_a = var_new(offset)           // offset of this for averaging, should step by this.strides(3)
          for (i <- DataLoop(this.shape(2))) {
            for (j <- DataLoop(this.shape(3))) {
              sum += this.data(offset_a + j)
            }
            offset_a += this.shape.strides(2)
          }
          resTensor.data(res_offset) = sum / scale
          offset += this.shape.strides(1)
          res_offset += 1
        }
      }
      resTensor
    }

    @virtualize
    def global_ave() = {
      // the result should be 1D (channel), by averaging the numbers in (width * height)
      assert(this.rank == 3, "assume this is Tensor with 3D (channel * width * height)")

      val res = backend.mallocArray[Float](this.shape(0))
      // looping over each channel
      for (channel <- DataLoop(this.shape(0))) {
        val offset = var_new(this.shape.strides(0) * channel)
        val sum = var_new(0.0f)
        for (i <- DataLoop(this.shape(1))) {
          for (j <- DataLoop(this.shape(2))) {
            sum += this.data(offset + j)
          }
          offset += this.shape.strides(1)
        }
        res(channel) = sum / (this.shape.strides(0))
      }
      Tensor(res, this.shape(0))
    }

    // FIXME: the MNIST example precomput the mean and std
    // I thought that normalize would need to compute it first and then
    // modify the data to match the one requested.
    // SO here what is expected is to have mean = 0 and std = 1 knowing that
    // the current mean is m and the current std is s
    @virtualize
    def normalize(m: Float, s: Float, inPlace: Boolean = false) = {
      if (inPlace) {
        this.mapInPlace(x => (x - m)/s)
        this
      } else {
        this.map(x => (x - m)/s)
      }
    }

    // fused ops
    def linearTanh(x: Tensor, b: Tensor) = {
      // this is W. We want (W.dot(x)+b).tanh()
      assert(this.rank == 2 && x.rank == 1 && b.rank == 1, "limited generalization")
      generateRawComment("forward for linearTanh")
      // val res_dot = backend.mallocArray[Float](this.shape(0))
      // val res_add = backend.mallocArray[Float](this.shape(0))
      val res_tanh = backend.mallocArray[Float](this.shape(0))
      val offSet = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val sum = var_new(0.0f)
        for (j <- DataLoop(this.shape(1))) {
          sum += this.data(offSet + j) * x.data(j)
        }
        // res_dot(i) = sum
        // res_add(i) = b.data(i) + sum
        res_tanh(i) = Math.tanh(b.data(i) + sum).toFloat
        offSet += this.shape(1)
      }
      (Tensor(res_tanh, this.shape(0)))
    }

    def linear2Tanh(x: Tensor, W2: Tensor, x2: Tensor, b: Tensor) = {
      // this is W. We want (W.dot(x) + W2.dot(x2) + b).tanh()
      assert(this.rank == 2 && x.rank == 1 && W2.rank == 2 && x2.rank == 1 && b.rank == 1, "limited generalization")
      generateRawComment("forward for linear2Tanh")
      val res_tanh = backend.mallocArray[Float](this.shape(0))
      val offSet = var_new(0)
      for (i <- DataLoop(this.shape(0))) {
        val sum = var_new(0.0f)
        for (j <- DataLoop(this.shape(1))) {
          sum += this.data(offSet + j) * x.data(j)
          sum += W2.data(offSet + j) * x2.data(j)
        }
        res_tanh(i) = Math.tanh(b.data(i) + sum).toFloat
        offSet += this.shape(1)
      }
      Tensor(res_tanh, this.shape(0))
    }
  }

  object Tensor {
    // this function actually SHOULD NOT be different for different backend
    def apply(data: Rep[Array[Float]], dims: Int*) =
      backend.arrayToTensor(data, dims: _*)

    def dimCompatible(a: Tensor, b: Tensor) = {
      (a.shape == b.shape) || a.isScalar || b.isScalar
    }

    def dimBroadcast(a: Seq[Int], b: Seq[Int]): Option[(Dimensions, Dimensions, Dimensions)] = {
      def bc(a: Seq[Int], b: Seq[Int], trail: List[Int]): List[Int] = {
        if (a.size == 0) b.toList ++ trail
        else if (b.size == 0) a.toList ++ trail
        else if (a.last == 1) bc(a.init, b.init, b.last :: trail)
        else if (b.last == 1) bc(a.init, b.init, a.last :: trail)
        else if (a.last == b.last) bc(a.init, b.init, a.last :: trail)
        else List(-1) // indicate dim not Compatible by broadcast
      }
      val res = bc(a, b, List())
      if (res == List(-1)) None
      else {
        // add dimensions of 1 to tensors with smaller rank
        if (a.size > b.size) Some((Dimensions(a), Dimensions(Seq.fill(a.size - b.size)(1) ++ b), Dimensions(res)))
        else if (a.size < b.size) Some((Dimensions(Seq.fill(b.size - a.size)(1) ++ a), Dimensions(b), Dimensions(res)))
        else Some((Dimensions(a), Dimensions(b), Dimensions(res)))
      }
    }

    def randseed(seed: Int) = unchecked[Unit]("srand(", seed, ")")
    def randseed() = unchecked[Unit]("srand(time(NULL))")
    def rand(dims: Int*) = randinit(dims.toSeq, 1.0f, None)
    def rand(dims: Seq[Int], scale: Float) = randinit(dims.toSeq, scale, None)
    def randinit(dim0: Int): Tensor = randinit(Seq(dim0), 1.0f, None)
    def randinit(dim0: Int, seed: Option[Int]): Tensor = randinit(Seq(dim0), 1.0f, seed)
    def randinit(dim0: Int, dim1: Int, scale: Float): Tensor = randinit(Seq(dim0, dim1), scale, None)
    def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor =
      backend.randinit(dims, scale, seed)

    def randn(dim0: Int, dim1: Int = 1, scale: Float = 1.0f, offset: Int = 0) = {
      val res = backend.mallocArray[Float](dim0 * dim1)
      for (i <- (0 until dim0 * dim1): Rep[Range]) res(i) = unchecked[Float]("d(gen)") * scale
      Tensor(res, dim0, dim1)
    }

    def randnorm(dims: Int*) = {
      val res = backend.mallocArray[Float](dims.product)
      for (i <- (0 until dims.product): Rep[Range]) res(i) = unchecked[Float]("d(gen)")
      Tensor(res, dims: _*)
    }

    def randPositive(dims: Int*) = {
      val scalarCount = dims.product
      val res = backend.mallocArray[Float](scalarCount)
      for (i <- (0 until scalarCount): Rep[Range]) res(i) = Random.rand()
      new Tensor(res, dims)
    }

    def fill(dims: Seq[Int], value: Rep[Float]): Tensor = backend.fill(dims, value)

    def fill(dims: Seq[Int], fFill: Seq[Rep[Int]] => Rep[Float]) = {
      val scalarCount = dims.product
      val res = backend.mallocArray[Float](scalarCount)

      var idx = var_new(0)
      def innerFill(args: Seq[Rep[Int]]) = {
        res(idx) = fFill(args)
        idx += 1
      }

      val dum = (dims :\ innerFill _) {
        case (up, f) =>
          (args: Seq[Rep[Int]]) => {
            for (i <- 0 until up: Rep[Range]) {
              f(args :+ i)
            }
          }
      }
      dum(Seq[Rep[Int]]())
      new Tensor(res, dims)
    }

    def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int) = backend.fillWithBias(dims, bias, dim)

    def scalar(value: Rep[Float]) = fill(Seq(1), value)

    def zeros(dims: Int*): Tensor = // fill(dims, 0.0f)
      new Tensor(backend.mallocArray[Float](dims.product), dims)
    def zeros_like(that: Tensor) = zeros(that.shape: _*)
    def ones(dims: Int*) = fill(dims, 1.0f)
    def ones_like(that: Tensor) = ones(that.shape: _*)
    def halves(dims: Int*) = fill(dims, 0.5f)

    def expand(vector: Tensor, dim1: Int) = {
      assert(vector.rank == 1)
      val res = backend.mallocArray[Float](vector.shape(0) * dim1)
      val off = var_new(0)
      for (j <- (0 until dim1): Rep[Range]){
        for (i <- (0 until vector.shape(0)): Rep[Range]) {
          res(off) = vector.data(i)
          off += 1
        }
      }
      new Tensor(res, dim1 +: vector.shape)
    }

    def copy(tensor: Tensor) = {
      val res = backend.mallocArray[Float](tensor.scalarCount)
      for (i <- DataLoop(tensor.scalarCount)) res(i) = tensor.data(i)
      new Tensor(res, tensor.shape)
    }

    def fromData(scalars: Float*): Tensor = backend.makeTensor(Seq(scalars.length), scalars: _*)

    def fromData(dims: Seq[Int], scalars: Float*): Tensor = backend.makeTensor(dims, scalars: _*)

    @virtualize
    def assertEqual(a: Tensor, b: Tensor, mark: String = "", tal: Float = 0.000001f) = {
      val errorPrefix = if (mark.nonEmpty) s"ERROR ($mark)" else "ERROR"
      assert(a.shape == b.shape, s"$errorPrefix: tensor shapes are not equal, ${a.shape.seq} != ${b.shape.seq}\\n")

      val i = var_new(0)
      while (i < a.scalarCount && { val diff = a.data(i) - b.data(i); diff > -tal && diff < tal }) {
        i += 1
      }
      if (i < a.scalarCount) {
        printf("%s: tensor data are not equal at index %d, %.4f != %.4f\\n", errorPrefix, i, a.data(i), b.data(i))
        error("")
      }
    }
  }

  // Tensor type is the similar to NumR, just replace RFloat with Tensor
  // also Tensor internally use array, which is mutable by default
  // so both field are val (not var) and can be updated by += -= *= /= setAsOne()
  // all instances of vectors will be shepherded by c++ smart pointers, alleviating the memory leak problem
  // type diff = cps[Unit]
  class TensorR(val x: Tensor, val d: Tensor) extends Serializable {
    var isInput: Boolean = false // true if it is an input (no need to compute gradient)

    def apply(i: Rep[Int]) = new TensorR(x(i), d(i))

    def clip_grad(bound: Float) = {
      d.clipAt(bound)
    }

    // fused ops (but slower!)
    def linearTanh(x: TensorR, b: TensorR) = shift { k: (TensorR => Unit) =>
      val y = TensorR(this.x.linearTanh(x.x, b.x)); k(y)
      generateRawComment("back prop for linearTanh")
      val offSet = var_new(0)
      for (i <- DataLoop(this.x.shape(0))) {
        val d_res_add = (1.0f - y.x.data(i) * y.x.data(i)) * y.d.data(i)
        b.d.data(i) += d_res_add
        for (j <- DataLoop(this.x.shape(1))) {
          x.d.data(j) += d_res_add * this.x.data(offSet + j)
          this.d.data(offSet + j) += d_res_add * x.x.data(j)
        }
        offSet += this.x.shape(1)
      }
    }

    // fused ops (but slower!)
    def linear2Tanh(x: TensorR, W2: TensorR, x2: TensorR, b: TensorR) = shift { k: (TensorR => Unit) =>
      val y = TensorR(this.x.linear2Tanh(x.x, W2.x, x2.x, b.x)); k(y)
      generateRawComment("back prop for linear2Tanh")
      val offSet = var_new(0)
      for (i <- DataLoop(this.x.shape(0))) {
        val d_res_add = (1.0f - y.x.data(i) * y.x.data(i)) * y.d.data(i)
        b.d.data(i) += d_res_add
        for (j <- DataLoop(this.x.shape(1))) {
          val idx = offSet + j
          x.d.data(j) += d_res_add * this.x.data(idx)
          this.d.data(idx) += d_res_add * x.x.data(j)
          x2.d.data(j) += d_res_add * W2.x.data(idx)
          W2.d.data(idx) += d_res_add * x2.x.data(j)
        }
        offSet += this.x.shape(1)
      }
    }

    def backpropElementWiseOpWithBroadCast(that: TensorR, y: TensorR, opThis: ((Rep[Float], Rep[Float], Rep[Float]) => Rep[Float]), opThat: ((Rep[Float], Rep[Float], Rep[Float]) => Rep[Float])): Unit = {
      // assume y.x = elementWiseOpWithBroadCast(this.x, that.x, someOp)
      // assume that opThis returns the increment of this.d; opThat returns the increment of that.d
      // both opThis and opThat takes this.x, that.x, and y.d as parameters
      // TODO (Fei Wang): in some cases (if this, or that are inputs (x, y), there gradients are not initialized/useless)
      Tensor.dimBroadcast(this.x.shape, that.x.shape) match {
        case None => throw new IllegalArgumentException(s"dimensions of tensors do not match! ${this.x.shape.seq} != ${that.x.shape.seq}")
        case Some((thisShape, thatShape, yShape)) => {
          def inplace(offThis: Rep[Int], offThat: Rep[Int], offY: Rep[Int], dim: Int): Unit = {
            val offthis = var_new[Int](offThis)
            val offthat = var_new[Int](offThat)
            val offy = var_new[Int](offY)
            for (i <- DataLoop(yShape(dim))) {
              if (dim == yShape.size - 1) {
                this.d.data(offthis) += opThis(this.x.data(offthis), that.x.data(offthat), y.d.data(offy))
                that.d.data(offthat) += opThat(this.x.data(offthis), that.x.data(offthat), y.d.data(offy))
              } else {
                inplace(offthis, offthat, offy, dim + 1)
              }
              offy += yShape.strides(dim)
              if (thisShape(dim) > 1) offthis += thisShape.strides(dim)
              if (thatShape(dim) > 1) offthat += thatShape.strides(dim)
            }
          }
          inplace(0, 0, 0, 0)
        }
      }
    }

    // that is bias
    def plusBias(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      backend.plusBias(this.x, that.x); k(this)  // note: plusBias is in-place
      backend.plusBias_grad(this, that)
    }

    def + (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x + that); k(y)
      this.d += y.d
    }
    def + (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x + that.x); k(y)
      generateRawComment("back prop for + op")
      backend.+=(this.d, y.d); backend.+=(that.d, y.d)
      // TODO: (Fei Wang): potential optimization (define specific +_grad to fuse to updates)
      // val opThis = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      // val opThat = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      // backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    def - (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x - that); k(y)
      this.d += y.d
    }
    def - (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x - that.x); k(y)
      generateRawComment("back prop for - op")
      backend.+=(this.d, y.d); backend.-=(that.d, y.d)
      // TODO: (Fei Wang): potential optimization (define specific +_grad to fuse to updates)
      // val opThis = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      // val opThat = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => -1.0f * c
      // backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // mark: HERE: following code need to be backend depend!
    // this is element wise multiplication
    def * (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x * that); k(y)
      // this.d += y.d * that  // TODO (Fei Wang) can be optimized to save space
      this.d.addMul(that, y.d)
    }
    def * (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x * that.x); k(y)
      // this.d.add_mult(that.x, y.d); that.d.add_mult(this.x, y.d)
      val opThis = (_: Rep[Float], b: Rep[Float], c: Rep[Float]) => c * b
      val opThat = (a: Rep[Float], _: Rep[Float], c: Rep[Float]) => c * a
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // element wise division
    def / (that: Rep[Float]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x / that); k(y)
      // this.d += y.d / that  // TODO (Fei Wang) can be optimized to save space
      this.d.addMul(1.0f / that, y.d)
    }
    def / (that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x / that.x); k(y)
      // this.d.add_div(y.d, that.x); that.d.minus_mult_div_square(this.x, y.d, that.x)
      val opThis = (_: Rep[Float], b: Rep[Float], c: Rep[Float]) => c / b
      val opThat = (a: Rep[Float], b: Rep[Float], c: Rep[Float]) => -1.0f * a * c / (b * b)
      backpropElementWiseOpWithBroadCast(that, y, opThis, opThat)
    }

    // `dot` represents the following:
    // - vector-vector dot product.
    //   [V] dot [V] => [1] (scalar)
    // - matrix-vector multiplication.
    //   [M1 x M2] dot [M2] => [M1]
    // - matrix-matrix multiplication.
    //   [M1 x M2] dot [M2 x M3] => [M1 x M3]
    def dot(that: TensorR): TensorR @diff = shift { (k: TensorR => Unit) =>
      val res = backend.dot(x, that.x)
      val y = TensorR(res); k(y)

      // back-propagate
      backend.dot_grad(this, that, y)
    }

    def gemm(that: TensorR, transX: Boolean, transY: Boolean, alpha: Float): TensorR @diff = shift { (k: TensorR => Unit) =>
      val ty = TensorR(x.gemm(that.x, transX, transY, alpha)); k(ty)
      generateRawComment(s"backprop for gemm ${x.shape.seq}, ${that.x.shape.seq}")
      backend.gemm_grad(this, transX, that, transY, alpha, ty)
    }

    def trans(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.trans()); k(y)
      // back-propagate
      backend.trans_grad(this, y)
    }

    def exp(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(backend.exp(x)); k(y)
      generateRawComment("backprop for exp")
      backend.exp_grad(this, y)
    }

    def log(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(backend.log(x)); k(y)
      generateRawComment("backprop for log")
      backend.log_grad(this, y)
    }

    def sqrt(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(backend.sqrt(x)); k(y)
      generateRawComment("backprop for sqrt")
      backend.sqrt_grad(this, y)
    }

    def square(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.square()); k(y)
      generateRawComment("backprop for square")
      backend.square_grad(this, y)
    }

    def relu(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(this.x.relu()); k(y)
      backend.relu_grad(this, y)
    }

    def tanh(): TensorR @diff = shift { (k : TensorR => Unit) =>
      val y = TensorR(x.tanh()); k(y)
      backend.tanh_grad(this, y)
    }

    def sigmoid(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sigmoid()); k(y)
      backend.sigmoid_grad(this, y)
    }

    def sum(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = new TensorR(x.sum(), Tensor.zeros(1)); k(y)
      generateRawComment("'sum' gradient.")
      backend.sum_grad(this, y)
    }

    def mean(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = new TensorR(x.mean(), Tensor.zeros(1)); k(y)
      generateRawComment("'mean' gradient")
      backend.mean_grad(this, y)
    }

    def sum(dim: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.sum(dim)); k(y)

      // back-propagate
      val higherDims = this.x.shape.take(dim)
      val higherDimsSquashed = higherDims.product
      val resDims = higherDims ++ this.x.shape.drop(dim + 1)
      // looping over the dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the dimension to be summed
        val offres = var_new(high * (if (dim == 0) y.x.scalarCount else y.x.shape.strides(dim - 1)))
        val offthis = var_new(high * (if (dim == 0) this.x.scalarCount else this.x.shape.strides(dim - 1)))
        for (sum <- DataLoop(this.x.shape(dim))) {
          // looping over the dims lower than dim
          for (low <- DataLoop(this.x.shape.strides(dim))) {
            this.d.data(offthis + low) += y.d.data(offres + low)
          }
          offthis += this.x.shape.strides(dim)
        }
      }
    }

    def batchNorm(scale: TensorR, bias: TensorR, runningMean: Tensor, runningVar: Tensor): TensorR @diff =
      shift { (k: TensorR => Unit) =>
        val (y, saveMean, saveInvVariance) = x.batchNormTraining(scale.x, bias.x, runningMean, runningVar)
        val ty = TensorR(y); k(ty);
        backend.batchNorm_grad(this, ty, scale, bias, saveMean, saveInvVariance)
      }

    def batchNormAv(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.batchNormAv()); k(y)

      // back-propagate
      val base: Rep[Float] = this.x.shape(0) * this.x.shape(2) * this.x.shape(3) * 1.0f

      for (batch <- DataLoop(this.x.shape(0))) {
        val offsetBatch = batch * this.x.shape.strides(0)
        for (channel <- DataLoop(this.x.shape(1))) {
          val offset = offsetBatch + channel * this.x.shape.strides(1)
          for (lower <- DataLoop(this.x.shape.strides(1))) {
            this.d.data(offset + lower) += y.d.data(channel) / base
          }
        }
      }
    }

    def softmax_batch(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.softmax_batch()); k(y)
      backend.softmax_grad(this, y)
    }

    def logSoftmax(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.logSoftmax()); k(y)

      val s = y.d.sum().data(0)
      for (i <- 0 until y.x.scalarCount: Rep[Range]) {
        this.d.data(i) += y.d.data(i) - Math.exp(y.x.data(i)).toFloat * s
      }
    }

    def logSoftmaxB(): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.logSoftmaxB()); k(y)
      backend.logSoftmax_grad(this, y)
    }

    def resize(dims: Int*): TensorR @diff = shift { (k: TensorR => Unit) =>
      k(new TensorR(this.x.resize(dims : _*), this.d.resize(dims : _*)))
    }

    def nllLossB(target: Rep[Array[Int]]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.nllLossB(target)); k(y)
      generateRawComment("'nllLossB' gradient.")
      backend.nllLoss_grad(this, y, target)
    }

    def nllLoss(target: Rep[Int]): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(x.nllLoss(target)); k(y)
      assert(y.x.isScalar, "y need to be a scalar")
      this.d.data(target) += -1.0f * y.d.data(0)
    }

    @virtualize
    def averagePoolBK(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]] = None): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = TensorR(this.x.averagePool_batch(kernels, strides, pads))
      k(y)

      // back prop
      backend.averagePool2D_batch_grad(this, y, kernels, strides, pads match {case None => Seq(0,0,0,0); case Some(pads) => pads})
    }

    @virtualize  // conv with batch, bias, and pads
    def convBBP(kernel: TensorR, bias: Option[TensorR], strides: Seq[Int], pads: Seq[Int]): TensorR@diff = shift { (k: TensorR => Unit) =>
      assert(this.isInput || this.d.scalarCount == this.x.scalarCount, "For convBBP, THIS is either input or intermediate stage")
      assert(this.x.rank == 4, "For convBBP, THIS is dim 4: batch, channel, row, col")
      assert(pads.tail.forall(x => x == pads.head), "pads should be the same in all directions")
      val (output, finputOption) = bias match {
        case Some(bias) => backend.conv2D_batch(x, kernel.x, Some(bias.x), strides, pads)
        case None =>       backend.conv2D_batch(x, kernel.x, None, strides, pads)
      }
      val y = TensorR(output); k(y)

      generateRawComment("conv2D back-propagate")
      finputOption match {
        case None => backend.conv2D_batch_grad(this, None, kernel, y, bias, (pads.head, pads.head), (strides.head, strides.last), dilations = (1, 1))
        case Some(finput) => backend.conv2D_batch_grad(this, Some(TensorR(finput)), kernel, y, bias, (pads.head, pads.head), (strides.head, strides.last), dilations = (1, 1))
      }
    }

    @virtualize  // maxpool with kernel size potentially different from strides, and works with batch dimension! can have optional paddings
    def maxPoolBK(kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]] = None): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = backend.maxPool2D_batch(x, kernels, strides, pads)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      pads match {
        case None => backend.maxPool2D_batch_grad(this, ty, sidx, kernels, strides, Seq(0, 0, 0, 0))
        case Some(pads) => backend.maxPool2D_batch_grad(this, ty, sidx, kernels, strides, pads)
      }
    }

    @virtualize
    def maxPool(strideRow: Int, strideCol: Int): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, sidx) = this.x.maxPool(strideRow, strideCol)

      val ty = TensorR(y)
      k(ty)

      for (i <- 0 until y.scalarCount: Rep[Range]) {
        this.d.data(sidx(i)) += ty.d.data(i)
      }
    }

    @virtualize
    def concat(dim: Int, others: TensorR*): TensorR @diff = shift { (k: TensorR => Unit) =>
      val y = this.x.concat(dim, others.map(t => t.x): _*)
      val ty = TensorR(y)
      k(ty)

      // back propagate
      backend.concat_grad(dim, this +: others, ty)
    }

    @virtualize
    def global_ave_batch(): TensorR @diff = shift { k: (TensorR => Unit) =>
      val y = this.x.global_ave_batch()
      val ty = TensorR(y)
      k(ty)

      // back propagate
      val scale = 1.0f / this.x.shape.strides(1)
      val offset = var_new(0)                      // offset of this, should step by this.x.strides(2)
      val res_offset = var_new(0)                  // offset of res, should step by 1
      // looping over each batch
      for (batch <- DataLoop(this.x.shape(0))) {
        // looping over each channel
        for (channel <- DataLoop(this.x.shape(1))) {
          // reflect gradient of ty to this, by scale
          val offset_a = var_new(offset)           // offset of this, should step by this.x.strides(3)
          for (i <- DataLoop(this.x.shape(2))) {
            for (j <- DataLoop(this.x.shape(3))) {
              this.d.data(offset_a + j) += ty.d.data(res_offset) * scale
            }
            offset_a += this.x.shape.strides(2)
          }
          offset += this.x.shape.strides(1)
          res_offset += 1
        }
      }
    }

    @virtualize
    def dropout(prob: Float): TensorR @diff = shift { (k: TensorR => Unit) =>
      val (y, helper, size) = backend.dropout(this.x, prob)
      val ty = TensorR(y); k(ty)
      // back prop
      backend.dropout_grad(this, ty, prob, helper, size)
    }

    def print(msg: String = "", derivative: Boolean = false): Unit = {
      this.x.print(msg)
      if (derivative) {
        if (msg == "")
          printf("=========\\n")
        this.d.print(s"derivative $msg")
      }
    }

    def clear_all() = {
      x.clear()
      d.clear()
    }

    def clear_grad() = {
      d.clear()
    }
  }

  object TensorR {
    def apply(a: Tensor, isInput: Boolean = false): TensorR = {
      val d = if (isInput) Tensor.zeros(2) else Tensor.zeros_like(a)
      val res = new TensorR(a, d)
      res.isInput = isInput
      res
    }
    def apply(a: Rep[Array[Float]], dim0: Int, dim1: Int): TensorR = {
      new TensorR(Tensor(a, dim0, dim1), Tensor.zeros(dim0, dim1))
    }

    // def apply(dim0: Int, dim1: Int): TensorR = {
    //   new TensorR(Tensor.zeros(dim0, dim1), Tensor.zeros(dim0, dim1))
    // }

  }

  // change fun signature for memory leak issue (no more returning of array, just update the array provided by the caller)
  // this is in accordance of the destination-passing style
  // the fun take array[array[double]] of size 2, with the first array to be the x, and the second array to be the d
  def FUNc(f: TensorR => Unit): (TensorR => Unit) = { (x:TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun { (x: Rep[Array[Array[Float]]]) =>
      val tensor = new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*))
      f(tensor)
    }
    val in = NewArray[Array[Float]](2)
    in(0) = x.x.data; in(1) = x.d.data
    f1(in) // f1 should take Array[Array[Float]] and update the gradient of x
  }

  def FUNm(f: ArrayBuffer[TensorR] => Unit): (ArrayBuffer[TensorR] => Unit) = { (x: ArrayBuffer[TensorR]) =>
    val dims = x.map(_.x.shape.toSeq)
    val f1 = fun { (x: Rep[Array[Array[Float]]]) =>
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until dims.length): Range) {
        tensors.append(new TensorR(Tensor(x(u * 2), dims(u) : _*), Tensor(x(u*2+1), dims(u) : _*)))
      }
      f(tensors)
    }
    val in = NewArray[Array[Float]](2 * dims.length)
    for (u <- (0 until dims.length): Range) {
      in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
    }
    f1(in)
  }

  @virtualize
  def IF(c: Rep[Boolean])(a: =>TensorR @diff)(b: =>TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    val k1 = FUNc(k)

    if (c) RST(k1(a)) else RST(k1(b))
  }

  @virtualize
  def IFm(c: Rep[Boolean])(a: => ArrayBuffer[TensorR] @diff)(b: => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
    val k1 = FUNm(k)
    if (c) RST(k1(a)) else RST(k1(b))
  }

  @virtualize
  def LOOP(init: TensorR)(c: TensorR => Rep[Boolean])(b: TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>

    lazy val loop: TensorR => Unit = FUNc { (x: TensorR) =>
      if (c(x)) RST(loop(b(x))) else RST(k(x))
    }
    loop(init)
  }

  def FUNs(f: Rep[Int] => TensorR => Unit): (Rep[Int] => TensorR => Unit) = { (i: Rep[Int]) => (x:TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun { (i: Rep[Int], x: Rep[Array[Array[Float]]]) =>
      val tensor = new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*))
      f(i)(tensor)
    }
    val in = NewArray[Array[Float]](2)
    in(0) = x.x.data; in(1) = x.d.data
    f1(i, in)
  }

  @virtualize
  def LOOPS(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k:(TensorR => Unit) =>
    lazy val loop: Rep[Int] => TensorR => Unit = FUNs { (i: Rep[Int]) => (x: TensorR) =>
      if (i < c) { RST(loop(i+1)(b(i)(x))) } else RST(k(x))
    }
    loop(0)(init)
  }

  def FUNsm(f: Rep[Int] => ArrayBuffer[TensorR] => Unit): (Rep[Int] => ArrayBuffer[TensorR] => Unit) = { (i: Rep[Int]) => (x:ArrayBuffer[TensorR]) =>
    val dims = x.map(_.x.shape.seq)
    val f1 = fun { (i: Rep[Int], x: Rep[Array[Array[Float]]]) =>
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until dims.length): Range) {
        tensors.append(new TensorR(Tensor(x(u*2), dims(u) : _*), Tensor(x(u*2+1), dims(u) : _*)))
      }
      f(i)(tensors)
    }
    val in = NewArray[Array[Float]](2 * dims.length)
    for (u <- (0 until dims.length): Range) {
      in(u*2) = x(u).x.data; in(u*2+1) = x(u).d.data
    }
    f1(i, in)
  }

  @virtualize
  def LOOPSM(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
  ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>
    lazy val loop: Rep[Int] => ArrayBuffer[TensorR] => Unit = FUNsm { (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) =>
      if (i < c) {
        RST(loop(i+1)(b(i)(x)))
      } else {
        RST(k(x))
      }
    }
    loop(0)(init)
  }

  def FUN0(f: ((TensorR => Unit) => TensorR => Unit)): ((TensorR => Unit) => TensorR => Unit) = { k1: (TensorR => Unit) => (x: TensorR) =>
    val dims = x.x.shape.toSeq
    val f1 = fun { (t1: Rep[Array[Array[Float]] => Unit], xx: Rep[Array[Array[Float]]]) =>
      val t2: (TensorR => Unit) = { (x: TensorR) =>
        val temp = NewArray[Array[Float]](2)
        temp(0) = x.x.data; temp(1) = x.d.data
        t1(temp)
      }
      val t3: (TensorR => Unit) = f(t2)
      t3(new TensorR(Tensor(xx(0), dims: _*), Tensor(xx(1), dims: _*)))
    }
    val k2: Rep[Array[Array[Float]] => Unit] = fun { (x: Rep[Array[Array[Float]]]) =>
      k1(new TensorR(Tensor(x(0), dims: _*), Tensor(x(1), dims: _*)))
    }
    val temp = NewArray[Array[Float]](2)
    temp(0) = x.x.data; temp(1) = x.d.data
    f1(k2, temp)
  }

  def FUNl(f: (Rep[Int] => (TensorR => Unit) => TensorR => Unit)): (Rep[Int] => (TensorR => Unit) => TensorR => Unit) = {i: Rep[Int] => k1: (TensorR => Unit) => (x: TensorR) =>

    val dims = x.x.shape.toSeq

    val f1 = fun { (i: Rep[Int], t1: Rep[((Array[Float], Array[Float])) => Unit], x0: Rep[Array[Float]], x1: Rep[Array[Float]]) =>
      val t2: (TensorR => Unit) = { (x: TensorR) =>
        t1(x.x.data, x.d.data)
      }
      val t3: (TensorR => Unit) = f(i)(t2)
      t3(new TensorR(Tensor(x0, dims: _*), Tensor(x1, dims: _*)))
    }

    val k2: Rep[((Array[Float], Array[Float])) => Unit] = fun { (x1: Rep[Array[Float]], x2: Rep[Array[Float]]) =>
      k1(new TensorR(Tensor(x1, dims: _*), Tensor(x2, dims: _*)))
    }
    f1(i, k2, x.x.data, x.d.data)
  }

  @virtualize
  def LOOPL(init: TensorR)(c: Rep[Int])(b: Rep[Int] => TensorR => TensorR @diff): TensorR @diff = shift { k: (TensorR => Unit) =>
    lazy val loop: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (gc: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
      def sh_loop: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => loop(i)(k)(x)}
      RST(k (IF(gc < c) { b(gc)(sh_loop(gc+1)) } { init }) )
    }
    loop(0)(k)(init)
  }

  @virtualize
  def LOOPT(start: Rep[Int])(init: TensorR)(lch: Rep[Array[Int]], rch: Rep[Array[Int]])(b: (TensorR, TensorR, Rep[Int]) => TensorR @diff): TensorR @diff = shift {
    k: (TensorR => Unit) =>

      lazy val tree: Rep[Int] => (TensorR => Unit) => TensorR => Unit = FUNl{ (i: Rep[Int]) => (k: TensorR => Unit) => (x: TensorR) =>
        def sh_tree: (Rep[Int] => TensorR @diff) = (i: Rep[Int]) => shift{(k: TensorR => Unit) => tree(i)(k)(x)}
        RST(k( IF(i >= 0) { b(sh_tree(lch(i)), sh_tree(rch(i)), i) } { init } ))
      }
      tree(start)(k)(init)
  }

  def FUNlm(f: (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit)):
  (Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit) = {i: Rep[Int] => k1: (ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

    val length = x.length
    val dims = x.map(_.x.shape.toSeq)
    val f1 = fun { (i: Rep[Int], t1: Rep[Array[Array[Float]] => Unit], xx: Rep[Array[Array[Float]]]) =>
      val t2: (ArrayBuffer[TensorR] => Unit) = { (x: ArrayBuffer[TensorR]) =>
        val aa = NewArray[Array[Float]](2*length)
        for (u <- (0 until length): Range) {
          aa(u*2) = x(u).x.data; aa(u*2+1) = x(u).d.data
        }
        t1(aa)
      }
      val t3: (ArrayBuffer[TensorR] => Unit) = f(i)(t2)
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until length): Range) {
        tensors.append(new TensorR(Tensor(xx(u*2), dims(u): _*), Tensor(xx(u*2+1), dims(u): _*)))
      }
      t3(tensors)
    }
    val k2: Rep[Array[Array[Float]] => Unit] = fun { (x: Rep[Array[Array[Float]]]) =>
      val tensors = ArrayBuffer[TensorR]()
      for (u <- (0 until length): Range) {
        tensors.append(new TensorR(Tensor(x(u*2), dims(u): _*), Tensor(x(u*2+1), dims(u): _*)))
      }
      k1(tensors)
    }
    val arrays = NewArray[Array[Float]](2*length)
    for (u <- (0 until length): Range) {
      arrays(u*2) = x(u).x.data; arrays(u*2+1) = x(u).d.data
    }
    f1(i, k2, arrays)
  }

  @virtualize
  def LOOPLM(start: Rep[Int])(init: ArrayBuffer[TensorR])(c: Rep[Int])(b: Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff):
  ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

    lazy val loop: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

      def sh_loop: (Rep[Int] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => shift{ (k: ArrayBuffer[TensorR] => Unit) => loop(i)(k)(x) }

      RST(k( IFm(i < c) { b(i)(sh_loop(i+1)) } { init } ))
    }
    loop(start)(k)(init)
  }

  @virtualize
  def LOOPTM(start: Rep[Int])(init: ArrayBuffer[TensorR])(lch: Rep[Array[Int]], rch: Rep[Array[Int]])
  (b: (ArrayBuffer[TensorR], ArrayBuffer[TensorR], Rep[Int]) => ArrayBuffer[TensorR] @diff): ArrayBuffer[TensorR] @diff = shift { k: (ArrayBuffer[TensorR] => Unit) =>

      lazy val tree: Rep[Int] => (ArrayBuffer[TensorR] => Unit) => ArrayBuffer[TensorR] => Unit = FUNlm { (i: Rep[Int]) => (k: ArrayBuffer[TensorR] => Unit) => (x: ArrayBuffer[TensorR]) =>

        def sh_tree: (Rep[Int] => ArrayBuffer[TensorR] => ArrayBuffer[TensorR] @diff) = (i: Rep[Int]) => (x: ArrayBuffer[TensorR]) => shift{(k: ArrayBuffer[TensorR] => Unit) => tree(i)(k)(x)}

        RST(k( IFm (i >= 0) { b(sh_tree(lch(i))(init), sh_tree(rch(i))(init), i) } { init } ))
      }
      tree(start)(k)(init)
  }

  def gradR(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = new TensorR(x, Tensor.zeros_like(x))
    reset {
      val y = f(x1)
      y.d.setAsOne()
      ()
    }
    x1.d
  }

  // same as gradR function, except that we return the final result of f, not the gradient of input
  // gradient of input is supposed to be dummy value here
  // gradient of useful tensors are in closure, and can be accessed directly from outside of this function
  def gradR_loss(f: TensorR => TensorR @diff)(x: Tensor): Tensor = {
    val x1 = TensorR(x) // this should be a dummy tensor
    // val result = Tensor.zeros(1)                  // this should be the loss
    generateRawComment("allocate memory to save the final loss in CPU Tensor")
    val res = BackendCPU().mallocArray[Float](1)
    val result = Tensor(res, 1)
    reset {
      val y = f(x1)
      assert(y.x.scalarCount == 1, s"Loss function must return a Tensor of size 1, got ${y.x.scalarCount}")
      y.d.setAsOne()
      generateRawComment(s"backend is $backend")
      // result.copy_data(y.x)
      if (backend.isInstanceOf[BackendCPU]) BackendCPU().copyFloatArray(res, y.x.data, 1)
      else unchecked[Unit]("CUDA_CALL(cudaMemcpy(", res, ", ", y.x.data, ", ", 1, " * sizeof(float), cudaMemcpyDeviceToHost))")
      ()
    }
    result
  }

  def getMallocAddr(): Rep[Long] = {
    unchecked[Long]("(long)mallocAddr")
  }

  def resetMallocAddr(addr: Rep[Long]) = {
    unchecked[Unit]("memset((void*)", addr, ", 0, ", getMallocAddr() - addr, ")")
    unchecked[Unit]("mallocAddr = (void*)", addr)
  }
}

trait TensorDslCublas extends TensorDsl with GPUOps {

  def getCudaMallocAddr(): Rep[Long] = {
    unchecked[Long]("(long)gpuMallocAddr")
  }

  def resetCudaMallocAddr(addr: Rep[Long]) = {
    unchecked[Unit]("cudaMemset((void*)", addr, ", 0, ", getCudaMallocAddr() - addr, ")")
    unchecked[Unit]("gpuMallocAddr = (void*)", addr)
  }

  // NOTE: `cudaMemset` is not very useful because it only works with an integer array/value.
  protected def cudaMemset(array: Rep[Array[Int]], value: Rep[Int], n: Int): Rep[Unit] =
    unchecked[Unit]("CUDA_CALL(cudaMemset((void **)&", array, ", ", value, ", ", n, " * sizeof(int)))")

  protected def cublasSetPointerModeDevice(): Rep[Unit] =
    unchecked[Unit]("cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE)")

  protected def cublasSetPointerModeHost(): Rep[Unit] =
    unchecked[Unit]("cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST)")

  class ArrayTransferOps[T: Manifest](array: Rep[Array[T]]) {
    // Get a CPU-allocated copy of this array.
    def toCPU(length: Rep[Int]): Rep[Array[T]] = {
      val res = BackendCPU().mallocArray[T](length)
      gpu_array_copy_device_to_host(array, res, length)
      res
    }

    // Get a GPU-allocated copy of this array.
    def toGPU(length: Rep[Int]): Rep[Array[T]] = {
      val res = BackendGPU.mallocArray[T](length)
      gpu_array_copy_host_to_device(array, res, length)
      res
    }

    // Move the underlying data of this array to the CPU.
    def moveToCPU(length: Rep[Int]): Unit = {
      val res = BackendCPU().mallocArray[T](length)
      gpu_array_copy_device_to_host(array, res, length)
      unchecked[Unit](array, " = ", res)
    }

    // Move the underlying data of this array to the GPU.
    def moveToGPU(length: Rep[Int]): Unit = {
      val res = BackendGPU.mallocArray[T](length)
      gpu_array_copy_host_to_device(array, res, length)
      unchecked[Unit](array, " = ", res)
    }
  }
  implicit def arrayToTransferOps[T: Manifest](array: Rep[Array[T]]) = new ArrayTransferOps(array)

  // Tensor backend transfer operations.
  class TensorTransferOps(t: Tensor) {
    // Get a CPU-allocated copy of this tensor.
    def toCPU(): Tensor = {
      generateRawComment("Tensor 'toCPU' invocation.")
      new Tensor(t.data.toCPU(t.scalarCount), t.shape)
    }

    // Get a GPU-allocated copy of this tensor.
    def toGPU(): Tensor = {
      generateRawComment("Tensor 'toGPU' invocation.")
      // val res = BackendGPU.mallocArray[Float](t.scalarCount)
      new Tensor(t.data.toGPU(t.scalarCount), t.shape)
    }

    // Move the underlying data of this tensor to the CPU.
    def moveToCPU(): Unit = {
      generateRawComment("Tensor 'moveToCPU' invocation.")
      t.data.moveToCPU(t.scalarCount)
    }

    // Move the underlying data of this tensor to the GPU.
    def moveToGPU(): Unit = {
      generateRawComment("Tensor 'moveToGPU' invocation.")
      t.data.moveToGPU(t.scalarCount)
    }
  }
  implicit def tensorToTransferOps(t: Tensor) = new TensorTransferOps(t)

  class TensorRTransferOps(t: TensorR) {
    def toCPU(): TensorR = new TensorR(t.x.toCPU(), t.d.toCPU())
    def toGPU(): TensorR = {
      val temp = new TensorR(t.x.toGPU(), t.d.toGPU())
      temp.isInput = t.isInput
      temp
    }
    def moveToCPU(): Unit = { t.x.moveToCPU(); t.d.moveToCPU() }
    def moveToGPU(): Unit = { t.x.moveToGPU(); t.d.moveToGPU() }
  }
  implicit def tensorRToTransferOps(t: TensorR) = new TensorRTransferOps(t)

  /**
    * cuBLAS tensor operation backend. WIP.
    */
  class BackendCublas protected() extends Backend {
    override def setup(): Unit = generateRawCode(
      """cublasHandle_t cublasHandle;
        |CUBLAS_CALL(cublasCreate(&cublasHandle));
        |CUDA_CALL(cudaMalloc(&gpuMallocBase, HEAP_SIZE));
        |CUDA_CALL(cudaMemset(gpuMallocBase, 0, HEAP_SIZE));
        |gpuMallocAddr = gpuMallocBase;
      """.stripMargin)

    override def cleanup(): Unit = generateRawCode(
      """CUBLAS_CALL(cublasDestroy(cublasHandle));
        |CUDA_CALL(cudaFree(gpuMallocBase));
      """.stripMargin)

    override def mallocArray[T: Manifest](length: Rep[Int]): Rep[Array[T]] = NewGPUArray[T](length)

    override def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Int): Unit =
      gpu_array_copy_device_to_device(src, dest, length)

    override def arrayToTensor(array: Rep[Array[Float]], dims: Int*): Tensor = new Tensor(array, dims)
      // BackendCPU().arrayToTensor(array, dims: _*).toGPU()

    override def makeTensor(dims: Seq[Int], scalars: Float*): Tensor =
      BackendCPU().makeTensor(dims, scalars: _*).toGPU()

    // TOOO: Compare performance with GPU allocation + `fillInPlace`.
    override def fill(dims: Seq[Int], value: Rep[Float]): Tensor =
      BackendCPU().fill(dims, value).toGPU()

    override def fillWithBias(dims: Seq[Int], bias: Tensor, dim: Int): Tensor =
      BackendCPU().fillWithBias(dims, bias.toCPU(), dim).toGPU()

    override def fillInPlace(x: Tensor, value: Rep[Float]): Unit = {
      // TODO: Consider different grid/block parameters.
      unchecked[Unit](s"arrayFill<<<${x.scalarCount}, 1>>>(", x.data, ", ", value, ")")
    }

    // TODO: Implement random initialization using cuRAND API.
    override def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor =
      BackendCPU().randinit(dims, scale, seed).toGPU()

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot
    // NOTE: `sdot` fails when the cuBLAS pointer mode is host (as opposed to device).
    // Investigate performance impact.
    def sdot(n: Int, a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      generateRawComment("calling Sdot API function")
      unchecked[Unit]("CUBLAS_CALL(cublasSdot(cublasHandle, ", n, ",", a, ",", 1, ",", b, ",", 1, ",", result, "))")
    }

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      val res = BackendCPU().mallocArray[Float](1)
      generateRawComment("calling sdot from vectorVectorDot function")
      sdot(x.scalarCount, x.data, y.data, res)
      Tensor(res, 1).toGPU()  // TODO (Fei Wang): if use GPU memory for result, there is segfault
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
    def sgemv(m: Int, n: Int, matrix: Rep[Array[Float]], vector: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemv(cublasHandle, CUBLAS_OP_T, ",
        n, ",", m, ",", one, ",",
        matrix, ",", n, ",", vector, ",", 1, ",", zero, ",", result, ",", 1, "))")
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = x.shape(1)
      val res = mallocArray[Float](m)
      sgemv(m, n, x.data, y.data, res)
      Tensor(res, m)
    }

    // Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    def sgemm(m: Int, n: Int, k: Int, a: Rep[Array[Float]], b: Rep[Array[Float]], result: Rep[Array[Float]]) = {
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
        n, ",", m, ",", k, ",", one, ",",
        b, ",", n, ",", a, ",", k, ",", zero, ",", result, ",", n, "))")
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      val m = x.shape(0)
      val n = y.shape(1)
      val k = y.shape(0)
      val res = mallocArray[Float](m * n)
      sgemm(m, n, k, x.data, y.data, res)
      Tensor(res, m, n)
    }

    override def dot_grad(x: TensorR, y: TensorR, output: TensorR): Unit = {
      // use CuBLAS instead
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      (x.x.rank, y.x.rank) match {
        case (1, 1) =>
          val dim = x.x.shape(0)
          val scale = output.d.toCPU()
          // x.d.addMul(output.d.data(0), y.x)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim, ",", 1, ",", one, ",",
            x.d.data, ",", dim, ",", scale.data, ", ", y.x.data, ", ", dim, ", ", x.d.data, ",", dim, "))")
          // y.d.addMul(output.d.data(0), x.x)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim, ",", 1, ",", one, ",",
            y.d.data, ",", dim, ",", scale.data, ", ", x.x.data, ", ", dim, ", ", y.d.data, ",", dim, "))")
        case (2, 1) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1)
          // that.d.add_composion(this.x, y.d)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemv(cublasHandle, CUBLAS_OP_N, ",
            dim2, ",", dim1, ",", one, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", 1, ",", one, ",", y.d.data, ",", 1, "))")
          // x.d.add_cartesian(y.x, output.d);
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim2, ", ", dim1, ", ", 1, ", ", one, ", ",
            y.x.data, ", ", dim2, ", ", output.d.data, ", ", 1, ", ", one, ", ", x.d.data, ", ", dim2, "))")
        case (2, 2) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          generateRawComment("backprop of matrix-matrix-dot")
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            dim2, ",", dim1, ",", dim3, ",", one, ",",
            y.x.data, ",", dim3, ",", output.d.data, ",", dim3, ",", one, ",", x.d.data, ",", dim2, "))")
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            dim3, ",", dim2, ",", dim1, ",", one, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim2, ",", one, ",", y.d.data, ",", dim3, "))")
      }
    }

    // Compute broadcasting strides.
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorIterator.cpp#L396
    // TODO: Generalize for different-ranked tensors. Currently, broadcasting works only for same-rank tensors.
    def getBroadcastingStrides(shape: Dimensions): Seq[Int] = {
      shape.strides.zipWithIndex.map { case (s, i) =>
        if (shape(i) == 1) 0 else s
      }
    }

    def launchUnaryKernel(res: Tensor, x: Tensor)(op: String => Seq[Any]): Unit = {
      assert(res.shape == x.shape, s"Unary kernel incompatible shapes: ${res.shape.seq}, ${x.shape.seq}")

      // Store shapes as local variables.
      val resShape = res.shape
      // Convert shapes to Rep[Array[Int]].
      val resDims = Array(resShape.map(unit(_)): _*)
      // Compute strides.
      val strides = NewArray[Array[Int]](2)
      val tmp = Array(getBroadcastingStrides(resShape).map(unit(_)): _*)
      strides(0) = tmp
      strides(1) = tmp
      // Launch kernel.
      // NOTE: Hacky way to propagate `Rep[Float]` as an argument to `unchecked`.
      unchecked[Unit](
        Seq("{\n" +
        "OffsetCalculator<2> calc(", resShape.length, ",", resDims, ",", strides, "); \n" +
        "launch_kernel<128, 4>(", resShape.scalarCount, ", [=]__device__(int idx) {\n" +
        "  auto offsets = calc.get(idx);\n" +
        "  float* out = (float*)&", res.data, "[offsets[0]];\n" +
        "  float* in = (float*)&", x.data, "[offsets[1]];\n" +
        "  *out = ") ++ op("(*in)") ++ Seq(";\n" +
        "});\n" +
        "}"): _*)
    }

    def elementwiseUnaryOp(x: Tensor)(op: String => Seq[Any]): Tensor = {
      val resData = mallocArray[Float](x.scalarCount)
      val res = Tensor(resData, x.shape: _*)
      launchUnaryKernel(res, x)(op)
      res
    }

    def elementwiseInplaceUnaryOp(x: Tensor)(op: String => Seq[Any]): Unit = {
      launchUnaryKernel(x, x)(op)
    }

    def launchBinaryKernel(res: Tensor, x: Tensor, y: Tensor)(op: (String, String) => String): Unit = {
      // Store shapes as local variables.
      val resShape = res.shape
      val xShape = x.shape
      val yShape = y.shape
      // Convert shapes to Rep[Array[Int]].
      val resDims = Array(resShape.map(unit(_)): _*)
      val xDims = Array(xShape.map(unit(_)): _*)
      val yDims = Array(yShape.map(unit(_)): _*)
      // Compute strides.
      val strides = NewArray[Array[Int]](3)
      strides(0) = Array(getBroadcastingStrides(resShape).map(unit(_)): _*)
      strides(1) = Array(getBroadcastingStrides(xShape).map(unit(_)): _*)
      strides(2) = Array(getBroadcastingStrides(yShape).map(unit(_)): _*)
      // Launch kernel.
      unchecked[Unit](
        "{\n" +
        "OffsetCalculator<3> calc(", resShape.length, ",", resDims, ",", strides, "); \n" +
        "launch_kernel<128, 4>(", resShape.scalarCount, ", [=]__device__(int idx) {\n" +
        "  auto offsets = calc.get(idx);\n" +
        "  float* out = (float*)&", res.data, "[offsets[0]];\n" +
        "  float* in1 = (float*)&", x.data, "[offsets[1]];\n" +
        "  float* in2 = (float*)&", y.data, "[offsets[2]];\n" +
        s"  *out = ${op("(*in1)", "(*in2)")};\n" +
        "});\n" +
        "}")
    }

    def elementwiseBinaryOp(x: Tensor, y: Tensor)(op: (String, String) => String): Tensor = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case None => throw new IllegalArgumentException(s"Shapes cannot be broadcasted: ${x.shape.seq}, ${y.shape.seq}")
        case Some((xShape, yShape, resShape)) =>
          val resData = mallocArray[Float](resShape.scalarCount)
          val res = Tensor(resData, resShape: _*)
          launchBinaryKernel(res, x, y)(op)
          res
      }
    }

    def elementwiseInplaceBinaryOp(x: Tensor, y: Tensor)(op: (String, String) => String): Unit = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case None => throw new IllegalArgumentException(s"Shapes cannot be broadcasted: ${x.shape.seq}, ${y.shape.seq}")
        case Some((xShape, yShape, resShape)) =>
          assert(x.shape == resShape, s"Output shape ${xShape.seq} does not match broadcast shape: ${resShape.seq}")
          launchBinaryKernel(x, x, y)(op)
      }
    }

    override def +(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " + ", y))
    override def +(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " + " + _ }

    override def +=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " + ", y))
    override def +=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " + " + _ }

    override def -(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " - ", y))
    override def -(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " - " + _ }

    override def -=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " - ", y))
    override def -=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " - " + _ }

    override def *(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " * ", y))
    override def *(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " * " + _ }

    override def *=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " * ", y))
    override def *=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " * " + _ }

    override def /(x: Tensor, y: Rep[Float]): Tensor = elementwiseUnaryOp(x)(s => Seq(s + " / ", y))
    override def /(x: Tensor, y: Tensor): Tensor = elementwiseBinaryOp(x, y) { _ + " / " + _ }

    override def /=(x: Tensor, y: Rep[Float]): Unit = elementwiseInplaceUnaryOp(x)(s => Seq(s + " / ", y))
    override def /=(x: Tensor, y: Tensor): Unit = elementwiseInplaceBinaryOp(x, y) { _ + " / " + _ }

    override def plusBias(main: Tensor, bias: Tensor): Tensor = { ??? }
    override def plusBias_grad(main: TensorR, bias: TensorR): Unit = { ??? }

    override def geam(x: Tensor, transX: Boolean, alpha: Float, y: Tensor, transY: Boolean, beta: Float, output: Tensor): Unit = {
      val alpha1 = NewArray[Float](1); alpha1(0) = alpha
      val beta1 = NewArray[Float](1); beta1(0) = beta
      (transX, transY) match {
        case (false, false) =>
          assert(x.shape == y.shape && x.shape == output.shape, "TODO: only handle uniform shape (no transpose) for now")
          val m = x.shape(0)
          val n = x.shape.drop(1).product
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", n, ",", beta1, ", ", y.data, ", ", n, ", ", output.data, ",", n, "))")
        case (false, true) =>
          assert(x.rank == 2 && y.rank == 2 && x.shape(0) == y.shape(1) && x.shape(1) == y.shape(0))
          val m = x.shape(0)
          val n = x.shape(1)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", n, ",", beta1, ", ", y.data, ", ", m, ", ", output.data, ",", n, "))")
        case (true, false) =>
          assert(x.rank == 2 && y.rank == 2 && x.shape(0) == y.shape(1) && x.shape(1) == y.shape(0))
          val m = x.shape(1)
          val n = x.shape(0)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", m, ",", beta1, ", ", y.data, ", ", n, ", ", output.data, ",", n, "))")
        case (true, true) =>
          assert(x.rank == 2 && y.rank == 2 && x.shape == y.shape)
          val m = x.shape(1)
          val n = x.shape(0)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            n, ",", m, ",", alpha1, ",",
            x.data, ",", m, ",", beta1, ", ", y.data, ", ", m, ", ", output.data, ",", n, "))")
      }
    }

    override def trans(x: Tensor): Tensor = {
      assert(x.rank == 2, s"trans only supported for 2D matrix, got ${x.shape.seq}")
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape.reverse: _*)
      generateRawComment("trans casted as geam call")
      this.geam(x, true, 1.0f, x, true, 0.0f, res)
      res
    }

    override def trans_grad(x: TensorR, y: TensorR): Unit = {
      assert(x.x.rank == 2 && y.x.rank == 2 && x.x.shape.reverse == y.x.shape)
      this.geam(x.d, false, 1.0f, y.d, true, 1.0f, x.d)
    }

    override def gemm(x: Tensor, transX: Boolean, y: Tensor, transY: Boolean, alpha: Float): Tensor = {
      (transX, transY) match {
        case (false, false) =>
          val m = x.shape(0)
          val n = y.shape(1)
          val k = y.shape(0)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", n, ",", x.data, ",", k, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
        case (false, true) =>
          val m = x.shape(0)
          val n = y.shape(0)
          val k = y.shape(1)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", k, ",", x.data, ",", k, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
        case (true, false) =>
          val m = x.shape(1)
          val n = y.shape(1)
          val k = y.shape(0)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", n, ",", x.data, ",", m, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
        case (true, true) =>
          val m = x.shape(1)
          val n = y.shape(0)
          val k = y.shape(1)
          val res = mallocArray[Float](m * n)
          val zero = NewArray[Float](1); zero(0) = 0
          val Alpha = NewArray[Float](1); Alpha(0) = alpha
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            n, ",", m, ",", k, ",", Alpha, ",",
            y.data, ",", k, ",", x.data, ",", m, ",", zero, ",", res, ",", n, "))")
          Tensor(res, m, n)
      }
    }

    override def gemm_grad(x: TensorR, transX: Boolean, y: TensorR, transY: Boolean, alpha: Float, output: TensorR): Unit = {
      val alpha1 = NewArray[Float](1); alpha1(0) = alpha;
      val one = NewArray[Float](1); one(0) = 1.0f;
      generateRawComment("backprop of gemm")
      (transX, transY) match {
        case (false, false) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            dim2, ",", dim1, ",", dim3, ",", alpha1, ",",
            y.x.data, ",", dim3, ",", output.d.data, ",", dim3, ",", one, ",", x.d.data, ",", dim2, "))")
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            dim3, ",", dim2, ",", dim1, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim2, ",", one, ",", y.d.data, ",", dim3, "))")
        case (false, true) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(0)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim2, ",", dim1, ",", dim3, ",", alpha1, ",",
            y.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", one, ",", x.d.data, ",", dim2, "))")
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ",
            dim2, ",", dim3, ",", dim1, ",", alpha1, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", one, ",", y.d.data, ",", dim2, "))")
        case (true, false) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(1)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ",
            dim1, ",", dim2, ",", dim3, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim3, ",", one, ",", x.d.data, ",", dim1, "))")
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ",
            dim3, ",", dim2, ",", dim1, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim1, ",", one, ",", y.d.data, ",", dim3, "))")
        case (true, true) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(0)
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            dim1, ",", dim2, ",", dim3, ",", alpha1, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim2, ",", one, ",", x.d.data, ",", dim1, "))")
          unchecked[Unit](
            "CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, ",
            dim2, ",", dim3, ",", dim1, ",", alpha1, ",",
            x.x.data, ",", dim1, ",", output.d.data, ",", dim3, ",", one, ",", y.d.data, ",", dim2, "))")
      }
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) = ???
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = ???
    override def maxPool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]]) = ???
    override def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]], kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = ???

    override def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor = ???
    override def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = ???

    override def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = ???
    override def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor]) = ???
    override def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor]): Unit = ???

    override def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int]) = ???
    override def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit = ???

    override def relu(x: Tensor): Tensor = ???
    override def tanh(x: Tensor): Tensor = ???
    override def sigmoid(x: Tensor): Tensor = ???
    override def relu_grad(input: TensorR, res: TensorR): Unit = ???
    override def tanh_grad(input: TensorR, res: TensorR): Unit = ???
    override def sigmoid_grad(input: TensorR, res: TensorR): Unit = ???

    override def softmax(x: Tensor): Tensor = ???
    override def logSoftmax(x: Tensor): Tensor = ???
    override def softmax_grad(input: TensorR, res: TensorR): Unit = ???
    override def logSoftmax_grad(input: TensorR, res: TensorR): Unit = ???

    override def exp(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Exp)
    override def exp_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.ExpGrad)

    override def log(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Log)
    override def log_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.LogGrad)

    override def sqrt(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Sqrt)
    override def sqrt_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.SqrtGrad)

    override def square(x: Tensor) = elementwiseOpNoBroadcast(x, ElementWiseNoBroadCastOpt.Square)
    override def square_grad(x: TensorR, y: TensorR): Unit = elementwiseOpNoBroadcastGrad(x, y, ElementWiseNoBroadCastOpt.SquareGrad)

    object ElementWiseNoBroadCastOpt extends Enumeration {
      val Log = Value("LOG")
      val LogGrad = Value("LOG_GRAD")
      val Exp = Value("EXP")
      val ExpGrad = Value("EXP_GRAD")
      val Sqrt = Value("SQRT")
      val SqrtGrad = Value("SQRT_GRAD")
      val Square = Value("SQUARE")
      val SquareGrad = Value("SQUARE_GRAD")
    }

    def elementwiseOpNoBroadcast(input: Tensor, op: ElementWiseNoBroadCastOpt.Value, inplace: Boolean = false): Tensor = {
      if (input.scalarCount <= 512 * 65535) {
        val numBlocks = (input.scalarCount + 511) / 512
        val res = if (inplace) input.data else mallocArray[Float](input.scalarCount)
        op match {
          case ElementWiseNoBroadCastOpt.Log =>
            unchecked[Unit](s"elementwise_1D_1D_log<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ");\n")
          case ElementWiseNoBroadCastOpt.Exp =>
            unchecked[Unit](s"elementwise_1D_1D_exp<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ");\n")
          case ElementWiseNoBroadCastOpt.Sqrt =>
            unchecked[Unit](s"elementwise_1D_1D_sqrt<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ");\n")
          case ElementWiseNoBroadCastOpt.Square =>
            unchecked[Unit](s"elementwise_1D_1D_square<<<${numBlocks},", "512>>>(", input.data, ",", res, ", ", input.scalarCount, ");\n")
          case _ => ???
        }
        Tensor(res, input.shape: _*)
      } else {
        assert(false, s"not implemented for tensors larger than 1D grid * 1D block, got ${input.shape}")
        ???
      }
    }

    def elementwiseOpNoBroadcastGrad(input: TensorR, output: TensorR, op: ElementWiseNoBroadCastOpt.Value): Unit = {
      if (input.x.scalarCount <= 512 * 65535) {
        val numBlocks = (input.x.scalarCount + 511) / 512
        op match {
          case ElementWiseNoBroadCastOpt.LogGrad =>
            unchecked[Unit](s"elementwise_1D_1D_log_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ");\n")
          case ElementWiseNoBroadCastOpt.ExpGrad =>
            unchecked[Unit](s"elementwise_1D_1D_exp_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ");\n")
          case ElementWiseNoBroadCastOpt.SqrtGrad =>
            unchecked[Unit](s"elementwise_1D_1D_sqrt_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ");\n")
          case ElementWiseNoBroadCastOpt.SquareGrad =>
            unchecked[Unit](s"elementwise_1D_1D_square_grad<<<${numBlocks},", "512>>>(", input.x.data, ", ", input.d.data, ", ", output.x.data, ", ", output.d.data, ", ", input.x.scalarCount, ");\n")
          case _ => ???
        }
      } else {
        assert(false, s"not implemented for tensors larger than 1D grid * 1D block, got ${input.x.shape}")
        ???
      }
    }

    override def nllLoss(x: Tensor, target: Rep[Array[Int]]): Tensor = {
      assert(x.rank == 2, "Input must be a 2-D tensor")

      val batchSize = x.shape(0)
      val res = Tensor(mallocArray[Float](batchSize), batchSize)
      unchecked[Unit](
        s"nllLoss<<<${batchSize}, 1>>>(", x.data, ",", x.shape.strides(0), ",",
        res.data, ",", target, ")")
      res
    }

    override def nllLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Int]]): Unit = {
      unchecked[Unit](
        s"nllLoss_grad<<<${input.d.shape(0)}, 1>>>(", input.d.shape.strides(0), ",",
        res.d.data, ",", target, ",", input.d.data, ")")
    }

    override def sum(x: Tensor): Tensor = ???
    override def sum_grad(input: TensorR, res: TensorR): Unit = ???
    override def mean(x: Tensor): Tensor = ???
    override def mean_grad(input: TensorR, res: TensorR): Unit = ???

    override def concat(dim: Int, tensors: Seq[Tensor]): Tensor = {
      assert(dim == 1, "TODO (Fei Wang): only support dim = 1 so far")
      assert(tensors.size == 2, "TODO: (Fei Wang): only support two tensor concatenation so far")
      assert(tensors(0).rank == 4 && tensors(1).rank == 4, "TODO: (Fei Wang): only support 4D concat so far")

      val dim0 = tensors(0).shape(0)
      val dim1 = tensors(0).shape(1) + tensors(1).shape(1)
      val dim2 = tensors(0).shape(2)
      val dim3 = tensors(0).shape(3)
      val resShape = Seq(dim0, dim1, dim2, dim3)
      val res = this.mallocArray[Float](resShape.product)

      // concatenate with user-define kernel function (1D grid 3D block)
      unchecked[Unit](
        "{\n",
        "dim3 grid(", dim1 * dim2, ", ", dim0, "); \n",
        "concat2D_1D<<<grid", ", ", dim3, ">>>(", tensors(0).data, ",", tensors(1).data, ",", res, ", ", dim2, ", ", tensors(0).shape(1), ");\n",
        "}")
      Tensor(res, dim0, dim1, dim2, dim3)
    }

    override def concat_grad(dim: Int, tensorRs: Seq[TensorR], output: TensorR): Unit = {
      assert(dim == 1, "TODO (Fei Wang): only support dim = 1 so far")
      assert(tensorRs.size == 2, "TODO: (Fei Wang): only support two tensor concatenation so far")
      assert(tensorRs(0).x.rank == 4 && tensorRs(1).x.rank == 4, "TODO: (Fei Wang): only support 4D concat so far")

      val dim0 = tensorRs(0).x.shape(0)
      val dim1 = tensorRs(0).x.shape(1) + tensorRs(1).x.shape(1)
      val dim2 = tensorRs(0).x.shape(2)
      val dim3 = tensorRs(0).x.shape(3)

      // concatenate with user-define kernel function (1D grid 3D block)
      unchecked[Unit](
        "{\n",
        "dim3 grid(", dim1 * dim2, ", ", dim0, ");\n",
        "concat2D_1D_grad<<<grid", ", ", dim3, ">>>(", tensorRs(0).d.data, ",", tensorRs(1).d.data, ",", output.d.data, ", ", dim2, ", ", tensorRs(0).x.shape(1), ");\n",
        "}")
    }
  }

  object BackendCublas {
    def apply() = new BackendCublas
  }

  // Define default GPU backend.
  def BackendGPU: Backend = BackendCublas()
  backend = BackendGPU
}

trait TensorDslCudnn extends TensorDslCublas {
  /**
    * cuDNN tensor operation backend. WIP.
    * Extends `BackendCublas` to leverage cuBLAS primitives.
    */
  class BackendCudnn protected() extends BackendCublas {
    override def setup(): Unit = {
      super.setup()
      generateRawCode("cudnnHandle_t cudnnHandle;\nCUDNN_CALL(cudnnCreate(&cudnnHandle));")
    }

    override def cleanup(): Unit = {
      super.cleanup()
      generateRawCode("CUDNN_CALL(cudnnDestroy(cudnnHandle));")
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnAddTensor
    // Note: this function performs in-place addition for `res`.
    def cudnnAddBiasTensor(bias: Tensor, res: Tensor, scale: Float = 1.0f): Unit = {
      val (biasShape, resShape) = if (bias.shape == res.shape) {
        (bias.shape.padTo(4, 1), res.shape.padTo(4, 1))
      } else {
        assert(bias.rank == 1, "if not equal shape, bias must be rank 1")
        // assert(res.rank >= 2, "if not equal shape, res must be rank 2 or more")
        (Seq(1, bias.shape(0), 1, 1), res.shape.padTo(4, 1))
      }
      val scaled = NewArray[Float](1); scaled(0) = scale
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t bias_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${biasShape(0)}, ${biasShape(1)}, ${biasShape(2)}, ${biasShape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${resShape(0)}, ${resShape(1)}, ${resShape(2)}, ${resShape(3)}));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnAddTensor(\n" +
          "    cudnnHandle, ", scaled, ", bias_desc, ", bias.data, ", ", one, ", out_desc, ", res.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionForward
    def cudnnConvolutionForward(input: Tensor, filter: Tensor, res: Tensor, bias: Option[Tensor] = None,
                                padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      assert(input.rank == 4, s"Convolution input must have rank 4, but got ${input.rank}")
      assert(res.rank == 4, s"Convolution result must have rank 4, but got ${res.rank}")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.shape(0)}, ${input.shape(1)}, ${input.shape(2)}, ${input.shape(3)}));
          |
          |cudnnFilterDescriptor_t filt_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
          |CUDNN_CALL(cudnnSetFilter4dDescriptor(
          |    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          |    ${filter.shape(0)}, ${filter.shape(1)}, ${filter.shape(2)}, ${filter.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${res.shape(0)}, ${res.shape(1)}, ${res.shape(2)}, ${res.shape(3)}));
          |
          |cudnnConvolutionDescriptor_t conv_desc;
          |CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
          |CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          |    conv_desc,
          |    ${padding._1}, ${padding._2}, ${strides._1}, ${strides._2}, ${dilations._1}, ${dilations._2},
          |    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
          |
          |// Algorithm.
          |cudnnConvolutionFwdAlgo_t algo;
          |CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
          |    cudnnHandle,
          |    in_desc, filt_desc, conv_desc, out_desc,
          |    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          |    cudnnHandle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
          |void *ws_data = myGpuMalloc(ws_size);
          |""".stripMargin) ++
        Seq(
          "// Execute convolution.\n" +
          "CUDNN_CALL(cudnnConvolutionForward(\n" +
          "    cudnnHandle,\n" +
          "    ", one, ", in_desc, ", input.data, ", filt_desc, ", filter.data, ",\n" +
          "    conv_desc, algo, ws_data, ws_size,\n" +
          "    ", zero, ", out_desc, ", res.data, "));\n" +
          "}"): _*
      )
    }

    override def plusBias(main: Tensor, bias: Tensor): Tensor = {
      // use cudnnAddTensor (bias is the first parameter, main tensor is the second parameter, addition is in-place on main tensor)
      cudnnAddBiasTensor(bias, main)
      main
    }

    override def plusBias_grad(main: TensorR, bias: TensorR): Unit = {
      // use cudnnConvolutionBackwardBias (or TensorReduce) for bias.d += main.d
      cudnnConvolutionBackwardBias(bias.d, main.d)
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardBias
    // This is effectively the gradient of `cudnnAddBiasTensor`.
    def cudnnConvolutionBackwardBias(biasGrad: Tensor, resGrad: Tensor): Unit = {
      val biasShape: Seq[Int] =
        if (biasGrad.rank == 1) Seq(1, biasGrad.shape(0), 1, 1)
        else if (biasGrad.rank == 4) biasGrad.shape
        else { assert(false, s"Bias gradient is neither rank 1 or rank 4, got ${biasGrad.shape}"); ???}
      assert(resGrad.rank >= 2, "Convolution result gradient must have rank no less than 2")
      if (biasGrad.rank == 1) assert(resGrad.shape(1) == biasGrad.shape(0), "Convolution result gradient shape(1) must equal to Bias gradient shape(0)")
      val resGradShape = resGrad.shape.padTo(4, 1)
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t grad_bias_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${biasShape(0)}, ${biasShape(1)}, ${biasShape(2)}, ${biasShape(3)}));
          |
          |cudnnTensorDescriptor_t grad_out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${resGradShape(0)}, ${resGradShape(1)}, ${resGradShape(2)}, ${resGradShape(3)}));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardBias(\n" +
          "    cudnnHandle, ", one, ", grad_out_desc, ", resGrad.data, ",\n",
          "    ", one, ", grad_bias_desc, ", biasGrad.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
    def cudnnConvolutionBackwardData(inputGrad: Tensor, filter: Tensor, resGrad: Tensor,
                                     padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      assert(resGrad.rank == 4, s"Convolution result gradient must have rank 4, but got ${resGrad.rank}")
      assert(inputGrad.rank == 4, s"Convolution input gradient must have rank 4, but got ${inputGrad.rank}")
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnFilterDescriptor_t filt_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
          |CUDNN_CALL(cudnnSetFilter4dDescriptor(
          |    filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          |    ${filter.shape(0)}, ${filter.shape(1)}, ${filter.shape(2)}, ${filter.shape(3)}));
          |
          |cudnnTensorDescriptor_t grad_in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${inputGrad.shape(0)}, ${inputGrad.shape(1)}, ${inputGrad.shape(2)}, ${inputGrad.shape(3)}));
          |
          |cudnnTensorDescriptor_t grad_out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${resGrad.shape(0)}, ${resGrad.shape(1)}, ${resGrad.shape(2)}, ${resGrad.shape(3)}));
          |
          |cudnnConvolutionDescriptor_t conv_desc;
          |CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
          |CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          |    conv_desc,
          |    ${padding._1}, ${padding._2}, ${strides._1}, ${strides._2}, ${dilations._1}, ${dilations._2},
          |    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
          |
          |// Algorithm.
          |cudnnConvolutionBwdDataAlgo_t algo;
          |CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
          |    cudnnHandle,
          |    filt_desc, grad_out_desc, conv_desc, grad_in_desc,
          |    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
          |    cudnnHandle, filt_desc, grad_out_desc, conv_desc, grad_in_desc, algo, &ws_size));
          |void *ws_data = myGpuMalloc(ws_size);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardData(\n" +
          "    cudnnHandle,\n" +
          "    ", one, ", filt_desc, ", filter.data, ", grad_out_desc, ", resGrad.data, ",\n" +
          "    conv_desc, algo, ws_data, ws_size,\n" +
          "    ", one, ", grad_in_desc, ", inputGrad.data, "));\n" +
          "}"): _*
      )
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardFilter
    def cudnnConvolutionBackwardFilter(filterGrad: Tensor, input: Tensor, resGrad: Tensor,
                                       padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      assert(resGrad.rank == 4, "Convolution result gradient must have rank 4")
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnFilterDescriptor_t grad_filt_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
          |CUDNN_CALL(cudnnSetFilter4dDescriptor(
          |    grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          |    ${filterGrad.shape(0)}, ${filterGrad.shape(1)}, ${filterGrad.shape(2)}, ${filterGrad.shape(3)}));
          |
          |cudnnTensorDescriptor_t grad_out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${resGrad.shape(0)}, ${resGrad.shape(1)}, ${resGrad.shape(2)}, ${resGrad.shape(3)}));
          |
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.shape(0)}, ${input.shape(1)}, ${input.shape(2)}, ${input.shape(3)}));
          |
          |cudnnConvolutionDescriptor_t conv_desc;
          |CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
          |CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          |    conv_desc,
          |    ${padding._1}, ${padding._2}, ${strides._1}, ${strides._2}, ${dilations._1}, ${dilations._2},
          |    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
          |
          |// Algorithm.
          |cudnnConvolutionBwdFilterAlgo_t algo;
          |CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
          |    cudnnHandle,
          |    in_desc, grad_out_desc, conv_desc, grad_filt_desc,
          |    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          |    cudnnHandle, in_desc, grad_out_desc, conv_desc, grad_filt_desc, algo, &ws_size));
          |void *ws_data = myGpuMalloc(ws_size);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnConvolutionBackwardFilter(\n" +
          "    cudnnHandle,\n" +
          "    ", one, ", in_desc, ", input.data, ", grad_out_desc, ", resGrad.data, ",\n" +
          "    conv_desc, algo, ws_data, ws_size,\n" +
          "    ", one, ", grad_filt_desc, ", filterGrad.data, "));\n" +
          "}"): _*
      )
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) = {
      // TODO: Dedupe assertions/shape calculations with CPU implementation.
      assert(input.rank == 4, "Input must be 4-D (first dimension is batch size)")
      assert(kernel.rank == 4, "Kernel must be 4-D")
      bias match {
        case Some(bias) =>
          assert(bias.rank == 1, s"Bias should be 1-D, got ${bias.shape}")
          assert(bias.shape(0) == kernel.shape(0), "Bias length must equal number of out-channels")
        case None => ()
      }

      assert(strides.size == 2, "Strides should have length 2: [strideRow, strideColumn]")
      assert(pads.size == 4, "Pads should have length 4: [padTop, padBottom, padLeft, padRight]")
      val ((strideRow:Int) :: (strideCol:Int) :: Nil) = strides.take(2).toList
      val ((padUp:Int) :: (padDown:Int) :: (padLeft:Int) :: (padRight:Int) :: Nil) = pads.take(4).toList
      assert(strideRow >= 1, "Row stride must be at least 1")
      assert(strideCol >= 1, "Column stride must be at least 1")
      assert(padUp == padDown && padUp == padLeft && padUp == padRight, "All paddings must be equal (for now)")

      assert(kernel.shape(1) == input.shape(1), s"In-channel count mismatch: input.shape[1] ${input.shape(1)} should match kernel.shape[1] ${kernel.shape(1)}")
      assert(input.shape(2) + 2 * padUp >= kernel.shape(2) && input.shape(3) + 2 * padUp >= kernel.shape(3),
        s"Image too small for conv: input is ${input.shape}, kernel is ${kernel.shape}")

      // Execute `cudnnConvolutionForward`.
      val resWidth = convSize(input.shape(2) + padLeft + padRight, kernel.shape(2), strideRow)
      val resHeight = convSize(input.shape(3) + padUp + padDown, kernel.shape(3), strideCol)
      val resShape = Seq(input.shape(0), kernel.shape(0), resWidth, resHeight)
      val resData = mallocArray[Float](resShape.product)
      val res = Tensor(resData, resShape: _*)
      cudnnConvolutionForward(input, kernel, res, padding = (padUp, padLeft), strides = (strideCol, strideRow), dilations = (1, 1))

      // If bias is defined, execute `cudnnAddBiasTensor`.
      bias match {
        case None =>
        case Some(bias) => cudnnAddBiasTensor(bias, res)
      }
      (res, None)
    }

    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      assert(input.x.rank == 4, s"convolution input values should be 4D, but got ${input.x.rank}")
      assert(input.isInput || input.d.rank == 4, s"convolution input gradients is either ignored (for training data) or should be 4D, but got ${input.d.rank}")
      if (!input.isInput) cudnnConvolutionBackwardData(input.d, filter.x, res.d, padding, strides, dilations)
      cudnnConvolutionBackwardFilter(filter.d, input.x, res.d, padding, strides, dilations)
      bias match {
        case None =>
        case Some(bias) =>
          cudnnConvolutionBackwardBias(bias.d, res.d)
      }
      // input.x.toCPU().printHead(10, "forward, input of conv")
      // input.d.toCPU().printHead(10, "backprop, input of conv")
      // filter.x.toCPU().printHead(10, "forward, filter of conv")
      // filter.d.toCPU().printHead(10, "backprop, filter of conv")
      // res.x.toCPU().printHead(10, "forward, res of conv")
      // res.d.toCPU().printHead(10, "backprop, res of conv")
      // error("")
    }

    object PoolModes extends Enumeration {
      val Max = Value("CUDNN_POOLING_MAX")
      val AverageIP = Value("CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING")
      val AverageEP = Value("CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING")
      val MaxD = Value("CUDNN_POOLING_MAX_DETERMINISTIC")
    }

    object NanOpt extends Enumeration {
      val NotProp = Value("CUDNN_NOT_PROPAGATE_NAN")
      val Prop = Value("CUDNN_PROPAGATE_NAN")
    }

    def Pool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]], mode: PoolModes.Value, nanOpt: NanOpt.Value): Tensor = {
      val (windowHeight :: windowWidth :: Nil) = kernel.take(2).toList
      val (verticalPadding, horizontalPadding) = pads match {
        case None => (0, 0)
        case Some(pads) => (pads(0), pads(2))
      }
      val (verticalStride :: horizontalStride :: Nil) = strides.take(2).toList
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      val (outputHeight, outputWidth) = pads match {
        case None => (convSize(input.shape(2), kernel(0), strides(0)), convSize(input.shape(3), kernel(1), strides(1)))
        case Some(pads) => (convSize(input.shape(2), kernel(0), strides(0), pads(0)), convSize(input.shape(3), kernel(1), strides(1), pads(2)))
      }
      val output = Tensor.zeros(input.shape(0), input.shape(1), outputHeight, outputWidth)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.shape(0)}, ${input.shape(1)}, ${input.shape(2)}, ${input.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${output.shape(0)}, ${output.shape(1)}, ${output.shape(2)}, ${output.shape(3)}));
          |
          |cudnnPoolingDescriptor_t poolingDesc;
          |CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
          |CUDNN_CALL(cudnnSetPooling2dDescriptor(
          |    poolingDesc, ${mode.toString}, ${nanOpt.toString},
          |    ${windowHeight}, ${windowWidth}, ${verticalPadding},
          |    ${horizontalPadding}, ${verticalStride}, ${horizontalStride}
          |));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnPoolingForward(\n" +
          "    cudnnHandle, \n" +
          "    poolingDesc, \n" +
          "    ", one, ", in_desc, ", input.data, ", ", zero, ", out_desc, ", output.data, "));\n" +
          "}"): _*)
      output
    }

    override def maxPool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]]): (Tensor, Option[Rep[Array[Int]]]) = {
      assert(input.rank == 4, "Currently, maxpool2D only supports inputs of rank 4")
      (Pool2D_batch(input, kernel, strides, pads, PoolModes.Max, NanOpt.NotProp), None)
    }

    def Pool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int], mode: PoolModes.Value, nanOpt: NanOpt.Value): Unit = {
      val (windowHeight :: windowWidth :: Nil) = kernel.take(2).toList
      val (verticalPadding, horizontalPadding) = (pads(0), pads(2))
      val (verticalStride :: horizontalStride :: Nil) = strides.take(2).toList
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.x.shape(0)}, ${input.x.shape(1)}, ${input.x.shape(2)}, ${input.x.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${output.x.shape(0)}, ${output.x.shape(1)}, ${output.x.shape(2)}, ${output.x.shape(3)}));
          |
          |cudnnPoolingDescriptor_t poolingDesc;
          |CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
          |CUDNN_CALL(cudnnSetPooling2dDescriptor(
          |    poolingDesc, ${mode.toString}, ${nanOpt.toString},
          |    ${windowHeight}, ${windowWidth}, ${verticalPadding},
          |    ${horizontalPadding}, ${verticalStride}, ${horizontalStride}
          |));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnPoolingBackward(\n" +
          "    cudnnHandle, \n" +
          "    poolingDesc, \n" +
          "    ", one, ", out_desc, ", output.x.data, ", out_desc, ", output.d.data, ", in_desc, ", input.x.data,
          "  , ", zero, ", in_desc, ", input.d.data, "));\n" +
          "}"): _*)
    }

    override def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]], kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      Pool2D_batch_grad(input, output, kernel, strides, pads, PoolModes.Max, NanOpt.NotProp)
    }

    override def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor = {
      assert(input.rank == 4, "Current, averagePool2D_batch only supports inputs of rank 4")
      Pool2D_batch(input, kernel, strides, Some(pads), PoolModes.AverageEP, NanOpt.NotProp)
    }

    override def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      Pool2D_batch_grad(input, output, kernel, strides, pads, PoolModes.AverageEP, NanOpt.NotProp)
    }

    def cudnnBatchNormalizationForwardInference(x: Tensor, res: Tensor, scale: Tensor, bias: Tensor,
                                                runningMean: Tensor, runningVar: Tensor,
                                                momentum: Double = 1.0, epsilon: Double = 1e-5): Unit = {
      val biasShape = if (bias.rank == 1) Seq(1, bias.shape(0), 1, 1) else {???}
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${x.shape(0)}, ${x.shape(1)}, ${x.shape(2)}, ${x.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${res.shape(0)}, ${res.shape(1)}, ${res.shape(2)}, ${res.shape(3)}));
          |
          |cudnnTensorDescriptor_t sbmv_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${biasShape(0)}, ${biasShape(1)}, ${biasShape(2)}, ${biasShape(3)}));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationForwardInference(\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", one, ", in_desc, ", x.data, ", out_desc, ", res.data, ", sbmv_desc, ", scale.data, ",\n" +
          "    ", bias.data, ", ", runningMean.data, ", ", runningVar.data, ", ", epsilon, "));\n" +
          "}"): _*)
    }

    // TODO (Fei Wang): What is proper value for momentum (or should be called exponentialAverageFactor) here?
    def cudnnBatchNormalizationForwardTraining(x: Tensor, res: Tensor, scale: Tensor, bias: Tensor,
                                               runningMean: Tensor, runningVar: Tensor, saveMean: Tensor, saveInvVariance: Tensor,
                                               momentum: Double = 0.1, epsilon: Double = 1e-5): Unit = {
      val biasShape = if (bias.rank == 1) Seq(1, bias.shape(0), 1, 1) else {???}
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${x.shape(0)}, ${x.shape(1)}, ${x.shape(2)}, ${x.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${res.shape(0)}, ${res.shape(1)}, ${res.shape(2)}, ${res.shape(3)}));
          |
          |cudnnTensorDescriptor_t sbmv_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${biasShape(0)}, ${biasShape(1)}, ${biasShape(2)}, ${biasShape(3)}));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationForwardTraining(\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", zero, ", in_desc, ", x.data, ", out_desc, ", res.data, ", sbmv_desc, ", scale.data, ",\n" +
          "    ", bias.data, ", ", momentum, ", ", runningMean.data, ", ", runningVar.data, ", ", epsilon, ",\n" +
          "    ", saveMean.data, ", ", saveInvVariance.data, "));\n" +
          "}"): _*)
    }

    def cudnnBatchNormalizationBackward(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR,
                                        saveMean: Tensor, saveInvVariance: Tensor,
                                        momentum: Double = 1.0, epsilon: Double = 1e-5): Unit = {
      val biasShape = if (bias.x.rank == 1) Seq(1, bias.x.shape(0), 1, 1) else {???}
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.x.shape(0)}, ${input.x.shape(1)}, ${input.x.shape(2)}, ${input.x.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${res.x.shape(0)}, ${res.x.shape(1)}, ${res.x.shape(2)}, ${res.x.shape(3)}));
          |
          |cudnnTensorDescriptor_t sbmv_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&sbmv_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    sbmv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${biasShape(0)}, ${biasShape(1)}, ${biasShape(2)}, ${biasShape(3)}));
          |
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnBatchNormalizationBackward(\n" +
          // "    cudnnHandle, CUDNN_BATCHNORM_PER_ACTIVATION,\n" +
          "    cudnnHandle, CUDNN_BATCHNORM_SPATIAL,\n" +
          "    ", one, ", ", one, ", ", one, ", ", one, ", in_desc, ", input.x.data, ",\n" +
          "    out_desc, ", res.d.data, ", in_desc, ", input.d.data, ", sbmv_desc, ", scale.x.data, ",\n" +
          "    ", scale.d.data, ",", bias.d.data, ", ", epsilon, ", ", saveMean.data, ", ", saveInvVariance.data, "));\n" +
          "}"): _*)
    }

    override def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = {
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      cudnnBatchNormalizationForwardInference(x, res, scale, bias, runningMean, runningVar)
      res
    }

    override def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor]) = {
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      val saveMean = Tensor(mallocArray[Float](bias.scalarCount), bias.shape: _*)
      val saveInvVariance = Tensor(mallocArray[Float](bias.scalarCount), bias.shape: _*)
      cudnnBatchNormalizationForwardTraining(x, res, scale, bias, runningMean, runningVar, saveMean, saveInvVariance)
      (res, Some(saveMean), Some(saveInvVariance))
    }

    override def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR,
                                saveMean: Option[Tensor], saveInvVariance: Option[Tensor]): Unit = {
      (saveMean, saveInvVariance) match {
        case (Some(saveMean), Some(saveInvVariance)) => cudnnBatchNormalizationBackward(input, res, scale, bias, saveMean, saveInvVariance)
        case _ => ???
      }
    }

    override def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int]) = {
      val output = Tensor.zeros_like(input)
      val reserveSpace: Rep[Array[Float]] = unchecked[Array[Float]]("(float*)NULL")
      val sizeInBytes: Rep[Int] = unchecked[Int]("0")
      val padShape = input.shape.padTo(4, 1) // pad the dimension to 4D
      unchecked[Unit](
        s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${padShape(0)}, ${padShape(1)}, ${padShape(2)}, ${padShape(3)}));
          |
          |size_t stateSizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetStatesSize(
          |    cudnnHandle, &stateSizeInBytes
          |));
          |void* state = myGpuMalloc(stateSizeInBytes);
          |
          |size_t sizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(
          |    in_desc, &sizeInBytes
          |));
          |void* reserveSpace = myGpuMalloc(sizeInBytes);
          |
          |""".stripMargin,
          reserveSpace, " = (float*)reserveSpace;\n",
          sizeInBytes, " = (int)sizeInBytes;\n",
        s"""
          |cudnnDropoutDescriptor_t dropoutDesc;
          |CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
          |CUDNN_CALL(cudnnSetDropoutDescriptor(
          |    dropoutDesc, cudnnHandle, ${prob}, state, stateSizeInBytes, time(NULL)
          |));
          |""".stripMargin,

          "CUDNN_CALL(cudnnDropoutForward(\n" +
          "    cudnnHandle,\n" +
          "    dropoutDesc,\n" +
          "    in_desc, ", input.data, ", in_desc, ", output.data, ", ", "reserveSpace, sizeInBytes));\n" +
          "}")
      (output, reserveSpace, sizeInBytes)
    }

    override def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit = {
      val padShape = input.x.shape.padTo(4, 1) // pad the dimension to 4D
      unchecked[Unit](
        s"""
          |{
          |cudnnTensorDescriptor_t in_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${padShape(0)}, ${padShape(1)}, ${padShape(2)}, ${padShape(3)}));
          |
          |size_t stateSizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetStatesSize(
          |    cudnnHandle, &stateSizeInBytes
          |));
          |void* state = myGpuMalloc(stateSizeInBytes);
          |
          |size_t sizeInBytes;
          |CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(
          |    in_desc, &sizeInBytes
          |));
          |void* reserveSpace = myGpuMalloc(sizeInBytes);
          |
          |cudnnDropoutDescriptor_t dropoutDesc;
          |CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
          |CUDNN_CALL(cudnnSetDropoutDescriptor(
          |    dropoutDesc, cudnnHandle, ${prob}, state, stateSizeInBytes, time(NULL)
          |));
          |""".stripMargin,
          "CUDNN_CALL(cudnnDropoutBackward(\n" +
          "    cudnnHandle,\n" +
          "    dropoutDesc,\n" +
          "    in_desc, ", output.d.data, ", in_desc, ", input.d.data, ", (void*)", helper, ", (size_t)", size, "));\n" +
          "}")
    }

    object Activation extends Enumeration {
      val Sigmoid = Value("CUDNN_ACTIVATION_SIGMOID")
      val Relu = Value("CUDNN_ACTIVATION_RELU")
      val Tanh = Value("CUDNN_ACTIVATION_TANH")
      val ClippedRelu = Value("CUDNN_ACTIVATION_CLIPPED_RELU")
      val Elu = Value("CUDNN_ACTIVATION_ELU")
    }

    def cudnnActivationForward(x: Tensor, activation: Activation.Value): Tensor = {
      assert(x.rank == 4, "Currently, activation functions only support tensors of rank 4")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${x.shape(0)}, ${x.shape(1)}, ${x.shape(2)}, ${x.shape(3)}));
          |
          |cudnnActivationDescriptor_t act_desc;
          |CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
          |CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
          |                                        /*mode=*/ ${activation.toString},
          |                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
          |                                        /*relu_coef=*/ 0));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnActivationForward(\n" +
          "    cudnnHandle, act_desc,\n" +
          "    ", one, ", x_desc, ", x.data, ", ", zero, ", x_desc, ", res.data, "));\n" +
          "}"): _*
      )
      res
    }

    def cudnnActivationBackward(input: TensorR, res: TensorR, activation: Activation.Value): Unit = {
      assert(input.x.rank == 4, "Currently, activation functions only support tensors of rank 4")
      assert(input.x.shape == res.x.shape,
        "Currently, input and result shapes must be equal: ${input.x.shape}, ${res.x.shape}")
      assert(input.d.shape == res.d.shape,
        s"Currently, input gradient and result gradient shapes must be equal: ${input.d.shape}, ${res.d.shape}")
      val one = NewArray[Float](1); one(0) = 1
      val inputGrad = Tensor(mallocArray[Float](input.d.scalarCount), input.d.shape: _*)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.x.shape(0)}, ${input.x.shape(1)}, ${input.x.shape(2)}, ${input.x.shape(3)}));
          |
          |cudnnActivationDescriptor_t act_desc;
          |CUDNN_CALL(cudnnCreateActivationDescriptor(&act_desc));
          |CUDNN_CALL(cudnnSetActivationDescriptor(act_desc,
          |                                        /*mode=*/ ${activation.toString},
          |                                        /*reluNanOpt=*/ CUDNN_PROPAGATE_NAN,
          |                                        /*relu_coef=*/ 0));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnActivationBackward(\n" +
          "    cudnnHandle, act_desc,\n" +
          "    ", one, ", x_desc, ", res.x.data, ", x_desc, ", res.d.data, ", x_desc, ", input.x.data, ",\n",
          "    ", one, ", x_desc, ", input.d.data, "));\n" +
          "}"): _*
      )
    }

    override def relu(x: Tensor): Tensor = {
      cudnnActivationForward(x, Activation.Relu)
    }
    override def relu_grad(input: TensorR, res: TensorR): Unit = {
      cudnnActivationBackward(input, res, Activation.Relu)
    }

    override def tanh(x: Tensor): Tensor = {
      cudnnActivationForward(x, Activation.Tanh)
    }
    override def tanh_grad(input: TensorR, res: TensorR): Unit = {
      cudnnActivationBackward(input, res, Activation.Tanh)
    }

    override def sigmoid(x: Tensor): Tensor = {
      cudnnActivationForward(x, Activation.Sigmoid)
    }
    override def sigmoid_grad(input: TensorR, res: TensorR): Unit = {
      cudnnActivationBackward(input, res, Activation.Sigmoid)
    }

    // override def exp(x: Tensor) = ???
    // override def exp_grad(x: TensorR, y: TensorR): Unit = ???

    // override def log(x: Tensor) = ???
    // override def log_grad(x: TensorR, y: TensorR): Unit = ???

    // override def sqrt(x: Tensor) = ???
    // override def sqrt_grad(x: TensorR, y: TensorR): Unit = ???

    // override def square(x: Tensor) = ???
    // override def square_grad(x: TensorR, y: TensorR): Unit = ???

    object SoftmaxMode extends Enumeration {
      val Fast = Value("CUDNN_SOFTMAX_FAST")
      val Accurate = Value("CUDNN_SOFTMAX_ACCURATE")
      val Log = Value("CUDNN_SOFTMAX_LOG")
    }

    def cudnnSoftmaxForward(x: Tensor, mode: SoftmaxMode.Value): Tensor = {
      assert(x.rank == 4, "Currently, softmax functions only support tensors of rank 4")
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      val res = Tensor(mallocArray[Float](x.scalarCount), x.shape: _*)
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${x.shape(0)}, ${x.shape(1)}, ${x.shape(2)}, ${x.shape(3)}));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnSoftmaxForward(\n" +
          s"    cudnnHandle, ${mode.toString}, CUDNN_SOFTMAX_MODE_CHANNEL,\n" +
          "    ", one, ", x_desc, ", x.data, ", ", zero, ", x_desc, ", res.data, "));\n" +
          "}"): _*
      )
      res
    }

    def cudnnSoftmaxBackward(input: TensorR, res: TensorR, mode: SoftmaxMode.Value): Unit = {
      assert(input.x.rank == 4, "Currently, softmax only support tensors of rank 4")
      // NOTE: shape assertions are relaxed.
      // Assume that {input/result * forward/backward} values all have the same shape.
      // The shape of the input forward value is used in the generated code.
      /*
      assert(input.x.shape == res.x.shape,
        s"Currently, input and result shapes must be equal: ${input.x.shape}, ${res.x.shape}")
      assert(input.d.shape == res.d.shape,
        s"Currently, input gradient and result gradient shapes must be equal: ${input.d.shape}, ${res.d.shape}")
      */
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${input.x.shape(0)}, ${input.x.shape(1)}, ${input.x.shape(2)}, ${input.x.shape(3)}));
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnSoftmaxBackward(\n" +
          s"    cudnnHandle, ${mode.toString}, CUDNN_SOFTMAX_MODE_CHANNEL,\n" +
          "    ", one, ", x_desc, ", res.x.data, ", x_desc, ", res.d.data, ",\n" +
          "    ", one, ", x_desc, ", input.d.data, "));\n" +
          "}"): _*
      )
    }

    // NOTE: The `input.rank == 2` assertion currently matches the CPU implementation.
    // Conceptually, softmax should support arbitrary rank tensors and an arbitrary reduction axis.
    // This would require some shape padding/transformation.
    def softmaxHelper(x: Tensor, mode: SoftmaxMode.Value): Tensor = {
      assert(x.rank == 2, "Softmax input must be 2-D: [batchSize, logits]")
      val tmpIn = x.reshape(x.shape(0), x.shape(1), 1, 1)
      val tmpOut = cudnnSoftmaxForward(tmpIn, mode)
      val res = tmpOut.reshape(x.shape: _*)
      res
    }

    override def softmax(x: Tensor): Tensor = softmaxHelper(x, SoftmaxMode.Accurate)
    override def logSoftmax(x: Tensor): Tensor = softmaxHelper(x, SoftmaxMode.Log)

    // TODO: Relax rank assertions, see `softmaxHelper` above.
    def softmaxBackwardHelper(input: TensorR, res: TensorR, mode: SoftmaxMode.Value): Unit = {
      assert(input.x.rank == 2, "Softmax input must be 2-D: [batchSize, logits]")
      val tmpInX = input.x.reshape(input.x.shape(0), input.x.shape(1), 1, 1)
      val tmpIn = new TensorR(tmpInX, input.d)
      cudnnSoftmaxBackward(tmpIn, res, mode)
    }

    override def softmax_grad(input: TensorR, res: TensorR): Unit =
      softmaxBackwardHelper(input, res, SoftmaxMode.Accurate)
    override def logSoftmax_grad(input: TensorR, res: TensorR): Unit = {
      softmaxBackwardHelper(input, res, SoftmaxMode.Log)
    }

    // Reference: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnReduceTensorOp_t
    object ReductionOp extends Enumeration {
      val Add = Value("CUDNN_REDUCE_TENSOR_ADD")
      val Mul = Value("CUDNN_REDUCE_TENSOR_MUL")
      val Min = Value("CUDNN_REDUCE_TENSOR_MIN")
      val Max = Value("CUDNN_REDUCE_TENSOR_MAX")
      val Avg = Value("CUDNN_REDUCE_TENSOR_AVG")
      // Maximum of absolute values.
      val Amax = Value("CUDNN_REDUCE_TENSOR_AMAX")
      // Addition of absolute values.
      val Norm1 = Value("CUDNN_REDUCE_TENSOR_NORM1")
      // Square root of sum of squares.
      val Norm2 = Value("CUDNN_REDUCE_TENSOR_NORM2")
      // Multiplication, ignoring zero elements.
      val MulNoZeros = Value("CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS")
    }

    // TODO: Relax rank 4 requirement after implementing tensor descriptor helper functions.
    // `cudnnReduceTensor` supports tensors up to dimension 8.
    def cudnnReduceTensor(x: Tensor, op: ReductionOp.Value, indices: Seq[Int], flatten: Boolean = true): Tensor = {
      assert(x.rank == 4, "Currently, reduction only support tensors of rank 4")
      assert(indices.forall(i => x.shape.indices.contains(i)),
        s"Indices out of bounds: $indices, tensor shape is ${x.shape}")
      val unflatShape = x.shape.zipWithIndex.map { case (dim, i) =>
        if (indices.contains(i)) 1 else dim
      }
      val res = Tensor(mallocArray[Float](unflatShape.product), unflatShape: _*)
      val zero = NewArray[Float](1); zero(0) = 0
      val one = NewArray[Float](1); one(0) = 1
      unchecked[Unit](
        Seq(s"""
          |{
          |cudnnTensorDescriptor_t x_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${x.shape(0)}, ${x.shape(1)}, ${x.shape(2)}, ${x.shape(3)}));
          |
          |cudnnTensorDescriptor_t out_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
          |CUDNN_CALL(cudnnSetTensor4dDescriptor(
          |    out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          |    ${res.shape(0)}, ${res.shape(1)}, ${res.shape(2)}, ${res.shape(3)}));
          |
          |cudnnReduceTensorDescriptor_t reduce_desc;
          |CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&reduce_desc));
          |CUDNN_CALL(cudnnSetReduceTensorDescriptor(
          |    reduce_desc, ${op.toString}, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
          |    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
          |
          |void *indices = nullptr; // Don't store indices.
          |
          |// Workspace.
          |size_t ws_size;
          |CUDNN_CALL(cudnnGetReductionWorkspaceSize(
          |    cudnnHandle, reduce_desc, x_desc, out_desc, &ws_size));
          |void *ws_data = myGpuMalloc(ws_size);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnReduceTensor(\n" +
          s"    cudnnHandle, reduce_desc, indices, 0, ws_data, ws_size,\n" +
          "    ", one, ", x_desc, ", x.data, ", ", zero, ", out_desc, ", res.data, "));\n" +
          "}"): _*
      )
      val resShape = x.shape.zipWithIndex.flatMap { case (dim, i) =>
        if (indices.contains(i)) if (flatten) None else Some(1) else Some(dim)
      }

      // TODO: Remove if expression when rank-0 tensor support is fixed.
      if (resShape.isEmpty) Tensor(res.data, 1)
      else Tensor(res.data, resShape: _*)
    }

    override def sum(x: Tensor): Tensor = {
      val xx = x.resize(x.shape.padTo(4, 1): _*)
      val res = cudnnReduceTensor(xx, ReductionOp.Add, xx.shape.indices)
      res.resize(1)
    }

    override def sum_grad(input: TensorR, res: TensorR): Unit = {
      generateRawComment("backprop for sum op")
      cudnnAddBiasTensor(res.d, input.d)
    }

    override def mean(x: Tensor): Tensor = {
      val xx = x.resize(x.shape.padTo(4, 1): _*)
      val res = cudnnReduceTensor(xx, ReductionOp.Avg, xx.shape.indices)
      res.resize(1)
    }

    override def mean_grad(input: TensorR, res: TensorR): Unit = {
      generateRawComment("backprop for mean op")
      cudnnAddBiasTensor(res.d, input.d, scale = 1.0f / input.x.scalarCount)
    }

    object RNNMode extends Enumeration {
      val RnnRelu = Value("CUDNN_RNN_RELU")
      val RnnTanh = Value("CUDNN_RNN_TANH")
      val Lstm = Value("CUDNN_LSTM")
      val Gru = Value("CUDNN_GRU")
    }

    def cudnnRNNForwardInference(x: Tensor, hx: Tensor, w: Tensor,
                                 numLayers: Int, hiddenSize: Int,
                                 dropout: Float = 0f,
                                 mode: RNNMode.Value = RNNMode.RnnRelu,
                                 bidirectional: Boolean = false): Tensor = {
      assert(x.rank == 3, "RNN input should have rank 3: [seqLength x batchSize x inputSize]")
      assert(hx.rank == 3, "RNN hidden state should have rank 3: [numLayers x batchSize x hiddenSize]")
      assert(x.shape(1) == hx.shape(1), "RNN hidden state second dimension should equal input second dimension (batch size)")
      val seqLength = x.shape(0)
      val batchSize = x.shape(1)
      val inputSize = x.shape(2)
      val numDirections = if (bidirectional) 2 else 1

      val resShape = Seq(seqLength, batchSize, hiddenSize * numDirections)
      val res = Tensor(mallocArray[Float](resShape.product), resShape: _*)
      unchecked[Unit](
        Seq(s"""
          |{
          |size_t dropoutStateSize;
          |CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
          |void* dropoutStates = myGpuMalloc(dropoutStateSize);
          |
          |cudnnDropoutDescriptor_t dropout_desc;
          |CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
          |CUDNN_CALL(cudnnSetDropoutDescriptor(
          |    dropout_desc, cudnnHandle, $dropout, dropoutStates, dropoutStateSize, time(NULL)));
          |
          |cudnnRNNDescriptor_t rnn_desc;
          |CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
          |CUDNN_CALL(cudnnSetRNNDescriptor(
          |    cudnnHandle, rnn_desc,
          |    /*hiddenSize*/ $hiddenSize, /*numLayers*/ $numLayers,
          |    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL,
          |    ${mode.toString}, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
          |
          |cudnnTensorDescriptor_t x_descs[$seqLength];
          |for (int i = 0; i < $seqLength; i++) {
          |  cudnnTensorDescriptor_t x_desc;
          |  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |  int dims[] = {$batchSize, $inputSize, 1};
          |  int strides[] = {dims[1] * dims[2], dims[2], 1};
          |  CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |      x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, dims, strides));
          |  x_descs[i] = x_desc;
          |}
          |
          |cudnnTensorDescriptor_t hx_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
          |int hx_dims[] = {${numLayers * numDirections}, $batchSize, $hiddenSize};
          |int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
          |CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));
          |
          |cudnnTensorDescriptor_t cx_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&cx_desc));
          |int cx_dims[] = {${numLayers * numDirections}, $batchSize, $hiddenSize};
          |int cx_strides[] = {cx_dims[1] * cx_dims[2], cx_dims[2], 1};
          |CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |    cx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, cx_dims, cx_strides));
          |
          |size_t paramsSize;
          |CUDNN_CALL(cudnnGetRNNParamsSize(
          |    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
          |assert(paramsSize / sizeof(float) == ${w.scalarCount} && "Expected parameter size mismatch");
          |
          |cudnnFilterDescriptor_t w_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
          |int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
          |CUDNN_CALL(cudnnSetFilterNdDescriptor(
          |    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));
          |
          |cudnnTensorDescriptor_t y_descs[$seqLength];
          |for (int i = 0; i < $seqLength; i++) {
          |  cudnnTensorDescriptor_t y_desc;
          |  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
          |  int dims[] = {$batchSize, ${hiddenSize * numDirections}, 1};
          |  int strides[] = {dims[1] * dims[2], dims[2], 1};
          |  CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |      y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, dims, strides));
          |  y_descs[i] = y_desc;
          |}
          |
          |cudnnTensorDescriptor_t hy_desc = hx_desc;
          |cudnnTensorDescriptor_t cy_desc = cx_desc;
          |
          |size_t workspaceSize;
          |CUDNN_CALL(cudnnGetRNNWorkspaceSize(
          |    cudnnHandle, rnn_desc, $seqLength, x_descs, &workspaceSize));
          |void* workspace = myGpuMalloc(workspaceSize);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnRNNForwardInference(\n" +
          s"    cudnnHandle, rnn_desc, $seqLength, x_descs, ", x.data, ",\n" +
          "    hx_desc,", hx.data, ", cx_desc, NULL, w_desc,", w.data, ", y_descs, ", res.data, ",\n" +
          "    hy_desc, NULL, cy_desc, NULL, workspace, workspaceSize));\n" +
          "}"): _*)
      res
    }

    def cudnnRNNForwardTraining(x: Tensor, hx: Tensor, w: Tensor,
                                numLayers: Int, hiddenSize: Int,
                                dropout: Float = 0f,
                                mode: RNNMode.Value = RNNMode.RnnRelu,
                                bidirectional: Boolean = false): Tensor = {
      assert(x.rank == 3, "RNN input should have rank 3: [seqLength x batchSize x inputSize]")
      assert(hx.rank == 3, "RNN hidden state should have rank 3: [numLayers * numDirections x batchSize x hiddenSize]")
      val seqLength = x.shape(0)
      val batchSize = x.shape(1)
      val inputSize = x.shape(2)
      val numDirections = if (bidirectional) 2 else 1

      val resShape = Seq(seqLength, batchSize, hiddenSize * numDirections)
      val res = Tensor(mallocArray[Float](resShape.product), resShape: _*)
      unchecked[Unit](
        Seq(s"""
          |{
          |size_t dropoutStateSize;
          |CUDNN_CALL(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStateSize));
          |void* dropoutStates = myGpuMalloc(dropoutStateSize);
          |
          |cudnnDropoutDescriptor_t dropout_desc;
          |CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));
          |CUDNN_CALL(cudnnSetDropoutDescriptor(
          |    dropout_desc, cudnnHandle, $dropout, dropoutStates, dropoutStateSize, time(NULL)));
          |
          |cudnnRNNDescriptor_t rnn_desc;
          |CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc));
          |CUDNN_CALL(cudnnSetRNNDescriptor(
          |    cudnnHandle, rnn_desc,
          |    /*hiddenSize*/ $hiddenSize, /*numLayers*/ $numLayers,
          |    dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL,
          |    ${mode.toString}, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
          |
          |cudnnTensorDescriptor_t x_descs[$seqLength];
          |for (int i = 0; i < $seqLength; i++) {
          |  cudnnTensorDescriptor_t x_desc;
          |  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
          |  int dims[] = {$batchSize, $inputSize, 1};
          |  int strides[] = {dims[1] * dims[2], dims[2], 1};
          |  CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |      x_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, dims, strides));
          |  x_descs[i] = x_desc;
          |}
          |
          |// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
          |// The second dimension must match the first dimension of the tensors described in xDesc.
          |// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
          |cudnnTensorDescriptor_t hx_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc));
          |int hx_dims[] = {${numLayers * numDirections}, $batchSize, $hiddenSize};
          |int hx_strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
          |CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |    hx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, hx_dims, hx_strides));
          |
          |// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
          |// The second dimension must match the first dimension of the tensors described in xDesc.
          |// The third dimension must match the hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc.
          |cudnnTensorDescriptor_t cx_desc;
          |CUDNN_CALL(cudnnCreateTensorDescriptor(&cx_desc));
          |int cx_dims[] = {${numLayers * numDirections}, $batchSize, $hiddenSize};
          |int cx_strides[] = {cx_dims[1] * cx_dims[2], cx_dims[2], 1};
          |CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |    cx_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, cx_dims, cx_strides));
          |
          |size_t paramsSize;
          |CUDNN_CALL(cudnnGetRNNParamsSize(
          |    cudnnHandle, rnn_desc, x_descs[0], &paramsSize, CUDNN_DATA_FLOAT));
          |assert(paramsSize / sizeof(float) == ${w.scalarCount} && "Expected parameter size mismatch");
          |
          |cudnnFilterDescriptor_t w_desc;
          |CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
          |int w_dims[] = {int(paramsSize / sizeof(float)), 1, 1};
          |CUDNN_CALL(cudnnSetFilterNdDescriptor(
          |    w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, /*nbDims*/ 3, w_dims));
          |
          |cudnnTensorDescriptor_t y_descs[$seqLength];
          |for (int i = 0; i < $seqLength; i++) {
          |  cudnnTensorDescriptor_t y_desc;
          |  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
          |  int dims[] = {$batchSize, ${hiddenSize * numDirections}, 1};
          |  int strides[] = {dims[1] * dims[2], dims[2], 1};
          |  CUDNN_CALL(cudnnSetTensorNdDescriptor(
          |      y_desc, CUDNN_DATA_FLOAT, /*nbDims*/ 3, dims, strides));
          |  y_descs[i] = y_desc;
          |}
          |
          |cudnnTensorDescriptor_t hy_desc = hx_desc;
          |cudnnTensorDescriptor_t cy_desc = cx_desc;
          |
          |// Reserve space used by `ForwardTraining` function.
          |size_t reserveSize;
          |CUDNN_CALL(cudnnGetRNNTrainingReserveSize(
          |    cudnnHandle, rnn_desc, $seqLength, x_descs, &reserveSize));
          |void* reserveSpace = myGpuMalloc(reserveSize);
          |
          |size_t workspaceSize;
          |CUDNN_CALL(cudnnGetRNNWorkspaceSize(
          |    cudnnHandle, rnn_desc, $seqLength, x_descs, &workspaceSize));
          |void* workspace = myGpuMalloc(workspaceSize);
          |""".stripMargin) ++
        Seq(
          "CUDNN_CALL(cudnnRNNForwardTraining(\n" +
          s"    cudnnHandle, rnn_desc, $seqLength, x_descs, ", x.data, ",\n" +
          "    hx_desc,", hx.data, ", cx_desc, NULL, w_desc, ", w.data, ", y_descs, ", res.data, ",\n" +
          "    hy_desc, NULL, cy_desc, NULL, workspace, workspaceSize, reserveSpace, reserveSize));\n" +
          "}"): _*)
      res
    }
  }

  object BackendCudnn {
    def apply() = new BackendCudnn
  }

  // Define default GPU backend.
  override def BackendGPU: Backend = BackendCudnn()
  backend = BackendGPU
}
