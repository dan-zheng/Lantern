package lantern

import scala.util.continuations._
import scala.util.continuations

import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Seq => NSeq}
import scala.math._

import org.scalatest.FunSuite

import java.io.PrintWriter;
import java.io.File;

// this is Xilun's attempt of Neural Turing Machine.
// Adapted from @loudinthecloud's PyTorch implementation:
// https://github.com/loudinthecloud/pytorch-ntm/tree/master/ntm

class NTMtest extends FunSuite {
	val ntm_test = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {
		class NTM {}
		class Memory {}
		class BaseHead {}
		class ReadHead extends BaseHead {}
		class WriteHead extends BaseHead {}

		// main entrance to program.
		@virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
    	// copy task
    	
    }
	}
}