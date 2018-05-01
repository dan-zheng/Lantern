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

// this is Xilun's attempt of Neural Turing Machine in Lantern.
// Adapted from @loudinthecloud's PyTorch implementation:
// https://github.com/loudinthecloud/pytorch-ntm/tree/master/ntm

class NTMtest extends FunSuite {
	val ntm_test = new DslDriverC[String, Unit] with TensorExp with ScannerLowerExp {
		class NTM {
			/** initialize state of each component **/
			def init: Rep[Unit] = {}

			// returns output and the new state
			// read_mem --> linear layer --> output
			def forward(input: Tensor, prev_state: TensorR): (TensorR, TensorR) = {}
		}

		class Controller () {
			// controller is basically an LSTM cell.
			// the role of controller is to output parameters for read heads and write heads, 
			// and thus compute the output and the loss.
			// the loss will be used to backpropagate and update the hidden/cell state (learned params) of controller. 
			// Question: which tensors (in each component) require gradients?
		}

		class Memory (val N: Int, val M: Int) {
			// initialize addressing parameters
			val bias = Tensor.rand(2.0f/sqrt(N + M), N, M)
			// Memory should be a Tensor or Variable???
			val memory = TensorR(bias.copy(bias))

			def read(w: TensorR): TensorR = w dot TensorR(memory)

			def write(w: TensorR, e: TensorR, a: TensorR): Rep[Unit] = {
				// dim(w) = (1, N), dim(e) = dim(a) = (1, M)
				// cart: vector * vector
				// dot: Matrix * vector or vector * vector

				val weight_t = new TensorR(w.x.trans, w.d.trans)
				// erase and add are differentiable. 
				val erase = (weight_t dot e)
				val add = (weight_t dot a)
				// is it correct to construct TensorR for (1 - w^T dot e)???
				val tmp_mem = memory * (TensorR(Tensor.ones(erase.x)) - erase)
				val new_mem = tmp_mem + add
				// update tensor memory with gradients
				memory.x.copy_data(new_mem.x)
				memory.d.copy_data(new_mem.d)
			}
		}

		class BaseHead (val memory: Memory) {
			// addressing
			// tensor size: 
			// k: M length vector
			// beta, g: scalar
			// s: 3 length vector
			// gamma: scalar
			// e,a: M length vector
			def address(k: TensorR, beta: TensorR, g: TensorR, s: TensorR, gamma: TensorR, w_prev: TensorR): TensorR = {
				// this function calculate the new weightings w.
				// compute cosine similarity. remember to plus a small number (to smoothen).
				// then softmax.
				// why is this necessary to clone tensor K???
				val _k = new TensorR(Tensor.copy(k.x), Tensor.copy(k.d))
				val _beta = beta.relu
				val _g = g.sigmoid
				val _s = s.softmax
				val _gamma = TensorR(gamma.ones(gamma.x)) + gamma.softplus
				// content base addressing
				val wc = (_beta dot cos_similarity(_k, memory)).softmax
				// interpolation
				val wg = (_g dot wc) + ((1 - _g) dot w_prev)
				// @TODO: shift
				val w_blur = ???
				// sharpen
				val w_pow = w_blur.map(x => Math.pow(x, _gamma))
				val w_pow_sum = w_pow.sum
				val w = w_pow.map(x => x / w_pow_sum)

				def cos_similarity(k: TensorR, M: TensorR): TensorR = {}

				w
			}
		}
		class ReadHead (m: Memory) extends BaseHead (m) {
			// params is a list of param. 
			// read function returns the new weightings w and read memory location based on the new w.
			def forward(params: NSeq[TensorR], w_prev: TensorR): (TensorR, TensorR) = {
				// params: k, β, g, s, γ
				val w = address(params(0), params(1), params(2), params(3), params(4), w_prev)
				val read = memory.read(w)
				(read, w)
			}
		}

		class WriteHead (m: Memory) extends BaseHead (m) {
			// write function writes memory and returns the new weightings
			def forward(params: NSeq[TensorR], w_prev: TensorR): TensorR = {
				// params: k, β, g, s, γ, e, a
				val w = address(params(0), params(1), params(2), params(3), params(4), params(5), params(6), w_prev)
				// need a sigmoid layer on e???
				memory.write(w, e.sigmoid, a)
				w
			}
		}

		// main entrance to program.
		@virtualize
    def snippet(a: Rep[String]): Rep[Unit] = {
    	// copy task

    }
	}
}