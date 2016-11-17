using MathNet.Numerics.LinearAlgebra;

namespace MLP.Training
{
    public class BackpropagationResult
    {
        public Vector<double>[] Biases { get; set; }

        public Matrix<double>[] Weights { get; set; }

        public Vector<double> Solution { get; set; }

        public Vector<double> Input { get; set; }
    }
}
