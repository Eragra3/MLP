using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;
using static MLP.HelperFunctions;

namespace MLP
{
    public class Layer
    {
        /// <summary>
        /// Matrix
        /// dimensions: O x I
        /// </summary>
        /// <remarks>
        /// I == N
        /// Input count is same as neuron count
        /// </remarks>
        [JsonProperty]
        private Matrix<double> _weights;

        /// <summary>
        /// Vector
        /// dimensions: I
        /// </summary>
        [JsonProperty]
        private Vector<double> _biases;

        public Layer(int inputsCount, int outputsCount)
        {
            _weights = new DenseMatrix(outputsCount, inputsCount);
            _biases = new DenseVector(inputsCount);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs">Vector dimensions: I</param>
        /// <returns></returns>
        public Vector<double> Feedforward(Vector<double> inputs)
        {
            var I = _weights.ColumnCount;
            var O = _weights.RowCount;

            var output = new DenseVector(O);
            for (int i = 0; i < O; i++)
            {
                var neuronWeights = _weights.Column(i);
                var neuronBias = _biases[i];

                for (int j = 0; j < I; j++)
                {
                    output[i] += inputs[j] * neuronWeights[j] + neuronBias;
                }
            }

            return Sigmoid(output);
        }

        public Vector<double> GetOutput(Vector<double> inputs)
        {
            var I = _weights.ColumnCount;
            var O = _weights.RowCount;

            var output = new DenseVector(O);
            for (int i = 0; i < O; i++)
            {
                var neuronWeights = _weights.Column(i);
                var neuronBias = _biases[i];

                for (int j = 0; j < I; j++)
                {
                    output[i] += inputs[j] * neuronWeights[j] + neuronBias;
                }
            }
            return output;
        }

        public Vector<double> GetActivation(Vector<double> output)
        {
            return Sigmoid(output);
        }
    }
}
