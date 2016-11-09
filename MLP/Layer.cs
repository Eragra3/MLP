using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MLP.Serialization;
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
        /// O == N
        /// Output count is same as neuron count
        /// </remarks>
        [JsonProperty]
        [JsonConverter(typeof(DenseMatrixConverter))]
        public DenseMatrix _weights;

        /// <summary>
        /// Vector
        /// dimensions: O x 1
        /// </summary>
        [JsonProperty]
        [JsonConverter(typeof(DenseVectorConverter))]
        public DenseVector _biases;

        [JsonConstructor]
        private Layer() { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputsCount">Number of inputs to each vector</param>
        /// <param name="outputsCount">Same as neuron count in this layer</param>
        public Layer(int inputsCount, int outputsCount)
        {
            _weights = DenseMatrix.CreateRandom(outputsCount, inputsCount, new ContinuousUniform());
            _biases = DenseVector.CreateRandom(outputsCount, new ContinuousUniform());
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs">Vector dimensions: I x 1</param>
        /// <returns></returns>
        public Vector<double> Feedforward(Vector<double> inputs)
        {
            var output = GetOutput(inputs);

            return GetActivation(output);
        }

        public Vector<double> GetOutput(Vector<double> inputs)
        {
            //O x I * I x 1 = O x 1
            //O x 1 o O x 1 = O x 1
            var output = _weights.Multiply(inputs).Map2((w, b) => w + b, _biases);

            return output;
        }

        public Vector<double> GetActivation(Vector<double> output)
        {
            return Sigmoid(output);
        }
    }
}
