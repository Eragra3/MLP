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
        [JsonConverter(typeof(MatrixConverter))]
        public Matrix<double> Weights;

        /// <summary>
        /// Vector
        /// dimensions: O x 1
        /// </summary>
        [JsonProperty]
        [JsonConverter(typeof(VectorConverter))]
        public Vector<double> Biases;

        private readonly int _inputsCount;
        private readonly int _neuronsCount;

        private readonly ActivationFunction _activationFunction;
        public ActivationFunction CurrentActivationFunction => _activationFunction;
        private readonly double _normalStDev;

        private IContinuousDistribution _continuousDistribution;
        private IContinuousDistribution ContinuousUniform
        {
            get
            {
                if (_continuousDistribution != null) return _continuousDistribution;

                _continuousDistribution = new Normal(0.0, _normalStDev);

                return _continuousDistribution;
            }
            set { _continuousDistribution = value; }
        }

        [JsonConstructor]
        private Layer()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputsCount">Number of inputs to each vector</param>
        /// <param name="outputsCount">Same as neuron count in this layer</param>
        /// <param name="isOutputLayer">Is it last nn layer</param>
        public Layer(int inputsCount, int outputsCount, ActivationFunction activationFunction, double normalStDev)
        {
            _activationFunction = activationFunction;
            _inputsCount = inputsCount;
            _neuronsCount = outputsCount;
            _normalStDev = normalStDev;

            Weights = GetNewWeightsMatrix(false);
            Biases = GetNewBiasesVector(false);
        }

        public Matrix GetNewWeightsMatrix(bool allZeroes)
        {
            if (allZeroes) return new DenseMatrix(_neuronsCount, _inputsCount);
            return DenseMatrix.CreateRandom(_neuronsCount, _inputsCount, ContinuousUniform);
        }

        public Vector GetNewBiasesVector(bool allZeroes)
        {
            if (allZeroes) return new DenseVector(_neuronsCount);
            return DenseVector.CreateRandom(_neuronsCount, ContinuousUniform);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs">Vector dimensions: I x 1</param>
        /// <returns></returns>
        public Vector<double> Feedforward(Vector<double> inputs)
        {
            var output = GetNet(inputs);

            return GetActivation(output);
        }

        public Vector<double> GetNet(Vector<double> inputs)
        {
            //(O x I * I x 1) o 0 x 1 = O x 1
            return Weights * inputs + Biases;
        }

        /// <summary>
        /// Takes net as input (this layer output before activation function)
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        public Vector<double> GetActivation(Vector<double> output)
        {
            return ActivationFunction(output);
        }

        public Vector<double> ActivationFunction(Vector<double> output)
        {
            switch (_activationFunction)
            {
                case MLP.ActivationFunction.Sigmoid:
                    return HelperFunctions.Sigmoid(output);
                case MLP.ActivationFunction.Tanh:
                    return HelperFunctions.Tanh(output);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public Vector<double> ActivationFunctionPrime(Vector<double> output)
        {
            switch (_activationFunction)
            {
                case MLP.ActivationFunction.Sigmoid:
                    return HelperFunctions.SigmoidPrime(output);
                case MLP.ActivationFunction.Tanh:
                    return HelperFunctions.TanhPrime(output);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}
