using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
        private float[][] _weights;
        /// <summary>
        /// Vector
        /// dimensions: I
        /// </summary>
        [JsonProperty]
        private float[] _biases;

        public Layer(int inputsCount, int outputsCount)
        {
            _weights = MatrixHelper.GetRandomMatrix(outputsCount, inputsCount);
            _biases = MatrixHelper.GetRandomVector(inputsCount);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs">Vector dimensions: I</param>
        /// <returns></returns>
        public float[] Feedforward(float[] inputs)
        {
            var I = _weights[0].Length;
            var O = _weights.Length;

            var output = new float[O];
            for (int i = 0; i < O; i++)
            {
                var neuronWeights = _weights[i];
                var neuronBias = _biases[i];

                for (int j = 0; j < I; j++)
                {
                    output[i] += inputs[j] * neuronWeights[j] + neuronBias;
                }
            }

            return Sigmoid(output);
        }
    }
}
