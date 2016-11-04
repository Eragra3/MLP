using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MLP.HelperFunctions;

namespace MLP
{
    public class HiddenLayer
    {
        /// <summary>
        /// Matrix
        /// dimensions: I x O
        /// </summary>
        /// <remarks>
        /// I == N
        /// Input count is same as neuron count
        /// </remarks>
        private float[][] _weights;
        /// <summary>
        /// Vector
        /// dimensions: I
        /// </summary>
        private float[] _biases;

        public HiddenLayer(int inputsCount, int outputsCount)
        {
            _weights = MatrixHelper.GetRandomMatrix(inputsCount, outputsCount);
            _biases = MatrixHelper.GetRandomVector(inputsCount);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs">Vector dimensions: I</param>
        /// <returns></returns>
        public float[] Feedforward(float[] inputs)
        {
            var O = _weights[0].Length;
            var I = _weights.Length;

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
