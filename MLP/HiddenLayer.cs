using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
            _weights = AlgebraHelper.GetRandomMatrix(inputsCount, outputsCount);
            _biases = AlgebraHelper.GetRandomVector(inputsCount);
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

        public static float[] Sigmoid(float[] input)
        {
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 1.0f / (1.0f + (float)Math.Exp(-input[i]));
            }

            return input;
        }

        public static float[] SigmoidPrime(float[] input)
        {
            var sig1 = new float[input.Length];

            //sigmoid
            for (int i = 0; i < input.Length; i++)
            {
                sig1[i] = 1.0f / (1.0f + (float)Math.Exp(-input[i]));
            }

            for (int i = 0; i < input.Length; i++)
            {
                input[i] = sig1[i] * (1 - 1.0f / (1.0f + (float)Math.Exp(-input[i])));
            }

            return input;
        }
    }
}
