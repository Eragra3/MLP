using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public static class HelperFunctions
    {
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
