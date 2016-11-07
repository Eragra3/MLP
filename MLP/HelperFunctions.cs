using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace MLP
{
    public static class HelperFunctions
    {
        public static Vector<double> Sigmoid(Vector<double> input)
        {
            Vector<double> result = input.Map(x => 1.0 / (1.0 + Math.Exp(x)));
            return result;
        }

        public static Vector<double> SigmoidPrime(Vector<double> input)
        {
            var sigmoid = Sigmoid(input);

            Vector<double> result = sigmoid.PointwiseMultiply(sigmoid.Map(x => 1 - x));

            return result;
        }
    }
}
