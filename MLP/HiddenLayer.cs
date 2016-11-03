using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class HiddenLayer
    {
        private float[][] _weights; //I x O
        private float[] _biases; // I

        private Random _rng = new Random();

        public HiddenLayer(int inputsCount, int outputsCount)
        {
            _weights = AlgebraHelper.GetRandomMatrix(inputsCount, outputsCount);
            _biases = AlgebraHelper.GetRandomVector(inputsCount);
        }
    }
}
