using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace MLP
{
    public class InputModel
    {
        public Vector<double> Values { get; set; }

        public Vector<double> ExpectedSolution { get; set; }

        public int Label { get; set; }
    }

    public class MnistImage : InputModel
    {
        public int Height { get; set; }
        public int Width { get; set; }
        public string FileName { get; set; }
    }
}
