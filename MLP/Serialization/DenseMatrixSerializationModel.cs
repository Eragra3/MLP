using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Storage;

namespace MLP.Serialization
{
    public class DenseMatrixSerializationModel
    {
        public double[] Storage { get; set; }

        public int ColumnCount { get; set; }

        public int RowCount { get; set; }

        public DenseMatrixSerializationModel() { }
    }
}
