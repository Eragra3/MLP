using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class TrainingResult
    {
        public Mlp Mlp { get; set; }

        public int Epochs { get; set; }

        public double[] EpochErrors { get; set; }
    }
}
