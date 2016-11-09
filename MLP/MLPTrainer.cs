using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLP.MnistHelpers;

namespace MLP
{
    public static class MLPTrainer
    {
        public static MLP TrainOnMnist(string pathToDirectory, params int[] sizes)
        {
            var mlp = new MLP(sizes);

            return mlp;
        }

        public static MLPEvaluationModel Evaluate(MLP mlp, string pathToDirectory)
        {
            var testData = MnistParser.ReadAll(pathToDirectory);

            var correctSolutions = 0;

            for (int i = 0; i < testData.Length; i++)
            {
                var model = testData[i];

                var decision = mlp.Compute(model.Values);

                if (decision == model.Label) correctSolutions++;
            }

            var result = new MLPEvaluationModel
            {
                Correct = correctSolutions,
                All = testData.Length
            };

            return result;
        }

        public struct MLPEvaluationModel
        {
            public int Correct;
            public int All;
            public int Incorrect => All - Correct;
            public double Percentage => Math.Round((double)Correct / All, 2);
        }
    }
}
