using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MLP.MnistHelpers;

namespace MLP
{
    public static class MlpTrainer
    {
        public static TrainingStatistics TrainOnMnist(MlpOptions options)
        {
            var isVerbose = options.IsVerbose;

            var mlp = new Mlp(options.LearningRate, options.Momentum, options.ErrorThreshold, options.Sizes);

            var trainingSet = MnistParser.ReadAll(options.TrainingPath);
            var validationSet = MnistParser.ReadAll(options.ValidationPath);

            var trainingModel = new TrainingModel
            {
                MaxEpochs = options.MaxEpochs,
                ErrorThreshold = options.ErrorThreshold,
                ValidationSet = validationSet,
                TrainingSet = trainingSet,
                IsVerbose = isVerbose
            };

            var trainingResult = mlp.Train(trainingModel);

            var statistics = new TrainingStatistics
            {
                TrainingResult = trainingResult
            };

            return statistics;
        }

        public static MlpEvaluationModel Evaluate(Mlp mlp, InputModel[] testData)
        {
            var correctSolutions = 0;

            for (int i = 0; i < testData.Length; i++)
            {
                var model = testData[i];

                var decision = mlp.Compute(model.Values);

                if (decision == model.Label) correctSolutions++;
            }

            var result = new MlpEvaluationModel
            {
                Correct = correctSolutions,
                All = testData.Length
            };

            return result;
        }

        public class MlpEvaluationModel
        {
            public int All { get; set; }
            public int Correct { get; set; }

            public int Incorrect => All - Correct;
            public double Percentage => Math.Round((double)Correct / All, 2);
        }
    }
}
