using System;
using MLP.MnistHelpers;

namespace MLP.Training
{
    public static class MlpTrainer
    {
        public static TrainingResult TrainOnMnist(MlpOptions options)
        {
            var isVerbose = options.IsVerbose;

            var mlp = new Mlp(
                options.ActivationFunction,
                options.NormalStDeviation,
                options.Sizes);

            var trainingSet = MnistParser.ReadAll(options.TrainingPath);
            var testSet = MnistParser.ReadAll(options.TestPath);
            var validationSet = MnistParser.ReadAll(options.ValidationPath);

            var trainingModel = new TrainingModel
            {
                MaxEpochs = options.MaxEpochs,
                ErrorThreshold = options.ErrorThreshold,
                ValidationSet = validationSet,
                TrainingSet = trainingSet,
                TestSet = testSet,
                IsVerbose = isVerbose,
                BatchSize = options.BatchSize,
                LearningRate = options.LearningRate,
                Momentum = options.Momentum,
                EvaluateOnEachEpoch = options.EvaluateOnEachEpoch
            };

            var trainingResult = mlp.Train(trainingModel);

            return trainingResult;
        }

        public static EvaluationModel Evaluate(Mlp mlp, InputModel[] testData)
        {
            var correctSolutions = 0;

            for (int i = 0; i < testData.Length; i++)
            {
                var model = testData[i];

                var decision = mlp.Compute(model.Values);

                if (decision == model.Label) correctSolutions++;
            }

            var result = new EvaluationModel
            {
                Correct = correctSolutions,
                All = testData.Length
            };

            return result;
        }

        public class EvaluationModel
        {
            public int All { get; set; }
            public int Correct { get; set; }

            public int Incorrect => All - Correct;
            public double Percentage => Math.Round((double)Correct / All * 100, 2);
        }
    }
}
