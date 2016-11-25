using System;
using MLP.MnistHelpers;

namespace MLP.Training
{
    public static class MlpTrainer
    {
        private static MnistImage[] _trainingSet;
        private static string _trainingSetPath;
        private static MnistImage[] _testSet;
        private static string _testSetPath;
        private static MnistImage[] _validationSet;
        private static string _validationSetPath;

        public static TrainingResult TrainOnMnist(MlpOptions options)
        {
            var isVerbose = options.IsVerbose;
            var normalize = options.NormalizeInput;

            var mlp = new Mlp(
                options.ActivationFunction,
                options.NormalStDeviation,
                options.Sizes);
            if (_trainingSetPath != options.TrainingPath || _trainingSet == null)
            {
                _trainingSet = MnistParser.ReadAll(options.TrainingPath, normalize);
                _trainingSetPath = options.TrainingPath;
            }
            var trainingSet = _trainingSet;
            if (_testSetPath != options.TestPath || _testSet == null)
            {
                _testSet = MnistParser.ReadAll(options.TestPath, normalize);
                _testSetPath = options.TestPath;
            }
            var testSet = _testSet;
            if (_validationSetPath != options.ValidationPath || _validationSet == null)
            {
                _validationSet = MnistParser.ReadAll(options.ValidationPath, normalize);
                _validationSetPath = options.ValidationPath;
            }
            var validationSet = _validationSet;

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
