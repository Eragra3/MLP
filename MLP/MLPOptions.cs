namespace MLP
{
    public class MlpOptions
    {
        public double LearningRate { get; set; }

        public double Momentum { get; set; }

        public double ErrorThreshold { get; set; }

        public int MaxEpochs { get; set; }

        public int[] Sizes { get; set; }

        public string TrainingPath { get; set; }

        public string ValidationPath { get; set; }

        public string TestPath { get; set; }

        public bool IsVerbose { get; set; }

        public int BatchSize { get; set; }

        public ActivationFunction ActivationFunction { get; set; }

        public double NormalStDeviation { get; set; }

        public bool EvaluateOnEachEpoch { get; set; }

        public MlpOptions(double learningRate, double momentum, double errorThreshold, int[] sizes, string trainingPath, string validationPath, string testPath, int maxEpochs, bool isVerbose, int batchSize, ActivationFunction activationFunction, double normalStDeviation, bool evaluateOnEachEpoch)
        {
            LearningRate = learningRate;
            Momentum = momentum;
            ErrorThreshold = errorThreshold;
            Sizes = sizes;
            TrainingPath = trainingPath;
            ValidationPath = validationPath;
            TestPath = testPath;
            MaxEpochs = maxEpochs;
            IsVerbose = isVerbose;
            BatchSize = batchSize;
            ActivationFunction = activationFunction;
            NormalStDeviation = normalStDeviation;
            EvaluateOnEachEpoch = evaluateOnEachEpoch;
        }
    }
}
