namespace MLP.Training
{
    public class TrainingModel
    {
        public InputModel[] TrainingSet { get; set; }

        public InputModel[] TestSet { get; set; }

        public InputModel[] ValidationSet { get; set; }

        public double ErrorThreshold { get; set; }

        public int MaxEpochs { get; set; }

        public bool IsVerbose { get; set; }

        public int BatchSize { get; set; }

        public double LearningRate { get; set; }

        public double Momentum { get; set; }

        public bool EvaluateOnEachEpoch { get; set; }
    }
}