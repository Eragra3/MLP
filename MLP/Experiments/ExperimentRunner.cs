using System.IO;
using System.Text;
using MLP.Training;

namespace MLP
{

    /// <summary>
    /// Runs experiments on (almost)MNIST data set
    /// </summary>
    public static class ExperimentRunner
    {
        public static void RunLearningRateExperiment(
            double[] learningRates,
            MlpOptions options,
            int repetitions,
            string logPath
            )
        {
            //disable early learning end
            options.ErrorThreshold = 0;

            File.Create(logPath);

            for (int i = 0; i < learningRates.Length; i++)
            {
                var learningRate = learningRates[i];

                var trainingOptions = new MlpOptions(
                    learningRate,
                    options.Momentum,
                    options.ErrorThreshold,
                    options.Sizes,
                    options.TrainingPath,
                    options.ValidationPath,
                    options.TestPath,
                    options.MaxEpochs,
                    options.IsVerbose,
                    options.BatchSize,
                    options.ActivationFunction,
                    options.NormalStDeviation,
                    true
                    );

                var trainingResponses = new TrainingResult[repetitions];

                //gather data
                for (int j = 0; j < repetitions; j++)
                {
                    var mlp = new Mlp(options.ActivationFunction, options.NormalStDeviation, options.Sizes);

                    var trainingResponse = MlpTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;
                }

                //log data
                StringBuilder log = new StringBuilder("sep=|");
                log.AppendLine();
                log.Append("epoch");
                for (int j = 0; j < trainingResponses.Length; j++)
                {
                    log.Append("|evaluation_" + j + "|error_" + j);
                }
                log.AppendLine();
                for (int j = 0; j < trainingResponses[0].Epochs + 1; j++)
                {
                    log.Append(j);
                    for (int n = 0; n < trainingResponses.Length; n++)
                    {
                        var result = trainingResponses[n];
                        log.Append("|" + result.Evaluations[j] + "|" + result.EpochErrors[j]);
                    }
                    log.AppendLine();
                }
            }
        }
    }
}
