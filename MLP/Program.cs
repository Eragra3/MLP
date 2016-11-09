using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLP.Cmd;
using MLP.MnistHelpers;
using TTRider.FluidCommandLine;
using System.IO;
using Newtonsoft.Json;

namespace MLP
{
    class Program
    {
        private const string DATA_PATH = "../../";
        private const string TEST_DATA_PATH = DATA_PATH + "TestData";
        private const string TRAINING_DATA_PATH = DATA_PATH + "TrainingData";
        private const string VALIDATION_PATH = DATA_PATH + "ValidationData";

        static void Main(string[] args)
        {
            Command command = Command.Help;
            string nnJsonPath = "";
            bool isVerbose = false;
            string outputPath = "";
            string imagePath = "";
            bool print = false;
            bool dump = false;
            bool evaluate = false;

            //mlp params
            int[] layersSizes = { 70, 15, 10 };
            double learningRate = 0.25;
            double momentum = 0.001;
            double errorThreshold = 1;

            int maxEpochs = 100;

            ICommandLine commandLine = CommandLine
                .Help("h")
                .Help("help")
                .Command("test", () => command = Command.Test, true, "Test your MLP")
                    .DefaultParameter("mlp", json => nnJsonPath = json, "MLP data in json format", "Json")
                    .Parameter("image", path => imagePath = path, "Path to image", "Path to image")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                    .Option("e", () => evaluate = true, "Evaluate using MNIST dataset")
                    .Option("evaluate", () => evaluate = true, "Evaluate using MNIST dataset")
                .Command("train", () => command = Command.Train, "Train new MLP")
                    .DefaultParameter("output", path => outputPath = path, "Output file to save trained mlp")
                    .Parameter("sizes", sizes => layersSizes = JsonConvert.DeserializeObject<int[]>(sizes), "Number of layer and its sizes, default to [70,5,10]", "Sizes")
                    .Parameter("learning-rate", val => learningRate = double.Parse(val), "Learning rate")
                    .Parameter("momentum", val => momentum = double.Parse(val), "Momenum parameter")
                    .Parameter("error-threshold", val => errorThreshold = double.Parse(val), "Error threshold to set learning stop criteria")
                    .Parameter("max-epochs", val => maxEpochs = int.Parse(val), "Progra will terminate learning if reaches this epoch")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                .Command("view", () => command = Command.View, "Show MNIST imag")
                    .DefaultParameter("path", path => imagePath = path, "Path to image")
                    .Option("p", () => print = true, "Display grayscale interpretation")
                    .Option("print", () => print = true, "Display grayscale interpretation")
                    .Option("d", () => dump = true, "Dump image data")
                    .Option("dump", () => dump = true, "Dump image data")
                .End();

            commandLine.Run(args);

            switch (command)
            {
                case Command.Train:
                    {
                        try
                        {
                            File.Create(outputPath).Close();
                        }
                        catch (Exception)
                        {
                            Console.WriteLine($"Path is invalid");
                            return;
                        }

                        var options = new MlpOptions(
                            learningRate,
                            momentum,
                            errorThreshold,
                            layersSizes,
                            TRAINING_DATA_PATH,
                            VALIDATION_PATH,
                            TEST_DATA_PATH,
                            maxEpochs,
                            isVerbose
                            );

                        var statistics = MlpTrainer.TrainOnMnist(options);

                        var mlp = statistics.TrainingResult.Mlp;

                        File.WriteAllText(outputPath, mlp.ToJson());

                        break;
                    }
                case Command.Test:
                    {
                        Mlp mlp;
                        try
                        {
                            var json = File.ReadAllText(nnJsonPath);
                            mlp = JsonConvert.DeserializeObject<Mlp>(json);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(e);
                            Console.WriteLine($"Path is invalid");
                            return;
                        }

                        if (!string.IsNullOrEmpty(imagePath))
                        {
                            if (!File.Exists(imagePath))
                            {
                                Console.WriteLine($"File {imagePath} does not exist!");
                                return;
                            }

                            var image = MnistParser.ReadImage(imagePath);

                            var decision = mlp.Compute(image.Values);

                            Console.WriteLine($"Result - {decision}");
                            Console.WriteLine($"Expected - {image.Label}");
                        }

                        if (evaluate)
                        {
                            var evaluation = MlpTrainer.Evaluate(mlp, TEST_DATA_PATH);

                            Console.WriteLine($"Solutions - {evaluation.Correct} / {evaluation.All}");
                            Console.WriteLine($"Fitness - {evaluation.Percentage}");
                        }

                        break;
                    }
                case Command.View:
                    {
                        if (string.IsNullOrEmpty(imagePath))
                        {
                            Console.WriteLine($"Path to image not set");
                            return;
                        }
                        if (!File.Exists(imagePath))
                        {
                            Console.WriteLine($"File {imagePath} does not exist!");
                            return;
                        }

                        var model = MnistParser.ReadImage(imagePath);

                        if (dump)
                        {
                            var modelDump = MnistViewer.Dump(model);
                            Console.Write(modelDump);
                        }

                        if (print)
                        {
                            var modelMatrix = MnistViewer.ToMatrix(model.Values, model.Width);
                            Console.Write(modelMatrix);
                        }
                        break;
                    }
                case Command.Help:
                    commandLine.Run("help");
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}
