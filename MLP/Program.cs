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
        public const string MnistDirectory = "../../Data";

        static void Main(string[] args)
        {
            Command command = Command.Help;
            string nnJsonPath = "";
            bool isVerbose = false;
            string outputPath = "";
            string imagePath = "";
            bool print = false;
            int[] layersSizes = { 70, 15, 10 };
            bool evaluate = false;

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
                    .DefaultParameter("sizes", sizes => layersSizes = JsonConvert.DeserializeObject<int[]>(sizes), "Number of layer and its sizes, default to [70,5,10]", "Sizes")
                    .DefaultParameter("input-path", path => imagePath = path, "Number of layer and its sizes, default to [70,5,10]", "Sizes")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                .Command("view", () => command = Command.View, "Show MNIST imag")
                    .DefaultParameter("path", path => imagePath = path, "Path to image")
                    .Option("p", () => print = true, "Display grayscale interpretation")
                    .Option("print", () => print = true, "Display grayscale interpretation")
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
                        catch (Exception e)
                        {
                            Console.WriteLine(e);
                            Console.WriteLine($"Path is invalid");
                            return;
                        }

                        var mlp = MLPTrainer.TrainOnMnist(MnistDirectory, layersSizes);

                        File.WriteAllText(outputPath, mlp.ToJson());

                        break;
                    }
                case Command.Test:
                    {
                        MLP mlp;
                        try
                        {
                            var json = File.ReadAllText(nnJsonPath);
                            mlp = JsonConvert.DeserializeObject<MLP>(json);
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

                            var decision = mlp.Compute(image.ValuesFloats);

                            Console.WriteLine($"Result - {decision}");
                            Console.WriteLine($"Expected - {image.Label}");
                        }

                        if (evaluate)
                        {
                            var evaluation = MLPTrainer.Evaluate(mlp, MnistDirectory);

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

                        if (print)
                        {
                            var modelMatrix = MnistViewer.Print(model);
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
