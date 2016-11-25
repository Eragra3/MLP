using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms.DataVisualization.Charting;
using System.Text;
using System.Threading.Tasks;
using MLP.Training;

namespace MLP.Charter
{
    public static class Charter
    {
        public static void GenerateExperimentPlot(
            TrainingResult[] trainingResults,
            string path,
            int min,
            int max)
        {
            IList<DataPoint>[] series = new IList<DataPoint>[trainingResults.Length];
            path += ".png";

            for (int index = 0; index < trainingResults.Length; index++)
            {
                var trainingResult = trainingResults[index];
                series[index] = new List<DataPoint>(trainingResult.Evaluations.Length);
                var oneSeries = series[index];
                for (int epoch = 0; epoch < trainingResult.Evaluations.Length; epoch++)
                {
                    var evaluation = trainingResult.Evaluations[epoch];
                    var dataPoint = new DataPoint(epoch, evaluation.Percentage);
                    oneSeries.Add(dataPoint);
                }
            }

            GeneratePlot(series, path, min, max);
        }

        public static void GeneratePlot(IList<DataPoint>[] seriesArray, string path, int min, int max)
        {
            using (var ch = new Chart())
            {
                ch.ChartAreas.Add(new ChartArea());
                for (int i = 0; i < seriesArray.Length; i++)
                {
                    var series = seriesArray[i];
                    var s = new Series();
                    foreach (var pnt in series) s.Points.Add(pnt);
                    ch.Series.Add(s);
                    ch.Series[i].ChartType = SeriesChartType.Line;
                }
                ch.ChartAreas[0].AxisY.Minimum = min;
                ch.ChartAreas[0].AxisY.Maximum = max;
                ch.ChartAreas[0].AxisY.Interval = 10;
                ch.ChartAreas[0].AxisX.Minimum = 0;

                ch.SaveImage(path, ChartImageFormat.Png);
            }
        }
    }
}
