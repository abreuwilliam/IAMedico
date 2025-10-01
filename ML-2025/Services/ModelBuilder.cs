using System;
using System.IO;
using Microsoft.ML;
using ML_2025.Models;

namespace ML_2025.Services
{
    public static class ModelBuilder
    {
  
        public static bool Treinar(string pastaModelos, string csvPath)
        {
            if (!File.Exists(csvPath))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[ML] Arquivo de treino não encontrado: {csvPath}");
                Console.ResetColor();
                return false;
            }

            var ml = new MLContext(seed: 42);

            var data = ml.Data.LoadFromTextFile<SymptomData>(
                path: csvPath,
                hasHeader: true,
                separatorChar: ',');

            var split = ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 42);

            var pipeline =
                ml.Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(ml.Transforms.Text.FeaturizeText("Features", "Text"))
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "Label", featureColumnName: "Features"))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(split.TrainSet);

            var metrics = ml.MulticlassClassification.Evaluate(
                model.Transform(split.TestSet), labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"[ML] MicroAcc={metrics.MicroAccuracy:F3}  MacroAcc={metrics.MacroAccuracy:F3}  LogLoss={metrics.LogLoss:F3}");

            Directory.CreateDirectory(pastaModelos);
            var modelPath = Path.Combine(pastaModelos, "model.zip");
            ml.Model.Save(model, split.TrainSet.Schema, modelPath);

            Console.WriteLine($"[ML] Modelo salvo em: {modelPath}");
            return true;
        }
    }
}
