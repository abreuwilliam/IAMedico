using System;
using System.Collections.Generic;
using System.Linq; 
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.ML;
using Microsoft.ML.Data; 
using ML_2025.Models;

namespace ML_2025.Pages
{
    public class IndexModel : PageModel
    {
        private readonly PredictionEngine<SymptomData, SymptomPrediction>? _engine;

        public IndexModel(IServiceProvider sp)
        {
            _engine = sp.GetService<PredictionEngine<SymptomData, SymptomPrediction>>();
        }

        [BindProperty] public string? InputText { get; set; }

        public DiagnosisResult? DxResult { get; private set; }
        public string? MlStatusMessage { get; private set; }

        public void OnGet() { }

        public IActionResult OnPost()
        {
            if (string.IsNullOrWhiteSpace(InputText))
                return Page();

            if (_engine is null)
            {
                MlStatusMessage = "Modelo indisponível. Verifique se o 'diagnosis_train.csv' existe e se o 'model.zip' foi treinado/carregado.";
                return Page();
            }

            var pred = _engine.Predict(new SymptomData { Text = InputText });

            if (pred.Score is null || pred.Score.Length == 0)
            {
                MlStatusMessage = "Score vazio. Verifique o pipeline (SdcaMaximumEntropy) e a coluna 'Score'.";
                DxResult = new DiagnosisResult
                {
                    Triage = "consulta_ambulatorial",
                    Predictions = new List<PredictionItem>
                    {
                        new PredictionItem { Label = pred.PredictedLabel, Probability = 0.0 }
                    }
                };
                return Page();
            }

       
            string[] labels;
            try
            {
                var scoreCol = _engine.OutputSchema["Score"];
                VBuffer<ReadOnlyMemory<char>> slotNames = default;
                scoreCol.GetSlotNames(ref slotNames);
                labels = slotNames.DenseValues().Select(v => v.ToString()).ToArray();
            }
            catch
            {
                labels = Array.Empty<string>();
            }

            if (labels.Length != pred.Score.Length)
                labels = Enumerable.Range(0, pred.Score.Length).Select(i => $"classe_{i}").ToArray();

      
            try
            {
                Console.WriteLine("[ML] Labels: " + string.Join(", ", labels));
                Console.WriteLine("[ML] Scores: " + string.Join(", ", pred.Score.Select(s => s.ToString("F4"))));
            }
            catch { /* ignore */ }

            var items = Enumerable.Range(0, pred.Score.Length)
                .Select(i => new PredictionItem { Label = labels[i], Probability = pred.Score[i] })
                .OrderByDescending(p => p.Probability)
                .Take(3)
                .ToList();

            DxResult = new DiagnosisResult
            {
                Triage = "consulta_ambulatorial",
                Predictions = items
            };

            return Page();
        }
    }

    public class DiagnosisResult
    {
        public string? Triage { get; set; }
        public List<PredictionItem> Predictions { get; set; } = new();
    }

    public class PredictionItem
    {
        public string Label { get; set; } = "";
        public double Probability { get; set; }
    }
}
