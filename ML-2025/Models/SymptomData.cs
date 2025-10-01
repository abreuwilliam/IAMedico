using Microsoft.ML.Data;

namespace ML_2025.Models
{
    public class SymptomData
    {
        [LoadColumn(0)]
        public string Label { get; set; } = "";   // rótulo em TEXTO (gripe, asma, ...)

        [LoadColumn(1)]
        public string Text  { get; set; } = "";
    }

    public class SymptomPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; } = "";

        // Probabilidades por classe (mesma ordem do esquema do modelo)
        public float[]? Score { get; set; }
    }
}
