using Microsoft.ML;
using ML_2025.Models;
using ML_2025.Services;
using ML_2025.Helpers;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages();


var baseDir = builder.Environment.ContentRootPath;
var pastaModelos = Path.Combine(baseDir, "MLModels");
Directory.CreateDirectory(pastaModelos);

var csvPath     = Path.Combine(pastaModelos, "diagnosis_train.csv");
var modelPath   = Path.Combine(pastaModelos, "model.zip");
var classesPath = Path.Combine(pastaModelos, "classes.txt");

Console.WriteLine($"[ML] baseDir   = {baseDir}");
Console.WriteLine($"[ML] csvPath   = {csvPath}");
Console.WriteLine($"[ML] modelPath = {modelPath}");
Console.WriteLine($"[ML] classes   = {classesPath}");


if (!File.Exists(modelPath))
{
    var ok = ModelBuilder.Treinar(pastaModelos, csvPath);
    if (!ok) Console.WriteLine("[ML] Não treinou: CSV ausente.");
}


if (File.Exists(modelPath))
{
    var ml = new MLContext();
    var model = ml.Model.Load(modelPath, out _);
    var engine = ml.Model.CreatePredictionEngine<SymptomData, SymptomPrediction>(model);

    builder.Services.AddSingleton(engine);

    if (File.Exists(classesPath))
    {
        var classMap = File.ReadAllLines(classesPath)
                           .Where(l => !string.IsNullOrWhiteSpace(l))
                           .ToArray();
        builder.Services.AddSingleton<string[]>(classMap);
    }
    else
    {
        Console.WriteLine("[ML] WARN: classes.txt não encontrado; probabilidades não serão mapeadas.");
    }
}

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();
app.MapRazorPages();


app.MapPost("/predict", async (PredictRequest req, PredictionEngine<SymptomData, SymptomPrediction> engine, string[]? classMap) =>
{
    if (string.IsNullOrWhiteSpace(req.Text))
        return Results.BadRequest("Texto vazio");

    var input = new SymptomData { Text = req.Text };
    var result = engine.Predict(input);

    string resposta = result.PredictedLabel;

    
    await Logger.SalvarLogAsync(req.Text, resposta);

    return Results.Ok(new
    {
        userInput = req.Text,
        predicted = resposta,
        probabilities = classMap != null && result.Score != null
            ? classMap.Zip(result.Score, (classe, prob) => new { classe, prob })
            : null
    });
});

app.Run();
