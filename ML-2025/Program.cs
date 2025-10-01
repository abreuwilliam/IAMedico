using Microsoft.ML;
using ML_2025.Models;
using ML_2025.Services;

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

    // >>> registra o mapa de classes (na MESMA ordem do Score)
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

app.Run();
